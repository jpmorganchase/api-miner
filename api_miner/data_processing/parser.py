import re
import json
import sys
from collections import defaultdict, namedtuple, OrderedDict
from abc import ABC, abstractmethod
import numpy as np
from spacy.lang.en import English
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from api_miner.configs.internal_configs import KEY_FOR_SPEC_NAME, SEPARATOR, SENT_SEPARATOR, KEY_FOR_QUALITY, USER_QUERY, LOWERCASE, LEMMA, STEM, R_ERROR_COUNT, NUM_OCCURRENCE, REFERENCES, PROPERTIES
from api_miner.configs.openapi_configs import APISwaggerConfigs

API_endpoint = namedtuple(
    'API', 
    ['content', 'from_spec', 'tree_context', 'text_context', 'full_text_context', 'quality_score']
)

class APIParser(ABC):
    def __init__(self, openapi_configs: dict):
        self.openapi_configs = openapi_configs
        self._camelcase_pattern = re.compile(r'([a-z])([A-Z])')
        self._lemmatizer = WordNetLemmatizer()
        self._stemmer = PorterStemmer()
        self._nlp = English()

    @abstractmethod
    def featurize_endpoints_in_spec(self, spec) -> dict:
        """
        Featurize endpoints in spec, obtaining an API_endpoint object per endpoint 
        """
        pass

class APIParserSwagger(APIParser):
    def __init__(self):
        super().__init__(openapi_configs = APISwaggerConfigs())
    
    def featurize_endpoints_in_spec(self, spec) -> dict:
        '''
        For every endpoint in spec, obtain API_endpoint object

        Inputs:
        - spec: specification to featurize

        Outputs:
        - endpoint_objects: dictionary
            - key: endpoints in spec
            - values: API_endpoint objects containing contexts to use to generate features for downstream tasks
        '''
        endpoint_objects = {}

        if isinstance(spec, list): 
            spec = spec[0]
        
        if not self._is_valid_spec(spec): return endpoint_objects

        # Extract spec components
        spec_filename = spec.get(KEY_FOR_SPEC_NAME, USER_QUERY)
        quality_scores = spec.get(KEY_FOR_QUALITY, 0) 
        endpoint_specs = spec[self.openapi_configs.paths_key]
        defn_obj = spec.get(self.openapi_configs.defn_key, {})
        props_per_defn = self.get_definition_properties(defn_obj = defn_obj)
  
        for endpoint_name, endpoint_spec in endpoint_specs.items():
            if not endpoint_name:
                continue
            elif not isinstance(endpoint_spec, dict):
                continue
                
            tree_context, text_context, full_text_context = self.get_context_tokens(endpoint_spec=endpoint_spec, props_per_defn=props_per_defn)

            endpoint_objects[endpoint_name] = API_endpoint(
                content=endpoint_spec,
                from_spec=spec_filename,
                tree_context= SEPARATOR.join(tree_context), 
                text_context= SENT_SEPARATOR.join(text_context), 
                full_text_context = SENT_SEPARATOR.join(full_text_context),
                quality_score=quality_scores
            )

        return endpoint_objects

    def _is_valid_spec(self, spec) -> bool:
        '''
        Check if spec is the correct format and contains all required components

        Inputs:
        - spec: specification to check

        Ouputs:
        - boolean: True if spec is valid, False otherwise
        '''
        if not isinstance(spec, dict):
            return False
        if self.openapi_configs.paths_key not in spec:
            return False
        if not isinstance(spec[self.openapi_configs.paths_key], dict):
            return False
        return True
        
    def get_definition_properties(self, defn_obj: dict) -> dict:
        """
        Get all properties (including properties from $ref) for every definition name in defn_obj

        Inputs: 
        - defn_obj: definitions component of the spec
        
        Outputs:
        - props_per_defn: properties per definition
            - key: definition name, values: [properties]
        """
        props_per_defn = defaultdict(set)
        props_refs_per_defn = self.find_properties_and_refs(defn_obj=defn_obj)
    
        for defn_name, defn_obj in props_refs_per_defn.items():
            ref_stack = list(defn_obj[REFERENCES])
            ref_properties = defn_obj[PROPERTIES]
            defns_seen = {defn_name}

            while ref_stack:
                ref = ref_stack.pop()
                if ref not in defns_seen:
                    defns_seen.add(ref)
                    # Add properties of nested reference
                    if ref in props_refs_per_defn:
                        ref_properties.update(props_refs_per_defn[ref][PROPERTIES]) 
                    nested_refs = list(props_refs_per_defn[ref][REFERENCES]) if ref in props_refs_per_defn else []
                    ref_stack += nested_refs
            props_per_defn[defn_name] = ref_properties

        return props_per_defn

    def find_properties_and_refs(self, defn_obj: dict) -> dict:
        """
        Given defn_obj, find all the property and reference names contained in each defn

        Inputs: 
        - defn_obj: definitions component of the spec

        Outputs:
        - prop_ref_per_defn:
            - key = defn name
            - values = {properties: {}, references = {}}
        """

        props_refs_per_defn = defaultdict(lambda:{PROPERTIES: set(), REFERENCES: set()})
        for key, obj in defn_obj.items():
            if not isinstance(obj, dict): continue

            prop = obj.get(self.openapi_configs.properties_key, None)
          
            if not prop or not isinstance(prop, dict): continue
          
            for prop_name, prop_obj in prop.items():
                # Add direct property names
                props_refs_per_defn[key][PROPERTIES].add(prop_name)

                if not isinstance(prop_obj, dict): continue
                
                # Add definition reference names
                ref_names= self._get_parsed_ref_names(obj=prop_obj, only_defn_refs= True)
                if ref_names:
                    props_refs_per_defn[key][REFERENCES].update(ref_names)

        return props_refs_per_defn

    def get_context_tokens(self, endpoint_spec: dict, props_per_defn: dict):
        '''
        Get context from endpoint

        Inputs:
        - endpoint_spec: specification component of 1 endpoint
        - props_per_defn: properties per definition
            - key: definition name, values: [properties]

        Outputs:
        - tree_context: [num tree tokens]
        - text_context: [num sentences (only keywords)]
        - full_text_context = [num full sentences]
        '''
        tree_context = []
        text_context = []
        full_text_context = []

        for operation_key, operation_item in endpoint_spec.items():
            if not isinstance(operation_item, dict):
                continue

            for child_key, child_item in operation_item.items():
                if isinstance(child_item, list):
                    for item in child_item:
                        if isinstance(item, str):
                            self._add_tokens(leaf_item=item, tree_context=tree_context, prefix=child_key)
                        elif isinstance(item, dict):
                            self._add_tokens(leaf_item=item.get(self.openapi_configs.name_key, ''), tree_context=tree_context, prefix=child_key)
                            self.add_ref_tokens(item=item, props_per_defn=props_per_defn, tree_context=tree_context, prefix=child_key)
                        else:
                            continue
            
                elif isinstance(child_item, dict):
                    for key, item in child_item.items():
                        if isinstance(item, str):
                            self._add_tokens(leaf_item=item, tree_context=tree_context, prefix=child_key)
                        if isinstance(item, dict): 
                            self._add_tokens(leaf_item=key, tree_context=tree_context, prefix=child_key)
                            prefix = operation_key + '_' + child_key + '_' + key
                            self.add_ref_tokens(item=item, props_per_defn=props_per_defn, tree_context=tree_context, prefix=prefix)
                        else:
                            continue
                elif isinstance(child_item, str):
                    self.add_sentence(sent=child_item, text_context=text_context, full_text_context=full_text_context)
        
        return tree_context, text_context, full_text_context

    def add_ref_tokens(self, item:dict, props_per_defn: dict, tree_context: list, prefix: str):
        '''
        Add ref name token and property tokens under ref, if it exists
        '''
        ref_names = self._get_parsed_ref_names(obj = item)
        for ref in ref_names:
            # add ref name
            ref_tokens = {ref}
            
            # add property tokens of ref if ref is definition
            ref_tokens.update(props_per_defn.get(ref, {}))

            self._add_tokens(
                leaf_item=ref_tokens, 
                tree_context=tree_context, 
                prefix=prefix
            ) 

    def _add_tokens(self, leaf_item, tree_context: list, prefix: str):
        '''
        Add leaf tokens to tree context and leaf context
        '''
        if isinstance(leaf_item, list) or isinstance(leaf_item, set):
            for token in leaf_item:
                self._add_tokens(leaf_item=token, tree_context=tree_context, prefix=prefix)
        elif isinstance(leaf_item, str):
            for leaf_token in self._camelcase_tokenize(leaf_item):
                leaf_token = self._remove_special_chars(leaf_token)
                if LEMMA:
                    leaf_token = self._lemmatizer.lemmatize(leaf_token, 'v')
                if leaf_token and leaf_token != '':
                    tree_context.append('_'.join([prefix, leaf_token]))

    def add_sentence(self, sent:str, text_context: list, full_text_context: list):
        """
        Tokenize sentence and add to text context
        Add tokens from sentence to leaf context
        """
        sent = self._remove_special_chars(sent)
        full_text_context.append(sent)
        keyword_tokens = self._tokenize_sentence(sent)
        text_context.append(' '.join(keyword_tokens))

    def _get_parsed_ref_names(self, obj: dict, only_defn_refs = False) -> set:
        ref_full_names = set()
        parsed_refs = set()
        # Add direct ref object if it exists
        self._add_ref_name(obj=obj, names=ref_full_names)
        # Add ref in child item 
        for key, item in obj.items():
            if not isinstance(item, dict): continue
            self._add_ref_name(obj=item, names=ref_full_names)
            # check ref under specific tokens specified by 2.0
            self._add_ref_name(obj=item.get(self.openapi_configs.items_key, {}), names=ref_full_names)
            self._add_ref_name(obj=item.get(self.openapi_configs.schema_key, {}), names=ref_full_names)
            self._add_ref_name(obj=item.get(self.openapi_configs.additionalProperties_key, {}), names=ref_full_names)
         
        if only_defn_refs:
            defn_prefix = "#/" + self.openapi_configs.defn_key + "/"
            parsed_refs = {r.split('/')[-1] for r in ref_full_names if r.startswith(defn_prefix)}
        else:
            # get final names of all refs if / exists, else keep full name
            parsed_refs = {r.split('/')[-1] if '/' in r else r for r in ref_full_names }
        
        return parsed_refs
    
    def _add_ref_name(self, obj: dict, names : set):
        if isinstance(obj, list):
            for o in obj:
                self._add_ref_name(o, names)

        elif isinstance(obj, dict):
            ref_name = obj.get(self.openapi_configs.ref_key, None)
            if ref_name and isinstance(ref_name, str):
                names.add(ref_name)

    def _tokenize_sentence(self, text: str) -> str:
        tokens = []
        # Spacy Implementation
        for token in self._nlp(text):
            if token.is_stop or token.is_punct or token.is_digit:
                continue
            for t in self._camelcase_tokenize(token.text):
                if STEM:
                    t = self._stemmer.stem(t)
                elif LEMMA:
                    t = self._lemmatizer.lemmatize(t, 'v')
                tokens.append(t)
        return tokens

    def _remove_special_chars(self, text: str) -> str:
        # TODO check for common chars in tech, and remove
        text = text.replace('\n', '')
        text = text.replace('$', '')
        text = text.replace('#', '')
        return text

    def _camelcase_tokenize(self, token):
        """
        Split variable name by camelcase
        """
        # Insert space between camel case
        tokens = self._camelcase_pattern.sub(r"\1_\2", token)
        tokens = tokens.replace('-', '_')
        if LOWERCASE:
            tokens = tokens.lower()
        return tokens.split('_')
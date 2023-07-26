import json
import random
from types import GeneratorType
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from tqdm import tqdm
from collections import Counter

from api_miner.configs.internal_configs import SEPARATOR, KEY_FOR_SPEC_NAME, SENT_SEPARATOR, ANCESTORS_KEY, NUM_OCCURRENCE, ANCESTOR_CV
from api_miner.data_processing.database import write_to_disk, append_to_disk, setup_db, close_db, check_db_count

APIFeature = namedtuple(
    'APIFeature', 
    ['endpoints', 'tree_cm', 'text_cm',  'full_text_contexts']
)
APIFeatures = namedtuple(
    'APIFeatures', 
    ['endpoints', 'tree_cm', 'text_cm', 'full_text_contexts', 'idx2id', 'id2idx', 'spec2indices']
)

API_endpoint_save_info = namedtuple(
    'API', 
    [ 'endpoint_name', 'content', 'from_spec', 'quality_score']
)

class APIVectorizer(object):
    def __init__(self):
        self.idx_to_endpoint = OrderedDict() # maintain insertion order
        self.endpoint_to_idx = OrderedDict()
        self.spec2indices = defaultdict(list)
        self.quality_scores = []
        self.idx = 0
        self.min_df_percentile = 0.3

    def fit(self, specs, parser, save_to_db= True):
        tree_contexts = []
        text_contexts = []
        full_text_contexts = []
        self._featurize_specs(specs=specs, parser=parser, tree_contexts=tree_contexts, text_contexts=text_contexts, full_text_contexts=full_text_contexts, save_to_db=save_to_db)
        
        # TREE VECTORS
        self.tree_vectorizer = CountVectorizer(
            tokenizer=custom_tokenizer,
            lowercase=False, 
            min_df= 10 ,  
            max_df=0.9
        )
        self.tree_cm = self.tree_vectorizer.fit_transform(tree_contexts)
        self.tree_coo = self.compute_coo(self.tree_cm)

        # TEXT VECTORS
        self.text_vectorizer = CountVectorizer(
            lowercase=True,
            min_df= 15, 
            max_df=0.9, 
            ngram_range=(1,2)
        ) 
        self.text_cm = self.text_vectorizer.fit_transform(text_contexts)
        self.text_coo = self.compute_coo(self.text_cm)
        self.full_text_contexts = full_text_contexts

    def _featurize_specs(self, specs, parser, tree_contexts:list, text_contexts:list, full_text_contexts: list, save_to_db= True):
        """
        Get features for every spec and save to database
        """
        if not isinstance(specs, list) and not isinstance(specs, GeneratorType):
            raise ValueError("Expect input to be a list/generator")

        try:
            if save_to_db:
                setup_db()
            for spec in tqdm(specs):
                filename = spec[0][KEY_FOR_SPEC_NAME] if isinstance(spec, list) else spec[KEY_FOR_SPEC_NAME]
                if not spec: continue
                
                endpoint_objects = parser.featurize_endpoints_in_spec(spec)
                for endpoint_name, endpoint_obj in endpoint_objects.items():
                    endpoint_save_info = API_endpoint_save_info(
                        endpoint_name = endpoint_name,
                        content=endpoint_obj.content,
                        from_spec=endpoint_obj.from_spec, 
                        quality_score=endpoint_obj.quality_score
                    )
                    exist_idx = self.endpoint_to_idx.get(endpoint_name)
                    
                    # Aggregate context for duplicate api endpoint
                    if exist_idx is not None:
                        text_contexts[exist_idx] += SENT_SEPARATOR + endpoint_obj.text_context
                        tree_contexts[exist_idx] += SEPARATOR + endpoint_obj.tree_context
                        full_text_contexts[exist_idx] += SENT_SEPARATOR + endpoint_obj.full_text_context
                        self.quality_scores[exist_idx] += [endpoint_obj.quality_score]
                        
                        if save_to_db:
                            append_to_disk(endpoint_name, endpoint_save_info)
                        continue

                    # Create Mapping
                    self.idx_to_endpoint[self.idx] = endpoint_name
                    self.endpoint_to_idx[endpoint_name] = self.idx
                    self.spec2indices[endpoint_obj.from_spec].append(self.idx)
                    
                    # Create Corpus Matrices
                    tree_contexts.append(endpoint_obj.tree_context)
                    text_contexts.append(endpoint_obj.text_context)
                    full_text_contexts.append(endpoint_obj.full_text_context)
                    self.quality_scores.append([endpoint_obj.quality_score])
                    
                    # Add context to DB
                    if save_to_db:
                        write_to_disk(endpoint_name, endpoint_save_info)
                    self.idx += 1
 
        except Exception as e:
            print(e)

    @staticmethod
    def compute_coo(matrix: np.array):
        """
        Compute co-occurrence matrix
        """
        matrix_copy = matrix.copy()
        matrix_copy[matrix_copy > 0] = 1
        coo = (matrix_copy.T * matrix_copy)
        coo.setdiag(0)
        return coo

    def _find_min_df(self, cm):
        '''
        NOTE: this method is currently not used
        Find min_df value based on count matrix
        - This method assumes that cm has a high left skew (ie. many of the tokens are very unique and occur only in few documents) 
        - Get min_df that removes percentile from the largest 
        '''
        count_per_token = cm.toarray().sum(axis = 0)
        frequency_per_count = Counter(count_per_token)
        frequency_per_count_sorted = sorted(frequency_per_count.items()) # order by key (number of dfs that the token belongs to)
        total_frequency_in_percentile = self.min_df_percentile * sum(list(frequency_per_count.values()))

        min_df = 1
        curr_total_frequency = 0
        for count, frequency in frequency_per_count_sorted:
            if curr_total_frequency > total_frequency_in_percentile:
                break
            curr_total_frequency += frequency
            min_df = count

        return min_df

def custom_tokenizer(x):
    return x.split(SEPARATOR)

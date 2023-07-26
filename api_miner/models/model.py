from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from joblib import Parallel, delayed
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from joblib import Parallel, delayed
import os

from .ppmi import PPMI
from .bert import BERT
from .configs import tfidf, ppmi, bert, bert_original, bert_sent, tfidf_bert, ppmi_bert, none_val, avg_emb, \
    TEXT_EMB_PATH, TEXT_EMB_INFO_PATH, TEXT_CONFIG, BERT_CONFIG, PPMI_CONFIG, KEYWORD_EMB_CONFIG, EMB_CONCAT_CONFIG

class Model(ABC):
    @abstractmethod
    def fit(self):
        pass

class FusionModel(object):
    """
    Fuse multiple source of evidence to generate similarity score
    """
    def __init__(self, model_config, text_emb_model = None):
        self.model_config = model_config
        self.tree_model = None
        self.text_model = None

        if self.model_config.tree_config != none_val:
            self.tree_model = Tree2Vec(self.model_config)
        if self.model_config.text_config != none_val:
            self.text_model = Text2Vec(self.model_config, text_emb_model)

    def fit(self, vectorizer):
        self.endpoint_to_idx = vectorizer.endpoint_to_idx  
        self.endpoints = list(self.endpoint_to_idx.keys())
        if self.model_config.tree_config != none_val:
            self.tree_model.fit(cm=vectorizer.tree_cm, coo=vectorizer.tree_coo)
        if self.model_config.text_config != none_val:
            self.text_model.fit(vectorizer=vectorizer)
        self.quality_scores = vectorizer.quality_scores

    def find_fusion_similarity_scores(self, q_api_features, fuzzy_config: str, tree_weight: float, text_weight: float, fuzzy_weight: float, quality_weight: float):
        """
        Given API features from user query (api_features), find similarity scores between query and endpoints in data

        Inputs:
        - q_api_features: query features
        - fuzzy_config: 'True' if fuzzy matching is used, none_val otherwise
        - tree_weight: weighting to apply to tree similarity scores
        - text_weight: weighting to apply to text similarity scores
        - fuzzy_weight: weighting to apply to fuzzy similarity scores
        - quality_weight: weighting to apply to quality scores

        Outputs: 
        - similarity_scores: [query num_endpoints, data num_endpoints]
        """    
        try:
            scores = []
            if self.model_config.tree_config != none_val: 
                scores.append(self.find_tree_scores(q_cm=q_api_features.tree_cm) * tree_weight)
            if self.model_config.text_config != none_val:
                scores.append(self.find_text_scores(q_cm=q_api_features.text_cm, q_full_text_contexts= q_api_features.full_text_contexts) * text_weight)
            if fuzzy_config != none_val:
                scores.append(self.find_fuzzy_scores(q_endpoints=q_api_features.endpoints) * fuzzy_weight)

            sum_scores= self._add_quality_scores(scores=scores, quality_scores=self.quality_scores, quality_weight=quality_weight)
            similarity_scores = self.fusion_fn(sum_scores)
    
            return similarity_scores
        except Exception as e:
            import pdb
            pdb.set_trace()
            print(e)
    
    def find_tree_scores(self, q_cm: np.ndarray):
        '''
        Find similarity scores between query and data endpoints using tree features

        Inputs: 
        - q_cm: count matrix of tree path tokens in query
        
        Outputs:
        - similarity_scores: [query num_endpoints, data num_endpoints]
        '''
        qtree_vector = self.tree_model.transform(q_cm) 
        dtree_vector = self.tree_model.embeddings  
        return cosine_similarity(qtree_vector, dtree_vector) 

    def find_text_scores(self, q_cm: np.ndarray, q_full_text_contexts: list):
        '''
        Find similarity scores between query and data endpoints using text features

        Inputs: 
        - q_cm: count matrix of text tokens in query
        
        Outputs:
        - similarity_scores: [query num_endpoints, data num_endpoints]
        '''
        qtext_vector = self.text_model.transform(q_cm, q_full_text_contexts)
        dtext_vector = self.text_model.embeddings

        if type(qtext_vector) not in [np.array, np.ndarray]:
            qtext_vector = qtext_vector.toarray()
        if type(dtext_vector) not in [np.array, np.ndarray]:
            dtext_vector = dtext_vector.toarray()

        qtext_vector = sparse.csr_matrix(np.nan_to_num(qtext_vector))
        dtext_vector = sparse.csr_matrix(np.nan_to_num(dtext_vector))
        return cosine_similarity(qtext_vector, dtext_vector)

    def find_fuzzy_scores(self, q_endpoints):
        """
        Find fuzzy matching scores for every endpoint in query against every endpoint in data
        - Find which endpoints in data are most similar to endpoints in query
        
        Inputs:
        - q_endpoints: endpoint names in query 

        Outputs: 
        - fuzzy_scores : [num_endpoints_in_query, num_endpoints_in_data]
        """
        d_endpoints = self.endpoints
        fuzzy_scores= []
        for q_endpoint in q_endpoints:
            score = self.fuzzy_match(q_endpoint, d_endpoints)
            fuzzy_scores.append(score)
        return np.array(fuzzy_scores)

    @staticmethod
    def fuzzy_match(q_endpoint:str, d_endpoints: list):
        '''
        Compute fuzzy matching between endpoint in query and endpoints in data

        Inputs:
        - q_endpoint: endpoint in query 
        - d_endpoints: endpoints in data

        Outputs:
        - similarity_scores: [1, data num_endpoints]
        '''
        fuzzy_fn = lambda e: fuzz.token_set_ratio(q_endpoint, e) / 100
        score = Parallel(n_jobs=20, prefer="threads")(delayed(fuzzy_fn)(e) for e in d_endpoints)
        return score

    def _add_quality_scores(self, scores: list, quality_scores: np.array, quality_weight: float):
        """
        Add quality scores to similarity scores
        - Sum all scores together to get combined score of quality and similarity per endpoint in data

        Inputs:
        - scores: [num_evaluations, q_num_endpoints,  d_num_endpoints]
        - quality_scores: [q_num_endpoints, d_num_endpoints] 
        - quality_weight: weighting to apply to quality scores

        Outputs:
        - scores: [query num_endpoints, data num_endpoints]
        """
        q_num_endpoints = scores[0].shape[0]
        scores = np.sum(np.asarray(scores), axis= 0)
        if any(quality_scores):
            quality_scores = np.array([np.mean(scores_per_endpoint) for scores_per_endpoint in quality_scores]).reshape(1, -1) * quality_weight
            
            # NOTE: duplicate quality score for each query endpoint (to rank results for all endpoint in query)
            quality_scores = np.tile(quality_scores, (q_num_endpoints, 1))
            scores += quality_scores
        return scores

    @staticmethod
    def fusion_fn(sum_scores: np.ndarray):
        """
        Calculate normalized fusion scores given 'sum_scores' that combines both quality and similarity of every endpoint in data
        
        Inputs:
        - sum_scores: [num_endpoints_in_query, num_endpoints_data]
        Outputs:
        - norm_fusion_score: [num_endpoints_in_query, num_endpoints_data]
        """
        fusion_score = np.exp(sum_scores)
        normalise_factor = np.sum(fusion_score) 
        norm_fusion_score = fusion_score / normalise_factor
        return norm_fusion_score 

class Tree2Vec(Model):
    """
    Convert tree context into vectors
    """
    def __init__(self, model_config):
        self._config = model_config.tree_config
        if self._config == tfidf:
            self._tfidf = TfidfTransformer(smooth_idf=False)
        elif self._config ==  ppmi:
            self._ppmi = PPMI(
                laplace_k=model_config.laplace_k, 
                k_count=model_config.keep_top_k,
                alpha=model_config.alpha,
                none_val = none_val, 
                emb_config = model_config.ppmi_config
            )
            
    def fit(self, cm: np.ndarray, coo:np.ndarray):
        if self._config == tfidf:
            self.embeddings = self._tfidf.fit_transform(cm)

        elif self._config == ppmi:
            self._ppmi.fit(coo=coo) 
            self.embeddings = self._ppmi.compute_embeddings(cm=cm)
        
    def transform(self, q_cm: np.ndarray):
        if self._config ==  tfidf:
            return self._tfidf.transform(q_cm)
        elif self._config ==  ppmi:
            return self._ppmi.compute_embeddings(cm=q_cm)

class Text2Vec(Model):
    """
    Convert text context into vectors
    """
    def __init__(self, model_config, text_emb_model):
        self.emb_model = text_emb_model

    def fit(self, vectorizer):
        self.embeddings = self.emb_model.embeddings

    def transform(self, q_cm: np.ndarray, q_full_text_contexts: list):
        return self.emb_model.transform(q_cm=q_cm, q_full_text_contexts=q_full_text_contexts)


import numpy as np
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import pickle
from scipy import sparse
from sklearn.cross_decomposition import CCA

from .ppmi import PPMI
from .bert import BERT
from .configs import tfidf, ppmi, bert, tfidf_bert, ppmi_bert, none_val, t_svd, cca, concat

class TextEmbeddings():
    def __init__(self, model_config):
        self._config = model_config
        self.keyword_embeddings = None
        self.bert_embeddings = None
        self.embeddings = None

        if self._config.text_config == tfidf or self._config.text_config == tfidf_bert:
            self._tfidf = TfidfTransformer(smooth_idf=False)
        if self._config.text_config == ppmi or self._config.text_config == ppmi_bert:
            self._ppmi = PPMI(
                laplace_k=model_config.laplace_k, 
                k_count=model_config.keep_top_k,
                alpha=model_config.alpha,
                none_val = none_val, 
                emb_config = model_config.ppmi_config
            )
        if self._config.text_config == tfidf_bert or self._config.text_config == ppmi_bert or self._config.text_config == bert:
            self._bert = BERT(model_config.bert_config)
        
    def fit(self, vectorizer):
        cm=vectorizer.text_cm
        coo=vectorizer.text_coo
        full_text_contexts=vectorizer.full_text_contexts

        # single embedding
        if self._config.text_config == tfidf:
            self.embeddings = self._tfidf.fit_transform(cm)
        elif self._config.text_config == ppmi:
            self._ppmi.fit(coo=coo) 
            self.embeddings = self._ppmi.compute_embeddings(cm=cm)
        elif self._config.text_config == bert:
            self.embeddings = self._bert.compute_embeddings(full_text_contexts=full_text_contexts)
        
        # combine embedding
        elif self._config.text_config == tfidf_bert:
            self.keyword_embeddings = self._tfidf.fit_transform(cm)
            self.bert_embeddings = self._bert.compute_embeddings(full_text_contexts=full_text_contexts)
            self.embeddings = self.compute_combined_embeddings(keyword_embeddings=self.keyword_embeddings, bert_embeddings=self.bert_embeddings, fit= True)
        elif self._config.text_config == ppmi_bert:
            self._ppmi.fit(coo=coo) 
            self.keyword_embeddings = self._ppmi.compute_embeddings(cm=cm)
            self.bert_embeddings = self._bert.compute_embeddings(full_text_contexts=full_text_contexts)
            self.embeddings = self.compute_combined_embeddings(keyword_embeddings=self.keyword_embeddings, bert_embeddings=self.bert_embeddings, fit= True)

    def transform(self, q_cm: np.ndarray, q_full_text_contexts: list):
        # single embedding
        if self._config.text_config == tfidf:
            return self._tfidf.transform(q_cm)
        elif self._config.text_config == ppmi:
            return self._ppmi.compute_embeddings(cm=q_cm)
        elif self._config.text_config == bert:
            return self._bert.compute_embeddings(full_text_contexts=q_full_text_contexts)

        # combine embedding
        elif self._config.text_config == tfidf_bert:
            keyword_embeddings = self._tfidf.transform(q_cm)
            bert_embeddings = self._bert.compute_embeddings(full_text_contexts=q_full_text_contexts)
            return self.compute_combined_embeddings(keyword_embeddings= keyword_embeddings, bert_embeddings=bert_embeddings, fit= False)
        elif self._config.text_config == ppmi_bert:
            keyword_embeddings = self._ppmi.compute_embeddings(cm=q_cm)
            bert_embeddings = self._bert.compute_embeddings(full_text_contexts=q_full_text_contexts)
            return self.compute_combined_embeddings(keyword_embeddings= keyword_embeddings, bert_embeddings=bert_embeddings, fit= False)

    def compute_combined_embeddings(self, keyword_embeddings, bert_embeddings, fit = True, cca_dim = 300):
        try:
            # 1. preprocess keyword embeddings
            if self._config.keyword_emb_config == t_svd:
                '''
                Linear dimensionality reduction using truncated SVD
                - does not center data before computing SVD = work well with sparse matrices (centering sparse matrices = memory explosion)
                - works well on term count / TFIDF matrices 
                '''
                n_components = self._get_n_components(keyword_embeddings)
                if fit:
                    self.tsvd = TruncatedSVD(n_components=n_components)
                    self.tsvd.fit(keyword_embeddings) # keyword_embeddings = [n_endpoints, n_features] =  [n_samples, n_features]
                keyword_embeddings = self.tsvd.transform(keyword_embeddings) # np.array [n_endpoints, new_dim]

            
            # 2. combine embeddings
            if self._config.emb_concat_config == cca:
                '''
                Linearly project keyword and bert embeddings in the same space using CCA (canonical correlation analysis) 
                - The number of components for CCA is chosen as the smaller dimension between keyword and bert embeddings
                    - eg. if keyword_emb = [num_endpoints, dim1], bert_emb = [num_endpoints, dim2] and dim1 < dim2 
                    -> transformed(keyword_emb) = [num_endpoints, dim1], transformed(bert_emb) = [num_endpoints, dim1]
                '''
                # convert to array for cca
                if isinstance(keyword_embeddings, sparse.csr_matrix):
                    keyword_embeddings = keyword_embeddings.toarray()
                if isinstance(bert_embeddings, sparse.csr_matrix):
                    bert_embeddings = bert_embeddings.toarray()
                # make sure no nan values exist
                keyword_embeddings = np.nan_to_num(keyword_embeddings)
                bert_embeddings = np.nan_to_num(bert_embeddings)

                if fit:
                    self.cca = CCA(n_components=cca_dim)
                    self.cca.fit(keyword_embeddings, bert_embeddings) 
                
                keyword_embeddings, bert_embeddings = self.cca.transform(keyword_embeddings, bert_embeddings)
                return self.concatenate_embeddings(keyword_embeddings, bert_embeddings)
            elif self._config.emb_concat_config == concat:
                return self.concatenate_embeddings(keyword_embeddings, bert_embeddings)
        
        except Exception as e: 
            import pdb
            pdb.set_trace()
            print('ERROR IN FITTING TEXT EMB: ', e)

    def _get_n_components(self, keyword_embedding, goal_variance = 0.95): # potentially 0.8-0.9
        '''
        Get lowest n_components that has at least goal_variance for truncated SVD
        Reference: https://chrisalbon.com/code/machine_learning/feature_engineering/select_best_number_of_components_in_tsvd/
        '''
        tsvd = TruncatedSVD(n_components=keyword_embedding.shape[1] - 1) # NOTE fit on one less than # of features
        truncated_emb = tsvd.fit(keyword_embedding)

        total_variance = 0
        n_components = 0
        for v in tsvd.explained_variance_ratio_: # tsvd.explained_variance_ratio_ [num_features-1]
            total_variance += v
            n_components += 1
            if total_variance> goal_variance:
                break
        return n_components
    
    @staticmethod
    def concatenate_embeddings(emb1, emb2) -> sparse.csr_matrix:
        '''
        Concatenate 2 embeddings (ie. keyword and bert embeddings)

        Inputs:
        - emb1 = [num_endpoints, dim1] (csr matrix)
        - emb2 = [num_endpoints, dim2] (csr matrix)
        Outputs:
        - [num_endpoints, dim1 + dim2] (csr matrix)
        '''
        if not isinstance(emb1, sparse.csr_matrix):
            emb1 = sparse.csr_matrix(emb1)
        if not isinstance(emb1, sparse.csr_matrix):
            emb1 = sparse.csr_matrix(emb1)

        return sparse.hstack((emb1, emb2)) # hstack = same number of rows, concat cols

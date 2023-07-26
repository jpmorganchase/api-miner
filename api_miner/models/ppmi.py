import numpy as np
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from .configs import avg_emb 

class PPMI():
    '''
    Positive Pointwise Mutual Information
    '''
    def __init__(self, laplace_k, k_count, alpha, none_val: str, emb_config: str):
        self._none_val = none_val
        self.laplace_k = laplace_k
        self.k_count = k_count
        self.alpha = alpha
        self.emb_config = emb_config
        
    def fit(self, coo: np.ndarray):
        """
        Generate self.ppmi_matrix given coo (co-occurrence matrix)

        Inputs:
        - coo: co occurrence matrix
        """
        if self.laplace_k != self._none_val:
            # NOTE: Apply laplace approach to all entries (incl. zero)
            coo.data = coo.data + self.laplace_k

        if self.k_count != self._none_val:
            coo = coo.todense()
            self.keep_top_k_per_row(coo)

        self.n = np.sum(coo)
        prob_matrix = coo / self.n # p(w, c)
        prob_word = np.sum(coo, axis=1) / self.n # p(w)
        
        self.count_context = np.sum(coo, axis=0)
        if self.alpha != self._none_val:
            # NOTE: Weighting PPMI to give rare context words sightly higher probability  
            self.count_context = np.power(self.count_context, self.alpha)
            weighted_n = np.sum(self.count_context)
            prob_context = self.count_context / weighted_n # p(c)
        else:
            prob_context = self.count_context / self.n # p(c)

        ppmi_matrix = np.log2(np.nan_to_num(prob_matrix / (prob_word * prob_context))) # log2 [P(w,c) / [P(w)P(c)]]
        ppmi_matrix[ppmi_matrix < 0] = 0
        ppmi_matrix = np.nan_to_num(ppmi_matrix)
        self.ppmi_matrix = sparse.csr_matrix(ppmi_matrix)
    
    def compute_embeddings(self, cm: np.ndarray) -> np.ndarray:
        """
        Compute ppmi embeddings for the spec by calculating the dot product between cm and ppmi_matrix
        - self.ppmi_matrix: [num_unique_tokens, num_unique_tokens]
            - PPMI matrix (ie. log2 [P(w,c) / [P(w)P(c)]])

        Inputs: 
        - cm (count matrix): [num_endpoints, num_unique_tokens]
            - counts frequency of unique token appearing in each endpoint
        Ouputs: 
        - embeddings: [num_endpoints, num_unique_tokens]
            - generate PPMI embedding representation of each endpoint
            - for endpoint (ie. row of count matrix),
                - summarize PPMI [dim,dim] to obtain single value per dim -> [1, dim]
                - single value = cm[i] * [emb value of token in that dim] = [1, dim] * [dim, 1] = [1]
                - single value = sum of token in that dim weighted by count of token that appear in endpoint i 
        """
        cm = cm.copy()
        num_endpoints, _ = cm.shape
        embeddings = np.zeros(cm.shape)
        for i in range(num_endpoints):
            #  (cm[i] * self.ppmi_matrix).toarray() = dot product => [1, 7737]
            embeddings[i] += (cm[i] * self.ppmi_matrix).toarray().flatten()
     
        if self.emb_config == avg_emb:
            embeddings /= np.sum(embeddings, axis=1).reshape(-1, 1) 

        embeddings = np.nan_to_num(embeddings) # Fill nan with 0

        return embeddings

    def keep_top_k_per_row(self, matrix: np.ndarray):
        """
        Keep top k largest values per row. Make numbers lower than threshold 0.
        """
        for row in np.asarray(matrix):
            threshold = np.sort(row)[::-1][self.k_count]
            row[row < threshold] = 0

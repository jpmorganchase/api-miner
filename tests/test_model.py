import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import sys
sys.path.append('..')

from experiments.configs import ModelConfig, TextEmbeddingsConfig
from api_miner.models.model import FusionModel, Text2Vec
from api_miner.models.text_embeddings import TextEmbeddings
from api_miner.models.configs import tfidf, ppmi, bert, bert_original, bert_sent, tfidf_bert, ppmi_bert, \
     none_val, avg_emb, t_svd, cca
from api_miner.models.bert import BERT
from api_miner.data_processing.vectorize import APIFeatures, APIFeature, APIVectorizer

MODEL_CONFIG = ModelConfig(
    tree_config=ppmi, 
    text_config=ppmi_bert, 
    bert_config = bert_sent, 
    ppmi_config = avg_emb, 
    laplace_k = none_val,
    alpha = 0.1,
    keep_top_k =none_val)
    
TEXT_EMB_CONFIG = TextEmbeddingsConfig(
            text_config= ppmi_bert, 
            bert_config= bert_sent, 
            ppmi_config = avg_emb,
            keyword_emb_config = t_svd, 
            emb_concat_config = cca, 
            laplace_k = none_val,
            alpha = 0.1,
            keep_top_k =none_val)
TEXT_EMB_MODEL = TextEmbeddings(TEXT_EMB_CONFIG)
TEXTMODEL = Text2Vec(MODEL_CONFIG, TEXT_EMB_MODEL)
FUSION_MODEL = FusionModel(MODEL_CONFIG)

def test_compute_combined_embeddings():
    keyword_embeddings = np.asarray([[1,2,3,5,6], [4,5,6,4,3], [1,2,3,7,5]])
    bert_embeddings = sparse.csr_matrix([[1,2], [2,1], [3,1]])
    num_samples = 3
    keyword_emb_dim = 5
    bert_emb_dim =  2

    # t_svd + cca
    config = TextEmbeddingsConfig(
            text_config= ppmi_bert, 
            bert_config= bert_sent, 
            ppmi_config = avg_emb,
            keyword_emb_config = t_svd, 
            emb_concat_config = cca, 
            laplace_k = none_val,
            alpha = 0.1,
            keep_top_k =none_val)
    emb_model = TextEmbeddings(config)
    embeddings = emb_model.compute_combined_embeddings(keyword_embeddings, bert_embeddings)
    transformed_keyword_emb_dim = 3
    assert(embeddings.shape == (num_samples, min(bert_emb_dim, transformed_keyword_emb_dim) * 2))

    # cca
    config = TextEmbeddingsConfig(
            text_config= ppmi_bert, 
            bert_config= bert_sent, 
            ppmi_config = avg_emb,
            keyword_emb_config = none_val, 
            emb_concat_config = cca, 
            laplace_k = none_val,
            alpha = 0.1,
            keep_top_k =none_val)
    emb_model = TextEmbeddings(config)
    embeddings = emb_model.compute_combined_embeddings(keyword_embeddings, bert_embeddings)
    assert(embeddings.shape == (num_samples, min(bert_emb_dim, keyword_emb_dim) * 2))

def test_concatenate_embeddings():
    '''
    Concatenate emb1 and emb2
    emb1 = [num_endpoints, dim1]
    emb2 = [num_endpoints, dim2]
    output = [num_endpoints, dim1 + dim2]
    '''
    emb1 = csr_matrix([[1,1], [1,1], [1,1]])
    emb2 = csr_matrix([[2,2,2], [2,2,2], [2,2,2]])
    expected_output = csr_matrix([[1,1,2,2,2], [1,1,2,2,2], [1,1,2,2,2]])
    output = TEXT_EMB_MODEL.concatenate_embeddings(emb1, emb2)
    assert((output.toarray() == expected_output.toarray()).all())
    assert((output.shape[0] == emb1.shape[0]) and (output.shape[0] == emb2.shape[0]))
    assert(output.shape[1] == emb1.shape[1] + emb2.shape[1])

def test_bert_embeddings():
    '''
    full_text_contexts = [num_endpoints]
    Outputs = [num_endpoint, 768 (bert embedding dim)] = sparse 
    '''
    BERTMODEL = BERT(bert_type = bert_original)
    full_text_contexts = ['I like cheese', 'and rice too']
    output = BERTMODEL.compute_embeddings(full_text_contexts)
    assert(output.shape == (2, 768))
    assert(type(output) == csr_matrix)

def test_bert_embeddings_long():
    '''
    Test for when input sentence is longer than 512 characters
    full_text_contexts = [num_endpoints]
    Outputs = [num_endpoint, 768 (bert embedding dim)] = sparse 
    '''
    BERTMODEL = BERT(bert_type = bert_original)
    full_text_contexts = [' '.join(['i'] * 530)]
    output = BERTMODEL.compute_embeddings(full_text_contexts)
    assert(output.shape == (1, 768))
    assert(type(output) == csr_matrix)

def test_sent_bert_embeddings():
    '''
    full_text_contexts = [num_endpoints]
    Outputs = [num_endpoint, 768 (bert embedding dim)]
    '''
    
    BERTMODEL = BERT(bert_type = bert_sent)
    full_text_contexts = ['I like cheese', 'and rice too']
    output = BERTMODEL.compute_embeddings(full_text_contexts)
    assert(output.shape == (2, 384))
    assert(type(output) == csr_matrix)

def test_sent_bert_embeddings_long():
    '''
    Test for when input sentence is longer than 512 characters
    full_text_contexts = [num_endpoints]
    Outputs = [num_endpoint, 768 (bert embedding dim)]
    '''
    
    BERTMODEL = BERT(bert_type = bert_sent)
    full_text_contexts = [' '.join(['i'] * 530)]
    output = BERTMODEL.compute_embeddings(full_text_contexts)
    assert(output.shape == (1, 384))
    assert(type(output) == csr_matrix)

if __name__ == "__main__":
    test_compute_combined_embeddings()

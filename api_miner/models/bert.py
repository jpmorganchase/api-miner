import numpy as np
import torch
from collections import namedtuple
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from scipy import sparse
from tqdm import tqdm

from api_miner.configs.internal_configs import BERT_PATH, SENT_BERT_PATH
from .configs import bert, bert_original, bert_sent

class BERT():
    """
    Bert model to obtain embeddings to featurize natural language texts
    """
    def __init__(self, bert_type: str):
        self._bert_type = bert_type
        if self._bert_type == bert_original:
            self._tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
            self._model = BertModel.from_pretrained(BERT_PATH)
            self._model.eval()
        
        elif self._bert_type == bert_sent:
            '''
            Sentence BERT: uses siamese and triplet network structures (modified BERT network) to derive semantically meaningful sentences
            https://www.sbert.net/docs/pretrained_models.html
            - paraphrase-mpnet-base-v2: best quality
            - paraphrase-MiniLM-L6-v2: quick model with high quality = better for us

            Model downloaded: 19-May-2021 20:01, 83426730
            '''
            self._model = SentenceTransformer(SENT_BERT_PATH)

    def compute_embeddings(self, full_text_contexts: list):
        """
        Computes BERT embeddings given list of sentences for each endpoint
        
        Inputs:
        - full_text_contexts: [num_endpoints], each containing string of natural language texts contained in endpoint

        Outputs:
        - embeddings: [num_endpoints, bert_dim]
        """
        if self._bert_type == bert_original:
            embeddings = self._compute_bert_embeddings(full_text_contexts)
        elif self._bert_type == bert_sent:
            embeddings = self._compute_sent_bert_embeddings(full_text_contexts)
        return embeddings

    def _compute_bert_embeddings(self, full_text_contexts: list):
        '''
        Compute embeddings for full_text_contexts using original BERT
        - Sentence embeddings are computed by calculating the average of its context token embeddings
        - When the string is longer than context length (512), it gets first 511 tokens + final [SEP] token

        Inputs:
        - full_text_contexts: [num_endpoints], each containing string of natural language texts contained in endpoint

        Outputs:
        - embeddings: [num_endpoints, 768]
        '''
        embeddings = []
        for sent in full_text_contexts:
            input_ids = torch.tensor(self._tokenizer.encode(sent)).unsqueeze(0)
            if input_ids.shape[1] > 512:
                # NOTE only encode up to 512
                input_ids = input_ids.reshape(input_ids.shape[1])
                # first 511 tokens and last SEP token
                input_ids = torch.cat((input_ids[:511], input_ids[-1].reshape(1))).reshape(1, 512)
            embs = self._model(input_ids).last_hidden_state.detach().numpy()
            
            # Sentence embedding = average of context embeddings (exept [CLS] and [SEP] tokens) 
            context_embs = embs[:, 1:-1, :]
            sentence_embs = np.sum(context_embs, axis = 1) / context_embs.shape[1] 
            
            embeddings.append(sentence_embs.reshape(sentence_embs.shape[1]))
      
        return sparse.csr_matrix(embeddings)

    def _compute_sent_bert_embeddings(self, full_text_contexts:list):
        '''
        Compute embeddings for full_text_contexts using BERT SENT
        Context length limit: automatically taken care of (https://www.sbert.net/examples/applications/computing-embeddings/README.html) 
        - This limits transformers to inputs of certain lengths. 
        - A common value for BERT & Co. are 512 word pieces, which corresponde to about 300-400 words (for English). 
        - Longer texts than this are truncated to the first x word pieces.

        Inputs:
        - full_text_contexts: [num_endpoints], each containing string of natural language texts contained in endpoint

        Outputs:
        - embeddings: [num_endpoints, 384]
        '''
        embeddings = self._model.encode(full_text_contexts)
      
        return sparse.csr_matrix(embeddings)
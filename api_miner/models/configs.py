from collections import namedtuple
import os
import platform
from pathlib import Path

PROJECT_NAME = 'Senatus_S3'

# -------------- MODEL VARIABLES -------------- # 

tfidf = 'tfidf'
ppmi = 'ppmi'
bert = 'bert'
bert_original = 'bert_original'
bert_sent = 'bert_sent'
tfidf_bert = 'tfidf_bert'
ppmi_bert = 'ppmi_bert'
none_val = 'n.a'
avg_emb = 'avg_emb'
t_svd = 't_svd'
cca = 'cca'
concat = 'concat'
bert_original = 'bert_original'
bert_sent = 'bert_sent'
fuzzy_true = 'True'

TEXT_CONFIG = 'text_config'
BERT_CONFIG = 'bert_config'
PPMI_CONFIG = 'ppmi_config'
KEYWORD_EMB_CONFIG= 'keyword_emb_config'
EMB_CONCAT_CONFIG = 'emb_concat_config'

# -------------- MODEL CONFIGS TO EXPLORE (CAN EDIT) -------------- # 

tree_config_options = [tfidf, ppmi]
text_config_options = [tfidf, ppmi, bert, tfidf_bert, ppmi_bert]
bert_config_options = [bert_original, bert_sent]
ppmi_config_options = [avg_emb]
keyword_emb_config_options = [t_svd]
emb_concat_config_options = [cca, concat]

def _get_path_separator():
    os = platform.system()
    return '\\' if os == 'Windows' else '/'

def _get_root():
    root = str(Path(__file__).parent.parent.parent.absolute())
    return root

PATH_SEP = _get_path_separator()
ROOT = _get_root()

def _get_text_emb_path():
    return os.path.normpath(ROOT + PATH_SEP + PROJECT_NAME + PATH_SEP +  'experiments'  + PATH_SEP  + 'text_embs')

def _get_text_emb_info_path():
    return os.path.normpath(ROOT + PATH_SEP + PROJECT_NAME + PATH_SEP +  'experiments'  + PATH_SEP  + 'logs' + PATH_SEP + 'text_emb_info.csv')

TEXT_EMB_PATH = _get_text_emb_path()
TEXT_EMB_INFO_PATH = _get_text_emb_info_path()




# -------------- MODEL CONFIGS -------------- # 
class ModelConfig:
    def __init__(self, tree_config: str, text_config:str, bert_config: str, ppmi_config: str, laplace_k: int, alpha: float, keep_top_k: int):
        # Model types
        self.tree_config = tree_config
        self.text_config = text_config
        self.bert_config = bert_config
        self.ppmi_config = ppmi_config
        self.laplace_k = laplace_k
        self.alpha = alpha
        self.keep_top_k = keep_top_k

class TextEmbeddingsConfig:
    def __init__(self, text_config:str, bert_config: str, ppmi_config: str, keyword_emb_config: str, emb_concat_config: str, laplace_k: int, alpha: float, keep_top_k: int):
        self.text_config = text_config
        self.bert_config = bert_config
        self.ppmi_config = ppmi_config
        self.keyword_emb_config = keyword_emb_config
        self.emb_concat_config = emb_concat_config
        self.laplace_k = laplace_k
        self.alpha = alpha
        self.keep_top_k = keep_top_k 
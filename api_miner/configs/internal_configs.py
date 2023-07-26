import os
import platform
from pathlib import Path
from collections import namedtuple

# EXPERIMENT INFO
DATASET_DIR = 'data'
DB_FOLDER = 'database'

# PATHS
def get_path_separator():
    os = platform.system()
    return '\\' if os == 'Windows' else '/'

def get_project_root():
    root = str(Path(__file__).parent.parent.parent.absolute())
    return root

PATH_SEP = get_path_separator()
ROOT = get_project_root()

def _get_data_path():
    return os.path.normpath(ROOT + PATH_SEP + DATASET_DIR + PATH_SEP + 'processed' + PATH_SEP + 'valid_specs')

def _get_grade_path():
    return os.path.normpath(ROOT + PATH_SEP + DATASET_DIR + PATH_SEP + 'processed' + PATH_SEP + 'grades')

def _get_public_dataset_repo_path():
    return os.path.normpath(ROOT + PATH_SEP + DATASET_DIR + PATH_SEP + 'raw')

def _get_exp_logs_path():
    return os.path.normpath(ROOT + PATH_SEP + 'experiments' + PATH_SEP + 'logs')

def _get_bert_path():
    return os.path.normpath(ROOT + PATH_SEP + 'lib' + PATH_SEP + 'bert')

def _get_sent_bert_path():
    return os.path.normpath(ROOT + PATH_SEP + 'lib' + PATH_SEP + 'sent_bert')

def _get_dataset_info_file_path():
   return os.path.normpath(ROOT + PATH_SEP + 'experiments' + PATH_SEP + 'logs' + PATH_SEP + 'dataset_info.csv')

def _get_grades_info_path():
   return os.path.normpath(ROOT + PATH_SEP + 'experiments' + PATH_SEP + 'logs' + PATH_SEP + 'grades_info.csv')

def _get_validation_set_info_path():
   return os.path.normpath(ROOT + PATH_SEP + 'experiments' + PATH_SEP + 'logs' + PATH_SEP + 'validation_set_info.csv')

def _get_validation_endpoints_path():
   return os.path.normpath(ROOT + PATH_SEP + 'experiments' + PATH_SEP + 'logs' + PATH_SEP + 'validation_set_endpoints.pickle')

DATA_PATH = _get_data_path()
PUBLIC_REPO_PATH = _get_public_dataset_repo_path()
REPO_PATH_TOKENS = PUBLIC_REPO_PATH.split(PATH_SEP)
EXP_LOGS_PATH = _get_exp_logs_path()
BERT_PATH = _get_bert_path()
SENT_BERT_PATH = _get_sent_bert_path()
DATASET_INFO_PATH = _get_dataset_info_file_path()
VALIDATION_SET_INFO_FILE = _get_validation_set_info_path()
VALIDATION_SET_ENDPOINTS_FILE = _get_validation_endpoints_path()
GRADE_PATH = _get_grade_path()
GRADE_FILE_PATH = GRADE_PATH + PATH_SEP + 'grades.csv'
GRADES_INFO_PATH = _get_grades_info_path()
GRADED_SPEC_FILE_PATH = 'file_path'
GRADED_SPEC_GRADE = 'grade'

# INTERNAL VARIABLES
KEY_FOR_SPEC_NAME='from_spec'
CONTENT = 'content'
USER_QUERY='user_query'
KEY_FOR_QUALITY='quality_score'
SEPARATOR = "[SEP]"
SENT_SEPARATOR = ". "
GRADE_EXT = ".grade"
LOWERCASE = True
LEMMA = True
STEM = False
R_ERROR_COUNT = 0
MIN_GRADE = 0.5
ANCESTORS_KEY = 'ancestors'
NUM_OCCURRENCE = 'num_occurrence'
ANCESTOR_CV = 'ancestor_cv'
JSON = '.json'
YAML = '.yaml'
SWAGGER = 'swagger'
REFERENCES  = 'references'
PROPERTIES = 'properties'
VALIDATION_SET_KEY = 'validation_set'
EVALUATION_SET_KEY = 'evaluation_set'
MASKED_RETRIEVAL_KEY = 'masked_retrieval'
MANGLED_RETRIEVAL_KEY = 'mangled_retrieval'
USER_STUDY_KEY = 'user_study'
import os
import platform
from pathlib import Path

DATASET_DIR = 'data'
DB_FOLDER = 'database'

def get_path_separator():
    os = platform.system()
    return '\\' if os == 'Windows' else '/'

PATH_SEP = get_path_separator()
def get_database_path():
    root = get_project_root()
    return os.path.normpath(root + PATH_SEP + DATASET_DIR + PATH_SEP + 'processed' + PATH_SEP + DB_FOLDER + PATH_SEP + 'data.db')

def get_project_root():
    root = str(Path(__file__).parent.parent.parent.absolute())
    return root

DB_PATH = get_database_path()
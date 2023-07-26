import numpy as np
import os
import shutil
from datetime import date 
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('..')
import json
from pathlib import Path

from api_miner.configs.internal_configs import DATASET_INFO_PATH, PATH_SEP, REPO_PATH_TOKENS, PUBLIC_REPO_PATH, DATA_PATH, GRADE_FILE_PATH, JSON, YAML, SWAGGER
from api_miner.data_processing.utils import get_spec_at_path

def generate_dataset_from_repo():
    '''
    - Extract only Swagger 2.x versions
    - nouns in urls, verbs out of urls
    - give examples for add get responses
    '''
    if not os.path.exists(DATASET_INFO_PATH):
        print('Processing raw data and saving valid OpenAPI spec...')
        total_files = valid_files  = 0
        for root, dirs, files in tqdm(os.walk(PUBLIC_REPO_PATH)):
            for f in files:
                total_files += 1
                if f.endswith(YAML) or f.endswith(JSON): 
                    full_file_path = os.path.join(root,f)

                    spec =  _get_valid_spec(file_path= full_file_path)
                    if spec:
                        valid_files += 1
                        _save_spec_as_json(spec=spec, full_file_path=full_file_path)

        _save_dataset_info(total_files= total_files, valid_files=valid_files)

    else:
        print('Repo has already been processed and saved at: ', DATA_PATH)

def _save_spec_as_json(spec:dict, full_file_path:str) -> str:
    '''
    Inputs:
    - spec: API specification obtained from repo
    - full_file_path: full file path of the spec
    '''
    # Create new folder for API if it doesn't exist
    api_path_tokens = [p for p in full_file_path.split(PATH_SEP) if p not in REPO_PATH_TOKENS]
    save_folder_path = DATA_PATH + PATH_SEP + '_'.join(api_path_tokens[:-2]) 
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Save processed JSON spec into new path
    save_file_name = api_path_tokens[-2:]
    save_file_name[-1] = ''.join(save_file_name[-1].split('.')[:-1] + [JSON])
    save_file_name = '_'.join(save_file_name)
    save_file_path = Path(save_folder_path + PATH_SEP + save_file_name)
    try:
        save_file_path.write_text(json.dumps(spec, default=str))
    except Exception as e:
        print('Error in saving JSON: ', full_file_path)
        print(e)
        print('Did not save spec...')

def _get_valid_spec(file_path = str):
    '''
    Valid:
    - Swagger 2.x
    '''
    spec = get_spec_at_path(file_path=file_path)
    version = spec.get(SWAGGER, '')

    if not version.startswith('2.') or not spec:
        return None
    return spec

def _save_dataset_info(total_files: int, valid_files: int):
    '''
    Save info about the dataset in dataset_info file

    Inputs:
    - total_files: total number of files in repo
    - valid_files: total number of valid files saved
    '''
    cols = ['date_run', 'num_total_files', 'num_valid_files', 'num_unique_apis']
    vals =  [date.today(), total_files, valid_files, len(os.listdir(DATA_PATH))]

    dataset_info = pd.DataFrame(columns = cols)
    dataset_info = dataset_info.append(dict(zip(cols, vals)), ignore_index=True)
    dataset_info.to_csv(DATASET_INFO_PATH, index = False)

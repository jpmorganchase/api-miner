import os
import json
import pickle
import logging
from collections import defaultdict
import yaml
from pathlib import Path
from scipy import sparse
import numpy as np

from api_miner.configs.internal_configs import KEY_FOR_SPEC_NAME, KEY_FOR_QUALITY, DATA_PATH, GRADE_PATH, GRADE_EXT, MIN_GRADE, JSON, YAML
from api_miner.data_processing.vectorize import APIFeatures, APIFeature
SEPARATOR = "[SEP]"

def get_spec_at_path(file_path: str) -> dict:
    '''
    Return specification as dictionary at file_path
    '''
    spec = {}
    if file_path.endswith(YAML):
        spec, e1 = convert_yaml_to_json(file_path=file_path)
        if not spec:
            spec, e2 = convert_yaml_to_json_utf_8(file_path=file_path)
        if not spec:
            print("Error in loading spec to json", file_path)
            print('Error 1: ', e1)
            print('Error 1: ', e2)
        
    elif file_path.endswith(JSON):
        file_content = open(file_path, 'rb')
        spec = json.load(file_content)
    else:
        print('Skipping invalid file type at: ', file_path)
    return spec

def convert_yaml_to_json(file_path: str):
    spec = {}
    error = ''
    try:
        file_content = open(file_path)
        spec = yaml.load(file_content, Loader=yaml.BaseLoader)
    except Exception as e:
        error = e
        pass
    return spec, error

def convert_yaml_to_json_utf_8(file_path: str):
    spec = {}
    error = ''
    try:
        file_content = open(file_path, encoding = 'utf-8')
        spec = yaml.safe_load(file_content)
    except Exception as e:
        error = e
        pass
    return spec, error

def transform_specs_to_features(specs, parser, vectorizer):
    """
    Tranform a specs (eg. user query) into API Features
    """
    tree_contexts = []
    text_contexts = []
    full_text_contexts = []
    endpoints = []
    id2idx = {}
    # TODO what is this id -> knowing where the endpoint came from (which spec)
    idx2id = {}
    spec2indices = defaultdict(list)
    idx = 0
    
    for spec in specs: 
        if not spec: continue

        endpoint_objects = parser.featurize_endpoints_in_spec(spec)

        if not endpoint_objects: continue 

        for endpoint, endpoint_obj in endpoint_objects.items():
            tree_contexts.append(endpoint_obj.tree_context)
            text_contexts.append(endpoint_obj.text_context)
            full_text_contexts.append(endpoint_obj.full_text_context)
            endpoints.append(endpoint)
            
            id_ = SEPARATOR.join([endpoint_obj.from_spec, endpoint])
            id2idx[id_] = idx
            idx2id[idx] = id_

            # All the indices of endpoints in spec
            spec2indices[endpoint_obj.from_spec].append(idx)
            idx += 1
    
    return APIFeatures(
        tree_cm=vectorizer.tree_vectorizer.transform(tree_contexts) if len(tree_contexts) > 0 else sparse.csr_matrix(np.asarray([[0]* vectorizer.tree_cm.shape[1] ])),
        text_cm=vectorizer.text_vectorizer.transform(text_contexts) if len(text_contexts) > 0 else sparse.csr_matrix(np.asarray([[0]* vectorizer.text_cm.shape[1] ])),
        full_text_contexts = full_text_contexts  if len(full_text_contexts) > 0 else np.asarray([['']* vectorizer.text_cm.shape[1] ]),
        endpoints = endpoints,
        idx2id=idx2id,
        id2idx=id2idx,
        spec2indices=spec2indices
    )
    
def transform_endpoint_name_to_features(endpoint_name: str, vectorizer):
    '''
    Given endpoint name that exists in the database, return features that were computed
    '''
    idx = vectorizer.endpoint_to_idx.get(endpoint_name)
    
    if idx is None: # Cater to index 0
        raise ValueError('Endpoint ', endpoint_name, 'does not exist in the database')
        return None

    return APIFeature(
        tree_cm=vectorizer.tree_cm[idx],
        text_cm=vectorizer.text_cm[idx],
        full_text_contexts =  [vectorizer.full_text_contexts[idx]],
        endpoints=[endpoint_name]
    )
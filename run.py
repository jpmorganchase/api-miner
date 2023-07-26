import numpy as np
import time
from experiments.setup import initialize_components, fit_models
from experiments.evaluate import run_experiments
from experiments.user_study import setup_user_study, evaluate_user_study

def setup(openapi_version: int, optim_hyperparams: bool):
    '''
    Specify OpenAPI version to obtain data_loader, validator, parser, vectorizer
    - NOTE: only openAPI version 2 is currently supported
    - can set up dataset (ie. pull from repo) if dataset_setup = False
    - can parse through database and save vectorizer (if not saved previously)
    - fit all models in advance
    '''
    print('Initializing data ...')
    data_loader, validator, parser, vectorizer = initialize_components(openapi_version=openapi_version)
    fit_models(vectorizer, parser, data_loader, optim_hyperparams= optim_hyperparams)

    return data_loader, validator, parser, vectorizer

if __name__ == "__main__":
    '''
    TODO add requirement.txt and add pytorch, transformers, etc
    '''
    openapi_version = 2
    optim_hyperparams = False

    global DATA_LOADER, VALIDATOR, PARSER, VECTORIZER
    DATA_LOADER, VALIDATOR, PARSER, VECTORIZER = setup(openapi_version, optim_hyperparams)
    
    # # NOTE: RUN RETRIEVAL TASKS:
    # run_experiments(
    #     experiment_type = 'masked_retrieval', 
    #     vectorizer=VECTORIZER, 
    #     parser=PARSER, 
    #     data_loader = DATA_LOADER, 
    #     n_samples=100, 
    #     optim_hyperparams = optim_hyperparams, 
    #     run_tree=True, 
    #     run_text=True, 
    #     run_fuzzy=True)

    # run_experiments(
    #     experiment_type = 'mangled_retrieval', 
    #     vectorizer=VECTORIZER, 
    #     parser=PARSER, 
    #     data_loader = DATA_LOADER, 
    #     n_samples=100, 
    #     optim_hyperparams = optim_hyperparams, 
    #     run_tree=True, 
    #     run_text=True, 
    #     run_fuzzy=True)

    # NOTE: SET UP USER STUDY
    # setup_user_study(
    #     vectorizer=VECTORIZER, 
    #     parser=PARSER, 
    #     data_loader = DATA_LOADER, 
    #     optim_hyperparams = optim_hyperparams, 
    #     n_annotators = 3,
    #     n_sets_per_annotator=10, 
    #     n_retrieved = 5,
    #     n_overlapping_samples = 5, 
    #     rerun_query = True)

    # NOTE: EVALUATE USER STUDY
    # evaluate_user_study()

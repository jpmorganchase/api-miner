# TODO clean up config names
import os
import pandas as pd
import pickle
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy import stats

import ast
from ast import literal_eval
from experiments.utils import get_model, get_text_emb_model
from experiments.records import get_query_info, order_results_by_avg_recall, \
    get_average_recall_of_masked_and_mangled, setup_model_configs
from experiments.configs import ModelConfig, VECTORIZER_FILE, DATASET_INFO_FILE, GRADES_INFO_FILE, MODELS_INFO_PATH, TEXT_EMB_INFO_PATH,\
    EXP_LOGS, PATH_SEP, BACKUP_PATH, PATH_SEP, MODELS_PATH, QUERY_INFO_FILE, SCORE_DIST_PATH, BACKUP_PATH, \
    RECALL_1, RECALL_5, RECALL_10, OVERALL_RECALL, BERT_CONFIG, LEAF_CONFIG, FUZZY_CONFIG, TEXT_CONFIG, TREE_CONFIG, \
    PPMI_CONFIG, AVG_SEARCH_TIME, none_val, PPMI_CONFIG, KEYWORD_EMB_CONFIG, EMB_CONCAT_CONFIG, DATE_RUN, TEXT_EMB_PATH, \
    LAPLACE_K, ALPHA, KEEP_TOP_K, EVALUATION_SET_KEY

from api_miner.models.bert import BERT

def summarize_data_setup():
    print('------------------------------------------------')
    print('DATASET SUMMARY:')
    dataset_df = pd.read_csv(DATASET_INFO_FILE)
    for c in list(dataset_df.columns):
        print(c, ': ', dataset_df[c][0])

    print('------------------------')
    print('GRADES SUMMARY:')
    grades_df = pd.read_csv(GRADES_INFO_FILE)
    for c in list(grades_df.columns):
        print(c, ': ', grades_df[c][0])

def summarize_vectorizer(file_name: str):
    print('------------------------')
    print('VECTORIZER SUMMARY')
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            vectorizer = pickle.load(f)

        print('Number of specs: ', len(vectorizer.spec2indices))
        print('Number of endpoints: ', len(vectorizer.endpoint_to_idx))
        print('Number of leafs: ', len(vectorizer.leaf2idx))
        print('Number of ancestors: ', len(vectorizer.ancestors2idx))
        print('Number of full text contexts: ', len(vectorizer.full_text_contexts))
        print('Number of structural matrices: ', len(vectorizer.structural_matrices))
        print('Dimension of structural matrix: ', vectorizer.structural_matrices[0].shape)
        print('tree_cm shape: ', vectorizer.tree_cm.shape)
        print('text_cm shape: ', vectorizer.text_cm.shape)
        print('leaf_cm shape: ', vectorizer.leaf_cm.shape)

        # tree token distribution
        tree_counts = vectorizer.tree_cm.toarray().sum(axis = 0)
        _plot_histogram(tree_counts, 'Tree tokens histogram')

        text_counts = vectorizer.text_cm.toarray().sum(axis = 0)
        _plot_histogram(text_counts, 'Text tokens histogram')

    else:
        print('Vectorizer not found')

def summarize_model_info(file_name: str):
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        print('vectorizer not found')
        return
    
    model_info_df = pd.read_csv(MODELS_INFO_PATH)

    print('------------------------------------------------')
    print('MODEL INFO SUMMARY:')
    print('Number of models trained: ', len(model_info_df))

    print('------------------------')
    print('FIT TIME SUMMARY:')
    print('Average fit time: ', model_info_df['model_fit_time'].mean())
    print('Min fit time: ', model_info_df['model_fit_time'].min())
    print('Max fit time: ', model_info_df['model_fit_time'].max())

    print('------------------------')
    print('Checking size of embeddings...')
    num_endpoints = len(vectorizer.endpoint_to_idx)
    num_tree_tokens = vectorizer.tree_cm.shape[1]
    num_text_tokens = vectorizer.text_cm.shape[1]
    bert_original_dim = 768
    bert_sent_dim = 384
    # tree emb check
    tree_emb_expected = {
        'tfidf': ['(' + str(num_endpoints) + ', ' + str(num_tree_tokens) + ')'], 
        'ppmi': ['(' + str(num_endpoints) + ', ' + str(num_tree_tokens) + ')'], 
        none_val: [none_val, np.nan, None], 
    }
    model_info_df['tree_check'] = model_info_df.apply(lambda row: True if row['tree_emb'] in tree_emb_expected[row[TREE_CONFIG]] else False, axis = 1)
    
    # text emb check
    text_emb_expected = {
        'tfidf_n.a': ['(' + str(num_endpoints) + ', ' + str(num_text_tokens) + ')'], 
        'ppmi_n.a': ['(' + str(num_endpoints) + ', ' + str(num_text_tokens) + ')'], 
        'bert_bert_original': ['(' + str(num_endpoints) + ', ' + str(bert_original_dim) + ')'], 
        'bert_bert_sent': ['(' + str(num_endpoints) + ', ' + str(bert_sent_dim) + ')'], 
        'ppmi_bert_bert_original': ['(' + str(num_endpoints) + ', ' + str(num_text_tokens + bert_original_dim) + ')'], 
        'ppmi_bert_bert_sent': ['(' + str(num_endpoints) + ', ' + str(num_text_tokens + bert_sent_dim) + ')'], 
        'tfidf_bert_bert_original': ['(' + str(num_endpoints) + ', ' + str(num_text_tokens + bert_original_dim) + ')'], 
        'tfidf_bert_bert_sent': ['(' + str(num_endpoints) + ', ' + str(num_text_tokens + bert_sent_dim) + ')'],
        'n.a_n.a': [none_val, np.nan, None], 
    }
    model_info_df['text_check'] = model_info_df.apply(lambda row: True if row['text_emb'] in text_emb_expected[row[TEXT_CONFIG] + '_' + row[BERT_CONFIG]] else False, axis = 1)


    if False in model_info_df[['tree_check', 'text_check']]:
        print('Incorrect embedding dimensions exist. Check model_info file')
    
    else:
        print('SHAPE OF ALL MODEL EMBEDDINGS ARE AS EXPECTED')
        
def summarize_retrieval_task_formal(experiment_table_file = None, experiment_table_2= None):
    display_cols = [TREE_CONFIG, TEXT_CONFIG, BERT_CONFIG, FUZZY_CONFIG, PPMI_CONFIG, KEYWORD_EMB_CONFIG, EMB_CONCAT_CONFIG]
    if experiment_table_file is None:
        print('Experiment: Average recall of masked and mangled')
        experiment_table = get_average_recall_of_masked_and_mangled(model_config_columns=display_cols)
        display_cols += [OVERALL_RECALL]
    else:
        print('Experiment: ', experiment_table_file)
        experiment_table = pd.read_csv(experiment_table_file)
        order_results_by_avg_recall(experiment_table)
        display_cols += [RECALL_1]

    tree_active = (experiment_table[TREE_CONFIG] != none_val)
    tree_inactive = (experiment_table[TREE_CONFIG] == none_val)
    text_active = (experiment_table[TEXT_CONFIG] != none_val)
    text_inactive = (experiment_table[TEXT_CONFIG] == none_val)
    fuzzy_active = (experiment_table[FUZZY_CONFIG] == 'True' )
    fuzzy_inactive = (experiment_table[FUZZY_CONFIG] == none_val)
    bert_active = (experiment_table[BERT_CONFIG] != none_val)
    bert_inactive = (experiment_table[BERT_CONFIG] == none_val)

    if DATE_RUN in experiment_table.columns:
        complete_results = experiment_table.loc[experiment_table[DATE_RUN] != none_val]
        print('Complete progress: ', str(len(complete_results)), ' / ', len(experiment_table))

    top_results = experiment_table.iloc[:10]
    print('Total experiments run: ', len(experiment_table))
    print('TOP 10 RESULTS')
    print(tabulate(top_results[display_cols], headers = 'keys', tablefmt = 'psql'))
    
    print('------------------------------------------------------------------------')
    print('BOTTOM 10 RESULTS')
    bottom_results = experiment_table.iloc[-10:]
    print(tabulate(bottom_results[display_cols], headers = 'keys', tablefmt = 'psql'))

    print('------------------------------------------------------------------------')
    print('TEXT ONLY RESULTS')
    text_only = experiment_table.loc[text_active & tree_inactive  & fuzzy_inactive]
    print(tabulate(text_only[display_cols], headers = 'keys', tablefmt = 'psql'))

    print('------------------------------------------------------------------------')
    print('TREE ONLY RESULTS')
    tree_only = experiment_table.loc[text_inactive & tree_active & fuzzy_inactive]
    print(tabulate(tree_only[display_cols], headers = 'keys', tablefmt = 'psql'))

    print('------------------------------------------------------------------------')
    print('FUZZY ONLY RESULTS')
    fuzzy_only = experiment_table.loc[text_inactive & tree_inactive & fuzzy_active]
    print(tabulate(fuzzy_only[display_cols], headers = 'keys', tablefmt = 'psql'))
    
    print('------------------------------------------------------------------------')
    print('FUZZY + TREE ONLY RESULTS')
    fuzzy_tree = experiment_table.loc[text_inactive & tree_active & fuzzy_active]
    print(tabulate(fuzzy_tree[display_cols], headers = 'keys', tablefmt = 'psql'))

    print('------------------------------------------------------------------------')
    print('FUZZY + TEXT ONLY RESULTS')
    fuzzy_text = experiment_table.loc[text_active & tree_inactive & fuzzy_active]
    print(tabulate(fuzzy_text[display_cols], headers = 'keys', tablefmt = 'psql'))

    print('------------------------------------------------------------------------')
    print('TREE + TEXT ONLY RESULTS')
    tree_text = experiment_table.loc[text_active & tree_active & fuzzy_inactive]
    print(tabulate(tree_text[display_cols], headers = 'keys', tablefmt = 'psql'))

    print('------------------------------------------------------------------------')
    print('TREE + TEXT + FUZZY ONLY RESULTS')
    tree_text_fuzzy = experiment_table.loc[text_active & tree_active & fuzzy_active]
    print(tabulate(tree_text_fuzzy[display_cols], headers = 'keys', tablefmt = 'psql'))

def _convert_str_list_to_array(l):
    if type(l) == str:
        return np.asarray([ float(n) for n in l.replace('[', '').replace(']', '').replace('\n', '').split(' ') if n != ''])
    else:
        return np.asarray([])

def summarize_query_info(experiment_type: str):
    query = get_query_info(experiment_type=experiment_type, set_type= EVALUATION_SET_KEY)
    
    if query:
        original_endpoints = query[0]
        query_endpoints = query[1]
        original_specs = query[2]
        query_specs = query[3]

        print('QUERY INFO: ')
        print('num_samples: ', len(query_endpoints))

        for i in range(len(query_endpoints)):
            query_endpoint = query_endpoints[i]
            original_endpoint = original_endpoints[i]

            print('Endpoint: ', query_endpoint)
            print('---------------------------------------------')
            print('ORIGINAL PATHS: ')

            operations, responses = _find_operation_responses(original_specs[i]['paths'][original_endpoint])
            print('Operations: ', len(operations), ' = ')
            print('Responses: ', len(responses), ' = ')

            print(original_specs[i]['paths'])
            print('--------------------')
            print('MASKED PATHS: ')

            operations, responses = _find_operation_responses(query_specs[i]['paths'][query_endpoint])
            print('Operations: ', len(operations), ' = ')
            print('Responses: ', len(responses), ' = ')
            print(query_specs[i]['paths'])
            
            if experiment_type != 'user_study':
                print('---------------------------------------------')
                print('ORIGINAL DEFNS: ')
                defns, props = _find_definition_properties(original_specs[i])
                print('Definitions: ', len(defns), ' = ')
                print('Properties: ', len(props), ' = ')

                # for d, d_obj in original_specs[i]['definitions'].items():
                #     print(original_specs[i]['definitions'])
                #     print('--------------------')
                #     break
                print('MASKED DEFNS: ')
                defns, props = _find_definition_properties(query_specs[i])
                print('Definitions: ', len(defns), ' = ')
                print('Properties: ', len(props), ' = ')

                # for d, d_obj in query_specs[i]['definitions'].items():
                #     print(query_specs[i]['definitions'])
                #     print('--------------------')
                #     break 

            break

def _find_definition_properties(spec:dict):
    defns = list(spec['definitions'].keys())
    props = []
    for d in defns:
        if 'properties' in spec['definitions'][d]:
            props += list(spec['definitions'][d]['properties'].keys())
    return defns, props

def _find_operation_responses(path_spec: dict):
    operations = list(path_spec.keys())
    responses = []
    for o in operations:
        if 'responses' in list(path_spec[o]):
            responses += list(path_spec[o]['responses'].keys())

    return operations, responses

def _plot_bar_graphs(x: np.array, y: np.array, title: str, dim_threshold= None):
    plt.ylim([min(y), max(y)])
    plt.bar(x,y)
    if dim_threshold:
        plt.axvline(x=dim_threshold, ymin=min(y), ymax=max(y), color = 'r')
    plt.xlabel('embedding dimension')
    plt.ylabel('embedding value')
    plt.title(title)
    plt.show()

def _plot_histogram(frequencies: np.array, title: str):
    plt.hist(frequencies, bins='auto', range = (0, 1000))
    plt.gca().set(title=title, ylabel='Frequency')
    plt.show()

if __name__ == "__main__":
    experiment_type = 'masked_retrieval'
    vectorizer_file = VECTORIZER_FILE
    experiment_file =  EXP_LOGS + PATH_SEP + experiment_type + '.csv'
    df = pd.read_csv(TEXT_EMB_INFO_PATH)
  
    summarize_query_info(experiment_type)
    summarize_retrieval_task_formal(experiment_file)
    summarize_retrieval_task_formal() # for average recall
    
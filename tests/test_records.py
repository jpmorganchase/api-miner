import sys
sys.path.append('..')
import numpy as np
import os
import pandas as pd

from experiments.records import setup_retrieval_tasks_table, get_text_embeddings_configs
from experiments.configs import EXP_LOGS, PATH_SEP, TREE_CONFIG, TEXT_CONFIG, BERT_CONFIG, FUZZY_CONFIG, PPMI_CONFIG,\
    LAPLACE_K, ALPHA, KEEP_TOP_K, MIN_MAX_SCALE, DATE_RUN, TREE_WEIGHT, TEXT_WEIGHT, FUZZY_WEIGHT, \
    QUALITY_WEIGHT, EXP_ID, QUERY_INFO_FILE, DATE_RUN, EXPERIMENT_TYPE_KEY, QUERY_ENDPOINTS_KEY, QUERY_SPECS_KEY, \
    ORIGINAL_SPECS_KEY, NUM_SAMPLES, SCORE_DIST_PATH, ORIGINAL_ENDPOINTS_KEY, BACKUP_PATH, RECALL_1, RECALL_5, \
    RECALL_10, OVERALL_RECALL, USER_STUDY_KEY, KEYWORD_EMB_CONFIG, EMB_CONCAT_CONFIG

from api_miner.models.configs import none_val, tfidf_bert, \
    ppmi_bert, bert, ppmi, tfidf, avg_emb, bert_sent, \
    bert_original, fuzzy_true, t_svd, cca, concat 

def test_setup_retrieval_tasks_table_tree():
    experiment_table = setup_retrieval_tasks_table(
        run_tree=True, 
        run_text=False, 
        run_fuzzy=False, 
        optim_hyperparams = False)
    
    assert(set(experiment_table[TREE_CONFIG].unique()) == {tfidf, ppmi} )
    assert(set(experiment_table[TEXT_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[FUZZY_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[BERT_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[PPMI_CONFIG].unique()) == {avg_emb, none_val})
    assert(set(experiment_table[KEYWORD_EMB_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[EMB_CONCAT_CONFIG].unique()) == {none_val})
    assert( set(experiment_table[TREE_WEIGHT].unique()) == {0.9} )
    assert( set(experiment_table[QUALITY_WEIGHT].unique()) == {0.1} )
    assert( set(experiment_table[TEXT_WEIGHT].unique()) == {0.0} )
    assert( set(experiment_table[FUZZY_WEIGHT].unique()) == {0.0} )
    assert( len(set(experiment_table[EXP_ID].unique())) == len(experiment_table[EXP_ID].unique()) ) # exp id is unique
    # check no duplicate tests
    assert(len(experiment_table) == len(experiment_table.drop_duplicates()))

    # tests with ppmi should be avg_emb
    ppmi_exp = experiment_table.loc[experiment_table[TREE_CONFIG] == ppmi]
    assert(set(ppmi_exp[PPMI_CONFIG].unique()) == { avg_emb})

    # tests without ppmi should have none_val
    non_ppmi_exp = experiment_table.loc[(experiment_table[TREE_CONFIG] != ppmi)]
    assert(set(non_ppmi_exp[PPMI_CONFIG].unique()) == {none_val})
    
def test_setup_retrieval_tasks_table_text():
    experiment_table = setup_retrieval_tasks_table(
        run_tree=False, 
        run_text=True, 
        run_fuzzy=False, 
        optim_hyperparams = False)
    
    assert( set(experiment_table[TREE_CONFIG].unique()) == {none_val} )
    assert( set(experiment_table[TEXT_CONFIG].unique()) == {tfidf, ppmi, bert, ppmi_bert, tfidf_bert})
    assert( set(experiment_table[FUZZY_CONFIG].unique()) == {none_val})
    assert( set(experiment_table[BERT_CONFIG].unique()) == {bert_original, bert_sent, none_val})
    assert(set(experiment_table[PPMI_CONFIG].unique()) == {avg_emb, none_val})
    assert(set(experiment_table[KEYWORD_EMB_CONFIG].unique()) == {t_svd, none_val})
    assert(set(experiment_table[EMB_CONCAT_CONFIG].unique()) == {cca, concat , none_val})
    assert( set(experiment_table[TREE_WEIGHT].unique()) == {0.0} )
    assert( set(experiment_table[QUALITY_WEIGHT].unique()) == {0.1} )
    assert( set(experiment_table[TEXT_WEIGHT].unique()) == {0.9} )
    assert( set(experiment_table[FUZZY_WEIGHT].unique()) == {0.0} )
    assert( len(set(experiment_table[EXP_ID].unique())) == len(experiment_table[EXP_ID].unique()) ) # exp id is unique
    # check no duplicate tests
    assert(len(experiment_table) == len(experiment_table.drop_duplicates()))

    # tests with ppmi should be avg_emb
    ppmi_exp = experiment_table.loc[(experiment_table[TEXT_CONFIG] == ppmi) | (experiment_table[TEXT_CONFIG] == ppmi_bert)]
    assert(set(ppmi_exp[PPMI_CONFIG].unique()) == {avg_emb})

    # tests without ppmi should have none_val
    non_ppmi_exp = experiment_table.loc[(experiment_table[TEXT_CONFIG] != ppmi) & (experiment_table[TEXT_CONFIG] != ppmi_bert)]
    assert(set(non_ppmi_exp[PPMI_CONFIG].unique()) == {none_val})

    # tests with single emb -> KEYWORD_EMB_CONFIG = none_val, EMB_CONCAT_CONFIG = none_val
    single_emb = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == ppmi) | (experiment_table[TEXT_CONFIG] == tfidf)]
    assert(set(single_emb[KEYWORD_EMB_CONFIG].unique()) == {none_val})
    assert(set(single_emb[EMB_CONCAT_CONFIG].unique()) == {none_val})
    
    # tests with text double emb combine -> KEYWORD_EMB_CONFIG = {t_svd, none_val}, EMB_CONCAT_CONFIG = {cca, concat}
    double_emb = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == ppmi_bert) | (experiment_table[TEXT_CONFIG] == tfidf_bert)]
    assert(set(double_emb[KEYWORD_EMB_CONFIG].unique()) == {t_svd, none_val})
    assert(set(double_emb[EMB_CONCAT_CONFIG].unique()) == {cca, concat})

def test_setup_retrieval_tasks_table_fuzzy():
    experiment_table = setup_retrieval_tasks_table(
        run_tree=False, 
        run_text=False, 
        run_fuzzy=True, 
        optim_hyperparams = False)
    
    assert(set(experiment_table[TREE_CONFIG].unique()) == {none_val} )
    assert(set(experiment_table[TEXT_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[FUZZY_CONFIG].unique()) == {fuzzy_true})
    assert(set(experiment_table[BERT_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[PPMI_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[PPMI_CONFIG].unique()) == {none_val})    
    assert(set(experiment_table[KEYWORD_EMB_CONFIG].unique()) == {none_val})
    assert(set(experiment_table[EMB_CONCAT_CONFIG].unique()) == {none_val})
    assert( set(experiment_table[TREE_WEIGHT].unique()) == {0.0} )
    assert( set(experiment_table[QUALITY_WEIGHT].unique()) == {0.1} )
    assert( set(experiment_table[TEXT_WEIGHT].unique()) == {0.0} )
    assert( set(experiment_table[FUZZY_WEIGHT].unique()) == {0.9} )
    assert( len(set(experiment_table[EXP_ID].unique())) == len(experiment_table[EXP_ID].unique()) ) # exp id is unique
    # check no duplicate tests
    assert(len(experiment_table) == len(experiment_table.drop_duplicates()))

def test_setup_retrieval_tasks_table_combo():
    experiment_table = setup_retrieval_tasks_table(
        run_tree=True, 
        run_text=True, 
        run_fuzzy=True, 
        optim_hyperparams = False)
    
    assert(set(experiment_table[TREE_CONFIG].unique()) == {tfidf, ppmi, none_val} )
    assert(set(experiment_table[TEXT_CONFIG].unique()) == {tfidf, ppmi, tfidf_bert, ppmi_bert,bert, none_val})
    assert(set(experiment_table[FUZZY_CONFIG].unique()) == {fuzzy_true, none_val})
    assert(set(experiment_table[BERT_CONFIG].unique()) == {bert_original, bert_sent, none_val})
    assert(set(experiment_table[PPMI_CONFIG].unique()) == {avg_emb, none_val})
    assert(set(experiment_table[KEYWORD_EMB_CONFIG].unique()) == {t_svd, none_val})
    assert(set(experiment_table[EMB_CONCAT_CONFIG].unique()) == {cca, concat , none_val})
    assert( set(experiment_table[QUALITY_WEIGHT].unique()) == {0.1} )
    # check no duplicate tests
    assert(len(experiment_table) == len(experiment_table.drop_duplicates()))

    # NOTE some add close to 0.8999 when split 3 ways
    sum_weights = experiment_table[[TREE_WEIGHT, TEXT_WEIGHT, FUZZY_WEIGHT]].sum(axis = 1).unique()
    assert ( np.asarray([np.isclose(s, 0.9) for s in sum_weights]).all() ) 
    assert( len(set(experiment_table[EXP_ID].unique())) == len(experiment_table[EXP_ID].unique()) ) # exp id is unique

    # tests with ppmi should be avg_emb
    tree_ppmi = (experiment_table[TREE_CONFIG] == ppmi)
    text_ppmi = (experiment_table[TEXT_CONFIG] == ppmi) | (experiment_table[TEXT_CONFIG] == ppmi_bert)
    ppmi_exp = experiment_table.loc[tree_ppmi | text_ppmi ]
    assert(set(ppmi_exp[PPMI_CONFIG].unique()) == {avg_emb})

    # tests without ppmi should have none_val
    tree_ppmi = (experiment_table[TREE_CONFIG] != ppmi)
    text_ppmi = (experiment_table[TEXT_CONFIG] != ppmi) & (experiment_table[TEXT_CONFIG] != ppmi_bert)
    non_ppmi_exp = experiment_table.loc[tree_ppmi & text_ppmi ]
    assert(set(non_ppmi_exp[PPMI_CONFIG].unique()) == {none_val})

    # tests without text -> KEYWORD_EMB_CONFIG = none_val, EMB_CONCAT_CONFIG = none_val
    no_text = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == none_val)]
    assert(set(no_text[KEYWORD_EMB_CONFIG].unique()) == {none_val})
    assert(set(no_text[EMB_CONCAT_CONFIG].unique()) == {none_val})

    # tests with single emb -> KEYWORD_EMB_CONFIG = none_val, EMB_CONCAT_CONFIG = none_val
    single_emb = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == ppmi) | (experiment_table[TEXT_CONFIG] == tfidf)]
    assert(set(single_emb[KEYWORD_EMB_CONFIG].unique()) == {none_val})
    assert(set(single_emb[EMB_CONCAT_CONFIG].unique()) == {none_val})
    
    # tests with text double emb combine -> KEYWORD_EMB_CONFIG = {t_svd, none_val}, EMB_CONCAT_CONFIG = {cca, concat}
    double_emb = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == ppmi_bert) | (experiment_table[TEXT_CONFIG] == tfidf_bert)]
    assert(set(double_emb[KEYWORD_EMB_CONFIG].unique()) == {t_svd, none_val})
    assert(set(double_emb[EMB_CONCAT_CONFIG].unique()) == {cca, concat})

def test_query_info():
    '''
    Make sure there is at most 1 query in record per unique experiment
    '''
    if os.path.isfile(QUERY_INFO_FILE):
        query_info = pd.read_csv(QUERY_INFO_FILE)
        query_masked = query_info.loc[query_info[EXPERIMENT_TYPE_KEY] == 'masked']
        assert(len(query_masked) == 1 or len(query_masked) == 0)

        query_mangled = query_info.loc[query_info[EXPERIMENT_TYPE_KEY] == 'mangled']
        assert(len(query_mangled) == 1 or len(query_mangled) == 0)

        query_user_study = query_info.loc[query_info[EXPERIMENT_TYPE_KEY] == 'user_study']
        assert(len(query_user_study) == 1 or len(query_user_study) == 0)

def test_get_text_embeddings_configs():
    experiment_table = get_text_embeddings_configs()

    # check no duplicate tests
    assert(len(experiment_table) == len(experiment_table.drop_duplicates()))

    # tests with ppmi should be avg_emb
    ppmi_exp = experiment_table.loc[(experiment_table[TEXT_CONFIG] == ppmi) | (experiment_table[TEXT_CONFIG] == ppmi_bert)]
    assert(set(ppmi_exp[PPMI_CONFIG].unique()) == {avg_emb})

    # tests without ppmi should have none_val
    non_ppmi_exp = experiment_table.loc[(experiment_table[TEXT_CONFIG] != ppmi) & (experiment_table[TEXT_CONFIG] != ppmi_bert)]
    assert(set(non_ppmi_exp[PPMI_CONFIG].unique()) == {none_val})

    # tests with single emb -> KEYWORD_EMB_CONFIG = none_val, EMB_CONCAT_CONFIG = none_val
    single_emb = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == ppmi) | (experiment_table[TEXT_CONFIG] == tfidf)]
    assert(set(single_emb[KEYWORD_EMB_CONFIG].unique()) == {none_val})
    assert(set(single_emb[EMB_CONCAT_CONFIG].unique()) == {none_val})
    
    # tests with text double emb combine -> KEYWORD_EMB_CONFIG = {t_svd, none_val}, EMB_CONCAT_CONFIG = {cca, concat}
    double_emb = experiment_table.loc[ (experiment_table[TEXT_CONFIG] == ppmi_bert) | (experiment_table[TEXT_CONFIG] == tfidf_bert)]
    assert(set(double_emb[KEYWORD_EMB_CONFIG].unique()) == {t_svd, none_val})
    assert(set(double_emb[EMB_CONCAT_CONFIG].unique()) == {cca, concat})

if __name__ == "__main__":
    test_get_text_embeddings_configs()
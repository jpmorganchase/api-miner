import code
import json
import numpy as np
from collections import namedtuple
from experiments.utils import transform_endpoint_name_to_features, transform_specs_to_features, retrieve_endpoints_from_database, normalize_scores
from experiments.setup import initialize_demo
from experiments.sample_generator import MaskedSpecGenerator, MangledSpecGenerator, EndpointNameGenerator, UserStudySpecGenerator

def endpoint_demo():
    generator = EndpointNameGenerator(VECTORIZER)
    endpoint = generator.get_randoms(num_samples = 1)[0]
    print('----------------------------------------------------------------------------------------')
    print('USER QUERY: ', endpoint)
    query_feature = transform_endpoint_name_to_features(endpoint_name=endpoint, vectorizer=VECTORIZER)
    display_retrieved_endpoints(query_feature=query_feature, query= endpoint)

def endpoint_modified_spec_demo(modify_type: str, n_retrieved=5):
    if modify_type == 'masked':
        generator = MaskedSpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    elif modify_type == 'mangled':
        generator = MangledSpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    elif modify_type == 'early_draft':
        generator = UserStudySpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    else: 
        print('Invalid modification type. Please choose from: masked, mangled')
        
    original_endpoints, modified_endpoints, original_specs, modified_specs = generator.get_randoms(num_samples=1)
    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')
    print('USER QUERY: ')
    print('Endpoint name: ', modified_endpoints[0])
    print(json.dumps(modified_specs[0], sort_keys=True, indent=4))
    query_feature = transform_specs_to_features(specs = modified_specs, parser = PARSER, vectorizer = VECTORIZER)
    display_retrieved_endpoints(query_feature=query_feature, query =  modified_specs[0])
 
def display_retrieved_endpoints(query_feature, query, n_retrieved=5):
    
    retrieved_endpoint_names_per_endpoint, top_similarity_scores_per_endpoint = retrieve_endpoints_from_database(
            model= MODEL, 
            vectorizer= VECTORIZER, 
            query_feature = query_feature, 
            config= DEMO_CONFIG, 
            n_retrieved=n_retrieved, 
            retrieve_specs = True)

    retrieved_endpoints = retrieved_endpoint_names_per_endpoint[0]
    relevance_scores = normalize_scores(top_similarity_scores_per_endpoint[0])

    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')
    print("RETRIEVED ENDPOINTS: ")
    for i in range(len(retrieved_endpoints)):
        e = retrieved_endpoints[i][0]
        score = relevance_scores[i]
        print('Endpoint name: ', e['endpoint_name'])
        print('Relevance score: ', score)
        print('Quality: ', e['quality_score'])
        print('File: ', e['from_spec'])
        print(json.dumps(e['content'], sort_keys=True, indent=4))
        print('----------------------------------------------------------------------------------------')
           
if __name__ == "__main__":
    global DATA_LOADER, VALIDATOR, PARSER, VECTORIZER, MODEL, DEMO_CONFIG
    DATA_LOADER, VALIDATOR, PARSER, VECTORIZER, MODEL, DEMO_CONFIG = initialize_demo()

    banner = """
    Interactive API Spector Demo
    
    endpoint_demo()
    - Generates a random endpoint from the database, and retrieves relevant endpoint specs

    endpoint_modified_spec_demo(modify_type='early_draft')
    - Replicate an early draft of an endpoint spec by removing large sections and masking tokens, and retrieves relevant endpoint specs

    endpoint_modified_spec_demo(modify_type='masked')
    - Modifies random endpoint spec from the database by randomly removing sections and masking tokens, and retrieves relevant endpoint specs

    endpoint_modified_spec_demo(modify_type='mangled')
    - Modifies random endpoint spec from the database by randomly removing sections and mangling tokens, and retrieves relevant endpoint specs
    """
    code.interact(banner=banner, local=locals())
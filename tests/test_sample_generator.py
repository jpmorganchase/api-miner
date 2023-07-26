import sys
sys.path.append('..')
import numpy as np
import copy
from collections import Counter
import json

from experiments.sample_generator import MaskedSpecGenerator, MangledSpecGenerator, UserStudySpecGenerator
from api_miner.data_processing.vectorize import APIVectorizer
from setup import PARSER

DATA_LOADER = ''
VECTORIZER = APIVectorizer()

spec = {
        "paths":{
            "sallymoon1":{
                "get":{
                    "description": "I like cheese", 
                    "parameters": [{"name": "blue"}, {"name": "green"}], 
                    "responses":{
                        "200":{
                            "description" : "I am response 200"
                        }, 
                        "300":{
                            "description" : "I am response 300"
                        }
                    }
                }, 
                "post":{
                    "description": "I like cheese", 
                    "parameters": [{"name": "blue"}, {"name": "green"}], 
                    "responses":{
                        "200":{
                            "description" : "I am response 200"
                        }, 
                        "300":{
                            "description" : "I am response 300"
                        }
                    }
                }
            }
        }, 
        "definitions":{
            "defn1":{
                "properties":{
                    "a1": [], 
                    "a2": [], 
                    "a3": [], 
                }
            }, 
            "defn2":{
                "properties":{
                    "a1": [], 
                    "a2": [], 
                    "a3": [], 
                }
            }
        }
    }

sample_spec = {
    "paths":{
        "main/user/{id}":{
            "get":{
                "description": "Retrieve website main page. Provide user information. ", 
                "parameters": [{"name": "id", "description": "Unique id of the user that is used to validate"}, {"name": "user_name", "description": "name of the user, nickname or full name"}], 
                "responses":{
                    "200":{
                        "description" : "I am response 200"
                    }, 
                    "300":{
                        "description" : "I am response 300"
                    }
                }
            }, 
            "post":{
                "description": "Post images to the website. Provide post info and comments. ", 
                "parameters": [{"name": "image", "description": "image in jpeg format that can be edited and cropped"}, {"date": "green"}], 
                "responses":{
                    "200":{
                        "description" : "I am response 200"
                    }, 
                    "300":{
                        "description" : "I am response 300"
                    }
                }
            }
        }
    }, 
    "definitions":{
        "defn1":{
            "properties":{
                "a1": [], 
                "a2": [], 
                "a3": [], 
            }
        }, 
        "defn2":{
            "properties":{
                "a1": [], 
                "a2": [], 
                "a3": [], 
            }
        }
    }
}

sample_spec_endpoint_name = 'main/user/{id}'
spec_endpoint_name = 'sallymoon1'

def test_mask_spec():
    '''
    Test MaskedSpecGenerator
    paths masking:
    - remove random percent of operations
    - for each operation: 
        - mask description or summary by generator.percent_replace with generator.mask_token 
        - delete percent of the responses and mask description of the other percent
    definition masking:
    - remove random percent of definition
    - for each definition:
        - remove generator.percent_replace of properties 
    '''
    generator = MaskedSpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    
    modified_spec = copy.deepcopy(spec)
    _ = generator.modify_spec(modified_spec, spec_endpoint_name)

    # endpoint name masking
    original_path_name = list(spec['paths'].keys())[0]
    modified_path_name = list(modified_spec['paths'].keys())[0]
    assert( len(list(modified_path_name)) == len(original_path_name) -  int(np.floor(len(original_path_name) * generator.percent_endpoint_name_replace)) ) 


    # PATHS MASKING
    # percent of the operations are deleted
    operations = list(modified_spec['paths'][modified_path_name].keys())
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name]) ))
    assert(len(spec['paths'][original_path_name]) - num_to_drop == len(operations))
    operation_name = operations[0]

    # mask description of operation
    num_tokens_to_mask = int(np.floor(generator.percent_replace *  len(spec['paths'][original_path_name][operation_name]['description'].split(' '))))
    assert(modified_spec['paths'][modified_path_name][operation_name]['description'].count(generator.mask_token) == num_tokens_to_mask)
    
    # remove percent of the responses
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name][operation_name]['responses']) ))
    assert( len(spec['paths'][original_path_name][operation_name]['responses']) - num_to_drop == len(modified_spec['paths'][modified_path_name][operation_name]['responses']))

    # mask description of the other percent of responses
    response_modified = list(modified_spec['paths'][modified_path_name][operation_name]['responses'].keys())[0]
    num_tokens_to_mask = int(np.floor(generator.percent_replace *  len(spec['paths'][original_path_name][operation_name]['responses'][response_modified]['description'].split(' '))))
    assert(modified_spec['paths'][modified_path_name][operation_name]['responses'][response_modified]['description'].count(generator.mask_token) == num_tokens_to_mask)

    # remove percent of the parameters
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name][operation_name]['parameters']) ))
    assert(len(spec['paths'][original_path_name][operation_name]['parameters'])- num_to_drop == len(modified_spec['paths'][modified_path_name][operation_name]['parameters']))

    # DEFN MASKING
    # percent of the definitions are deleted
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['definitions']) ))
    definitions = list(modified_spec['definitions'].keys())
    assert(len(definitions) == num_to_drop)
    defn_name = definitions[0]

    # properties of every defn remaining are dropped
    num_properties_to_drop = int(np.floor(generator.percent_replace *  len(spec['definitions'][defn_name]['properties']))) 
    assert(len(modified_spec['definitions'][defn_name]['properties']) == len(spec['definitions'][defn_name]['properties']) - num_properties_to_drop)

def test_mangled_spec():
    generator = MangledSpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    modified_spec = copy.deepcopy(spec)
    _ = generator.modify_spec(modified_spec, spec_endpoint_name)

    # endpoint name masking
    original_path_name = list(spec['paths'].keys())[0]
    modified_path_name = list(modified_spec['paths'].keys())[0]
    num_chars_to_mangle = int(np.floor(generator.percent_endpoint_name_mangle * len(original_path_name)))
    num_mangled = 0
    for i in range(len(list(modified_path_name))):
        if modified_path_name[i] != original_path_name[i]:
            num_mangled += 1
    
    assert( num_chars_to_mangle ==  num_mangled) 

    # PATHS MANGLING
    # percent of the operations are deleted
    operations = list(modified_spec['paths'][modified_path_name].keys())
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name]) ))
    assert(len(spec['paths'][original_path_name]) - num_to_drop == len(operations))
    operation_name = operations[0]

    # mangle description of operation = all but num_tokens_to_mangle should be same token
    num_tokens_to_mangle = int(np.floor(generator.percent_mangle *  len(spec['paths'][original_path_name][operation_name]['description'].split(' '))))
    modified_tokens= set(modified_spec['paths'][modified_path_name][operation_name]['description'].split(' '))
    original_tokens = set(spec['paths'][original_path_name][operation_name]['description'].split(' '))
    
    assert( len(modified_tokens.intersection(original_tokens)) == len(modified_tokens) - num_tokens_to_mangle)
    
    # mangle description of one of the response = all but num_tokens_to_mangle should be same token
    responses= list(modified_spec['paths'][modified_path_name][operation_name]['responses'].keys())
    
    # remove percent of the responses
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name][operation_name]['responses']) ))
    assert( len(spec['paths'][original_path_name][operation_name]['responses']) - num_to_drop == len(modified_spec['paths'][modified_path_name][operation_name]['responses']))
    
    # mangle the other percent
    response_modified = list(modified_spec['paths'][modified_path_name][operation_name]['responses'].keys())[0]
    num_tokens_to_modify = int(np.floor(generator.percent_mangle *  len(spec['paths'][original_path_name][operation_name]['responses'][response_modified]['description'].split(' '))))
    modified_tokens = set(modified_spec['paths'][modified_path_name][operation_name]['responses'][response_modified]['description'].split(' '))
    original_tokens = set(spec['paths'][original_path_name][operation_name]['responses'][response_modified]['description'].split(' '))
    assert(len(modified_tokens.intersection(original_tokens)) == len(modified_tokens) - num_tokens_to_modify)

    # remove percent of the parameters
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name][operation_name]['parameters']) ))
    assert(len(spec['paths'][original_path_name][operation_name]['parameters'])- num_to_drop == len(modified_spec['paths'][modified_path_name][operation_name]['parameters']))

    # DEFN MANGLING
    # percent of the definitions are deleted
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['definitions']) ))
    definitions = list(modified_spec['definitions'].keys())
    assert(len(definitions) == num_to_drop)
    defn_name = definitions[0]

    # mangle num_properties_to_modify of properties
    num_properties_to_modify = int(np.floor(generator.percent_mangle *  len(spec['definitions'][defn_name]['properties']))) 
    original_properties = set(spec['definitions'][defn_name]['properties'])
    modified_properties = set(modified_spec['definitions'][defn_name]['properties'])
    assert( len(modified_properties.intersection(original_properties)) == len(modified_properties) - num_properties_to_modify)

def test_user_study_modification():
    generator = UserStudySpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    
    modified_spec = copy.deepcopy(spec)
    _ = generator.modify_spec(modified_spec, spec_endpoint_name)

    # endpoint name masking
    original_path_name = list(spec['paths'].keys())[0]
    modified_path_name = list(modified_spec['paths'].keys())[0]
    assert( len(list(modified_path_name)) ==  int(np.floor(len(original_path_name) * generator.percent_endpoint_name_keep)) ) 

    # percent of the operations are deleted
    operations = list(modified_spec['paths'][modified_path_name].keys())
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name]) ))
    assert(len(spec['paths'][original_path_name]) - num_to_drop == len(operations))
    operation_name = operations[0]

    # remove percent of the parameters
    num_to_drop = int(np.floor(generator.drop_percent * len(spec['paths'][original_path_name][operation_name]['parameters']) ))
    assert(len(spec['paths'][original_path_name][operation_name]['parameters']) - num_to_drop == len(modified_spec['paths'][modified_path_name][operation_name]['parameters']))

    # responses are deleted
    assert('responses' not in list(modified_spec['paths'][modified_path_name][operation_name].keys()))
    
    # definitions are deleted
    assert('definitions' not in list(modified_spec.keys()))

def view_add_spelling_error():
    generator = MangledSpecGenerator(VECTORIZER, PARSER)
    word = 'dolphins'
    output = generator.add_spelling_error(word)
    assert(len(Counter(word) - Counter(output))== 1) # differ by 1 letter type
    assert(list(   (Counter(word) - Counter(output)).values()   )[0] == 1) # differ by 1 char of that letter

    word = 'hhh'
    output = generator.add_spelling_error(word)
    assert(len(Counter(word) - Counter(output))== 1) # differ by 1 letter type
    assert(list(   (Counter(word) - Counter(output)).values()   )[0] == 1) # differ by 1 char of that letter


def view_find_synonymn():
    generator = MangledSpecGenerator(VECTORIZER, PARSER)
    original = ['active', 'dog', 'cheese', 'api', 'analysis', '100', 'ping', 'ml', 'RESTful', 'JSON', 'http', 'post', 'real-time', 'license', 'url', 'parameters', 'application']

    for o in original:
        print(o,  ' --> ', generator.find_synonymn(o))


def view_masked_example():
    generator = MaskedSpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    
    modified_spec = copy.deepcopy(sample_spec)
    _ = generator.modify_spec(modified_spec, sample_endpoint_name)

    print('ORIGINAL SPEC')
    print(json.dumps(sample_spec['paths'], indent=4, sort_keys=True))
    
    print('-------------------------------------------')
    print('MODIFIED SPEC')
    print(json.dumps(modified_spec['paths'], indent=4, sort_keys=True))

def view_mangled_example():
    generator = MangledSpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    
    modified_spec = copy.deepcopy(sample_spec)
    _ = generator.modify_spec(modified_spec, sample_endpoint_name)

    print('ORIGINAL SPEC')
    print(json.dumps(sample_spec['paths'], indent=4, sort_keys=True))
    
    print('-------------------------------------------')
    print('MODIFIED SPEC')
    print(json.dumps(modified_spec['paths'], indent=4, sort_keys=True))

def view_user_study_example():
    generator = UserStudySpecGenerator(VECTORIZER, PARSER, DATA_LOADER)
    
    modified_spec = copy.deepcopy(sample_spec)
    _ = generator.modify_spec(modified_spec, sample_endpoint_name)
    
    print('ORIGINAL SPEC')
    print(json.dumps(sample_spec['paths'], indent=4, sort_keys=True))
    
    print('-------------------------------------------')
    print('MODIFIED SPEC')
    print(json.dumps(modified_spec['paths'], indent=4, sort_keys=True))

if __name__ == "__main__":
    view_mangled_example()

import sys
sys.path.append('..')
from collections import defaultdict
from setup import SAMPLE_SPEC, OPENAPI_CONFIGS, PARSER

def test_featurize_endpoints_in_spec():
    '''
    Check for return datatypes given sample spec
    '''
    endpoint_objects = PARSER.featurize_endpoints_in_spec(SAMPLE_SPEC)
    obj = endpoint_objects['/list.json']
    assert(type(obj.content) ==  dict)
    assert(type(obj.from_spec) == str)
    assert(type(obj.tree_context) == str)
    assert(type(obj.full_text_context) == str)
    assert(type(obj.text_context) == str)

def test_get_context_tokens():
    '''
    Parse correct contexts given endpoint spec
    '''
    props_per_defn = {
        'leaf': ['p1']
    }
    endpoint_spec = {
        "get": {
            "description": "I like leafs\n", 
            "responses": {"200": {"description": "OK", "schema": {"$ref": "#/definitions/leaf"}}}, 
            "tags": ["leaf"], 
            "summary": "Do you like cheese?"
        }
    }
    tree_context, text_context, full_text_context = PARSER.get_context_tokens(endpoint_spec=endpoint_spec, props_per_defn=props_per_defn)
    
    expected_tree_context = {'responses_200', 'get_responses_200_p1', 'get_responses_200_leaf', 'tags_leaf'}
    assert(set(tree_context) == expected_tree_context)

    expected_text_context = {'like leaf', 'like cheese'}
    assert(set(text_context) == expected_text_context)

    expected_full_text_context = {'I like leafs', 'Do you like cheese?'}
    assert(set(full_text_context) == expected_full_text_context)
    
def test_find_properties_and_refs():
    '''
    Retrieve direct properties and references for every defn in defn obj
    - using sample spec
    - only defns with properties get added 
    '''
    sample_defn_obj = SAMPLE_SPEC.get(OPENAPI_CONFIGS.defn_key)
    props_refs_per_defn = PARSER.find_properties_and_refs(sample_defn_obj)

    expected_keys = {'API', 'ApiVersion', 'Metrics'}
    assert(set(props_refs_per_defn.keys()) == expected_keys)

    defn = "API"
    props = {"added", "preferred", "versions"}
    refs = {"ApiVersion"}
    
    assert(props_refs_per_defn[defn]['references'] == refs)
    assert(props_refs_per_defn[defn]['properties'] == props)

def test_get_definition_properties_circular():
    '''
    Get propeties of all defn, including properties from nested references
    - Every defn in circular dependency should have properties from all defns
    '''
    defn_circular = {
        "Apple": {
            "properties": {"apple": {"description": "", "schema": {"$ref": "#/definitions/Orange"}}}
        }, 
        "Orange": {
            "properties": { "orange": {"description": "", "additionalProperties": {"$ref": "#/definitions/Bannanas"}}}
        }, 
        "Bannanas": {
            "properties": { "bannana": {"description": "", "items": {"$ref": "#/definitions/Apple"}}}
        }, 
    }
    props_per_defn = PARSER.get_definition_properties(defn_circular)

    expected_keys = {"Apple", "Orange", "Bannanas"}
    assert(set(props_per_defn.keys()) == expected_keys)
    
    # All defns in circular loop have all properties
    props = {"apple", "orange", "bannana"}
    assert(props_per_defn["Apple"] == props)
    assert(props_per_defn["Orange"] == props)
    assert(props_per_defn["Bannanas"] == props)

def test_add_ref_tokens_single():
    '''
    Given single ref token
    - add camelcased token of ref name and property of the ref as context
    '''
    props_per_defn = {
        "appleBannana": ["p1", "p2"]
    }
    prefix = "pre"
    expected_tree_context = ["pre_apple", "pre_bannana", "pre_p1", "pre_p2"]
 
    # ref in level 0
    tree_context = []
    ref_item = {"$ref": "#/definitions/appleBannana"}
    PARSER.add_ref_tokens(item=ref_item, props_per_defn=props_per_defn, tree_context=tree_context, prefix=prefix)
    assert(set(expected_tree_context) == set(expected_tree_context))

    # ref in level 1
    tree_context = []
    ref_item = {"description": "OK", "schema": {"$ref": "#/definitions/appleBannana"}}
    PARSER.add_ref_tokens(item=ref_item, props_per_defn=props_per_defn, tree_context=tree_context, prefix=prefix)
    assert(set(expected_tree_context) == set(expected_tree_context))

    # ref in 'additionalProperties', 'items' or 'schema' of level 2
    tree_context = []
    ref_item = {"description": "OK", "schema": { "additionalProperties": { "$ref": "#/definitions/appleBannana"}}}
    PARSER.add_ref_tokens(item=ref_item, props_per_defn=props_per_defn, tree_context=tree_context, prefix=prefix)
    assert(set(expected_tree_context) == set(expected_tree_context))

def test_add_ref_tokens_multiple():
    '''
    Given multiple ref tokens
    - add camelcased token of ref name and property of the ref as tree and leaf context
    '''
    props_per_defn = {
        "appleBannana": ["p1", "p2"], 
        "orange": ["o1", "o2"]
    }
    prefix = "pre"
    expected_tree_context = ["pre_apple", "pre_bannana", "pre_p1", "pre_p2", "pre_orange", "pre_o1", "pre_o2"]

    tree_context = []
    ref_item = {
        "description": "OK", 
        "schema": { 
            "additionalProperties": { "$ref": "#/definitions/appleBannana"}, 
            "items": { "$ref": "#/definitions/orange"}
        }
    }
    PARSER.add_ref_tokens(item=ref_item, props_per_defn=props_per_defn, tree_context=tree_context, prefix=prefix)
    assert(set(expected_tree_context) == set(expected_tree_context))

def test_add_sentence():
    '''
    Given sentence
    - add full sentence to full_text_context (removing special characters)
    - add keywords of sentence to text_context
    '''
    sent = "I went to the mall.\n"
    text_context = []
    full_text_context = []
    
    expected_text_context = ['go mall']
    expected_full_text_context = ["I went to the mall."]
    PARSER.add_sentence(sent=sent, text_context=text_context, full_text_context=full_text_context)

    assert(set(expected_text_context) == set(text_context))
    assert(set(expected_full_text_context) == set(full_text_context))

if __name__ == "__main__":
    test_add_sentence() 

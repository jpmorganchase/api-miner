import sys
sys.path.append('..')

from api_miner.data_processing.validator import APIValidatorSwagger
VALIDATOR = APIValidatorSwagger()

def test_contains_required_keys():
    assert(VALIDATOR.contains_required_keys(required_keys= {'a', 'b'}, keys={'a', 'b'}))
    assert(VALIDATOR.contains_required_keys(required_keys= {'a', 'b'}, keys={'a', 'b', 'c'}))
    assert(not VALIDATOR.contains_required_keys(required_keys= {'a', 'b'}, keys={'a', 'c'}))
    assert(not VALIDATOR.contains_required_keys(required_keys= {'a', 'b'}, keys={}))

def test_add_data_type_match_grade():
    expected_content = {'a': str, 'b': dict, 'c': int, 'd': bool}

    # All keys present, all correct data types = 1
    grade = []
    obj = {'a': 'yay', 'b': {}, 'c': 2, 'd': True}

    VALIDATOR.add_data_type_match_grade(expected_content, obj, grade)
    assert(grade[0] == float(1))

    # Some keys present, all corect data types = 1
    grade = []
    obj = {'a': 'yay', 'b': {}}
    VALIDATOR.add_data_type_match_grade(expected_content, obj, grade)
    assert(grade[0] == float(1))

    # All keys present, half correct data types = 0.5
    grade = []
    obj = {'a': 'yay', 'b': {}, 'c': 'bees', 'd': 3}
    VALIDATOR.add_data_type_match_grade(expected_content, obj, grade)
    assert(grade[0] == float(0.5))

    # No keys present = 0
    grade = []
    obj = {}
    VALIDATOR.add_data_type_match_grade(expected_content, obj, grade)
    assert(grade[0] == float(0))

def test_add_percent_keys_present_grade():
    all_keys = set(['a', 'b', 'c', 'd'])

    grade = []
    keys = set(['a', 'b', 'c', 'd'])
    VALIDATOR.add_percent_keys_present_grade(all_keys, keys, grade)
    assert(grade[0] == float(1))

    grade = []
    keys = set(['a', 'b'])
    VALIDATOR.add_percent_keys_present_grade(all_keys, keys, grade)
    assert(grade[0] == float(0.5))

    grade = []
    keys = set(['cheese'])
    VALIDATOR.add_percent_keys_present_grade(all_keys, keys, grade)
    assert(grade[0] == float(0))

    grade = []
    keys = set()
    VALIDATOR.add_percent_keys_present_grade(all_keys, keys, grade)
    assert(grade[0] == float(0))


if __name__ == "__main__":
    test_add_percent_keys_present_grade()
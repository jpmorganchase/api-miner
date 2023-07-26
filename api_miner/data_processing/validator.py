import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from api_miner.configs.openapi_configs import APISwaggerConfigs
from api_miner.data_processing.utils import get_spec_at_path
from api_miner.configs.internal_configs import DATA_PATH, GRADE_FILE_PATH, GRADES_INFO_PATH, GRADED_SPEC_FILE_PATH, \
    GRADED_SPEC_GRADE

class APIValidator(ABC):
    '''
    Validates API spec based on best practises of a specific version 
    '''
    def __init__(self, configs: dict, re_grade: bool, grade_keys_present = False):
        self._configs = configs
        self._grade_keys_present = grade_keys_present
        self._data_path = DATA_PATH
        self._grade_file_path = GRADE_FILE_PATH

        if re_grade or not os.path.isfile(self._grade_file_path):
            print('Grading dataset...')
            self.grade_dataset()
        
        self.grades = self._setup_grades()
    
    def grade_dataset(self):
        for root, dirs, files in tqdm(os.walk(self._data_path)):
            for file in files:
                full_file_path = os.path.normpath(os.path.join(root,file))
                spec = get_spec_at_path(file_path=full_file_path)
                self._save_spec_grade(spec=spec, spec_file_path=full_file_path)
        
        self._summarize_grades()

    def _save_spec_grade(self, spec:dict, spec_file_path:str, cols = [GRADED_SPEC_FILE_PATH, GRADED_SPEC_GRADE]):
        grade = self._validate(spec = spec)
        if os.path.isfile(self._grade_file_path):
            grades_df = pd.read_csv(self._grade_file_path)
        else:
            grades_df = pd.DataFrame(columns = cols)
        vals = [spec_file_path, grade]
        grades_df = grades_df.append(dict(zip(cols, vals)), ignore_index = True)
        grades_df.to_csv(self._grade_file_path, index = False)

    @abstractmethod
    def _validate(spec: dict) -> float:
        pass
    
    @staticmethod
    def contains_required_keys(required_keys: set, keys: set):
        return required_keys.issubset(keys)

    @staticmethod
    def add_percent_keys_present_grade(all_keys: set, keys: set, grades: list):
        grades.append(len(all_keys & keys) / len(all_keys))
    
    def add_data_type_match_grade(self, expected_content: dict, obj: dict, grades: list):
        '''
        For every expected key that exists in obj, check the match of datatype
        '''
        data_type_match = []
        for key, data_type in expected_content.items():
            if key in obj:
                if type(obj.get(key)) == data_type:
                    data_type_match.append(1)
                else:
                    data_type_match.append(0)
        grades.append(self._get_average(data_type_match))   

    @staticmethod
    def _get_average(grades: list):
        if len(grades) > 0:
            return sum(grades)/ len(grades)
        return 0

    @staticmethod
    def _get_weighted_average_grade(grades: list, weights: list):
        if len(grades) > 0:
            weighted_grades = np.array(grades) * np.array(weights)
            return sum(weighted_grades)
        return 0

    def _summarize_grades(self):
        grades_df = pd.read_csv(self._grade_file_path)
        cols = ['Num specs graded', 'Min grade', 'Max grade', 'Avg grade', 'Std grade']
        vals =  [len(grades_df), min(grades_df['grade']), max(grades_df['grade']),np.mean(grades_df['grade']),np.std(grades_df['grade'])]

        grades_info = pd.DataFrame(columns = cols)
        grades_info = grades_info.append(dict(zip(cols, vals)), ignore_index=True)
        grades_info.to_csv(GRADES_INFO_PATH, index = False)

    def _setup_grades(self):
        grades_df = pd.read_csv(self._grade_file_path)
        grades = dict(zip(grades_df['file_path'], grades_df['grade']))
        return grades

    def retrieve_grade(self, file_path: str):
        '''
        Return grade from 0 to 1, 1 being the best
        '''
        return self.grades[file_path]
    
class APIValidatorSwagger(APIValidator):
    '''
    API validator based on Swagger 2.0 best practises
    - https://swagger.io/specification/v2/

    '''
    def __init__(self, graded=True, re_grade=False):
        super().__init__(configs=APISwaggerConfigs(), re_grade=re_grade)
    
    def _validate(self, spec: dict, info_path_weights=[0.3, 0.7]) -> float:
        '''
        Calculate quality score of spec from 0 to 1, 1 being the best
        '''
        if not self.contains_required_keys(self._configs.api_required_keys, spec.keys() ):
            return 0
        
        # Grade quality of required components 
            # NOTE: quality of subcomponents are not calculated, except operations in path
        info_grade = self._grade_info(spec)
        path_grade = self._grade_paths(spec)
        grades = [info_grade, path_grade]
        return self._get_weighted_average_grade(grades, info_path_weights)

    def _grade_info(self, spec: dict):
        '''
        Grade info section of the API specification
        - grade is 0 if it does not contain all the required keys
        - grade is data_type_match_grade otherwise
        '''
        grades = []
        info_obj = spec.get(self._configs.info_key, {})
        info_keys = info_obj.keys()
        
        if not self.contains_required_keys(self._configs.info_required_keys, info_keys):
            return 0

        if self._grade_keys_present:
            self.add_percent_keys_present_grade(all_keys=self._configs.info_expected_content.keys(), keys = info_keys, grades=grades)
        self.add_data_type_match_grade(expected_content=self._configs.info_expected_content, obj=info_obj, grades=grades)

        return self._get_average(grades)
    
    def _grade_paths(self, spec: dict):
        '''
        Grade of paths is the average grade every path in paths
        Grade of a path is the average grade of every operation in path
        '''
        grades = []
        paths_obj = spec.get(self._configs.paths_key, {})
        
        for path_key, path_obj in paths_obj.items():
            path_grades = []
            for key, obj in path_obj.items():
                if key in self._configs.operation_types:
                    self._add_operation_grade(grades=path_grades, operation_obj=obj)
            grades.append(self._get_average(path_grades))

        return self._get_average(grades)
    
    def _add_operation_grade(self, grades: list, operation_obj: dict):
        '''
        Grade operation of a specification
        - grade is 0 if it does not contain all the required keys
        - grade is data_type_match_grade otherwise
        '''
        operation_grades = []

        if not self.contains_required_keys(self._configs.operation_required_keys, operation_obj.keys()):
            grades.append(0)
            return
        
        responses = operation_obj.get(self._configs.responses_key, {})
        for response_key, response_obj in responses.items():
            if not self.contains_required_keys(self._configs.responses_required_keys, response_obj.keys()):
                grades.append(0)
                return

        if self._grade_keys_present:
            self.add_percent_keys_present_grade(all_keys= self._configs.operation_expected_content.keys(), keys=operation_obj.keys(), grades=operation_grades)
        self.add_data_type_match_grade(expected_content=self._configs.operation_expected_content, obj=operation_obj, grades=operation_grades)

        grades.append(self._get_average(operation_grades))



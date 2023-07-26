from types import GeneratorType
import os

from api_miner.configs.internal_configs import DATA_PATH, KEY_FOR_QUALITY, KEY_FOR_SPEC_NAME
from api_miner.data_processing.utils import get_spec_at_path

class APILoader():
    def __init__(self, validator, call_validator = True, filter_by_grade=False):
        self._validator = validator
        self._call_validator = call_validator
        self._filter_by_grade = filter_by_grade
        self._data_path = DATA_PATH

    def load_specs(self) -> GeneratorType:
        '''
        Load API specs from data_path and return as a generator
        '''
        for root, dirs, files in os.walk(self._data_path):
            for file in files:
                file_path = os.path.normpath(os.path.join(root, file))
                if not self._filter_by_grade:
                    yield self.load_spec(file_path)
                    continue 
                
                if self._is_filtered_by_grade(file_path):
                    continue

                yield self.load_spec(file_path)

    def _is_filtered_by_grade(self, file_path: str) -> bool:
        is_filtered = False

        grade = self._validator.retrieve_grade(file_path= file_path)
        if grade > MIN_GRADE:
            logger.debug(f'API Validator Grade {grade} is lower than {MIN_GRADE}')
            is_filtered = True
        
        return is_filtered

    def load_spec(self, file_path):
        """
        Load single API spec given file path, in JSON format
        """
        try:
            spec = get_spec_at_path(file_path)
            # get quality value and file name on each dimension 
            self._append_metadata(spec, file_path)

        except Exception as e:
            print('load spec error')

        return spec

    def _append_metadata(self, spec, file_path: str):
        '''
        Append file_path, API validator score on each spec
        '''
        if not spec:
            return
        
        if isinstance(spec, list):
            for d in spec:
                self._append_metadata(d, file_path)
        else:
            if self._call_validator:
                spec[KEY_FOR_QUALITY] = self._validator.retrieve_grade(file_path= file_path)
            spec[KEY_FOR_SPEC_NAME] = file_path

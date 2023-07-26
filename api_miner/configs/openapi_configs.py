class APIConfigs():
    def __init__(self, configs: dict):
        self._configs = self._valid_configs(configs)
        self.api_required_keys = self._configs['api_required_keys']
        self.info_required_keys = self._configs['info_required_keys']
        self.info_expected_content = self._configs['info_expected_content']
        self.operation_required_keys = self._configs['operation_required_keys']
        self.operation_types = self._configs['operation_types']
        self.operation_expected_content = self._configs['operation_expected_content']
        self.responses_required_keys = self._configs['responses_required_keys']
        self.paths_key = self._configs['paths_key']
        self.defn_key = self._configs['defn_key']
        self.ref_key = self._configs['ref_key']
        self.info_key = self._configs['info_key']
        self.responses_key = self._configs['responses_key']

    def _valid_configs(self, configs: dict):
        required_configs = {'api_required_keys', 'info_required_keys', 'info_expected_content', 'operation_required_keys', 'operation_types', 'operation_expected_content', 'responses_required_keys', 'paths_key', 'defn_key', 'ref_key', 'info_key', 'responses_key'}
        if not required_configs.issubset(configs.keys()):
            raise ValueError('API config does not contain all required keys: ', required_configs)
        return configs

class APISwaggerConfigs(APIConfigs):
    def __init__(self):
        self._api_type_key = 'swagger'
        self._defn_key = "definitions"
        self._paths_key = "paths"
        self._ref_key = "$ref"
        self.properties_key = "properties"
        self._info_key = 'info'
        self._title_key = 'title'
        self._version_key = 'version'
        self.desc_key = 'description'
        self._terms_of_service_key = 'termsOfService'  # pragma: allowlist secret
        self._contact_key = 'contact'
        self._licence_key = 'license'
        self._responses_key = 'responses'
        self._tags_key = 'tags'
        self.summary_key = 'summary'
        self._external_docs_key = 'externalDocs'
        self._operation_id_key = 'operationId'
        self._consumes_key = 'consumes'
        self._produces_key = 'produces'
        self.parameters_key = 'parameters'
        self._schemes_key = 'schemes'
        self._deprecated_key = 'deprecated'
        self._security_key = 'security'
        self.schema_key = 'schema'
        self.name_key = 'name'
        self.items_key = 'items'
        self.additionalProperties_key = 'additionalProperties'

        configs = {
            'api_required_keys': {self._info_key, self._paths_key, self._api_type_key}, 
            'info_required_keys': {self._title_key, self._version_key}, 
            'info_expected_content':  {
                self._title_key: str, 
                self.desc_key: str, 
                self._terms_of_service_key: str, 
                self._contact_key: dict, 
                self._licence_key: dict, 
                self._version_key: str
            }, 
            'operation_required_keys': {self._responses_key}, 
            'operation_types':  {'get', 'put', 'post', 'delete', 'options', 'head', 'patch'}, 
            'operation_expected_content': {
                self._tags_key: list, 
                self.summary_key:str, 
                self.desc_key:str, 
                self._external_docs_key:dict, 
                self._operation_id_key: str, 
                self._consumes_key:list, 
                self._produces_key:list, 
                self.parameters_key:dict, 
                self._responses_key:dict, 
                self._schemes_key:list, 
                self._deprecated_key:bool, 
                self._security_key:dict
                }, 
            'responses_required_keys': {self.desc_key}, 
            'paths_key': self._paths_key, 
            'defn_key': self._defn_key, 
            'ref_key': self._ref_key, 
            'info_key': self._info_key, 
            'responses_key': self._responses_key
        }
        super().__init__(configs = configs)

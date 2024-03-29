from api_miner.data_processing.parser import APIParserSwagger
from api_miner.data_processing.data_loader import APILoader
from api_miner.data_processing.vectorize import APIVectorizer
from api_miner.data_processing.validator import APIValidatorSwagger

VECTORIZER = APIVectorizer()
PARSER = APIParserSwagger()
OPENAPI_CONFIGS = PARSER.openapi_configs

SAMPLE_SPEC = {
    "swagger": "2.0", 
    "schemes": ["https"], 
    "host": "api.apis.guru", 
    "basePath": "/v2/", 
    "info": {
        "contact": {"email": "mike.ralphson@gmail.com", "name": "APIs.guru", "url": "https://APIs.guru"}, 
        "description": "Wikipedia for Web APIs. Repository of API specs in OpenAPI 3.0 format.\n\n**Warning**: If you want to be notified about changes in advance please join our [Slack channel](https://join.slack.com/t/mermade/shared_invite/zt-g78g7xir-MLE_CTCcXCdfJfG3CJe9qA).\n\nClient sample: [[Demo]](https://apis.guru/simple-ui) [[Repo]](https://github.com/APIs-guru/simple-ui)\n", "license": {"name": "CC0 1.0", "url": "https://github.com/APIs-guru/openapi-directory#licenses"}, 
        "title": "APIs.guru", 
        "version": "2.0.2", 
        "x-apisguru-categories": ["open_data", "developer_tools"], 
        "x-logo": {"url": "https://apis.guru/branding/logo_vertical.svg"}, 
        "x-origin": [{"format": "swagger", "url": "https://api.apis.guru/v2/swagger.json", "version": "2.0"}], 
        "x-providerName": "apis.guru", 
        "x-tags": ["API", "Catalog", "Directory", "REST", "Swagger", "OpenAPI"]
    }, 
    "externalDocs": {"url": "https://github.com/APIs-guru/openapi-directory/blob/master/API.md"}, 
    "produces": ["application/json; charset=utf-8", "application/json"], 
    "security": [], 
    "tags": [{"description": "Actions relating to APIs in the collection", "name": "APIs"}], 
    "paths": {
        "/list.json": {
            "get": {
                "description": "List all APIs in the directory.\nReturns links to OpenAPI specification for each API in the directory.\nIf API exist in multiple versions `preferred` one is explicitly marked.\n\nSome basic info from OpenAPI spec is cached inside each object.\nThis allows to generate some simple views without need to fetch OpenAPI spec for each API.\n", 
                "operationId": "listAPIs", 
                "responses": {"200": {"description": "OK", "schema": {"$ref": "#/definitions/APIs"}}}, 
                "summary": "List all APIs", 
                "tags": ["APIs"]
            }
        }, 
        "/metrics.json": {
            "get": {
                "description": "Some basic metrics for the entire directory.\nJust stunning numbers to put on a front page and are intended purely for WoW effect :)\n", 
                "operationId": "getMetrics", 
                "responses": {"200": {"description": "OK", "schema": {"$ref": "#/definitions/Metrics"}}}, 
                "summary": "Get basic metrics", "tags": ["APIs"]
            }
        }
    }, 
    "definitions": {
        "API": {
            "additionalProperties": "false", 
            "description": "Meta information about API", 
            "properties": {
                "added": {"description": "Timestamp when the API was first added to the directory", "format": "date-time", "type": "string"}, 
                "preferred": {"description": "Recommended version", "type": "string"}, 
                "versions": {"additionalProperties": {"$ref": "#/definitions/ApiVersion"}, "description": "List of supported versions of the API", "minProperties": "1", "type": "object"}
            }, 
            "required": ["added", "preferred", "versions"], 
            "type": "object"
        }, 
        "APIs": {
            "additionalProperties": {"$ref": "#/definitions/API"}, 
            "description": "List of API details.\nIt is a JSON object with API IDs(`<provider>[:<service>]`) as keys.\n",
            "example": {"googleapis.com:drive": {"added": "2015-02-22T20:00:45.000Z", "preferred": "v3", "versions": {"v2": {"added": "2015-02-22T20:00:45.000Z", "info": {"title": "Drive", "version": "v2", "x-apiClientRegistration": {"url": "https://console.developers.google.com"}, "x-logo": {"url": "https://api.apis.guru/v2/cache/logo/https_www.gstatic.com_images_icons_material_product_2x_drive_32dp.png"}, "x-origin": {"format": "google", "url": "https://www.googleapis.com/discovery/v1/apis/drive/v2/rest", "version": "v1"}, "x-preferred": "false", "x-providerName": "googleapis.com", "x-serviceName": "drive"}, "swaggerUrl": "https://api.apis.guru/v2/specs/googleapis.com/drive/v2/swagger.json", "swaggerYamlUrl": "https://api.apis.guru/v2/specs/googleapis.com/drive/v2/swagger.yaml", "updated": "2016-06-17T00:21:44.000Z"}, "v3": {"added": "2015-12-12T00:25:13.000Z", "info": {"title": "Drive", "version": "v3", "x-apiClientRegistration": {"url": "https://console.developers.google.com"}, "x-logo": {"url": "https://api.apis.guru/v2/cache/logo/https_www.gstatic.com_images_icons_material_product_2x_drive_32dp.png"}, "x-origin": {"format": "google", "url": "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest", "version": "v1"}, "x-preferred": "true", "x-providerName": "googleapis.com", "x-serviceName": "drive"}, "swaggerUrl": "https://api.apis.guru/v2/specs/googleapis.com/drive/v3/swagger.json", "swaggerYamlUrl": "https://api.apis.guru/v2/specs/googleapis.com/drive/v3/swagger.yaml", "updated": "2016-06-17T00:21:44.000Z"}}}}, 
            "minProperties": "1", "type": "object"
        }, 
        "ApiVersion": {
            "additionalProperties": "false", 
            "properties": {
                "added": {
                    "description": "Timestamp when the version was added", 
                    "format": "date-time", 
                    "type": "string"
                }, 
                "externalDocs": {"description": "Copy of `externalDocs` section from OpenAPI definition", "minProperties": "1", "type": "object"}, 
                "info": {"description": "Copy of `info` section from OpenAPI definition", "minProperties": "1", "type": "object"}, 
                "swaggerUrl": {"description": "URL to OpenAPI definition in JSON format", "format": "url", "type": "string"}, 
                "swaggerYamlUrl": {"description": "URL to OpenAPI definition in YAML format", "format": "url", "type": "string"}, 
                "updated": {"description": "Timestamp when the version was updated", "format": "date-time", "type": "string"}
            },
            "required": ["added", "updated", "swaggerUrl", "swaggerYamlUrl", "info"], 
            "type": "object"
        }, 
        "Metrics": {
            "additionalProperties": "false", 
            "description": "List of basic metrics", 
            "example": {"numAPIs": "238", "numEndpoints": "6448", "numSpecs": "302"}, 
            "properties": {
                "numAPIs": {"description": "Number of APIs", "minimum": "1", "type": "integer"}, 
                "numEndpoints": {"description": "Total number of endpoints inside all specifications", "minimum": "1", "type": "integer"}, 
                "numSpecs": {"description": "Number of API specifications including different versions of the same API", "minimum": "1", "type": "integer"}
            }, 
            "required": ["numSpecs", "numAPIs", "numEndpoints"], 
            "type": "object"}
        }, 
        "from_spec": "sample", 
        "quality_score": 1
    }



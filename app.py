import os
import json
import pickle
import logging

import numpy as np
from flask import Flask, Response, request
from flasgger import swag_from, Swagger

from experiments.setup import initialize_demo
from experiments.utils import transform_endpoint_name_to_features, transform_specs_to_features, retrieve_endpoints_from_database, normalize_scores
from experiments.sample_generator import MaskedSpecGenerator, MangledSpecGenerator, EndpointNameGenerator, UserStudySpecGenerator

app = Flask(__name__,
            static_url_path='/static')
app.logger.setLevel(logging.DEBUG)
RESULT_LIMIT = 10

@app.route('/')
def index():
    """UI Landing page"""
    return app.send_static_file('index.html')

def search_by_endpoint(query_endpt, n_retrieved=RESULT_LIMIT):
    if not isinstance(query_endpt, str):
        return Response(
            f"Bad Request - Expect string as input.",
            status=400
        )
    # TODO: Replace with actual endpoint retrieval inference
    query_feature = transform_endpoint_name_to_features(endpoint_name=endpoint, vectorizer=VECTORIZER)
    if not query_feature:
        return Response(
            f"Unseen Endpoint - Please supply the OpenAPI Specification.",
            status=404
        )
    endpoint_results_str = _get_endpoint_results(query_feature, n_retrieved)
        
    return Response(endpoint_results_str)

def search_by_spec(query_spec_str, n_retrieved=RESULT_LIMIT):
    query_spec = json.loads(query_spec_str)
    if not isinstance(query_spec, dict) or not query_spec:
        return Response(
            f"Bad Request - Expect one JSON String specification as input.",
            status=400
        )
    query_feature = transform_specs_to_features(specs = [query_spec], parser = PARSER, vectorizer = VECTORIZER)

    if not query_feature:
        return Response(
            f"Malformed Input - Supply Valid OpenAPI Spec",
            status=422
        )
    endpoint_results_str = _get_endpoint_results(query_feature, n_retrieved)

    return Response(endpoint_results_str)

def _build_cors_prelight_response():
    response = make_response()
    # response.headers.add("Access-Control-Allow-Origin", "*")
    # response.headers.add("Access-Control-Allow-Headers", "*")
    # response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _get_endpoint_results(query_feature, n_retrieved):
    retrieved_endpoint_names_per_endpoint, top_similarity_scores_per_endpoint = retrieve_endpoints_from_database(
            model= MODEL, 
            vectorizer= VECTORIZER, 
            query_feature = query_feature, 
            config= DEMO_CONFIG, 
            n_retrieved=n_retrieved, 
            retrieve_specs = True)

    retrieved_endpoints = retrieved_endpoint_names_per_endpoint[0]
    relevance_scores = normalize_scores(top_similarity_scores_per_endpoint[0])

    endpoint_results = []
    for i in range(len(retrieved_endpoints)):
        results = {}
        results['endpoint'] = retrieved_endpoints[i][0]['endpoint_name']
        results['similarity'] = relevance_scores[i]
        results['spec_snippets'] = retrieved_endpoints[i][0]['content']
        endpoint_results.append(results)

    return json.dumps(endpoint_results)

def _is_json(string):
    try:
        json.loads(string)
        return True
    except ValueError or TypeError:
        return False


@app.route('/search', methods=['POST', 'OPTIONS'])
def search():
    """Search for OpenAPI Specification Snippets by endpoints or specification
    ---
    description: Search for OpenAPI Specification Snippets by endpoints or specification
    definitions:
        EndpointResult:
            type: object
            properties:
                endpoint:
                    type: string
                similarity_score:
                    type: float
                spec_snippets:
                    items:
                    $ref: '#/definitions/SpecSnippet'
        SpecSnippet:
            type: object
            properties:
                endpoint_info:
                    type: object
                ref_def:
                    type: object
    parameters:
        - name: body
          in: body
          required: true
          schema:
            id: openApi
            required:
                - query_code
            properties:
                query_code:
                    type: string
                    description: query code snippet
                    example: '{"paths": {"/heart": {"get": {"summary": "Heartbeating endpoint"}}}}'

    responses:
        "200":
          description: "Return search results."
          schema:
            type: "array"
            items:
              $ref: "#/definitions/EndpointResult"
        "400":
          description: "Bad Request - Expect one JSON  String specification as input."
        "404":
          description: "Unseen Endpoint - Please supply the OpenAPI Specification."
        "422":
          description: "Malformed Input - Supply Valid OpenAPI Spec"
    """

    if request.method == "OPTIONS":
        return _build_cors_prelight_response()

    req_json = request.get_json()
    query_code = req_json.get('query_code')

    if not _is_json(query_code):
        resp = search_by_endpoint(query_code)
    else:
        resp = search_by_spec(query_code)
    return resp

def test():
    query_code = json.dumps({"paths": {"/imgtrend": {"get": {"summary": "Return trending images"}}}})

    if not _is_json(query_code):
        resp = search_by_endpoint(query_code)
    else:
        resp = search_by_spec(query_code)

if __name__ == "__main__":
    global DATA_LOADER, VALIDATOR, PARSER, VECTORIZER, MODEL, DEMO_CONFIG
    DATA_LOADER, VALIDATOR, PARSER, VECTORIZER, MODEL, DEMO_CONFIG = initialize_demo()
    app.run(host='0.0.0.0', port=9000)
    # test()


import argparse
import requests
from fastapi import FastAPI, HTTPException
import uvicorn
from convert import *
import json
from urllib.parse import urljoin
from typing import Union



app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument("--hf_pipeline", type=str, help="Hugging Face pipeline name")
parser.add_argument("--model_deployed_url", type=str, help="URL of the deployed model")
args = parser.parse_args()
    
# Set the pipeline name and model URL from the command-line arguments
pipeline_name = args.hf_pipeline
model_deployed_url = args.model_deployed_url

model_deployed_name = {
    "zero-shot-classification": "zero-shot-cl-test",
    "text-generation": "text-gen-test",
    "object-detection": "test-object-detect",
    "token-classification": "token-cl-test"
}

def forward_request_to_model(model_deployed_url, v2_input):
    model_deployed_url = urljoin(model_deployed_url , '/v2/models/'+ model_deployed_name[pipeline_name] +'/infer')
    print("\nForwarding request to model..." + model_deployed_url)
    v2_input = json.dumps(v2_input)
    try:
        response = requests.post(model_deployed_url, data=v2_input)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/")
def predict(inputs: Union[dict, str]):
    print("\nReceived request for pipeline: "+ pipeline_name)
    print("\nReceived inputs: " + str(inputs))
    v2_input = convert_to_v2_input(pipeline_name, inputs)
    print("\nConverted to V2 input: " + str(v2_input))
    response = forward_request_to_model(model_deployed_url, v2_input)
    return response



import os
import requests
from fastapi import FastAPI,HTTPException
from convert import *
import json
import uvicorn
from urllib.parse import urljoin
app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"))

model_deployed_name = {
    "zero-shot-classification": "zero-shot-cl-test",
    "text-generation": "text-gen-test",
    "object-detection": "test-object-detect",
    "token-classification": "token-cl-test"
}

def forward_request_to_model(inputs, v2_input):
    model_deployed_url = urljoin(inputs["model_deployed_url"] , '/v2/models/'+ model_deployed_name[inputs["hf_pipeline"]] +'/infer')
    print("\nForwarding request to model..." + model_deployed_url)
    v2_input = json.dumps(v2_input)
    try:
        response = requests.post(model_deployed_url, data=v2_input)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(inputs: dict):
    print("\nReceived request for pipeline: "+ inputs['hf_pipeline'])
    print("\nReceived inputs: " + str(inputs))
    v2_input = convert_to_v2_input(inputs)
    print("\nConverted to V2 input: " + str(v2_input))
    response = forward_request_to_model(inputs, v2_input)
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
import argparse
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from convert import convert_to_v2_input

app = FastAPI()


class HFInput(BaseModel):
    inputs: dict


def forward_request_to_model(model_deployed_url, v2_input):
    try:
        response = requests.post(model_deployed_url, json=v2_input)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/")
def predict(hf_pipeline: str, model_deployed_url: str, input_data: HFInput):
    v2_input = convert_to_v2_input(hf_pipeline, input_data.inputs)
    response = forward_request_to_model(model_deployed_url, v2_input)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_pipeline", type=str, help="Hugging Face pipeline name")
    parser.add_argument("--model_deployed_url", type=str, help="URL of the deployed model")
    args = parser.parse_args()

    # Set the pipeline name and model URL from the command-line arguments
    pipeline_name = args.hf_pipeline
    model_deployed_url = args.model_deployed_url

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from mlserver.types import MetadataTensor
from fastapi import FastAPI, HTTPException
import PIL.Image

def convert_to_v2_input(hf_pipeline, input_data):
    if hf_pipeline == "zero-shot-classification":
        v2_input = {
            "inputs": [
                MetadataTensor(
                    name="array_inputs",
                    shape=[-1],
                    datatype="BYTES",
                    parameters={"content_type": "str"},
                    data=[input_data["inputs"]],
                ),
                MetadataTensor(
                    name="candidate_labels",
                    shape=[-1],
                    datatype="BYTES",
                    parameters={"content_type": "str"},
                    data=input_data["parameters"]["candidate_labels"],
                ),
            ],
            "outputs": [],
        }
    elif hf_pipeline == "object-detection":
        if isinstance(input_data[0], PIL.Image.Image):
            v2_input = {
                "inputs": [
                    MetadataTensor(
                    name="inputs",
                    shape=[-1],
                    datatype="BYTES",
                    parameters=dict(content_type="pillow_image"),
                    data= input_data
                )
                ],
                "outputs": [],
            }
    elif hf_pipeline == "text-generation":
        v2_input = {
            "inputs": [
                MetadataTensor(
                    name="args",
                    shape=[-1],
                    datatype="BYTES",
                    parameters={"content_type": "str"},
                    data=[input_data["inputs"]],
                )
            ],
            "outputs": [],
        }
    elif hf_pipeline == "token-classification":
        v2_input = {
            "inputs": [
                MetadataTensor(
                    name="args",
                    shape=[-1],
                    datatype="BYTES",
                    parameters={"content_type": "str"},
                    data=[input_data["inputs"]],
                )
            ],
            "outputs": [],
        }
    else:
        # Handle unsupported pipeline
        raise HTTPException(status_code=400, detail="Unsupported pipeline")

    return v2_input

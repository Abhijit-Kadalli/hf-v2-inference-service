from fastapi import HTTPException
import PIL.Image
import json
from pydantic import BaseModel as _BaseModel
from typing import Any, Dict, List, Optional

#WILL ONLY WORK WITH LINUX
#from mlserver.types.dataplane import MetadataTensor

from pydantic import Extra, Field

class BaseModel(_BaseModel):
    """
    Override Pydantic's BaseModel class to ensure all payloads exclude unset
    fields by default.

    From:
        https://github.com/pydantic/pydantic/issues/1387#issuecomment-612901525
    """

    def dict(self, exclude_unset=True, exclude_none=True, **kwargs):
        return super().dict(
            exclude_unset=exclude_unset, exclude_none=exclude_none, **kwargs
        )

    def json(self, exclude_unset=True, exclude_none=True, **kwargs):
        return super().json(
            exclude_unset=exclude_unset, exclude_none=exclude_none, **kwargs
        )

class Parameters(BaseModel):
    class Config:
        extra = Extra.allow

    content_type: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None

class MetadataTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    parameters: Optional["Parameters"] = None

class MetadataTensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MetadataTensor):
            return obj.dict()
        return super().default(obj)

def convert_to_v2_input(hf_pipeline, input_data):
    if hf_pipeline == "zero-shot-classification":
        #input_data = json.loads(input_data)
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
        print("Converted to V2 input: " + str(v2_input))
        return json.dumps(v2_input, cls=MetadataTensorEncoder)

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
        print("Converted to V2 input: " + str(v2_input))
        return json.dumps(v2_input, cls=MetadataTensorEncoder)
    elif hf_pipeline == "text-generation":
        input_data = json.loads(input_data)
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
        print("Converted to V2 input: " + str(v2_input))
        return json.dumps(v2_input, cls=MetadataTensorEncoder)
    elif hf_pipeline == "token-classification":
        input_data = json.loads(input_data)
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
        print("Converted to V2 input: " + str(v2_input))
        return json.dumps(v2_input, cls=MetadataTensorEncoder)
    else:
        # Handle unsupported pipeline
        raise HTTPException(status_code=400, detail="Unsupported pipeline")

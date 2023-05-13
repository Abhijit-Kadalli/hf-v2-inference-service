from fastapi import HTTPException
from PIL import Image
import json
# from pydantic import BaseModel as _BaseModel
# from typing import Any, Dict, List, Optional

# #WILL ONLY WORK WITH LINUX
# #from mlserver.types.dataplane import MetadataTensor

# from pydantic import Extra, Field

# class BaseModel(_BaseModel):
#     """
#     Override Pydantic's BaseModel class to ensure all payloads exclude unset
#     fields by default.

#     From:
#         https://github.com/pydantic/pydantic/issues/1387#issuecomment-612901525
#     """

#     def dict(self, exclude_unset=True, exclude_none=True, **kwargs):
#         return super().dict(
#             exclude_unset=exclude_unset, exclude_none=exclude_none, **kwargs
#         )

#     def json(self, exclude_unset=True, exclude_none=True, **kwargs):
#         return super().json(
#             exclude_unset=exclude_unset, exclude_none=exclude_none, **kwargs
#         )

# class Parameters(BaseModel):
#     class Config:
#         extra = Extra.allow

#     content_type: Optional[str] = None
#     headers: Optional[Dict[str, Any]] = None

# class MetadataTensor(BaseModel):
#     name: str
#     datatype: str
#     shape: List[int]
#     parameters: Optional["Parameters"] = None

def convert_to_v2_input(hf_pipeline, input_data):
    print("\nReceived request for pipeline: " + hf_pipeline + " with input: " + str(input_data))
    if hf_pipeline == "zero-shot-classification":
        v2_input =   {
            "parameters": {
            "content_type": "application/json",
            "headers": {}
            },
            "inputs": [
            {
                "name": "array_inputs",
                "shape": [
                -1
                ],
                "datatype": "BYTES",
                "parameters": {
                "content_type": "application/json",
                "headers": {}
                },
                "data": input_data["inputs"]
            },
            {
                "name": "candidate_labels",
                "shape": [
                -1
                ],
                "datatype": "BYTES",
                "parameters": {
                "content_type": "application/json",
                "headers": {}
                },
                "data": json.dumps(input_data['parameters']['candidate_labels'])
            }
            ],
            "outputs": [
            {
                "name": "output_*",
                "parameters": {
                "content_type": "application/json",
                "headers": {}
                }
            }
            ]
        }
        return v2_input

    elif hf_pipeline == "object-detection":
        v2_input = {
        "id": "string",
        "parameters": {
            "content_type": "string",
            "headers": {}
        },
        "inputs": [
            {
            "name": "inputs",
            "shape": [
                0
            ],
            "datatype": "BOOL",
            "parameters": {
                "content_type": "bytes",
                "headers": {}
            },
            "data": input_data  
            }
        ],
        "outputs": [
            {
            "name": "string",
            "parameters": {
                "content_type": "string",
                "headers": {}
            }
            }
        ]
        }
        return json.dumps(v2_input)
    
    elif hf_pipeline == "text-generation":
        v2_input = {
            "inputs": [
                {
                "name": "text_inputs",
                "shape": [
                    0
                ],
                "datatype": "BOOL",
                "parameters": {
                    "content_type": "string",
                    "headers": {}
                },
                "data": input_data['inputs']
                }
            ],
            "outputs": [
                {
                "name": "string",
                "parameters": {
                    "content_type": "string",
                    "headers": {}
                }
                }
            ]
            }
        return v2_input
    
    elif hf_pipeline == "token-classification":
        v2_input = {
        "id": "string",
        "parameters": {
            "content_type": "string",
            "headers": {}
        },
        "inputs": [
            {
            "name": "inputs",
            "shape": [
                0
            ],
            "datatype": "BOOL",
            "parameters": {
                "content_type": "string",
                "headers": {}
            },
            "data": input_data['inputs']
            }
        ],
        "outputs": [
            {
            "name": "string",
            "parameters": {
                "content_type": "string",
                "headers": {}
            }
            }
        ]
        }
        return v2_input
    else:
        # Handle unsupported pipeline
        raise HTTPException(status_code=400, detail="Unsupported pipeline")

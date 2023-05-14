from fastapi import HTTPException
import json
import base64

def isBase64(sb):
        try:
                if isinstance(sb, str):
                        # If there's any unicode here, an exception will be thrown and the function will return false
                        sb_bytes = bytes(sb, 'utf-8')
                elif isinstance(sb, bytes):
                        sb_bytes = sb
                else:
                        raise ValueError("Argument must be string or bytes")
                return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
        except Exception:
                return False
        
def convert_to_v2_input(input_data):
    if input_data['hf_pipeline'] == "zero-shot-classification":
        v2_input = {
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
            ]
        }
        return v2_input
    
    elif input_data['hf_pipeline'] == "text-generation" or "token-classification":
        v2_input = {
            "parameters": {
                "content_type": "application/json",
                "headers": {}
            },
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
            ]
        }
        return v2_input
    elif input_data['hf_pipeline'] == "object-detection":
        if isinstance(input_data['inputs'], str) and input_data.startswith("http"):
            v2_input = {
                "parameters": {
                    "content_type": "string",
                    "headers": {}
                },
                "inputs": [
                    {
                    "name": "inputs",
                    "shape": [
                        -1
                    ],
                    "datatype": "BYTES",
                    "parameters": {
                        "content_type": "str"
                    },
                    "data": input_data
                    }
                ],
                "outputs": []
            }
        elif isBase64(input_data):
            v2_input = {
                "parameters": {
                    "content_type": "string",
                    "headers": {}
                },
                "inputs": [
                    {
                    "name": "inputs",
                    "shape": [
                        -1,
                        -1,
                        -1
                    ],
                    "datatype": "BYTES",
                    "parameters": {
                        "content_type": "pillow_image"
                    },
                    "data": input_data["inputs"]
                    }
                ],
                "outputs": []
            }
        return v2_input    
    else:
        raise HTTPException(status_code=400, detail="hf_pipeline not supported")
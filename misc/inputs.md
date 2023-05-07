# HF input formats
## BaptisteDoyen/camembert-base-xnli [zero-shot-classification] 
```python
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
payload = {
    "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
    "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
}
output = query(payload)
```


## TahaDouaji/detr-doc-table-detection [object-detection]
```python
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("cats.jpg")
```

## sshleifer/tiny-gpt2 [text-generation]
```python
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

payload= {
	"inputs": "Can you please let us know more details about your ",
}
output = query(payload)
```

## d4data/biomedical-ner-all [token-classification]
```python
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
payload = {
    "inputs": "My name is Sarah Jessica Parker but you can call me Jessica",
}
output = query(payload)
```

# v2 input formats
```json
"zero-shot-classification": dict(
        inputs=[
            MetadataTensor(
                name="array_inputs",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="str"),
            ),
            MetadataTensor(
                name="candidate_labels",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="str"),
            ),
        ],
        outputs=[],
    ),
```

```json
"object-detection": dict(
        inputs=[
            # file path inputs
            MetadataTensor(
                name="inputs",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="str"),
            ),
            # file content bytes inputs
            MetadataTensor(
                name="inputs",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="base64"),
            ),
            # pillow image  inputs
            MetadataTensor(
                name="inputs",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="pillow_image"),
            ),
        ],
        outputs=[],
    ),
```
```json
"text-classification": dict(
        inputs=[
            MetadataTensor(
                name="args",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="str"),
            ),
        ],
        outputs=[
            MetadataTensor(
                name="outputs",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="hg_json"),
            ),
        ],
    ),
```
```json
"token-classification": dict(
        inputs=[
            MetadataTensor(
                name="args",
                shape=[-1],
                datatype="BYTES",
                parameters=dict(content_type="str"),
            ),
        ],
        outputs=[],
    )
```
import requests

API_URL = "https://api-inference.huggingface.co/models/TahaDouaji/detr-doc-table-detection"
headers = {"Authorization": "Bearer {api}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    print(data)
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("image.jpg")
print(output)
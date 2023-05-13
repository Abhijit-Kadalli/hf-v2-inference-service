import base64
def query(filename):
    with open(filename, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    print(encoded_string)

query("image.png")
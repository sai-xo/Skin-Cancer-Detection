import requests

url = "https://example.com/path/to/rainbowflow_model.h5"
response = requests.get(url)

with open("rainbowflow_model.h5", "wb") as f:
    f.write(response.content)

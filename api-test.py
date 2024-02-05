import requests

url = "http://127.0.0.1:8000/predict/"

with open("./images/samples/53-origin.jpg", 'rb') as fobj:
    resp = requests.post(url, files={'file': fobj})

print(resp.json())
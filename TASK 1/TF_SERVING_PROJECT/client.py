import json
import numpy as np
import requests

data = json.dumps({
    "signature_name": "serving_default",
    "instances": [np.zeros((28, 28)).tolist()]
})

headers = {"content-type": "application/json"}
response = requests.post("http://localhost:8501/v1/models/my_model:predict", data=data, headers=headers)

print("Prediction response:")
print(response.json())

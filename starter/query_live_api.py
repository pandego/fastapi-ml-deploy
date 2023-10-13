import requests
import json

url = "https://fastapi-ml-deploy.onrender.com/"
method = "predict"
sample_data = {
  "age": 50,
  "workclass": "Self-emp-not-inc",
  "fnlgt": 83311,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 13,
  "native_country": "United-States"
}

headers = {"Content-type": "application/json"}
response = requests.post(url+method, data=json.dumps(sample_data), headers=headers)
print("Status Code: ", response.status_code)
print("Prediction: ", response.text)
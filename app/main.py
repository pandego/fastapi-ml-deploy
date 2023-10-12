import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

from src.ml.data import process_data
from src.ml.model import inference

try:
    # Load your trained model and encoders here
    model = load("model/model.pkl")
    encoder = load("model/encoder.pkl")
    lb = load("model/label_encoder.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(e)

app = FastAPI()


class PredictionInput(BaseModel):
    age: int = Field(None, example=50)
    workclass: str = Field(None, example="Self-emp-not-inc")
    fnlgt: int = Field(None, example=83311)
    education: str = Field(None, example="Bachelors")
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example="Married-civ-spouse")
    occupation: str = Field(None, example="Exec-managerial")
    relationship: str = Field(None, example="Husband")
    race: str = Field(None, example="White")
    sex: str = Field(None, example="Male")
    capital_gain: int = Field(None, example=0)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=13)
    native_country: str = Field(None, example="United-States")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Machine Learning Model API!"}


@app.post("/predict/")
def predict(data: PredictionInput):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    try:
        # Process the input data using your process_data function and encoders
        data = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}
        data = pd.DataFrame.from_dict(data)

        X, _, _, _ = process_data(data, categorical_features=cat_features, label=None, training=False,
                                  encoder=encoder, lb=lb)
        predictions = inference(model, X)
        prediction = lb.inverse_transform(predictions)[0].split(" ")[-1]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

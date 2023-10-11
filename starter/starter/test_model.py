import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data

def test_train_model():
    # Generate dummy data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = train_model(X, y)

    # Check if the function returns a model
    assert isinstance(model, BaseEstimator), "train_model should return a model"

def test_inference():
    # Generate dummy model and data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = train_model(X, y)
    preds = inference(model, X)

    # Check if predictions have the correct shape and type
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert preds.shape == y.shape, "Predictions should have the same shape as y"

def test_compute_model_metrics():
    # Dummy true labels and predictions
    y_true = np.array([0, 1, 0, 1])
    y_preds = np.array([0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)

    # Check types and value ranges
    for metric in [precision, recall, fbeta]:
        assert isinstance(metric, float), "Metrics should be of type float"
        assert 0 <= metric <= 1, "Metric values should be between 0 and 1"

if __name__ == "__main__":
    test_train_model()
    test_inference()
    test_compute_model_metrics()
    print("All tests passed!")

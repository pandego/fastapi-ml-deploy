import numpy as np
import pandas as pd
from joblib import load

from ml.data import process_data
from ml.model import compute_model_metrics


def compute_metrics_on_slices(model, data, feature_name, encoder, lb):
    """
    Computes model metrics on slices of data based on unique values of a given feature.

    Parameters:
    - model: Trained machine learning model.
    - data: Original dataframe.
    - feature_name: Name of the feature on which to slice.
    - encoder: The trained encoder.
    - lb: The trained label binarizer.

    Returns:
    - A dictionary with unique feature values as keys and their corresponding metrics as values.
    """
    unique_values = data[feature_name].unique()
    metrics_dict = {}

    for value in unique_values:
        # Filter data based on the feature's unique value
        sliced_data = data[data[feature_name] == value]

        # Process the sliced data
        X_slice, y_slice, _, _ = process_data(
            sliced_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )

        # Predict and compute metrics
        preds = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        metrics_dict[value] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        }

    return metrics_dict


if __name__ == "__main__":
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

    # Load in the data.
    data = pd.read_csv('src/data/census.csv')

    # Stripping spaces from column names
    data.columns = data.columns.str.strip()

    # Load the model and encoders
    model = load("src/model/model.pkl")
    encoder = load("src/model/encoder.pkl")
    lb = load("src/model/label_encoder.pkl")

    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=False,
                              encoder=encoder, lb=lb)

    for feature in cat_features:
        metrics_dict = compute_metrics_on_slices(model, data, feature, encoder, lb)

        # Append outputs to slice_output.txt
        with open("src/src/slice_output.txt", "a") as file:
            for value, metrics in metrics_dict.items():
                file.write(f"Metrics for '{feature}' ={value}:\n")
                file.write(f"Precision: {metrics['precision']:.4f}\n")
                file.write(f"Recall: {metrics['recall']:.4f}\n")
                file.write(f"F1 Score: {metrics['fbeta']:.4f}\n\n")
                file.write("------#####------\n\n")

    print("Metrics saved to slice_output.txt!")

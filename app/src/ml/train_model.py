from sklearn.model_selection import train_test_split

# Add the necessary imports for the src code.
import pandas as pd
from data import process_data
from model import train_model, save_model, save_encoder, compute_model_metrics


# Add code to load in the data.
data = pd.read_csv('src/data/census.csv')

# Stripping spaces from column names
data.columns = data.columns.str.strip()  # TODO: opportunity to use DVC

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model
model = train_model(X_train, y_train)

# Save the trained model and encoders
save_model(model, 'src/model/model.pkl')
save_encoder(encoder, 'src/model/encoder.pkl')
save_encoder(lb, 'src/model/label_encoder.pkl')

# Compute metrics on test set
preds = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {fbeta:.4f}")
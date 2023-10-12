from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_read_root():
    """Test the GET method."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Machine Learning Model API!"}


def test_predict_positive():
    """Test the POST method for a positive inference."""
    # Adjust the input payload for a "positive" prediction (e.g., salary >50K)
    data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == ">50K"


def test_predict_negative():
    """Test the POST method for a negative inference."""
    # Adjust the input payload for a "negative" prediction (e.g., salary <=50K)
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 234567,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Craft-repair",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == "<=50K"

if __name__ == "__main__":
    test_read_root()
    test_predict_positive()
    test_predict_negative()
    print("All tests passed!")

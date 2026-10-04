from pathlib import Path
import joblib

def test_model_exists() :
    model_path = Path("model.pkl")

    assert model_path.exists()

def test_model_can_be_loaded():
    model = joblib.load("model.pkl")

    assert model is not None

def test_model_can_predict():
    model = joblib.load("model.pkl")

    sample = [[
        8.3,
        41.0,
        6.9,
        1.0,
        322.0,
        2.5,
        37.8,
        -122.4
    ]]

    prediction = model.predict(sample)

    assert len(prediction) == 1
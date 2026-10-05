from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
from app import main

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status" : "ok"}

def test_home():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Diabetes Model Demo" in response.text

def valid_payload():
    return {
        "age": 45,
        "gender": "Male",
        "polyuria": "Yes",
        "polydipsia": "Yes",
        "sudden_weight_loss": "No",
        "weakness": "Yes",
        "polyphagia": "No",
        "genital_thrush": "No",
        "visual_blurring": "No",
        "itching": "No",
        "irritability": "No",
        "delayed_healing": "No",
        "partial_paresis": "No",
        "muscle_stiffness": "No",
        "alopecia": "No",
        "obesity": "No"
    }

def invalid_payload():
    return {
        "age": 45,
        "gender": "Male",
        "polyuria": "Maybe",
        "polydipsia": "Yes",
        "sudden_weight_loss": "No",
        "weakness": "Yes",
        "polyphagia": "No",
        "genital_thrush": "No",
        "visual_blurring": "No",
        "itching": "No",
        "irritability": "No",
        "delayed_healing": "No",
        "partial_paresis": "No",
        "muscle_stiffness": "No",
        "alopecia": "No",
        "obesity": "No"
    }

def invalid_age():
    return {
        
        "gender": "Male",
        "polyuria": "Maybe",
        "polydipsia": "Yes",
        "sudden_weight_loss": "No",
        "weakness": "Yes",
        "polyphagia": "No",
        "genital_thrush": "No",
        "visual_blurring": "No",
        "itching": "No",
        "irritability": "No",
        "delayed_healing": "No",
        "partial_paresis": "No",
        "muscle_stiffness": "No",
        "alopecia": "No",
        "obesity": "No"
    }


def test_predict_valid():
    with TestClient(app) as client:
        response = client.post("/predict" , json = valid_payload())

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] in ["Positive" , "Negative"]
    assert data["disclaimer"] == "Educational prediction only, not a diagnosis"

def test_predict_invalid_category():
    with TestClient(app) as client:
        response = client.post("/predict" , json = invalid_payload())

    assert response.status_code == 422

def test_predict_missing_age():
    with TestClient(app) as client:
        response = client.post("/predict" , json = invalid_age())

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "age"]
        and error["type"] == "missing"
        for error in response.json()["detail"]
    )

def test_predict_model_failure():
    with patch.object(main.model , "predict" , side_effect = RuntimeError("Simulated internal failure")):
        with TestClient(app) as client:
            response = client.post("/predict" , json = valid_payload())

    assert response.status_code == 500
    assert response.json() == { "detail" : "Unable to generate a prediction"}

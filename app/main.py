from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from pathlib import Path
import pandas as pd
import joblib
import logging
from fastapi.responses import FileResponse
import os

default_model_path = (Path(__file__).resolve().parent.parent / "models" / "diabetes_pipeline.joblib")

model_path = Path(os.environ.get("MODEL_PATH", str(default_model_path))).resolve()

model = joblib.load(model_path)

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status" : "ok"
    }

class AgeRequest(BaseModel):
    age: int = Field(strict = True, ge = 1, le = 120)

@app.post("/practice")
def practice(request : AgeRequest):
    return {
        "received_age" : request.age,
        "message" : "Input received successfully"
    }

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    age : int = Field(strict = True, ge = 1 , le = 120)
    gender: Literal["Male" , "Female"]
    polyuria: Literal ["Yes" , "No"]
    polydipsia: Literal["Yes", "No"]
    sudden_weight_loss: Literal["Yes", "No"]
    weakness: Literal["Yes", "No"]
    polyphagia: Literal["Yes", "No"]
    genital_thrush: Literal["Yes", "No"]
    visual_blurring: Literal["Yes", "No"]
    itching: Literal["Yes", "No"]
    irritability: Literal["Yes", "No"]
    delayed_healing: Literal["Yes", "No"]
    partial_paresis: Literal["Yes", "No"]
    muscle_stiffness: Literal["Yes", "No"]
    alopecia: Literal["Yes", "No"]
    obesity: Literal["Yes", "No"]

class PredictionResponse(BaseModel):
    predicted_class : Literal["Positive" , "Negative"]
    disclaimer : str



@app.post("/validate-input")
def validate_input(request: PredictionRequest):
    return request.model_dump()

MODEL_COLUMNS = {
    "age": "Age",
    "gender": "Gender",
    "polyuria": "Polyuria",
    "polydipsia": "Polydipsia",
    "sudden_weight_loss": "sudden weight loss",
    "weakness": "weakness",
    "polyphagia": "Polyphagia",
    "genital_thrush": "Genital thrush",
    "visual_blurring": "visual blurring",
    "itching": "Itching",
    "irritability": "Irritability",
    "delayed_healing": "delayed healing",
    "partial_paresis": "partial paresis",
    "muscle_stiffness": "muscle stiffness",
    "alopecia": "Alopecia",
    "obesity": "Obesity"
}

@app.post("/predict" , response_model =PredictionResponse)
def predict(request: PredictionRequest):

    try:

        data = request.model_dump()

        inputs = pd.DataFrame([data])
        inputs = inputs.rename(columns=MODEL_COLUMNS)

        prediction = int(model.predict(inputs)[0])

    except Exception:
        logger.exception("Prediction Failed")

        raise HTTPException(status_code = 500 , detail = "Unable to generate a prediction") from None

    if prediction == 1:
        label = "Positive"
    else:
        label = "Negative"

    return {
        "predicted_class" : label ,
        "disclaimer" : "Educational prediction only, not a diagnosis"
    }

@app.get("/" , response_class=FileResponse)
def home():
    page_path = Path(__file__).resolve().parent / "index.html"
    return FileResponse(page_path)

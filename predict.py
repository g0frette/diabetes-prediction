from pathlib import Path
import pandas as pd
import joblib

model_path = Path(__file__).resolve().parent / "models" / "diabetes_pipeline.joblib"
loaded_model = joblib.load(model_path)

new_person = pd.DataFrame([{
    "Age": 45,
    "Gender": "Male",
    "Polyuria": "Yes",
    "Polydipsia": "Yes",
    "sudden weight loss": "No",
    "weakness": "Yes",
    "Polyphagia": "No",
    "Genital thrush": "No",
    "visual blurring": "No",
    "Itching": "No",
    "Irritability": "No",
    "delayed healing": "No",
    "partial paresis": "No",
    "muscle stiffness": "No",
    "Alopecia": "No",
    "Obesity": "No"
}])


predictions= loaded_model.predict(new_person)[0]

if predictions == 1:
    print("Predicted class: Positive")

else:
    print("Predicted class: Negative")

print("Educational tool only, not a diagnotic")

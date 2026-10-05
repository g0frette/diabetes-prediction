# Diabetes Prediction

An educational machine-learning project with a FastAPI API, browser form, and Python client. It predicts dataset labels, not a medical diagnosis. 

## Project structure

- `app/main.py`: API and input validation.
- `app/index.html`: browser form.
- `client.py`: example HTTP client.
- `predict.py`: loads the saved model for a local example.
- `training/train_model.py`: trains, evaluates, saves, and verifies the pipeline.
- `models/diabetes_pipeline.joblib`: fitted preprocessing and decision tree.
- `data/diabetes_data_upload.csv`: original dataset.
- `tests/test_api.py`: API tests.
- `MODEL_CARD.md`: results, dataset attribution, and limitations.

## Set up

Use Python 3.13.15, the version used for this project. Run from this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
```

The requirements pin the main packages, not every indirect dependency. Only load trusted joblib files; they can execute code. The saved model should use the recorded library versions.

## Run the app locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ for the form, or http://127.0.0.1:8000/docs for API documentation.
Add `--reload` only during development.

The API loads `models/diabetes_pipeline.joblib` by default. Set `MODEL_PATH` to an absolute trusted file path to override it. The API never trains the model during requests.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Example clients

With the server running:

```powershell
.\.venv\Scripts\python.exe client.py
```

For an example that loads the saved pipeline directly, without the server:

```powershell
.\.venv\Scripts\python.exe predict.py
```

## Retrain (optional)

This command replaces the saved model file:

```powershell
.\.venv\Scripts\python.exe training/train_model.py
```

Do not import the training script into the API: its top-level code trains and saves.

## Results and limitations

The selected tree scored 84.31% accuracy and 88.57% Positive-class F1 on 51 test rows. These are small-dataset educational results, not evidence of clinical reliability. Test results were inspected during lessons. See the model card for full details.

Dataset: *Early Stage Diabetes Risk Prediction* (2020), UCI Machine Learning Repository, https://doi.org/10.24432/C5VG8H. Licensed CC BY 4.0; retain attribution. The original CSV is included unchanged.

## GitHub

This folder is the standalone project; no lesson folders or virtual environments are needed. No Git repository, commit, remote, or push was created during preparation. Review the files before adding them to your personal repository. A private repository does not automatically make a deployed API private.

The app is local only. Public deployment and deployed checks are still pending.

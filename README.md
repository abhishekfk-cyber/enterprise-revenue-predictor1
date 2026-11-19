# Enterprise Revenue Predictor

A small Flask app with a project overview page, an enterprise input form, and a prediction result page.

## Files

- `app.py` — Flask routes, input validation, and model inference
- `revenue_model.pkl` — fitted preprocessing and Random Forest model
- `model_metadata.json` — model feature names
- `requirements.txt` — Python dependencies
- `templates/` and `static/` — page templates and styling

## Run locally in VS Code on Windows

Open this folder in VS Code, then run these commands in its terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in a browser.

The model artifact is a Python pickle/joblib file. Only load model files you trust.
The model was trained with revenue values in mixed reported currencies, so its
prediction is not normalized to USD and the result page labels this limitation.

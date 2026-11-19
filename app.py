import json
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "revenue_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

with METADATA_PATH.open(encoding="utf-8") as metadata_file:
    METADATA = json.load(metadata_file)

FEATURE_COLUMNS = METADATA["features"]
NUMERIC_FIELDS = {
    "employees",
    "employees_year",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
}
BOOLEAN_FIELDS = {"is_standalone_entity"}
LABELS = {
    "employees": "Employees",
    "employees_year": "Employees data year",
    "company_type": "Company type",
    "enterprise_group_jurisdiction": "Jurisdiction",
    "enterprise_group_entity_status": "Entity status",
    "legal_units_count": "Legal units",
    "direct_subsidiaries_count": "Direct subsidiaries",
    "max_hierarchy_depth": "Maximum hierarchy depth",
    "is_standalone_entity": "Standalone entity",
}

app = Flask(__name__)


def build_form_fields(values=None):
    values = values or {}
    return [
        {
            "name": name,
            "label": LABELS.get(name, name.replace("_", " ").title()),
            "kind": (
                "number"
                if name in NUMERIC_FIELDS
                else "boolean"
                if name in BOOLEAN_FIELDS
                else "text"
            ),
            "value": values.get(name, ""),
        }
        for name in FEATURE_COLUMNS
    ]


def load_pipeline():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "revenue_model.pkl is missing. Place it beside app.py."
        )

    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline")
        artifact_features = artifact.get("feature_columns", FEATURE_COLUMNS)
    else:
        pipeline = artifact
        artifact_features = FEATURE_COLUMNS

    if pipeline is None or not callable(getattr(pipeline, "predict", None)):
        raise RuntimeError("The model file does not contain a prediction pipeline.")
    if list(artifact_features) != list(FEATURE_COLUMNS):
        raise RuntimeError(
            "The feature columns in the model and metadata do not match."
        )

    return pipeline


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/predict")
def predict_form():
    return render_template("predict.html", fields=build_form_fields(), errors=[])


@app.post("/result")
def result():
    input_values = {}
    errors = []

    for name in FEATURE_COLUMNS:
        raw_value = request.form.get(name, "").strip()
        if not raw_value:
            errors.append(f"{LABELS.get(name, name)} is required.")
            continue

        if name in NUMERIC_FIELDS:
            try:
                number = float(raw_value)
            except ValueError:
                errors.append(f"{LABELS.get(name, name)} must be a number.")
                continue
            if not math.isfinite(number):
                errors.append(f"{LABELS.get(name, name)} must be a finite number.")
                continue
            input_values[name] = number
        elif name in BOOLEAN_FIELDS:
            if raw_value not in {"true", "false"}:
                errors.append(f"Choose true or false for {LABELS.get(name, name)}.")
                continue
            input_values[name] = raw_value == "true"
        else:
            input_values[name] = raw_value

    if errors:
        return (
            render_template(
                "predict.html",
                fields=build_form_fields(request.form),
                errors=errors,
            ),
            400,
        )

    if not MODEL_PATH.is_file():
        return (
            render_template(
                "error.html",
                message="The model file is missing. Place revenue_model.pkl beside app.py.",
            ),
            503,
        )

    pipeline = load_pipeline()
    input_frame = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = float(pipeline.predict(input_frame)[0])

    return render_template("result.html", prediction=f"{prediction:,.2f}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

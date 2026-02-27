import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ml_utils import load_bundle


class PredictRequest(BaseModel):
    features: Dict[str, Any]


app = FastAPI(title="Privacy-Preserving Risk API", version="0.1.0")
MODEL_BUNDLE_PATH = os.environ.get("MODEL_BUNDLE", "models/dp_bundle.joblib")

bundle = None


@app.on_event("startup")
def startup() -> None:
    global bundle
    if not os.path.exists(MODEL_BUNDLE_PATH):
        bundle = None
        return
    bundle = load_bundle(MODEL_BUNDLE_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_path": MODEL_BUNDLE_PATH,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, Any]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")

    feature_cols = bundle["feature_cols"]
    row = {col: payload.features.get(col, None) for col in feature_cols}

    df = pd.DataFrame([row])
    X_t = bundle["preprocessor"].transform(df)
    probs = bundle["model"].predict_proba(X_t)[0]

    positive_idx = 1 if len(probs) > 1 else int(np.argmax(probs))
    risk_score = float(probs[positive_idx])
    prediction = int(risk_score >= 0.5)

    return {
        "prediction": prediction,
        "risk_score": risk_score,
        "positive_label": bundle.get("positive_label", "1"),
    }

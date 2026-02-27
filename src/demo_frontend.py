import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml_utils import load_bundle


BASELINE_BUNDLE_PATH = ROOT_DIR / "models" / "baseline_bundle.joblib"
DP_BUNDLE_PATH = ROOT_DIR / "models" / "dp_bundle.joblib"


@st.cache_resource
def load_model_bundle(path: Path):
    if not path.exists():
        return None
    return load_bundle(str(path))


def extract_schema(bundle: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    pre = bundle["preprocessor"]
    feature_cols = bundle["feature_cols"]

    numeric_cols: List[str] = []
    categorical_options: Dict[str, List[str]] = {}

    for name, transformer, cols in pre.transformers_:
        if name == "num":
            numeric_cols = list(cols)
        if name == "cat":
            cat_cols = list(cols)
            onehot = transformer.named_steps["onehot"]
            for col, cats in zip(cat_cols, onehot.categories_):
                categorical_options[col] = [str(v) for v in cats]

    return feature_cols, numeric_cols, categorical_options


def default_value_for_numeric(name: str) -> float:
    defaults = {
        "age": 50.0,
        "bmi": 28.0,
        "glucose": 120.0,
        "blood_pressure": 80.0,
        "insulin": 100.0,
    }
    return float(defaults.get(name, 0.0))


def predict(bundle: Dict[str, Any], features: Dict[str, Any]) -> Tuple[int, float]:
    row = {col: features.get(col, None) for col in bundle["feature_cols"]}
    df = pd.DataFrame([row])
    X_t = bundle["preprocessor"].transform(df)

    probs = bundle["model"].predict_proba(X_t)[0]
    positive_idx = 1 if len(probs) > 1 else int(np.argmax(probs))
    risk_score = float(probs[positive_idx])
    label = int(risk_score >= 0.5)
    return label, risk_score


def render_prediction_card(title: str, prediction: int, risk_score: float):
    st.subheader(title)
    st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
    st.metric("Risk Score", f"{risk_score:.3f}")
    st.progress(min(max(risk_score, 0.0), 1.0))


def main():
    st.set_page_config(page_title="Healthcare Risk Demo", layout="wide")
    st.title("Privacy-Preserving Healthcare Risk Prediction")
    st.caption("Live demo: compare non-private baseline vs differentially private model")

    baseline_bundle = load_model_bundle(BASELINE_BUNDLE_PATH)
    dp_bundle = load_model_bundle(DP_BUNDLE_PATH)

    if baseline_bundle is None and dp_bundle is None:
        st.error("No trained model bundles found. Train models first in the terminal.")
        st.code(
            "python3 train_baseline.py --data data/sample_health.csv --target outcome\n"
            "python3 train_dp.py --data data/sample_health.csv --target outcome --epsilon 2.0 --delta 1e-5"
        )
        return

    reference_bundle = dp_bundle if dp_bundle is not None else baseline_bundle
    feature_cols, numeric_cols, categorical_options = extract_schema(reference_bundle)

    st.markdown("### Patient Inputs")
    with st.form("prediction_form"):
        inputs: Dict[str, Any] = {}
        left_col, right_col = st.columns(2)

        for i, col in enumerate(feature_cols):
            target_col = left_col if i % 2 == 0 else right_col
            with target_col:
                if col in numeric_cols:
                    inputs[col] = st.number_input(
                        col,
                        value=default_value_for_numeric(col),
                        step=1.0 if col in {"age", "blood_pressure", "glucose", "insulin"} else 0.1,
                    )
                else:
                    options = categorical_options.get(col, ["yes", "no"])
                    inputs[col] = st.selectbox(col, options=options, index=0)

        submitted = st.form_submit_button("Run Prediction")

    if not submitted:
        st.info("Enter patient values and click Run Prediction.")
        return

    result_col_1, result_col_2 = st.columns(2)

    if baseline_bundle is not None:
        baseline_pred, baseline_risk = predict(baseline_bundle, inputs)
        with result_col_1:
            render_prediction_card("Baseline Model (No DP)", baseline_pred, baseline_risk)
            st.caption(f"Accuracy: {baseline_bundle['metrics']['accuracy']:.3f}")
    else:
        with result_col_1:
            st.warning("Baseline bundle not found.")

    if dp_bundle is not None:
        dp_pred, dp_risk = predict(dp_bundle, inputs)
        with result_col_2:
            render_prediction_card("DP Model (Private)", dp_pred, dp_risk)
            st.caption(f"Accuracy: {dp_bundle['metrics']['accuracy']:.3f}")
            privacy = dp_bundle.get("privacy", {})
            if privacy:
                st.caption(
                    f"Privacy budget: epsilon={privacy.get('epsilon')} | delta={privacy.get('delta')}"
                )
    else:
        with result_col_2:
            st.warning("DP bundle not found.")

if __name__ == "__main__":
    main()

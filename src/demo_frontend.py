import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dp_logistic import DPLogisticRegression
from src.ml_utils import (
    build_preprocessor,
    evaluate_binary,
    load_bundle,
    save_bundle,
    split_features_target,
)


BASELINE_BUNDLE_PATH = ROOT_DIR / "models" / "baseline_bundle.joblib"
DP_BUNDLE_PATH = ROOT_DIR / "models" / "dp_bundle.joblib"


@st.cache_resource
def load_model_bundle(path: Path):
    if not path.exists():
        return None
    return load_bundle(str(path))


def bootstrap_demo_models() -> None:
    if BASELINE_BUNDLE_PATH.exists() and DP_BUNDLE_PATH.exists():
        return

    demo_df = pd.DataFrame(
        [
            [45, 28.1, 140, 82, 130, "low", "yes", 1],
            [34, 22.4, 95, 76, 80, "high", "no", 0],
            [52, 31.7, 168, 88, 160, "low", "yes", 1],
            [29, 24.2, 101, 72, 85, "medium", "no", 0],
            [61, 33.5, 182, 92, 190, "low", "yes", 1],
            [40, 27.0, 120, 80, 110, "medium", "yes", 0],
            [55, 30.2, 150, 86, 145, "low", "yes", 1],
            [31, 23.8, 98, 74, 78, "high", "no", 0],
            [48, 29.5, 142, 84, 135, "medium", "yes", 1],
            [27, 21.9, 90, 70, 70, "high", "no", 0],
            [58, 34.1, 176, 90, 175, "low", "yes", 1],
            [36, 25.1, 108, 78, 95, "medium", "no", 0],
            [50, 30.8, 160, 87, 155, "low", "yes", 1],
            [33, 24.0, 99, 73, 82, "high", "no", 0],
            [62, 35.0, 188, 94, 200, "low", "yes", 1],
            [39, 26.5, 116, 79, 105, "medium", "no", 0],
            [47, 28.9, 146, 83, 138, "medium", "yes", 1],
            [30, 22.7, 93, 71, 75, "high", "no", 0],
            [56, 32.4, 170, 89, 168, "low", "yes", 1],
            [35, 24.8, 104, 77, 90, "medium", "no", 0],
        ],
        columns=[
            "age",
            "bmi",
            "glucose",
            "blood_pressure",
            "insulin",
            "activity_level",
            "family_history",
            "outcome",
        ],
    )

    X_train, X_test, y_train, y_test = split_features_target(
        demo_df, "outcome", test_size=0.25, random_state=42
    )
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_t, y_train.to_numpy())
    b_prob = baseline_model.predict_proba(X_test_t)[:, 1]
    b_pred = (b_prob >= 0.5).astype(int)
    baseline_metrics = evaluate_binary(y_test.to_numpy(), b_pred, b_prob)

    dp_model = DPLogisticRegression(
        epsilon=2.0,
        delta=1e-5,
        learning_rate=0.05,
        epochs=35,
        batch_size=8,
        random_state=42,
    )
    dp_model.fit(X_train_t, y_train.to_numpy())
    d_prob = dp_model.predict_proba(X_test_t)[:, 1]
    d_pred = (d_prob >= 0.5).astype(int)
    dp_metrics = evaluate_binary(y_test.to_numpy(), d_pred, d_prob)

    feature_cols = X_train.columns.tolist()
    classes = sorted([str(c) for c in np.unique(y_train)])

    save_bundle(
        str(BASELINE_BUNDLE_PATH),
        {
            "model_type": "baseline_logreg",
            "model": baseline_model,
            "preprocessor": preprocessor,
            "target_col": "outcome",
            "feature_cols": feature_cols,
            "positive_label": "1",
            "classes": classes,
            "metrics": baseline_metrics,
        },
    )

    save_bundle(
        str(DP_BUNDLE_PATH),
        {
            "model_type": "dp_logreg",
            "model": dp_model,
            "preprocessor": preprocessor,
            "target_col": "outcome",
            "feature_cols": feature_cols,
            "positive_label": "1",
            "classes": classes,
            "privacy": {"epsilon": 2.0, "delta": 1e-5, "epochs": 35, "batch_size": 8},
            "metrics": dp_metrics,
        },
    )


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

    bootstrap_demo_models()
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

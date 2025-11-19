import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Directory to store global backend model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pkl")


# Ensure models/ directory exists
def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


# Auto-detect which column should be treated as the label
def auto_split_label(df: pd.DataFrame):
    """
    Heuristic:
    - Choose the LAST column with <= 20 unique values (categorical or low-cardinality numeric)
    - This allows flexible handling of various datasets
    """
    for col in reversed(df.columns):
        if df[col].nunique() <= 20:
            return col
    return None  # No label could be inferred


# Build a preprocessing transformer dynamically based on dataset structure
def build_preprocessing(df: pd.DataFrame, label_col: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="drop"
    )
    return transformer


# Build an incremental-learning model pipeline
def build_model_pipeline(preprocessor):
    """
    SGDClassifier supports partial_fit() which allows continuous learning.
    This turns your backend model into a self-improving engine.
    """
    model = SGDClassifier(loss="log_loss")
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", model)
    ])
    return pipeline


# Train or update your lifelong backend model
def train_global_model(df: pd.DataFrame):
    """
    Main backend logic:
    - Detect label column
    - Create preprocessing dynamically
    - Split 80/20 (train/validation)
    - Load or initialize global model
    - Incrementally train model using partial_fit
    - Save updated model
    - Return validation accuracy
    """

    ensure_model_dir()

    # 1. Auto-detect label
    label_col = auto_split_label(df)
    if label_col is None:
        # Dataset not suitable for backend ML training
        return None

    # Separate features and label
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # 2. 80/20 backend split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Build preprocessing
    preprocessor = build_preprocessing(df, label_col)

    # 4. Load global model if exists
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        # First-time training
        model = build_model_pipeline(preprocessor)

        # First partial_fit needs class list
        Xt_init = preprocessor.fit_transform(X_train)
        model.named_steps["clf"].partial_fit(Xt_init, y_train, classes=np.unique(y_train))

        # Save early stage model
        joblib.dump(model, MODEL_PATH)

    # 5. Incrementally train model with new dataset
    try:
        Xt = model.named_steps["preprocess"].fit_transform(X_train)
        model.named_steps["clf"].partial_fit(Xt, y_train)
    except Exception:
        return None

    # 6. Validate accuracy
    try:
        Xv = model.named_steps["preprocess"].transform(X_val)
        preds = model.named_steps["clf"].predict(Xv)
        acc = accuracy_score(y_val, preds)
    except Exception:
        acc = None

    # 7. Save updated model
    joblib.dump(model, MODEL_PATH)

    return acc

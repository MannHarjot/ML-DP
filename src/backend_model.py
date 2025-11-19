import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "global_vectorizer.pkl")


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _auto_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Choose a label column heuristically:
    - Last column with <= 20 unique values.
    """
    for col in reversed(df.columns):
        if df[col].nunique(dropna=True) <= 20:
            return col
    return None


def _to_feature_dicts(df: pd.DataFrame):
    """
    Convert DataFrame rows into dictionaries {column: value}
    suitable for DictVectorizer. This is schema-flexible.
    """
    return df.to_dict(orient="records")


def train_global_model(df: pd.DataFrame) -> Optional[float]:
    """
    Train or update a global incremental model using 80% of the raw data.
    - Uses SGDClassifier + DictVectorizer (feature hashing-like).
    - Schema-flexible: can handle changing columns between datasets.
    - Never stores raw data; only stores model + vectorizer.
    """
    _ensure_model_dir()

    # 1. Find label column
    label_col = _auto_label_column(df)
    if label_col is None:
        return None

    # Optionally downsample very large datasets
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    X_df = df.drop(columns=[label_col])
    y = df[label_col]

    # 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # 2. Vectorizer
    if os.path.exists(VECTORIZER_PATH):
        vec: DictVectorizer = joblib.load(VECTORIZER_PATH)
        X_train_vec = vec.transform(_to_feature_dicts(X_train))
        X_val_vec = vec.transform(_to_feature_dicts(X_val))
    else:
        vec = DictVectorizer(sparse=True)
        X_train_vec = vec.fit_transform(_to_feature_dicts(X_train))
        X_val_vec = vec.transform(_to_feature_dicts(X_val))
        joblib.dump(vec, VECTORIZER_PATH)

    # 3. Model
    if os.path.exists(MODEL_PATH):
        model: SGDClassifier = joblib.load(MODEL_PATH)
    else:
        model = SGDClassifier(loss="log_loss")
        # First partial_fit needs the class list
        model.partial_fit(X_train_vec, y_train, classes=np.unique(y_train))

    # 4. Incremental training
    model.partial_fit(X_train_vec, y_train)

    # 5. Validation
    try:
        y_pred = model.predict(X_val_vec)
        acc = accuracy_score(y_val, y_pred)
    except Exception:
        acc = None

    # 6. Save updated model
    joblib.dump(model, MODEL_PATH)

    return acc

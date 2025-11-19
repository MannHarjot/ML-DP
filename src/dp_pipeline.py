import io
import pandas as pd
import numpy as np


def load_dataframe(uploaded_file, filename: str) -> pd.DataFrame:
    """
    Load a user-uploaded file (CSV or Excel) into a pandas DataFrame.
    The file is kept only in memory.
    """
    name = filename.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        # Try CSV as fallback
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    if df.empty:
        raise ValueError("Uploaded file appears to be empty.")

    return df


def validate_dataframe(df: pd.DataFrame):
    """
    Basic validation to check if the dataset is usable.
    Returns (is_valid: bool, warnings: list[str]).
    """
    warnings = []

    if df.shape[0] < 10:
        warnings.append("Dataset has fewer than 10 rows. DP may severely hurt utility.")

    if df.shape[1] < 2:
        warnings.append("Dataset has fewer than 2 columns. Not very useful for ML.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        warnings.append(
            "No numeric columns found. Current DP pipeline only adds noise to numeric columns."
        )

    return True, warnings

def detect_quasi_identifiers(df):
    qid_cols = []

    # Typical QID patterns
    common_qids = ["age", "gender", "sex", "zip", "zipcode", "city", "state",
                   "dob", "birth", "occupation", "education", "marital"]

    for col in df.columns:
        col_lower = col.lower()
        for q in common_qids:
            if q in col_lower:
                qid_cols.append(col)
                break

    # Also detect high-cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 20:
            qid_cols.append(col)

    return list(set(qid_cols))

def dp_randomized_response(val, epsilon, categories):
    """
    Differential privacy for categorical variables using randomized response.
    """
    if np.random.rand() < (np.exp(epsilon) / (np.exp(epsilon) + len(categories) - 1)):
        return val  # keep true value
    else:
        return np.random.choice(categories)

def apply_dp(df, epsilon):
    dp_df = df.copy()

    numeric_cols = dp_df.select_dtypes(include=[np.number]).columns
    categorical_cols = dp_df.select_dtypes(include=['object']).columns

    qid_cols = detect_quasi_identifiers(dp_df)

    # 1. Numeric DP noise
    for col in numeric_cols:
        sensitivity = dp_df[col].max() - dp_df[col].min()
        noise = np.random.laplace(0, sensitivity / epsilon, size=len(dp_df))
        dp_df[col] = dp_df[col] + noise

    # 2. Categorical DP noise
    for col in categorical_cols:
        categories = dp_df[col].dropna().unique()
        dp_df[col] = dp_df[col].apply(
            lambda x: dp_randomized_response(x, epsilon, categories)
        )

    # 3. Generalization for QIDs
    for col in qid_cols:
        if col.lower().startswith("age"):
            dp_df[col] = pd.cut(dp_df[col], bins=[0,20,30,40,50,60,70,100],
                                labels=["0-20","20-30","30-40","40-50","50-60","60-70","70+"])
        if col.lower().startswith("zip"):
            dp_df[col] = dp_df[col].astype(str).str[:3] + "***"

    return dp_df



def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for downloading via the interface.
    """
    csv_str = df.to_csv(index=False)
    return csv_str.encode("utf-8")

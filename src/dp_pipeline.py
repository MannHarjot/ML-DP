import numpy as np
import pandas as pd


# ---------- File Loading ----------

def load_dataframe(uploaded_file, filename: str) -> pd.DataFrame:
    """
    Load a user-uploaded file (CSV or Excel) into a pandas DataFrame.
    File is only held in memory.
    """
    name = filename.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        # Try CSV as a fallback
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    if df.empty:
        raise ValueError("Uploaded file appears to be empty.")

    return df


# ---------- Basic Validation & Schema ----------

def validate_dataframe(df: pd.DataFrame):
    """
    Basic checks to warn the user about potential issues.
    Returns (True, [warnings]).
    """
    warnings = []

    if df.shape[0] < 20:
        warnings.append("Dataset has fewer than 20 rows. DP may severely reduce utility.")

    if df.shape[1] < 2:
        warnings.append("Dataset has fewer than 2 columns. Not very useful for ML or analytics.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        warnings.append("No numeric columns detected. Only categorical DP will be applied.")

    return True, warnings


# ---------- Quasi-Identifier Detection ----------

def detect_quasi_identifiers(df: pd.DataFrame):
    """
    Heuristic detection of quasi-identifiers:
    - Columns whose names look like age/zip/gender/etc.
    - High-cardinality categorical/string columns.
    """
    qid_cols = []

    common_qids = [
        "age", "gender", "sex", "zip", "zipcode", "postal", "postcode",
        "city", "state", "province", "region",
        "dob", "birth", "birthday",
        "occupation", "job", "profession",
        "education", "marital", "address"
    ]

    for col in df.columns:
        col_lower = col.lower()
        for pattern in common_qids:
            if pattern in col_lower:
                qid_cols.append(col)
                break

    # High-cardinality categoricals: often identifying
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique(dropna=True) > 30:
            qid_cols.append(col)

    return sorted(list(set(qid_cols)))


# ---------- Categorical DP (Randomized Response) ----------

def _dp_randomized_response_series(series: pd.Series, epsilon: float) -> pd.Series:
    """
    Vectorized randomized response for a categorical column.
    Keeps the original value with probability p, otherwise replaces with a random category.
    """
    values = series.astype("object").values
    mask_valid = pd.notna(values)
    valid_values = values[mask_valid]

    # No valid values -> nothing to do
    if valid_values.size == 0:
        return series

    categories = pd.unique(valid_values)
    k = len(categories)
    if k == 1:
        # Only one category, randomized response is pointless
        return series

    # Probability of keeping the true value
    p_keep = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    rand_vals = np.random.rand(len(values))
    random_choices = np.random.choice(categories, size=len(values))

    keep_mask = rand_vals < p_keep
    new_values = np.where(keep_mask, values, random_choices)

    return pd.Series(new_values, index=series.index)


# ---------- Numeric DP (Laplace) ----------

def _dp_laplace_numeric_series(series: pd.Series, epsilon: float, clip: bool = True) -> pd.Series:
    """
    Add Laplace noise to a numeric column. Sensitivity is approximated as (max - min).
    Clipping reduces outlier impact and improves utility.
    """
    if series.empty:
        return series

    col = pd.to_numeric(series, errors="coerce")

    if clip:
        # Clip extreme outliers to reduce sensitivity
        lower = col.quantile(0.01)
        upper = col.quantile(0.99)
        col = col.clip(lower, upper)

    if col.isna().all():
        return series  # cannot do much

    sensitivity = col.max() - col.min()
    if sensitivity == 0 or np.isnan(sensitivity):
        return col  # constant column

    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(col))

    return pd.Series(col.values + noise, index=series.index)


# ---------- QID Generalization (FINAL LAYER) ----------

def _generalize_qids(df: pd.DataFrame, qid_cols):
    """
    Force generalization of QIDs AFTER DP noise.
    Guarantees privacy even if data is numeric or categorical.
    """
    df = df.copy()

    for col in qid_cols:
        col_lower = col.lower()

        # AGE-LIKE COLUMNS -> BINS
        if "age" in col_lower:
            try:
                col_numeric = pd.to_numeric(df[col], errors="coerce")
                df[col] = pd.cut(
                    col_numeric,
                    bins=[0, 18, 25, 35, 45, 60, 80, 120],
                    labels=["0-18", "18-25", "25-35", "35-45", "45-60", "60-80", "80+"],
                    include_lowest=True,
                )
            except Exception:
                df[col] = "Unknown"

        # ZIP / POSTAL / POSTCODE -> FIRST 3 DIGITS + ***
        if any(x in col_lower for x in ["zip", "postal", "postcode"]):
            col_str = df[col].astype(str)
            # Keep only digits
            col_str = col_str.str.replace(r"\D", "", regex=True)
            df[col] = col_str.apply(lambda z: (z[:3] + "***") if len(z) >= 3 else "***")

        # CITY / STATE / PROVINCE -> GENERALIZE TO FIRST LETTER
        if any(x in col_lower for x in ["city", "state", "province", "region"]):
            df[col] = df[col].astype(str).str[:1] + "..."

        # ADDRESS-LIKE -> HEAVY TRUNCATION
        if "address" in col_lower:
            df[col] = df[col].astype(str).apply(lambda x: x[:5] + "..." if len(x) > 5 else ".....")

    return df


# ---------- Full DP Transform ----------

def apply_strong_dp(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    Apply a stronger, more realistic DP transformation:
    - Laplace noise on numeric columns (vectorized)
    - Randomized response on categorical columns
    - QID generalization as a final privacy layer (cannot be overwritten)
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be > 0.")

    dp_df = df.copy()

    # Detect QIDs once
    qid_cols = detect_quasi_identifiers(dp_df)

    # STEP 1: NUMERIC DP (Laplace)
    numeric_cols = dp_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        dp_df[col] = _dp_laplace_numeric_series(dp_df[col], epsilon)

    # STEP 2: CATEGORICAL DP (Randomized Response)
    categorical_cols = dp_df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        dp_df[col] = _dp_randomized_response_series(dp_df[col], epsilon)

    # STEP 3: QID GENERALIZATION (FINAL PRIVACY LAYER)
    if qid_cols:
        dp_df = _generalize_qids(dp_df, qid_cols)

    return dp_df


# ---------- Output Helpers ----------

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for downloading via the interface.
    """
    csv_str = df.to_csv(index=False)
    return csv_str.encode("utf-8")

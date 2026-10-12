"""
Preprocessor — applies dataset-specific preprocessing based on AI decisions:
  - Missing value imputation
  - Categorical encoding
  - Feature scaling (standard / minmax / robust)
  - Class imbalance handling (SMOTE / oversample / undersample)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler, OrdinalEncoder
)


def preprocess(
    df: pd.DataFrame,
    decisions: dict,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Preprocess the dataframe according to AI decisions.

    Returns:
        X_processed  : pd.DataFrame with processed features
        y            : pd.Series with target
        meta         : preprocessing metadata (encoders, scalers, etc.)
    """
    target_col: str = decisions["target_column"]
    preprocessing_steps: list = decisions.get("recommended_preprocessing", [])
    scaler_type: str = decisions.get("scaler_type", "standard")
    handle_imbalance: bool = decisions.get("handle_imbalance", False)
    imbalance_method: str | None = decisions.get("imbalance_method", None)
    task_type: str = decisions.get("task_type", "classification")

    meta: dict = {
        "preprocessing_steps": preprocessing_steps,
        "scaler_type": scaler_type,
        "label_encoders": {},
        "target_encoder": None,
        "scaler": None,
        "dropped_cols": [],
        "one_hot_cols": [],
        "original_shape": df.shape,
        "imbalance_handled": False,
    }

    df = df.copy()

    # ── 1. Split into X, y ──────────────────────────────────────────────────
    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # ── 2. Drop columns with > 60 % missing ─────────────────────────────────
    high_missing = [c for c in X.columns if X[c].isnull().mean() > 0.60]
    if high_missing:
        X = X.drop(columns=high_missing)
        meta["dropped_cols"] = high_missing

    # ── 3. Separate numeric / categorical ───────────────────────────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── 4. Imputation ────────────────────────────────────────────────────────
    if X.isnull().any().any() or "imputation" in preprocessing_steps:
        if num_cols:
            num_imp = SimpleImputer(strategy="median")
            X[num_cols] = num_imp.fit_transform(X[num_cols])
            meta["num_imputer"] = num_imp
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
            meta["cat_imputer"] = cat_imp

    # ── 5. Categorical Encoding ──────────────────────────────────────────────
    use_ohe = "one_hot_encoding" in preprocessing_steps
    use_le = "label_encoding" in preprocessing_steps

    ohe_processed_cols = []
    for col in cat_cols:
        nunique = X[col].nunique()
        if use_ohe and nunique <= 10:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False).astype(int)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            ohe_processed_cols.append(col)
            meta["one_hot_cols"].append(col)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            meta["label_encoders"][col] = le

    # Update num_cols after encoding
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # ── 6. Encode target ─────────────────────────────────────────────────────
    if task_type == "classification":
        le_target = LabelEncoder()
        if y_raw.dtype == object or str(y_raw.dtype) == "category":
            y = pd.Series(le_target.fit_transform(y_raw.astype(str)), name=target_col)
        else:
            y = pd.Series(le_target.fit_transform(y_raw), name=target_col)
        meta["target_encoder"] = le_target
    else:
        y = y_raw.astype(float).reset_index(drop=True)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # ── 7. Scaling ───────────────────────────────────────────────────────────
    if num_cols:
        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        X[num_cols] = scaler.fit_transform(X[num_cols])
        meta["scaler"] = scaler
        meta["num_cols_scaled"] = num_cols

    # ── 8. Class imbalance ───────────────────────────────────────────────────
    if task_type == "classification" and handle_imbalance and imbalance_method:
        try:
            if imbalance_method == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=42)
            elif imbalance_method == "oversample":
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=42)
            elif imbalance_method == "undersample":
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=42)
            else:
                sampler = None

            if sampler is not None:
                X_arr, y_arr = sampler.fit_resample(X, y)
                X = pd.DataFrame(X_arr, columns=X.columns)
                y = pd.Series(y_arr, name=target_col)
                meta["imbalance_handled"] = True
                meta["imbalance_method_used"] = imbalance_method
        except Exception as e:
            # If SMOTE fails (e.g., too few samples), skip gracefully
            meta["imbalance_error"] = str(e)

    meta["final_shape"] = X.shape
    meta["feature_names"] = list(X.columns)

    return X, y, meta

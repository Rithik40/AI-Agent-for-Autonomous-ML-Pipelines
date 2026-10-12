"""
Dataset Analyzer — uses Groq LLM to inspect dataset statistics and
decide which preprocessing steps, feature selection method, and ML
models are best suited for the particular dataset.
"""
from __future__ import annotations

import json
import re
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from groq import Groq

import config


# ─────────────────────────────── helpers ────────────────────────────────────

def _json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _build_dataset_summary(df: pd.DataFrame) -> dict:
    """Build a rich but token-efficient summary of the dataframe."""
    summary: dict = {
        "shape": list(df.shape),
        "columns": [],
        "missing_values": {},
        "class_distribution": None,
        "numeric_stats": {},
    }

    # Per-column profile
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = int(df[col].nunique())
        missing = int(df[col].isnull().sum())
        col_info = {
            "name": col,
            "dtype": dtype,
            "nunique": nunique,
            "missing": missing,
        }
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            col_info["mean"] = round(float(df[col].mean()), 4)
            col_info["std"] = round(float(df[col].std()), 4)
            col_info["min"] = round(float(df[col].min()), 4)
            col_info["max"] = round(float(df[col].max()), 4)
        else:
            top_vals = df[col].value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        summary["columns"].append(col_info)

    # Missing value summary
    mv = df.isnull().sum()
    summary["missing_values"] = {c: int(v) for c, v in mv[mv > 0].items()}

    # Numeric correlation (top correlated pairs)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        top_corr = (
            upper.stack()
            .sort_values(ascending=False)
            .head(10)
        )
        summary["top_correlations"] = [
            {"col1": str(idx[0]), "col2": str(idx[1]), "corr": round(float(val), 4)}
            for idx, val in top_corr.items()
        ]

    return _json_serializable(summary)


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    # Try to find a JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try raw JSON in text
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


# ─────────────────────────────── main analyzer ───────────────────────────────

def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze the dataframe using Groq LLM.

    Returns a decision dict:
    {
        "task_type": "classification" | "regression",
        "target_column": str,
        "recommended_preprocessing": [str, ...],
        "needs_feature_selection": bool,
        "feature_selection_method": str | null,
        "recommended_models": [str, ...],
        "reasoning": str,
        "scaler_type": "standard" | "minmax" | "robust",
        "handle_imbalance": bool,
        "imbalance_method": "smote" | "oversample" | "undersample" | null,
    }
    """
    client = Groq(api_key=config.GROQ_API_KEY)

    summary = _build_dataset_summary(df)
    num_features = df.shape[1] - 1  # excluding target

    prompt = textwrap.dedent(f"""
    You are an expert data scientist. Analyze the following dataset summary and decide
    the best ML pipeline for it. Respond ONLY with a valid JSON object — no other text.

    Dataset Summary:
    {json.dumps(summary, indent=2)}

    Number of features (excluding target): {num_features}
    Feature selection threshold: {config.FEATURE_SELECTION_THRESHOLD}

    Instructions:
    1. Identify the target column (the label/output column; typically the last column,
       or any column named "target","label","class","output","y", etc.).
    2. Determine if this is a classification or regression task.
    3. Recommend preprocessing steps needed (from: ["imputation", "label_encoding",
       "one_hot_encoding", "standardization", "minmax_scaling", "robust_scaling",
       "smote", "oversample", "undersample"]).
    4. Set "needs_feature_selection" to true ONLY if num_features > {config.FEATURE_SELECTION_THRESHOLD}.
       If already low, set to false.
    5. If feature selection is needed, pick one of: "selectkbest", "rfe", "pca".
    6. Recommend 3 ML models best suited to this dataset (classification models:
       "LogisticRegression","RandomForest","XGBoost","LightGBM","SVM","KNN",
       "GradientBoosting","DecisionTree"; regression models: "LinearRegression",
       "Ridge","Lasso","RandomForestRegressor","XGBoostRegressor","SVR",
       "GradientBoostingRegressor").
    7. Choose scaler: "standard" for normal data, "robust" if outliers present,
       "minmax" for neural-net-style or [0,1] bounded data.
    8. For classification: set handle_imbalance=true and pick method if class imbalance
       ratio > 2:1; otherwise false.

    Return ONLY this JSON (fill in actual values):
    {{
        "task_type": "classification",
        "target_column": "<column_name>",
        "recommended_preprocessing": ["imputation", "label_encoding", "standardization"],
        "needs_feature_selection": false,
        "feature_selection_method": null,
        "recommended_models": ["RandomForest", "XGBoost", "LogisticRegression"],
        "scaler_type": "standard",
        "handle_imbalance": false,
        "imbalance_method": null,
        "reasoning": "Brief explanation of choices"
    }}
    """).strip()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )

    raw_text = response.choices[0].message.content
    result = _extract_json(raw_text)

    # ALWAYS use the last column as the target — predictable & unambiguous
    target_col = df.columns[-1]
    result["target_column"] = target_col

    # Auto-correct task_type based on actual target column dtype
    target_series = df[target_col]
    if target_series.dtype == object or str(target_series.dtype) == "category":
        # Text label -> must be classification
        result["task_type"] = "classification"
    elif target_series.nunique() <= 20 and str(target_series.dtype).startswith("int"):
        # Small set of integers -> classification (e.g. 0/1 or 1-5 ratings)
        result["task_type"] = "classification"
    # else keep what AI decided (float target with many values -> regression)

    # If we corrected to classification, sanitize model list
    CLASSIFICATION_MODEL_NAMES = {
        "LogisticRegression", "RandomForest", "GradientBoosting",
        "DecisionTree", "SVM", "KNN", "XGBoost", "LightGBM",
    }
    REGRESSION_MODEL_NAMES = {
        "LinearRegression", "Ridge", "Lasso", "RandomForestRegressor",
        "GradientBoostingRegressor", "SVR", "XGBoostRegressor",
        "LGBMRegressor", "DecisionTreeRegressor",
    }
    if result["task_type"] == "classification":
        result["recommended_models"] = [
            m for m in result.get("recommended_models", [])
            if m in CLASSIFICATION_MODEL_NAMES
        ] or ["RandomForest", "XGBoost", "LogisticRegression"]
    else:
        result["recommended_models"] = [
            m for m in result.get("recommended_models", [])
            if m in REGRESSION_MODEL_NAMES
        ] or ["RandomForestRegressor", "XGBoostRegressor", "Ridge"]

    # Enforce feature selection threshold
    num_features = df.shape[1] - 1
    if num_features <= config.FEATURE_SELECTION_THRESHOLD:
        result["needs_feature_selection"] = False
        result["feature_selection_method"] = None

    return result


# ─────────────────────────── auto-retry analyzer ─────────────────────────────

def retry_analyze_dataset(
    df: pd.DataFrame,
    previous_decisions: dict,
    previous_cv_score: float,
) -> dict:
    """
    Called when the best CV score is below the performance threshold.
    Re-prompts Groq with context about the poor previous attempt and
    asks it to suggest an improved pipeline strategy.
    Returns a new decisions dict (same format as analyze_dataset).
    """
    client = Groq(api_key=config.GROQ_API_KEY)
    summary = _build_dataset_summary(df)

    prev_models     = previous_decisions.get("recommended_models", [])
    prev_scaler     = previous_decisions.get("scaler_type", "standard")
    prev_imbalance  = previous_decisions.get("handle_imbalance", False)
    prev_fs         = previous_decisions.get("feature_selection_method", "none")

    prompt = textwrap.dedent(f"""
    You are an expert data scientist debugging a poor ML pipeline.

    The previous pipeline attempt achieved only CV score = {previous_cv_score:.4f}, which is below the acceptable threshold.

    Previous decisions that FAILED:
    - Models tried: {prev_models}
    - Scaler: {prev_scaler}
    - Handle imbalance: {prev_imbalance}
    - Feature selection: {prev_fs}

    Dataset summary (same dataset):
    {json.dumps(summary, indent=2)}

    Your task: suggest a DIFFERENT strategy.
    - Try different models NOT in the failed list if possible
    - Consider switching scaler (e.g., try robust or minmax)
    - Reconsider whether to use SMOTE or other balancing
    - Keep the same target column (last column)
    - Return ONLY valid JSON, no explanation.

    Required JSON keys:
    {{
      "task_type": "classification" | "regression",
      "target_column": "...",
      "recommended_preprocessing": [...],
      "needs_feature_selection": true | false,
      "feature_selection_method": "selectkbest" | "rfe" | "pca" | null,
      "recommended_models": [...],
      "reasoning": "...",
      "scaler_type": "standard" | "minmax" | "robust",
      "handle_imbalance": true | false,
      "imbalance_method": "smote" | "oversample" | "undersample" | null
    }}
    """)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )

    result = _extract_json(response.choices[0].message.content)

    # Apply the same auto-corrections as analyze_dataset
    target_col = df.columns[-1]
    result["target_column"] = target_col

    target_series = df[target_col]
    if target_series.dtype == object or str(target_series.dtype) == "category":
        result["task_type"] = "classification"
    elif target_series.nunique() <= 20 and str(target_series.dtype).startswith("int"):
        result["task_type"] = "classification"

    CLASSIFICATION_MODEL_NAMES = {
        "LogisticRegression", "RandomForest", "GradientBoosting",
        "DecisionTree", "SVM", "KNN", "XGBoost", "LightGBM",
    }
    REGRESSION_MODEL_NAMES = {
        "LinearRegression", "Ridge", "Lasso", "RandomForestRegressor",
        "GradientBoostingRegressor", "SVR", "XGBoostRegressor",
        "LGBMRegressor", "DecisionTreeRegressor",
    }
    if result["task_type"] == "classification":
        result["recommended_models"] = [
            m for m in result.get("recommended_models", [])
            if m in CLASSIFICATION_MODEL_NAMES
        ] or ["GradientBoosting", "RandomForest", "XGBoost"]
    else:
        result["recommended_models"] = [
            m for m in result.get("recommended_models", [])
            if m in REGRESSION_MODEL_NAMES
        ] or ["GradientBoostingRegressor", "RandomForestRegressor", "Ridge"]

    num_features = df.shape[1] - 1
    if num_features <= config.FEATURE_SELECTION_THRESHOLD:
        result["needs_feature_selection"] = False
        result["feature_selection_method"] = None

    return result

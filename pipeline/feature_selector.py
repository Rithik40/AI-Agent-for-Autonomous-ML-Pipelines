"""
Feature Selector — applied ONLY when the number of features exceeds the threshold.
Methods: SelectKBest (chi2 / f_classif / f_regression), RFE, PCA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE, SelectKBest, f_classif, f_regression, mutual_info_classif,
    mutual_info_regression,
)

import config


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    task_type: str,
    k: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply feature selection.

    Args:
        X          : feature matrix
        y          : target series
        method     : "selectkbest" | "rfe" | "pca"
        task_type  : "classification" | "regression"
        k          : number of features to keep (auto-set if None)

    Returns:
        X_selected : dataframe with selected features
        fs_meta    : metadata (selector object, selected feature names, etc.)
    """
    n_features = X.shape[1]

    # Auto-determine k: keep sqrt(n_features) but at least 5, at most 30
    if k is None:
        k = max(5, min(30, int(np.sqrt(n_features))))

    k = min(k, n_features)  # Never exceed actual feature count

    fs_meta: dict = {
        "method": method,
        "n_original": n_features,
        "n_selected": k,
        "selected_features": [],
    }

    if method == "selectkbest":
        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected_arr = selector.fit_transform(X, y)
        selected_mask = selector.get_support()
        selected_cols = X.columns[selected_mask].tolist()
        X_selected = pd.DataFrame(X_selected_arr, columns=selected_cols)
        fs_meta["scores"] = dict(zip(X.columns.tolist(), selector.scores_.tolist()))
        fs_meta["selector"] = selector

    elif method == "rfe":
        if task_type == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
        X_selected_arr = selector.fit_transform(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        X_selected = pd.DataFrame(X_selected_arr, columns=selected_cols)
        fs_meta["ranking"] = dict(zip(X.columns.tolist(), selector.ranking_.tolist()))
        fs_meta["selector"] = selector

    elif method == "pca":
        pca = PCA(n_components=k, random_state=42)
        X_selected_arr = pca.fit_transform(X)
        selected_cols = [f"PC{i+1}" for i in range(k)]
        X_selected = pd.DataFrame(X_selected_arr, columns=selected_cols)
        fs_meta["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        fs_meta["cumulative_variance"] = float(np.sum(pca.explained_variance_ratio_))
        fs_meta["selector"] = pca

    else:
        # fallback: no selection
        X_selected = X.copy()
        selected_cols = X.columns.tolist()

    fs_meta["selected_features"] = selected_cols
    return X_selected, fs_meta

"""
Model Trainer — exposes ALL available models. Accepts a user-chosen subset.
Returns full CV results and the trained best model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ─────────────────────── full model registries ───────────────────────────────

CLASSIFICATION_MODELS: dict[str, callable] = {
    "LogisticRegression":  lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest":        lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "GradientBoosting":    lambda: GradientBoostingClassifier(n_estimators=200, random_state=42),
    "DecisionTree":        lambda: DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "SVM":                 lambda: SVC(probability=True, class_weight='balanced', random_state=42),
    "KNN":                 lambda: KNeighborsClassifier(n_neighbors=5),
}

REGRESSION_MODELS: dict[str, callable] = {
    "LinearRegression":          lambda: LinearRegression(),
    "Ridge":                     lambda: Ridge(alpha=1.0),
    "Lasso":                     lambda: Lasso(alpha=0.01, max_iter=5000),
    "RandomForestRegressor":     lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoostingRegressor": lambda: GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR":                       lambda: SVR(),
    "DecisionTreeRegressor":     lambda: DecisionTreeRegressor(random_state=42),
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBoost"] = lambda: XGBClassifier(
        n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0
    )
    REGRESSION_MODELS["XGBoostRegressor"] = lambda: XGBRegressor(
        n_estimators=200, random_state=42, verbosity=0
    )

if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODELS["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, verbose=-1
    )
    REGRESSION_MODELS["LGBMRegressor"] = lambda: LGBMRegressor(
        n_estimators=200, random_state=42, verbose=-1
    )


def get_all_models_for_task(task_type: str) -> list[str]:
    """Return all available model names for the given task type."""
    if task_type == "classification":
        return list(CLASSIFICATION_MODELS.keys())
    return list(REGRESSION_MODELS.keys())


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    decisions: dict,
    selected_models: list[str] | None = None,
) -> dict:
    """
    Train the user-selected models (or all available if None).
    Returns best model + CV results.
    """
    task_type: str = decisions.get("task_type", "classification")
    model_pool = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS

    # Build the set to train: respect user selection
    if selected_models:
        models_to_try = {n: model_pool[n] for n in selected_models if n in model_pool}
    else:
        # Default: AI recommended + all available (all models)
        models_to_try = dict(model_pool)

    if not models_to_try:
        models_to_try = dict(model_pool)   # fallback

    # Cross-validation setup
    from sklearn.model_selection import train_test_split
    X_arr = X.values
    y_arr = y.values
    
    # 80/20 train/test split here instead of evaluator, so test set is fully held out from CV
    stratify = y_arr if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=stratify
    )

    if task_type == "classification":
        n_classes = int(y.nunique())
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        n_classes = None
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
        scoring = "r2"

    cv_results: dict = {}
    best_name: str = ""
    best_score: float = -np.inf
    trained_models = {}

    for name, factory in models_to_try.items():
        try:
            model = factory()
            # strictly CV on X_train to prevent info leakage
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))
            cv_results[name] = {"mean": round(mean_score, 4), "std": round(std_score, 4)}
            if mean_score > best_score:
                best_score = mean_score
                best_name  = name
            
            # Train model on full training set
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            cv_results[name] = {"mean": 0.0, "std": 0.0, "error": str(e)}

    # Fallback if best_name is empty (all failed)
    if not best_name and trained_models:
        best_name = list(trained_models.keys())[0]

    result = {
        "best_model_name":  best_name,
        "best_model":       trained_models.get(best_name),
        "trained_models":   trained_models, # Pass all models back
        "cv_results":       cv_results,
        "task_type":        task_type,
        "feature_names":    list(X.columns),
        "best_cv_score":    best_score,
        "X_train":          X_train,
        "y_train":          y_train,
        "X_test":           X_test,
        "y_test":           y_test,
    }
    if n_classes is not None:
        result["n_classes"] = n_classes

    return result


# ─────────────────────── hyperparameter tuning ───────────────────────────────

PARAM_GRIDS = {
    # Classification
    "LogisticRegression": {
        "C": stats.loguniform(1e-4, 1e2),
        "solver": ["lbfgs", "liblinear"],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6],
    },
    "DecisionTree": {
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10, 20],
    },
    "SVM": {
        "C": stats.loguniform(1e-3, 1e2),
        "gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 5)),
        "kernel": ["linear", "rbf"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 50, 100],
    },
    # Regression
    "LinearRegression": {}, # Typically no hyperparameters to tune
    "Ridge": {
        "alpha": stats.loguniform(1e-3, 1e2)
    },
    "Lasso": {
        "alpha": stats.loguniform(1e-4, 1e1)
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    },
    "SVR": {
        "C": stats.loguniform(1e-3, 1e2),
        "epsilon": [0.01, 0.1, 0.2, 0.5],
        "kernel": ["linear", "rbf"]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoostRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
    },
    "LGBMRegressor": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 50, 100],
    }
}


def tune_model(train_result: dict, decisions: dict) -> dict:
    """
    Takes the baseline best model and performs RandomizedSearchCV over predefined param grids.
    Returns the updated train_result dictionary with the `tuned_model` added.
    """
    model_name = train_result["best_model_name"]
    model = train_result["best_model"]
    X_train = train_result["X_train"]
    y_train = train_result["y_train"]
    task_type = train_result["task_type"]

    param_grid = PARAM_GRIDS.get(model_name, {})
    
    # If there are no tunable parameters, just return the dict untouched
    if not param_grid:
        train_result["tuned_model"] = model
        train_result["tuned_cv_score"] = train_result["best_cv_score"]
        return train_result
        
    n_iter = 20 # reasonable default for quick Telegram bot iteration
    
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scoring = "r2"

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
    )
    
    search.fit(X_train, y_train)
    
    train_result["tuned_model"] = search.best_estimator_
    train_result["tuned_cv_score"] = search.best_score_
    
    return train_result

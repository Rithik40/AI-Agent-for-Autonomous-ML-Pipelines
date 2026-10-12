"""
Evaluator — computes comprehensive metrics for classification and regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def evaluate(
    train_result: dict,
) -> dict:
    """
    Evaluate ALL trained models on the test split, return full metrics.
    """
    task_type = train_result["task_type"]
    n_classes = train_result.get("n_classes", 2)
    trained_models = dict(train_result["trained_models"])
    X_test = train_result["X_test"]
    y_test = train_result["y_test"]
    best_name = train_result["best_model_name"]
    cv_results = train_result.get("cv_results", {})
    
    # If tuning occurred, explicitly label baseline vs tuned models
    if "tuned_model" in train_result:
        baseline_name = f"{best_name} (Baseline)"
        trained_models[baseline_name] = trained_models.pop(best_name)
        if best_name in cv_results:
            cv_results[baseline_name] = cv_results.pop(best_name)
        
        tuned_name = f"{best_name} (Tuned)"
        trained_models[tuned_name] = train_result["tuned_model"]
        cv_results[tuned_name] = {"mean": round(float(train_result["tuned_cv_score"]), 4), "std": 0.0}
        
        best_name = tuned_name

    all_metrics = {}
    best_metrics = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        model_metrics: dict = {}

        if task_type == "classification":
            avg = "binary" if n_classes == 2 else "macro"
            model_metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            model_metrics["precision"] = round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            model_metrics["recall"] = round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4)
            model_metrics["f1"] = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)

            if name == best_name:
                model_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)
                model_metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

            # ROC-AUC
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    if n_classes == 2:
                        model_metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
                        model_metrics["y_proba"] = y_proba[:, 1]
                    else:
                        model_metrics["roc_auc"] = round(float(roc_auc_score(
                            y_test, y_proba, multi_class="ovr", average="macro"
                        )), 4)
                        model_metrics["y_proba"] = y_proba
                elif hasattr(model, "decision_function"):
                    y_scores = model.decision_function(X_test)
                    if n_classes == 2:
                        model_metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_scores)), 4)
                        model_metrics["y_proba"] = y_scores
                    else:
                        model_metrics["roc_auc"] = None
                        model_metrics["y_proba"] = None
                else:
                    model_metrics["roc_auc"] = None
                    model_metrics["y_proba"] = None
            except Exception:
                model_metrics["roc_auc"] = None
                model_metrics["y_proba"] = None

        else:  # regression
            residuals = y_test - y_pred
            model_metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            model_metrics["mse"] = round(float(mean_squared_error(y_test, y_pred)), 4)
            model_metrics["rmse"] = round(float(np.sqrt(model_metrics["mse"])), 4)
            model_metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)
            model_metrics["residuals"] = residuals
            model_metrics["y_pred"] = y_pred

        all_metrics[name] = model_metrics
        if name == best_name:
            best_metrics = model_metrics

    metrics = {
        "task_type": task_type,
        "n_test_samples": len(y_test),
        "n_train_samples": len(train_result["y_train"]),
        "X_test": X_test,
        "X_train": train_result["X_train"],
        "y_test": y_test,
        "y_pred": best_metrics.get("y_pred"), # Not available in best_metrics, need to add if needed
        "y_train": train_result["y_train"],
        "cv_results": cv_results,
        "all_metrics": all_metrics,
    }
    # Provide backward compatibility for plotter
    if task_type == "classification":
        metrics["confusion_matrix"] = best_metrics.get("confusion_matrix")
        metrics["classification_report"] = best_metrics.get("classification_report")
        metrics["accuracy"] = best_metrics.get("accuracy", 0)
        metrics["precision"] = best_metrics.get("precision", 0)
        metrics["recall"] = best_metrics.get("recall", 0)
        metrics["f1"] = best_metrics.get("f1", 0)
        metrics["roc_auc"] = best_metrics.get("roc_auc")
        metrics["y_proba"] = best_metrics.get("y_proba")
    else:
        metrics["mae"] = best_metrics.get("mae", 0)
        metrics["mse"] = best_metrics.get("mse", 0)
        metrics["rmse"] = best_metrics.get("rmse", 0)
        metrics["r2"] = best_metrics.get("r2", 0)
        metrics["residuals"] = best_metrics.get("residuals")
        best_pred = trained_models[best_name].predict(X_test)
        metrics["y_pred"] = best_pred

    return metrics

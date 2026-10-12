"""
Plotter — white-themed plots. Values printed on top of ALL bar charts.
Plots: class distribution, confusion matrix, ROC curve, feature importance,
       model comparison (CV), correlation heatmap, residuals, predicted vs actual.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ──────────── white/light style ──────────────────────────────────────────────
ACCENT   = "#1a56db"
ACCENT2  = "#e94560"
BG       = "white"
AX_BG    = "#f8fafc"
TEXT_COL = "#1a1a2e"
GRID_COL = "#e2e8f0"
PALETTE  = [
    "#1a56db", "#e94560", "#15803d", "#b45309",
    "#7c3aed", "#0284c7", "#be123c", "#065f46",
    "#92400e", "#4f46e5",
]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   AX_BG,
    "axes.edgecolor":   "#cbd5e1",
    "axes.labelcolor":  TEXT_COL,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "text.color":       TEXT_COL,
    "grid.color":       GRID_COL,
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titlecolor":  ACCENT,
    "axes.titleweight": "bold",
})
sns.set_theme(style="whitegrid", palette=PALETTE)


def _savefig(fig: plt.Figure, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def _bar_values(ax, bars, fmt="{:.4f}", pad_factor=0.005):
    """Print value on top of every bar in `bars`."""
    y_max = max((b.get_height() for b in bars), default=1)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * pad_factor,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=8.5, color=TEXT_COL, fontweight="bold",
        )


def _hbar_values(ax, bars, fmt="{:.4f}", pad_factor=0.003):
    """Print value to the right of every horizontal bar."""
    x_max = max((b.get_width() for b in bars), default=1)
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + x_max * pad_factor,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha="left", va="center",
            fontsize=8, color=TEXT_COL, fontweight="bold",
        )


# ─────────────────────── individual plots ────────────────────────────────────

def plot_class_distribution(
    y_before: pd.Series,
    y_after: Optional[pd.Series],
    out_dir: str,
    target_col: str,
) -> str:
    path = os.path.join(out_dir, "class_distribution.png")

    if y_after is not None and not y_before.equals(y_after):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        datasets = [(y_before, "Before Balancing"), (y_after, "After Balancing")]
        for ax, (data, title) in zip(axes, datasets):
            counts = data.value_counts().sort_index()
            bars = ax.bar([str(x) for x in counts.index], counts.values,
                          color=PALETTE[:len(counts)], edgecolor="white", linewidth=1.2)
            _bar_values(ax, bars, fmt="{:.0f}")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(target_col, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
        fig.suptitle("Class Distribution", fontsize=14, color=ACCENT, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = y_before.value_counts().sort_index()
        bars = ax.bar([str(x) for x in counts.index], counts.values,
                      color=PALETTE[:len(counts)], edgecolor="white", linewidth=1.2)
        _bar_values(ax, bars, fmt="{:.0f}")
        ax.set_title("Class Distribution", fontsize=13)
        ax.set_xlabel(target_col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)

    return _savefig(fig, path)


def plot_confusion_matrix(cm: np.ndarray, out_dir: str,
                           class_labels: Optional[list] = None) -> str:
    path = os.path.join(out_dir, "confusion_matrix.png")
    n = cm.shape[0]
    labels = class_labels or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#e2e8f0",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    return _savefig(fig, path)


def plot_roc_curve(y_test: np.ndarray, all_metrics: dict,
                   n_classes: int, out_dir: str, best_model_name: str) -> str:
    path = os.path.join(out_dir, "roc_curve.png")
    fig, ax = plt.subplots(figsize=(9, 6))

    if n_classes == 2:
        for i, (name, m_res) in enumerate(all_metrics.items()):
            y_proba = m_res.get("y_proba")
            if y_proba is None: continue
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            color = PALETTE[i % len(PALETTE)]
            lw = 3 if name == best_model_name else 1.5
            alpha = 1.0 if name == best_model_name else 0.7
            ax.plot(fpr, tpr, color=color, linewidth=lw, alpha=alpha,
                    label=f"{name} (AUC = {roc_auc:.4f})")
    else:
        # Multiclass: Plot only the best model's classes to avoid extreme clutter
        from sklearn.preprocessing import label_binarize
        classes = list(range(n_classes))
        y_bin = label_binarize(y_test, classes=classes)
        best_proba = all_metrics.get(best_model_name, {}).get("y_proba")
        if best_proba is not None:
            for i, color in zip(range(n_classes), PALETTE):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], best_proba[:, i])
                roc_auc_i = auc(fpr_i, tpr_i)
                ax.plot(fpr_i, tpr_i, color=color, linewidth=2,
                        label=f"Class {i} (AUC = {roc_auc_i:.3f})")
        ax.set_title(f"ROC Curve (Best Model: {best_model_name})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    if n_classes == 2:
        ax.set_title("ROC Curve (All Models on Test Set)")
    ax.legend(loc="lower right", fontsize=9)
    return _savefig(fig, path)


def plot_feature_importance(model, feature_names: list[str],
                             out_dir: str, top_n: int = 20) -> Optional[str]:
    path = os.path.join(out_dir, "feature_importance.png")

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None:
        return None

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, len(top_features) * 0.38)))
    # Cycle through PALETTE if top_n > len(PALETTE)
    colors_list = [PALETTE[i % len(PALETTE)] for i in range(len(top_features))]
    
    bars = ax.barh(range(len(top_features)), top_importances[::-1],
                   color=list(reversed(colors_list)), edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(f"Top {len(top_features)} Feature Importances")
    _hbar_values(ax, bars, fmt="{:.4f}")
    return _savefig(fig, path)


def plot_model_comparison(metrics: dict, out_dir: str, task_type: str) -> str:
    path = os.path.join(out_dir, "model_comparison.png")
    cv_results = metrics.get("cv_results", {})
    all_metrics = metrics.get("all_metrics", {})
    
    names = list(cv_results.keys())
    cv_means = [cv_results[n]["mean"] for n in names]
    cv_stds  = [cv_results[n]["std"]  for n in names]
    
    if task_type == "classification":
        test_scores = [all_metrics.get(n, {}).get("accuracy", 0) for n in names]
        metric_name = "Accuracy"
    else:
        test_scores = [all_metrics.get(n, {}).get("r2", 0) for n in names]
        metric_name = "R² Score"

    # Sort by test score descending
    sorted_data = sorted(zip(names, cv_means, cv_stds, test_scores), key=lambda x: x[3], reverse=True)
    names_s, cv_means_s, cv_stds_s, test_scores_s = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(max(9, len(names_s) * 1.6), 6.5))
    x = np.arange(len(names_s))
    width = 0.35

    bars1 = ax.bar(x - width/2, cv_means_s, width, yerr=cv_stds_s, capsize=4,
                   color=PALETTE[0], edgecolor="white", label="CV Mean")
    bars2 = ax.bar(x + width/2, test_scores_s, width,
                   color=PALETTE[1], edgecolor="white", label="Test Set")

    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"Model Comparison: CV vs Test Set ({metric_name})")
    ax.legend()
    
    _bar_values(ax, bars1, fmt="{:.3f}")
    _bar_values(ax, bars2, fmt="{:.3f}")
    
    ax.set_ylim(0, max(max(cv_means_s), max(test_scores_s)) * 1.15)
    return _savefig(fig, path)


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, out_dir: str) -> str:
    path = os.path.join(out_dir, "residuals_plot.png")
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.6, color=ACCENT, s=30, edgecolor="none")
    ax1.axhline(y=0, color=ACCENT2, linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Predicted Values", fontsize=10)
    ax1.set_ylabel("Residuals", fontsize=10)
    ax1.set_title("Residuals vs Predicted (Best Model)")

    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color=ACCENT, edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Residual Value", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title("Residuals Distribution")

    fig.suptitle("Regression Residual Analysis", fontsize=14, color=ACCENT, fontweight="bold")
    return _savefig(fig, path)


def plot_predicted_vs_actual(y_test: np.ndarray, all_metrics: dict, out_dir: str, best_model_name: str) -> str:
    path = os.path.join(out_dir, "predicted_vs_actual.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for i, (name, m_res) in enumerate(all_metrics.items()):
        y_pred = m_res.get("y_pred", y_test - m_res.get("residuals", np.zeros_like(y_test)))
        alpha = 0.8 if name == best_model_name else 0.4
        s = 40 if name == best_model_name else 20
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(y_test, y_pred, alpha=alpha, color=color, s=s, edgecolor="none", label=name)
        
    mn = float(y_test.min())
    mx = float(y_test.max())
    # adjust max based on predictions too
    all_preds = [y_test - m.get("residuals", np.zeros_like(y_test)) for m in all_metrics.values() if "residuals" in m]
    if all_preds:
        mx = max(mx, max([p.max() for p in all_preds]))
        mn = min(mn, min([p.min() for p in all_preds]))
        
    ax.plot([mn, mx], [mn, mx], color="black", linewidth=2, linestyle="--", label="Perfect Fit")
    ax.set_xlabel("Actual Values", fontsize=10)
    ax.set_ylabel("Predicted Values", fontsize=10)
    ax.set_title("Predicted vs Actual (All Models on Test Set)")
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    return _savefig(fig, path)


def plot_correlation_heatmap(X_raw: pd.DataFrame, out_dir: str, max_cols: int = 20) -> Optional[str]:
    """Pearson correlation heatmap for numeric features."""
    path = os.path.join(out_dir, "correlation_heatmap.png")
    num_df = X_raw.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    # Cap columns to avoid unreadable plots
    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]

    corr = num_df.corr()
    size = max(8, min(num_df.shape[1], max_cols) * 0.65)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        corr, ax=ax, annot=(num_df.shape[1] <= 15),
        fmt=".2f", cmap="coolwarm", center=0,
        linewidths=0.4, linecolor="#e2e8f0",
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Feature Correlation Heatmap")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    return _savefig(fig, path)


# ─────────────────────── main orchestrator ───────────────────────────────────

def generate_all_plots(
    metrics: dict,
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    y_balanced: Optional[pd.Series],
    model,
    feature_names: list[str],
    decisions: dict,
    out_dir: str,
    best_model_name: str,
) -> list[str]:
    plots: list[str] = []
    task_type  = decisions.get("task_type", "classification")
    target_col = decisions.get("target_column", "target")

    if task_type == "classification":
        try:
            plots.append(plot_class_distribution(y_raw, y_balanced, out_dir, target_col))
        except Exception as e:
            print(f"[Plot] class_distribution: {e}")

    if task_type == "classification" and "confusion_matrix" in metrics:
        try:
            plots.append(plot_confusion_matrix(metrics["confusion_matrix"], out_dir))
        except Exception as e:
            print(f"[Plot] confusion_matrix: {e}")

    if task_type == "classification" and metrics.get("all_metrics"):
        try:
            n_classes = max(2, int(np.max(metrics["y_test"])) + 1)
            plots.append(plot_roc_curve(
                metrics["y_test"], metrics["all_metrics"], n_classes, out_dir, best_model_name
            ))
        except Exception as e:
            print(f"[Plot] roc_curve: {e}")

    if feature_names:
        try:
            p = plot_feature_importance(model, feature_names, out_dir)
            if p:
                plots.append(p)
        except Exception as e:
            print(f"[Plot] feature_importance: {e}")

    if metrics:
        try:
            plots.append(plot_model_comparison(metrics, out_dir, task_type))
        except Exception as e:
            print(f"[Plot] model_comparison: {e}")

    if isinstance(X_raw, pd.DataFrame) and X_raw.shape[1] >= 2:
        try:
            p = plot_correlation_heatmap(X_raw, out_dir)
            if p:
                plots.append(p)
        except Exception as e:
            print(f"[Plot] correlation_heatmap: {e}")

    if task_type == "regression":
        try:
            plots.append(plot_residuals(metrics["y_test"], metrics["y_pred"], out_dir))
        except Exception as e:
            print(f"[Plot] residuals: {e}")
        try:
            plots.append(plot_predicted_vs_actual(
                metrics["y_test"], metrics["all_metrics"], out_dir, best_model_name
            ))
        except Exception as e:
            print(f"[Plot] predicted_vs_actual: {e}")

    return plots

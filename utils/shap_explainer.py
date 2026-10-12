"""
shap_explainer.py — Generates SHAP-based natural language explanations for single predictions.

Works with tree-based models (XGBoost, LightGBM, RandomForest, GradientBoosting)
and falls back to KernelExplainer for linear/SVM models.
Returns a text summary + optionally a plot path.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


def explain_prediction(
    model,
    sample_df: pd.DataFrame,
    feature_names: list[str],
    output_dir: str | None = None,
    top_n: int = 10,
    display_df: pd.DataFrame | None = None,
) -> tuple[str, str | None]:
    """
    Generate a SHAP explanation for a single sample prediction.

    Returns:
        (text_explanation, plot_path_or_None)
    """
    try:
        import shap
    except ImportError:
        return ("_(SHAP not installed — run `pip install shap` for explanations.)_", None)

    # ── Pick the right explainer ──────────────────────────────────────────────
    model_type = type(model).__name__

    try:
        tree_types = (
            "XGBClassifier", "XGBRegressor",
            "LGBMClassifier", "LGBMRegressor",
            "RandomForestClassifier", "RandomForestRegressor",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor",
        )

        if model_type in tree_types:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_df)
        else:
            # Fallback for LR, SVM, KNN etc.
            # Since data is scaled (mean=0), zero array is a good proxy for the training mean
            background_df = pd.DataFrame(np.zeros_like(sample_df.values), columns=feature_names)
            masker = shap.maskers.Independent(background_df, max_samples=50)
            explainer = shap.Explainer(model.predict, masker)
            shap_values = explainer(sample_df).values

        # For binary / multiclass: take first class or the positive class
        if isinstance(shap_values, list):
            # Multiclass: pick the predicted class
            pred_class = int(model.predict(sample_df)[0])
            sv = shap_values[pred_class][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            pred_class = int(model.predict(sample_df)[0])
            sv = shap_values[0, :, pred_class]
        else:
            sv = shap_values[0] if shap_values.ndim == 2 else shap_values

        # ── Build text summary ────────────────────────────────────────────────
        feat_importance = sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        if display_df is None:
            display_df = sample_df

        lines = []
        for feat, val in feat_importance:
            arrow = "⬆️ pushes prediction UP" if val > 0 else "⬇️ pushes prediction DOWN"
            
            # format the display value cleanly
            disp_val = display_df.at[0, feat]
            if isinstance(disp_val, float):
                disp_str = f"{disp_val:.4g}"
            else:
                disp_str = f"{disp_val}"
                
            lines.append(f"  • `{feat}` ({disp_str}): {arrow} by `{abs(val):.4f}`")

        actual_top_n = min(top_n, len(feat_importance))
        text = (
            f"🔍 *Why this prediction?* (Top {actual_top_n} drivers)\n\n"
            + "\n".join(lines)
            + "\n\n_Positive = pushed toward predicted class, Negative = pushed away._"
        )

        # ── Optional: save waterfall plot ─────────────────────────────────────
        plot_path = None
        if output_dir:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                explanation = shap.Explanation(
                    values=sv,
                    base_values=float(explainer.expected_value[pred_class])
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else float(explainer.expected_value),
                    data=sample_df.values[0],
                    feature_names=feature_names,
                )
                plt.figure(figsize=(9, 5))
                shap.plots.waterfall(explanation, show=False)
                plt.tight_layout()
                plot_path = os.path.join(output_dir, "shap_explanation.png")
                plt.savefig(plot_path, dpi=130, bbox_inches="tight")
                plt.close("all")
            except Exception:
                plot_path = None

        return text, plot_path

    except Exception as e:
        return (f"_(SHAP explanation unavailable: {e})_", None)

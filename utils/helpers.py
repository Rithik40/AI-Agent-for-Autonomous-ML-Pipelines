"""
Utility helpers: temp file management, text formatting, persistent output dirs.
"""
from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime

import config


# Where persistent reports go
OUTPUT_ROOT = r"D:\Users\Rithik\Desktop\Output"


def make_session_dir() -> str:
    """Create a unique temp directory for this pipeline run."""
    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(config.TEMP_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def make_output_dir(filename: str) -> str:
    """Create/return a persistent output folder named after the dataset."""
    # Strip extension and sanitize
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in dataset_name).strip()
    output_dir = os.path.join(OUTPUT_ROOT, safe_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def cleanup_session(session_dir: str) -> None:
    """Remove the session temp directory."""
    try:
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
    except Exception:
        pass


def format_metrics_message(
    decisions: dict,
    train_result: dict,
    metrics: dict,
    preprocess_meta: dict,
    fs_meta: dict | None,
) -> str:
    """Format a concise Telegram text message summarising results."""
    task_type = decisions.get("task_type", "classification")
    best_model_base = train_result.get("best_model_name", "Unknown")
    is_tuned = "tuned_model" in train_result
    best_model = f"{best_model_base} (Tuned)" if is_tuned else best_model_base
    
    target = decisions.get("target_column", "?")
    orig_shape = preprocess_meta.get("original_shape", ("?", "?"))

    lines = [
        "─" * 34,
        f"📊 *ML Pipeline Results*",
        f"─" + "─" * 33,
        f"*Task Type:* {task_type.capitalize()}",
        f"*Target Column:* `{target}`",
        f"*Dataset:* {orig_shape[0]} rows × {orig_shape[1]} cols",
        f"*Best Model:* `{best_model}`",
        "",
        "━━━ *Preprocessing* ━━━",
        f"• Scaler: {decisions.get('scaler_type', 'N/A').capitalize()}",
        f"• Class Balancing: {'Yes (' + preprocess_meta.get('imbalance_method_used', '') + ')' if preprocess_meta.get('imbalance_handled') else 'No'}",
        f"• Dropped Cols: {len(preprocess_meta.get('dropped_cols', []))}",
        f"• Final Features: {preprocess_meta.get('final_shape', ('?', '?'))[1]}",
    ]

    if fs_meta:
        lines += [
            "",
            "━━━ *Feature Selection* ━━━",
            f"• Method: {fs_meta.get('method', 'N/A').upper()}",
            f"• Features: {fs_meta.get('n_original', '?')} → {fs_meta.get('n_selected', '?')}",
        ]

    lines += [
        "",
        f"━━━ *Evaluation Metrics* (Test Set) ━━━",
    ]

    if is_tuned:
        baseline_name = f"{best_model_base} (Baseline)"
        tuned_name = f"{best_model_base} (Tuned)"
        base_m = metrics.get("all_metrics", {}).get(baseline_name, {})
        tuned_m = metrics.get("all_metrics", {}).get(tuned_name, {})
        
        lines += ["*(Tuned vs Baseline)*"]
        
        if task_type == "classification":
            lines += [
                f"• Accuracy:  `{tuned_m.get('accuracy', 'N/A'):.4f}` (was `{base_m.get('accuracy', 'N/A'):.4f}`)",
                f"• Precision: `{tuned_m.get('precision', 'N/A'):.4f}` (was `{base_m.get('precision', 'N/A'):.4f}`)",
                f"• Recall:    `{tuned_m.get('recall', 'N/A'):.4f}` (was `{base_m.get('recall', 'N/A'):.4f}`)",
                f"• F1 Score:  `{tuned_m.get('f1', 'N/A'):.4f}` (was `{base_m.get('f1', 'N/A'):.4f}`)",
                f"• ROC-AUC:   `{tuned_m.get('roc_auc', 'N/A')}` (was `{base_m.get('roc_auc', 'N/A')}`)",
            ]
        else:
            lines += [
                f"• MAE:  `{tuned_m.get('mae', 'N/A'):.4f}` (was `{base_m.get('mae', 'N/A'):.4f}`)",
                f"• MSE:  `{tuned_m.get('mse', 'N/A'):.4f}` (was `{base_m.get('mse', 'N/A'):.4f}`)",
                f"• RMSE: `{tuned_m.get('rmse', 'N/A'):.4f}` (was `{base_m.get('rmse', 'N/A'):.4f}`)",
                f"• R²:   `{tuned_m.get('r2', 'N/A'):.4f}` (was `{base_m.get('r2', 'N/A'):.4f}`)",
            ]
    else:
        if task_type == "classification":
            lines += [
                f"• Accuracy:  `{metrics.get('accuracy', 'N/A'):.4f}`",
                f"• Precision: `{metrics.get('precision', 'N/A'):.4f}`",
                f"• Recall:    `{metrics.get('recall', 'N/A'):.4f}`",
                f"• F1 Score:  `{metrics.get('f1', 'N/A'):.4f}`",
                f"• ROC-AUC:   `{metrics.get('roc_auc', 'N/A')}`",
            ]
        else:
            lines += [
                f"• MAE:  `{metrics.get('mae', 'N/A'):.4f}`",
                f"• MSE:  `{metrics.get('mse', 'N/A'):.4f}`",
                f"• RMSE: `{metrics.get('rmse', 'N/A'):.4f}`",
                f"• R²:   `{metrics.get('r2', 'N/A'):.4f}`",
            ]

    # CV summary
    cv = train_result.get("cv_results", {})
    if cv:
        lines += ["", "━━━ *CV Scores (5-fold)* ━━━"]
        for name, scores in sorted(cv.items(), key=lambda x: x[1].get("mean", 0), reverse=True):
            star = " ⭐" if name == best_model else ""
            if "error" not in scores:
                lines.append(f"• {name}: `{scores['mean']:.4f}` ±{scores['std']:.4f}{star}")

    lines += [
        "",
        "📄 _Full PDF report attached above._",
        "─" * 34,
    ]

    return "\n".join(lines)

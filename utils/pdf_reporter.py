"""
PDF Report Generator — white-themed, professional multi-page PDF.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from PIL import Image as PilImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

# ──────────────── colour palette (white theme) ───────────────────────────────
C_BG        = colors.white
C_ACCENT    = colors.HexColor("#1a56db")     # strong blue
C_SECONDARY = colors.HexColor("#e8f0fe")     # light blue tint
C_HEADER_BG = colors.HexColor("#1a56db")     # table headers
C_ROW_ALT   = colors.HexColor("#f0f4ff")     # alternating row
C_TEXT      = colors.HexColor("#1a1a2e")     # near-black
C_MID       = colors.HexColor("#6b7280")     # grey
C_GREEN     = colors.HexColor("#15803d")
C_YELLOW    = colors.HexColor("#b45309")
C_RED       = colors.HexColor("#dc2626")
C_WHITE     = colors.white
C_GRID      = colors.HexColor("#cbd5e1")


def _make_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"],
            textColor=C_ACCENT, fontSize=28, spaceAfter=10,
            alignment=TA_CENTER, fontName="Helvetica-Bold",
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", textColor=C_TEXT, fontSize=13, spaceAfter=6,
            alignment=TA_CENTER, fontName="Helvetica",
        ),
        "section_header": ParagraphStyle(
            "section_header", textColor=C_ACCENT, fontSize=15, spaceAfter=8,
            spaceBefore=14, fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body", textColor=C_TEXT, fontSize=10, spaceAfter=4,
            fontName="Helvetica", leading=14,
        ),
        "mono": ParagraphStyle(
            "mono", textColor=C_TEXT, fontSize=8, spaceAfter=4,
            fontName="Courier", leading=12,
            backColor=colors.HexColor("#f8fafc"),
            borderPad=6,
        ),
        "bold_label": ParagraphStyle(
            "bold_label", textColor=C_YELLOW, fontSize=10, spaceAfter=3,
            fontName="Helvetica-Bold",
        ),
        "plot_label": ParagraphStyle(
            "plot_label", textColor=C_ACCENT, fontSize=11, spaceAfter=4,
            spaceBefore=8, fontName="Helvetica-Bold",
        ),
    }


def _table_style(highlight_row: int | None = None) -> TableStyle:
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), C_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_ROW_ALT]),
        ("TEXTCOLOR", (0, 1), (-1, -1), C_TEXT),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, C_GRID),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, 0), [C_HEADER_BG]),
    ]
    if highlight_row is not None:
        cmds += [
            ("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dcfce7")),
            ("TEXTCOLOR", (0, highlight_row), (-1, highlight_row), C_GREEN),
            ("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"),
        ]
    return TableStyle(cmds)


def _hr(story: list) -> None:
    story.append(HRFlowable(color=C_ACCENT, thickness=1.2, width="100%", spaceAfter=8))


def _section(story: list, title: str, styles: dict) -> None:
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(title, styles["section_header"]))
    _hr(story)


def _embed_plot(path: str, width_cm: float = 15) -> Optional[Image]:
    if not path or not os.path.exists(path):
        return None
    try:
        with PilImage.open(path) as img:
            w, h = img.size
        aspect = h / w
        w_pt = width_cm * cm
        return Image(path, width=w_pt, height=w_pt * aspect)
    except Exception:
        return None


# ──────────────────────── main builder ───────────────────────────────────────

def build_pdf(
    output_path: str,
    filename: str,
    decisions: dict,
    preprocess_meta: dict,
    fs_meta: Optional[dict],
    train_result: dict,
    metrics: dict,
    plot_paths: list[str],
) -> str:
    styles = _make_styles()
    story = []

    # ── COVER PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("ML Pipeline Analysis Report", styles["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Dataset: <b>{filename}</b>", styles["cover_sub"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        styles["cover_sub"],
    ))
    story.append(Spacer(1, 1.5 * cm))

    task_type = decisions.get("task_type", "N/A").upper()
    target    = decisions.get("target_column", "N/A")
    best_model_base = train_result.get("best_model_name", "N/A")
    best_model = f"{best_model_base} (Tuned)" if "tuned_model" in train_result else best_model_base
    
    orig_shape = preprocess_meta.get("original_shape", ("?", "?"))

    cover_data = [
        ["Parameter", "Value"],
        ["Task Type", task_type],
        ["Target Column", target],
        ["Dataset Shape", f"{orig_shape[0]} rows × {orig_shape[1]} cols"],
        ["Best Model", best_model],
        ["Scaler Used", decisions.get("scaler_type", "N/A").capitalize()],
        ["Class Balancing", "Yes" if preprocess_meta.get("imbalance_handled") else "No"],
        ["Feature Selection", "Yes" if fs_meta else "No"],
    ]
    tbl = Table(cover_data, colWidths=[7 * cm, 9 * cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(PageBreak())

    # ── SECTION 1: AI ANALYSIS ──────────────────────────────────────────────
    _section(story, "1 · AI Analysis & Pipeline Decisions", styles)
    reasoning = decisions.get("reasoning", "No reasoning provided.")
    story.append(Paragraph(reasoning, styles["body"]))
    story.append(Spacer(1, 0.2 * cm))
    rec_steps  = ", ".join(decisions.get("recommended_preprocessing", []))
    rec_models = ", ".join(decisions.get("recommended_models", []))
    story.append(Paragraph(f"<b>Recommended Preprocessing:</b> {rec_steps}", styles["body"]))
    story.append(Paragraph(f"<b>Models Evaluated:</b> {rec_models}", styles["body"]))

    # ── SECTION 2: PREPROCESSING ─────────────────────────────────────────────
    _section(story, "2 · Preprocessing Summary", styles)
    final_shape = preprocess_meta.get("final_shape", ("?", "?"))
    dropped     = preprocess_meta.get("dropped_cols", [])
    ohe_cols    = preprocess_meta.get("one_hot_cols", [])
    # Feature-level label-encoded columns
    le_cols     = list(preprocess_meta.get("label_encoders", {}).keys())
    # Also include target column if it was label-encoded
    if preprocess_meta.get("target_encoder") is not None:
        target_col_name = decisions.get("target_column", "target")
        le_cols_display = le_cols + [f"{target_col_name} (target)"]
    else:
        le_cols_display = le_cols

    prep_data = [
        ["Step", "Details"],
        ["Original Shape", f"{orig_shape[0]} rows × {orig_shape[1]} cols"],
        ["Final Shape (after preprocessing)", f"{final_shape[0]} rows × {final_shape[1]} cols"],
        ["Dropped Columns (>60% missing)", ", ".join(dropped) if dropped else "None"],
        ["One-Hot Encoded", ", ".join(ohe_cols) if ohe_cols else "None"],
        ["Label Encoded", ", ".join(le_cols_display) if le_cols_display else "None"],
        ["Scaler", decisions.get("scaler_type", "N/A").capitalize()],
        ["Imbalance Method", preprocess_meta.get("imbalance_method_used", "Not Applied")],
    ]
    tbl2 = Table(prep_data, colWidths=[8 * cm, 8 * cm])
    tbl2.setStyle(_table_style())
    story.append(tbl2)

    # ── SECTION 3: FEATURE SELECTION ─────────────────────────────────────────
    _section(story, "3 · Feature Selection", styles)
    if fs_meta:
        story.append(Paragraph(
            f"Method: <b>{fs_meta.get('method', 'N/A').upper()}</b> | "
            f"Original: <b>{fs_meta.get('n_original', '?')}</b> → "
            f"Selected: <b>{fs_meta.get('n_selected', '?')}</b>",
            styles["body"],
        ))
        if "cumulative_variance" in fs_meta:
            story.append(Paragraph(
                f"PCA cumulative explained variance: {fs_meta['cumulative_variance']:.2%}",
                styles["body"],
            ))
        sel_feats = fs_meta.get("selected_features", [])
        if sel_feats:
            story.append(Paragraph(
                f"<b>Selected Features:</b> {', '.join(sel_feats[:30])}{'...' if len(sel_feats) > 30 else ''}",
                styles["body"],
            ))
    else:
        story.append(Paragraph(
            "Feature selection was not applied — the dataset already has a low number of features.",
            styles["body"],
        ))

    # ── SECTION 4: MODEL TRAINING ─────────────────────────────────────────────
    _section(story, "4 · Model Training & Cross-Validation Results", styles)
    cv_res = metrics.get("cv_results", {})
    story.append(Paragraph(
        f"5-fold cross-validation on {len(cv_res)} model(s). "
        f"Best model: <b>{best_model}</b> (CV score: {train_result.get('best_cv_score', 0):.4f})",
        styles["body"],
    ))

    if cv_res:
        cv_data = [["Model", "CV Mean Score", "CV Std Dev", "Status"]]
        sorted_models = sorted(cv_res.items(), key=lambda x: x[1].get("mean", 0), reverse=True)
        best_row_idx = None
        for i, (model_name, scores) in enumerate(sorted_models):
            if model_name == best_model:
                best_row_idx = i + 1   # +1 for header row
            err = scores.get("error")
            cv_data.append([
                model_name,
                f"{scores.get('mean', 0):.4f}" if not err else "ERROR",
                f"±{scores.get('std', 0):.4f}" if not err else "-",
                "⭐ Best" if model_name == best_model and not err else ("" if not err else str(err)[:30]),
            ])
        cv_table = Table(cv_data, colWidths=[5.5 * cm, 4 * cm, 3.5 * cm, 3 * cm])
        cv_table.setStyle(_table_style(highlight_row=best_row_idx))
        story.append(cv_table)

    story.append(PageBreak())

    # ── SECTION 5: EVALUATION METRICS ────────────────────────────────────────
    task_type_lower = decisions.get("task_type", "classification")
    _section(story, "5 · Model Evaluation on Test Set (80/20 Split)", styles)
    story.append(Paragraph(
        f"Test samples: <b>{metrics.get('n_test_samples', '?')}</b>  |  "
        f"Training samples: <b>{metrics.get('n_train_samples', '?')}</b>",
        styles["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # All Models Comparison Table
    story.append(Paragraph("<b>Test Set Performance (All Selected Models):</b>", styles["bold_label"]))
    all_metrics = metrics.get("all_metrics", {})
    
    if task_type_lower == "classification":
        compare_data = [["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
        best_idx = None
        for i, (name, m_res) in enumerate(all_metrics.items()):
            if name == best_model: best_idx = i + 1
            acc = f"{m_res.get('accuracy', 0):.4f}"
            prec = f"{m_res.get('precision', 0):.4f}"
            rec = f"{m_res.get('recall', 0):.4f}"
            f1 = f"{m_res.get('f1', 0):.4f}"
            roc = f"{m_res['roc_auc']:.4f}" if m_res.get("roc_auc") else "N/A"
            compare_data.append([
                name + (" ⭐" if name == best_model else ""),
                acc, prec, rec, f1, roc
            ])
        comp_tbl = Table(compare_data, colWidths=[4*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        comp_tbl.setStyle(_table_style(highlight_row=best_idx))
        story.append(comp_tbl)
        
    else:  # regression
        compare_data = [["Model", "MAE", "MSE", "RMSE", "R² Score"]]
        best_idx = None
        for i, (name, m_res) in enumerate(all_metrics.items()):
            if name == best_model: best_idx = i + 1
            mae = f"{m_res.get('mae', 0):.4f}"
            mse = f"{m_res.get('mse', 0):.4f}"
            rmse = f"{m_res.get('rmse', 0):.4f}"
            r2 = f"{m_res.get('r2', 0):.4f}"
            compare_data.append([
                name + (" ⭐" if name == best_model else ""),
                mae, mse, rmse, r2
            ])
        comp_tbl = Table(compare_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        comp_tbl.setStyle(_table_style(highlight_row=best_idx))
        story.append(comp_tbl)

    # Classification report is stored inside all_metrics for the best model
    best_metrics_detail = all_metrics.get(best_model, metrics)
    clf_report = best_metrics_detail.get(
        "classification_report",
        metrics.get("classification_report")   # fallback for old format
    )
    if task_type_lower == "classification" and clf_report:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(
            f"<b>Full Classification Report — Best Model: {best_model} ⭐</b>",
            styles["bold_label"]
        ))
        story.append(Paragraph(
            "(Confusion matrix and per-class metrics are shown for the best-performing model only, "
            "since the comparison table above already covers all models.)",
            styles["body"]
        ))
        story.append(Spacer(1, 0.2 * cm))
        report_text = clf_report.replace("\n", "<br/>").replace(" ", "&nbsp;")
        story.append(Paragraph(report_text, styles["mono"]))

    story.append(PageBreak())

    # ── SECTION 6: PLOTS ─────────────────────────────────────────────────────
    _section(story, "6 · Visualisations", styles)

    plot_labels = {
        "class_distribution.png":  "Class Distribution (Before / After Balancing)",
        "confusion_matrix.png":    f"Confusion Matrix — Best Model: {best_model} ⭐",
        "roc_curve.png":           "ROC Curves — All Models (Test Set)",
        "feature_importance.png":  f"Feature Importance — Best Model: {best_model} ⭐",
        "model_comparison.png":    "Model Comparison — CV Score vs Test Score (All Models)",
        "correlation_heatmap.png": "Feature Correlation Heatmap",
        "residuals_plot.png":      f"Residuals vs Predicted — Best Model: {best_model} ⭐",
        "predicted_vs_actual.png": "Predicted vs Actual Values — All Models (Test Set)",
    }

    for plot_path in plot_paths:
        if not os.path.exists(plot_path):
            continue
        basename = os.path.basename(plot_path)
        label = plot_labels.get(basename, basename)
        story.append(Paragraph(label, styles["plot_label"]))
        img = _embed_plot(plot_path, width_cm=15)
        if img:
            story.append(img)
        story.append(Spacer(1, 0.5 * cm))

    # ── BUILD ────────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title="ML Pipeline Analysis Report",
        author="ML Bot",
    )

    def _add_page_decorations(canvas, doc):
        canvas.saveState()
        # Header bar
        canvas.setFillColor(C_ACCENT)
        canvas.rect(0, A4[1] - 1.4 * cm, A4[0], 1.4 * cm, fill=1, stroke=0)
        canvas.setFillColor(C_WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(2 * cm, A4[1] - 0.85 * cm, "ML Pipeline Analysis Report — Groq AI Powered")
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 0.85 * cm,
                               datetime.now().strftime("%Y-%m-%d"))
        # Footer
        canvas.setFillColor(C_MID)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(A4[0] / 2, 0.7 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_add_page_decorations, onLaterPages=_add_page_decorations)
    return output_path

"""
export_helper.py — Saves trained model artifacts & generates a deployable zip package.

Saves to: D:\\Users\\Rithik\\Desktop\\Output\\<dataset_name>\\
Zip contains:
  - best_model.pkl         (trained sklearn estimator)
  - scaler.pkl             (fitted scaler)
  - metadata.json          (feature names, target col, task type, encoders info)
  - predict.py             (ready-to-use boilerplate inference script)
  - requirements.txt       (dependencies to run predict.py)
"""
from __future__ import annotations

import json
import os
import textwrap
import zipfile
from datetime import datetime

import joblib
import pandas as pd
import sklearn
import xgboost
import lightgbm


# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_ROOT = r"D:\Users\Rithik\Desktop\Output"


# ── Core save + zip function ──────────────────────────────────────────────────

def save_and_zip(
    output_dir: str,
    filename: str,
    train_result: dict,
    preprocess_meta: dict,
    decisions: dict,
    orig_df: pd.DataFrame = None,
) -> str:
    """
    Persist the best model + preprocessing artifacts and bundle them into a zip.

    Returns the path to the created .zip file.
    """
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    export_dir = os.path.join(output_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    best_model     = train_result["best_model"]
    best_model_name = train_result["best_model_name"]
    feature_names  = train_result.get("feature_names", [])
    task_type      = decisions.get("task_type", "classification")
    target_col     = decisions.get("target_column", "target")
    scaler         = preprocess_meta.get("scaler")
    label_encoders = preprocess_meta.get("label_encoders", {})
    target_encoder = preprocess_meta.get("target_encoder")

    # ── 1. Save models ────────────────────────────────────────────────────────
    model_path = os.path.join(export_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    
    if "tuned_model" in train_result:
        tuned_path = os.path.join(export_dir, "tuned_model.pkl")
        joblib.dump(train_result["tuned_model"], tuned_path)

    # ── 2. Save scaler ────────────────────────────────────────────────────────
    scaler_path = os.path.join(export_dir, "scaler.pkl")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
    else:
        scaler_path = None  # no scaler used

    # ── 3. Save label encoders (feature + target) ─────────────────────────────
    encoders_path = None
    if label_encoders:
        encoders_path = os.path.join(export_dir, "label_encoders.pkl")
        joblib.dump(label_encoders, encoders_path)

    target_encoder_path = None
    if target_encoder is not None:
        target_encoder_path = os.path.join(export_dir, "target_encoder.pkl")
        joblib.dump(target_encoder, target_encoder_path)

    # ── 4. Save metadata JSON ─────────────────────────────────────────────────
    metadata = {
        "dataset_name":      dataset_name,
        "model_name":        best_model_name,
        "task_type":         task_type,
        "target_column":     target_col,
        "feature_names":     feature_names,
        "has_scaler":        scaler is not None,
        "has_label_encoders": bool(label_encoders),
        "has_target_encoder": target_encoder is not None,
        "target_classes":    (
            list(target_encoder.classes_) if target_encoder is not None else None
        ),
        "has_tuned_model":   "tuned_model" in train_result,
        "exported_at":       datetime.now().isoformat(),
    }
    meta_path = os.path.join(export_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ── 5. Generate predict.py & app.py ─────────────────────────────────────────
    predict_py = _generate_predict_script(metadata)
    predict_path = os.path.join(export_dir, "predict.py")
    with open(predict_path, "w", encoding="utf-8") as f:
        f.write(predict_py)
        
    app_py = _generate_gradio_script(metadata, orig_df)
    app_path = os.path.join(export_dir, "app.py")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(app_py)

    # ── 6. Generate requirements.txt ─────────────────────────────────────────
    req_path = os.path.join(export_dir, "requirements.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(
            f"scikit-learn=={sklearn.__version__}\n"
            f"xgboost=={xgboost.__version__}\n"
            f"lightgbm=={lightgbm.__version__}\n"
            "joblib>=1.3\n"
            "numpy>=1.24\n"
            "pandas>=2.0\n"
            "gradio>=4.0\n"
            "huggingface_hub==0.25.2\n"
        )

    # ── 7. Bundle into zip ────────────────────────────────────────────────────
    zip_path = os.path.join(output_dir, f"{dataset_name}_deployment.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(export_dir):
            zf.write(os.path.join(export_dir, fname), arcname=fname)

    return zip_path


# ── Boilerplate inference script generator ───────────────────────────────────

def _generate_predict_script(meta: dict) -> str:
    feature_names   = meta["feature_names"]
    task_type       = meta["task_type"]
    has_scaler      = meta["has_scaler"]
    has_le          = meta["has_label_encoders"]
    has_te          = meta["has_target_encoder"]
    target_col      = meta["target_column"]
    model_name      = meta["model_name"]
    classes         = meta.get("target_classes")
    classes_repr    = repr(classes) if classes else "None"

    scale_block = textwrap.dedent("""
    # Scale features
    sample_scaled = scaler.transform(sample_df)
    sample_df = pd.DataFrame(sample_scaled, columns=feature_names)
    """) if has_scaler else ""

    le_block = textwrap.dedent("""
    # Apply feature label encoders for categorical columns
    for col, le in label_encoders.items():
        if col in sample_df.columns:
            sample_df[col] = le.transform(sample_df[col].astype(str))
    """) if has_le else ""

    decode_block = textwrap.dedent(f"""
    # Decode predicted label back to original class name
    prediction = target_encoder.inverse_transform(prediction)[0]
    print(f"Predicted {target_col}: {{prediction}}")
    """) if has_te else f"\nprint(f\"Predicted {target_col}: {{prediction[0]}}\")\n"

    proba_block = textwrap.dedent(f"""
    # Probability scores (if supported)
    if hasattr(model, 'predict_proba'):
        classes = {classes_repr}
        proba = model.predict_proba(sample_df)[0]
        for cls, p in zip(classes or range(len(proba)), proba):
            print(f"  P({{cls}}) = {{p:.4f}}")
    """) if task_type == "classification" else ""

    return textwrap.dedent(f"""\
    \"\"\"
    Auto-generated predict.py — {model_name} model for '{meta['dataset_name']}'
    Generated: {meta['exported_at']}

    Usage:
        python predict.py
    Edit the `sample_input` dict below to run your own prediction.
    \"\"\"
    import json
    import joblib
    import numpy as np
    import pandas as pd

    # ── Load artifacts ──────────────────────────────────────────────────────
    model   = joblib.load("best_model.pkl")
    {'tuned_model = joblib.load("tuned_model.pkl")' if meta.get('has_tuned_model') else ''}
    {'scaler  = joblib.load("scaler.pkl")' if has_scaler else ''}
    {'label_encoders = joblib.load("label_encoders.pkl")' if has_le else ''}
    {'target_encoder = joblib.load("target_encoder.pkl")' if has_te else ''}

    with open("metadata.json") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]

    # ── Provide your input here ─────────────────────────────────────────────
    sample_input = {{
        col: 0 for col in feature_names  # <-- replace 0 with actual values
    }}
    # Example:  sample_input = {{{", ".join(f'"{c}": 0' for c in feature_names[:3])}...}}

    # ── Run prediction ──────────────────────────────────────────────────────
    sample_df = pd.DataFrame([sample_input])[feature_names]
    {le_block}
    {scale_block}

    active_model = {'tuned_model' if meta.get('has_tuned_model') else 'model'}
    prediction = active_model.predict(sample_df)
    {decode_block}
    {proba_block.replace("model.predict_proba", "active_model.predict_proba")}
    """)


def _generate_gradio_script(meta: dict, orig_df: pd.DataFrame = None) -> str:
    feature_names   = meta["feature_names"]
    has_scaler      = meta["has_scaler"]
    has_le          = meta["has_label_encoders"]
    has_te          = meta["has_target_encoder"]
    target_col      = meta["target_column"]
    model_name      = meta["model_name"]

    scale_block = textwrap.dedent("""
        # Scale features
        sample_scaled = scaler.transform(sample_df)
        sample_df = pd.DataFrame(sample_scaled, columns=feature_names)
    """) if has_scaler else ""

    le_block = textwrap.dedent("""
        # Apply feature label encoders for categorical columns
        for col, le in label_encoders.items():
            if col in sample_df.columns:
                sample_df[col] = le.transform(sample_df[col].astype(str))
    """) if has_le else ""

    decode_block = textwrap.dedent(f"""
        # Decode predicted label back to original class name
        prediction = target_encoder.inverse_transform(prediction)[0]
    """) if has_te else "\n        prediction = prediction[0]\n"

    # Create Gradio inputs code
    inputs_lines = []
    MAX_CATS = 12
    for f in feature_names:
        input_str = f'gr.Textbox(label="{f}")'
        if orig_df is not None and f in orig_df.columns:
            col_series = orig_df[f]
            dtype = col_series.dtype
            uniques = col_series.dropna().unique().tolist()
            if len(uniques) <= MAX_CATS:
                if pd.api.types.is_numeric_dtype(dtype):
                    choices_str = ", ".join(str(u) for u in sorted(uniques))
                    val_str = str(sorted(uniques)[0])
                else:
                    choices_str = ", ".join(f'"{str(u)}"' for u in sorted(uniques, key=str))
                    val_str = f'"{str(sorted(uniques, key=str)[0])}"'
                
                input_str = f'gr.Dropdown(label="{f}", choices=[{choices_str}], value={val_str})'
            else:
                if pd.api.types.is_numeric_dtype(dtype):
                    _min, _max = col_series.min(), col_series.max()
                    if pd.notna(_min):
                        if pd.api.types.is_integer_dtype(dtype):
                            input_str = f'gr.Slider(label="{f}", minimum={_min}, maximum={_max}, step=1, value={col_series.median():.0f})'
                        else:
                            input_str = f'gr.Slider(label="{f}", minimum={_min}, maximum={_max}, value={col_series.median():.2f})'
        inputs_lines.append(input_str)
        
    if meta.get("has_tuned_model"):
        inputs_lines.append('gr.Radio(choices=["Baseline", "Tuned"], value="Tuned", label="Model Version")')

    inputs_code = ",\n            ".join(inputs_lines)
    
    return textwrap.dedent(f"""\
    \"\"\"
    Auto-generated app.py — Gradio UI for {model_name} model 
    Generated: {meta['exported_at']}

    Usage:
        python app.py
    This will launch a local web server with a UI for your model.
    \"\"\"
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import gradio as gr

    # ── Load artifacts ──────────────────────────────────────────────────────
    model   = joblib.load("best_model.pkl")
    {'tuned_model = joblib.load("tuned_model.pkl")' if meta.get('has_tuned_model') else ''}
    {'scaler  = joblib.load("scaler.pkl")' if has_scaler else ''}
    {'label_encoders = joblib.load("label_encoders.pkl")' if has_le else ''}
    {'target_encoder = joblib.load("target_encoder.pkl")' if has_te else ''}

    with open("metadata.json") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]

    def predict(*args):
        # ── Map inputs to dataframe ─────────────────────────────────────────────
        if {meta.get('has_tuned_model')}:
            model_version = args[-1]
            args = args[:-1]
            active_model = tuned_model if model_version == "Tuned" else model
        else:
            active_model = model

        row = {{f: a for f, a in zip(feature_names, args)}}
        sample_df = pd.DataFrame([row])[feature_names]

        # ── Preprocess ──────────────────────────────────────────────────────────
{textwrap.indent(le_block, '        ')}
{textwrap.indent(scale_block, '        ')}

        # ── Predict ─────────────────────────────────────────────────────────────
        prediction = active_model.predict(sample_df)
{textwrap.indent(decode_block, '        ')}
        return str(prediction)

    # ── Build Gradio Interface ──────────────────────────────────────────────
    demo = gr.Interface(
        fn=predict,
        inputs=[
            {inputs_code}
        ],
        outputs=gr.Textbox(label="Predicted {target_col}"),
        title="{meta['dataset_name']} Predictor",
        description="Live inference endpoint powered by {model_name}.",
        theme=gr.themes.Soft()
    )

    if __name__ == "__main__":
        demo.launch()
    """)

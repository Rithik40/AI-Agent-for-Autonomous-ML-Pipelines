"""
Bot handlers — ALL blocking pipeline calls run in asyncio.to_thread()
so the event loop is never blocked (no more Telegram TimedOut errors).
"""
from __future__ import annotations

import asyncio
import os
import functools

import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, ConversationHandler

import config
from pipeline import (
    dataset_analyzer, preprocessor, feature_selector,
    model_trainer, evaluator, plotter,
)
from pipeline.model_trainer import get_all_models_for_task
from utils import helpers
from utils.helpers import make_output_dir
from utils.pdf_reporter import build_pdf
from utils import export_helper
from utils import shap_explainer
from pipeline import dataset_analyzer as _ds_analyzer

# Conversation states
SELECTING_MODELS = 1
INFERENCE_MODE   = 2
WIZARD_PREDICTION = 3
WANT_TUNING      = 4

# user_data keys
UD_DF             = "df"
UD_DECISIONS      = "decisions"
UD_SESSION_DIR    = "session_dir"
UD_FILENAME       = "filename"
UD_SELECTED       = "selected_models"
UD_ALL_MODELS     = "all_models"
UD_INFERENCE_META = "inference_meta"   # stores model artifacts for live prediction


# ─────────────────── commands ────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 *ML Pipeline Bot* — powered by Groq AI\n\n"
        "Send me a *CSV file with a label column* and I will:\n\n"
        "1️⃣  Analyse your dataset with Groq AI\n"
        "2️⃣  Let you *choose which ML models* to train\n"
        "3️⃣  Apply smart preprocessing\n"
        "4️⃣  Select features if high-dimensional\n"
        "5️⃣  Train & compare models with 5-fold CV\n"
        "6️⃣  Send a *PDF report* with all metrics & plots\n"
        "7️⃣  Enter *Live Predictor* mode — query the trained model\n\n"
        "📎 Drop a CSV file here to start!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Exit inference/prediction mode."""
    context.user_data.pop(UD_INFERENCE_META, None)
    await update.message.reply_text(
        "👋 *Exited Inference Mode.* Send a new CSV file to start another pipeline!",
        parse_mode=ParseMode.MARKDOWN,
    )
    return ConversationHandler.END


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *Help*\n\n"
        "*Supported:* `.csv` files with headers\n"
        "*Target column:* Auto-detected by AI (last column fallback)\n"
        "*Model selection:* Pick which models to evaluate after upload\n\n"
        "Commands: `/start`  `/help`\n"
        "Max file size: 20 MB (Telegram limit)",
        parse_mode=ParseMode.MARKDOWN,
    )


# ─────────────────── keyboard builder ────────────────────────────────────────

def _build_model_keyboard(
    all_models: list[str],
    selected: set[str],
    recommended: list[str],
) -> InlineKeyboardMarkup:
    buttons = []
    row = []
    for i, name in enumerate(all_models):
        is_sel = name in selected
        is_rec = name in recommended
        if is_sel and is_rec:
            icon = "✅⭐"
        elif is_sel:
            icon = "✅"
        elif is_rec:
            icon = "⭐"
        else:
            icon = "☐"
        row.append(InlineKeyboardButton(f"{icon} {name}", callback_data=f"toggle:{name}"))
        if len(row) == 2 or i == len(all_models) - 1:
            buttons.append(row)
            row = []

    buttons.append([
        InlineKeyboardButton("☑️ Select All",   callback_data="sel_all"),
        InlineKeyboardButton("🔳 Deselect All", callback_data="desel_all"),
    ])
    buttons.append([
        InlineKeyboardButton("✅ Confirm & Run Pipeline", callback_data="confirm"),
    ])
    return InlineKeyboardMarkup(buttons)


def _format_selection_message(task_type: str, target: str, recommended: list[str], selected: set[str]) -> str:
    """Format the text message above the keyboard to clearly show selected models."""
    rec_str = ", ".join(f"`{m}`" for m in recommended) or "None"
    
    sel_list = sorted(list(selected))
    sel_str = ", ".join(f"`{m}`" for m in sel_list) if sel_list else "⚠️ _None selected_"
    
    return (
        f"✅ AI analysis complete!\n"
        f"   *Task:* {task_type.capitalize()}  |  *Target:* `{target}`\n\n"
        f"🤖 *AI Recommended:* {rec_str}\n"
        f"_(⭐ = AI recommended)_\n\n"
        f"✅ *Currently Selected:* {sel_str}\n\n"
        f"*Tap buttons to toggle, then press Confirm:*"
    )


# ─────────────────── CSV upload handler ──────────────────────────────────────

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".csv"):
        await update.message.reply_text(
            "⚠️ Please send a *CSV file* (`.csv` extension).",
            parse_mode=ParseMode.MARKDOWN,
        )
        return ConversationHandler.END

    session_dir = helpers.make_session_dir()
    csv_path = os.path.join(session_dir, doc.file_name)

    status_msg = await update.message.reply_text(
        "⏳ *Downloading your dataset…*", parse_mode=ParseMode.MARKDOWN
    )

    # ── Download CSV ──────────────────────────────────────────────────────────
    try:
        file = await context.bot.get_file(doc.file_id)
        await file.download_to_drive(csv_path)
    except Exception as e:
        await status_msg.edit_text(f"❌ Download failed: {e}")
        helpers.cleanup_session(session_dir)
        return ConversationHandler.END

    # ── Load CSV (in thread — pandas can be slow on large files) ─────────────
    try:
        df = await asyncio.to_thread(pd.read_csv, csv_path)
        if df.empty or df.shape[1] < 2:
            await status_msg.edit_text("❌ CSV is empty or has fewer than 2 columns.")
            helpers.cleanup_session(session_dir)
            return ConversationHandler.END
    except Exception as e:
        await status_msg.edit_text(f"❌ Could not parse CSV: {e}")
        helpers.cleanup_session(session_dir)
        return ConversationHandler.END

    await status_msg.edit_text(
        f"✅ Loaded `{doc.file_name}` — *{df.shape[0]} rows × {df.shape[1]} cols*\n\n"
        "🧠 Analysing dataset with Groq AI… _(may take ~15 seconds)_",
        parse_mode=ParseMode.MARKDOWN,
    )

    # ── AI Analysis — RUNS IN THREAD (blocking Groq HTTP call) ───────────────
    try:
        decisions = await asyncio.to_thread(dataset_analyzer.analyze_dataset, df)
    except Exception as e:
        await status_msg.edit_text(
            f"❌ Groq AI analysis failed:\n`{e}`\n\n"
            "_Check your GROQ_API_KEY and network connection._",
            parse_mode=ParseMode.MARKDOWN,
        )
        helpers.cleanup_session(session_dir)
        return ConversationHandler.END

    task_type   = decisions.get("task_type", "classification")
    target      = decisions.get("target_column", "?")
    recommended = decisions.get("recommended_models", [])
    all_models  = get_all_models_for_task(task_type)

    selected: set[str] = set(m for m in recommended if m in all_models)
    if not selected:
        selected = set(all_models)

    context.user_data[UD_DF]          = df
    context.user_data[UD_DECISIONS]   = decisions
    context.user_data[UD_SESSION_DIR] = session_dir
    context.user_data[UD_FILENAME]    = doc.file_name
    context.user_data[UD_SELECTED]    = selected
    context.user_data[UD_ALL_MODELS]  = all_models

    keyboard = _build_model_keyboard(all_models, selected, recommended)
    msg_text = _format_selection_message(task_type, target, recommended, selected)

    await status_msg.edit_text(
        text=msg_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard,
    )
    return SELECTING_MODELS


# ─────────────────── callback: toggle / confirm ───────────────────────────────

async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    all_models  = context.user_data[UD_ALL_MODELS]
    selected    = context.user_data[UD_SELECTED]
    decisions   = context.user_data[UD_DECISIONS]
    recommended = decisions.get("recommended_models", [])
    task_type   = decisions.get("task_type", "classification")
    target      = decisions.get("target_column", "?")
    
    data = query.data

    if data.startswith("toggle:"):
        name = data.split(":", 1)[1]
        selected.discard(name) if name in selected else selected.add(name)
        context.user_data[UD_SELECTED] = selected
        
        keyboard = _build_model_keyboard(all_models, selected, recommended)
        msg_text = _format_selection_message(task_type, target, recommended, selected)
        
        await query.edit_message_text(
            text=msg_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )
        return SELECTING_MODELS

    elif data == "sel_all":
        context.user_data[UD_SELECTED] = set(all_models)
        keyboard = _build_model_keyboard(all_models, set(all_models), recommended)
        msg_text = _format_selection_message(task_type, target, recommended, set(all_models))
        
        await query.edit_message_text(
            text=msg_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )
        return SELECTING_MODELS

    elif data == "desel_all":
        context.user_data[UD_SELECTED] = set()
        keyboard = _build_model_keyboard(all_models, set(), recommended)
        msg_text = _format_selection_message(task_type, target, recommended, set())
        
        await query.edit_message_text(
            text=msg_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )
        return SELECTING_MODELS

    elif data == "confirm":
        selected_list = sorted(selected)
        if not selected_list:
            await query.answer("⚠️ Please select at least one model!", show_alert=True)
            return SELECTING_MODELS

        await query.edit_message_text(
            text=f"✅ Models confirmed: *{', '.join(selected_list)}*\n\n"
                 "⚙️ Running pipeline…",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=None,
        )
        return await _run_pipeline(query, context, selected_list)

    return SELECTING_MODELS


# ─────────────────── pipeline runner (all heavy ops in threads) ───────────────

async def _run_pipeline(query, context: ContextTypes.DEFAULT_TYPE, selected_models: list[str]) -> int:
    df          = context.user_data[UD_DF]
    decisions   = context.user_data[UD_DECISIONS]
    session_dir = context.user_data[UD_SESSION_DIR]
    filename    = context.user_data[UD_FILENAME]
    chat_id     = query.message.chat_id

    # Persistent output folder named after the dataset
    output_dir = make_output_dir(filename)

    async def _msg(text: str, reply_markup=None) -> None:
        await context.bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

    # ── Step 2: Preprocessing ─────────────────────────────────────────────────
    await _msg("⚙️ *Step 2/5:* Preprocessing data…")
    try:
        result = await asyncio.to_thread(preprocessor.preprocess, df, decisions)
        X, y, preprocess_meta = result
        y_before_balancing = df[decisions["target_column"]].copy()
    except Exception as e:
        await _msg(f"❌ Preprocessing failed:\n`{e}`")
        helpers.cleanup_session(session_dir)
        return ConversationHandler.END

    # ── Step 3: Feature selection ─────────────────────────────────────────────
    fs_meta = None
    await _msg("🔍 *Step 3/5:* Feature selection (if needed)…")
    if decisions.get("needs_feature_selection") and decisions.get("feature_selection_method"):
        try:
            X, fs_meta = await asyncio.to_thread(
                feature_selector.select_features, X, y,
                decisions["feature_selection_method"],
                decisions.get("task_type", "classification"),
            )
        except Exception as e:
            await _msg(f"⚠️ Feature selection skipped: `{e}`")

    # ── Step 4: Model training ────────────────────────────────────────────────
    await _msg(
        f"🚀 *Step 4/5:* Training {len(selected_models)} model(s):\n"
        f"*{', '.join(selected_models)}*\n_(5-fold CV — may take a minute…)_"
    )
    try:
        train_result = await asyncio.to_thread(
            model_trainer.train_models, X, y, decisions, selected_models
        )
    except Exception as e:
        await _msg(f"❌ Model training failed:\n`{e}`")
        helpers.cleanup_session(session_dir)
        return ConversationHandler.END

    best = train_result["best_model_name"]
    best_cv = train_result["best_cv_score"]

    # ── Auto-Retry if CV score is poor ────────────────────────────────────────
    RETRY_THRESHOLD = getattr(config, "RETRY_THRESHOLD", 0.60)
    if best_cv < RETRY_THRESHOLD:
        retry_reasoning = ""
        try:
            await _msg(
                f"\u26a0\ufe0f Best CV score `{best_cv:.4f}` is below threshold `{RETRY_THRESHOLD:.2f}`.\n\n"
                "\U0001f9e0 Asking Groq AI for an improved strategy\u2026"
            )
            new_decisions = await asyncio.to_thread(
                _ds_analyzer.retry_analyze_dataset,
                df, decisions, best_cv,
            )
            retry_reasoning = new_decisions.get("reasoning", "")
            new_models = new_decisions.get("recommended_models", selected_models)

            await _msg(
                f"\U0001f504 *Retry Strategy:*\n_{retry_reasoning[:300]}_\n\n"
                f"New models to try: *{', '.join(new_models)}*\n"
                "Re-running pipeline with improved settings\u2026"
            )

            # Re-preprocess with new decisions
            retry_result = await asyncio.to_thread(
                preprocessor.preprocess, df, new_decisions
            )
            X_r, y_r, preprocess_meta_r = retry_result

            # Re-train with new models
            train_result_r = await asyncio.to_thread(
                model_trainer.train_models, X_r, y_r, new_decisions, new_models
            )

            # Only accept the retry if it actually improves things
            if train_result_r["best_cv_score"] > best_cv:
                train_result  = train_result_r
                decisions     = new_decisions
                preprocess_meta = preprocess_meta_r
                X             = X_r
                y             = y_r
                best          = train_result["best_model_name"]
                best_cv       = train_result["best_cv_score"]
                await _msg(
                    f"\u2705 Retry succeeded! New best: *{best}* "
                    f"(CV: `{best_cv:.4f}` vs original `{train_result['best_cv_score']:.4f}`)"
                )
            else:
                await _msg(
                    f"\u26a0\ufe0f Retry didn't improve (CV: `{train_result_r['best_cv_score']:.4f}`). "
                    "Continuing with original results."
                )
        except Exception as e:
            await _msg(f"\u26a0\ufe0f Auto-retry failed: `{e}`. Continuing with original results.")

    # ── Ask about tuning ──────────────────────────────────────────────────────
    keyboard = [
        [
            InlineKeyboardButton("Yes, Tune Hyperparameters", callback_data="tune_yes"),
            InlineKeyboardButton("No, Skip to Evaluation", callback_data="tune_no"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await _msg(
        f"✅ Best baseline model: *{best}* (CV: {train_result['best_cv_score']:.4f})\n\n"
        "Would you like to run **Hyperparameter Tuning** on this model to squeeze out more performance?\n"
        "_(This may take 1-3 minutes)_",
        reply_markup=reply_markup
    )
    
    # Store everything needed for tuning & evaluation
    context.user_data["train_result"] = train_result
    context.user_data["metrics_meta"] = {
        "X": X,
        "y": y,
        "y_before_balancing": y_before_balancing,
        "preprocess_meta": preprocess_meta,
        "fs_meta": fs_meta,
    }
    
    return WANT_TUNING

# ─────────────────── tuning handler ──────────────────────────────────────────

async def handle_tuning_decision(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id

    async def _msg(m: str):
        return await context.bot.send_message(chat_id, m, parse_mode=ParseMode.MARKDOWN)

    # Reload pipeline state
    session_dir     = context.user_data[UD_SESSION_DIR]
    filename        = context.user_data[UD_FILENAME]
    decisions       = context.user_data[UD_DECISIONS]
    train_result    = context.user_data["train_result"]
    metrics_meta    = context.user_data["metrics_meta"]
    
    X                  = metrics_meta["X"]
    y                  = metrics_meta["y"]
    y_before_balancing = metrics_meta["y_before_balancing"]
    preprocess_meta    = metrics_meta["preprocess_meta"]
    fs_meta            = metrics_meta["fs_meta"]
    
    output_dir = os.path.join(config.TEMP_DIR, session_dir)
    
    if query.data == "tune_yes":
        await _msg("⚙️ Running Hyperparameter Tuning...\n_(This will take a moment)_")
        try:
            train_result = await asyncio.to_thread(
                model_trainer.tune_model, train_result, decisions
            )
            await _msg(
                f"✅ Tuning Complete! New CV Score: `{train_result.get('tuned_cv_score', 0):.4f}` "
                f"(was `{train_result['best_cv_score']:.4f}`)"
            )
        except Exception as e:
            await _msg(f"⚠️ Tuning failed: `{e}`. Falling back to baseline model.")

    best = train_result["best_model_name"]

    # ── Step 5: Evaluation ────────────────────────────────────────────────────
    await _msg("📈 *Step 5/5:* Evaluating & building PDF report…")
    try:
        metrics = await asyncio.to_thread(
            evaluator.evaluate,
            train_result
        )
    except Exception as e:
        await _msg(f"❌ Evaluation failed:\n`{e}`")
        helpers.cleanup_session(session_dir)
        return ConversationHandler.END

    # ── Generate plots (saved to persistent output dir)  ─────────────────────
    y_balanced = y if preprocess_meta.get("imbalance_handled") else None
    
    # We plot the absolute best model (tuned if exists, else baseline)
    plot_model = train_result.get("tuned_model", train_result["best_model"])
    plot_model_name = f"{best} (Tuned)" if "tuned_model" in train_result else best
    
    plot_paths = await asyncio.to_thread(
        plotter.generate_all_plots,
        metrics, X,
        pd.Series(y_before_balancing) if y_before_balancing is not None else y,
        y_balanced,
        plot_model,
        train_result.get("feature_names", list(X.columns)),
        decisions,
        output_dir,
        plot_model_name,
    )

    # ── Build PDF (saved to persistent output dir) ────────────────────────────
    pdf_path = os.path.join(output_dir, f"ml_report_{filename.replace('.csv', '')}.pdf")
    try:
        await asyncio.to_thread(
            build_pdf,
            pdf_path, filename, decisions, preprocess_meta,
            fs_meta, train_result, metrics, plot_paths,
        )
    except Exception as e:
        await _msg(f"⚠️ PDF build failed: `{e}`. Sending text summary only.")
        pdf_path = None

    # ── Send PDF ──────────────────────────────────────────────────────────────
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            await context.bot.send_document(
                chat_id, document=f,
                filename=f"ml_report_{filename.replace('.csv','')}.pdf",
                caption=(
                    f"📄 *ML Report — {filename}*\n"
                    f"Best model: `{best}`"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )

    # ── Export: Save model + zip ──────────────────────────────────────────────
    await _msg("🔧 Saving model artifacts & generating deployment zip…")
    try:
        zip_path = await asyncio.to_thread(
            export_helper.save_and_zip,
            output_dir, filename, train_result, preprocess_meta, decisions, context.user_data.get(UD_DF)
        )
        # Send zip to user
        with open(zip_path, "rb") as zf:
            await context.bot.send_document(
                chat_id, document=zf,
                filename=os.path.basename(zip_path),
                caption=(
                    f"📦 *Deployment Package — {filename}*\n"
                    f"Contains: `best_model.pkl`, `scaler.pkl`, `metadata.json`, "
                    f"`predict.py`, `app.py` (Gradio UI)\n"
                    "Extract and run `python app.py` to get live predictions in your browser!"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
    except Exception as e:
        await _msg(f"⚠️ Could not generate deployment zip: `{e}`")
        zip_path = None

    # ── Send text summary ─────────────────────────────────────────────────────
    await context.bot.send_message(
        chat_id,
        helpers.format_metrics_message(decisions, train_result, metrics, preprocess_meta, fs_meta),
        parse_mode=ParseMode.MARKDOWN,
    )

    helpers.cleanup_session(session_dir)

    # ── Transition to INFERENCE_MODE ──────────────────────────────────────────
    final_model = train_result.get("tuned_model", train_result["best_model"])
    
    # Store everything the inference handler needs
    context.user_data[UD_INFERENCE_META] = {
        "model":          final_model,
        "model_name":     best + (" (Tuned)" if "tuned_model" in train_result else " (Baseline)"),
        "task_type":      decisions.get("task_type", "classification"),
        "target_column":  decisions.get("target_column", "target"),
        "feature_names":  train_result.get("feature_names", []),
        "scaler":         preprocess_meta.get("scaler"),
        "label_encoders": preprocess_meta.get("label_encoders", {}),
        "target_encoder": preprocess_meta.get("target_encoder"),
        "dataset_name":   filename,
    }

    feature_list = ", ".join(f"`{f}`" for f in train_result.get("feature_names", [])[:8])

    # ── Build a real example + feature guide from the uploaded dataset ────────
    feature_names_all = train_result.get("feature_names", [])
    orig_df: pd.DataFrame = context.user_data.get(UD_DF)
    real_example = ""
    feature_guide_lines = []
    feature_guides_dict = {}

    MAX_CATS = 12   

    if orig_df is not None and feature_names_all:
        sample_cols = [c for c in feature_names_all if c in orig_df.columns]
        if sample_cols:
            sample_row = orig_df[sample_cols].iloc[0]

            example_parts = [f"{col} {val}" for col, val in sample_row.items()][:6]
            real_example = ", ".join(example_parts)

            for col in sample_cols:
                col_series = orig_df[col]
                dtype = col_series.dtype
                
                uniques = col_series.dropna().unique().tolist()
                
                if len(uniques) <= MAX_CATS:
                    if pd.api.types.is_numeric_dtype(dtype):
                        type_label = "integer" if pd.api.types.is_integer_dtype(dtype) else "decimal"
                        opts_str = ", ".join(f"{u}" for u in sorted(uniques))
                    else:
                        type_label = "text"
                        opts_str = ", ".join(f'"{str(u)}"' for u in sorted(uniques, key=str))
                        
                    options = f" → options: {opts_str}"
                else:
                    if pd.api.types.is_numeric_dtype(dtype):
                        _min, _max = col_series.min(), col_series.max()
                        if pd.api.types.is_integer_dtype(dtype):
                            type_label = "integer"
                            options = f" (range: {_min} to {_max})" if pd.notna(_min) else ""
                        else:
                            type_label = "decimal"
                            options = f" (range: {_min:.2g} to {_max:.2g})" if pd.notna(_min) else ""
                    else:
                        type_label = "text"
                        options = f" (free text, {col_series.nunique()} unique values)"

                guide_str = f"{type_label}{options}"
                feature_guide_lines.append(f"  • `{col}` — {guide_str}")
                feature_guides_dict[col] = guide_str
                
    context.user_data[UD_INFERENCE_META]["feature_guides"] = feature_guides_dict

    example_text   = f"`{real_example}`" if real_example else "`col1 value1, col2 value2, ...`"
    feature_guide  = "\n".join(feature_guide_lines) if feature_guide_lines else "_Feature info unavailable_"

    await context.bot.send_message(
        chat_id,
        f"🎯 *Live Predictor Mode Active!*\n\n"
        f"Best model: `{best}` is ready for inference.\n\n"
        "*Feature Guide:*\n"
        f"{feature_guide}\n\n"
        "*How to use:*\n"
        "Describe your data in plain English, e.g.:\n"
        f"{example_text}\n"
        "_(That example is a real row from your dataset!)_\n\n"
        "💡 *Or type /predict* to use the step-by-step Wizard mode!\n\n"
        "I will extract the values, run them through the model, and reply with a live prediction! 🚀\n\n"
        "Type /exit to leave Inference Mode.",
        parse_mode=ParseMode.MARKDOWN,
    )
    return INFERENCE_MODE



# ─────────────────── inference chat handler ──────────────────────────────────

async def handle_inference_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Called whenever the user sends a text message while in INFERENCE_MODE.
    Uses Groq to parse the natural-language input into a feature dict, then
    runs the trained model + preprocessing pipeline to return a prediction.
    """
    user_text = update.message.text.strip()
    inf = context.user_data.get(UD_INFERENCE_META)

    if not inf:
        await update.message.reply_text(
            "⚠️ No trained model in memory. Please upload a CSV to start a new pipeline.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return ConversationHandler.END

    feature_names   = inf["feature_names"]
    model           = inf["model"]
    scaler          = inf["scaler"]
    label_encoders  = inf["label_encoders"]
    target_encoder  = inf["target_encoder"]
    task_type       = inf["task_type"]
    target_col      = inf["target_column"]
    model_name      = inf["model_name"]

    if user_text.lower() == "/predict":
        context.user_data["wizard_answers"] = {}
        context.user_data["wizard_index"] = 0
        first_feature = feature_names[0]
        guide_str = inf.get("feature_guides", {}).get(first_feature, "")
        guide_text = f"\n_({guide_str})_" if guide_str else ""
        
        await update.message.reply_text(
            f"🧙‍♂️ *Wizard Mode: Step 1 of {len(feature_names)}*\n\n"
            f"Please enter a value for:\n👉 `{first_feature}`{guide_text}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WIZARD_PREDICTION

    # ── Step 1: Use Groq LLM to extract feature values from user text ─────────
    extracted = context.user_data.pop("wizard_result", None)

    if not extracted:
        await update.message.reply_text("🔍 _Parsing your input…_", parse_mode=ParseMode.MARKDOWN)

        from groq import Groq
        import json as _json

        client = Groq(api_key=config.GROQ_API_KEY)

        prompt = (
            f"The user wants a prediction from a trained ML model. The model has these features:\n"
            f"{feature_names}\n\n"
            f"The user described their input as:\n\"{user_text}\"\n\n"
            "Extract the values and return ONLY a JSON object with feature names as keys and numeric/string values as values. "
            "If a feature is not mentioned, use 0 for numeric or \"unknown\" for text. "
            "Return ONLY the JSON, no explanation."
        )

        try:
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
            )
            raw = response.choices[0].message.content
            # Extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("LLM did not return a valid JSON dict")
            extracted = _json.loads(json_match.group(0))
        except Exception as e:
            await update.message.reply_text(
                f"❌ Could not parse your input: `{e}`\n\n"
                "Try: `Feature1 value, Feature2 value, ...`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return INFERENCE_MODE

    # ── Step 2: Build the sample row, matching training feature order ─────────
    import numpy as np
    import pandas as pd

    try:
        row = {f: extracted.get(f, 0) for f in feature_names}
        sample_df = pd.DataFrame([row])[feature_names]

        # Apply feature label encoders (same as training)
        for col, le in (label_encoders or {}).items():
            if col in sample_df.columns:
                val = str(sample_df.at[0, col])
                if val in le.classes_:
                    sample_df.at[0, col] = le.transform([val])[0]
                else:
                    sample_df.at[0, col] = 0

        display_df = sample_df.copy()

        # Apply scaler (same as training)
        if scaler is not None:
            sample_scaled = scaler.transform(sample_df)
            sample_df = pd.DataFrame(sample_scaled, columns=feature_names)

        # ── Step 3: Predict ───────────────────────────────────────────────────
        prediction_raw = model.predict(sample_df)[0]

        # Decode target label if label encoder was used
        if target_encoder is not None:
            try:
                prediction = target_encoder.inverse_transform([int(prediction_raw)])[0]
            except Exception:
                prediction = prediction_raw
        else:
            prediction = prediction_raw

        # ── Step 4: Build response ────────────────────────────────────────────
        emoji = "🟢" if task_type == "classification" else "📊"

        # Optional: probability scores
        proba_text = ""
        if task_type == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(sample_df)[0]
                classes = target_encoder.classes_ if target_encoder is not None else list(range(len(proba)))
                proba_lines = "\n".join(
                    f"  `{cls}`: {p:.1%}" for cls, p in zip(classes, proba)
                )
                proba_text = f"\n\n*Confidence Scores:*\n{proba_lines}"
            except Exception:
                pass

        # Parsed input summary
        parsed_summary = ", ".join(f"`{k}={v}`" for k, v in list(extracted.items())[:6])
        suffix = '…' if len(extracted) > 6 else ''

        await update.message.reply_text(
            f"{emoji} *Prediction Result*\n\n"
            f"\U0001f4e5 Input: {parsed_summary}{suffix}\n\n"
            f"\U0001f52e *Predicted {target_col}:* `{prediction}`\n"
            f"\U0001f916 Model: `{model_name}`"
            f"{proba_text}\n\n"
            "_Send another query or /exit to stop._",
            parse_mode=ParseMode.MARKDOWN,
        )

        # ── SHAP Explanation ─────────────────────────────────────────
        try:
            inf_meta = context.user_data.get(UD_INFERENCE_META, {})
            output_dir = make_output_dir(inf_meta.get("dataset_name", "unknown"))
            shap_text, shap_plot = await asyncio.to_thread(
                shap_explainer.explain_prediction,
                model, sample_df, feature_names, output_dir, 10, display_df
            )
            await update.message.reply_text(shap_text, parse_mode=ParseMode.MARKDOWN)
            if shap_plot and os.path.exists(shap_plot):
                with open(shap_plot, "rb") as img_f:
                    await context.bot.send_photo(
                        chat_id=update.message.chat_id,
                        photo=img_f,
                        caption="\U0001f4ca SHAP Waterfall — feature contributions to this prediction",
                    )
        except Exception as shap_err:
            # SHAP is optional — silently skip if it fails
            pass

    except Exception as e:
        await update.message.reply_text(
            f"\u274c Prediction failed: `{e}`\n"
            "Please check your input format and try again.",
            parse_mode=ParseMode.MARKDOWN,
        )

    return INFERENCE_MODE


# ─────────────────── wizard prediction handler ──────────────────────────────────

async def handle_wizard_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles step-by-step feature input."""
    user_text = update.message.text.strip()
    if user_text.lower() == "/exit":
        await update.message.reply_text("Exited wizard. Back in open inference mode.", parse_mode=ParseMode.MARKDOWN)
        return INFERENCE_MODE

    inf = context.user_data.get(UD_INFERENCE_META)
    answers = context.user_data.get("wizard_answers", {})
    idx = context.user_data.get("wizard_index", 0)
    feature_names = inf["feature_names"]

    # Save the answer for the current feature
    current_feature = feature_names[idx]
    
    # Try converting to float if possible, else keep string
    try:
        val = float(user_text) if "." in user_text else int(user_text)
    except ValueError:
        val = user_text
        
    answers[current_feature] = val
    idx += 1
    
    # If there are more features, ask the next one
    if idx < len(feature_names):
        context.user_data["wizard_answers"] = answers
        context.user_data["wizard_index"] = idx
        next_feature = feature_names[idx]
        guide_str = inf.get("feature_guides", {}).get(next_feature, "")
        guide_text = f"\n_({guide_str})_" if guide_str else ""
        
        await update.message.reply_text(
            f"🧙‍♂️ *Wizard Mode: Step {idx + 1} of {len(feature_names)}*\n\n"
            f"Please enter a value for:\n👉 `{next_feature}`{guide_text}\n"
            "_(Type /exit to cancel)_",
            parse_mode=ParseMode.MARKDOWN,
        )
        return WIZARD_PREDICTION
        
    # If done, run prediction
    await update.message.reply_text("✨ _Running prediction with your inputs..._", parse_mode=ParseMode.MARKDOWN)
    
    # Temporarily pass the extracted dictionary directly into inference mode
    context.user_data["wizard_result"] = answers
    
    # Clear wizard state
    context.user_data.pop("wizard_answers", None)
    context.user_data.pop("wizard_index", None)

    return await handle_inference_chat(update, context)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            f"❌ Unexpected error: `{context.error}`", parse_mode=ParseMode.MARKDOWN
        )

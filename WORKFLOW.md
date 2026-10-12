# 🔄 Workflow: What Happens When You Upload a CSV

This document tracks every operation that occurs — and exactly **where** it runs.

---

## 🌐 Location Legend

| Symbol | Location | Meaning |
|--------|----------|---------|
| 💻 | **Your Laptop** | Runs locally in Python (bot process, sklearn, pandas) |
| 🤖 | **Groq API** | Runs on Groq's cloud servers (LLM inference) |
| 📱 | **Telegram Servers** | Runs on Telegram's infrastructure |

---

## 📤 Phase 0 — You Upload the CSV

| # | What Happens | Where |
|---|-------------|-------|
| 0.1 | You select the CSV file in Telegram and press Send | 📱 Telegram App (your phone/PC) |
| 0.2 | Telegram uploads the file to its own servers and stores it | 📱 Telegram Servers |
| 0.3 | Telegram notifies your bot that a new document arrived | 📱 → 💻 Telegram pushes update to your bot |
| 0.4 | `handle_document()` is triggered in `bot/handlers.py` | 💻 Your Laptop |

---

## 📥 Phase 1 — Download & Load

| # | What Happens | Where |
|---|-------------|-------|
| 1.1 | Bot calls `context.bot.get_file(doc.file_id)` to get a download URL | 💻 → 📱 Bot sends request to Telegram |
| 1.2 | Telegram returns a temporary download URL | 📱 → 💻 |
| 1.3 | Bot downloads the CSV file to your laptop's `tmp/` folder | 💻 Your Laptop (file saved locally) |
| 1.4 | `pandas.read_csv()` loads the file into a DataFrame | 💻 Your Laptop (RAM) |
| 1.5 | Bot validates: file must have ≥2 columns and not be empty | 💻 Your Laptop |
| 1.6 | Bot sends "⏳ Downloading…" and then "✅ Loaded 500 rows × 12 cols" status message to you | 💻 → 📱 Telegram |

---

## 🧠 Phase 2 — AI Dataset Analysis

| # | What Happens | Where |
|---|-------------|-------|
| 2.1 | `_build_dataset_summary(df)` runs — computes shape, dtypes, missing counts, numeric stats, top correlations | 💻 Your Laptop |
| 2.2 | Summary is serialized to a JSON string | 💻 Your Laptop |
| 2.3 | Bot sends the JSON + analysis prompt to Groq API via HTTP | 💻 → 🤖 Groq Cloud |
| 2.4 | Groq's `llama-3.3-70b-versatile` model reads the dataset summary and decides: task type, scaler, models, imbalance handling, feature selection | 🤖 Groq Cloud |
| 2.5 | Groq returns a JSON response string | 🤖 → 💻 |
| 2.6 | Bot parses the JSON, applies safety overrides (forces target = last column, validates model names) | 💻 Your Laptop |
| 2.7 | Bot sends "🧠 Analysing with Groq AI…" message | 💻 → 📱 Telegram |

---

## 🎛️ Phase 3 — Model Selection Keyboard

| # | What Happens | Where |
|---|-------------|-------|
| 3.1 | Bot builds an `InlineKeyboardMarkup` with all available models for the task type | 💻 Your Laptop |
| 3.2 | AI-recommended models are starred (⭐) in the keyboard | 💻 Your Laptop |
| 3.3 | Bot sends the keyboard message to you | 💻 → 📱 Telegram |
| 3.4 | You tap model buttons to toggle them on/off | 📱 Telegram App |
| 3.5 | Each tap sends a `CallbackQuery` to your bot | 📱 → 💻 |
| 3.6 | `handle_model_selection()` updates the selected set and refreshes the keyboard | 💻 Your Laptop |
| 3.7 | You tap "✅ Confirm & Run Pipeline" | 📱 → 💻 |

---

## ⚙️ Phase 4 — Preprocessing

| # | What Happens | Where |
|---|-------------|-------|
| 4.1 | `preprocessor.preprocess(df, decisions)` is called in a background thread (`asyncio.to_thread`) | 💻 Your Laptop (thread pool) |
| 4.2 | Missing numeric values are filled with the **median** of each column | 💻 Your Laptop |
| 4.3 | Missing text values are filled with the **mode** (most common value) | 💻 Your Laptop |
| 4.4 | Text/categorical columns are label-encoded with `sklearn.LabelEncoder` | 💻 Your Laptop |
| 4.5 | Target column `y` is label-encoded if it's text | 💻 Your Laptop |
| 4.6 | Features are scaled (StandardScaler / MinMaxScaler / RobustScaler — whichever Groq chose) | 💻 Your Laptop |
| 4.7 | If Groq said to handle imbalance: SMOTE / RandomOverSampler / RandomUnderSampler is applied | 💻 Your Laptop |
| 4.8 | All fitted transformers (scaler, encoders) saved in `preprocess_meta` for inference reuse | 💻 Your Laptop (RAM) |
| 4.9 | Bot sends "⚙️ Step 2/5: Preprocessing…" status | 💻 → 📱 Telegram |

---

## 🔍 Phase 5 — Feature Selection *(only if n_features > 15)*

| # | What Happens | Where |
|---|-------------|-------|
| 5.1 | Bot checks `decisions["needs_feature_selection"]` | 💻 Your Laptop |
| 5.2 | If True, `feature_selector.select_features()` runs in a thread | 💻 Your Laptop (thread pool) |
| 5.3 | Chosen method runs: SelectKBest / RFE / PCA — reduces the feature matrix `X` | 💻 Your Laptop |
| 5.4 | `fs_meta` stores the fitted selector | 💻 Your Laptop (RAM) |
| 5.5 | Bot sends "🔍 Step 3/5: Feature selection…" status | 💻 → 📱 Telegram |
| *(skip)* | If n_features ≤ 15, this entire phase is skipped silently | — |

---

## 🚀 Phase 6 — Model Training & Cross-Validation

| # | What Happens | Where |
|---|-------------|-------|
| 6.1 | `model_trainer.train_models()` runs in a background thread | 💻 Your Laptop (thread pool) |
| 6.2 | Data is split 80% train / 20% test | 💻 Your Laptop |
| 6.3 | For each selected model — 5-fold CV runs, scores collected | 💻 Your Laptop |
| 6.4 | Each model is then fitted on the full training set | 💻 Your Laptop |
| 6.5 | Best model selected by highest mean CV score | 💻 Your Laptop |
| 6.6 | Bot sends "🚀 Step 4/5: Training X models… (5-fold CV)" status | 💻 → 📱 Telegram |

---

## 🔄 Phase 7 — Auto-Retry *(only if best CV score < 0.60)*

| # | What Happens | Where |
|---|-------------|-------|
| 7.1 | Bot detects CV score < 0.60 threshold | 💻 Your Laptop |
| 7.2 | `retry_analyze_dataset()` sends a new prompt to Groq describing what failed | 💻 → 🤖 Groq Cloud |
| 7.3 | Groq suggests a different strategy (different models, different scaler, different balancing) | 🤖 Groq Cloud |
| 7.4 | Groq returns new decisions JSON | 🤖 → 💻 |
| 7.5 | Bot re-preprocesses the data with new decisions | 💻 Your Laptop |
| 7.6 | Bot re-trains with newly suggested models | 💻 Your Laptop |
| 7.7 | If retry score > original score → adopt retry results; else keep originals | 💻 Your Laptop |
| 7.8 | Bot sends retry status messages throughout | 💻 → 📱 Telegram |
| *(skip)* | If CV ≥ 0.60, this phase is skipped entirely | — |

---

## ⚡ Phase 8 — Hyperparameter Tuning *(user's choice)*

| # | What Happens | Where |
|---|-------------|-------|
| 8.1 | Bot sends "Would you like to run Hyperparameter Tuning?" with Yes/No buttons | 💻 → 📱 Telegram |
| 8.2 | You tap Yes or No | 📱 → 💻 |
| 8.3 | If Yes: `model_trainer.tune_model()` runs `RandomizedSearchCV` (50 iterations, 5-fold CV) | 💻 Your Laptop (can take 1–3 min) |
| 8.4 | Best hyperparameters found, tuned model stored as `train_result["tuned_model"]` | 💻 Your Laptop |
| 8.5 | Bot reports new tuned CV score vs baseline | 💻 → 📱 Telegram |
| *(skip)* | If No: skips directly to evaluation with baseline model | — |

---

## 📊 Phase 9 — Evaluation

| # | What Happens | Where |
|---|-------------|-------|
| 9.1 | `evaluator.evaluate(train_result)` runs in a thread | 💻 Your Laptop |
| 9.2 | Every trained model makes predictions on `X_test` | 💻 Your Laptop |
| 9.3 | Metrics computed per model: Accuracy / F1 / ROC-AUC (classification) or MAE / R² (regression) | 💻 Your Laptop |
| 9.4 | Confusion matrix computed for the best model | 💻 Your Laptop |
| 9.5 | All results stored in `metrics` dict | 💻 Your Laptop (RAM) |
| 9.6 | Bot sends "📈 Step 5/5: Evaluating…" status | 💻 → 📱 Telegram |

---

## 📈 Phase 10 — Plot Generation

| # | What Happens | Where |
|---|-------------|-------|
| 10.1 | `plotter.generate_all_plots()` runs in a thread | 💻 Your Laptop |
| 10.2 | Class distribution chart saved as PNG | 💻 Your Laptop |
| 10.3 | Confusion matrix heatmap saved as PNG | 💻 Your Laptop |
| 10.4 | ROC curve (all models) saved as PNG | 💻 Your Laptop |
| 10.5 | Feature importance chart saved as PNG | 💻 Your Laptop |
| 10.6 | Model comparison bar chart saved as PNG | 💻 Your Laptop |
| 10.7 | Correlation heatmap saved as PNG | 💻 Your Laptop |
| 10.8 | Residuals plot saved as PNG *(regression only)* | 💻 Your Laptop |
| 10.9 | Predicted vs Actual scatter saved as PNG *(regression only)* | 💻 Your Laptop |
| 10.10 | All PNGs stored in `output/<dataset_name>/` folder | 💻 Your Laptop (disk) |

---

## 📄 Phase 11 — PDF Report Generation

| # | What Happens | Where |
|---|-------------|-------|
| 11.1 | `build_pdf()` runs in a thread using `reportlab` | 💻 Your Laptop |
| 11.2 | Cover page, AI decisions table, preprocessing summary built | 💻 Your Laptop |
| 11.3 | Per-model metrics table and best model section built | 💻 Your Laptop |
| 11.4 | All 8 PNG plots embedded into the PDF | 💻 Your Laptop |
| 11.5 | PDF saved to disk: `output/<name>/ml_report_<name>.pdf` | 💻 Your Laptop (disk) |
| 11.6 | Bot uploads and sends the PDF to you via Telegram | 💻 → 📱 Telegram |
| 11.7 | File travels through Telegram's servers to your device | 📱 Telegram Servers |

---

## 📦 Phase 12 — Deployment Package Export

| # | What Happens | Where |
|---|-------------|-------|
| 12.1 | `export_helper.save_and_zip()` runs in a thread | 💻 Your Laptop |
| 12.2 | Model pickled → `best_model.pkl` | 💻 Your Laptop |
| 12.3 | Scaler pickled → `scaler.pkl` | 💻 Your Laptop |
| 12.4 | Label encoders pickled → `label_encoders.pkl` | 💻 Your Laptop |
| 12.5 | `metadata.json` written (feature names, model name, task type, CV score) | 💻 Your Laptop |
| 12.6 | `predict.py` generated (CLI prediction script) | 💻 Your Laptop |
| 12.7 | `app.py` generated (Gradio web UI) | 💻 Your Laptop |
| 12.8 | All files zipped into `<dataset_name>_deployment.zip` | 💻 Your Laptop |
| 12.9 | Bot uploads and sends the ZIP to you via Telegram | 💻 → 📱 Telegram |

---

## 💬 Phase 13 — Text Summary

| # | What Happens | Where |
|---|-------------|-------|
| 13.1 | `helpers.format_metrics_message()` formats all results into a Telegram message | 💻 Your Laptop |
| 13.2 | Bot sends the text summary (best model, CV score, test metrics, preprocessing steps used) | 💻 → 📱 Telegram |
| 13.3 | Temporary session files (downloaded CSV, etc.) are deleted from `tmp/` | 💻 Your Laptop |

---

## 🎯 Phase 14 — Live Prediction Mode (Inference)

*The bot now waits for you to type a query.*

### 14A — Free-Text Prediction

| # | What Happens | Where |
|---|-------------|-------|
| 14A.1 | You type: `"age 35, salary 60000, department IT"` | 📱 Telegram App |
| 14A.2 | Message arrives at your bot | 📱 → 💻 |
| 14A.3 | `handle_inference_chat()` triggered | 💻 Your Laptop |
| 14A.4 | Bot sends your text + feature list to Groq: *"Extract these feature values as JSON"* | 💻 → 🤖 Groq Cloud |
| 14A.5 | Groq parses the natural language and returns: `{"age": 35, "salary": 60000, "department": "IT"}` | 🤖 Groq Cloud |
| 14A.6 | JSON returned to bot | 🤖 → 💻 |
| 14A.7 | "IT" is encoded with the saved `LabelEncoder` | 💻 Your Laptop |
| 14A.8 | Values are scaled with the saved `Scaler` | 💻 Your Laptop |
| 14A.9 | `model.predict()` runs | 💻 Your Laptop |
| 14A.10 | If classification + `predict_proba` exists: confidence scores computed | 💻 Your Laptop |
| 14A.11 | If target was label-encoded: result is decoded back to original label | 💻 Your Laptop |
| 14A.12 | Prediction result message sent to you | 💻 → 📱 Telegram |

### 14B — Wizard Mode (`/predict`)

| # | What Happens | Where |
|---|-------------|-------|
| 14B.1 | You type `/predict` | 📱 → 💻 |
| 14B.2 | Bot asks for each feature one at a time, showing type and valid values | 💻 → 📱 Telegram |
| 14B.3 | You type a value for each feature | 📱 → 💻 (repeated per feature) |
| 14B.4 | After all features collected, same encode → scale → predict pipeline runs | 💻 Your Laptop |
| 14B.5 | Prediction result sent | 💻 → 📱 Telegram |
| *(Note)* | Wizard mode does NOT call Groq — values are entered directly | — |

---

## 🔍 Phase 15 — SHAP Explanation *(after every prediction)*

| # | What Happens | Where |
|---|-------------|-------|
| 15.1 | `shap_explainer.explain_prediction()` runs in a thread | 💻 Your Laptop |
| 15.2 | For tree models: `shap.TreeExplainer(model)` computes exact SHAP values | 💻 Your Laptop |
| 15.3 | For other models: `shap.Explainer` with Independent masker runs | 💻 Your Laptop |
| 15.4 | Top-N feature contributions formatted as text (⬆️ pushes UP / ⬇️ pushes DOWN) | 💻 Your Laptop |
| 15.5 | SHAP waterfall plot saved as PNG | 💻 Your Laptop |
| 15.6 | Text explanation sent to you | 💻 → 📱 Telegram |
| 15.7 | Waterfall plot image sent to you | 💻 → 📱 Telegram |
| *(fallback)* | If SHAP fails for any reason, it is silently skipped — prediction still delivered | 💻 Your Laptop |

---

## 🗺️ Full Flow Summary

```
You (Telegram App)
      │
      │  Upload CSV
      ▼
Telegram Servers ──────────────────────────────────── stores file, notifies bot
      │
      │  Update pushed to bot
      ▼
💻 Your Laptop (bot/handlers.py)
      │
      ├── Download CSV from Telegram          [💻 ← 📱]
      ├── Load with pandas                    [💻]
      ├── Build dataset summary               [💻]
      │
      ├── Send summary to Groq                [💻 → 🤖]
      │       Groq analyzes, returns JSON     [🤖 → 💻]
      │
      ├── Send model keyboard to user         [💻 → 📱]
      │       User toggles + confirms         [📱 → 💻]
      │
      ├── Preprocess (impute/encode/scale)    [💻]
      ├── Feature selection (if needed)       [💻]
      ├── Train models + 5-fold CV            [💻]
      │
      ├── (if CV < 0.60) Retry via Groq       [💻 → 🤖 → 💻]
      │
      ├── Ask user about tuning               [💻 → 📱]
      │       User replies                    [📱 → 💻]
      ├── (if yes) Hyperparameter tuning      [💻]
      │
      ├── Evaluate all models                 [💻]
      ├── Generate 8 plots as PNGs            [💻]
      ├── Build PDF report                    [💻]
      ├── Create deployment ZIP               [💻]
      │
      ├── Send PDF to user                    [💻 → 📱]
      ├── Send ZIP to user                    [💻 → 📱]
      ├── Send text summary to user           [💻 → 📱]
      │
      └── Enter INFERENCE MODE
              │
              │  User types query              [📱 → 💻]
              ├── Send query to Groq           [💻 → 🤖 → 💻]
              ├── Encode + Scale + Predict     [💻]
              ├── SHAP explanation             [💻]
              └── Send result + SHAP plot      [💻 → 📱]
```

---

## 📊 Where Does the Heavy Work Happen?

| Task | Location | Approx Time |
|------|----------|-------------|
| File download | Telegram → Laptop | 1–5 sec |
| Dataset analysis (LLM) | Groq Cloud | 5–15 sec |
| Preprocessing | Laptop | < 5 sec |
| Feature selection | Laptop | < 10 sec |
| Model training (5-fold CV) | Laptop | 30 sec – 3 min |
| Auto-retry analysis (LLM) | Groq Cloud | 5–15 sec |
| Hyperparameter tuning | Laptop | 1–5 min |
| Evaluation | Laptop | < 5 sec |
| Plot generation (8 plots) | Laptop | 5–15 sec |
| PDF building | Laptop | 5–10 sec |
| Deployment zip creation | Laptop | < 5 sec |
| Inference parsing (LLM) | Groq Cloud | 2–5 sec |
| Prediction + SHAP | Laptop | 1–10 sec |

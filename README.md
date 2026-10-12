# ML Pipeline Telegram Bot 🤖

An AI-powered Telegram bot that receives a labeled CSV dataset, analyses it using **Groq AI (llama-3.3-70b-versatile)**, runs a full automated machine learning pipeline, and delivers a comprehensive **PDF report**, **visual plots**, a **deployment-ready package**, and a **live prediction mode** — all inside Telegram.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Tech Stack](#architecture--tech-stack)
3. [Full Feature List](#full-feature-list)
4. [How It Works — Step by Step](#how-it-works--step-by-step)
5. [ML Pipeline Details](#ml-pipeline-details)
   - [Step 1: AI Dataset Analysis (Groq)](#step-1-ai-dataset-analysis-groq)
   - [Step 2: Preprocessing](#step-2-preprocessing)
   - [Step 3: Feature Selection](#step-3-feature-selection)
   - [Step 4: Model Training & Cross-Validation](#step-4-model-training--cross-validation)
   - [Step 5: Auto-Retry with Groq](#step-5-auto-retry-with-groq)
   - [Step 6: Hyperparameter Tuning](#step-6-hyperparameter-tuning)
   - [Step 7: Evaluation](#step-7-evaluation)
   - [Step 8: Plot Generation](#step-8-plot-generation)
   - [Step 9: PDF Report](#step-9-pdf-report)
   - [Step 10: Deployment Package Export](#step-10-deployment-package-export)
   - [Step 11: Live Prediction Mode](#step-11-live-prediction-mode)
   - [Step 12: SHAP Explainability](#step-12-shap-explainability)
6. [Supported Models](#supported-models)
7. [Project Structure](#project-structure)
8. [Setup & Installation](#setup--installation)
9. [Environment Variables](#environment-variables)
10. [Usage Guide](#usage-guide)
11. [Key Design Decisions](#key-design-decisions)
12. [Known Limitations](#known-limitations)

---

## Project Overview

This project automates the entire ML lifecycle for any labeled CSV dataset. A user simply sends a CSV file to the Telegram bot. The bot then:

1. **Analyses** the dataset using a Groq-hosted LLM to understand its structure
2. **Presents** a model selection menu where the user can toggle which algorithms to train
3. **Preprocesses** the data (imputation, encoding, scaling, imbalance handling)
4. **Selects features** if the dimensionality is high
5. **Trains & compares** multiple ML models using 5-fold cross-validation
6. **Auto-retries** with a different Groq-suggested strategy if the score is poor
7. **Optionally tunes** hyperparameters of the best model via `RandomizedSearchCV`
8. **Evaluates** the final model and generates all diagnostic plots
9. **Sends** a multi-page dark-themed PDF report with all metrics and embedded plots
10. **Sends** a zip deployment package with `predict.py`, a Gradio `app.py`, and all model artifacts
11. **Enters Live Prediction Mode** where the user can type natural-language queries or use a step-by-step wizard to get real-time predictions with SHAP explanations

---

## Architecture & Tech Stack

| Layer | Technology |
|-------|------------|
| Bot Framework | `python-telegram-bot` v21 (`ConversationHandler`, async polling) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| ML Core | `scikit-learn`, `xgboost`, `lightgbm` |
| Imbalance Handling | `imbalanced-learn` (SMOTE, oversampling, undersampling) |
| Data Processing | `pandas`, `numpy` |
| Plotting | `matplotlib`, `seaborn` |
| PDF Generation | `reportlab` |
| Explainability | `shap` (TreeExplainer + KernelExplainer fallback) |
| Deployment Export | `pickle`, `zipfile`, `gradio` (generated `app.py`) |
| Async Architecture | `asyncio.to_thread()` — all blocking ML calls run in thread pools |
| Config | `python-dotenv` |

### Why asyncio.to_thread?

All heavy ML operations (Groq API calls, pandas processing, sklearn training) are blocking. Running them directly in an `async` Telegram handler would block the entire event loop, causing Telegram `TimedOut` errors. Every pipeline step is wrapped in `asyncio.to_thread()` to offload blocking work to a thread pool while keeping the event loop free for other updates.

---

## Full Feature List

- 🧠 **Groq AI Dataset Analysis** — LLM inspects column statistics, dtypes, correlations, and missing values to decide the optimal preprocessing strategy and models
- 🎛️ **Interactive Model Selection** — Inline keyboard with toggle buttons; AI-recommended models are starred (⭐)
- ⚙️ **Smart Preprocessing** — Missing value imputation (median/mode), `LabelEncoder` for ordinal/high-cardinality categoricals, `StandardScaler` / `MinMaxScaler` / `RobustScaler` (AI-chosen), SMOTE / random over/under-sampling for class imbalance
- 🔍 **Feature Selection** — Applied only when `n_features > 15`; supports `SelectKBest` (ANOVA F-test for classification, f_regression for regression), `RFE` (Recursive Feature Elimination with a Random Forest estimator), or `PCA`
- 🚀 **Automated Model Training** — Trains all user-selected models with 5-fold stratified cross-validation; scores are collected and compared
- 🔄 **Auto-Retry Loop** — If the best CV score < 0.60, the bot re-prompts Groq with a description of what failed and asks for a different strategy; re-preprocesses and retrains; only adopts the retry if it actually improves things
- ⚡ **Hyperparameter Tuning** — User-prompted; runs `RandomizedSearchCV` (50 iterations) on the best model from cross-validation
- 📊 **Comprehensive Evaluation** — Classification: Accuracy, Precision (macro), Recall (macro), F1 (macro), ROC-AUC, Confusion Matrix, per-model test set metrics. Regression: MAE, MSE, RMSE, R²
- 📈 **8 Diagnostic Plots** — Class distribution (before/after balancing), Confusion Matrix heatmap, ROC curves (all models), Feature importance (top 20), Model comparison bar chart (CV vs Test), Correlation heatmap, Residuals analysis (regression), Predicted vs Actual scatter (regression)
- 📄 **Dark-themed PDF Report** — Multi-page `reportlab` PDF with a cover page, decisions table, preprocessing summary, feature selection summary, per-model metrics table, best model section, and all embedded plots
- 📦 **Deployment Package** — Auto-generated `.zip` containing: `best_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `metadata.json`, `predict.py` (command-line prediction script), `app.py` (Gradio web UI)
- 🎯 **Live Predictor Mode** — Natural language input parsed by Groq LLM into a feature dict, scaled and fed to the trained model, result returned with confidence probabilities
- 🧙 **Wizard Mode (`/predict`)** — Step-by-step feature input with type hints and valid options/ranges for each feature
- 🔍 **SHAP Explainability** — After every prediction, returns top-N feature contributions in plain text + a waterfall plot image (uses `TreeExplainer` for tree models, `KernelExplainer` fallback for linear/SVM)

---

## How It Works — Step by Step

```
User sends CSV
      │
      ▼
[handle_document] ← downloads file, reads with pandas, validates shape
      │
      ▼
[analyze_dataset] ← Groq LLM call (asyncio.to_thread)
      │  Returns: task_type, target_column, scaler_type,
      │           handle_imbalance, feature_selection_method, recommended_models
      │
      ▼
[Inline Keyboard] ← user toggles models, presses Confirm
      │
      ▼
[_run_pipeline]
      │
      ├── [preprocessor.preprocess]      → X, y, preprocess_meta
      │      • imputation, encoding, scaling, SMOTE/balance
      │
      ├── [feature_selector.select_features]  → X (reduced), fs_meta
      │      • SelectKBest / RFE / PCA (only if n_features > 15)
      │
      ├── [model_trainer.train_models]   → train_result
      │      • 5-fold CV, collects cv_scores per model
      │      • picks best model by mean CV score
      │
      ├── [Auto-Retry if CV < 0.60]
      │      • retry_analyze_dataset (Groq) → new strategy
      │      • re-preprocess + re-train, adopt if better
      │
      ├── [User: Tune Yes/No?]
      │      • tune_yes → model_trainer.tune_model (RandomizedSearchCV 50 iter)
      │
      ├── [evaluator.evaluate]           → metrics dict
      │      • train/test split from train_result
      │      • computes all metrics for all trained models
      │
      ├── [plotter.generate_all_plots]   → list of PNG paths
      │
      ├── [build_pdf]                    → PDF path
      │
      ├── [export_helper.save_and_zip]   → ZIP path
      │      • pickles model, scaler, encoders
      │      • generates predict.py & app.py
      │
      └── [INFERENCE_MODE]
             • Natural language → Groq parses → feature dict → model.predict
             • /predict → Wizard mode (step-by-step)
             • SHAP explanation after every prediction
```

---

## ML Pipeline Details

### Step 1: AI Dataset Analysis (Groq)

**File:** `pipeline/dataset_analyzer.py`

The bot builds a rich JSON summary of the dataset including:
- Shape, column dtypes, missing value counts
- Numeric stats (mean, std, min, max per column)
- Top-5 value counts for categorical columns
- Top-10 Pearson correlations between numeric pairs

This JSON is sent to Groq's `llama-3.3-70b-versatile` model with a strict prompt asking for a JSON-only response containing:

```json
{
  "task_type": "classification" | "regression",
  "target_column": "<last column, always>",
  "recommended_preprocessing": ["imputation", "label_encoding", "standardization"],
  "needs_feature_selection": false,
  "feature_selection_method": null,
  "recommended_models": ["RandomForest", "XGBoost", "LogisticRegression"],
  "scaler_type": "standard" | "minmax" | "robust",
  "handle_imbalance": false,
  "imbalance_method": "smote" | "oversample" | "undersample" | null,
  "reasoning": "Brief explanation"
}
```

**Safety overrides applied after LLM response:**
- Target column is always forced to `df.columns[-1]` (avoids LLM hallucination)
- `task_type` is auto-corrected based on actual target column dtype (object/category → classification; int with ≤20 unique values → classification)
- Model names are filtered against known registry sets to prevent invalid models
- Feature selection is forced off if `n_features ≤ 15`

---

### Step 2: Preprocessing

**File:** `pipeline/preprocessor.py`

The `preprocess(df, decisions)` function returns `(X, y, preprocess_meta)`.

Steps applied in order:
1. **Separate** features `X` and target `y`
2. **Missing value imputation** — numeric columns → median; categorical columns → mode
3. **Label encoding** — for columns with dtype `object` or `category`, fits a `LabelEncoder` per column. Encoders stored in `preprocess_meta["label_encoders"]` for inference reuse
4. **Target encoding** — if `y` is non-numeric (classification), a separate `LabelEncoder` is fitted on `y` and stored in `preprocess_meta["target_encoder"]`
5. **Scaling** — applies `StandardScaler`, `MinMaxScaler`, or `RobustScaler` based on `decisions["scaler_type"]`; scaler stored in `preprocess_meta["scaler"]`
6. **Imbalance handling** — if `decisions["handle_imbalance"]` is True:
   - `smote` → `SMOTE(random_state=42)`
   - `oversample` → `RandomOverSampler(random_state=42)`
   - `undersample` → `RandomUnderSampler(random_state=42)`

`preprocess_meta` keys: `scaler`, `label_encoders`, `target_encoder`, `imbalance_handled`, `feature_names`

---

### Step 3: Feature Selection

**File:** `pipeline/feature_selector.py`

Applied only if `decisions["needs_feature_selection"] == True` (i.e., original feature count > 15).

| Method | Implementation | Notes |
|--------|---------------|-------|
| `selectkbest` | `SelectKBest(f_classif)` for classification, `SelectKBest(f_regression)` for regression | Selects top `min(k, n_features)` features |
| `rfe` | `RFE(estimator=RandomForestClassifier/Regressor, n_features_to_select=k)` | Recursive elimination |
| `pca` | `PCA(n_components=k)` | Returns PC components; feature names become `PC1, PC2, ...` |

Returns `(X_reduced, fs_meta)` where `fs_meta` contains the fitted selector object for later reference.

---

### Step 4: Model Training & Cross-Validation

**File:** `pipeline/model_trainer.py`

`train_models(X, y, decisions, selected_models)` — performs:

1. **Train/test split** — `80/20`, `stratify=y` for classification
2. **Cross-validation** — `StratifiedKFold(n_splits=5)` for classification, `KFold(n_splits=5)` for regression; scoring is `accuracy` / `r2`
3. **Fits each selected model** on the full training set after CV
4. Collects per-model: CV mean, CV std, fitted model object, test predictions

**Returns `train_result` dict:**
```python
{
  "best_model_name": str,
  "best_cv_score": float,
  "best_model": fitted_model,
  "all_models": {name: fitted_model},
  "X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...,
  "cv_results": {name: {"mean": float, "std": float}},
  "feature_names": list[str],
  "task_type": str,
}
```

#### Classification Models
| Name | Class |
|------|-------|
| LogisticRegression | `LogisticRegression(max_iter=1000)` |
| RandomForest | `RandomForestClassifier(n_estimators=100)` |
| XGBoost | `XGBClassifier(eval_metric='logloss', verbosity=0)` |
| LightGBM | `LGBMClassifier(verbose=-1)` |
| SVM | `SVC(probability=True)` |
| KNN | `KNeighborsClassifier()` |
| GradientBoosting | `GradientBoostingClassifier()` |
| DecisionTree | `DecisionTreeClassifier()` |

#### Regression Models
| Name | Class |
|------|-------|
| LinearRegression | `LinearRegression()` |
| Ridge | `Ridge()` |
| Lasso | `Lasso()` |
| RandomForestRegressor | `RandomForestRegressor(n_estimators=100)` |
| XGBoostRegressor | `XGBRegressor(verbosity=0)` |
| GradientBoostingRegressor | `GradientBoostingRegressor()` |
| SVR | `SVR()` |

---

### Step 5: Auto-Retry with Groq

If `best_cv_score < 0.60` (configurable via `config.RETRY_THRESHOLD`):

1. Calls `retry_analyze_dataset(df, previous_decisions, previous_cv_score)` — sends a different prompt to Groq describing the failed attempt and asking for an improved strategy
2. Re-preprocesses with new decisions
3. Re-trains with newly recommended models
4. Only adopts retry result if `retry_cv_score > original_cv_score`
5. Notifies the user either way

---

### Step 6: Hyperparameter Tuning

`model_trainer.tune_model(train_result, decisions)` — uses `RandomizedSearchCV`:

- **50 iterations**, 5-fold CV
- **Classification param grids** (example for RandomForest): `n_estimators [50,100,200]`, `max_depth [None,5,10,15]`, `min_samples_split [2,5,10]`, `min_samples_leaf [1,2,4]`
- **Regression param grids** similarly defined per model type
- Stores tuned model as `train_result["tuned_model"]`
- Reports tuned CV score vs baseline CV score to the user

---

### Step 7: Evaluation

**File:** `pipeline/evaluator.py`

`evaluate(train_result)` — uses `X_test` and `y_test` from `train_result`:

**Classification metrics (per model):**
- Accuracy, Precision (macro avg), Recall (macro avg), F1 (macro avg)
- ROC-AUC (binary: direct; multiclass: `ovr` average)
- `y_proba` (predict_proba output for ROC curve plotting)
- Confusion matrix for the best model

**Regression metrics (per model):**
- MAE, MSE, RMSE, R²
- `y_pred`, residuals stored for plotting

**Returns `metrics` dict:**
```python
{
  "task_type": str,
  "y_test": np.ndarray,
  "y_pred": np.ndarray,           # best model predictions
  "confusion_matrix": np.ndarray, # classification only
  "all_metrics": {model_name: {accuracy/r2/...}},
  "cv_results": {model_name: {mean, std}},
}
```

---

### Step 8: Plot Generation

**File:** `pipeline/plotter.py`

All plots use a clean white/light theme with a blue accent (`#1a56db`). Values are printed on top of all bar charts.

| Plot | Trigger | Description |
|------|---------|-------------|
| `class_distribution.png` | Classification | Bar chart of class counts before (and after if balanced) |
| `confusion_matrix.png` | Classification | Seaborn heatmap with class labels |
| `roc_curve.png` | Classification | ROC curves for all models (binary) or all classes of best model (multiclass) |
| `feature_importance.png` | Any (if model has importances) | Top-20 features; uses `feature_importances_` or `coef_` |
| `model_comparison.png` | Any | Grouped bar: CV mean ± std vs Test set score for each model |
| `correlation_heatmap.png` | Any (≥2 numeric features) | Pearson correlation heatmap; annotated if ≤15 features |
| `residuals_plot.png` | Regression | Residuals vs Predicted scatter + Residuals histogram |
| `predicted_vs_actual.png` | Regression | Scatter of actual vs predicted for all models |

---

### Step 9: PDF Report

**File:** `utils/pdf_reporter.py`

Built with `reportlab`. Dark-themed multi-page document:

1. **Cover page** — Dataset name, task type, best model name, timestamp
2. **AI Decisions** — Table of all Groq decisions (scaler, imbalance method, etc.)
3. **Preprocessing Summary** — What was applied and what was stored
4. **Feature Selection Summary** (if applicable)
5. **Model Training Results** — Table of CV scores for all models
6. **Best Model Metrics** — Full metrics table for the winning model
7. **All Plots** — Each PNG embedded at full width with a caption

---

### Step 10: Deployment Package Export

**File:** `utils/export_helper.py`

`save_and_zip(output_dir, filename, train_result, preprocess_meta, decisions, df)` creates a `.zip` containing:

| File | Description |
|------|-------------|
| `best_model.pkl` | Pickled trained model (tuned if available, else baseline) |
| `scaler.pkl` | Pickled fitted scaler |
| `label_encoders.pkl` | Dict of fitted `LabelEncoder` objects (one per categorical feature) |
| `metadata.json` | `task_type`, `target_column`, `feature_names`, `model_name`, `best_cv_score` |
| `predict.py` | CLI script: `python predict.py --input "col1=val1, col2=val2"` |
| `app.py` | Gradio web UI: numeric sliders, categorical dropdowns, `python app.py` |

The generated scripts load the pickled artifacts, apply the same `LabelEncoder` → scaler pipeline, and call `model.predict()`.

---

### Step 11: Live Prediction Mode

After the pipeline completes, the bot enters `INFERENCE_MODE`. The user can:

**Free-text mode:** Type anything like `"age 35, salary 60000, department Engineering"`. The bot sends this to Groq with a prompt listing all feature names and asks it to extract a JSON dict. The dict is then encoded, scaled, and fed to the model.

**Wizard mode (`/predict`):** The bot asks for each feature one at a time, showing the data type, valid options, and value ranges. After all features are collected, it runs the same prediction pipeline.

**Response includes:**
- Predicted value/class
- Confidence probabilities (for classification models that support `predict_proba`)
- Parsed input summary
- SHAP explanation (text + waterfall plot)

---

### Step 12: SHAP Explainability

**File:** `utils/shap_explainer.py`

After every prediction, `explain_prediction(model, sample_df, feature_names, output_dir)` is called:

- For tree-based models (XGBoost, LightGBM, RandomForest, GradientBoosting, DecisionTree): uses `shap.TreeExplainer` — fast and exact
- For others (Logistic Regression, SVM, KNN): uses `shap.Explainer` with an `Independent` masker (background = zeros, since data is already scaled to mean ≈ 0)
- Returns a Telegram-formatted text listing top-N features with `⬆️ pushes UP` / `⬇️ pushes DOWN` labels and SHAP values
- Saves and sends a SHAP waterfall plot image

---

## Supported Models

### Classification
`LogisticRegression`, `RandomForest`, `XGBoost`, `LightGBM`, `SVM`, `KNN`, `GradientBoosting`, `DecisionTree`

### Regression
`LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `XGBoostRegressor`, `GradientBoostingRegressor`, `SVR`

---

## Project Structure

```
├── main.py                      # Bot entry point; ConversationHandler setup; IPv4 fix; retry loop
├── config.py                    # Loads .env; TEMP_DIR; CV_FOLDS; FEATURE_SELECTION_THRESHOLD
├── requirements.txt
├── setup.bat                    # Creates venv + installs packages
├── .env                         # TELEGRAM_BOT_TOKEN, GROQ_API_KEY (not committed)
├── .env.example                 # Template
│
├── bot/
│   ├── __init__.py
│   └── handlers.py              # All Telegram handlers; full pipeline orchestration
│
├── pipeline/
│   ├── __init__.py
│   ├── dataset_analyzer.py      # Groq LLM analysis + retry analysis
│   ├── preprocessor.py          # Imputation, encoding, scaling, SMOTE
│   ├── feature_selector.py      # SelectKBest / RFE / PCA
│   ├── model_trainer.py         # CV training, model registry, hyperparameter tuning
│   ├── evaluator.py             # Metrics computation for all models
│   └── plotter.py               # 8 diagnostic plot functions + orchestrator
│
└── utils/
    ├── __init__.py
    ├── helpers.py               # Session dir management, temp file cleanup, message formatting
    ├── pdf_reporter.py          # Reportlab PDF builder
    ├── export_helper.py         # Pickle, zip, generate predict.py & app.py
    └── shap_explainer.py        # SHAP TreeExplainer / KernelExplainer + waterfall plot
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- A Telegram bot token (create via [@BotFather](https://core.telegram.org/bots/tutorial))
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Quick Setup

```bat
:: Clone/download the project, then:
setup.bat
```

This script creates a virtual environment and installs all dependencies.

### Manual Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```powershell
copy .env.example .env
# Edit .env and fill in your keys
```

### Run

```powershell
venv\Scripts\activate
python main.py
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | ✅ Yes |
| `GROQ_API_KEY` | Groq API key for LLM calls | ✅ Yes |

---

## Usage Guide

1. Start the bot: send `/start` in Telegram
2. Upload any labeled `.csv` file (max 20 MB — Telegram limit)
3. Wait for AI analysis (~15 seconds)
4. **Toggle models** using the inline keyboard; AI recommendations are starred ⭐
5. Press **Confirm & Run Pipeline**
6. Optionally choose **Yes** to tune hyperparameters
7. Receive:
   - 📊 Text summary with all metrics
   - 📄 PDF report
   - 📦 Deployment zip
8. Enter **Live Predictor Mode**:
   - Type natural language like `"age 30, income 50000, education Masters"`
   - Or type `/predict` for step-by-step Wizard mode
   - Type `/exit` to stop

### Dataset Requirements
- Must be a `.csv` file with a header row
- Must have at least 2 columns
- **Target column must be the last column**
- Supports both classification (categorical/integer label) and regression (float label)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| All ML calls in `asyncio.to_thread()` | Prevents blocking the Telegram event loop; avoids `TimedOut` errors |
| IPv4 monkey-patch on `socket.getaddrinfo` | Windows asyncio does not gracefully fall back from IPv6 to IPv4; the patch forces IPv4 globally at startup |
| Retry loop on `app.run_polling(close_loop=False)` | Allows the bot to survive temporary network outages at startup without crashing |
| Target column always = last column | Removes ambiguity in LLM target detection; predictable behavior for any dataset |
| Auto-correct `task_type` from dtype | Prevents LLM from misclassifying regression as classification based on column name alone |
| SHAP is optional (silent failure) | SHAP can fail on edge-case models or data; wrapping in try/except ensures predictions always succeed |
| Persistent `output_dir` per dataset | Reports, plots, and zips are stored by dataset name (not session), so they survive cleanup |
| Session-scoped `tmp` dir | CSVs and intermediate files are cleaned up per session to avoid disk bloat |

---

## Known Limitations

1. **PCA feature selection breaks inference** — PCA transforms features into principal components (`PC1, PC2, …`), which cannot be input by users in natural language. The LLM is instructed to prefer `selectkbest` or `rfe`, but if it recommends PCA, live prediction will have meaningless feature names.

2. **SHAP KernelExplainer is slow** — For non-tree models (SVM, KNN), SHAP uses a background-based explainer which can take 10–30 seconds. It is silently skipped if it exceeds runtime.

3. **20 MB file size limit** — Hard limit imposed by Telegram's Bot API for document uploads.

4. **Gradio app.py requires `gradio` installed** — The deployment package generates a `app.py` but `gradio` is not bundled; the user must `pip install gradio` separately.

5. **Single user session per chat** — The bot stores model artifacts in `context.user_data` which is per-user/per-chat. Uploading a new CSV during inference mode replaces the previous session.

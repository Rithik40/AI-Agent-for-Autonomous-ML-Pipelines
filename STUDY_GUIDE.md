# 📖 Understanding Your ML Pipeline Bot — A Personal Study Guide

This guide is written **for you** — to help you understand exactly what your project does, how each piece works, and why it was built that way. Read this before your interview.

---

## 🤔 What Is This Project, In Simple Terms?

Imagine you have a dataset — say, a CSV file of patient records where the last column says whether they have a disease or not. Normally, to build an ML model from this, you'd have to:

1. Open the file in Jupyter, look at the data
2. Clean it (fix missing values, encode text columns)
3. Choose a scaler, decide if the classes are imbalanced
4. Write code to train 5-10 models, compare them
5. Plot confusion matrices, ROC curves, feature importance charts
6. Export the final model
7. Write a script so someone else can use it

**That's hours of work for every new dataset.**

Your bot does ALL of that automatically — inside Telegram. You send a CSV file, and the bot sends back a PDF report, plots, a deployment zip, and even lets you type questions like *"what would the prediction be for a 35-year-old male with salary 60000?"* and get a real answer.

The secret ingredient is **Groq AI** — a free, fast LLM API that the bot uses to *read the dataset* and *decide* what kind of preprocessing and models to use, instead of you having to figure that out manually.

---

## 🧠 The Big Picture — How the Bot Thinks

When you send a CSV, here's what the bot does internally (in plain English):

```
You: [sends CSV file]

Bot thinks:
  1. "Let me download this file and load it with pandas."
  2. "Let me look at the columns, dtypes, missing values, correlations..."
  3. "Let me ask Groq AI: what kind of problem is this? Which models should I use?"
  4. "Groq says: classification, use RandomForest + XGBoost, apply SMOTE, use StandardScaler."
  5. "Let me show the user a keyboard to pick which models they want."

You: [picks models, clicks Confirm]

Bot thinks:
  6. "Clean the data — fill missing values, encode text columns, scale numbers."
  7. "Train each model the user picked using 5-fold cross-validation."
  8. "The best model got 0.55 CV score — that's below 0.60. Ask Groq for a retry."
  9. "Groq says: try GradientBoosting with RobustScaler instead."
  10. "Retrain. New score: 0.72. That's better — use these results."

You: [clicks Yes to hyperparameter tuning]

Bot thinks:
  11. "Run RandomizedSearchCV on the best model for 50 iterations."
  12. "New tuned score: 0.76. "
  13. "Evaluate on the test set — compute accuracy, F1, ROC-AUC, confusion matrix."
  14. "Generate all the plots — confusion matrix, ROC curve, feature importance, etc."
  15. "Build a PDF report with all of this."
  16. "Pickle the model, generate predict.py and app.py, zip it all up."
  17. "Send everything to the user."
  18. "Switch to Live Prediction Mode — wait for the user to type queries."

You: "age 30, salary 50000, department IT"

Bot thinks:
  19. "Send this text to Groq and ask it to extract a JSON dict of feature values."
  20. "Groq returns: {'age': 30, 'salary': 50000, 'department': 'IT'}"
  21. "Encode 'IT' with the label encoder. Scale the numbers with the scaler."
  22. "Run model.predict(). Result: 1 (disease positive)."
  23. "Run SHAP to explain WHY — which features pushed the prediction."
  24. "Send result + explanation + a SHAP waterfall plot image."
```

That's the entire flow. Every step.

---

## 📁 Project Files — What Each One Does

### `main.py` — The Front Door
This is where the bot starts. It does three things:
1. **IPv4 patch** — patches Python's DNS resolver to only use IPv4, because your network blocks IPv6 connections to Telegram (this is why the bot was crashing before)
2. **Sets up the ConversationHandler** — tells the bot which function to call based on the current "state" (waiting for CSV? waiting for model selection? in inference mode?)
3. **Starts polling** — tells Telegram to keep checking for new messages and pass them to the right handler

Think of `main.py` as the traffic controller. It doesn't do any ML work — it just routes messages.

### `config.py` — Settings
Loads your `.env` file and defines global constants:
- `TELEGRAM_BOT_TOKEN` — the token to authenticate your bot
- `GROQ_API_KEY` — the key to call Groq's LLM
- `TEMP_DIR` — where temporary files (downloaded CSVs, etc.) are stored
- `FEATURE_SELECTION_THRESHOLD = 15` — only apply feature selection if the dataset has more than 15 features
- `CV_FOLDS = 5` — use 5-fold cross-validation

### `bot/handlers.py` — The Brain
This is the largest and most important file. It contains all the Telegram handler functions. Here's what each one does:

| Function | Triggered When | What It Does |
|----------|---------------|--------------|
| `cmd_start` | User sends `/start` | Sends a welcome message explaining how to use the bot |
| `cmd_help` | User sends `/help` | Sends usage instructions |
| `cmd_exit` | User sends `/exit` | Exits inference/prediction mode |
| `handle_document` | User uploads a CSV | Downloads it, loads with pandas, calls Groq analysis, shows model selection keyboard |
| `handle_model_selection` | User clicks a keyboard button | Toggles model selection, or confirms and runs the pipeline |
| `_run_pipeline` | After user confirms models | Steps 2-5: preprocessing → feature selection → training → asks about tuning |
| `handle_tuning_decision` | User clicks Yes/No on tuning | Steps 5+: tunes (if yes), evaluates, plots, builds PDF, sends everything, enters inference mode |
| `handle_inference_chat` | User sends text in inference mode | Calls Groq to parse it → encodes → scales → predicts → SHAP |
| `handle_wizard_prediction` | User in `/predict` wizard mode | Collects one feature at a time, then runs prediction |
| `handle_error` | Any uncaught error | Sends error message to user |

### `pipeline/dataset_analyzer.py` — The AI Analyst
This file talks to Groq. It:
1. Builds a summary of your dataset (shape, dtypes, missing values, top correlations)
2. Sends that summary to Groq with a prompt like *"You are an expert data scientist. Here's the dataset. Tell me what kind of problem this is and what pipeline to use. Return only JSON."*
3. Parses the JSON response
4. **Applies safety checks** — overrides the LLM's target column choice (always uses the last column), fixes wrong model names, turns off feature selection if n_features is small

It also has `retry_analyze_dataset()` — called when the CV score is bad. Same idea but the prompt says *"the previous attempt with these settings got a poor score. Suggest something different."*

### `pipeline/preprocessor.py` — Data Cleaning
Takes the raw dataframe and returns clean `X` (features) and `y` (target). Steps:
1. Separate `X` and `y` (target = last column)
2. Fill missing values (numbers → median, text → most common value)
3. Label-encode text columns (e.g., "Male"→0, "Female"→1). Saves the encoder so inference uses the same mapping
4. Label-encode `y` if it's text (e.g., "Yes"→1, "No"→0). Saves this encoder too
5. Scale features with the scaler Groq recommended (StandardScaler / MinMaxScaler / RobustScaler)
6. Handle class imbalance with SMOTE or oversampling/undersampling if Groq said to

**Why save the encoders and scaler?** Because when the user later types a prediction query, you need to apply the exact same transformations — otherwise the model will get garbage input.

### `pipeline/feature_selector.py` — Dimensionality Reduction
Only runs if the dataset has more than 15 features. Three options:
- **SelectKBest** — picks the top K features based on statistical tests (ANOVA F-score for classification, F-regression for regression)
- **RFE** — trains a Random Forest and recursively removes the least important features
- **PCA** — mathematically compresses features into fewer "principal components" (note: these don't have meaningful names, which is why natural-language inference is tricky with PCA)

### `pipeline/model_trainer.py` — Training
`train_models()` does this:
1. Splits data 80/20 (train/test)
2. For each model the user selected:
   - Run 5-fold cross-validation, collect CV scores
   - Fit the model on the full training set
3. Pick the best model by mean CV score
4. Return everything: fitted models, CV results, train/test splits

`tune_model()` takes the best model and runs `RandomizedSearchCV` with 50 random hyperparameter combinations to find the best settings.

### `pipeline/evaluator.py` — Measuring Performance
Uses the `X_test` and `y_test` stored in `train_result` (the 20% held-out set that was never used in training). Computes:
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC, per-class breakdown, confusion matrix
- **Regression**: MAE (Mean Absolute Error), MSE, RMSE, R² score

Runs metrics for every trained model so the comparison plot can show all of them.

### `pipeline/plotter.py` — Visualizations
Makes 8 different plots as PNG files and saves them to a folder. The plots are:

| Plot | What It Shows |
|------|---------------|
| Class Distribution | How many samples per class (before and after balancing if SMOTE was used) |
| Confusion Matrix | How many predictions were right/wrong for each class pair |
| ROC Curve | Trade-off between true positives and false positives (higher AUC = better) |
| Feature Importance | Which features influenced the model most |
| Model Comparison | Bar chart comparing all models' CV score vs test set score |
| Correlation Heatmap | Which features are correlated with each other (helps spot redundant features) |
| Residuals Plot | For regression — how far off predictions are and if errors are random |
| Predicted vs Actual | For regression — scatter plot of real values vs model predictions |

### `utils/pdf_reporter.py` — The PDF
Uses `reportlab` to build a multi-page PDF report. Contains:
- Cover page with dataset name, task type, best model, timestamp
- AI decisions table (what Groq recommended)
- Preprocessing summary
- Feature selection info
- Per-model metrics table
- Best model detailed metrics
- All 8 plots embedded as images

### `utils/export_helper.py` — Deployment Package
`save_and_zip()` creates a ZIP file with everything needed to use the model outside the bot:
- `best_model.pkl` — the trained model (pickled)
- `scaler.pkl` — the fitted scaler
- `label_encoders.pkl` — the fitted encoders for each column
- `metadata.json` — feature names, task type, model name, CV score
- `predict.py` — a command-line script to make predictions
- `app.py` — a Gradio web interface with sliders and dropdowns; run `python app.py` to get a browser UI

### `utils/shap_explainer.py` — Why Did the Model Predict That?
SHAP (SHapley Additive exPlanations) is a technique from game theory that explains model predictions by figuring out how much each feature "contributed" to the final prediction.

For example: *"age pushed the prediction UP by 0.23, salary pushed it DOWN by 0.11, department had no effect."*

Two modes:
- **TreeExplainer** — fast, exact, works for RandomForest/XGBoost/LightGBM/GradientBoosting
- **KernelExplainer** — slower, approximate, fallback for LogisticRegression/SVM/KNN

Returns a text message and a waterfall plot image. If it fails for any reason, it silently skips (the prediction still works).

---

## 🔄 Conversation States — How the Bot Remembers Where It Is

Telegram bots are stateless by default — every message is independent. The `ConversationHandler` adds state management. Your bot has 4 states:

```
[No state]         → User uploads CSV → [SELECTING_MODELS]
[SELECTING_MODELS] → User confirms    → [WANT_TUNING]
[WANT_TUNING]      → User clicks      → [INFERENCE_MODE]
[INFERENCE_MODE]   → User types /predict → [WIZARD_PREDICTION]
[Any state]        → User types /exit → [No state / End]
```

User data (the dataframe, the model, the encoders, etc.) is stored in `context.user_data` — a dictionary that persists for the entire conversation per user.

---

## 🔑 Key Concepts to Understand for the Interview

### What is Cross-Validation and Why Use It?
Cross-validation splits the training data into K equal parts (K=5 in your case). It trains the model K times, each time using a different part as validation. The average score across all K runs is the CV score.

**Why?** Because a single train/test split can be lucky or unlucky depending on which data ended up in which split. CV gives a much more reliable estimate of how the model will perform on unseen data.

### What is SMOTE?
Synthetic Minority Over-sampling Technique. If your dataset has 90% class A and 10% class B, the model will be biased toward always predicting class A. SMOTE creates synthetic new samples for the minority class by interpolating between existing minority samples, bringing the classes into balance.

### What is StandardScaler?
It transforms each feature so it has mean=0 and standard deviation=1. This is important for models like Logistic Regression, SVM, and KNN that are sensitive to the scale of features. Tree-based models (RandomForest, XGBoost) don't need scaling but it doesn't hurt them.

### What is RandomizedSearchCV?
Instead of trying every possible combination of hyperparameters (GridSearchCV, which is exponentially slow), RandomizedSearchCV tries N random combinations from the defined ranges. With 50 iterations and 5-fold CV, it runs 250 model fits total to find the best hyperparameters.

### What is a Hyperparameter?
A setting of the model that is NOT learned from data — it's set before training. Examples:
- `n_estimators=100` in RandomForest (how many trees to build)
- `max_depth=5` (how deep each tree can be)
- `C=1.0` in LogisticRegression (regularization strength)

### Why is asyncio.to_thread() Used?
Python's Telegram bot library runs on an `asyncio` event loop. If you call a slow function (like sklearn's `fit()`) directly in an async function, it blocks the entire event loop — the bot can't process any other messages until it finishes. `asyncio.to_thread()` runs the slow function in a separate thread, so the event loop is free to handle other requests while training happens.

### What is the IPv4 Patch?
Windows's `asyncio` implementation (specifically the `anyio`/`httpcore` library used by `python-telegram-bot`) prefers IPv6 connections. When the DNS lookup for `api.telegram.org` returns both an IPv4 and IPv6 address, it tries IPv6 first. If IPv6 routing is broken (as it is on your network), instead of gracefully falling back to IPv4, it crashes with `ConnectError`.

The fix patches Python's `socket.getaddrinfo()` function — the function that DNS lookups go through — to filter out all IPv6 results. This forces all connections to use IPv4 where your network actually works.

---

## 🗣️ How to Explain This Project in an Interview

### "Tell me about this project."
> "I built a Telegram bot that automates the full machine learning pipeline for any labeled CSV dataset. Instead of manually writing preprocessing and training code for every dataset, the bot uses Groq's LLM API to analyze the dataset and decide the optimal strategy — which scaler to use, whether to handle class imbalance, which models to try. It then trains and cross-validates multiple models, optionally tunes hyperparameters, generates diagnostic plots, and sends a PDF report. After training, it enters a live prediction mode where users can query the model in plain English and get predictions with SHAP explanations."

### "What was the hardest technical challenge?"
> "Keeping the Telegram event loop non-blocking was the main architectural challenge. All the heavy ML operations — sklearn training, Groq API calls, pandas processing — are synchronous and blocking. Running them directly in async handlers would freeze the bot. I solved this by wrapping every blocking call in `asyncio.to_thread()`, which runs them in a thread pool and keeps the event loop free. I also had to fix a network issue where Windows's async I/O framework was trying IPv6 connections to Telegram's servers, which my network doesn't support, causing silent crashes at startup."

### "How does the AI decide which models to use?"
> "I build a JSON summary of the dataset — shape, column types, missing values, class distribution, top correlations — and send it to Groq's llama-3.3-70b-versatile model with a strict prompt. The prompt asks it to return only a JSON object specifying the task type, recommended models, scaler type, whether to handle class imbalance, and whether feature selection is needed. I apply safety overrides on top of the LLM's response — the target column is always forced to the last column regardless of what the LLM says, and model names are validated against a known registry."

### "What metrics do you use to evaluate models?"
> "For classification: accuracy, macro-averaged precision, recall, and F1-score, plus ROC-AUC. I compute these for every model on the same held-out test set. For regression: MAE, MSE, RMSE, and R². I also collect per-model CV scores during training, so the comparison plot shows both the cross-validation performance and test set performance side by side."

### "How does the live prediction work?"
> "When a user types something like 'age 35, income 60000, department HR', the bot sends that text to Groq with a prompt that lists all the model's feature names and asks it to extract a JSON dict of feature values. Groq returns something like `{'age': 35, 'income': 60000, 'department': 'HR'}`. The bot then applies the same label encoders and scaler that were fitted during training, feeds the result to the trained model, and returns the prediction. It also runs SHAP to explain which features drove the prediction."

### "What is SHAP and why did you include it?"
> "SHAP stands for SHapley Additive exPlanations. It's based on game theory — it assigns a contribution value to each feature for a specific prediction, showing how much each feature pushed the prediction up or down from the baseline. I included it because black-box predictions without explanation are hard to trust. With SHAP, a user can see that the model predicted 'high risk' because their age and blood pressure were high, not just get a 1 or 0. For tree-based models I use TreeExplainer which is fast and exact; for others I use a KernelExplainer fallback."

---

## ⚡ Quick Reference — Numbers to Remember

| Thing | Value |
|-------|-------|
| Cross-validation folds | 5 |
| Feature selection threshold | 15 features |
| Hyperparameter tuning iterations | 50 (RandomizedSearchCV) |
| Auto-retry threshold | CV score < 0.60 |
| Max file size | 20 MB (Telegram limit) |
| LLM model | llama-3.3-70b-versatile (Groq) |
| Train/test split | 80% / 20% |
| Classification models available | 8 |
| Regression models available | 7 |
| Plots generated | 8 |
| Deployment files in zip | 6 (model, scaler, encoders, metadata, predict.py, app.py) |

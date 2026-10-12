# How the ML Pipeline Bot Works 🤖

The bot orchestrates a fully automated, start-to-finish Machine Learning pipeline. Here is exactly how it works behind the scenes, step-by-step:

### 1. File Upload & Intake (`bot/handlers.py`)
- When you send a `.csv` file, the bot uses Telegram's API to securely download it into a unique, temporary session folder.
- It loads the CSV into a pandas `DataFrame` and counts the rows, columns, and missing values.

### 2. AI Dataset Analysis (`pipeline/dataset_analyzer.py`)
- The bot builds a lightweight "summary map" of your dataset (column names, data types, unique values, missing percentages, top correlations).
- It sends this summary to the **Groq AI** (`llama-3.3-70b-versatile` model). None of your actual raw data is sent — only the metadata statistics.
- The AI plays the role of a Senior Data Scientist. It replies with a JSON object dictating:
  - Is this Classification or Regression?
  - Which column is the target/label?
  - What preprocessing is needed (e.g., scaling type, balancing)?
  - Which ML models are best suited for this specific dataset shape and distribution?

### 3. Interactive Model Selection (`bot/handlers.py`)
- The bot builds a Telegram **Inline Keyboard** grid displaying all available ML models for your task type (Classification vs Regression).
- Models that Groq AI recommended are marked with a ⭐ and pre-selected (✅) by default.
- You can toggle any models on/off. When you click **Confirm**, the bot locks in your choices and begins the heavy lifting.

### 4. Smart Preprocessing (`pipeline/preprocessor.py`)
The pipeline automatically applies the AI's recommended preprocessing steps:
- **Imputation:** Fills missing values (media for numbers, most-frequent for categories).
- **Encoding:** Converts text categories into numbers (Label Encoding for categories with many unique values, One-Hot Encoding for fewer).
- **Scaling:** Normalizes numeric data (StandardScaler, MinMaxScaler, or RobustScaler depending on outlier likelihood).
- **Class Balancing:** If it's a classification task with highly skewed classes (e.g., 90% Class A, 10% Class B), it applies techniques like **SMOTE** to balance the dataset so models learn fairly.

### 5. Feature Selection (`pipeline/feature_selector.py`)
- If your dataset is very wide (e.g., >15 columns), the bot automatically applies feature selection.
- It uses methods like **SelectKBest**, **Recursive Feature Elimination (RFE)**, or **PCA** to find the most mathematically important columns and discards the noise. This makes models train faster and often improves accuracy.

### 6. Model Training & Cross-Validation (`pipeline/model_trainer.py`)
- The bot isolates your selected models (e.g., XGBoost, Random Forest, Logistic Regression).
- It splits your data and performs strict **5-fold Cross-Validation** (training and testing on 5 different slices of data).
- It calculates the average CV score for every model to prevent overfitting.
- The model with the absolute highest average CV score is crowned the **Best Model**, and then trained one final time on the entire training set.

### 7. Evaluation & PDF Reporting (`pipeline/evaluator.py`, `pipeline/plotter.py`, `utils/pdf_reporter.py`)
- The Best Model is tested one last time on a completely held-out test set (the 20% of data it has never seen).
- Metrics like Accuracy, Precision, Recall, F1, and ROC-AUC are calculated.
- The bot generates 7+ professional, dark-accented visual plots: Confusion matrices, Feature Importance bar charts, ROC curves, Correlation heatmaps, etc.
- It compiles everything into a beautiful **multi-page PDF** using `reportlab`, sends it back to you via Telegram, and instantly deletes the temporary session folder from the server for privacy.

---
_Note: Because Machine Learning math is heavily synchronous and blocks normal code execution, the bot uses `asyncio.to_thread()` to run these intensive steps in invisible background workers. This allows the bot to keep chatting with other users perfectly smoothly while your models are training!_

## 🛠️ Tools & Tech Stack Used

This project is built using a modern Python data science and bot stack:

### Core Frameworks
- `python-telegram-bot` (v20+): Handles the async Telegram interface, inline keyboards, and document downloads.
- `groq`: Official Python client for the Groq API, used to query the Llama 3.3 70B model at extreme speeds.

### Data Science & Machine Learning
- `pandas` & `numpy`: Data manipulation, statistical summaries, and array operations.
- `scikit-learn` (`sklearn`): Core ML framework for preprocessing (Scalers, Encoders, Imputers), Cross-Validation, and training classical models (Random Forest, SVM, Logistic Regression, etc.).
- `xgboost` & `lightgbm`: Advanced gradient boosting frameworks used for high-performance classification and regression.
- `imbalanced-learn` (`imblearn`): Handles class imbalance (e.g., using SMOTE to oversample minority classes).

### Visualisation & PDF Generation
- `matplotlib` & `seaborn`: Generating professional statistical plots (Confusion Matrix, Heatmaps, ROC plots).
- `reportlab`: Programmatic generation of the final multi-page PDF report.
- `Pillow` (`PIL`): Image processing to embed matplotlib plots into the PDF.

---

## 🧠 Why is this an "AI Agent" Project?

Historically, a script or bot simply executes hardcoded logic: "If X happens, do Y." 

This project goes beyond a script and qualifies as an **AI Agent** because it exhibits **perception, reasoning, and autonomous execution:**

1. **Perception (State Observation):**
   Instead of just taking a CSV and blindly running `fit()`, the pipeline first "looks" at the data. It generates an abstract statistical summary (missing ratios, cardinalities, dtypes, correlations) which forms the agent's observation of its environment.

2. **Reasoning & Planning (LLM as the Brain):**
   The agent sends this observation to the Groq LLM (Llama 3). The LLM acts as the "brain," reasoning about the dataset's specific traits. Based strictly on the dataset summary, the agent autonomously decides the following:
   
   - **Task Type:** Determines if the problem requires `Classification` (predicting discrete labels) or `Regression` (predicting continuous numbers). *(Note: We now auto-override this to classification if the target column is text, catching edge-case LLM hallucinations).*
   - **Scaling Strategy:** Chooses the best normalization method (`Standard`, `MinMax`, or `Robust`). For example, if it notices high maximums relative to the mean, it chooses `Robust` scaler to ignore outliers.
   - **Class Imbalance Handling:** Decides if the dataset needs synthetic balancing (`true`/`false`). If yes, it selects the exact algorithm to use (e.g., `smote` to generate synthetic data points for a minority class like "fraud instances").
   - **Feature Selection Switch:** Decides whether to turn on mathematical feature selection (`true`/`false`) if the number of columns exceeds a configured threshold, to prevent the "curse of dimensionality."
   - **Feature Selection Method:** If feature selection is needed, it chooses the approach (e.g., `pca` for linear correlations, or `tree` for tree-based feature trimming).
   - **Model Selection (The Shortlist):** Out of 10+ available models, it selects a bespoke shortlist of 3-4 algorithms mathematically suited for this specific dataset's shape (e.g., recommending `LightGBM` for huge datasets, but `LogisticRegression` for tiny, linearly separable ones).
   - **Pipeline Reasoning:** Generates a human-readable string explaining exactly *why* it made the choices above, giving transparency to the "black box."

3. **Autonomous Execution (Tool Use):**
   The agent parses the LLM's structured JSON decisions and translates them into actionable Python code execution. It automatically configures `scikit-learn` pipelines, handles edge cases (like text labels needing encoding), dynamically toggles class balancing or feature selection based on the data's dimensions, and trains the models without any human code intervention.

In summary: It observes (pandas stats), thinks (Groq LLM), and acts (scikit-learn pipeline execution) without human micro-management.

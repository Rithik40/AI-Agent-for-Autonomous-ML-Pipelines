Rithik B
CB.PS.I5DAS22140

Project Title: AI Agent for Autonomous Machine Learning Pipelines

This project is an end-to-end automated machine learning (AutoML) system integrated with a Telegram bot. It enables users to upload labeled CSV datasets, which are autonomously analyzed by an AI agent using the Groq Llama 3.3 model to determine the optimal preprocessing and model selection strategy. The system handles missing value imputation, categorical encoding, feature scaling, and class balancing using SMOTE. It trains and compares a suite of models including Random Forest, XGBoost, and LightGBM via 5-fold cross-validation and hyperparameter tuning. The bot generates a comprehensive multi-page PDF report with statistical visualizations and provides a downloadable deployment package containing the trained model and a standalone Gradio web application. Additionally, it features a "Live Predictor" mode that uses natural language processing to extract feature values and return real-time predictions.

Tools Used:
* Python – Core programming language
* python-telegram-bot – Telegram bot interface
* Groq API (Llama 3.3) – AI reasoning engine
* Scikit-learn – Machine learning & preprocessing
* Gradio – Interactive model deployment
* ReportLab – PDF report generation

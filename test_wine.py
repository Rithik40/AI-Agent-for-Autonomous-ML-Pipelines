import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pipeline.preprocessor import preprocess
from pipeline.model_trainer import train_models
from pipeline.evaluator import evaluate

if __name__ == '__main__':
    # Make a synthetic 1000x5 difficult classification dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, n_redundant=1, n_classes=5, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
    df['quality_label'] = y

    decisions = {
        "task_type": "classification",
        "target_column": "quality_label",
        "recommended_preprocessing": ["standardization"],
        "scaler_type": "standard",
        "handle_imbalance": False,
    }

    X_proc, y_proc, meta = preprocess(df, decisions)
    print("Preprocessing done.")

    train_res = train_models(X_proc, y_proc, decisions)
    print("Training done.")

    metrics = evaluate(train_res)
    print("Evaluation done.")

    print("\n--- CV Scores ---")
    for name, res in train_res['cv_results'].items():
        print(f"{name}: {res['mean']} ± {res['std']}")

    print("\n--- Test Scores ---")
    for name, m in metrics['all_metrics'].items():
        print(f"{name}: Acc={m['accuracy']}, F1={m['f1']}")

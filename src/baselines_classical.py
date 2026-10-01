import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"

def load_data():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train_df, val_df, test_df

def compute_metrics(y_true, y_pred):
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    kendall_t, _ = kendalltau(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "kendall_tau": float(kendall_t),
        "mae": float(mae)
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_df, val_df, test_df = load_data()

    # Concatenate Topic + Argument text for TF-IDF
    train_text = train_df['topic'] + " " + train_df['argument']
    val_text = val_df['topic'] + " " + val_df['argument']
    test_text = test_df['topic'] + " " + test_df['argument']

    print("Fitting TF-IDF Vectorizer (max_features=1000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    y_train = train_df['quality_score'].values
    y_val = val_df['quality_score'].values
    y_test = test_df['quality_score'].values

    results = {}

    # 1. Ridge Regression
    print("\nTraining Ridge Regression Baseline...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    
    ridge_val_preds = ridge.predict(X_val)
    ridge_test_preds = ridge.predict(X_test)
    
    results["Ridge"] = {
        "val": compute_metrics(y_val, ridge_val_preds),
        "test": compute_metrics(y_test, ridge_test_preds)
    }
    print(f"Ridge Test Metrics -> Pearson: {results['Ridge']['test']['pearson']:.4f}, Spearman: {results['Ridge']['test']['spearman']:.4f}, MAE: {results['Ridge']['test']['mae']:.4f}")

    # 2. Linear SVR
    print("\nTraining Linear SVR Baseline...")
    svr = LinearSVR(C=1.0, max_iter=2000, random_state=42)
    svr.fit(X_train, y_train)
    
    svr_val_preds = svr.predict(X_val)
    svr_test_preds = svr.predict(X_test)
    
    results["LinearSVR"] = {
        "val": compute_metrics(y_val, svr_val_preds),
        "test": compute_metrics(y_test, svr_test_preds)
    }
    print(f"LinearSVR Test Metrics -> Pearson: {results['LinearSVR']['test']['pearson']:.4f}, Spearman: {results['LinearSVR']['test']['spearman']:.4f}, MAE: {results['LinearSVR']['test']['mae']:.4f}")

    out_file = os.path.join(RESULTS_DIR, "classical_baseline.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classical baseline results to {out_file}")

if __name__ == "__main__":
    main()

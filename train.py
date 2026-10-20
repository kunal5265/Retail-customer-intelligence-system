"""
Retail Customer Intelligence & Propensity Model Pipeline
Generates synthetic transaction logs, extracts temporal cycle drift features,
trains a LightGBM classifier with probability calibration, and evaluates metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, brier_score_loss
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

def generate_synthetic_transactions(num_customers=800, num_records=8000):
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    customers = [f"CUST-{1000 + i}" for i in range(num_customers)]
    records = []
    
    for _ in range(num_records):
        cust = np.random.choice(customers)
        offset = np.random.randint(0, 365)
        invoice_date = start_date + timedelta(days=offset)
        amount = round(float(np.random.exponential(scale=65) + 15), 2)
        records.append({
            "customer_id": cust,
            "invoice_date": invoice_date,
            "amount": amount,
            "discount_applied": int(np.random.choice([0, 1], p=[0.75, 0.25]))
        })
        
    return pd.DataFrame(records).sort_values("invoice_date")

def extract_features(df, cutoff_days=30):
    max_date = df["invoice_date"].max()
    split_date = max_date - timedelta(days=cutoff_days)
    
    history_df = df[df["invoice_date"] <= split_date]
    future_df = df[df["invoice_date"] > split_date]
    future_buyers = set(future_df["customer_id"].unique())
    
    features = []
    for cust_id, group in history_df.groupby("customer_id"):
        if len(group) < 2:
            continue
            
        dates = group["invoice_date"].sort_values().tolist()
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        ipi_mean = float(np.mean(intervals)) if intervals else 30.0
        ipi_std = float(np.std(intervals)) if intervals else 0.0
        
        recency = (split_date - dates[-1]).days
        frequency = len(group)
        monetary = float(group["amount"].sum())
        aov = float(group["amount"].mean())
        discount_ratio = float(group["discount_applied"].mean())
        cycle_drift = recency / ipi_mean if ipi_mean > 0 else 1.0
        
        features.append({
            "customer_id": cust_id,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "aov": aov,
            "ipi_mean": ipi_mean,
            "ipi_std": ipi_std,
            "discount_ratio": discount_ratio,
            "cycle_drift": cycle_drift,
            "target_repurchase_30d": 1 if cust_id in future_buyers else 0
        })
        
    return pd.DataFrame(features)

def main():
    print("1. Generating synthetic POS transaction stream...")
    df_tx = generate_synthetic_transactions()
    
    print("2. Extracting behavioral cadence and cycle-drift features...")
    df_features = extract_features(df_tx)
    print(f"   Cohort size: {len(df_features)} profiles.")
    
    feature_cols = ["recency", "frequency", "monetary", "aov", "ipi_mean", "ipi_std", "discount_ratio", "cycle_drift"]
    X = df_features[feature_cols]
    y = df_features["target_repurchase_30d"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("3. Training base LightGBM classifier with isotonic probability calibration...")
    base_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
    calibrated_model = CalibratedClassifierCV(estimator=base_lgbm, method="isotonic", cv=3)
    calibrated_model.fit(X_train, y_train)
    
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    print("\n" + "="*50)
    print("MODEL VALIDATION RESULTS")
    print("="*50)
    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"Brier Score   : {brier:.4f} (Probability calibration error)")
    print("="*50)
    
    joblib.dump(calibrated_model, "calibrated_lgbm.pkl")
    print("Model serialized and saved to 'calibrated_lgbm.pkl'.\n")

if __name__ == "__main__":
    main()

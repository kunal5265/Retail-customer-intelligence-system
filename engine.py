"""
AETHER // Enterprise Propensity & Causal Uplift Modeling Engine
Dual-Head Architecture:
  1. Isotonic Calibrated LightGBM (Probability of Purchase)
  2. Causal T-Learner (Individual Treatment Effect / Uplift Modeling)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, brier_score_loss
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

def generate_point_in_time_dataset(n_customers=1200, n_records=15000):
    """Generates leak-free point-in-time transaction logs."""
    np.random.seed(42)
    end_date = datetime.now()
    cutoff_date = end_date - timedelta(days=30)
    start_date = cutoff_date - timedelta(days=180)
    
    customers = [f"CUST-{2000 + i}" for i in range(n_customers)]
    records = []
    
    for _ in range(n_records):
        cust = np.random.choice(customers)
        # Random dates spread across 210 days
        tx_date = start_date + timedelta(days=int(np.random.uniform(0, 210)))
        amount = round(float(np.random.exponential(scale=70) + 20), 2)
        records.append({
            "customer_id": cust,
            "invoice_date": tx_date,
            "amount": amount
        })
        
    df = pd.DataFrame(records).sort_values("invoice_date")
    
    # Strictly split into historical observation vs future target window
    history = df[df["invoice_date"] <= cutoff_date]
    future = df[df["invoice_date"] > cutoff_date]
    future_buyers = set(future["customer_id"].unique())
    
    features = []
    for cust_id, group in history.groupby("customer_id"):
        if len(group) < 2:
            continue
            
        dates = sorted(group["invoice_date"].tolist())
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        ipi_mean = float(np.mean(intervals)) if intervals else 30.0
        ipi_std = float(np.std(intervals)) if intervals else 0.0
        
        recency = (cutoff_date - dates[-1]).days
        frequency = len(group)
        monetary = float(group["amount"].sum())
        aov = float(group["amount"].mean())
        cycle_drift = recency / ipi_mean if ipi_mean > 0 else 1.0
        
        # Synthetic treatment assignment (A/B testing simulation: 50% got promo voucher)
        treatment = int(np.random.binomial(1, 0.5))
        
        features.append({
            "customer_id": cust_id,
            "recency_days": recency,
            "frequency_orders": frequency,
            "monetary_total": round(monetary, 2),
            "aov": round(aov, 2),
            "ipi_mean": round(ipi_mean, 2),
            "ipi_std": round(ipi_std, 2),
            "cycle_drift_ratio": round(cycle_drift, 3),
            "treatment": treatment,
            "target_repurchase_30d": 1 if cust_id in future_buyers else 0
        })
        
    return pd.DataFrame(features)

class ProductionEngine:
    def __init__(self):
        self.features = [
            "recency_days", "frequency_orders", "monetary_total", 
            "aov", "ipi_mean", "ipi_std", "cycle_drift_ratio"
        ]
        self.propensity_model = None
        self.uplift_treatment = None
        self.uplift_control = None

    def fit_and_evaluate(self, df):
        X = df[self.features]
        y = df["target_repurchase_30d"]
        t = df["treatment"]

        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t, test_size=0.25, random_state=42, stratify=y
        )

        # 1. Calibrated Propensity Head (Isotonic LightGBM)
        base = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
        self.propensity_model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
        self.propensity_model.fit(X_train, y_train)

        # 2. Causal T-Learner (Uplift Modeling)
        self.uplift_treatment = LGBMClassifier(n_estimators=60, max_depth=4, random_state=42, verbose=-1)
        self.uplift_control = LGBMClassifier(n_estimators=60, max_depth=4, random_state=42, verbose=-1)

        self.uplift_treatment.fit(X_train[t_train == 1], y_train[t_train == 1])
        self.uplift_control.fit(X_train[t_train == 0], y_train[t_train == 0])

        # Validation Metrics
        probs = self.propensity_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        print("\n" + "="*50)
        print("   AETHER ADVANCED MODEL METRICS")
        print("="*50)
        print(f"ROC-AUC Score          : {auc:.4f}")
        print(f"Brier Calibration Loss : {brier:.4f}")
        print("="*50 + "\n")

        joblib.dump(self, "engine_artifact.pkl")
        print("Engine serialized to 'engine_artifact.pkl'.")

if __name__ == "__main__":
    df = generate_point_in_time_dataset()
    engine = ProductionEngine()
    engine.fit_and_evaluate(df)

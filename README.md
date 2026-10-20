# Aether // Retail Customer Intelligence & Repurchase Propensity System
**Machine Learning Mini-Project | Customer Analytics & Behavioral Propensity Modeling**

[![Live Demo](https://img.shields.io/badge/Demo-Live_Web_App-00F0FF)](https://kunal5265.github.io/Retail-customer-intelligence-system/)
[![Python](https://img.shields.io/badge/Python-3.9+-05FFA1)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Calibrated_LightGBM-FFB800)](https://lightgbm.readthedocs.io/)

---

## 1. Problem Statement
Retail organizations waste marketing spend by retargeting customers indiscriminately:
- Customers planning to purchase naturally (*organic loyalists*) receive margin-eroding discount vouchers.
- Customers silently drifting away receive re-engagement prompts too late.

This project solves this by constructing a unified predictive system modeling **individual customer cycle drift** ($R_{\text{ratio}} = R / \mu_{\text{IPI}}$) to predict the **probability of repurchase within the next 30 days**.

---

## 2. Feature Engineering Pipeline

Rather than relying purely on static RFM segmentation, this system builds temporal cadence metrics:

| Feature Name | Mathematical Definition | Clinical / Business Rationale |
| :--- | :--- | :--- |
| **Recency ($R$)** | $\text{days since last invoice}$ | Baseline inactivity indicator |
| **Frequency ($F$)** | $\sum \text{invoices}$ | Historical engagement volume |
| **Monetary ($M$)** | $\sum \text{spend}$ | Gross lifetime contribution |
| **Mean Interval ($\mu_{\text{IPI}}$)** | $\frac{1}{n-1} \sum (t_i - t_{i-1})$ | Individual customer shopping cadence |
| **Cycle Drift ($R_{\text{ratio}}$)** | $\frac{R}{\mu_{\text{IPI}}}$ | Dynamic churn indicator ($>1.5\times$ indicates high lapse risk) |
| **Discount Reliance** | $\frac{\text{Discounted Orders}}{\text{Total Orders}}$ | Price sensitivity coefficient |

---

## 3. Modeling & Probability Calibration

- **Base Classifier:** LightGBM with tree depth regularization to prevent overfitting on outlier frequent shoppers.
- **Calibration Engine:** `CalibratedClassifierCV` using Isotonic Regression so output scores represent true conversion likelihoods.
- **Evaluation Metrics:**
  - **ROC-AUC:** Measures ranking discrimination (~0.89).
  - **Brier Score:** Evaluates mean squared error of predicted probabilities (< 0.08).

---

## 4. Live Interactive UI & Web Console
The deployed web console provides:
1. **Behavioral Topography Canvas:** Visualizes customer cycle drift vs. calibrated purchase propensity.
2. **Local Explainability (SHAP Simulation):** Feature-level contributions per customer profile.
3. **Marketing Yield Simulator:** Arbitrage engine calculating optimal promo discount cutoffs to maximize margin.

🔗 **Live Deployment:** [https://kunal5265.github.io/Retail-customer-intelligence-system/](https://kunal5265.github.io/Retail-customer-intelligence-system/)

---

## 5. How to Run Locally

```bash
# 1. Clone repository
git clone [https://github.com/kunal5265/Retail-customer-intelligence-system.git](https://github.com/kunal5265/Retail-customer-intelligence-system.git)
cd Retail-customer-intelligence-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate feature store and train model
python train.py

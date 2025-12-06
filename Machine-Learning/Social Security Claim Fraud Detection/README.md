
# Social Security Claim Fraud Detection (Synthetic SSA Project)

This project builds a complete **fraud detection pipeline** on a synthetic Social Security Administration (SSA)-style dataset of benefit claims.

## Project Goals

- Predict which claims are likely to be fraudulent (`is_fraud`)
- Handle imbalanced data (fraud is rare, ~10% of claims)
- Focus on fraud-relevant evaluation metrics (ROC AUC, PR AUC, recall for fraud)
- Use SHAP to explain model behavior and highlight key risk drivers
- Export fraud risk scores for BI dashboards or case review queues

## Dataset

The file `data/ssa_claim_fraud_dataset.csv` contains 12,000 synthetic claim records with columns such as:

- `age`, `sex`, `state`
- `benefit_type`, `years_contributed`
- `prior_claim_count`, `prior_denial_count`
- `overpayment_flag`, `suspicious_ip_flag`, `doc_inconsistency_count`
- `filing_channel`, `employment_status`
- `claim_amount`, `monthly_benefit`
- `is_fraud` (target label)

Fraud labels are generated using realistic combinations of risk factors (e.g., high claim amount with low contributions, prior denials, suspicious IP, mismatched bank country).

## Folder Structure

```text
ssa_fraud_project/
├── data/
│   ├── ssa_claim_fraud_dataset.csv   # Synthetic SSA claims dataset
│   └── fraud_predictions.csv         # Model output (created by notebook)
├── notebooks/
│   └── fraud_model.ipynb             # Main fraud detection notebook
├── src/
│   ├── preprocess.py                 # Data loading & preprocessing helpers
│   ├── train.py                      # Model training helpers
│   ├── evaluate.py                   # Evaluation utilities
│   └── shap_explain.py               # SHAP explainability helpers
└── powerbi/  
    └── FRAUD RISK DASHBOARD.pbit     # Power BI Report
```

## ML Approach

1. **Preprocessing**
   - Drop IDs for modeling (`claim_id`, `person_id` kept for joins only)
   - One-hot encode categorical variables
   - Keep numeric features as-is

2. **Imbalanced Learning**
   - Train/test split with stratification on `is_fraud`
   - Use SMOTE to oversample the fraud class on the training set

3. **Models**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (primary fraud model)

4. **Metrics (fraud-focused)**
   - ROC AUC
   - Precision-Recall AUC (PR AUC)
   - Classification report (precision, recall, F1)
   - Confusion matrix

5. **Explainability**
   - Use SHAP (TreeExplainer) on XGBoost
   - SHAP summary and bar plot to identify top fraud drivers

6. **Outputs**
   - `data/fraud_predictions.csv` with:
     - `claim_id`, `person_id`
     - `fraud_flag_pred` (0/1)
     - `fraud_probability` (0–1)
     - `fraud_risk_level` (Low / Medium / High)

## How to Run

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

- In VS Code, Jupyter Notebook, or Google Colab, open `notebooks/fraud_model.ipynb`
- Run all cells from top to bottom

This will:

- Train multiple fraud detection models
- Evaluate performance
- Generate SHAP explainability plots
- Export `fraud_predictions.csv`

## Potential BI / Dashboard Use

You can load `fraud_predictions.csv` into Power BI or another BI tool to build:

- Overall fraud rate & risk distribution
- Fraud risk by state, benefit type, filing channel
- Top risk drivers (via SHAP output)
- High-risk claims table for investigators

## Notes

- All data is synthetic and generated for educational and portfolio purposes only.
- No real SSA or PII data is used.

# FinWise Lending — Notebooks Directory

## 1. Purpose

The `notebooks/` directory is reserved for exploratory data analysis (EDA), data quality visualization, iterative feature exploration, and prototype modeling experiments.

In Week 1, this directory serves as a defined workspace ready for the upcoming Week 3 exploratory milestones.

---

## 2. Planned Experimentation Workflow (Week 3+)

The notebooks are designed to follow a structured, numbered lifecycle:

1. `01_exploratory_data_analysis.ipynb`:
   - Inspect data distributions, summary statistics, and missing value mechanisms.
   - Profile the candidate datasets from the client URLs.
   - Confirm target column candidates (`loan_status` vs `is_default`) and class distributions.
2. `02_feature_engineering_and_leakage_audit.ipynb`:
   - Identify and isolate post-origination leakage features.
   - Construct financial underwriting ratios (e.g., debt-to-income, credit utilization).
   - Test categorical encoding strategies.
3. `03_baseline_and_xgboost_benchmarks.ipynb`:
   - Establish a Logistic Regression baseline model.
   - Train and hyperparameter-tune the XGBoost classifier.
   - Evaluate Gini, ROC-AUC, PR-AUC, and calibration curves.
4. `04_shap_explainability_and_fairness.ipynb`:
   - Compute TreeSHAP summary, dependence, and local waterfall plots.
   - Conduct disparate impact and equal opportunity audits across age and gender.

---

## 3. Best Practices & Governance Rules

- **Code Migration:** Notebooks are for prototyping only. Any reusable data transformations or evaluation utilities must be refactored into production modules in `src/`.
- **Reproducibility:** Always set random seeds (e.g., `random_state=42`) across cross-validation splits and model initializations.
- **Data Safety:** Never embed raw customer data, credentials, or confidential output in notebook cells. Ensure notebook outputs do not leak PII before committing.
- **Version Control:** Clear unnecessary heavy binary outputs before committing `.ipynb` files, or use paired `.py` scripts via Jupytext where applicable.

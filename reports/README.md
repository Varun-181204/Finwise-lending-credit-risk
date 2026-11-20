# FinWise Lending — Reports & Audit Artifacts

## 1. Purpose & Governance Scope

The `reports/` directory stores formal model validation reports, compliance documentation, fairness audit summaries, and client milestone delivery artifacts.

In credit risk modeling, comprehensive reporting is a regulatory prerequisite to satisfy Model Risk Management (MRM) standards and ensure model auditability over time.

---

## 2. Report Artifacts

### 1. `dataset_audit.md` (Week 3 - Completed)
- Comprehensive, privacy-safe audit of all 3 candidate client datasets.
- Schema definitions, missing value checks, and aggregate distributions.
- Empirical join key incompatibility analysis and blocker decision between `application_id` and `applicant_id`.
- Formal isolation of `finwise_fraud_transactions.csv`.

### 2. `baseline_report.md` (Week 3 - Completed)
- Dataset provenance and selection of `finwise_loan_applications.csv` as the authoritative modeling source.
- Preprocessing, feature auditing, and leakage-prevention pipeline documentation.
- Logistic Regression baseline architecture and validation threshold selection ($t^* = 0.40$).
- Empirical evaluation results on holdout test partition (ROC-AUC: 0.9277, Gini: 0.8554 surpassing the $\ge 0.45$ target).
- Model coefficient interpretation, confusion matrix, limitations, and client clarification questions.

### 3. `model_quality_audit.md` (Week 3 - Completed)
- Forensic model quality, target leakage, and evaluation integrity audit.
- Feature-by-feature temporal and derivation assessment.
- Mathematical verification of `dti_ratio` and model sensitivity ablation study (Gini = 0.8181 without DTI).
- Dataset provenance limitations, synthetic artifacts, and partition independence verification.
- Fairness audit readiness: age audit ready; gender audit blocked due to join mismatch.
- Formal questions for Cynaris Solutions and recommended corrective actions.

### 2. `model_benchmark_report.md` (Week 3/6)
- Comparative performance between the Logistic Regression baseline and the tuned XGBoost classifier.
- Formal evaluation of Gini ($\ge 0.45$ target), ROC-AUC, PR-AUC, confusion matrix, and calibration curve.

### 2. `fairness_and_bias_audit.md` (Week 6)
- Disparate Impact Ratio (DIR) analysis across age and gender subgroups.
- Equal Opportunity Difference (EOD) comparisons.
- Identification of potential proxy features and documentation of demographic data limitations.

### 3. `explainability_validation_report.md` (Week 6)
- Global TreeSHAP feature importance ranking.
- Validation of per-prediction SHAP waterfall charts and risk driver attributions for underwriter decision support.

### 4. `system_performance_report.md` (Week 9 & 12)
- FastAPI endpoint throughput and latency benchmarks under 50 concurrent requests.
- End-to-end integration test audit logs.

---

## 3. Git Exclusions & Storage Policy

- The root `.gitignore` excludes heavy binary report artifacts (`.html`, `.pdf`, `.png`, `.jpg`).
- Only markdown summary reports, structured configuration metadata, and README documentation are tracked in version control.

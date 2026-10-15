# FinWise Lending — Reports & Audit Artifacts

## 1. Purpose & Governance Scope

The `reports/` directory stores formal model validation reports, compliance documentation, fairness audit summaries, and client milestone delivery artifacts.

In credit risk modeling, comprehensive reporting is a regulatory prerequisite to satisfy Model Risk Management (MRM) standards and ensure model auditability over time.

---

## 2. Planned Report Artifacts

The following reports are scheduled for generation during upcoming milestones:

### 1. `model_benchmark_report.md` (Week 3)
- Comparative performance between the Logistic Regression baseline and the tuned XGBoost classifier.
- Formal evaluation of:
  - Gini coefficient (evaluating progress toward the client's $\ge 0.45$ target).
  - ROC-AUC and PR-AUC curves.
  - Confusion matrix and cost-benefit trade-off curves.
  - Probability calibration curve and Brier score.

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

# FinWise Lending — Credit Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/pytest-38%20passing-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()
[![Milestone](https://img.shields.io/badge/Milestone-Week%203%20Complete-success.svg)]()

## 1. Project Objective & Client Acceptance Criteria

FinWise Lending is developing an intelligent, regulatory-compliant credit risk underwriting decision platform. The system ingests loan application data, predicts borrower default risk, categorizes applicants into transparent risk tiers, produces instance-level explainability via SHAP waterfall charts, and exposes real-time decision support to underwriters via an API and frontend dashboard.

### Client Acceptance Criteria:
* **Ranking & Discrimination Target:** Achieve a **test-set Gini coefficient of $\ge 0.45$** (equivalent to **ROC-AUC $\ge 0.725$**) on the holdout evaluation partition.
* **Instance-Level Explainability:** Every credit prediction must generate an individual **SHAP waterfall chart per prediction** illustrating exact feature attributions to support underwriter evaluations.
* **Transparent Risk Classification:** Categorization of applicants into **Low**, **Medium**, and **High** risk bands (thresholds calibrated from empirical default rates and agreed with client credit policy).
* **High-Concurrency Serving:** FastAPI service architected, optimized, and load-tested to handle **50 concurrent API requests** reliably.
* **Fairness & Non-Discrimination:** Systematic bias audits evaluating disparate impact and equal opportunity across **gender and age** cohorts.
* **Jurisdiction-Agnostic Governance:** Regulatory compliance requirements depend on operating jurisdiction and require formal client and compliance team confirmation. Explainability frameworks (such as SHAP) provide mathematical feature attribution but do not alone guarantee legal or regulatory compliance.

> [!IMPORTANT]
> **Implementation Status Note:** The **test-set Gini $\ge 0.45$** client milestone is satisfied by the baseline model, which achieves **$\text{Gini} = 0.8554$** ($\text{ROC-AUC} = 0.9277$) on the holdout test set of `data/finwise_loan_applications.csv`. Non-linear modeling with XGBoost and instance-level SHAP waterfall charts remain scheduled for Week 6.

---

## 2. System Architecture

The planned data and inference flow progresses from raw application ingestion to real-time risk evaluation in the loan officer dashboard.

![FinWise Lending Architecture](docs/architecture.svg)

*For the complete architectural design, component responsibilities, API schemas, and editable Mermaid diagram source, see [`docs/architecture.md`](docs/architecture.md).*

---

## 3. Technology Stack

### Active Stack (Weeks 1–3 Ingestion, Preprocessing & Baseline):
* **Language:** Python 3.14
* **Data Processing & ML:** `pandas` ($\ge 2.2.0$), `scikit-learn` ($\ge 1.4.0$)
* **Testing Framework:** `pytest` ($\ge 8.0.0$)
* **Environment Management:** Python standard `venv`

### Future Production Stack (Planned for Weeks 6–12):
* **Advanced ML Engine (Week 6):** `xgboost` (gradient boosting)
* **Model Explainability (Week 6):** `shap` (TreeSHAP exact attributions & waterfall charts)
* **Backend Serving (Week 9):** `fastapi`, `uvicorn`, `pydantic` (architected for 50 concurrent requests)
* **Frontend Dashboard (Week 12):** React, TypeScript, Tailwind CSS, Recharts

---

## 4. Actual Implementation Status

| Component | Target Week | Current Status | Description |
| :--- | :---: | :---: | :--- |
| **Research & Literature Review** | Week 1 | **Completed** | Full review of XGBoost, SHAP, explainability, leakage, and fairness in `docs/research_review.md`. |
| **System Architecture Design** | Week 1 | **Completed** | Full architecture with vector diagram (`docs/architecture.svg`), bitmap (`docs/architecture.png`), and API contracts in `docs/architecture.md`. |
| **Evaluation Plan** | Week 1 | **Completed** | Metric formulas, Gini target ($\ge 0.45$), calibration, and fairness plans in `docs/evaluation_plan.md`. |
| **Risk & Assumptions Register** | Week 1 | **Completed** | Catalog of technical, business, and data risks in `docs/risks_and_assumptions.md`. |
| **Data Ingestion & Quality Engine** | Weeks 1–3 | **Completed** | Implemented `src.data_pipeline` with `load_csv`, `validate_loan_applications_schema`, and `verify_join_compatibility`. |
| **Preprocessing & Leakage Guards** | Week 3 | **Completed** | Implemented `src.preprocessing` with target/feature separation, feature auditing, scikit-learn `ColumnTransformer`, and stratified splitting. |
| **Baseline Modeling Engine** | Week 3 | **Completed** | Implemented `src.baseline_model` and `src.train_baseline` with balanced Logistic Regression, validation threshold selection ($t^*=0.40$), and test Gini of 0.8554 ($\ge 0.45$ target satisfied). |
| **Dataset Audit Report** | Week 3 | **Completed** | Authored `reports/dataset_audit.md` covering all 3 client datasets, privacy-safe distributions, join key mismatch analysis, and fraud isolation. |
| **Baseline Report** | Week 3 | **Completed** | Detailed report in `reports/baseline_report.md` with empirical test results (Gini 0.8554, ROC-AUC 0.9277), confusion matrix, feature coefficients, and limitations. |
| **Automated Test Suite** | Weeks 1–3 | **Completed** | 38 automated unit & integration tests passing across ingestion, schema validation, join blocker, preprocessing, and model training with 100% success rate. |
| **XGBoost & SHAP Explainability** | Week 6 | *Planned* | Tuned XGBoost classifier, probability calibration, Low/Medium/High risk bands, TreeSHAP waterfall charts, and gender/age fairness audit. |
| **FastAPI Microservice** | Week 9 | *Planned* | High-concurrency API supporting 50 concurrent requests (`/predict`, `/explain`, `/health`). |
| **React Loan Officer Dashboard**| Week 12 | *Planned* | Interactive dashboard with risk profile, waterfall charts, and underwriter decision support. |

---

## 5. Project Directory Structure

```
finwise-lending-credit-risk/
├── backend/                  # FastAPI service (Planned - Week 9)
│   └── README.md             # Microservice architecture and endpoint specs
├── data/                     # Local datasets directory (Excluded from Git)
│   └── README.md             # Data privacy policy and inspection guide
├── docs/                     # System design and research documentation
│   ├── architecture.md       # Architecture diagrams, component flow, and API contracts
│   ├── architecture.png      # High-resolution architecture diagram image
│   ├── architecture.svg      # Vector architecture diagram (client deliverable)
│   ├── evaluation_plan.md    # Metrics, Gini target (>=0.45), calibration, fairness
│   ├── research_review.md    # Academic review of XGBoost, SHAP, and lending literature
│   └── risks_and_assumptions.md # Risk matrix, assumptions, and mitigation strategies
├── frontend/                 # React Dashboard (Planned - Week 12)
│   └── README.md             # Dashboard UX features and component hierarchy
├── notebooks/                # Exploratory notebooks (Planned - Week 6)
│   └── README.md             # Experimentation lifecycle and reproducibility guidelines
├── reports/                  # Validation and audit reports
│   ├── baseline_report.md    # Week 3 baseline model, pipeline, and empirical results
│   ├── dataset_audit.md      # Week 3 privacy-safe dataset audit and join blocker analysis
│   └── README.md             # Model governance, audit logs, and compliance sign-offs
├── src/                      # Production Python modules
│   ├── __init__.py           # Package initialization exposing pipeline & model functions
│   ├── baseline_model.py     # Logistic Regression baseline, threshold selection & metrics
│   ├── data_pipeline.py      # Safe CSV ingestion, schema validation & join verification
│   ├── preprocessing.py      # Preprocessing, leakage filter & stratified split
│   └── train_baseline.py     # End-to-end baseline training runner on client dataset
├── tests/                    # Automated testing suite (38 unit & integration tests)
│   ├── __init__.py
│   ├── test_actual_data_validation.py # 7 tests for real data schema, join blocker & training
│   ├── test_baseline_model.py # 7 tests for baseline modeling & metrics
│   ├── test_data_pipeline.py  # 11 tests for ingestion & validation
│   └── test_preprocessing.py  # 13 tests for preprocessing, leakage & splitting
├── .gitignore                # Comprehensive exclusions (venv, caches, data, models, secrets)
├── README.md                 # Project root documentation
└── requirements.txt          # Minimal dependency manifest (pandas, scikit-learn, pytest)
```

---

## 6. Windows PowerShell Setup & Test Instructions

### Step 1: Clone & Navigate to Project
```powershell
cd C:\Users\HP\OneDrive\Documents\finwise-lending-credit-risk
```

### Step 2: Initialize Virtual Environment
```powershell
python -m venv .venv
```
*(If `python` is not directly on PATH, use `py -m venv .venv`)*

### Step 3: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Execute Test Suite
Run the full test suite using pytest:
```powershell
pytest tests -v
```

### Step 6: Train and Evaluate Baseline Model
Run the end-to-end baseline pipeline on the client loan applications dataset:
```powershell
python -m src.train_baseline
```

---

## 7. Multi-Week Project Roadmap

* **Week 1: Research + Architecture Design (Completed)**
  - Literature review (XGBoost, SHAP, fair lending).
  - System architecture design, vector diagram (`docs/architecture.svg`), and API boundary specifications.
  - Evaluation plan and client target definition (test-set Gini $\ge 0.45$).
  - Defensive data ingestion engine (`src/data_pipeline.py`) and passing unit tests.
* **Week 3: Data Pipeline & Baseline Modeling (Completed)**
  - Audited all 3 candidate client datasets in `reports/dataset_audit.md`.
  - Confirmed `default_flag` in `finwise_loan_applications.csv` as the true default target (23.8% prevalence).
  - Proved join key incompatibility between `application_id` and `applicant_id` (0% overlap); blocked unsafe merges.
  - Strictly isolated `finwise_fraud_transactions.csv` from origination credit default modeling.
  - Implemented schema validation and join verification in `src/data_pipeline.py`.
  - Built leakage-free scikit-learn preprocessing `ColumnTransformer` fitted strictly on `X_train`.
  - Executed stratified 70/15/15 train/val/test splitting (`random_state=42`).
  - Tuned operating threshold on validation split ($t^* = 0.40$).
  - Evaluated on holdout test set: achieved **Test Gini = 0.8554** ($\ge 0.45$ target satisfied) and **ROC-AUC = 0.9277**.
  - 38 automated unit & integration tests passing with 100% success rate.
  - Authored comprehensive documentation in `reports/dataset_audit.md` and `reports/baseline_report.md`.
* **Week 6: Advanced Modeling, Explainability & Fairness Auditing (Planned)**
  - Train non-linear XGBoost classifier with hyperparameter tuning on validation split.
  - Probability calibration (Platt scaling / Isotonic regression) and Brier score evaluation.
  - Risk categorization into **Low**, **Medium**, and **High** risk bands.
  - TreeSHAP local attribution and generation of a **SHAP waterfall chart per prediction**.
  - Comprehensive fairness audits across **gender and age** cohorts.
* **Week 9: FastAPI Microservice & Backend Deployment**
  - Build asynchronous REST API with Pydantic validation schemas.
  - High-concurrency optimization and load testing under **50 concurrent requests**.
  - Integration testing and containerization.
* **Week 12: React Dashboard & Final Production Delivery**
  - Loan officer web console with real-time risk gauges.
  - Interactive SHAP waterfall chart visualization for each application.
  - Final client milestone review, documentation handoff, and acceptance.

---

## 8. Data Safety, Privacy & Source URLs

### Authorized Client Data-Pack Sources:
1. `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_loan_applications.csv`
2. `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_credit_risk_features.csv`
3. `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_fraud_transactions.csv`

### Data Safety Rules:
* **No Commits of Raw Data:** Datasets must never be committed to version control. The `.gitignore` strictly excludes all data extensions.
* **No Speculative Schema Merging:** We do not assume these datasets share the same prediction task or key structure. Each file must be profiled independently.
* **No Silent Data Dropping:** The ingestion pipeline never silently drops missing values or duplicate records; all anomalies are summarized transparently in quality reports.

---

## 9. Research References

1. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). [arXiv:1603.02754](https://arxiv.org/abs/1603.02754).
2. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.* In Advances in Neural Information Processing Systems (NeurIPS 2017, pp. 4765–4774). [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).
3. **DMLC (Distributed Machine Learning Community).** *XGBoost: Scalable, Portable and Distributed Gradient Boosting Library.* GitHub Repository. [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost).

---

## 10. Regulatory & Underwriting Disclaimer

> [!CAUTION]
> **Regulatory Notice:**
> The FinWise Lending credit risk assessment software and models developed within this repository are research and engineering prototypes designed for underwriter decision support. This software is **not suitable for real-world automated credit decisions** without formal model validation, compliance certification under applicable fair lending and consumer credit laws in the operating jurisdiction, and appropriate banking supervisory approvals. TreeSHAP provides mathematical feature attributions but does not alone guarantee legal or regulatory compliance.

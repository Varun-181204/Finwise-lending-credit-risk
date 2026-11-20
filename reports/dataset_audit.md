# FinWise Lending — Week 3: Client Dataset Audit & Ingestion Analysis

**Project:** FinWise Lending — Credit Risk Assessment System
**Document:** Dataset Audit, Target Analysis & Join Feasibility Assessment
**Milestone:** Checkpoint 2 (Week 3: Data Pipeline + Baseline Model)
**Date:** November 2026
**Status:** Completed & Verified

---

## 1. Executive Summary

This document presents a comprehensive, privacy-safe audit of the three candidate client datasets provided by Cynaris for the FinWise Lending credit risk assessment project:
1. `data/finwise_loan_applications.csv`
2. `data/finwise_credit_risk_features.csv`
3. `data/finwise_fraud_transactions.csv`

In strict alignment with enterprise risk management practices, data governance regulations, and core project guidelines:
* **Zero PII Exposure:** No applicant-level rows, names, personal identifiers, or raw individual records are published in this report or exposed in system logs. All findings are presented strictly through aggregated summaries, distribution metrics, and schema evaluations.
* **Target Clarification:** The actual loan credit-default target is identified as `default_flag` in `finwise_loan_applications.csv` (762 non-defaults [76.2%], 238 defaults [23.8%]). In contrast, `finwise_credit_risk_features.csv` contains an underwriting origination decision (`approved` / `risk_category`), which represents application approval rather than loan performance.
* **Join Key Incompatibility & Blocker:** Empirical analysis proves that `finwise_loan_applications.csv` (keyed by `application_id`) and `finwise_credit_risk_features.csv` (keyed by `applicant_id`) have **0 exact matching keys (0.0% overlap)** and represent completely independent, non-overlapping sample populations. Merging these datasets via arbitrary row alignment or fabricated mappings would corrupt borrower data and is **strictly blocked**.
* **Fraud Dataset Isolation:** `finwise_fraud_transactions.csv` represents transaction-level payment fraud monitoring (e.g., account takeover, synthetic identity) and is **strictly isolated** from the loan credit-default model to avoid domain confusion and label leakage.
* **Authoritative Modeling Dataset:** `finwise_loan_applications.csv` is selected as the single valid, verified dataset for the Week 3 baseline and future credit risk models.

---

## 2. Dataset Ingestion & Dimension Overview

All three candidate datasets were ingested using the non-destructive ingestion module `src.data_pipeline.load_csv`. The dimensions, memory footprints, and structural integrity are summarized below:

| Dataset Name | File Path | Row Count | Column Count | Duplicate Rows | Null Cell Count | Primary Key / Identifier | Verified Target Column |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Loan Applications** | `data/finwise_loan_applications.csv` | 1,000 | 13 | 0 (0.0%) | 0 (0.0%) | `application_id` (Unique: 1,000) | `default_flag` (Binary 0/1) |
| **Credit Risk Features** | `data/finwise_credit_risk_features.csv` | 1,000 | 17 | 0 (0.0%) | 0 (0.0%) | `applicant_id` (Unique: 1,000) | `approved` (Binary 0/1) & `risk_category` |
| **Fraud Transactions** | `data/finwise_fraud_transactions.csv` | 500 | 12 | 0 (0.0%) | 452 (7.5%) | `txn_id` (Unique: 500) | `is_fraud` (Binary 0/1) |

---

## 3. Dataset 1 Audit: Loan Applications (`finwise_loan_applications.csv`)

### 3.1 Schema & Semantic Definition

The loan applications dataset represents retail credit applications at the time of origination. All fields are pre-decision features observable prior to loan disbursement.

| Column Name | Storage Type | Null Count | Unique Count | Semantic Domain / Description | Model Handling Strategy |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `application_id` | `object` (string) | 0 | 1,000 | Primary application key (`APP-0001` .. `APP-1000`) | **Excluded** (Primary Identifier) |
| `age` | `int64` | 0 | 37 | Applicant age in completed years (Range: 22–58) | **Excluded** from predictors; preserved for bias audit |
| `income_annual_inr` | `int64` | 0 | 7 | Annual applicant income in Indian Rupees (INR) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `employment_type` | `object` (string) | 0 | 4 | Employment sector classification | **Retained** (Categorical Predictor: One-Hot Encoded) |
| `employment_years` | `int64` | 0 | 20 | Years in current employment / career | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `existing_loans` | `int64` | 0 | 4 | Count of currently active credit lines (0–3) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `existing_emi_inr` | `int64` | 0 | 754 | Current monthly debt service commitments (INR) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `credit_score` | `int64` | 0 | 313 | Bureau credit score at application (Range: 453–900) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `loan_amount_inr` | `int64` | 0 | 6 | Requested principal loan amount (INR) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `loan_purpose` | `object` (string) | 0 | 6 | Stated purpose of credit facility | **Retained** (Categorical Predictor: One-Hot Encoded) |
| `ltv_ratio` | `float64` | 0 | 56 | Loan-to-Value ratio (Range: 0.40–0.95) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `dti_ratio` | `float64` | 0 | 167 | Debt-to-Income ratio (Range: 0.00–3.06) | **Retained** (Numerical Predictor: Median Impute + Scaler) |
| `default_flag` | `int64` | 0 | 2 | Ground truth credit default indicator (0/1) | **Target Variable** (`y`) |

### 3.2 Target Variable Distribution (`default_flag`)

* **Positive Class (`default_flag = 1`):** 238 applicants (**23.80%**). Represents borrowers who defaulted on loan repayment obligations.
* **Negative Class (`default_flag = 0`):** 762 applicants (**76.20%**). Represents borrowers who successfully serviced or are servicing obligations without default.
* **Imbalance Ratio:** ~3.2:1 (Negative to Positive). This moderate class imbalance is handled via balanced class weights (`class_weight='balanced'`) in the baseline model and validation-tuned decision thresholds.

### 3.3 Privacy-Safe Summary Statistics

Aggregated five-number summaries and distribution statistics across the 1,000 application records:

| Numerical Feature | Min | 25th %ile | Median | Mean | 75th %ile | Max | Std Dev | Non-Default Mean (`y=0`) | Default Mean (`y=1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `age` | 22.0 | 31.75 | 41.0 | 40.69 | 50.0 | 58.0 | 10.74 | 40.77 | 40.44 |
| `income_annual_inr` | 300,000 | 420,000 | 840,000 | 1,093,800 | 1,800,000 | 2,400,000 | 725,651 | 1,248,425 | 598,740 |
| `employment_years` | 1.0 | 6.0 | 11.0 | 10.79 | 16.0 | 20.0 | 5.73 | 10.79 | 10.80 |
| `existing_loans` | 0.0 | 1.0 | 2.0 | 1.52 | 2.0 | 3.0 | 1.09 | 1.37 | 1.98 |
| `existing_emi_inr` | 0.0 | 2,497 | 11,960 | 13,850 | 22,987 | 44,976 | 11,887 | 11,924 | 20,014 |
| `credit_score` | 453.0 | 627.0 | 680.0 | 681.49 | 737.0 | 900.0 | 79.15 | 688.00 | 660.66 |
| `loan_amount_inr` | 100,000 | 300,000 | 1,000,000 | 1,585,900 | 2,000,000 | 5,000,000 | 1,732,011 | 1,315,354 | 2,452,101 |
| `ltv_ratio` | 0.40 | 0.53 | 0.66 | 0.67 | 0.81 | 0.95 | 0.16 | 0.67 | 0.66 |
| `dti_ratio` | 0.00 | 0.13 | 0.30 | 0.47 | 0.64 | 3.06 | 0.48 | 0.29 | 1.04 |

### 3.4 Categorical Attribute Distributions

1. **`employment_type`:**
   * `government`: 262 (26.20%)
   * `self_employed`: 261 (26.10%)
   * `contract`: 247 (24.70%)
   * `salaried`: 230 (23.00%)
2. **`loan_purpose`:**
   * `personal`: 201 (20.10%)
   * `vehicle`: 173 (17.30%)
   * `home`: 166 (16.60%)
   * `education`: 160 (16.00%)
   * `medical`: 156 (15.60%)
   * `business`: 144 (14.40%)

---

## 4. Dataset 2 Audit: Credit Risk Features (`finwise_credit_risk_features.csv`)

### 4.1 Schema & Purpose

The credit risk features dataset contains credit bureau inquiry features and an underwriting decision:

| Column Name | Storage Type | Null Count | Unique Count | Description & Category |
| :--- | :--- | :---: | :---: | :--- |
| `applicant_id` | `object` | 0 | 1,000 | Applicant key (`FW-0001` .. `FW-1000`) |
| `age` | `int64` | 0 | 44 | Range: 22–65 |
| `gender` | `object` | 0 | 2 | `Male`: 628 (62.8%), `Female`: 372 (37.2%) |
| `marital_status` | `object` | 0 | 3 | `Married`: 499 (49.9%), `Divorced`: 265 (26.5%), `Single`: 236 (23.6%) |
| `employment_type` | `object` | 0 | 4 | `Business`: 265, `Salaried`: 249, `Self-Employed`: 247, `Retired`: 239 |
| `years_employed` | `int64` | 0 | 31 | Range: 0–30 |
| `monthly_income_inr`| `int64` | 0 | 998 | Range: 15,000–99,800 |
| `existing_emis_inr` | `int64` | 0 | 985 | Range: 0–35,000 |
| `loan_amount_inr` | `int64` | 0 | 1,000 | Continuous loan amounts (50,000–1,000,000) |
| `loan_tenure_months`| `int64` | 0 | 7 | Discrete tenures: 12, 24, 36, 48, 60, 72, 84 |
| `loan_purpose` | `object` | 0 | 6 | `Home`, `Personal`, `Business`, `Education`, `Medical`, `Vehicle` |
| `cibil_score` | `int64` | 0 | 470 | Bureau score (300–900) |
| `num_credit_cards` | `int64` | 0 | 6 | Range: 0–5 |
| `num_existing_loans`| `int64` | 0 | 5 | Range: 0–4 |
| `defaults_last_2yr` | `int64` | 0 | 4 | Count of historical defaults (0–3) |
| `risk_category` | `object` | 0 | 3 | `High`: 465 (46.5%), `Low`: 280 (28.0%), `Medium`: 255 (25.5%) |
| `approved` | `int64` | 0 | 2 | Underwriting decision (0: 647 [64.7%], 1: 353 [35.3%]) |

### 4.2 Target Analysis & Rejection of `approved` as Default Target

* **Target Confusion Risk:** `approved` represents loan approval (underwriting acceptance vs. rejection), **not loan default**.
* **Survivorship / Selection Bias:** In credit lending, rejected borrowers (`approved = 0`) never receive a loan and therefore **cannot default**. Using `approved` as a proxy for default would severely bias the model by learning underwriting rules rather than true credit performance.
* **Ground Truth Absence:** There is no default flag (e.g. `is_default`, `chargeoff_flag`, or days past due) in this dataset.

---

## 5. Dataset 3 Audit: Fraud Transactions (`finwise_fraud_transactions.csv`)

### 5.1 Schema & Semantic Domain

The fraud dataset records digital transaction attempts across retail payment channels:
* **Dimensions:** 500 rows × 12 columns
* **Identifiers:** `txn_id` (`TXN-0001` .. `TXN-0500`, 500 unique); `borrower_id` (`BORR-0001` .. `BORR-0390`, 390 unique borrowers)
* **Target:** `is_fraud` (0: 441 [88.2%], 1: 59 [11.8%])
* **Fraud Typology (`fraud_type`):** 452 nulls (for non-fraud transactions and unclassified events), 23 Synthetic ID, 14 Account Takeover, 11 Identity Theft
* **Transactional Attributes:** `amount_inr` (mean 74,710 INR), `channel` (web, mobile, POS, etc.), `city`, `device_fingerprint_match` (binary), `ip_location_match` (binary), `time_since_last_txn_hrs`.

### 5.2 Strict Fraud Isolation Confirmation

* **Domain Incompatibility:** Credit default modeling predicts the **long-term solvency and repayment behavior** of a verified borrower over a multi-month or multi-year term. Transaction fraud detection predicts **unauthorized or counterfeit payment execution** in real-time milliseconds.
* **Label Conflation Risk:** Training a credit risk model on transaction fraud labels (`is_fraud`) would produce an invalid scoring engine that penalizes legitimate borrowers experiencing fraud while ignoring credit insolvency.
* **Governance Mandate:** In strict adherence to Task 1 instructions, `finwise_fraud_transactions.csv` is **completely isolated** from the credit risk baseline model.

---

## 6. Join Key Analysis & Blocker Documentation

### 6.1 Feasibility Investigation

We evaluated whether `finwise_loan_applications.csv` could be merged with `finwise_credit_risk_features.csv` to augment applications with demographic (`gender`) and bureau (`defaults_last_2yr`) attributes.

```
Applications Key Format: 'APP-0001', 'APP-0002', ..., 'APP-1000' (1,000 unique)
Features Key Format:     'FW-0001',  'FW-0002',  ..., 'FW-1000'  (1,000 unique)
```

1. **Exact Key Match Overlap:**
   $$\text{Exact Overlap} = \text{Set}(\text{application\_id}) \cap \text{Set}(\text{applicant\_id}) = \emptyset \quad (\mathbf{0 \text{ matches}}, \mathbf{0.0\%})$$
2. **Normalized Numeric Match Experiment (Stripping Prefixes `APP-` vs `FW-`):**
   To test if keys were mapped 1:1 by numeric sequence (`0001` to `0001`):
   * **Age exact match rate:** **2.00%** (consistent with pure random chance across ages 22–58)
   * **Employment type match rate:** **6.50%** (different category sets: `government`/`contract` vs. `Retired`/`Business`)
   * **Loan amount match rate:** **0.00%** (discrete amounts in applications vs continuous amounts in features)
   * **Income match rate:** **0.00%** (`income_annual_inr / 12` vs `monthly_income_inr`)
3. **Card-Order / Row-Index Alignment:**
   Aligning rows by index creates fabricated associations between unrelated synthetic records, destroying empirical integrity.

### 6.2 Governance Decision: Formal Merge Block

Per strict project instructions:
> *"If no valid join key exists or identifiers do not match, do not force a merge with fabricated mappings or row index alignment. Stop the merge, document the blocker, and use the single appropriate dataset for credit-default modeling."*

* **Decision:** The merge is **formally halted and blocked**.
* **Diagnostic Code Implemented:** `src.data_pipeline.verify_join_compatibility` programmatically verifies overlap and blocks joins when overlap is below 50%.
* **Authoritative Modeling Dataset:** `data/finwise_loan_applications.csv` is adopted as the single, authoritative dataset for credit risk modeling.

---

## 7. Data Quality & Preprocessing Readiness Summary

| Quality Check Dimension | Result | Remediation / Implementation Details |
| :--- | :---: | :--- |
| **Missing Value Handling** | Zero Nulls | Baseline pipeline includes `SimpleImputer(strategy='median')` for numerical columns and `SimpleImputer(strategy='constant', fill_value='missing')` for categorical columns to safeguard future batches. |
| **Outlier & Skew Treatment** | Present in DTI & Income | Numerical pipeline standardizes all numerical features using `StandardScaler()`. |
| **Categorical Encoding** | 2 Features (`employment_type`, `loan_purpose`) | Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to prevent production inference crashes on unseen categories. |
| **Data Leakage Safeguards** | Verified Leakage-Free | Identifiers (`application_id`) are excluded to avoid memorization. Demographic attribute (`age`) is excluded from model training to prevent direct demographic discrimination and preserved for post-model fairness audits. |
| **Splitting Strategy** | Stratified 70/15/15 | `src.preprocessing.split_dataset` executes a two-step stratified split preserving the 23.8% default rate across Train (700), Validation (150), and Holdout Test (150). |

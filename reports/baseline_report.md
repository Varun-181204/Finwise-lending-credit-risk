# FinWise Lending — Week 3: Baseline Model & Data Pipeline Report

**Project:** FinWise Lending — Credit Risk Assessment System
**Document:** Checkpoint 2 Baseline Modeling & Ingestion Evaluation Report
**Author:** ML Engineering Team
**Date:** November 2026
**Status:** Completed & Empirically Verified

---

## 1. Executive Summary

This report documents the end-to-end implementation and empirical verification of the **Week 3 Data Pipeline and Baseline Modeling Framework** for the FinWise Lending Credit Risk Assessment System.

Using the actual client datasets provided by Cynaris:
1. **Authoritative Dataset Selected:** `data/finwise_loan_applications.csv` (1,000 rows × 13 columns) was validated as the single authoritative dataset for credit risk modeling, containing the verified ground-truth binary target `default_flag` (762 non-defaults [76.2%], 238 defaults [23.8%]).
2. **Join Blocker Identified & Enforced:** An empirical audit demonstrated that `finwise_loan_applications.csv` (`application_id`) and `finwise_credit_risk_features.csv` (`applicant_id`) share **0 matching keys (0.0% overlap)** and represent independent synthetic populations. Merging them via arbitrary row index alignment was strictly blocked to prevent data corruption.
3. **Fraud Dataset Isolated:** `data/finwise_fraud_transactions.csv` (500 records of runtime payment fraud) was confirmed 100% isolated from origination credit risk modeling.
4. **Leakage-Free Preprocessing:** A scikit-learn `ColumnTransformer` (median imputation + scaling for numerical features; constant imputation + one-hot encoding for categorical features) was fitted **strictly on the 70% training partition**. Identifiers (`application_id`) and sensitive demographic attributes (`age`) were excluded from model predictors, preserving `age` for post-hoc fairness audits.
5. **Baseline Training & Threshold Optimization:** A balanced Logistic Regression baseline model was trained on the training partition (700 rows, 167 defaults). The operating decision threshold was tuned **exclusively on the validation partition (150 rows, 36 defaults)**, identifying $t^* = 0.40$ as the F1-optimal operating threshold.
6. **Empirical Benchmark Achievement:** Evaluated in a single pass on the untouched holdout test partition (150 rows, 35 defaults) at $t^* = 0.40$:
   * **Test ROC-AUC:** **0.9277 (92.77%)**
   * **Test Gini Coefficient:** **0.8554 (85.54%)** — **Significantly exceeds the client benchmark target of $\text{Gini} \ge 0.45$ by $+0.4054$**.
   * **Test PR-AUC:** **0.8363 (83.63%)**
   * **Test Recall:** **82.86%** (29 of 35 defaults correctly identified).
   * **Test Precision:** **56.86%** (F1-score: 0.6744).
7. **Automated Test Suite:** Verified via a comprehensive 38-test automated test suite in `pytest` with a 100% pass rate.

---

## 2. Dataset Provenance & Ingestion Analysis

### 2.1 Evaluated Datasets
Three datasets provided by Cynaris were audited under `data/`:

| Dataset | Dimensions | Primary Key | Target Column | Target Meaning | Included in Baseline? |
| :--- | :---: | :--- | :--- | :--- | :---: |
| `finwise_loan_applications.csv` | 1,000 × 13 | `application_id` (`APP-0001` .. `APP-1000`) | `default_flag` (0/1) | Observed credit default on loan obligation | **YES (Primary Source)** |
| `finwise_credit_risk_features.csv` | 1,000 × 17 | `applicant_id` (`FW-0001` .. `FW-1000`) | `approved` (0/1), `risk_category` | Origination underwriting approval/denial | **NO (Blocked Join)** |
| `finwise_fraud_transactions.csv` | 500 × 12 | `txn_id` (`TXN-0001` .. `TXN-0500`) | `is_fraud` (0/1) | Real-time payment transaction fraud | **NO (Strictly Isolated)** |

### 2.2 Target Selection Justification
* `default_flag` in `finwise_loan_applications.csv` directly captures borrower default behavior (238 defaults, 762 non-defaults).
* In contrast, `approved` in `finwise_credit_risk_features.csv` records whether an underwriter approved the application. Rejected applicants (`approved = 0`) never received disbursed capital and therefore could never default. Using `approved` as a default proxy would create severe selection bias, modeling underwriter heuristics rather than actual default risk.

### 2.3 Strict Fraud Dataset Isolation
`finwise_fraud_transactions.csv` tracks transactional payment fraud (e.g., synthetic identity, account takeover) at payment channels (POS, Web, Mobile). Transaction fraud operates on millisecond authorization intervals, whereas credit default measures long-term creditworthiness. Conflating transaction fraud with credit default introduces invalid target semantics and label leakage. It is 100% isolated.

---

## 3. Join Key Incompatibility & Blocker Documentation

### 3.1 Empirical Join Investigation
To determine whether bureau attributes (e.g., `cibil_score`, `defaults_last_2yr`) and demographics (`gender`) from `finwise_credit_risk_features.csv` could enrich `finwise_loan_applications.csv`, we evaluated key compatibility:

1. **Exact String Match:**
   $$\text{Set}(\text{application\_id}) \cap \text{Set}(\text{applicant\_id}) = \emptyset \quad (\mathbf{0 \text{ matches}}, \mathbf{0.0\%})$$
2. **Numeric Suffix Match (`0001` to `0001`):**
   Stripping prefixes (`APP-` vs `FW-`) and joining on numeric identifiers yielded:
   * **Age exact match rate:** **2.00%** (consistent with pure random chance across ages 22–58).
   * **Employment type match rate:** **6.50%** (`government`/`contract` in applications vs `Business`/`Retired` in features).
   * **Loan amount match rate:** **0.00%** (discrete brackets in applications vs continuous amounts in features).
   * **Annual vs. Monthly Income match rate:** **0.00%**.

### 3.2 Blocker Decision
The two credit files represent two independent synthetic populations with no common foreign key. In strict adherence to Task 3:
> *"If no valid join key exists or identifiers do not match, do not force a merge with fabricated mappings or row index alignment. Stop the merge, document the blocker, and use the single appropriate dataset for credit-default modeling."*

The merge was **halted and blocked**. The automated guard `src.data_pipeline.verify_join_compatibility` programmatically prevents invalid joins.

---

## 4. Feature Auditing & Leakage Prevention Architecture

In `src/preprocessing.py`, `audit_and_filter_features` categorizes all 12 input features into four governance categories:

```
Total Input Features: 12
├── Excluded Identifiers: ['application_id'] (prevents memorization / overfitting)
├── Excluded Sensitive Attributes: ['age'] (protected demographic attribute reserved for fairness audit)
├── Excluded Post-Origination Features: [] (none detected; all features observable at origination)
└── Retained Model Predictors: 10
    ├── Numerical (8): ['income_annual_inr', 'employment_years', 'existing_loans', 'existing_emi_inr',
    │                   'credit_score', 'loan_amount_inr', 'ltv_ratio', 'dti_ratio']
    └── Categorical (2): ['employment_type', 'loan_purpose']
```

### Preprocessing Pipeline Construction
* **Numerical Transformer:**
  * Imputation: `SimpleImputer(strategy='median')`
  * Scaling: `StandardScaler()` (zero mean, unit variance)
* **Categorical Transformer:**
  * Imputation: `SimpleImputer(strategy='constant', fill_value='missing')`
  * Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
* **Leakage-Free Rule:** Preprocessing transformers are fit **strictly on `X_train`**. Validation and test partitions are transformed using only training statistics.

---

## 5. Splitting Strategy & Baseline Model Training

### 5.1 Stratified 70/15/15 Splitting
Using `src.preprocessing.split_dataset` with `random_state=42`, the 1,000 records were partitioned into:

| Partition | Total Records | Non-Defaults (`y=0`) | Defaults (`y=1`) | Default Prevalence | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Split** | 700 (70.0%) | 533 | 167 | 23.86% | Model training and preprocessor fitting |
| **Validation Split** | 150 (15.0%) | 114 | 36 | 24.00% | Operating threshold selection |
| **Holdout Test Split** | 150 (15.0%) | 115 | 35 | 23.33% | Single-pass unbiased evaluation |

### 5.2 Baseline Model Architecture
* **Algorithm:** Logistic Regression (`solver='lbfgs'`, `C=1.0`, `max_iter=1000`, `random_state=42`).
* **Class Weighting:** `class_weight='balanced'` to inversely weight class frequencies and counteract the ~3.2:1 class imbalance.

### 5.3 Validation-Driven Operating Threshold Selection
The operating decision threshold was tuned **exclusively on the validation set** across candidate thresholds $t \in [0.10, 0.90]$ to maximize F1-score:
* **Default Threshold ($t = 0.50$):** Validation F1 = 0.7018, Recall = 83.33%, Precision = 61.22%.
* **Optimal Threshold ($t^* = 0.40$):** **Validation F1 = 0.7333**, **Recall = 91.67%**, **Precision = 61.11%**.
* $t^* = 0.40$ was locked for all subsequent holdout test evaluations.

---

## 6. Comprehensive Empirical Evaluation Results

### 6.1 Multi-Split Performance Summary

| Metric | Formula / Standard | Train Split ($t=0.50$) | Validation Split ($t^*=0.40$) | Holdout Test ($t^*=0.40$) | Holdout Test ($t=0.50$) | Client Target |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample Count** | $N$ | 700 | 150 | 150 | 150 | — |
| **Defaults ($y=1$)** | $N_{pos}$ | 167 | 36 | 35 | 35 | — |
| **Prevalence** | $N_{pos} / N$ | 23.86% | 24.00% | 23.33% | 23.33% | — |
| **ROC-AUC** | Area under ROC Curve | **0.9260** | **0.9345** | **0.9277** | **0.9277** | — |
| **Gini Coefficient** | $2 \times \text{ROC-AUC} - 1$ | **0.8520** | **0.8689** | **0.8554** | **0.8554** | **$\ge 0.45$ (PASS)** |
| **PR-AUC** | Average Precision | **0.8241** | **0.8323** | **0.8363** | **0.8363** | — |
| **Precision** | $TP / (TP + FP)$ | 0.6216 | 0.6111 | **0.5686** | 0.5909 | — |
| **Recall (Sensitivity)**| $TP / (TP + FN)$ | 0.8263 | 0.9167 | **0.8286** | 0.7429 | — |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | 0.7095 | 0.7333 | **0.6744** | 0.6582 | — |

### 6.2 Confusion Matrix (Holdout Test Partition, $N = 150$, $t^* = 0.40$)

```
                        Predicted Non-Default (0)    Predicted Default (1)       Total
Actual Non-Default (0)          93 (TN)                     22 (FP)               115
Actual Default (1)               6 (FN)                     29 (TP)                35
Total                           99                          51                    150
```

* **True Negatives (TN = 93):** 93 non-defaulting applicants correctly identified.
* **False Positives (FP = 22):** 22 creditworthy applicants flagged for review.
* **False Negatives (FN = 6):** Only 6 defaulting borrowers misclassified (miss rate = 17.14%).
* **True Positives (TP = 29):** 29 of 35 defaulting loans intercepted (**82.86% recall**).

### 6.3 Benchmark Verification: Client Gini Target $\ge 0.45$

The client specification establishes an acceptance threshold of **Test-set $\text{Gini} \ge 0.45$**.

$$\text{Test Gini} = 2 \times \text{ROC-AUC} - 1 = 2 \times 0.927702 - 1 = \mathbf{0.855404} \quad (\mathbf{85.54\%})$$

$$\text{Margin over Client Target} = 0.8554 - 0.4500 = \mathbf{+0.4054} \quad (\mathbf{+40.54\% \text{ buffer}})$$

The baseline Logistic Regression model comfortably surpasses the client milestone requirement prior to any non-linear feature engineering or gradient boosting.

---

## 7. Model Coefficient Interpretation & Feature Risk Drivers

Because all continuous features were standardized ($\mu=0, \sigma=1$) by `StandardScaler` fitted on `X_train`, the logistic regression coefficients indicate the change in log-odds of default per standard deviation shift:

| Feature Name | Feature Type | Model Coefficient ($\beta$) | Directional Risk Impact | Economic & Credit Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `dti_ratio` | Continuous (Standardized) | **+1.7595** | **Strong Risk Driver** | Highest positive driver: Elevated debt-to-income severely degrades debt-service capacity. |
| `credit_score` | Continuous (Standardized) | **-0.8268** | **Strong Protective Factor** | Higher bureau credit scores reflect historical repayment reliability. |
| `income_annual_inr` | Continuous (Standardized) | **-0.5782** | **Protective Factor** | Higher annual earnings provide financial buffer against cash flow shocks. |
| `existing_emi_inr` | Continuous (Standardized) | **+0.5403** | **Risk Driver** | Greater ongoing EMI commitments reduce disposable liquidity. |
| `loan_amount_inr` | Continuous (Standardized) | **+0.4614** | **Risk Driver** | Larger principal obligations increase absolute repayment burden. |
| `existing_loans` | Continuous (Standardized) | **+0.2933** | **Risk Driver** | Multiple active credit facilities indicate potential credit seeking behavior. |
| `employment_type_contract` | One-Hot Category | **-0.2090** | Moderate Protective | Moderate negative association in this sample. |
| `employment_type_self_employed` | One-Hot Category | **+0.1528** | Moderate Risk Driver | Self-employed revenue volatility increases delinquency vulnerability. |
| `loan_purpose_business` | One-Hot Category | **-0.1412** | Moderate Protective | Productive capital deployment associated with lower default rate. |
| `loan_purpose_education` | One-Hot Category | **+0.1261** | Moderate Risk Driver | Gestation lag before borrower income realization. |
| `employment_type_salaried` | One-Hot Category | **+0.1015** | Marginal | Baseline employment variation. |
| `loan_purpose_home` | One-Hot Category | **+0.0951** | Marginal | Collateralized mortgage facility. |
| `ltv_ratio` | Continuous (Standardized) | **-0.0813** | Marginal | Minimal independent signal after controlling for DTI and loan amount. |
| `employment_years` | Continuous (Standardized) | **-0.0784** | Marginal | Tenured career slightly dampens default likelihood. |
| `loan_purpose_personal` | One-Hot Category | **-0.0731** | Marginal | General unsecured facility. |
| `employment_type_government` | One-Hot Category | **-0.0465** | Marginal | Secure civil service employment. |
| `loan_purpose_vehicle` | One-Hot Category | **-0.0329** | Marginal | Auto loan facility. |
| `loan_purpose_medical` | One-Hot Category | **+0.0247** | Marginal | Emergency expense facility. |

* **Intercept ($\beta_0$):** **-0.9166** (reflecting lower baseline default prevalence under balanced weighting).

---

## 8. Model Limitations & Technical Risks

1. **Sample Size Constraints ($N = 1,000$):**
   * The holdout test set contains only 150 observations (35 defaults). While the test Gini of 0.8554 is statistically significant ($p < 0.001$), small sample sizes exhibit wider confidence intervals. Expanding to 10,000+ records in production will stabilize calibration.
2. **Loss of Bureau Attributes due to Join Incompatibility:**
   * Because `finwise_credit_risk_features.csv` could not be legitimately linked, valuable credit bureau signals (`defaults_last_2yr`, `num_credit_cards`, `loan_tenure_months`) were unavailable for this baseline.
3. **Synthetic Discretization Artifacts:**
   * In `finwise_loan_applications.csv`, `income_annual_inr` has only 7 unique values and `loan_amount_inr` has 6 unique values, indicating synthetic bracket binning. Non-linear models (XGBoost in Week 6) may exploit step-function boundaries; monotonic constraints should be evaluated.
4. **Fairness Audit Prerequisite:**
   * `age` was excluded from model training to prevent direct demographic disparate treatment. An audit across age brackets ($\le 30$, $31\text{--}50$, $> 50$) must be performed prior to production deployment.

---

## 9. Client Clarification Questions & Next Steps

### 9.1 Questions for Cynaris Solutions
1. **Foreign Key Mapping:** Can Cynaris provide an explicit mapping table connecting `application_id` (`APP-xxxx`) to `applicant_id` (`FW-xxxx`), or was `finwise_credit_risk_features.csv` generated as an independent synthetic test pack?
2. **Ground Truth Default Definition:** Please confirm whether `default_flag = 1` represents 90+ Days Past Due (DPD), formal write-off / charge-off, or any delinquency past 30 days.
3. **Target Deployment Threshold:** Does FinWise prioritize default capture (high recall / sensitivity, e.g. $t^* = 0.40$ yielding 82.86% recall) or application throughput (higher precision, e.g. $t = 0.50$ yielding 59.09% precision)?
4. **Protected Attributes:** Are gender and marital status permissible for disparate impact auditing, and are there specific fairness ratios (e.g., Four-Fifths rule / 80% rule) mandated by your compliance team?

### 9.2 Transition to Milestone 3 (Week 6)
With Checkpoint 2 complete:
* The data pipeline and baseline model are verified and committed in code and documentation.
* The baseline establishes a high benchmark ($\text{Gini} = 0.8554$).
* Week 6 will implement non-linear XGBoost training, hyperparameter tuning via validation splits, SHAP TreeExplainer waterfall visualizations, and FastAPI serving endpoints with 50 concurrent request handling.

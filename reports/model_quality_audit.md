# FinWise Lending — Independent Model Quality & Data Leakage Audit

**Project:** FinWise Lending — Credit Risk Assessment System
**Document:** Independent Model Quality, Target Leakage, and Methodology Audit
**Milestone:** Checkpoint 2 (Week 3: Data Pipeline + Baseline Model)
**Date:** November 2026
**Status:** Completed & Empirically Verified

---

## 1. Executive Verdict

This independent audit evaluated the methodological rigor, data leakage posture, evaluation integrity, and fairness audit readiness of the FinWise Lending Week 3 baseline implementation.

### Key Conclusions:
1. **Methodological Rigor — SOUND:**
   * Preprocessing (`ColumnTransformer`) is fit **strictly on the 70% training partition**. No test or validation statistics leak into imputation or scaling.
   * Decision threshold selection ($t^* = 0.40$) is conducted **exclusively on the 15% validation partition**. The holdout test set is evaluated in a single final pass.
   * Cross-partition contamination check confirms **0 shared records or duplicate profiles** across partitions.
2. **Target & Post-Origination Leakage — NEGATIVE (LEAKAGE-FREE):**
   * All 10 retained model features represent information observable at the moment of credit inquiry and application underwriting.
   * No post-origination fields (e.g., payment status, delinquency history post-disbursement, collections, late fees) exist in the modeling dataset.
   * High model performance (Test ROC-AUC: **0.9277**, Test Gini: **0.8554**) is not caused by target leakage. An ablation test removing the strongest predictor (`dti_ratio`) confirmed the baseline still achieves **Test ROC-AUC: 0.9091** and **Test Gini: 0.8181**, proving that predictive power is distributed across multiple underwriting attributes.
3. **Dataset Provenance & Synthetic Nature — LIMITATIONS IDENTIFIED:**
   * The primary modeling dataset (`finwise_loan_applications.csv`) lacks temporal collection metadata, vintage tracking, and a formal regulatory definition of default (e.g., 90+ DPD vs. charge-off).
   * Feature distributions exhibit hallmarks of synthetic generation (discrete step-function binning in income and loan amounts, and identical LTV distributions across secured and unsecured facilities).
4. **Fairness Audit Readiness — PARTIALLY BLOCKED:**
   * **Age Fairness:** **READY**. `age` is available, excluded from model training, and ready for disparate impact and equal opportunity auditing.
   * **Gender Fairness:** **BLOCKED**. `gender` does not exist in `finwise_loan_applications.csv`. It exists only in `finwise_credit_risk_features.csv`, which has **0% key overlap** with applications. Mapping gender across datasets without a verified foreign key would fabricate applicant attributes and is strictly prohibited.

---

## 2. Feature-by-Feature Leakage Assessment

Every column in `data/finwise_loan_applications.csv` was audited for temporal origination timing, derivation from target, and leakage risk:

| Column Name | Data Type | Model Status | Temporal Timing | Derivation from `default_flag`? | Leakage Assessment & Technical Rationale |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `application_id` | `object` | **Excluded** | Application submission | None | **SAFE (Excluded):** Primary applicant key (`APP-0001` .. `APP-1000`). Excluded to prevent memorization and privacy exposure. |
| `age` | `int64` | **Excluded** | Application submission | None | **SAFE (Excluded):** Stated applicant age. Excluded from predictors to prevent demographic disparate treatment; preserved for fairness auditing. |
| `income_annual_inr` | `int64` | **Retained** | Pre-origination (Verified) | None | **SAFE:** Verified annual earnings at origination. Non-defaulters average 1.25M INR vs. defaulters 598k INR. Legitimate economic capacity indicator. |
| `employment_type` | `object` | **Retained** | Application submission | None | **SAFE:** Sector classification (`government`, `salaried`, `self_employed`, `contract`). Pre-origination stated attribute. |
| `employment_years` | `int64` | **Retained** | Application submission | None | **SAFE:** Years in career/job at application time. No correlation with default ($r = 0.001$). Completely leakage-free. |
| `existing_loans` | `int64` | **Retained** | Pre-origination (Bureau) | None | **SAFE:** Count of open credit lines (0–3) on bureau file at inquiry time. Valid measure of credit depth. |
| `existing_emi_inr` | `int64` | **Retained** | Pre-origination (Bureau) | None | **SAFE:** Total existing monthly debt obligations from bureau at inquiry. For all 230 applicants with `existing_loans == 0`, `existing_emi_inr` is strictly 0. |
| `credit_score` | `int64` | **Retained** | Pre-origination (Bureau) | None | **SAFE:** Bureau score (453–900) pulled at underwriting inquiry. Protective factor ($\beta = -0.8268$). Standard underwriting signal. |
| `loan_amount_inr` | `int64` | **Retained** | Application submission | None | **SAFE:** Requested facility principal. Larger facilities exhibit higher default propensity ($r = 0.280$). Pre-decision input. |
| `loan_purpose` | `object` | **Retained** | Application submission | None | **SAFE:** Declared use of funds (6 categories). Observable prior to approval. |
| `ltv_ratio` | `float64` | **Retained** | Pre-origination (Collateral) | None | **SAFE:** Loan-to-Value ratio (0.40–0.95). Computed from requested loan amount divided by appraised collateral value prior to approval. |
| `dti_ratio` | `float64` | **Retained** | Pre-origination (Calculated) | None | **SAFE (Underwriting Metric):** Total proposed Debt-to-Income ratio (FOIR). Detailed mathematical verification below. |
| `default_flag` | `int64` | **Target** | Post-origination (Observed) | N/A | **TARGET:** Ground truth default flag. Separated upfront; never enters feature matrix $X$. |

### Deep-Dive: Forensic Examination of `dti_ratio`

Because `dti_ratio` is the strongest predictor in the baseline ($\beta = +1.7595$, univariate $\text{AUC} = 0.9096$), we subjected it to forensic mathematical analysis to determine whether it embeds post-origination information:

1. **Mathematical Relationship to Other Inputs:**
   In retail credit underwriting, the Fixed Obligation to Income Ratio (FOIR / Total DTI) is computed as:
   $$\text{Total DTI} = \frac{\text{Existing Monthly Debt Service} + \text{Proposed New Loan EMI}}{\text{Monthly Income}}$$
   Empirical testing on `data/finwise_loan_applications.csv` reveals:
   $$\text{Implied New Loan EMI} = \left(\text{dti\_ratio} - \frac{\text{existing\_emi\_inr}}{\text{income\_annual\_inr} / 12}\right) \times \left(\frac{\text{income\_annual\_inr}}{12}\right)$$
   Dividing the implied new EMI by `loan_amount_inr` yields an average ratio of **$0.007975 \approx 0.80\%$ per month** across all 1,000 applicants (corresponding to standard commercial amortizing terms, e.g., ~9.6% APR over 20 years).
2. **Timing Determination:**
   `dti_ratio` is constructed entirely from three pre-origination attributes: existing debt obligations (bureau), requested loan principal (application), and annual income (pay stub / tax filing). It reflects the applicant's **projected debt burden if the requested loan is approved**. It contains no post-disbursement delinquency or collections data.
3. **Collinearity vs. Leakage:**
   `dti_ratio` is collinear with its constituent parts, but Variance Inflation Factor (VIF) analysis shows all features remain under $3.5$ (well below the critical collinearity threshold of $5.0$).
4. **Model Sensitivity Ablation Test:**
   To prove that the baseline model does not collapse without `dti_ratio`, we trained a model excluding `dti_ratio`:
   * **Test ROC-AUC (without `dti_ratio`):** **0.9091**
   * **Test Gini (without `dti_ratio`):** **0.8181** (exceeds client target $\ge 0.45$ by $+0.3681$)
   * **Test Recall (without `dti_ratio`):** **88.57%** (F1: 0.6889)
   * **Conclusion:** Model discrimination is fundamentally sound and robust. `dti_ratio` is a valid, highly informative pre-origination underwriting metric, not an artifact of leakage.

---

## 3. Dataset Provenance & Data Generation Limitations

The audit identified several structural limitations in the dataset provenance that must be communicated to the client:

### 3.1 Lack of Temporal and Lifecycle Metadata
* **No Timestamp Columns:** The dataset contains no origination dates, application timestamps, maturity dates, or default dates.
* **Lack of Vintage / Cohort Tracking:** In banking, credit performance is tracked across vintages (e.g., Q1 2023 vs. Q3 2024) to account for macroeconomic shifts. The absence of dates prevents out-of-time (OOT) validation.
* **Unconfirmed Performance Window:** There is no documentation establishing the duration of performance observation (e.g., 12 months after disbursement vs. 24 months vs. full term).

### 3.2 Unconfirmed Definition of Default
* The target is labeled simply as `default_flag` (0 or 1).
* In regulatory credit risk (Basel III / IFRS 9 / CECL), default has precise operational definitions:
  * 90+ Days Past Due (DPD) on any material obligation.
  * Borrower bankruptcy, restructuring, or legal recovery.
  * Accounting charge-off / write-off.
* The current dataset does not specify whether `default_flag = 1` includes early delinquencies (30–60 DPD) or strictly severe write-offs.

### 3.3 Synthetic Artifacts
* **Discrete Binning:** `income_annual_inr` takes only 7 distinct values (300k, 420k, 600k, 840k, 1.2M, 1.8M, 2.4M INR). `loan_amount_inr` takes only 6 distinct values (100k, 300k, 500k, 1M, 2M, 5M INR). Real-world borrower populations exhibit continuous income and loan distributions.
* **Universal LTV Distribution:** `ltv_ratio` is present with identical distributions across all loan purposes (mean ~0.67 for personal loans, education loans, and home mortgages). In banking practice, unsecured facilities (personal, medical) do not carry an LTV ratio.

---

## 4. Train / Validation / Test Integrity Audit

We verified the splitting, transformation, and evaluation execution order against strict anti-leakage principles:

```
Full Ingested Dataset (N = 1,000, 238 Defaults [23.8%])
│
├── Step 1: Feature / Target Separation (X = 12 features, y = default_flag)
├── Step 2: Feature Audit & Exclusion (Drop application_id, age) -> Retained: 10 features
│
└── Step 3: Stratified 70/15/15 Splitting (random_state=42)
    ├── Train Partition (N = 700, 167 Defaults [23.86%])
    │   └── Fitted: SimpleImputer + StandardScaler + OneHotEncoder + LogisticRegression
    │
    ├── Validation Partition (N = 150, 36 Defaults [24.00%])
    │   └── Applied: Transformed via Train stats; Evaluated candidate thresholds t ∈ [0.10, 0.90]
    │   └── Selected: t* = 0.40 (Optimal Validation F1 = 0.7333)
    │
    └── Holdout Test Partition (N = 150, 35 Defaults [23.33%])
        └── Applied: Transformed via Train stats; Evaluated ONCE at t* = 0.40
        └── Achieved: Gini = 0.8554 (Single-pass unbiased evaluation)
```

### Partition Integrity Checks:
1. **Split Stratification:** Exact proportion matching. The default rate across all three splits is virtually identical (~23.8% $\pm 0.5\%$).
2. **Preprocessor Isolation:** The test `test_leakage_safety_fit_only_on_train` passed in pytest. Fitting transformers on the combined dataset or testing data is strictly prohibited in code.
3. **Threshold Selection Independence:** Threshold $t^* = 0.40$ was locked on `X_val`, `y_val` before `X_test` was evaluated.
4. **Cross-Partition Duplicate Check:** An exhaustive set comparison verified **0 identical feature vectors** across the train, validation, and test partitions.

---

## 5. Metric Verification

We independently verified the calculation of all reported metrics against their mathematical and regulatory definitions:

| Metric Name | Mathematical Definition | Verified Test Value | Status vs. Client Benchmark |
| :--- | :--- | :---: | :---: |
| **ROC-AUC** | $\int_0^1 \text{TPR}(F^{-1}(u)) \, du$ | **0.927702** | Excellent discrimination |
| **Gini Coefficient** | $2 \times \text{ROC-AUC} - 1$ | **0.855404** | **Target $\ge 0.45$ SATISFIED (+0.4054)** |
| **PR-AUC** | $\sum_k (R_k - R_{k-1}) P_k$ | **0.836318** | High precision-recall area under imbalance |
| **Operating Threshold ($t^*$)** | $\arg\max_t F_1(y_{val}, \hat{y}_{val}(t))$ | **0.40** | Validation-optimized |
| **Precision** | $TP / (TP + FP) = 29 / (29 + 22)$ | **0.568627 (56.86%)** | Operating balance |
| **Recall (Sensitivity)** | $TP / (TP + FN) = 29 / (29 + 6)$ | **0.828571 (82.86%)** | 29 of 35 defaulting loans intercepted |
| **F1-Score** | $2 \cdot (P \cdot R) / (P + R)$ | **0.674419** | Strong generalization from validation |

### Confusion Matrix Reconciliation ($N = 150$):
$$\begin{pmatrix} TN & FP \\ FN & TP \end{pmatrix} = \begin{pmatrix} 93 & 22 \\ 6 & 29 \end{pmatrix}$$
$$\text{Total Observations} = 93 + 22 + 6 + 29 = 150 \quad (\text{Exact Match})$$

---

## 6. Fairness Audit Readiness

The client specification mandates bias audits across **gender** and **age**. We evaluated the data readiness for both protected classes:

### 6.1 Age Fairness Audit — READY
* **Availability:** `age` is present in `finwise_loan_applications.csv` across all 1,000 records (Range: 22–58 years).
* **Treatment in Model:** Excluded from the model feature matrix $X$ to prevent direct disparate treatment.
* **Audit Viability:** Ready for Disparate Impact Ratio (DIR) and Equal Opportunity Difference (EOD) auditing across age brackets (e.g., Young $\le 30$, Mid-career $31\text{--}50$, Mature $> 50$) in Week 6.

### 6.2 Gender Fairness Audit — CRITICAL BLOCKER
* **Absence in Primary Dataset:** `finwise_loan_applications.csv` **does not contain a gender column**.
* **Presence in Unrelated Dataset:** `gender` is present in `finwise_credit_risk_features.csv` (62.8% Male, 37.2% Female).
* **Join Incompatibility:** As proven in the dataset audit, `application_id` (`APP-xxxx`) and `applicant_id` (`FW-xxxx`) share **0% overlap**, and prefix-stripped alignment exhibits only a 2.0% age match (pure random chance). The two files represent independent sample populations.
* **Governance Finding:** Fabricating a join between these files to attach `gender` to loan applications would result in synthetic, fabricated demographic assignments for real or benchmark applicants. This would produce legally invalid and misleading fairness metrics.
* **Audit Verdict:** **A gender bias audit cannot be conducted on the credit default model until Cynaris provides a valid cross-reference table or an updated loan applications dataset containing gender annotations.**

---

## 7. Issues Requiring Clarification from Cynaris

We recommend presenting the following five formal technical questions to Cynaris Solutions:

1. **Foreign Key Mapping for Demographic Features:**
   Can Cynaris provide a verified mapping key connecting `application_id` (`APP-xxxx`) in `finwise_loan_applications.csv` to `applicant_id` (`FW-xxxx`) in `finwise_credit_risk_features.csv`? Without this mapping, demographic `gender` and bureau variables (`defaults_last_2yr`) cannot be incorporated into the credit default model or fairness audits.
2. **Ground Truth Definition of `default_flag`:**
   Does `default_flag = 1` correspond to 90+ Days Past Due (DPD) (Basel standard), formal charge-off / write-off, or any delinquency past 30 days?
3. **Performance Observation Window:**
   Over what time horizon was default observed (e.g., 12 months, 24 months, or lifetime of the loan)? Are all loans in this dataset fully matured, or does the data include active loans?
4. **Underwriting Optimization Priority:**
   Does FinWise prioritize default prevention (maximizing recall, e.g., $t^* = 0.40$ yielding 82.86% recall with 22 false alarms) or underwriter throughput (maximizing precision, e.g., $t = 0.50$ yielding 59.09% precision with 18 false alarms)?
5. **Regulatory Jurisdiction & Compliance Thresholds:**
   What regulatory jurisdiction governs the deployment (e.g., US ECOA/FCRA, EU AI Act, RBI Guidelines), and what specific quantitative fairness criteria (e.g., 80% / Four-Fifths rule for Disparate Impact) are mandated by compliance?

---

## 8. Recommended Corrective Actions & Roadmap

1. **Preserve Current Baseline Implementation:**
   * Retain the existing 10-feature baseline Logistic Regression model (`Gini = 0.8554`). The implementation is mathematically sound, leakage-free, and fully reproducible.
2. **Document `dti_ratio` Characteristics in Model Governance:**
   * Formalize in the Model Risk Management (MRM) documentation that `dti_ratio` is a calculated total debt service metric incorporating requested loan principal and verified income, rather than post-origination debt.
3. **Execute Age Bias Audit in Week 6:**
   * Proceed with comprehensive Disparate Impact and Equal Opportunity audits across age cohorts using the preserved `age` attribute.
4. **Formally Log Gender Data Blocker:**
   * Log the absence of verified gender data as an external client dependency, pausing the gender fairness audit until Cynaris delivers verified demographic annotations.
5. **Prepare XGBoost Architecture for Synthetic Binning (Week 6):**
   * When implementing non-linear gradient boosting in Week 6, configure monotonic constraints on `dti_ratio` (positive monotonic) and `credit_score` (negative monotonic) to prevent trees from overfitting to synthetic step-function boundaries in `income_annual_inr` and `loan_amount_inr`.

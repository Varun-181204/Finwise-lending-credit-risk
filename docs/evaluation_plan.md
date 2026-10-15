# FinWise Lending — Model Evaluation Plan

## 1. Objective & Client Acceptance Target

The primary purpose of the FinWise Lending evaluation framework is to establish objective, reproducible, and business-grounded criteria for evaluating credit risk classification models. In lending, an effective model must balance discriminatory power (ranking risk accurately), probability calibration (predicting true default likelihood), financial profitability (minimizing credit losses while approving creditworthy applicants), and regulatory compliance (algorithmic fairness across demographic groups).

> [!IMPORTANT]
> **Client Acceptance Criteria:**
> The client's explicit performance target is a **Gini coefficient of $\ge 0.45$** (which corresponds to an **ROC-AUC of $\ge 0.725$**).
> **Note on Implementation Status:** This threshold is recorded here strictly as the client's target objective for future model training (Week 3), **not** as an achieved or verified experimental result. Model training has not yet commenced.

---

## 2. Target Column Confirmation Protocol

Before any machine learning training, feature selection, or label encoding is executed, the project team must formally confirm the target column with the client and domain underwriters.

### 2.1 Rationale for Strict Confirmation
Credit datasets often contain multiple candidate outcome columns that represent distinct business concepts:
* `loan_status` (e.g., categorical strings such as "Fully Paid", "Charged Off", "Current", "Late (31-120 days)", "In Grace Period").
* `is_default` or `default_flag` (binary 0/1 indicator).
* `delinquency_2yrs` (historical delinquency rather than future loan default).
* `charge_off_amount` (loss magnitude rather than binary occurrence).

### 2.2 Operational Confirmation Steps
1. **Candidate Column Inspection:** The exploratory pipeline will tabulate all candidate target columns and their unique value distributions.
2. **Definition of Default:** Establish the exact definition of credit default:
   - Does "Default" comprise solely "Charged Off" accounts, or does it include "Default" and "Late (31-120 days)"?
   - How are active ("Current") loans handled? In mature portfolios, immature active loans with insufficient observation windows are typically excluded or analyzed via survival analysis to prevent censorship bias.
3. **Formal Sign-off:** The confirmed column name and binary mapping dictionary (e.g., `{'Charged Off': 1, 'Default': 1, 'Fully Paid': 0}`) will be documented in a configuration manifest before Week 3 modeling begins.

---

## 3. Dataset Splitting & Stratification Strategy

To prevent data leakage, overfitting, and overly optimistic performance estimates, the dataset will be split into three strictly isolated subsets:

```
[ Entire Confirmed Dataset (100%) ]
  ├── Training Set (70%): Used for model fitting, feature encoding parameter calculation, and baseline training.
  ├── Validation Set (15%): Used for hyperparameter tuning, early stopping in XGBoost, and probability calibration fitting.
  └── Test Set (15%): Kept strictly blind until final evaluation. Used once per candidate model for client reporting.
```

### 3.1 Stratification
Because loan default is a minority event (typically 2% to 10% prevalence), random splitting can introduce variance in default rates across folds.
* **Stratified Sampling:** All train/val/test splits will be stratified on the confirmed target variable to guarantee identical class distributions across all three partitions.
* **Temporal Holdout Consideration:** If the dataset contains origination dates (`issue_date`), an **Out-Of-Time (OOT)** test split will be prioritized over random stratified splitting. Evaluating the model on the most recent 6–12 months of loans mirrors real-world production performance across evolving macroeconomic conditions.

---

## 4. Performance Metrics & Mathematical Formulations

### 4.1 Discrimination & Ranking Metrics

#### 1. ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
Measures the probability that the model ranks a randomly chosen defaulting applicant higher than a randomly chosen non-defaulting applicant:
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d\text{FPR}$$
* Value range: $0.5$ (random guess) to $1.0$ (perfect separation).

#### 2. Gini Coefficient (Somers' D)
The standard metric used in banking risk management and Basel Accord credit rating systems:
$$\text{Gini} = 2 \times \text{ROC-AUC} - 1$$
* Value range: $0.0$ (no discriminatory power) to $1.0$ (perfect ranking).
* **Target Benchmark:** $\text{Gini} \ge 0.45 \implies \text{ROC-AUC} \ge \frac{0.45 + 1}{2} = 0.725$.

#### 3. PR-AUC (Precision-Recall Area Under Curve / Average Precision)
Because credit risk involves significant class imbalance, ROC-AUC can be overly optimistic due to a large pool of true negatives. PR-AUC focuses directly on the minority class:
$$\text{PR-AUC} = \sum_{k=1}^N (R_k - R_{k-1}) P_k$$
PR-AUC will be calculated on all model candidates to measure precision trade-offs across realistic lending thresholds.

### 4.2 Classification Metrics at Selected Decision Threshold ($t$)

For any chosen decision threshold $t \in [0, 1]$ where an applicant is classified as predicted default if $P(\text{Default}) \ge t$:

* **Confusion Matrix:**
  * True Positives ($TP$): Actual defaulters correctly flagged as high-risk.
  * False Positives ($FP$): Non-defaulters incorrectly flagged as high-risk (loss of profitable business).
  * True Negatives ($TN$): Non-defaulters correctly approved.
  * False Negatives ($FN$): Actual defaulters incorrectly approved (direct credit loss / charge-off).

* **Precision:**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  Measures what fraction of rejected high-risk applicants would have actually defaulted.

* **Recall (Sensitivity / True Positive Rate):**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  Measures the fraction of total bad loans successfully intercepted.

* **F1-Score:**
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Probability Calibration & Brier Score

In credit risk, ranking alone is insufficient. If a model predicts a 10% probability of default, exactly 10 out of 100 such borrowers must default empirically. Accurate probabilities are required to estimate **Expected Loss (EL)**:
$$\text{Expected Loss} = \text{PD} \times \text{EAD} \times \text{LGD}$$
where:
* $\text{PD}$ = Probability of Default.
* $\text{EAD}$ = Exposure at Default (loan balance).
* $\text{LGD}$ = Loss Given Default (typically 45%–75% for unsecured consumer credit).

### 5.1 Brier Score
Measures the mean squared difference between predicted default probabilities and actual binary outcomes ($y_i \in \{0, 1\}$):
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$$
Lower values indicate superior calibration.

### 5.2 Reliability Diagrams & Recalibration Methods
* **Reliability Diagrams (Calibration Curves):** Predicted probabilities are binned into deciles, and observed default frequencies are plotted against mean predicted probabilities.
* **Calibration Algorithms:** If raw tree outputs deviate from the diagonal, **Platt Scaling (logistic calibration)** or **Isotonic Regression** will be fit on the validation partition and tested on the holdout partition.

---

## 6. Business-Cost Threshold Optimization

Rather than using an arbitrary default threshold ($t = 0.50$), FinWise will optimize the decision threshold based on an explicit business cost-benefit matrix:

| Metric | Business Realization | Cost / Revenue Impact |
| :--- | :--- | :--- |
| **True Negative ($TN$)** | Approved applicant who repays loan in full | Net profit: $+ (\text{Loan Amount} \times \text{Interest Margin})$ |
| **True Positive ($TP$)** | Defaulter correctly declined | Zero loss: Avoided credit default loss |
| **False Positive ($FP$)** | Creditworthy applicant incorrectly declined | Opportunity cost: $- (\text{Loan Amount} \times \text{Interest Margin})$ |
| **False Negative ($FN$)** | Defaulter incorrectly approved | Direct credit loss: $- (\text{Loan Amount} \times \text{LGD})$ |

### Expected Profit Function:
$$\text{Net Business Value}(t) = \sum_{i \in \text{Approved}(t)} \left[ (1 - y_i) \cdot \text{Profit} - y_i \cdot \text{Loss} \right]$$
Because credit losses on defaulted loans ($FN$) typically dwarf individual interest margins by a factor of $4\times$ to $10\times$, the optimal threshold $t^*$ will be selected at the peak of the expected net financial value curve on the validation set.

---

## 7. Algorithmic Fairness & Bias Evaluation across Gender and Age

FinWise Lending commits to responsible, non-discriminatory credit underwriting. As specified in client requirements, fairness and bias audits will be conducted systematically across **Gender** and **Age** cohorts where demographic features are available in the dataset.

### 7.1 Key Fairness Metrics

#### 1. Disparate Impact Ratio (DIR)
Evaluates whether the favorable evaluation rate (classification as Low/Medium risk) for an unprivileged demographic subgroup is substantially lower than that of the reference subgroup:
$$\text{DIR} = \frac{P(\hat{Y} = 0 \mid A = \text{Unprivileged})}{P(\hat{Y} = 0 \mid A = \text{Privileged})}$$
*(where $\hat{Y} = 0$ signifies low/medium risk classification).*
* **Audit Benchmark:** A benchmark threshold of $\text{DIR} \ge 0.80$ is typically tracked as an initial indicator of potential disparate impact. Specific regulatory threshold mandates depend on operating jurisdiction and require client and compliance confirmation.

#### 2. Equal Opportunity Difference (EOD)
Evaluates whether creditworthy borrowers ($Y = 0$) experience equal rates of favorable evaluation regardless of demographic group:
$$\text{EOD} = |P(\hat{Y} = 0 \mid A = \text{Unprivileged}, Y = 0) - P(\hat{Y} = 0 \mid A = \text{Privileged}, Y = 0)|$$
Target is to minimize $\text{EOD} < 0.05$ across gender and age cohorts.

---

## 8. Analytical Limitations & Governance Constraints

1. **Jurisdiction-Specific Regulatory Mandates:**
   * *Governance Note:* Fair lending and anti-discrimination standards differ substantially across legal jurisdictions (e.g., US ECOA/FCRA regulations, European Union AI Act and GDPR directives, or regional banking supervisory rules). All specific compliance mandates must be formally confirmed with the client's compliance leadership.
   * *Explainability Limitations:* SHAP provides mathematical feature attributions; SHAP alone does not constitute or guarantee regulatory or legal compliance.
2. **Missing Demographic Data:**
   * *Limitation:* Depending on jurisdiction and dataset provenance, protected demographic characteristics may be partially unrecorded or subject to collection restrictions.
   * *Mitigation:* Audits will focus on documented, legally permissible fields (specifically age and gender cuts). Proxy modeling will not be applied without explicit compliance approval.
3. **Small Subgroup Sample Sizes:**
   * *Limitation:* Subgroups with limited sample representation can yield wide confidence intervals on fairness ratios.
   * *Mitigation:* Bootstrap resampling (1,000 iterations) will be used to compute 95% confidence intervals for all subgroup metrics.
4. **Reject Inference Bias:**
   * *Limitation:* Historical loan datasets only reflect repayment performance for applicants who were previously approved. The true default behavior of previously declined applicants is unobserved.
   * *Mitigation:* Document this selection bias explicitly in validation documentation and risk registers.

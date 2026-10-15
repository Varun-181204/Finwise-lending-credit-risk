# FinWise Lending — Project Risks & Assumptions

## 1. Overview

Deploying machine learning models in credit underwriting introduces enterprise-level financial, regulatory, and technical risks. This document catalog's foundational assumptions, identifies project risks, evaluates their potential severity and likelihood, and establishes mitigation actions for the FinWise Lending credit risk assessment project.

---

## 2. Foundational Project Assumptions

1. **Dataset Independence & Scope:**
   - The client provided three distinct data-pack source URLs:
     1. `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_loan_applications.csv`
     2. `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_credit_risk_features.csv`
     3. `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_fraud_transactions.csv`
   - **Crucial Assumption:** The project team **does not assume** these three datasets serve identical prediction tasks or share a common key schema without empirical inspection. For example, fraud transaction detection is an entirely distinct operational problem from credit default risk. Each dataset must be independently audited, profiled, and mapped.
2. **Target Variable Undetermined in Week 1:**
   - We do not assume or invent target column names or definitions prior to actual inspection. Target variable confirmation with client underwriters is mandatory before Week 3 model training.
3. **No Invented Scoring or Cutoff Rules:**
   - The project avoids unconfirmed assumptions such as 300–850 credit score mappings, A–E risk bands, arbitrary PD thresholds, or automatic approve/decline rules. The architecture adheres to the client-specified **Low, Medium, and High** risk bands, with thresholds to be determined through empirical analysis and client underwriting policy.
4. **Jurisdiction-Agnostic Regulatory Stance:**
   - Specific fair lending and transparency regulations depend on the jurisdiction where FinWise operates. Regulatory compliance requirements must be formally confirmed with client compliance leadership. TreeSHAP provides mathematical feature attribution, but explainability alone does not guarantee legal compliance.
5. **Prototype Scope & Human-in-the-Loop:**
   - The platform developed across Weeks 1–12 serves as an underwriter decision-support prototype. It does not enforce automated approval or decline decisions.

---

## 3. Risk Matrix & Mitigations

| Risk ID | Category | Risk Description | Severity | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | Data Quality | **Data Leakage from Post-Origination Fields**<br>Features populated after loan approval (e.g., late fees, collections, days past due) leak future information into training. | Critical | High | Strict temporal feature auditing. Any variable with a collection or timestamp after loan origination is blacklisted. |
| **RSK-02** | Methodology | **Reject Inference Bias (Survivorship)**<br>Training data only reflects outcomes of previously approved loans. The performance of previously rejected applicants is unobserved. | High | High | Document survivorship bias explicitly in model risk documentation. Explore semi-supervised reject inference or bounding methods if historical rejection scores exist. |
| **RSK-03** | Governance | **Jurisdiction-Specific Regulatory Non-Compliance**<br>Model decisions lack transparency or fail to meet the specific consumer credit explanation mandates of the operating jurisdiction. | Critical | Medium | Integrate TreeSHAP to produce a clear waterfall chart per prediction for underwriter decision support. Conduct formal compliance alignment with client legal officers; recognize that SHAP alone does not guarantee legal compliance. |
| **RSK-04** | Methodology | **Class Imbalance & Extreme Loss Asymmetry**<br>Default events represent a small fraction of loans. Models optimizing generic accuracy fail to catch defaults, causing severe credit losses. | High | High | Optimize decision thresholds using financial cost matrices rather than default $0.5$ cutoffs. Evaluate PR-AUC and Gini ($\ge 0.45$ target). |
| **RSK-05** | Engineering | **Production Concurrency Bottlenecks**<br>System throughput degrades under concurrent API evaluation, failing to achieve the client's **50 concurrent requests** target. | High | Medium | Architect FastAPI with asynchronous worker pooling, pre-warm models in memory, optimize TreeSHAP background evaluations, and run load tests under 50 simultaneous connections. |
| **RSK-06** | Fairness | **Disparate Impact & Proxy Discrimination**<br>Unintentional discrimination against protected classes (specifically gender and age) through correlated proxy features. | High | Medium | Execute automated fairness audits using Disparate Impact Ratio and Equal Opportunity Difference across gender and age cohorts. |
| **RSK-07** | Operations | **Dataset Schema Incompatibility**<br>Client datasets do not join cleanly or have inconsistent primary keys. | Medium | Medium | Automated data profiling via `src.data_pipeline` will inspect key overlap and null rates prior to merging. |
| **RSK-08** | Security | **Accidental Exposure of PII / Sensitive Financial Data**<br>Sensitive customer financial records committed to version control. | Critical | Low | Comprehensive `.gitignore` excluding all raw data formats (`.csv`, `.parquet`, `.xlsx`), strict repository scanning, and data anonymization. |

---

## 4. Next Milestone Preconditions (Week 3 Gateways)

Before any modeling scripts are authored in Week 3, the following gates must be cleared:
1. Local download and data-quality inspection of the client datasets using `src.data_pipeline.load_csv` and `generate_data_quality_summary`.
2. Formal sign-off on the confirmed target variable definition (`loan_status` vs `is_default`).
3. Validation that no post-origination features exist in the candidate feature pool.

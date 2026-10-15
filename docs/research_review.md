# FinWise Lending — Research Review

## 1. Executive Summary

Credit risk modeling operates at the intersection of statistical precision, business solvency, and strict legal compliance. Unlike generic machine learning domains where predictive accuracy alone suffices, financial lending models directly determine credit access, pricing, and loan terms for individual consumers and businesses. Consequently, algorithmic systems must be not only predictive and performant on tabular data, but also rigorously calibrated, explainable, robust against data leakage, and fair across demographic groups.

This document reviews the primary foundational literature and open-source infrastructure underpinning the FinWise Lending credit-risk pipeline:
1. **XGBoost: A Scalable Tree Boosting System** (Chen & Guestrin, 2016)
2. **A Unified Approach to Interpreting Model Predictions [SHAP]** (Lundberg & Lee, 2017)
3. **Official XGBoost Repository** (DMLC / Chen et al., 2014–Present)

Following the individual source evaluations, we provide an in-depth analysis of four pillars critical to credit-risk modeling: **Explainability**, **Data Leakage**, **Class Imbalance**, and **Algorithmic Fairness**.

---

## 2. Source-by-Source Literature & Tool Reviews

### Source 1: XGBoost Research Paper

* **Title:** XGBoost: A Scalable Tree Boosting System
* **Authors:** Tianqi Chen, Carlos Guestrin (University of Washington)
* **Year:** 2016 (Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794)
* **URL:** [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
* **Problem Addressed:** Gradient boosted decision trees (GBDT) are among the most effective algorithms for tabular, non-linear predictive problems. However, existing implementations struggled to scale to billions of examples, exhibited high memory footprints, lacked native sparsity handling for real-world missing data, and suffered from computational bottlenecks in exact greedy split searching.
* **Key Ideas:**
  1. *Regularized Objective Function:* Incorporates explicit $L_1$ and $L_2$ regularization penalties on leaf weights ($\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2$) directly into the second-order Taylor expansion objective, penalizing model complexity and reducing variance.
  2. *Weighted Quantile Sketch:* Proposes a novel theoretical framework to handle weighted datasets where instance weights correspond to the hessian (curvature) of the loss function, enabling fast candidate split generation with formal error bounds.
  3. *Sparsity-Aware Split Finding:* Employs a default split direction for missing or sparse values learned directly from the data, bypassing ad-hoc imputation and speeding up computation.
  4. *System Architecture Optimizations:* Introduces columnar block structures in memory, cache-aware prefetching to minimize CPU stalls, and out-of-core block compression and sharding across disk storage.
* **Relevance to FinWise:**
  - Loan applicant data is tabular, heterogeneous, and contains non-linear interactions (e.g., interaction between Debt-to-Income ratio and Revolving Credit Utilization).
  - Financial datasets frequently have legitimate missing fields (e.g., missing length of credit history or omitted secondary applicant fields). XGBoost handles these natively without introducing artificial imputation bias.
  - The second-order optimization and regularization are vital for stabilizing predictions on noisy financial targets.
* **Limitations:**
  - Standard tree building is greedy and can overfit rare, unrepresentative sub-cohorts if tree depth is unchecked.
  - Generates complex step-wise piecewise constant decision surfaces that do not naturally extrapolate continuous monotonic relationships (e.g., default risk should monotonically decrease as credit score increases) unless explicit monotonic constraints are configured.
  - High sensitivity to hyperparameter configurations (e.g., `learning_rate`, `max_depth`, `subsample`, `min_child_weight`).
* **Questions for Future Work:**
  - Can monotonic constraints (`monotone_constraints`) be strictly enforced on credit-scoring features to prevent economically irrational loan decisions?
  - How does tree depth affect probability calibration when converting margins to calibrated probabilities of default (PD)?

---

### Source 2: SHAP Research Paper

* **Title:** A Unified Approach to Interpreting Model Predictions
* **Authors:** Scott M. Lundberg, Su-In Lee (University of Washington)
* **Year:** 2017 (Advances in Neural Information Processing Systems 30 - NeurIPS 2017, pp. 4765–4774)
* **URL:** [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
* **Problem Addressed:** Complex ensemble and deep learning models operate as "black boxes." Existing local explanation methods (such as LIME, DeepLIFT, and Layer-Wise Relevance Propagation) lacked unified theoretical foundations, produced inconsistent attributions, and could assign lower importance to a feature even when the model's reliance on that feature strictly increased.
* **Key Ideas:**
  1. *Unified Additive Feature Attribution:* Demonstrates that six previously distinct interpretability approaches are all approximations of a single class of additive feature attribution methods.
  2. *Shapley Value Foundations:* Derives attributions from cooperative game theory (Shapley, 1953), proving that Shapley values are the *only* set of attributions that simultaneously satisfy three essential properties:
     - **Local Accuracy (Efficiency):** The sum of feature attributions equals the difference between the model's output for that instance and the base expected value: $f(x) = \phi_0 + \sum_{i=1}^M \phi_i$.
     - **Missingness:** Features missing from the input receive zero attribution ($\phi_i = 0$ when $x'_i = 0$).
     - **Consistency (Monotonicity):** If a model changes so that the marginal contribution of feature $i$ increases or stays the same regardless of other features, that feature's attribution cannot decrease.
  3. *TreeSHAP Formulation:* Introduces an algorithm capable of computing exact Shapley values for tree ensembles in polynomial time ($O(TLD^2)$, where $T$ is the number of trees, $L$ is maximum leaves, and $D$ is maximum tree depth) rather than exponential time.
* **Relevance to FinWise:**
  - Depending on the operating jurisdiction, lending regulations may mandate transparency into credit decisions, requiring lenders to provide the principal factors influencing adverse outcomes (e.g., adverse action notices in the US under ECOA/FCRA, explanation requirements under GDPR in the EU, or regional financial conduct rules).
  - *Governance & Compliance Note:* Regulatory requirements depend on the operating jurisdiction and require formal client and compliance team confirmation. SHAP provides mathematically rigorous feature attribution based on cooperative game theory, but SHAP alone does not guarantee legal or regulatory compliance.
  - TreeSHAP enables FinWise to extract mathematically consistent, instance-level top driving risk factors for individual loan applications, generating a clear **SHAP waterfall chart per prediction** for underwriter decision support.
* **Limitations:**
  - Standard TreeSHAP assumes feature independence when calculating expectations over background samples, which can allocate attribution across highly correlated financial features (e.g., `total_debt` and `monthly_installment`).
  - Explanations reflect model behavior, not causal relationships in the real world. A high SHAP value indicates what the model used, not necessarily an actionable counterfactual recommendation for the applicant.
* **Questions for Future Work:**
  - How can we convert continuous SHAP feature contributions into standardized, plain-language reason summaries aligned with client compliance policies?
  - Does using a localized background distribution (e.g., representative medians) alter the stability of adverse action codes across re-trainings?

---

### Source 3: Official XGBoost Open-Source Repository

* **Title:** XGBoost: Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library
* **Organization / Maintainers:** DMLC (Distributed Machine Learning Community) / Chen, He, Khotilovich, Tang, et al.
* **Year:** Initial Release 2014; Actively Maintained (2026)
* **URL:** [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)
* **Problem Addressed:** Translating theoretical tree boosting research into a hardened, portable, high-throughput, cross-language production library suitable for enterprise deployments and distributed computing.
* **Key Ideas:**
  1. *Versatile Tree Methods:* Implementation of `hist` (histogram-based building) matching or exceeding LightGBM speeds for high-dimensional tabular datasets.
  2. *Monotonic Constraints:* Native support for directional constraints (`monotone_constraints`), enforcing that predicted risk strictly increases or decreases as a specific feature varies.
  3. *Production Serialization:* Support for standardized JSON and UBJSON model serialization formats, guaranteeing schema stability and backward compatibility across versions.
  4. *Scikit-Learn API & Integration:* First-class `XGBClassifier` adhering to standard `fit`/`predict_proba` conventions, enabling seamless scikit-learn pipelines and cross-validation workflows.
* **Relevance to FinWise:**
  - Provides the robust engine for training the core credit scoring model.
  - JSON serialization allows the trained model artifact to be loaded reliably inside the FastAPI service without pickle security vulnerabilities.
  - Native CPU multithreading ensures low-latency inference (<50ms per applicant) required by the React loan evaluation dashboard.
* **Limitations:**
  - In environments with constrained container memory, histogram building across very wide feature sets can cause memory spikes.
  - Categorical feature handling (experimental `enable_categorical`) must be explicitly controlled to avoid silent type coercion bugs.
* **Questions for Future Work:**
  - Should the inference engine export to ONNX runtime or utilize native XGBoost C API bindings inside FastAPI to minimize serving latency?
  - How do we structure the automated retraining pipeline to validate JSON model schemas before deployment?

---

## 3. Core Credit-Risk Modeling Challenges

### 3.1 Explainability, Governance, and Regulatory Context

In consumer lending, an unexplainable model carries substantial financial, legal, and operational risks:
* **Jurisdiction-Specific Requirements:** Transparency obligations vary significantly across regulatory jurisdictions (e.g., adverse action notice rules under ECOA/FCRA in the United States, automated decision-making explanation rights under GDPR Article 22 in Europe, or consumer credit standards in other jurisdictions). All specific compliance mandates must be formally confirmed with the client's legal and compliance officers.
* **Limitations of Algorithmic Explainability:** While TreeSHAP delivers mathematically sound local and global feature attribution grounded in cooperative game theory, **SHAP alone does not guarantee legal or regulatory compliance**. Algorithmic explanations must be supported by comprehensive institutional model governance, policy alignment, and underwriter oversight.
* **Internal Governance and Auditability:** Model risk management teams (e.g., following supervisory guidance such as SR 11-7 or regional equivalents) require independent validation of feature importance, directional sanity, and boundary stability.
* **Local vs. Global Interpretability:**
  * *Global Explainability* (e.g., summary SHAP plots) informs risk officers whether the model aligns with established credit principles.
  * *Local Explainability* (a **SHAP waterfall chart per prediction**) powers individual applicant explanations in the FinWise underwriter dashboard, providing transparent decision support.

### 3.2 Data Leakage in Loan Origination

Data leakage occurs when information unavailable at the time of origination is included in training, leading to unrealistically optimistic validation metrics and catastrophic production failures.

* **Post-Origination Leakage:** Features that describe loan performance *after* the loan is disbursed—such as `repayment_status_month_3`, `collections_flag`, `days_past_due`, or `total_interest_paid`—must be strictly excluded from origination risk models.
* **Target Leakage:** Derived variables that partially encode the target variable (e.g., `account_closed_by_lender`).
* **Temporal / Split Leakage:** Credit environments undergo macroeconomic cycles (interest rate shifts, inflation). Standard random K-Fold cross-validation can leak temporal correlation between concurrent applications. Temporal validation (train on historical period, validate on subsequent period) or strictly isolated stratified splits are mandatory.

### 3.3 Class Imbalance in Credit Default Datasets

In prime and near-prime lending portfolios, default is a rare event—typically occurring in only 2% to 8% of applications.

* **Consequences of Unaddressed Imbalance:** Standard loss functions minimizing binary cross-entropy can achieve 95% accuracy by simply predicting non-default for every applicant, rendering the model useless for risk differentiation.
* **Mitigation Strategies in XGBoost:**
  * Parameter `scale_pos_weight = (count(negative) / count(positive))` scales the gradients for the minority (default) class, penalizing false negatives.
  * *Calibration Caution:* While synthetic oversampling (SMOTE) or downsampling alters the empirical prior, it distorts the predicted probabilities. In lending, because default probabilities directly feed into expected loss calculations ($\text{Expected Loss} = \text{PD} \times \text{EAD} \times \text{LGD}$), preserving or re-calibrating the true posterior probability using isotonic regression or Platt scaling is essential.

### 3.4 Fairness, Bias, and Disparate Impact

Lending algorithms can inadvertently replicate or amplify historical biases present in financial data.

* **Protected Attributes:** Attributes such as gender and age are common focal points of fair-lending oversight.
* **Proxy Discrimination:** Even if protected attributes are excluded from the training matrix, correlated proxy variables (e.g., location codes or certain employment categories) can introduce bias.
* **Fairness Metrics for FinWise:**
  1. *Disparate Impact Ratio (DIR):* Ratio of selection rate for the protected group compared to the non-protected group. (An 80% ratio is commonly utilized as an initial audit benchmark, but specific thresholds require client compliance confirmation).
  2. *Equal Opportunity:* Equality of True Positive Rates (or False Negative Rates) across demographic subgroups, ensuring creditworthy applicants have equal probability of favorable evaluation regardless of demographic group.
* **Audit Mandate:** FinWise must conduct systematic bias audits across **gender** and **age** cohorts while actively documenting data limitations.

---

## 4. Synthesis & Research Recommendations for FinWise

| Area | Week 1 Recommendation | Future Week Implementation |
| :--- | :--- | :--- |
| **Model Engine** | Document XGBoost parameters, regularized loss, and monotonic constraints. | Week 3: Train baseline & XGBoost classifier. |
| **Discrimination Target** | Record test-set Gini $\ge 0.45$ (ROC-AUC $\ge 0.725$) as client target. | Week 3 & 6: Holdout evaluation benchmark. |
| **Explainability** | Standardize on TreeSHAP for per-prediction waterfall charts. | Week 6: Integrate SHAP waterfall generation. |
| **Risk Classification** | Categorize applicants into Low, Medium, and High risk tiers. | Week 6: Risk banding calibrated with client policy. |
| **Pipeline Hygiene** | Enforce strict schema separation, leakage exclusion, and no-data-dropping ingestion. | Week 3: Preprocessing and feature engineering. |
| **Fairness & Bias** | Mandate systematic audits across gender and age cohorts. | Week 6: Fairness audit report generation. |
| **Serving Architecture** | Design asynchronous FastAPI service to handle 50 concurrent requests. | Week 9: API deployment with load testing. |
| **Presentation** | Provide interactive React underwriter decision-support dashboard. | Week 12: React frontend with waterfall charts. |

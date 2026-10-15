# FinWise Lending — Data Directory

## 1. Overview & Data Privacy Policy

The `data/` directory contains localized working copies of client datasets used for model exploration, validation, and training. 

> [!CAUTION]
> **Strict Data Privacy Mandate:**
> Under no circumstances should raw, processed, or intermediate client datasets be committed to Git. The root `.gitignore` is configured to exclude all `.csv`, `.parquet`, `.feather`, `.xlsx`, and `.zip` files.
> All applicant datasets contain confidential loan-level financial data. Treat all local files with strict security and privacy protocols.

---

## 2. Client Data-Pack Sources

The client provided three candidate data-pack URLs for this engagement:

1. **Loan Applications Data:**
   `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_loan_applications.csv`
2. **Credit Risk Features Data:**
   `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_credit_risk_features.csv`
3. **Fraud Transactions Data:**
   `https://cynarissolutions.com/candidate/client-project-pack/aiml/finwise_fraud_transactions.csv`

### Critical Analytical Note:
* **Task Independence:** Do **not** assume these three datasets serve the same prediction task or share the same granularity. Fraud transaction detection is operationally distinct from loan origination credit default risk.
* **No Invented Columns:** Column headers, data types, and target labels must be determined through empirical loading and client confirmation, not guessed or hardcoded in advance.
* **Pre-Week 3 Inspection Requirement:** Before commencing Week 3 feature engineering and model training, data scientists must inspect each dataset using `src.data_pipeline`, verify primary keys, evaluate missing values, and formally confirm the target variable with stakeholders.

---

## 3. Schema & Data-Quality Inspection Instructions

To inspect and profile a local dataset safely without mutating or dropping records, use the provided `src.data_pipeline` tools:

```python
from pathlib import Path
from src.data_pipeline import load_csv, generate_data_quality_summary

# 1. Define local file path
data_path = Path("data/finwise_loan_applications.csv")

# 2. Load dataset safely (validates non-empty, checks formatting)
df = load_csv(data_path)

# 3. Generate non-destructive quality profile
summary = generate_data_quality_summary(df)

# 4. Review quality metrics
print(f"Total Rows: {summary['row_count']}")
print(f"Total Columns: {summary['column_count']}")
print(f"Duplicate Rows: {summary['duplicate_rows']}")
print("Missing values per column:", summary['missing_values'])
print("Data types per column:", summary['data_types'])
```

---

## 4. Directory Sub-Structure (Local Working Copies)

When downloading datasets locally, organize them according to standard data tiering:
- `data/raw/`: Untouched, immutable downloads directly from the source URLs.
- `data/processed/`: Cleaned, feature-engineered matrices ready for model training (generated in Week 3).
- `data/interim/`: Intermediate merged tables or staging files.

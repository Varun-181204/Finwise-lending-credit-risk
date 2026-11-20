"""FinWise Lending - Credit Risk Assessment System.

Unit & Integration Tests: Actual Client Dataset Validation & Baseline Pipeline.
Ensures schema conformance, privacy preservation, join incompatibility detection,
and baseline model benchmark verification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    generate_privacy_safe_summary,
    load_csv,
    validate_loan_applications_schema,
    verify_join_compatibility,
)
from src.train_baseline import run_baseline_training


DATA_DIR = Path("data")
LOAN_APPS_PATH = DATA_DIR / "finwise_loan_applications.csv"
CREDIT_FEATS_PATH = DATA_DIR / "finwise_credit_risk_features.csv"
FRAUD_TXNS_PATH = DATA_DIR / "finwise_fraud_transactions.csv"


class TestActualDatasetIntegrity:
    """Validate real client datasets against structural and schema contracts."""

    def test_actual_loan_applications_loading_and_schema(self):
        """Verify that actual loan applications dataset loads and conforms to schema."""
        if not LOAN_APPS_PATH.exists():
            pytest.skip(f"Dataset not available at {LOAN_APPS_PATH}")

        df = load_csv(LOAN_APPS_PATH)
        assert len(df) == 1000
        assert len(df.columns) == 13

        report = validate_loan_applications_schema(df)
        assert report["status"] == "VALID"
        assert report["row_count"] == 1000
        assert report["target_distribution"] == {0: 762, 1: 238}

    def test_schema_validation_rejects_missing_columns(self):
        """Ensure schema validator flags missing critical columns."""
        df_invalid = pd.DataFrame({
            "application_id": ["APP-001"],
            "age": [30],
            "income_annual_inr": [500000],
            # missing default_flag and others
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_loan_applications_schema(df_invalid)

    def test_schema_validation_rejects_domain_violations(self):
        """Ensure schema validator enforces financial and demographic sanity bounds."""
        valid_row = {
            "application_id": "APP-001",
            "age": 35,
            "income_annual_inr": 600000,
            "employment_type": "salaried",
            "employment_years": 5,
            "existing_loans": 1,
            "existing_emi_inr": 10000,
            "credit_score": 720,
            "loan_amount_inr": 1000000,
            "loan_purpose": "home",
            "ltv_ratio": 0.70,
            "dti_ratio": 0.25,
            "default_flag": 0,
        }

        # Case 1: Underage applicant (<18)
        df_underage = pd.DataFrame([dict(valid_row, age=16)])
        with pytest.raises(ValueError, match="under legal age"):
            validate_loan_applications_schema(df_underage)

        # Case 2: Credit score out of bounds (>900)
        df_bad_score = pd.DataFrame([dict(valid_row, credit_score=950)])
        with pytest.raises(ValueError, match="standard range \\[300, 900\\]"):
            validate_loan_applications_schema(df_bad_score)

        # Case 3: Invalid target class
        df_bad_target = pd.DataFrame([dict(valid_row, default_flag=2)])
        with pytest.raises(ValueError, match="Binary credit default target"):
            validate_loan_applications_schema(df_bad_target)

    def test_privacy_safe_summary_omits_raw_records(self):
        """Ensure privacy-safe summarizer calculates aggregates without exposing raw row data."""
        if not LOAN_APPS_PATH.exists():
            pytest.skip("Dataset file not found.")

        df = load_csv(LOAN_APPS_PATH)
        summary = generate_privacy_safe_summary(df, dataset_name="Loan Applications")

        assert summary["dataset_name"] == "Loan Applications"
        assert summary["row_count"] == 1000
        assert summary["column_count"] == 13
        assert "credit_score" in summary["numerical_metrics"]
        assert "employment_type" in summary["categorical_frequencies"]

        # Ensure no individual applicant IDs or raw row lists are leaked
        assert "application_id" not in summary["numerical_metrics"]
        assert "records" not in summary
        assert "raw_data" not in summary

    def test_join_compatibility_blocker_detection(self):
        """Verify that incompatible join keys between datasets are detected and blocked."""
        if not LOAN_APPS_PATH.exists() or not CREDIT_FEATS_PATH.exists():
            pytest.skip("Dataset files not found.")

        df_apps = load_csv(LOAN_APPS_PATH)
        df_feats = load_csv(CREDIT_FEATS_PATH)

        join_result = verify_join_compatibility(
            df_apps,
            df_feats,
            app_key="application_id",
            feat_key="applicant_id",
        )

        assert join_result["is_join_feasible"] is False
        assert join_result["overlap_count"] == 0
        assert join_result["overlap_ratio"] == 0.0
        assert "incompatible" in join_result["diagnostics"].lower()
        assert "blocked" in join_result["diagnostics"].lower()

    def test_join_compatibility_accepts_valid_overlap(self):
        """Verify that join validator permits merging when keys demonstrably overlap."""
        df1 = pd.DataFrame({"app_id": ["A1", "A2", "A3", "A4"], "val": [1, 2, 3, 4]})
        df2 = pd.DataFrame({"feat_id": ["A1", "A2", "A3", "A5"], "score": [10, 20, 30, 40]})

        res = verify_join_compatibility(df1, df2, app_key="app_id", feat_key="feat_id")
        assert res["is_join_feasible"] is True
        assert res["overlap_count"] == 3
        assert res["overlap_ratio"] == 0.75


class TestActualBaselineTrainingPipeline:
    """Validate end-to-end execution of baseline model on the actual client dataset."""

    def test_end_to_end_baseline_training_execution(self):
        """Run baseline training pipeline on client dataset and verify all results."""
        if not LOAN_APPS_PATH.exists():
            pytest.skip("Dataset file not found.")

        results = run_baseline_training(
            data_path=LOAN_APPS_PATH,
            random_state=42,
        )

        # Verify split partitions
        splits = results["splits"]
        assert splits["train_count"] == 700
        assert splits["train_defaults"] == 167
        assert splits["val_count"] == 150
        assert splits["val_defaults"] == 36
        assert splits["test_count"] == 150
        assert splits["test_defaults"] == 35

        # Verify threshold tuning on validation
        opt_thresh = results["optimal_threshold"]
        assert 0.10 <= opt_thresh <= 0.90
        assert opt_thresh == 0.40

        # Verify test set performance metrics
        test_m = results["test_metrics"]
        assert test_m["sample_count"] == 150
        assert test_m["positive_count"] == 35
        assert test_m["roc_auc"] > 0.85
        assert test_m["gini"] >= 0.45  # Must meet client target benchmark
        assert test_m["pr_auc"] > 0.75
        assert test_m["recall"] >= 0.75

        # Verify confusion matrix sums to test size
        cm = test_m["confusion_matrix"]
        total_cm = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total_cm == 150
        assert cm["true_negatives"] == 93
        assert cm["false_positives"] == 22
        assert cm["false_negatives"] == 6
        assert cm["true_positives"] == 29

        # Verify feature coefficients
        coefs = results["feature_coefficients"]
        assert len(coefs) > 0
        feat_dict = {c["feature"]: c["coefficient"] for c in coefs}
        assert "dti_ratio" in feat_dict
        assert feat_dict["dti_ratio"] > 0  # Higher DTI increases default probability
        assert "credit_score" in feat_dict
        assert feat_dict["credit_score"] < 0  # Higher credit score reduces default probability

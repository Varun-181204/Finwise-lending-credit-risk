"""Unit tests for src.preprocessing module.

Verifies:
1. Feature and target separation with strict validation (missing targets, single-class targets).
2. Identifier uniqueness and duplicate detection.
3. Feature audit trail and exclusion of identifiers, sensitive attributes, and post-origination leakage.
4. Numerical and categorical imputation, scaling, and one-hot encoding without data loss.
5. Leakage prevention (preprocessor parameters fit strictly on training set).
6. Deterministic, stratified train/validation/test splitting.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    audit_and_filter_features,
    build_feature_preprocessor,
    separate_target_and_features,
    split_dataset,
    validate_identifiers,
)


@pytest.fixture
def synthetic_lending_data() -> pd.DataFrame:
    """Generate a clean synthetic credit dataset for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "applicant_id": [f"APP-{i:04d}" for i in range(n)],
            "annual_income": np.random.uniform(30000, 120000, n),
            "debt_to_income": np.random.uniform(0.1, 0.6, n),
            "home_ownership": np.random.choice(["RENT", "MORTGAGE", "OWN"], n),
            "gender": np.random.choice(["Male", "Female"], n),  # Sensitive attribute
            "age": np.random.randint(21, 65, n),  # Sensitive attribute
            "repayment_status_m3": np.random.choice([0, 1, 2], n),  # Post-origination
            "default_target": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )


class TestTargetSeparation:
    """Test suite for separate_target_and_features."""

    def test_valid_separation(self, synthetic_lending_data: pd.DataFrame):
        X, y = separate_target_and_features(synthetic_lending_data, "default_target")
        assert "default_target" not in X.columns
        assert len(y) == len(synthetic_lending_data)
        assert y.name == "default_target"
        assert set(y.unique()) == {0, 1}

    def test_missing_target_column_raises_error(
        self, synthetic_lending_data: pd.DataFrame
    ):
        with pytest.raises(ValueError) as exc:
            separate_target_and_features(synthetic_lending_data, "non_existent_target")
        assert "not found in DataFrame columns" in str(exc.value)

    def test_target_with_nulls_raises_error(
        self, synthetic_lending_data: pd.DataFrame
    ):
        df_corrupted = synthetic_lending_data.copy()
        df_corrupted.loc[0, "default_target"] = None
        with pytest.raises(ValueError) as exc:
            separate_target_and_features(df_corrupted, "default_target")
        assert "missing/null values" in str(exc.value)

    def test_single_class_target_raises_error(self):
        df_single = pd.DataFrame(
            {"feat1": [1, 2, 3], "target": [0, 0, 0]}
        )
        with pytest.raises(ValueError) as exc:
            separate_target_and_features(df_single, "target")
        assert "requires at least 2 distinct classes" in str(exc.value)

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            separate_target_and_features([1, 2, 3], "target")  # type: ignore


class TestIdentifierValidation:
    """Test suite for validate_identifiers."""

    def test_unique_identifiers_pass(self, synthetic_lending_data: pd.DataFrame):
        # Should not raise any error
        validate_identifiers(synthetic_lending_data, ["applicant_id"])

    def test_duplicate_identifiers_raise_error(
        self, synthetic_lending_data: pd.DataFrame
    ):
        df_dup = synthetic_lending_data.copy()
        df_dup.loc[1, "applicant_id"] = df_dup.loc[0, "applicant_id"]
        with pytest.raises(ValueError) as exc:
            validate_identifiers(df_dup, ["applicant_id"])
        assert "duplicate entries" in str(exc.value)

    def test_missing_identifier_column_raises_error(
        self, synthetic_lending_data: pd.DataFrame
    ):
        with pytest.raises(ValueError) as exc:
            validate_identifiers(synthetic_lending_data, ["missing_id"])
        assert "not found in DataFrame" in str(exc.value)


class TestFeatureAuditAndExclusion:
    """Test suite for audit_and_filter_features."""

    def test_audit_trail_and_exclusions(self, synthetic_lending_data: pd.DataFrame):
        X, _ = separate_target_and_features(synthetic_lending_data, "default_target")

        X_filtered, audit = audit_and_filter_features(
            X,
            id_columns=["applicant_id"],
            sensitive_columns=["gender", "age"],
            post_origination_columns=["repayment_status_m3"],
        )

        # Verify excluded columns are absent from feature matrix
        assert "applicant_id" not in X_filtered.columns
        assert "gender" not in X_filtered.columns
        assert "age" not in X_filtered.columns
        assert "repayment_status_m3" not in X_filtered.columns

        # Verify retained columns
        assert "annual_income" in X_filtered.columns
        assert "debt_to_income" in X_filtered.columns
        assert "home_ownership" in X_filtered.columns

        # Verify audit metadata
        assert audit["excluded_identifiers"] == ["applicant_id"]
        assert set(audit["excluded_sensitive"]) == {"gender", "age"}
        assert audit["excluded_post_origination"] == ["repayment_status_m3"]
        assert audit["retained_feature_count"] == len(X_filtered.columns)


class TestPreprocessorAndLeakage:
    """Test suite for build_feature_preprocessor and leakage prevention."""

    def test_preprocessor_handles_missing_values(self):
        train_df = pd.DataFrame(
            {
                "income": [50000.0, np.nan, 80000.0, 100000.0],
                "dti": [0.2, 0.4, np.nan, 0.3],
                "home": ["RENT", None, "MORTGAGE", "OWN"],
            }
        )

        preprocessor = build_feature_preprocessor(
            numerical_cols=["income", "dti"],
            categorical_cols=["home"],
        )

        transformed = preprocessor.fit_transform(train_df)
        assert isinstance(transformed, np.ndarray)
        assert not np.isnan(transformed).any()

    def test_leakage_safety_fit_only_on_train(self):
        """Verify preprocessor calculates scaling mean from train, not test."""
        train_df = pd.DataFrame({"income": [10.0, 20.0, 30.0]})  # Mean = 20.0
        test_df = pd.DataFrame({"income": [100.0, 200.0, 300.0]})  # Mean = 200.0

        preprocessor = build_feature_preprocessor(
            numerical_cols=["income"], categorical_cols=[]
        )
        preprocessor.fit(train_df)

        # Scaler mean must match train mean (20.0), unaffected by test
        fitted_mean = (
            preprocessor.named_transformers_["num"]
            .named_steps["scaler"]
            .mean_[0]
        )
        assert np.isclose(fitted_mean, 20.0)

        # Transform test data using train parameters
        transformed_test = preprocessor.transform(test_df)
        # Verify (100 - 20) / std is positive
        assert transformed_test[0, 0] > 0


class TestStratifiedSplitting:
    """Test suite for split_dataset."""

    def test_reproducible_split_sizes_and_stratification(
        self, synthetic_lending_data: pd.DataFrame
    ):
        X, y = separate_target_and_features(synthetic_lending_data, "default_target")

        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_dataset(
            X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
        )
        X_train2, _, _, _, _, _ = split_dataset(
            X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
        )

        # Check deterministic reproducibility
        pd.testing.assert_frame_equal(X_train1, X_train2)

        # Check expected counts: 70 train, 15 val, 15 test
        assert len(X_train1) == 70
        assert len(X_val1) == 15
        assert len(X_test1) == 15

        # Check stratification preserves positive class prevalence (~20%)
        overall_prev = y.mean()
        train_prev = y_train1.mean()
        test_prev = y_test1.mean()
        assert np.isclose(train_prev, overall_prev, atol=0.05)
        assert np.isclose(test_prev, overall_prev, atol=0.07)

    def test_invalid_split_proportions_raise_error(
        self, synthetic_lending_data: pd.DataFrame
    ):
        X, y = separate_target_and_features(synthetic_lending_data, "default_target")
        with pytest.raises(ValueError) as exc:
            split_dataset(X, y, train_size=0.8, val_size=0.3, test_size=0.1)
        assert "must sum to 1.0" in str(exc.value)

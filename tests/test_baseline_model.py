"""Unit tests for src.baseline_model module.

Verifies:
1. Baseline pipeline construction with balanced class weights.
2. Fitting strictly on training data without leakage.
3. Selection of operating decision threshold on validation data.
4. Correct calculation of evaluation metrics: ROC-AUC, Gini (2 * ROC-AUC - 1),
   PR-AUC, Precision, Recall, F1, and Confusion Matrix.
5. Robust handling of edge cases (e.g., single-class target).
6. End-to-end pipeline training and holdout evaluation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.baseline_model import (
    build_baseline_pipeline,
    calculate_evaluation_metrics,
    evaluate_baseline_pipeline,
    select_operating_threshold,
    train_baseline_model,
)
from src.preprocessing import build_feature_preprocessor, split_dataset


@pytest.fixture
def synthetic_training_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
):
    """Generate synthetic train, val, test datasets for modeling tests."""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame(
        {
            "income": np.random.uniform(20000, 100000, n),
            "debt": np.random.uniform(1000, 50000, n),
            "employment_type": np.random.choice(["SALARIED", "SELF", "OTHER"], n),
        }
    )
    # Default correlated with higher debt / lower income
    debt_ratio = X["debt"] / X["income"]
    prob = 1 / (1 + np.exp(-(3.0 * debt_ratio - 1.0)))
    y = pd.Series(
        np.random.binomial(1, prob.clip(0.05, 0.95), n), name="is_default"
    )

    return split_dataset(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )


class TestBaselinePipeline:
    """Test suite for pipeline construction and training."""

    def test_pipeline_construction(self):
        preprocessor = build_feature_preprocessor(
            numerical_cols=["income", "debt"],
            categorical_cols=["employment_type"],
        )
        pipeline = build_baseline_pipeline(preprocessor, class_weight="balanced")

        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps
        assert pipeline.named_steps["classifier"].class_weight == "balanced"

    def test_pipeline_fit_and_predict(self, synthetic_training_data):
        X_train, _, X_test, y_train, _, _ = synthetic_training_data
        preprocessor = build_feature_preprocessor(
            numerical_cols=["income", "debt"],
            categorical_cols=["employment_type"],
        )
        pipeline = build_baseline_pipeline(preprocessor)
        fitted_pipeline = train_baseline_model(pipeline, X_train, y_train)

        # Check probability outputs
        probas = fitted_pipeline.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        assert (probas >= 0.0).all() and (probas <= 1.0).all()
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


class TestOperatingThresholdSelection:
    """Test suite for select_operating_threshold."""

    def test_select_threshold_on_validation_only(self, synthetic_training_data):
        X_train, X_val, _, y_train, y_val, _ = synthetic_training_data
        preprocessor = build_feature_preprocessor(
            numerical_cols=["income", "debt"],
            categorical_cols=["employment_type"],
        )
        pipeline = build_baseline_pipeline(preprocessor)
        train_baseline_model(pipeline, X_train, y_train)

        best_t = select_operating_threshold(
            pipeline, X_val, y_val, optimization_metric="f1"
        )

        assert isinstance(best_t, float)
        assert 0.0 < best_t < 1.0

    def test_empty_validation_set_raises_error(self, synthetic_training_data):
        X_train, _, _, y_train, _, _ = synthetic_training_data
        preprocessor = build_feature_preprocessor(
            numerical_cols=["income", "debt"],
            categorical_cols=["employment_type"],
        )
        pipeline = build_baseline_pipeline(preprocessor)
        train_baseline_model(pipeline, X_train, y_train)

        with pytest.raises(ValueError) as exc:
            select_operating_threshold(
                pipeline, pd.DataFrame(), pd.Series(dtype=int)
            )
        assert "Validation set cannot be empty" in str(exc.value)


class TestEvaluationMetrics:
    """Test suite for calculate_evaluation_metrics and Gini relationship."""

    def test_metrics_calculation_and_gini_formula(self):
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        metrics = calculate_evaluation_metrics(
            y_true=y_true, y_pred_proba=y_pred_proba, threshold=0.5
        )

        # Verify Gini formula: Gini = 2 * ROC-AUC - 1
        assert metrics["roc_auc"] == 1.0
        assert metrics["gini"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["confusion_matrix"]["true_negatives"] == 4
        assert metrics["confusion_matrix"]["true_positives"] == 4
        assert metrics["confusion_matrix"]["false_positives"] == 0
        assert metrics["confusion_matrix"]["false_negatives"] == 0

    def test_single_class_edge_case(self):
        """Verify ranking metrics return None safely when class diversity is insufficient."""
        y_single = pd.Series([0, 0, 0, 0])
        y_pred_proba = np.array([0.2, 0.3, 0.4, 0.1])

        metrics = calculate_evaluation_metrics(
            y_true=y_single, y_pred_proba=y_pred_proba, threshold=0.5
        )

        # ROC-AUC, Gini, and PR-AUC cannot be computed on a single class
        assert metrics["roc_auc"] is None
        assert metrics["gini"] is None
        assert metrics["pr_auc"] is None
        assert metrics["sample_count"] == 4
        assert metrics["positive_count"] == 0

    def test_evaluate_baseline_pipeline_end_to_end(self, synthetic_training_data):
        X_train, X_val, X_test, y_train, y_val, y_test = synthetic_training_data
        preprocessor = build_feature_preprocessor(
            numerical_cols=["income", "debt"],
            categorical_cols=["employment_type"],
        )
        pipeline = build_baseline_pipeline(preprocessor)
        train_baseline_model(pipeline, X_train, y_train)

        best_t = select_operating_threshold(pipeline, X_val, y_val)
        test_metrics = evaluate_baseline_pipeline(
            pipeline, X_test, y_test, threshold=best_t
        )

        assert test_metrics["sample_count"] == len(X_test)
        assert test_metrics["operating_threshold"] == best_t
        assert test_metrics["roc_auc"] is not None
        assert test_metrics["gini"] is not None
        assert -1.0 <= test_metrics["gini"] <= 1.0

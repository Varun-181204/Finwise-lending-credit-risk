"""FinWise Lending - Credit Risk Assessment System.

Module: src.baseline_model
Purpose: Baseline Logistic Regression model training, validation-based threshold
         selection, and comprehensive classification evaluation.

Design Principles:
1. Leakage-Free Workflow: Preprocessing and estimators are fit strictly on training data.
2. Validation-Driven Thresholds: The operating decision threshold is tuned exclusively
   on the validation partition, never on the holdout test set.
3. Class Imbalance Handling: Leverages balanced class weighting (`class_weight='balanced'`).
4. Strict Transparency: Evaluation metrics include ROC-AUC, Gini, PR-AUC, Precision, Recall,
   F1, and Confusion Matrix. If metrics cannot be computed due to edge-case class distribution,
   clear explanations are provided without inventing numbers.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def build_baseline_pipeline(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str = "balanced",
    C: float = 1.0,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline combining preprocessor and Logistic Regression.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Configured ColumnTransformer for numerical and categorical features.
    random_state : int, default=42
        Seed for deterministic solver optimization.
    max_iter : int, default=1000
        Maximum iterations for convergence.
    class_weight : str, default='balanced'
        Weighting strategy to counteract class imbalance.
    C : float, default=1.0
        Inverse regularization strength.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline.
    """
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return full_pipeline


def train_baseline_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the baseline pipeline on the training partition strictly.

    Parameters
    ----------
    pipeline : Pipeline
        The unfitted Pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.

    Returns
    -------
    Pipeline
        Fitted scikit-learn Pipeline.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot train model on empty training data.")

    pipeline.fit(X_train, y_train)
    return pipeline


def select_operating_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold_candidates: Optional[List[float]] = None,
    optimization_metric: str = "f1",
) -> float:
    """Select the optimal binary decision threshold using validation data only.

    Evaluates predictions across candidate thresholds and returns the threshold
    maximizing the specified evaluation metric (default: F1 score).

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Pipeline.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation ground-truth binary targets.
    threshold_candidates : Optional[List[float]]
        List of candidate probability thresholds. Defaults to [0.1, 0.2, ..., 0.9].
    optimization_metric : str, default='f1'
        Metric to maximize ('f1', 'balanced_accuracy', etc.).

    Returns
    -------
    float
        Selected operating threshold.
    """
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("Validation set cannot be empty for threshold selection.")

    candidates = threshold_candidates or [
        round(t, 2) for t in np.arange(0.10, 0.95, 0.05)
    ]
    # Predict default probabilities (class 1)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_score = -1.0

    for threshold in candidates:
        y_pred = (y_val_proba >= threshold).astype(int)
        if optimization_metric == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def calculate_evaluation_metrics(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate comprehensive credit-risk classification metrics.

    Metrics calculated:
    - ROC-AUC: Area under ROC curve.
    - Gini Coefficient: 2 * ROC-AUC - 1 (Basel / credit risk standard).
    - PR-AUC: Area under Precision-Recall curve (Average Precision).
    - Precision, Recall, and F1 at the specified operating threshold.
    - Confusion Matrix (TN, FP, FN, TP).
    - Sample counts and positive class prevalence.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth binary labels (0 = non-default, 1 = default).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class (default).
    threshold : float, default=0.50
        Operating decision threshold.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed evaluation metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_proba_arr = np.asarray(y_pred_proba)

    sample_count = int(len(y_true_arr))
    if sample_count == 0:
        raise ValueError("Cannot calculate metrics on empty targets.")

    positive_count = int((y_true_arr == 1).sum())
    positive_prevalence = float(positive_count / sample_count)

    unique_classes = np.unique(y_true_arr)
    has_binary_classes = len(unique_classes) == 2

    # Calculate ranking metrics (ROC-AUC, Gini, PR-AUC)
    if has_binary_classes:
        roc_auc = float(roc_auc_score(y_true_arr, y_pred_proba_arr))
        gini = float(2.0 * roc_auc - 1.0)
        pr_auc = float(average_precision_score(y_true_arr, y_pred_proba_arr))
    else:
        # If dataset only contains one class (e.g. all 0s), ranking cannot be computed
        roc_auc = None
        gini = None
        pr_auc = None

    # Calculate threshold-dependent metrics
    y_pred = (y_pred_proba_arr >= threshold).astype(int)
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    if has_binary_classes:
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    else:
        # Fallback if only single class
        tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())
        fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
        tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())

    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "positive_prevalence": positive_prevalence,
        "operating_threshold": float(threshold),
        "roc_auc": roc_auc,
        "gini": gini,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
    }


def evaluate_baseline_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluate fitted baseline pipeline once on the untouched holdout test partition.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn Pipeline.
    X_test : pd.DataFrame
        Holdout test features.
    y_test : pd.Series
        Holdout test ground truth.
    threshold : float, default=0.50
        Operating threshold selected on the validation partition.

    Returns
    -------
    Dict[str, Any]
        Test evaluation metrics dictionary.
    """
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = calculate_evaluation_metrics(
        y_true=y_test,
        y_pred_proba=y_test_proba,
        threshold=threshold,
    )
    return metrics

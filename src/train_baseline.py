"""FinWise Lending - Credit Risk Assessment System.

Module: src.train_baseline
Purpose: End-to-end execution of the baseline Logistic Regression model pipeline
         on the actual Cynaris client loan applications dataset.

Design Principles:
1. Leakage Prevention: Transformers and model weights fit strictly on X_train.
2. Validation-Driven Operating Threshold: The decision threshold is selected on X_val
   and applied to X_test in a single evaluation pass.
3. Feature Governance: Identifiers are excluded to prevent memorization; sensitive
   demographic attributes (age) are excluded and preserved for fairness audits.
4. Reproducibility: Deterministic splitting and optimization seeds (random_state=42).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.baseline_model import (
    build_baseline_pipeline,
    calculate_evaluation_metrics,
    evaluate_baseline_pipeline,
    select_operating_threshold,
    train_baseline_model,
)
from src.data_pipeline import load_csv, validate_loan_applications_schema
from src.preprocessing import (
    audit_and_filter_features,
    build_feature_preprocessor,
    separate_target_and_features,
    split_dataset,
)


def run_baseline_training(
    data_path: Union[str, Path] = "data/finwise_loan_applications.csv",
    target_column: str = "default_flag",
    id_columns: Optional[List[str]] = None,
    sensitive_columns: Optional[List[str]] = None,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute the end-to-end baseline training and evaluation workflow.

    Parameters
    ----------
    data_path : str or Path
        Path to the validated loan applications CSV file.
    target_column : str, default='default_flag'
        Name of the binary credit default target column.
    id_columns : Optional[List[str]]
        Identifier columns to exclude from training (defaults to ['application_id']).
    sensitive_columns : Optional[List[str]]
        Demographic attributes to exclude from training (defaults to ['age']).
    train_size : float, default=0.70
        Training split proportion.
    val_size : float, default=0.15
        Validation split proportion.
    test_size : float, default=0.15
        Holdout test split proportion.
    random_state : int, default=42
        Seed for deterministic splitting and training.

    Returns
    -------
    Dict[str, Any]
        Dictionary with complete evaluation results across train, val, and test splits,
        confusion matrices, optimal threshold, feature coefficients, and fitted pipeline.
    """
    id_cols = id_columns if id_columns is not None else ["application_id"]
    sens_cols = sensitive_columns if sensitive_columns is not None else ["age"]

    # 1. Ingestion
    df = load_csv(data_path)

    # 2. Schema Validation
    schema_report = validate_loan_applications_schema(df)

    # 3. Target and Feature Separation
    X, y = separate_target_and_features(df, target_column=target_column)

    # 4. Feature Auditing & Leakage Prevention
    X_filtered, audit_trail = audit_and_filter_features(
        X,
        id_columns=id_cols,
        sensitive_columns=sens_cols,
        post_origination_columns=[],
    )

    # 5. Dynamic Type Identification
    numerical_cols = X_filtered.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_filtered.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # 6. Stratified Train / Validation / Test Splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X_filtered,
        y,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    # 7. Pipeline Construction
    preprocessor = build_feature_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )
    pipeline = build_baseline_pipeline(
        preprocessor=preprocessor,
        random_state=random_state,
        class_weight="balanced",
    )

    # 8. Training (Fitted Strictly on Training Data)
    pipeline = train_baseline_model(pipeline, X_train, y_train)

    # 9. Training Set Evaluation
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_metrics = calculate_evaluation_metrics(
        y_true=y_train,
        y_pred_proba=y_train_proba,
        threshold=0.50,
    )

    # 10. Validation Operating Threshold Optimization
    optimal_threshold = select_operating_threshold(
        pipeline=pipeline,
        X_val=X_val,
        y_val=y_val,
        optimization_metric="f1",
    )

    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_metrics = calculate_evaluation_metrics(
        y_true=y_val,
        y_pred_proba=y_val_proba,
        threshold=optimal_threshold,
    )

    # 11. Single-Pass Holdout Test Evaluation
    test_metrics = evaluate_baseline_pipeline(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=optimal_threshold,
    )

    # Test metrics at default 0.50 threshold for reference
    test_metrics_50 = evaluate_baseline_pipeline(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=0.50,
    )

    # 12. Model Coefficients Extraction
    ct = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    cat_encoder = ct.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + cat_feature_names
    coefficients = model.coef_[0]

    coef_list = [
        {"feature": feat, "coefficient": round(float(coef), 6)}
        for feat, coef in zip(all_feature_names, coefficients)
    ]
    coef_list.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return {
        "schema_validation": schema_report,
        "audit_trail": audit_trail,
        "splits": {
            "train_count": len(X_train),
            "train_defaults": int(y_train.sum()),
            "val_count": len(X_val),
            "val_defaults": int(y_val.sum()),
            "test_count": len(X_test),
            "test_defaults": int(y_test.sum()),
        },
        "optimal_threshold": optimal_threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_50": test_metrics_50,
        "feature_coefficients": coef_list,
        "intercept": round(float(model.intercept_[0]), 6),
        "pipeline": pipeline,
    }


if __name__ == "__main__":
    print("Executing FinWise Lending Baseline Model Training Pipeline...")
    results = run_baseline_training()
    print("\n=== BASELINE MODEL RESULTS ===")
    print(f"Optimal Validation Threshold: {results['optimal_threshold']:.2f}")
    print("\n--- Test Set Metrics (at optimal threshold) ---")
    tm = results["test_metrics"]
    print(f"ROC-AUC:   {tm['roc_auc']:.4f}")
    print(f"Gini:      {tm['gini']:.4f} (Client target >= 0.45: {'PASS' if tm['gini'] >= 0.45 else 'FAIL'})")
    print(f"PR-AUC:    {tm['pr_auc']:.4f}")
    print(f"Precision: {tm['precision']:.4f}")
    print(f"Recall:    {tm['recall']:.4f}")
    print(f"F1-Score:  {tm['f1']:.4f}")
    cm = tm["confusion_matrix"]
    print(f"Confusion Matrix: TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TP={cm['true_positives']}")

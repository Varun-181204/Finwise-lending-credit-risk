"""FinWise Lending Credit Risk Assessment Package.

Core modules for data ingestion, quality inspection, feature processing, and baseline modeling.
"""

from src.baseline_model import (
    build_baseline_pipeline,
    calculate_evaluation_metrics,
    evaluate_baseline_pipeline,
    select_operating_threshold,
    train_baseline_model,
)
from src.data_pipeline import (
    generate_data_quality_summary,
    generate_privacy_safe_summary,
    load_csv,
    validate_loan_applications_schema,
    verify_join_compatibility,
)
from src.preprocessing import (
    audit_and_filter_features,
    build_feature_preprocessor,
    separate_target_and_features,
    split_dataset,
    validate_identifiers,
)

__all__ = [
    "load_csv",
    "generate_data_quality_summary",
    "generate_privacy_safe_summary",
    "validate_loan_applications_schema",
    "verify_join_compatibility",
    "separate_target_and_features",
    "validate_identifiers",
    "audit_and_filter_features",
    "build_feature_preprocessor",
    "split_dataset",
    "build_baseline_pipeline",
    "train_baseline_model",
    "select_operating_threshold",
    "calculate_evaluation_metrics",
    "evaluate_baseline_pipeline",
]

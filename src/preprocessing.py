"""FinWise Lending - Credit Risk Assessment System.

Module: src.preprocessing
Purpose: Data preprocessing, feature-target separation, leakage prevention,
         and stratified train/validation/test splitting.

Design Principles:
1. Leakage Prevention: All transformers fit strictly on the training partition.
2. Transparency: Excluded features (identifiers, sensitive attributes, post-origination fields)
   are explicitly tracked in a documented audit trail.
3. Reproducibility: Deterministic splitting and transformation parameters with configurable seeds.
4. Robust Validation: Descriptive exceptions for missing targets, invalid inputs, or duplicate IDs.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def separate_target_and_features(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target variable from feature matrix with upfront validation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and target.
    target_column : str
        The name of the target column to extract.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature DataFrame X (excluding target) and target Series y.

    Raises
    ------
    TypeError
        If input df is not a pandas DataFrame.
    ValueError
        If target_column is not in df, contains null/NaN values, or has
        fewer than 2 distinct classes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}"
        )

    # Validate target missingness - target cannot have nulls
    null_count = int(df[target_column].isna().sum())
    if null_count > 0:
        raise ValueError(
            f"Target column '{target_column}' contains {null_count} missing/null values. "
            "Records with unconfirmed targets must be resolved or explicitly handled."
        )

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    # Validate target diversity
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Target column '{target_column}' has only {len(unique_classes)} unique class ({unique_classes}). "
            "Binary classification requires at least 2 distinct classes."
        )

    return X, y


def validate_identifiers(df: pd.DataFrame, id_columns: List[str]) -> None:
    """Validate that specified identifier columns exist and contain unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check.
    id_columns : List[str]
        List of identifier column names (e.g., ['applicant_id']).

    Raises
    ------
    ValueError
        If an identifier column is missing or contains duplicate records.
    """
    for col in id_columns:
        if col not in df.columns:
            raise ValueError(f"Identifier column '{col}' not found in DataFrame.")

        dup_count = int(df[col].duplicated().sum())
        if dup_count > 0:
            raise ValueError(
                f"Identifier column '{col}' contains {dup_count} duplicate entries. "
                "Primary applicant keys must be unique."
            )


def audit_and_filter_features(
    X: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    sensitive_columns: Optional[List[str]] = None,
    post_origination_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter out non-predictive or high-risk features and create an audit trail.

    Excludes:
    - Primary identifiers (which carry no generalizable signal and cause overfitting).
    - Sensitive attributes (e.g., gender, age), preserving them for standalone
      fairness audits rather than direct model predictors.
    - Post-origination features (which cause data leakage because they occur after loan approval).

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix.
    id_columns : Optional[List[str]]
        List of identifier column names.
    sensitive_columns : Optional[List[str]]
        List of demographic/protected column names.
    post_origination_columns : Optional[List[str]]
        List of post-origination/leakage column names.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Filtered feature matrix and audit trail dictionary documenting all exclusions.
    """
    id_cols = id_columns or []
    sens_cols = sensitive_columns or []
    post_cols = post_origination_columns or []

    audit_trail: Dict[str, Any] = {
        "original_feature_count": len(X.columns),
        "excluded_identifiers": [c for c in id_cols if c in X.columns],
        "excluded_sensitive": [c for c in sens_cols if c in X.columns],
        "excluded_post_origination": [c for c in post_cols if c in X.columns],
        "retained_feature_count": 0,
        "retained_features": [],
    }

    cols_to_drop = set(
        audit_trail["excluded_identifiers"]
        + audit_trail["excluded_sensitive"]
        + audit_trail["excluded_post_origination"]
    )

    X_filtered = X.drop(columns=list(cols_to_drop)).copy()
    audit_trail["retained_feature_count"] = len(X_filtered.columns)
    audit_trail["retained_features"] = list(X_filtered.columns)

    return X_filtered, audit_trail


def build_feature_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical features.

    Numerical pipeline:
    - SimpleImputer with median strategy (robust to outliers).
    - StandardScaler for zero-mean, unit-variance scaling.

    Categorical pipeline:
    - SimpleImputer with constant 'missing' strategy for null categories.
    - OneHotEncoder with handle_unknown='ignore' to prevent errors on unseen categories.

    Parameters
    ----------
    numerical_cols : List[str]
        List of numerical feature names.
    categorical_cols : List[str]
        List of categorical feature names.

    Returns
    -------
    ColumnTransformer
        Unfitted scikit-learn ColumnTransformer.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split dataset into reproducible train, validation, and test partitions.

    Applies a two-step stratified split:
    1. Split total into train (e.g., 70%) and temp (e.g., 30%).
    2. Split temp into validation (e.g., 50% of temp = 15% of total) and test (15% of total).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_size : float, default=0.70
        Proportion of dataset for training.
    val_size : float, default=0.15
        Proportion of dataset for validation.
    test_size : float, default=0.15
        Proportion of dataset for holdout testing.
    random_state : int, default=42
        Seed for reproducibility.
    stratify : bool, default=True
        Whether to stratify splits based on target class counts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        X_train, X_val, X_test, y_train, y_val, y_test

    Raises
    ------
    ValueError
        If split proportions do not sum to 1.0 or if minimum class counts
        are insufficient for stratification.
    """
    total_prop = train_size + val_size + test_size
    if not np.isclose(total_prop, 1.0, atol=1e-4):
        raise ValueError(
            f"Split sizes must sum to 1.0 (got train={train_size}, val={val_size}, test={test_size}, sum={total_prop})"
        )

    # Check minimum class count for stratification
    strat_y = y if stratify else None
    if stratify:
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            raise ValueError(
                f"Cannot stratify split: at least one class has fewer than 2 instances. Counts: {dict(class_counts)}"
            )

    # First split: train vs (val + test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=strat_y,
    )

    # Second split: val vs test
    val_ratio_of_temp = val_size / temp_size
    strat_temp = y_temp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_ratio_of_temp,
        random_state=random_state,
        stratify=strat_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

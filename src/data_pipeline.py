"""FinWise Lending - Credit Risk Assessment System.

Module: src.data_pipeline
Purpose: Initial data ingestion and automated data-quality profiling for credit risk datasets.

Design Principles:
1. Transparency: Never drop rows, columns, or values silently during ingestion.
2. Agnostic Ingestion: Do not hardcode any target column (e.g., 'default', 'loan_status')
   until exploratory data analysis and client alignment confirm the ground-truth definition.
3. Strict Validation: Validate file paths, file contents, and formats upfront, raising
   descriptive errors before downstream operations occur.
"""

from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file from the specified path into a pandas DataFrame.

    Performs upfront safety checks to ensure the file exists, is non-empty,
    and conforms to standard CSV formatting without silently modifying or
    dropping any applicant data.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The path to the CSV file to be loaded.

    Returns
    -------
    pd.DataFrame
        Loaded tabular dataset preserving original row and column counts.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the provided path.
    ValueError
        If the path points to a directory, if the file is empty (0 bytes),
        if the file contains no parseable columns/data, or if the CSV syntax is corrupted.
    """
    path = Path(file_path)

    # 1. Verify existence
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at path: '{path}'")

    # 2. Verify that the path points to a file, not a directory
    if not path.is_file():
        raise ValueError(f"Expected a file path, but encountered a directory: '{path}'")

    # 3. Check for 0-byte (completely empty) files before parsing
    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty (0 bytes): '{path}'")

    # 4. Attempt to read the CSV file with pandas
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as err:
        raise ValueError(
            f"CSV file contains no data or headers to parse: '{path}'"
        ) from err
    except pd.errors.ParserError as err:
        raise ValueError(
            f"CSV file is malformed and could not be parsed: '{path}'. Details: {err}"
        ) from err
    except Exception as err:
        raise ValueError(
            f"Failed to read CSV file at '{path}'. Error: {err}"
        ) from err

    # 5. Guard against files that had whitespace or comments resulting in 0 columns
    if len(df.columns) == 0:
        raise ValueError(
            f"CSV file at '{path}' was parsed but produced zero columns."
        )

    return df


def generate_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate an automated data-quality inspection summary for a given DataFrame.

    Calculates key health metrics without altering or filtering any records:
    - Total row count
    - Total column count
    - List of column names
    - Data type for each column (string representation)
    - Missing (null/NaN) count for each column
    - Count of duplicate rows across all columns

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    dict
        A dictionary containing:
        - 'row_count' (int): Total number of rows.
        - 'column_count' (int): Total number of columns.
        - 'column_names' (list of str): List of all column headers.
        - 'data_types' (dict): Mapping of column names to string data type names.
        - 'missing_values' (dict): Mapping of column names to count of null entries.
        - 'duplicate_rows' (int): Total number of duplicate rows across all columns.

    Raises
    ------
    TypeError
        If the provided argument is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected 'df' to be a pandas DataFrame, but received {type(df).__name__}."
        )

    summary: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "column_names": [str(col) for col in df.columns],
        "data_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {str(col): int(df[col].isna().sum()) for col in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return summary

def validate_loan_applications_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that a DataFrame conforms to the expected loan applications schema.

    Verifies required column presence, primary key uniqueness, binary target integrity,
    and domain-level financial sanity bounds without altering the underlying data.

    Parameters
    ----------
    df : pd.DataFrame
        The loan applications DataFrame to validate.

    Returns
    -------
    Dict[str, Any]
        Dictionary detailing validation results, row count, and verified checks.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If required columns are missing, primary keys are duplicated or null,
        target variable is corrupted, or domain ranges are violated.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected 'df' to be a pandas DataFrame, but received {type(df).__name__}."
        )

    expected_columns = [
        "application_id",
        "age",
        "income_annual_inr",
        "employment_type",
        "employment_years",
        "existing_loans",
        "existing_emi_inr",
        "credit_score",
        "loan_amount_inr",
        "loan_purpose",
        "ltv_ratio",
        "dti_ratio",
        "default_flag",
    ]

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed: Missing required columns: {missing_cols}"
        )

    # Validate primary identifier
    id_nulls = int(df["application_id"].isna().sum())
    if id_nulls > 0:
        raise ValueError(
            f"Schema validation failed: 'application_id' contains {id_nulls} null values."
        )

    id_dups = int(df["application_id"].duplicated().sum())
    if id_dups > 0:
        raise ValueError(
            f"Schema validation failed: 'application_id' contains {id_dups} duplicate values."
        )

    # Validate target variable
    target_nulls = int(df["default_flag"].isna().sum())
    if target_nulls > 0:
        raise ValueError(
            f"Schema validation failed: 'default_flag' contains {target_nulls} null values."
        )

    unique_targets = set(df["default_flag"].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Schema validation failed: 'default_flag' contains invalid classes {unique_targets}. "
            "Binary credit default target must only contain 0 and 1."
        )

    # Validate financial / demographic domain bounds
    if (df["age"] < 18).any():
        raise ValueError("Schema validation failed: Found applicants under legal age (18).")

    if (df["income_annual_inr"] <= 0).any():
        raise ValueError("Schema validation failed: 'income_annual_inr' must be positive.")

    if (df["credit_score"] < 300).any() or (df["credit_score"] > 900).any():
        raise ValueError(
            "Schema validation failed: 'credit_score' must fall within the standard range [300, 900]."
        )

    if (df["loan_amount_inr"] <= 0).any():
        raise ValueError("Schema validation failed: 'loan_amount_inr' must be strictly positive.")

    if (df["dti_ratio"] < 0.0).any():
        raise ValueError("Schema validation failed: 'dti_ratio' cannot be negative.")

    if (df["ltv_ratio"] < 0.0).any():
        raise ValueError("Schema validation failed: 'ltv_ratio' cannot be negative.")

    return {
        "status": "VALID",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target_distribution": {
            int(k): int(v) for k, v in df["default_flag"].value_counts().items()
        },
        "verified_columns": expected_columns,
    }


def verify_join_compatibility(
    df_apps: pd.DataFrame,
    df_feats: pd.DataFrame,
    app_key: str = "application_id",
    feat_key: str = "applicant_id",
) -> Dict[str, Any]:
    """Verify join key compatibility between application and feature datasets.

    Evaluates key presence, format overlap, and cardinality. If no valid,
    unambiguous key alignment exists, reports the incompatibility to prevent
    arbitrary or row-order merging.

    Parameters
    ----------
    df_apps : pd.DataFrame
        Loan applications dataset.
    df_feats : pd.DataFrame
        Credit risk features dataset.
    app_key : str, default='application_id'
        Join key in df_apps.
    feat_key : str, default='applicant_id'
        Join key in df_feats.

    Returns
    -------
    Dict[str, Any]
        Dictionary reporting join feasibility, overlap metrics, and diagnostics.

    Raises
    ------
    TypeError
        If either input is not a pandas DataFrame.
    ValueError
        If specified join keys are absent from their respective DataFrames.
    """
    if not isinstance(df_apps, pd.DataFrame):
        raise TypeError(f"Expected df_apps to be DataFrame, got {type(df_apps).__name__}.")
    if not isinstance(df_feats, pd.DataFrame):
        raise TypeError(f"Expected df_feats to be DataFrame, got {type(df_feats).__name__}.")

    if app_key not in df_apps.columns:
        raise ValueError(f"Join key '{app_key}' not found in applications DataFrame.")
    if feat_key not in df_feats.columns:
        raise ValueError(f"Join key '{feat_key}' not found in features DataFrame.")

    apps_keys = set(df_apps[app_key].dropna().astype(str))
    feats_keys = set(df_feats[feat_key].dropna().astype(str))

    overlap = apps_keys.intersection(feats_keys)
    overlap_count = len(overlap)
    min_size = min(len(apps_keys), len(feats_keys))
    overlap_ratio = float(overlap_count / min_size) if min_size > 0 else 0.0

    is_feasible = overlap_ratio >= 0.50

    if not is_feasible:
        reason = (
            f"Join incompatible: Only {overlap_count} overlapping keys found "
            f"({overlap_ratio:.1%} overlap between '{app_key}' and '{feat_key}'). "
            "Merging these datasets without a verified foreign key mapping would corrupt "
            "applicant records. Merge blocked to prevent fabricated associations; datasets must be processed separately."
        )
    else:
        reason = (
            f"Join verified: {overlap_count} overlapping keys ({overlap_ratio:.1%} overlap). "
            "Safe to merge on confirmed keys."
        )

    return {
        "is_join_feasible": is_feasible,
        "app_key": app_key,
        "feat_key": feat_key,
        "app_key_count": len(apps_keys),
        "feat_key_count": len(feats_keys),
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio,
        "diagnostics": reason,
    }


def generate_privacy_safe_summary(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
) -> Dict[str, Any]:
    """Generate an aggregated, privacy-safe data summary without exposing individual PII.

    Produces statistical summaries for numerical columns and frequency tables for
    categorical columns while strictly omitting row records, names, or individual values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    dataset_name : str, default='Dataset'
        Human-readable name of the dataset for reporting.

    Returns
    -------
    Dict[str, Any]
        Aggregated summary dictionary.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "missing_values": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_metrics": {},
        "categorical_frequencies": {},
    }

    # Aggregate numerical features
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 0:
            summary["numerical_metrics"][str(col)] = {
                "count": int(len(series)),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "mean": round(float(series.mean()), 4),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "std": round(float(series.std()), 4),
            }

    # Aggregate categorical features
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        val_counts = df[col].value_counts(dropna=False)
        total = len(df)
        summary["categorical_frequencies"][str(col)] = {
            str(cat): {
                "count": int(cnt),
                "proportion": round(float(cnt / total), 4),
            }
            for cat, cnt in val_counts.items()
        }

    return summary

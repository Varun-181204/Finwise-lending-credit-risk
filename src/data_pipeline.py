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

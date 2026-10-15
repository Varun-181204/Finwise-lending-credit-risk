"""Unit tests for src.data_pipeline module.

Verifies:
1. Valid CSV loading and structure preservation.
2. Error handling for missing files, directories, empty files, and malformed content.
3. Data quality summary calculations (dimensions, column types, missing values, duplicate rows).
4. Preservation of input data (no silent dropping of records).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_pipeline import generate_data_quality_summary, load_csv


class TestLoadCsv:
    """Test suite for load_csv function."""

    def test_load_valid_csv(self, tmp_path: Path):
        """Verify that a valid CSV file loads properly into a DataFrame."""
        csv_file = tmp_path / "valid_sample.csv"
        csv_content = (
            "applicant_id,annual_income,loan_amount,employment_years\n"
            "APP-001,85000,15000,5\n"
            "APP-002,42000,8000,2\n"
            "APP-003,110000,25000,10\n"
        )
        csv_file.write_text(csv_content, encoding="utf-8")

        df = load_csv(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == [
            "applicant_id",
            "annual_income",
            "loan_amount",
            "employment_years",
        ]
        assert df["annual_income"].iloc[0] == 85000

    def test_load_missing_file_raises_error(self, tmp_path: Path):
        """Verify that attempting to load a non-existent file raises FileNotFoundError."""
        non_existent_file = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_csv(non_existent_file)

        assert "not found at path" in str(exc_info.value)

    def test_load_directory_raises_error(self, tmp_path: Path):
        """Verify that passing a directory path raises ValueError."""
        test_dir = tmp_path / "sample_directory"
        test_dir.mkdir()

        with pytest.raises(ValueError) as exc_info:
            load_csv(test_dir)

        assert "Expected a file path, but encountered a directory" in str(
            exc_info.value
        )

    def test_load_empty_zero_byte_file_raises_error(self, tmp_path: Path):
        """Verify that loading a completely empty (0-byte) file raises ValueError."""
        empty_file = tmp_path / "empty_zero_bytes.csv"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_csv(empty_file)

        assert "empty (0 bytes)" in str(exc_info.value)

    def test_load_headers_only_csv(self, tmp_path: Path):
        """Verify that a CSV with headers but no rows loads as a 0-row DataFrame."""
        headers_only = tmp_path / "headers_only.csv"
        headers_only.write_text("col_a,col_b,col_c\n", encoding="utf-8")

        df = load_csv(headers_only)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["col_a", "col_b", "col_c"]

    def test_load_malformed_csv_raises_error(self, tmp_path: Path):
        """Verify that loading an unparseable/corrupted CSV raises ValueError."""
        malformed_file = tmp_path / "corrupted.csv"
        # Inconsistent field counts without quotes to trigger parser failure
        malformed_content = (
            "col_a,col_b\n"
            "val1,val2\n"
            "val1,val2,val3,val4,val5\n"
        )
        malformed_file.write_text(malformed_content, encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_csv(malformed_file)

        assert "malformed" in str(exc_info.value).lower() or "failed" in str(
            exc_info.value
        ).lower()


class TestGenerateDataQualitySummary:
    """Test suite for generate_data_quality_summary function."""

    def test_quality_summary_dimensions_and_metadata(self):
        """Verify summary accurately captures row count, column count, names, and data types."""
        df = pd.DataFrame(
            {
                "user_id": [101, 102, 103, 104],
                "score": [720.5, 680.0, 590.2, 810.0],
                "grade": ["A", "B", "C", "A"],
            }
        )

        summary = generate_data_quality_summary(df)

        assert summary["row_count"] == 4
        assert summary["column_count"] == 3
        assert summary["column_names"] == ["user_id", "score", "grade"]
        assert "int" in summary["data_types"]["user_id"]
        assert "float" in summary["data_types"]["score"]
        assert "object" in summary["data_types"]["grade"] or "str" in summary["data_types"]["grade"]
        assert summary["duplicate_rows"] == 0

    def test_quality_summary_missing_values(self):
        """Verify that null/NaN values are accurately tallied per column."""
        df = pd.DataFrame(
            {
                "col_complete": [1, 2, 3, 4],
                "col_one_missing": [10.0, None, 30.0, 40.0],
                "col_all_missing": [None, None, None, None],
            }
        )

        summary = generate_data_quality_summary(df)

        assert summary["missing_values"]["col_complete"] == 0
        assert summary["missing_values"]["col_one_missing"] == 1
        assert summary["missing_values"]["col_all_missing"] == 4

    def test_quality_summary_duplicate_rows(self):
        """Verify that duplicate rows are detected and counted accurately."""
        df = pd.DataFrame(
            {
                "applicant_id": ["APP-1", "APP-2", "APP-1", "APP-3", "APP-2"],
                "income": [50000, 60000, 50000, 70000, 60000],
            }
        )

        summary = generate_data_quality_summary(df)

        # There are 5 rows total, 2 of which are identical duplicates of earlier rows
        assert summary["row_count"] == 5
        assert summary["duplicate_rows"] == 2

    def test_quality_summary_invalid_type_raises_error(self):
        """Verify that passing an object other than a pandas DataFrame raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            generate_data_quality_summary({"key": "value"})  # type: ignore

        assert "Expected 'df' to be a pandas DataFrame" in str(exc_info.value)

    def test_quality_summary_does_not_mutate_or_drop_data(self):
        """Verify that generating a quality summary does not mutate the original DataFrame."""
        original_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "val": [10, None, 30],
            }
        )
        copied_df = original_df.copy()

        _ = generate_data_quality_summary(original_df)

        pd.testing.assert_frame_equal(original_df, copied_df)

"""FinWise Lending Credit Risk Assessment Package.

Core modules for data ingestion, quality inspection, and feature processing.
"""

from src.data_pipeline import generate_data_quality_summary, load_csv

__all__ = ["load_csv", "generate_data_quality_summary"]

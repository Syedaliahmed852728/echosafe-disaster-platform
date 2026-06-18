"""
Data validation utilities for Silver and Gold layers.
"""

import pandas as pd
from typing import List, Dict
from backend.config.logger import get_logger

logger = get_logger(__name__)


class DataQualityReport:
    """Stores and reports quality check results."""

    def __init__(self, layer: str, dataset_name: str):
        self.layer = layer
        self.dataset_name = dataset_name
        self.checks = []
        self.passed = 0
        self.failed = 0

    def add_check(self, name: str, passed: bool, details: str = ""):
        self.checks.append({"name": name, "passed": passed, "details": details})
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def is_valid(self) -> bool:
        return self.failed == 0

    def log_summary(self):
        status = "PASSED" if self.is_valid else "FAILED"
        logger.info(
            f"Quality Report [{self.layer}/{self.dataset_name}]: {status} "
            f"({self.passed}/{self.passed + self.failed} checks passed)"
        )
        for c in self.checks:
            icon = "OK" if c["passed"] else "FAIL"
            logger.info(f"  [{icon}] {c['name']}: {c['details']}")


def validate_weather_silver(df: pd.DataFrame) -> DataQualityReport:
    """Validate Silver weather dataset."""
    report = DataQualityReport("silver", "weather_cleaned")
    required = [
        "date",
        "region",
        "province",
        "latitude",
        "longitude",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "precipitation_mm",
        "rainfall_mm",
        "humidity_mean_percent",
        "wind_speed_mean_kmh",
    ]
    missing = [c for c in required if c not in df.columns]
    report.add_check(
        "required_columns",
        len(missing) == 0,
        f"Missing: {missing}" if missing else "All present",
    )
    report.add_check("non_empty", len(df) > 0, f"Rows: {len(df)}")

    if "date" in df.columns and "region" in df.columns:
        dups = df.duplicated(subset=["date", "region"]).sum()
        report.add_check("no_duplicates", dups == 0, f"Duplicates: {dups}")

    if "temperature_mean_c" in df.columns:
        invalid = (
            (df["temperature_mean_c"] < -20) | (df["temperature_mean_c"] > 55)
        ).sum()
        report.add_check("temperature_range", invalid == 0, f"Invalid temps: {invalid}")

    if "humidity_mean_percent" in df.columns:
        invalid = (
            (df["humidity_mean_percent"] < 0) | (df["humidity_mean_percent"] > 100)
        ).sum()
        report.add_check("humidity_range", invalid == 0, f"Invalid humidity: {invalid}")

    report.log_summary()
    return report


def validate_gold_dataset(
    df: pd.DataFrame, label_col: str, dataset_name: str
) -> DataQualityReport:
    """Validate any Gold dataset."""
    report = DataQualityReport("gold", dataset_name)
    report.add_check("non_empty", len(df) > 0, f"Rows: {len(df)}")
    if label_col in df.columns:
        labels = df[label_col].value_counts().to_dict()
        report.add_check("labels_present", len(labels) > 0, f"Labels: {labels}")
        null_count = df[label_col].isnull().sum()
        report.add_check(
            "no_null_labels", null_count == 0, f"Null labels: {null_count}"
        )
    else:
        report.add_check(
            "label_column_exists", False, f"Column '{label_col}' not found"
        )
    report.log_summary()
    return report

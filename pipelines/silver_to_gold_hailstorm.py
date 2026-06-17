#!/usr/bin/env python3
"""
Pipeline 10: Silver to Gold - Hailstorm Risk Dataset.

Builds the ML-ready hailstorm dataset used by:
  - the rule-based hail-severity classifier
  - the production hail-risk binary classifier (Random Forest)

Input  : data/silver/hailstorm_cleaned/hailstorm_events_cleaned.csv
Output : data/gold/hailstorm_risk/hailstorm_risk_dataset.csv

Each row is a station-day with the observed `hail_observed` label, the daily
surface aggregates, plus engineered seasonal / severity features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.logger import get_logger
from config.settings import SETTINGS
from ml_models.hailstorm.severity_classifier import classify_hail_severity
from pipelines.utils.data_validators import validate_gold_dataset

logger = get_logger("pipeline.10")

REQUIRED_FIELDS = (
    "event_id",
    "date",
    "latitude",
    "longitude",
    "temperature_max_c",
    "dew_point_mean_c",
    "rh_min_pct",
    "wind_speed_mean_ms",
    "wind_gust_max_ms",
    "surface_pressure_min_hpa",
    "precipitation_sum_mm",
    "thunder_hours",
    "hail_observed",
)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def engineer_hailstorm_features() -> Path:
    silver_path = (
        SETTINGS.pipeline.silver_dir
        / "hailstorm_cleaned"
        / "hailstorm_events_cleaned.csv"
    )
    df = pd.read_csv(silver_path)
    logger.info(f"Loaded {len(df)} silver station-days from {silver_path.name}")

    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            raise KeyError(f"Silver CSV missing required field: {col}")
    df = df.dropna(subset=list(REQUIRED_FIELDS))

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"].apply(_season)
    df["is_premonsoon"] = df["month"].isin([3, 4, 5]).astype(int)

    df["hail_severity_label"] = df.apply(
        lambda r: classify_hail_severity(
            gust_ms=float(r["wind_gust_max_ms"]),
            thunder_hours=int(r["thunder_hours"]),
            precipitation_mm=float(r["precipitation_sum_mm"]),
            observed=bool(int(r["hail_observed"])),
        ),
        axis=1,
    )

    cols = [
        "event_id",
        "date",
        "year",
        "month",
        "day_of_year",
        "season",
        "is_premonsoon",
        "icao",
        "station_name",
        "region",
        "province",
        "distance_to_region_km",
        "in_pakistan_bbox",
        "latitude",
        "longitude",
        "temperature_max_c",
        "temperature_min_c",
        "dew_point_mean_c",
        "rh_mean_pct",
        "rh_min_pct",
        "wind_speed_mean_ms",
        "wind_gust_max_ms",
        "surface_pressure_min_hpa",
        "surface_pressure_drop_hpa",
        "precipitation_sum_mm",
        "rain_sum_mm",
        "snow_sum_mm",
        "cloud_cover_mean_pct",
        "thunder_hours",
        "n_hail_reports",
        "wxcodes_examples",
        "hail_observed",
        "hail_severity_label",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)

    output_dir = SETTINGS.pipeline.gold_dir / "hailstorm_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hailstorm_risk_dataset.csv"
    df.to_csv(output_path, index=False)

    severity_dist = df["hail_severity_label"].value_counts().to_dict()
    pos = int(df["hail_observed"].sum())
    logger.info(
        f"Hailstorm Gold dataset: {len(df)} station-days, "
        f"{pos} positive hail days ({pos / max(len(df), 1):.2%}), "
        f"severity={severity_dist}"
    )
    validate_gold_dataset(df, "hail_severity_label", "hailstorm_risk")
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 10: Silver to Gold - Hailstorm")
    logger.info("Builds the hail-day classification dataset.")
    logger.info("=" * 60)
    engineer_hailstorm_features()
    logger.info("Pipeline 10 completed successfully")

#!/usr/bin/env python3
"""
Pipeline 09: Silver to Gold - Earthquake Risk Dataset

Builds the ML-ready earthquake dataset used by:
  - the existing severity classifier (rule-based; pipeline 13)
  - the upcoming magnitude estimator (Random Forest)

Input  : data/silver/earthquake_cleaned/earthquake_events_cleaned.csv
Output : data/gold/earthquake_risk/earthquake_risk_dataset.csv

Reminder: the earthquake module classifies severity of detected events.
It does NOT predict earthquake occurrence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from backend.config.settings import SETTINGS
from backend.config.logger import get_logger
from backend.ml_models.earthquake.severity_classifier import classify_severity
from backend.pipelines.utils.data_validators import validate_gold_dataset
from backend.pipelines.utils.incremental import covers_upstream, target_window

logger = get_logger("pipeline.09")

REQUIRED_FIELDS = ("event_id", "event_time", "magnitude", "latitude", "longitude")
SHALLOW_DEPTH_THRESHOLD_KM = 70.0


def _magnitude_bucket(mag: float) -> str:
    if mag >= 7.0:
        return "7+"
    if mag >= 6.0:
        return "6-7"
    if mag >= 5.0:
        return "5-6"
    if mag >= 4.0:
        return "4-5"
    return "3-4"


def _csv_window(path: Path, col: str):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=[col])
    except Exception:
        return None
    ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return (ts.min().to_pydatetime(), ts.max().to_pydatetime())


def engineer_earthquake_features() -> Path:
    silver_path = (
        SETTINGS.pipeline.silver_dir
        / "earthquake_cleaned"
        / "earthquake_events_cleaned.csv"
    )
    output_dir = SETTINGS.pipeline.gold_dir / "earthquake_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "earthquake_risk_dataset.csv"

    target = target_window()
    gold_window = _csv_window(output_path, "event_time")
    silver_window = _csv_window(silver_path, "event_time")
    if covers_upstream(gold_window, silver_window, target):
        logger.info(
            f"Gold earthquake already covers all silver events "
            f"(silver max {silver_window[1].date()}); skipping recompute."
        )
        return output_path

    df = pd.read_csv(silver_path)
    logger.info(f"Loaded {len(df)} silver events from {silver_path.name}")

    # Defensive: drop any rows the silver layer didn't catch.
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            raise KeyError(f"Silver CSV missing required field: {col}")
    df = df.dropna(subset=list(REQUIRED_FIELDS)).drop_duplicates(subset=["event_id"])

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"])

    # Temporal features (useful for time-based splits + report aggregations).
    df["year"] = df["event_time"].dt.year
    df["month"] = df["event_time"].dt.month
    df["quarter"] = df["event_time"].dt.quarter

    # Depth-derived features. depth_km may be NaN for some events.
    df["depth_km"] = pd.to_numeric(df.get("depth_km"), errors="coerce")
    df["is_shallow"] = (
        df["depth_km"].fillna(999) <= SHALLOW_DEPTH_THRESHOLD_KM
    ).astype(int)

    # Canonical severity labels (matches ml_models/earthquake/severity_classifier.py).
    df["earthquake_severity_label"] = df.apply(
        lambda r: classify_severity(
            float(r["magnitude"]),
            float(r["depth_km"]) if pd.notna(r["depth_km"]) else 100.0,
        ),
        axis=1,
    )
    df["magnitude_bucket"] = df["magnitude"].apply(_magnitude_bucket)

    cols = [
        "event_id",
        "event_time",
        "year",
        "month",
        "quarter",
        "region",
        "province",
        "distance_to_region_km",
        "in_pakistan_bbox",
        "latitude",
        "longitude",
        "magnitude",
        "magnitude_bucket",
        "depth_km",
        "is_shallow",
        "place",
        "source_type",
        "earthquake_severity_label",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("event_time").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    severity_dist = df["earthquake_severity_label"].value_counts().to_dict()
    logger.info(f"Earthquake Gold dataset: {len(df)} events, severity={severity_dist}")
    validate_gold_dataset(df, "earthquake_severity_label", "earthquake_risk")
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 09: Silver to Gold - Earthquake")
    logger.info("NOTE: Earthquake module classifies severity only.")
    logger.info("It does NOT predict earthquake occurrence.")
    logger.info("=" * 60)
    engineer_earthquake_features()
    logger.info("Pipeline 09 completed successfully")

#!/usr/bin/env python3
"""
Pipeline 11: Silver to Gold - Heatwave Risk Dataset.

Reads the silver region-day frame (with day-of-year climatology already
attached) and produces the ML-ready gold dataset:

  - Engineered predictors: Tmax/Tmin anomalies, 3- and 7-day rolling Tmax,
    diurnal range, consecutive-dry-day streak, calendar / seasonal features.
  - PMD heatwave label: a run of >= 5 consecutive days where Tmax exceeds the
    day-of-year normal by >= 5 deg C. Runs are also numbered (heatwave_event)
    so downstream code can group days by event.
  - Severity label: rule-based summary of Tmax anomaly + apparent (feels-like)
    temperature, including single-day hot events.

Input  : data/silver/heatwave_cleaned/heatwave_daily_cleaned.csv
Output : data/gold/heatwave_risk/heatwave_risk_dataset.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import SETTINGS
from ml_models.heatwave.features import EXCLUDED_REGIONS
from ml_models.heatwave.severity_classifier import classify_heatwave_severity
from pipelines.utils.data_validators import validate_gold_dataset

logger = get_logger("pipeline.11")

PMD_ANOMALY_THRESHOLD_C = 5.0
PMD_MIN_RUN_DAYS = 5

REQUIRED_FIELDS = (
    "event_id",
    "date",
    "region",
    "latitude",
    "longitude",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "tmax_normal",
    "tmax_p90",
)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _runs_of_true(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Label each True element with an event id if it belongs to a run of
    True of length >= ``min_len``; 0 otherwise."""
    out = np.zeros(mask.size, dtype=int)
    event = 0
    i = 0
    while i < mask.size:
        if mask[i]:
            j = i
            while j < mask.size and mask[j]:
                j += 1
            if (j - i) >= min_len:
                event += 1
                out[i:j] = event
            i = j
        else:
            i += 1
    return out


def _dry_streak(precip: np.ndarray) -> np.ndarray:
    """Running count of consecutive zero-precipitation days (resets on any rain)."""
    out = np.zeros(precip.size, dtype=int)
    streak = 0
    for i, p in enumerate(precip):
        if p is None or np.isnan(p) or p <= 0:
            streak += 1
        else:
            streak = 0
        out[i] = streak
    return out


def engineer_heatwave_features() -> Path:
    silver_path = (
        SETTINGS.pipeline.silver_dir
        / "heatwave_cleaned"
        / "heatwave_daily_cleaned.csv"
    )
    df = pd.read_csv(silver_path)
    logger.info(f"Loaded {len(df)} silver region-days from {silver_path.name}")

    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            raise KeyError(f"Silver CSV missing required field: {col}")
    df = df.dropna(subset=list(REQUIRED_FIELDS))

    # Drop alpine / high-altitude regions that trip the PMD anomaly rule on
    # cool baselines without any operational heatwave risk. See
    # ml_models/heatwave/features.py for the rationale and the canonical list.
    before_excl = len(df)
    df = df[~df["region"].isin(EXCLUDED_REGIONS)].copy()
    if len(df) < before_excl:
        logger.info(
            f"Excluded {before_excl - len(df)} rows from regions "
            f"{list(EXCLUDED_REGIONS)} (no operational heatwave risk)"
        )

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["region", "date"]).reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"].apply(_season)
    df["is_premonsoon"] = df["month"].isin([3, 4, 5, 6]).astype(int)

    df["tmax_anomaly"] = df["temperature_2m_max"] - df["tmax_normal"]
    df["diurnal_range_c"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Compute rolling means and per-region tmin climatology + label per region
    # so windows never bleed across regions.
    out_pieces = []
    for region, g in df.groupby("region", sort=False):
        g = g.sort_values("date").copy()
        g["tmax_roll3"] = g["temperature_2m_max"].rolling(3, min_periods=1).mean()
        g["tmax_roll7"] = g["temperature_2m_max"].rolling(7, min_periods=1).mean()
        # Day-of-year tmin normal computed lazily here (could also live in silver).
        tmin_normal = g.groupby("day_of_year")["temperature_2m_min"].transform("mean")
        g["tmin_normal"] = tmin_normal
        g["tmin_anomaly"] = g["temperature_2m_min"] - g["tmin_normal"]
        g["dry_streak"] = _dry_streak(g["precipitation_sum"].to_numpy(dtype=float))

        hot = (g["tmax_anomaly"] >= PMD_ANOMALY_THRESHOLD_C).to_numpy()
        hot = np.nan_to_num(hot, nan=0).astype(bool)
        g["heatwave_event"] = _runs_of_true(hot, PMD_MIN_RUN_DAYS)
        g["is_heatwave"] = (g["heatwave_event"] > 0).astype(int)
        out_pieces.append(g)
    df = pd.concat(out_pieces, ignore_index=True)

    df["heatwave_severity_label"] = df.apply(
        lambda r: classify_heatwave_severity(
            tmax_anomaly=float(r["tmax_anomaly"]),
            apparent_tmax=float(r["apparent_temperature_max"]),
            is_heatwave=bool(int(r["is_heatwave"])),
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
        "region",
        "province",
        "latitude",
        "longitude",
        "in_pakistan_bbox",
        "distance_to_region_km",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "shortwave_radiation_sum",
        "wind_speed_10m_max",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "tmax_normal",
        "tmax_p90",
        "tmin_normal",
        "tmax_anomaly",
        "tmin_anomaly",
        "tmax_roll3",
        "tmax_roll7",
        "diurnal_range_c",
        "dry_streak",
        "heatwave_event",
        "is_heatwave",
        "heatwave_severity_label",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["date", "region"]).reset_index(drop=True)

    output_dir = SETTINGS.pipeline.gold_dir / "heatwave_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "heatwave_risk_dataset.csv"
    df.to_csv(output_path, index=False)

    severity_dist = df["heatwave_severity_label"].value_counts().to_dict()
    pos = int(df["is_heatwave"].sum())
    n_events = int(df["heatwave_event"].max() or 0)
    logger.info(
        f"Heatwave Gold dataset: {len(df)} region-days, "
        f"{pos} heatwave days across {n_events} events "
        f"({pos / max(len(df), 1):.2%}), severity={severity_dist}"
    )
    validate_gold_dataset(df, "heatwave_severity_label", "heatwave_risk")
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 11: Silver to Gold - Heatwave")
    logger.info("Builds the PMD heatwave classification dataset.")
    logger.info("=" * 60)
    engineer_heatwave_features()
    logger.info("Pipeline 11 completed successfully")

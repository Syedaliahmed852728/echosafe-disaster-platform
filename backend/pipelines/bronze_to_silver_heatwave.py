#!/usr/bin/env python3
"""
Pipeline 07: Bronze to Silver - Heatwave Data.

Reads the bronze JSON produced by pipeline 04 and emits a cleaned silver CSV
keyed on (region, date). Adds the day-of-year climatology (smoothed normal
and 90th percentile of Tmax) so the gold pipeline can label heatwave events
without re-computing climatology on every run.

Climatology baseline: 1991 - (current_year - 1), 7-day pooled window.

Input  : data/bronze/heatwave/heatwave_daily_raw.json
Output : data/silver/heatwave_cleaned/heatwave_daily_cleaned.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import timezone

import numpy as np
import pandas as pd

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS
from backend.pipelines.utils.incremental import covers_upstream, target_window

logger = get_logger("pipeline.07")

CLIMATOLOGY_WINDOW_DAYS = 7

REQUIRED_DAILY_COLS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
)


def _doy_climatology(
    region_df: pd.DataFrame, window: int = CLIMATOLOGY_WINDOW_DAYS
) -> pd.DataFrame:
    """Day-of-year normal + 90th percentile of Tmax for one region."""
    doy = region_df["date"].dt.dayofyear.to_numpy()
    vals = region_df["temperature_2m_max"].to_numpy(dtype=float)
    normal = np.full(366, np.nan)
    p90 = np.full(366, np.nan)
    for d in range(1, 367):
        dist = np.minimum(np.abs(doy - d), 366 - np.abs(doy - d))
        sel = vals[dist <= window]
        sel = sel[~np.isnan(sel)]
        if sel.size:
            normal[d - 1] = sel.mean()
            p90[d - 1] = np.percentile(sel, 90)
    return pd.DataFrame(
        {"day_of_year": np.arange(1, 367), "tmax_normal": normal, "tmax_p90": p90}
    )


def _existing_csv_window(path: Path, col: str):
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


def _bronze_window(bronze_path: Path):
    if not bronze_path.exists():
        return None
    try:
        with open(bronze_path) as fh:
            rows = json.load(fh).get("daily_records", [])
    except Exception:
        return None
    dates = pd.to_datetime(
        pd.Series([r.get("date") for r in rows]), utc=True, errors="coerce"
    ).dropna()
    if dates.empty:
        return None
    return (dates.min().to_pydatetime(), dates.max().to_pydatetime())


def bronze_to_silver_heatwave() -> Path:
    bronze_path = (
        SETTINGS.pipeline.bronze_dir / "heatwave" / "heatwave_daily_raw.json"
    )
    silver_dir = SETTINGS.pipeline.silver_dir / "heatwave_cleaned"
    silver_dir.mkdir(parents=True, exist_ok=True)
    output_path = silver_dir / "heatwave_daily_cleaned.csv"

    target = target_window()
    silver_window = _existing_csv_window(output_path, "date")
    bronze_window = _bronze_window(bronze_path)
    if covers_upstream(silver_window, bronze_window, target):
        logger.info(
            f"Silver heatwave already covers all bronze rows "
            f"(bronze max {bronze_window[1].date()}); skipping recompute."
        )
        return output_path

    with open(bronze_path) as fh:
        raw = json.load(fh)

    daily = pd.DataFrame(raw.get("daily_records", []))
    if daily.empty:
        raise ValueError(f"No daily records in {bronze_path}")
    logger.info(f"Loaded {len(daily)} daily rows from bronze JSON")

    for col in REQUIRED_DAILY_COLS:
        if col not in daily.columns:
            raise KeyError(f"Bronze JSON missing required field: {col}")

    daily["date"] = pd.to_datetime(daily["date"], utc=True, errors="coerce")
    daily = daily.dropna(subset=["date"])
    for col in REQUIRED_DAILY_COLS:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["temperature_2m_max"])

    target_start = target[0].replace(tzinfo=timezone.utc)
    daily = daily[daily["date"] >= pd.Timestamp(target_start)].copy()

    # Per-region climatology so each location's "normal" is its own.
    climo_pieces = []
    for region, group in daily.groupby("region"):
        climo = _doy_climatology(group)
        climo["region"] = region
        climo_pieces.append(climo)
    climo_all = pd.concat(climo_pieces, ignore_index=True)

    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily = daily.merge(climo_all, on=["region", "day_of_year"], how="left")

    daily["event_id"] = (
        daily["region"].str.replace(" ", "_") + "_" + daily["date"].dt.strftime("%Y%m%d")
    )
    daily["in_pakistan_bbox"] = 1  # regions are all Pakistani by construction
    daily["distance_to_region_km"] = 0.0
    daily["data_layer"] = "silver"
    daily = daily.sort_values(["region", "date"]).reset_index(drop=True)

    cols = [
        "event_id",
        "date",
        "region",
        "province",
        "latitude",
        "longitude",
        "in_pakistan_bbox",
        "distance_to_region_km",
        "day_of_year",
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
        "data_layer",
    ]
    daily = daily[[c for c in cols if c in daily.columns]]
    daily.to_csv(output_path, index=False)
    logger.info(
        f"Silver heatwave saved: {output_path} "
        f"({len(daily)} region-days; {daily['region'].nunique()} regions)"
    )
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 07: Bronze to Silver - Heatwave")
    logger.info("=" * 60)
    bronze_to_silver_heatwave()
    logger.info("Pipeline 07 completed successfully")

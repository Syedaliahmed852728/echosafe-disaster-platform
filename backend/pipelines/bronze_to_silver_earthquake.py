#!/usr/bin/env python3
"""
Pipeline 05: Bronze to Silver - Earthquake Data

Reads the scraped USGS bronze JSON (which now spans Pakistan + Afghanistan +
parts of India, Iran, China, etc.) and produces a cleaned silver CSV.

Input  : data/bronze/earthquake/earthquake_events_raw.json
Output : data/silver/earthquake_cleaned/earthquake_events_cleaned.csv

Cleaning steps:
  - Drop rows with missing event_id / event_time / latitude / longitude / magnitude
  - Deduplicate by event_id (defensive; the scrape already dedupes)
  - Convert event_time from epoch-milliseconds to UTC datetime
  - For each event, compute the nearest Pakistan region using Haversine
    great-circle distance and store the distance in km
  - Flag events whose epicentre falls inside the Pakistan bounding box
  - Sort by event_time ascending so downstream time-based splits are stable
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import timezone

import numpy as np
import pandas as pd
from backend.config.settings import SETTINGS, PROJECT_ROOT
from backend.config.logger import get_logger
from backend.pipelines.utils.incremental import covers_upstream, target_window

logger = get_logger("pipeline.05")

# Loose Pakistan bbox (matches the old earthquake client default).
PAK_BBOX = {"min_lat": 23.5, "max_lat": 37.5, "min_lon": 60.5, "max_lon": 77.5}

REQUIRED_FIELDS = ("event_id", "event_time", "latitude", "longitude", "magnitude")


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two arrays of points."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _attach_nearest_region(df: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Add `region`, `province`, `distance_to_region_km` via Haversine search."""
    region_lats = regions["latitude"].to_numpy()
    region_lons = regions["longitude"].to_numpy()
    region_names = regions["region"].to_numpy()
    region_provs = regions["province"].to_numpy()

    nearest_region: list[str] = []
    nearest_prov: list[str] = []
    nearest_km: list[float] = []
    for lat, lon in zip(df["latitude"].to_numpy(), df["longitude"].to_numpy()):
        distances = _haversine_km(lat, lon, region_lats, region_lons)
        idx = int(np.argmin(distances))
        nearest_region.append(str(region_names[idx]))
        nearest_prov.append(str(region_provs[idx]))
        nearest_km.append(round(float(distances[idx]), 2))

    df = df.copy()
    df["region"] = nearest_region
    df["province"] = nearest_prov
    df["distance_to_region_km"] = nearest_km
    return df


def _existing_csv_window(path: Path, col: str):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=[col])
    except Exception:
        return None
    if df.empty:
        return None
    ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return (ts.min().to_pydatetime(), ts.max().to_pydatetime())


def _bronze_event_window(bronze_path: Path):
    if not bronze_path.exists():
        return None
    try:
        with open(bronze_path) as fh:
            events = json.load(fh).get("events", [])
    except Exception:
        return None
    ts = [e["event_time"] for e in events if e.get("event_time")]
    if not ts:
        return None
    from datetime import datetime, timezone as _tz
    return (
        datetime.fromtimestamp(min(ts) / 1000, tz=_tz.utc),
        datetime.fromtimestamp(max(ts) / 1000, tz=_tz.utc),
    )


def bronze_to_silver_earthquake() -> Path:
    bronze_path = (
        SETTINGS.pipeline.bronze_dir / "earthquake" / "earthquake_events_raw.json"
    )
    silver_dir = SETTINGS.pipeline.silver_dir / "earthquake_cleaned"
    silver_dir.mkdir(parents=True, exist_ok=True)
    output_path = silver_dir / "earthquake_events_cleaned.csv"

    target = target_window()
    silver_window = _existing_csv_window(output_path, "event_time")
    bronze_window = _bronze_event_window(bronze_path)
    # Bronze events carry hh:mm:ss; silver carries the same timestamps so no
    # slack is needed.
    if covers_upstream(silver_window, bronze_window, target):
        logger.info(
            f"Silver earthquake already covers all bronze events "
            f"(bronze max {bronze_window[1].date()}); skipping recompute."
        )
        return output_path

    with open(bronze_path) as fh:
        raw = json.load(fh)

    events = raw.get("events", [])
    logger.info(f"Loaded {len(events)} raw events from {bronze_path.name}")
    if not events:
        raise ValueError(f"No events found in {bronze_path}")

    df = pd.DataFrame(events)

    # Drop rows missing the fields any downstream module needs.
    initial = len(df)
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            raise KeyError(f"Bronze JSON missing required field: {col}")
    df = df.dropna(subset=list(REQUIRED_FIELDS))
    if len(df) < initial:
        logger.warning(f"Dropped {initial - len(df)} rows with missing required fields")

    # Deduplicate by event_id (keep latest in case of re-fetch updates).
    df = df.drop_duplicates(subset=["event_id"], keep="last")

    # Epoch-ms -> UTC datetime.
    df["event_time"] = pd.to_datetime(
        df["event_time"], unit="ms", utc=True, errors="coerce"
    )
    df = df.dropna(subset=["event_time"])

    # Coerce numerics; rows whose lat/lon/magnitude can't be parsed are dropped.
    for col in ("latitude", "longitude", "magnitude", "depth_km"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "magnitude"])

    # Nearest Pakistan region + great-circle distance.
    regions = pd.read_csv(PROJECT_ROOT / "data" / "reference" / "master_regions.csv")
    df = _attach_nearest_region(df, regions)

    # In-Pakistan flag for downstream filtering.
    df["in_pakistan_bbox"] = (
        df["latitude"].between(PAK_BBOX["min_lat"], PAK_BBOX["max_lat"])
        & df["longitude"].between(PAK_BBOX["min_lon"], PAK_BBOX["max_lon"])
    ).astype(int)

    # EchoSafe is Pakistan-only; drop any event whose epicentre is outside the
    # Pakistan bbox. Historical bronze JSONs may still contain foreign events
    # from the older multi-country scrape window, so the filter runs here.
    before_pk = len(df)
    df = df[df["in_pakistan_bbox"] == 1].copy()
    if len(df) < before_pk:
        logger.info(
            f"Filtered {before_pk - len(df)} non-Pakistan events "
            f"({len(df)} retained)"
        )

    df["data_layer"] = "silver"
    target_start, _ = target
    target_start = target_start.replace(tzinfo=timezone.utc)
    df = df[df["event_time"] >= pd.Timestamp(target_start)].copy()
    df = df.sort_values("event_time").reset_index(drop=True)

    cols = [
        "event_id",
        "event_time",
        "region",
        "province",
        "distance_to_region_km",
        "in_pakistan_bbox",
        "latitude",
        "longitude",
        "magnitude",
        "depth_km",
        "place",
        "source_type",
        "data_layer",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(output_path, index=False)

    in_pk = int(df["in_pakistan_bbox"].sum())
    logger.info(
        f"Silver earthquake saved: {output_path} "
        f"({len(df)} events; {in_pk} inside Pakistan bbox)"
    )
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 05: Bronze to Silver - Earthquake")
    logger.info("=" * 60)
    bronze_to_silver_earthquake()
    logger.info("Pipeline 05 completed successfully")

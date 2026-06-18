#!/usr/bin/env python3
"""
Pipeline 06: Bronze to Silver - Hailstorm Data.

Reads the bronze JSON produced by pipeline 03 and emits a cleaned silver CSV
keyed on (station, observation_date_utc). One row per station-day, carrying:
  - hail_observed (1 if any METAR within that day reported GR/GS, else 0)
  - daily aggregates of the Open-Meteo surface variables:
        temperature_max_c, temperature_min_c
        dew_point_mean_c
        rh_min_pct, rh_mean_pct
        wind_speed_mean_ms, wind_gust_max_ms
        surface_pressure_min_hpa, surface_pressure_drop_hpa
        precipitation_sum_mm, rain_sum_mm, snow_sum_mm
        cloud_cover_mean_pct
        thunder_hours (# of hours with WMO weather_code in {95,96,99})
  - nearest Pakistan region (Haversine, master_regions.csv)
  - in_pakistan_bbox flag

Input  : data/bronze/hailstorm/hailstorm_events_raw.json
Output : data/silver/hailstorm_cleaned/hailstorm_events_cleaned.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import timezone

import numpy as np
import pandas as pd

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS, PROJECT_ROOT
from backend.pipelines.utils.incremental import covers_upstream, target_window

logger = get_logger("pipeline.06")

PAK_BBOX = {"min_lat": 23.5, "max_lat": 37.5, "min_lon": 60.5, "max_lon": 77.5}
THUNDER_CODES = {95, 96, 99}


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _attach_nearest_region(df: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    region_lats = regions["latitude"].to_numpy()
    region_lons = regions["longitude"].to_numpy()
    region_names = regions["region"].to_numpy()
    region_provs = regions["province"].to_numpy()

    nearest_region, nearest_prov, nearest_km = [], [], []
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


def _daily_observations(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "icao",
                "date",
                "hail_observed",
                "n_hail_reports",
                "wxcodes_examples",
            ]
        )
    observations["event_time"] = pd.to_datetime(
        observations["event_time"], unit="ms", utc=True, errors="coerce"
    )
    observations = observations.dropna(subset=["event_time"]).copy()
    observations["date"] = observations["event_time"].dt.floor("D")
    agg = observations.groupby(["icao", "date"], as_index=False).agg(
        n_hail_reports=("wxcodes", "size"),
        wxcodes_examples=("wxcodes", lambda s: ",".join(sorted(set(s))[:3])),
    )
    agg["hail_observed"] = 1
    return agg


def _daily_surface_features(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    hourly = hourly.copy()
    hourly["event_time"] = pd.to_datetime(
        hourly["event_time"], unit="ms", utc=True, errors="coerce"
    )
    hourly = hourly.dropna(subset=["event_time"]).copy()
    hourly["date"] = hourly["event_time"].dt.floor("D")
    numeric_cols = [
        "temperature_c",
        "dew_point_c",
        "rh_pct",
        "wind_speed_ms",
        "wind_gust_ms",
        "surface_pressure_hpa",
        "precip_mm",
        "rain_mm",
        "snow_mm",
        "cloud_cover_pct",
        "weather_code",
    ]
    for c in numeric_cols:
        hourly[c] = pd.to_numeric(hourly[c], errors="coerce")
    hourly["is_thunder_hour"] = (
        hourly["weather_code"].isin(THUNDER_CODES).astype(int)
    )

    grouped = (
        hourly.groupby(["station", "date"], as_index=False)
        .agg(
            temperature_max_c=("temperature_c", "max"),
            temperature_min_c=("temperature_c", "min"),
            dew_point_mean_c=("dew_point_c", "mean"),
            rh_mean_pct=("rh_pct", "mean"),
            rh_min_pct=("rh_pct", "min"),
            wind_speed_mean_ms=("wind_speed_ms", "mean"),
            wind_gust_max_ms=("wind_gust_ms", "max"),
            surface_pressure_min_hpa=("surface_pressure_hpa", "min"),
            surface_pressure_max_hpa=("surface_pressure_hpa", "max"),
            precipitation_sum_mm=("precip_mm", "sum"),
            rain_sum_mm=("rain_mm", "sum"),
            snow_sum_mm=("snow_mm", "sum"),
            cloud_cover_mean_pct=("cloud_cover_pct", "mean"),
            thunder_hours=("is_thunder_hour", "sum"),
        )
        .rename(columns={"station": "icao"})
    )
    grouped["surface_pressure_drop_hpa"] = (
        grouped["surface_pressure_max_hpa"] - grouped["surface_pressure_min_hpa"]
    )
    grouped = grouped.drop(columns=["surface_pressure_max_hpa"])
    return grouped


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


def _bronze_hourly_window(bronze_path: Path):
    if not bronze_path.exists():
        return None
    try:
        with open(bronze_path) as fh:
            rows = json.load(fh).get("hourly_features", [])
    except Exception:
        return None
    ts = [r["event_time"] for r in rows if r.get("event_time") is not None]
    if not ts:
        return None
    from datetime import datetime, timezone as _tz
    return (
        datetime.fromtimestamp(min(ts) / 1000, tz=_tz.utc),
        datetime.fromtimestamp(max(ts) / 1000, tz=_tz.utc),
    )


def bronze_to_silver_hailstorm() -> Path:
    bronze_path = (
        SETTINGS.pipeline.bronze_dir / "hailstorm" / "hailstorm_events_raw.json"
    )
    silver_dir = SETTINGS.pipeline.silver_dir / "hailstorm_cleaned"
    silver_dir.mkdir(parents=True, exist_ok=True)
    output_path = silver_dir / "hailstorm_events_cleaned.csv"

    target = target_window()
    silver_window = _existing_csv_window(output_path, "date")
    bronze_window = _bronze_hourly_window(bronze_path)
    # Bronze events are hourly while silver is per day, so allow up to 1 day
    # of difference to avoid recomputing for the few intra-day hours on
    # bronze's most-recent date.
    if covers_upstream(silver_window, bronze_window, target, slack_days=1):
        logger.info(
            f"Silver hailstorm already covers all bronze hourly rows "
            f"(bronze max {bronze_window[1].date()}); skipping recompute."
        )
        return output_path

    with open(bronze_path) as fh:
        raw = json.load(fh)

    stations = pd.DataFrame(raw.get("stations", []))
    if stations.empty:
        raise ValueError(f"No stations in {bronze_path}")
    stations = stations.rename(columns={"sid": "icao"})

    observations = pd.DataFrame(raw.get("observations", []))
    hourly = pd.DataFrame(raw.get("hourly_features", []))
    logger.info(
        f"Loaded {len(observations)} hail observations / "
        f"{len(hourly)} hourly feature rows / {len(stations)} stations"
    )
    if hourly.empty:
        raise ValueError("No hourly features in bronze; cannot build silver layer")

    daily_obs = _daily_observations(observations)
    daily_feat = _daily_surface_features(hourly)

    df = daily_feat.merge(daily_obs, on=["icao", "date"], how="left")
    df["hail_observed"] = df["hail_observed"].fillna(0).astype(int)
    df["n_hail_reports"] = df["n_hail_reports"].fillna(0).astype(int)
    df["wxcodes_examples"] = df["wxcodes_examples"].fillna("")

    df = df.merge(stations[["icao", "name", "lat", "lon"]], on="icao", how="left")
    df = df.rename(
        columns={"lat": "latitude", "lon": "longitude", "name": "station_name"}
    )
    df = df.dropna(subset=["latitude", "longitude"])

    regions = pd.read_csv(PROJECT_ROOT / "data" / "reference" / "master_regions.csv")
    df = _attach_nearest_region(df, regions)

    df["in_pakistan_bbox"] = (
        df["latitude"].between(PAK_BBOX["min_lat"], PAK_BBOX["max_lat"])
        & df["longitude"].between(PAK_BBOX["min_lon"], PAK_BBOX["max_lon"])
    ).astype(int)

    before_pk = len(df)
    df = df[df["in_pakistan_bbox"] == 1].copy()
    if len(df) < before_pk:
        logger.info(
            f"Filtered {before_pk - len(df)} non-Pakistan rows ({len(df)} retained)"
        )

    df["event_id"] = df["icao"] + "_" + df["date"].dt.strftime("%Y%m%d")
    df["data_layer"] = "silver"
    target_start = target[0].replace(tzinfo=timezone.utc)
    df = df[df["date"] >= pd.Timestamp(target_start)].copy()
    df = df.sort_values(["date", "icao"]).reset_index(drop=True)

    cols = [
        "event_id",
        "date",
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
        "hail_observed",
        "n_hail_reports",
        "wxcodes_examples",
        "data_layer",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(output_path, index=False)

    pos = int(df["hail_observed"].sum())
    logger.info(
        f"Silver hailstorm saved: {output_path} "
        f"({len(df)} station-days; {pos} positive hail days)"
    )
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 06: Bronze to Silver - Hailstorm")
    logger.info("=" * 60)
    bronze_to_silver_hailstorm()
    logger.info("Pipeline 06 completed successfully")

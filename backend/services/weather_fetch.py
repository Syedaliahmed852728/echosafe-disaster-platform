"""Lightweight wrappers around the Open-Meteo forecast API.

We hit one endpoint per region and pull the daily + hourly slices the
risk-engine predictors need (Tmax, Tmin, humidity proxy, wind speed,
precipitation). The dashboard does similar work inline for the user-selected
city; this module is the multi-region batch equivalent used by
``backend.services.batch_predictions``.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS

logger = get_logger(__name__)

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
)

HOURLY_VARS = ("relative_humidity_2m",)


def fetch_region_current_weather(
    latitude: float, longitude: float, *, forecast_days: int = 1
) -> Dict[str, Any]:
    """Return today's weather summary for a single point."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": forecast_days,
        "past_days": 1,
        "timezone": "UTC",
    }
    resp = requests.get(
        OPEN_METEO_FORECAST,
        params=params,
        timeout=SETTINGS.pipeline.request_timeout_seconds * 4,
    )
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily") or {}
    times = daily.get("time", [])
    if not times:
        raise RuntimeError("Open-Meteo returned no daily slice")
    # Find today's index (or the latest available day).
    today_iso = datetime.now(timezone.utc).date().isoformat()
    idx = times.index(today_iso) if today_iso in times else len(times) - 1
    out: Dict[str, Any] = {"date": times[idx]}
    for var in DAILY_VARS:
        series = daily.get(var) or []
        out[var] = series[idx] if idx < len(series) else None
    # Mean humidity from hourly slice for today only.
    hourly = payload.get("hourly") or {}
    h_times = hourly.get("time", [])
    h_rh = hourly.get("relative_humidity_2m", [])
    if h_times and h_rh:
        prefix = times[idx]
        same_day = [v for t, v in zip(h_times, h_rh) if t.startswith(prefix) and v is not None]
        out["humidity_mean_percent"] = round(sum(same_day) / len(same_day), 1) if same_day else None
    else:
        out["humidity_mean_percent"] = None
    return out


def fetch_regions_current_weather(
    regions_df: pd.DataFrame, *, sleep_between: float = 0.4
) -> pd.DataFrame:
    """Return one row per region with today's weather features."""
    rows: List[Dict[str, Any]] = []
    for _, r in regions_df.iterrows():
        try:
            payload = fetch_region_current_weather(
                float(r["latitude"]), float(r["longitude"])
            )
        except Exception as exc:
            logger.warning(
                "Open-Meteo fetch failed for %s: %s", r.get("region"), exc
            )
            payload = {}
        rows.append(
            {
                "region": r["region"],
                "province": r["province"],
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "date": payload.get("date"),
                "temperature_max_c": payload.get("temperature_2m_max"),
                "temperature_min_c": payload.get("temperature_2m_min"),
                "temperature_mean_c": payload.get("temperature_2m_mean"),
                "apparent_temperature_max_c": payload.get("apparent_temperature_max"),
                "wind_speed_mean_kmh": payload.get("wind_speed_10m_max"),
                "precipitation_mm": payload.get("precipitation_sum"),
                "rainfall_mm": payload.get("precipitation_sum"),
                "humidity_mean_percent": payload.get("humidity_mean_percent"),
            }
        )
        time.sleep(sleep_between)
    return pd.DataFrame(rows)

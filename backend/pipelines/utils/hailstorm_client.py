"""
Hailstorm acquisition client (real data only).

Two real sources are stitched together:

1. Iowa Environmental Mesonet (IEM) ASOS archive — network ``PK__ASOS``.
   The METAR ``wxcodes`` field encodes hail observations:
       GR -> hail (>= 5 mm), GS -> small hail / graupel.
   Each Pakistani station is queried with a single bulk request over the
   10-year rolling window. Rows whose wxcodes carry GR/GS become observed
   positive labels in bronze.

2. Open-Meteo ERA5 archive (historical) and forecast (operational) — surface
   reanalysis variables that *are* populated for Pakistan in the public free
   tier:
       temperature_2m, relative_humidity_2m, dew_point_2m,
       wind_speed_10m, wind_direction_10m, wind_gusts_10m,
       surface_pressure, precipitation, rain, snowfall,
       weather_code, cloud_cover.
   Pressure-level CAPE / freezing-level / pressure-level winds are NOT
   served by the archive endpoint for Pakistan, so the EchoSafe pipeline
   uses surface proxies (gusts, pressure drop, low-RH instability) plus the
   WMO weather codes 95/96/99 which encode thunder / thunder-with-hail.

Bronze output: ``data/bronze/hailstorm/hailstorm_events_raw.json``::

    {
      "metadata": { source, retrieved_at, count, lookback_years, stations },
      "stations": [ { sid, name, lat, lon } ],
      "observations": [ { station, icao, lat, lon, event_time (ms epoch),
                          wxcodes, raw_metar, source_type } ],
      "hourly_features": [ { station, event_time, temperature_c,
                             dew_point_c, rh_pct, wind_speed_ms,
                             wind_gust_ms, surface_pressure_hpa,
                             precip_mm, weather_code, cloud_cover_pct } ]
    }

Both endpoints are required. If either fails the function raises so the
caller does not silently fall back to fabricated data.
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS
from backend.pipelines.utils.incremental import covers_window, missing_windows, target_window

logger = get_logger(__name__)

IEM_STATION_LIST = (
    "https://mesonet.agron.iastate.edu/geojson/network/PK__ASOS.geojson"
)
IEM_ASOS_ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OPEN_METEO_HISTORY = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes that imply thunder / hail.
THUNDER_CODES = {95, 96, 99}

# Variables Open-Meteo actually returns for Pakistan in the archive endpoint.
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
]


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


# ---------------------------------------------------------------------------
# IEM ASOS — observed hail labels
# ---------------------------------------------------------------------------

def _get_pakistani_stations() -> List[Dict[str, Any]]:
    resp = requests.get(IEM_STATION_LIST, timeout=30)
    resp.raise_for_status()
    feats = resp.json().get("features", [])
    stations: List[Dict[str, Any]] = []
    for f in feats:
        lon, lat = f["geometry"]["coordinates"]
        props = f["properties"]
        stations.append(
            {
                "sid": props["sid"],
                "name": props.get("sname", props["sid"]),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    if not stations:
        raise RuntimeError("IEM PK__ASOS returned zero stations")
    logger.info(f"IEM PK__ASOS network has {len(stations)} stations")
    return stations


def _is_hail(wxcodes: str) -> bool:
    if not wxcodes or wxcodes in ("M", ""):
        return False
    for group in wxcodes.split():
        g = group.upper()
        if "GR" in g or "GS" in g:
            return True
    return False


def _fetch_station_metars(sid: str, start: datetime, end: datetime) -> str:
    params = {
        "station": sid,
        "data": ["wxcodes", "metar"],
        "sts": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tz": "UTC",
        "format": "onlycomma",
        "missing": "empty",
        "trace": "empty",
        "latlon": "no",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(IEM_ASOS_ENDPOINT, params=params, timeout=180)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(f"IEM {sid} attempt {attempt}/3 failed: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"IEM exhausted retries for {sid}: {last_exc}")


def _parse_hail_observations(
    csv_text: str, station: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if not csv_text.strip():
        return []
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        logger.warning(f"Could not parse IEM CSV for {station['sid']}: {exc}")
        return []
    if df.empty or "wxcodes" not in df.columns:
        return []
    df["wxcodes"] = df["wxcodes"].fillna("").astype(str)
    df = df[df["wxcodes"].apply(_is_hail)]
    if df.empty:
        return []
    df["event_time"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"])
    obs = []
    for _, r in df.iterrows():
        obs.append(
            {
                "station": station["name"],
                "icao": station["sid"],
                "lat": station["lat"],
                "lon": station["lon"],
                "event_time": _ms(r["event_time"]),
                "wxcodes": r["wxcodes"],
                "raw_metar": str(r.get("metar", "")),
                "source_type": "iem_asos",
            }
        )
    return obs


# ---------------------------------------------------------------------------
# Open-Meteo — surface hourly features
# ---------------------------------------------------------------------------

def _ms_per_kmh(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v) / 3.6
    except (TypeError, ValueError):
        return None


def _rows_from_open_meteo(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    h = data.get("hourly", {})
    times = h.get("time", [])
    if not times:
        return []
    out: List[Dict[str, Any]] = []
    for i, ts in enumerate(times):
        out.append(
            {
                "event_time": ts,
                "temperature_c": h["temperature_2m"][i],
                "dew_point_c": h["dew_point_2m"][i],
                "rh_pct": h["relative_humidity_2m"][i],
                "wind_speed_ms": _ms_per_kmh(h["wind_speed_10m"][i]),
                "wind_gust_ms": _ms_per_kmh(h["wind_gusts_10m"][i]),
                "wind_direction_deg": h["wind_direction_10m"][i],
                "surface_pressure_hpa": h["surface_pressure"][i],
                "precip_mm": h["precipitation"][i],
                "rain_mm": h["rain"][i],
                "snow_mm": h["snowfall"][i],
                "weather_code": h["weather_code"][i],
                "cloud_cover_pct": h["cloud_cover"][i],
            }
        )
    return out


def _fetch_open_meteo_history(
    station: Dict[str, Any], start: datetime, end: datetime
) -> List[Dict[str, Any]]:
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(OPEN_METEO_HISTORY, params=params, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                f"Open-Meteo history {station['sid']} attempt {attempt}/3: {exc}"
            )
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(
            f"Open-Meteo exhausted retries for {station['sid']}: {last_exc}"
        )

    rows = _rows_from_open_meteo(data)
    base_ts = pd.Timestamp("1970-01-01", tz="UTC")
    for r in rows:
        ts = pd.Timestamp(r["event_time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        r["event_time"] = int((ts - base_ts).total_seconds() * 1000)
        r["station"] = station["sid"]
    return rows


def fetch_current_features(
    lat: float, lon: float, forecast_days: int = 2
) -> List[Dict[str, Any]]:
    """Live surface-variable pull used by the predictor for operational scoring."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": forecast_days,
        "past_days": 1,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_FORECAST, params=params, timeout=60)
    resp.raise_for_status()
    return _rows_from_open_meteo(resp.json())


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def _load_existing_bronze(path: Path) -> Dict[str, Any]:
    """Read the existing bronze JSON; return an empty skeleton if missing."""
    if not path.exists():
        return {"stations": [], "observations": [], "hourly_features": []}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning(f"Existing bronze unreadable; rebuilding from scratch: {exc}")
        return {"stations": [], "observations": [], "hourly_features": []}
    data.setdefault("stations", [])
    data.setdefault("observations", [])
    data.setdefault("hourly_features", [])
    return data


def _station_window(
    rows: List[Dict[str, Any]], key: str
) -> Optional[Tuple[datetime, datetime]]:
    if not rows:
        return None
    ts = [r["event_time"] for r in rows if r.get("event_time") is not None]
    if not ts:
        return None
    return (
        datetime.fromtimestamp(min(ts) / 1000, tz=timezone.utc),
        datetime.fromtimestamp(max(ts) / 1000, tz=timezone.utc),
    )


def _merge_hourly(
    existing: List[Dict[str, Any]], new: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Dedupe hourly rows by (station, event_time)."""
    by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in existing + new:
        sid = row.get("station")
        ts = row.get("event_time")
        if sid is None or ts is None:
            continue
        by_key[(str(sid), int(ts))] = row
    return list(by_key.values())


def _merge_observations(
    existing: List[Dict[str, Any]], new: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Dedupe IEM observations by (icao, event_time, wxcodes)."""
    by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for row in existing + new:
        sid = row.get("icao")
        ts = row.get("event_time")
        wx = row.get("wxcodes", "")
        if sid is None or ts is None:
            continue
        by_key[(str(sid), int(ts), str(wx))] = row
    return list(by_key.values())


def download_hailstorm_data(output_path: Optional[Path] = None) -> Path:
    """Scrape IEM + Open-Meteo into bronze, fetching only missing per-station windows.

    Raises if either source fails on the gaps it actually had to fill. Reuses
    everything already in the existing bronze JSON.
    """
    output_path = output_path or (
        SETTINGS.pipeline.bronze_dir / "hailstorm" / "hailstorm_events_raw.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start, end = target_window()
    target = (start, end)

    stations = _get_pakistani_stations()
    existing = _load_existing_bronze(output_path)
    existing_hourly_by_sid: Dict[str, List[Dict[str, Any]]] = {}
    for row in existing["hourly_features"]:
        existing_hourly_by_sid.setdefault(str(row.get("station", "")), []).append(row)
    existing_obs_by_sid: Dict[str, List[Dict[str, Any]]] = {}
    for row in existing["observations"]:
        existing_obs_by_sid.setdefault(str(row.get("icao", "")), []).append(row)

    new_hourly: List[Dict[str, Any]] = []
    new_observations: List[Dict[str, Any]] = []
    api_calls = 0

    for s in stations:
        sid = s["sid"]
        sid_hourly = existing_hourly_by_sid.get(sid, [])
        sid_obs = existing_obs_by_sid.get(sid, [])
        window = _station_window(sid_hourly, "event_time")
        if covers_window(window, target):
            logger.info(
                f"-> {sid:6s} {s['name']} cached "
                f"({len(sid_hourly)} hourly rows; "
                f"{window[0].date()}..{window[1].date()})"
            )
            continue

        gaps = missing_windows(window, target)
        logger.info(
            f"-> {sid:6s} {s['name']} -- {len(gaps)} gap(s) to fetch"
        )
        for g_start, g_end in gaps:
            if g_end <= g_start:
                continue
            metar_csv = _fetch_station_metars(sid, g_start, g_end)
            obs = _parse_hail_observations(metar_csv, s)
            if obs:
                logger.info(
                    f"     {len(obs)} hail observations "
                    f"({g_start.date()}..{g_end.date()})"
                )
            new_observations.extend(obs)

            hourly_rows = _fetch_open_meteo_history(s, g_start, g_end)
            logger.info(
                f"     {len(hourly_rows)} hourly feature rows "
                f"({g_start.date()}..{g_end.date()})"
            )
            new_hourly.extend(hourly_rows)
            api_calls += 1
            time.sleep(1)  # courtesy delay between fetches

    merged_hourly = _merge_hourly(existing["hourly_features"], new_hourly)
    merged_obs = _merge_observations(existing["observations"], new_observations)

    if not merged_hourly:
        raise RuntimeError("Open-Meteo returned no hourly rows for any station")

    payload = {
        "metadata": {
            "source": "IEM PK__ASOS + Open-Meteo ERA5 archive (surface)",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "lookback_years": SETTINGS.pipeline.lookback_years,
            "n_observations": len(merged_obs),
            "n_hourly_feature_rows": len(merged_hourly),
            "stations": [s["sid"] for s in stations],
            "n_new_hourly_rows": len(new_hourly),
            "n_new_observations": len(new_observations),
            "n_api_calls": api_calls,
        },
        "stations": stations,
        "observations": merged_obs,
        "hourly_features": merged_hourly,
    }

    with open(output_path, "w") as fh:
        json.dump(payload, fh, default=str)
    logger.info(
        f"Saved {len(merged_obs)} hail observations and "
        f"{len(merged_hourly)} hourly feature rows to {output_path} "
        f"(+{len(new_hourly)} new hourly, +{len(new_observations)} new obs)"
    )
    return output_path

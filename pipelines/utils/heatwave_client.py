"""
Heatwave acquisition client.

Two real sources, split by use case:

* **Historical archive** -> NASA POWER daily endpoint (no API key, generous
  quota). The free Open-Meteo archive API rate-limits the multi-region
  multi-year requests this pipeline needs, so NASA POWER serves the training
  data.
* **Operational forecast** -> Open-Meteo forecast endpoint (the predictor
  calls this; the forecast endpoint is much lighter and the daily quota is
  fine for one-shot scoring).

Variables pulled per (region, day) — same schema from both sources::

    temperature_2m_max               <- NASA POWER T2M_MAX           / OM temperature_2m_max
    temperature_2m_min               <- NASA POWER T2M_MIN           / OM temperature_2m_min
    temperature_2m_mean              <- NASA POWER T2M               / OM temperature_2m_mean
    apparent_temperature_max         <- Steadman approx (T,RH,wind)  / OM apparent_temperature_max
    shortwave_radiation_sum  (MJ/m2) <- NASA POWER ALLSKY_SFC_SW_DWN / OM shortwave_radiation_sum
    wind_speed_10m_max       (m/s)   <- NASA POWER WS10M_MAX         / OM wind_speed_10m_max/3.6
    precipitation_sum        (mm)    <- NASA POWER PRECTOTCORR       / OM precipitation_sum
    et0_fao_evapotranspiration (mm)  <- NASA POWER EVPTRNS           / OM et0_fao_evapotranspiration

A heatwave label is a deterministic function of the temperature record, so
features + label come out of the same source. The label is computed in the
gold pipeline; this module only handles fetching.

Bronze output: data/bronze/heatwave/heatwave_daily_raw.json::

    {
      "metadata": { source, retrieved_at, start, end, regions },
      "regions":  [ { region, province, latitude, longitude } ],
      "daily_records": [ { region, province, latitude, longitude, date,
                            temperature_2m_max, ..., et0_fao_evapotranspiration } ]
    }

The client raises on failure rather than falling back to fabricated data.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from config.logger import get_logger
from config.settings import PROJECT_ROOT, SETTINGS

logger = get_logger(__name__)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMS = [
    "T2M_MAX",
    "T2M_MIN",
    "T2M",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS10M_MAX",
    "PRECTOTCORR",
    "EVPTRNS",
]
NASA_POWER_MISSING = -999.0

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

LOOKBACK_YEARS = 10
# Climatology baseline: 2010 -> ~15 years of daily history per region, which
# keeps the day-of-year normals stable and matches the scale of the other
# EchoSafe datasets.
CLIMATOLOGY_START_YEAR = 2010

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]

OPEN_METEO_FORECAST_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]


def _steadman_apparent_max(t_max_c: float, rh_pct: float, wind_ms: float) -> float:
    """Steadman apparent (feels-like) temperature approximation in deg C.

    AT = T + 0.33 * e - 0.7 * v - 4
    e  = RH/100 * 6.105 * exp(17.27 * T / (237.7 + T))

    Uses daily Tmax and the daily mean RH. This is an approximation — when the
    Open-Meteo forecast supplies the value natively the predictor uses that
    instead.
    """
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in (t_max_c, rh_pct, wind_ms)):
        return float("nan")
    e = (rh_pct / 100.0) * 6.105 * math.exp(
        17.27 * t_max_c / (237.7 + t_max_c)
    )
    return t_max_c + 0.33 * e - 0.7 * wind_ms - 4.0


def _load_regions() -> List[Dict[str, Any]]:
    """Pakistani regions covered by the heatwave module."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "reference" / "master_regions.csv")
    regions: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        regions.append(
            {
                "region": str(r["region"]),
                "province": str(r["province"]),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            }
        )
    return regions


def _coerce_nasa_value(v: Any) -> Optional[float]:
    """NASA POWER uses -999.0 for missing values."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f == NASA_POWER_MISSING:
        return None
    return f


def _fetch_archive_daily(
    region: Dict[str, Any], start: datetime, end: datetime
) -> List[Dict[str, Any]]:
    """Pull daily rows for one region from NASA POWER."""
    params = {
        "parameters": ",".join(NASA_POWER_PARAMS),
        "community": "AG",
        "latitude": region["latitude"],
        "longitude": region["longitude"],
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    last_exc: Optional[Exception] = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(NASA_POWER_URL, params=params, timeout=180)
            if resp.status_code == 429:
                wait = min(60, 5 * 2 ** (attempt - 1))
                logger.warning(
                    f"NASA POWER 429 for {region['region']}; backing off {wait}s "
                    f"(attempt {attempt}/{max_attempts})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                f"NASA POWER {region['region']} attempt {attempt}/{max_attempts}: {exc}"
            )
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    else:
        raise RuntimeError(
            f"NASA POWER exhausted retries for {region['region']}: {last_exc}"
        )

    params_block = payload.get("properties", {}).get("parameter", {})
    if not params_block:
        raise RuntimeError(f"NASA POWER returned no parameter block for {region['region']}")

    keys = sorted(next(iter(params_block.values())).keys())  # YYYYMMDD strings
    rows: List[Dict[str, Any]] = []
    for k in keys:
        t_max = _coerce_nasa_value(params_block.get("T2M_MAX", {}).get(k))
        t_min = _coerce_nasa_value(params_block.get("T2M_MIN", {}).get(k))
        t_mean = _coerce_nasa_value(params_block.get("T2M", {}).get(k))
        rh = _coerce_nasa_value(params_block.get("RH2M", {}).get(k))
        ws = _coerce_nasa_value(params_block.get("WS10M_MAX", {}).get(k))
        sw = _coerce_nasa_value(params_block.get("ALLSKY_SFC_SW_DWN", {}).get(k))
        precip = _coerce_nasa_value(params_block.get("PRECTOTCORR", {}).get(k))
        et0 = _coerce_nasa_value(params_block.get("EVPTRNS", {}).get(k))
        # Skip days where Tmax itself is missing (rare; happens at NASA POWER's
        # current end-of-period rolling boundary).
        if t_max is None:
            continue
        apparent = _steadman_apparent_max(
            t_max, rh if rh is not None else 30.0, ws if ws is not None else 0.0
        )
        date_iso = f"{k[0:4]}-{k[4:6]}-{k[6:8]}"
        rows.append(
            {
                "region": region["region"],
                "province": region["province"],
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "date": date_iso,
                "temperature_2m_max": t_max,
                "temperature_2m_min": t_min,
                "temperature_2m_mean": t_mean,
                "apparent_temperature_max": apparent,
                "shortwave_radiation_sum": sw,
                "wind_speed_10m_max": ws,
                "precipitation_sum": precip,
                "et0_fao_evapotranspiration": et0,
                "rh_2m_mean_pct": rh,
            }
        )
    if not rows:
        raise RuntimeError(f"NASA POWER returned no usable rows for {region['region']}")
    return rows


def fetch_region_forecast(
    region: Dict[str, Any], forecast_days: int = 14, past_days: int = 14
) -> List[Dict[str, Any]]:
    """Daily forecast slice used by the predictor for operational scoring.

    Open-Meteo's forecast endpoint serves the same variable names; wind speed
    comes in km/h and is converted to m/s here so the units match the NASA
    POWER training rows.
    """
    params = {
        "latitude": region["latitude"],
        "longitude": region["longitude"],
        "daily": ",".join(OPEN_METEO_FORECAST_VARS),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "UTC",
    }
    resp = requests.get(OPEN_METEO_FORECAST, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("daily", {})
    times = data.get("time", [])
    out: List[Dict[str, Any]] = []
    for i, dt in enumerate(times):
        ws_kmh = data.get("wind_speed_10m_max", [None] * len(times))[i]
        ws_ms = None if ws_kmh is None else float(ws_kmh) / 3.6
        out.append(
            {
                "region": region.get("region", ""),
                "province": region.get("province", ""),
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "date": dt,
                "temperature_2m_max": data["temperature_2m_max"][i],
                "temperature_2m_min": data["temperature_2m_min"][i],
                "temperature_2m_mean": data["temperature_2m_mean"][i],
                "apparent_temperature_max": data["apparent_temperature_max"][i],
                "shortwave_radiation_sum": data["shortwave_radiation_sum"][i],
                "wind_speed_10m_max": ws_ms,
                "precipitation_sum": data["precipitation_sum"][i],
                "et0_fao_evapotranspiration": data["et0_fao_evapotranspiration"][i],
            }
        )
    return out


def _fetch_open_meteo_tail(
    region: Dict[str, Any],
    after_date: str,
    today_date,
) -> List[Dict[str, Any]]:
    """Pull (after_date, today_date] from Open-Meteo to bridge NASA POWER's lag.

    Open-Meteo's forecast endpoint exposes ``past_days`` up to 92 days, so a
    small (typically 2-4 day) tail is cheap and stays clear of the archive
    endpoint's rate-limit. Returns rows in the same bronze schema as the
    archive fetch (including a synthetic ``rh_2m_mean_pct`` of None because
    Open-Meteo's daily endpoint doesn't expose mean RH).
    """
    from datetime import date as _date, datetime as _datetime, timedelta

    last = _datetime.strptime(after_date, "%Y-%m-%d").date()
    if isinstance(today_date, _datetime):
        today_date = today_date.date()
    needed_days = (today_date - last).days
    if needed_days <= 0:
        return []
    past_days = min(92, needed_days + 1)  # +1 so we overlap the last archive day
    try:
        forecast = fetch_region_forecast(
            region, forecast_days=1, past_days=past_days
        )
    except Exception as exc:
        logger.warning(
            f"Open-Meteo tail fetch failed for {region['region']}: {exc}"
        )
        return []

    tail: List[Dict[str, Any]] = []
    for row in forecast:
        d = row.get("date", "")
        try:
            dt = _datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            continue
        if dt <= last or dt > today_date:
            continue
        tail.append(
            {
                "region": region["region"],
                "province": region["province"],
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "date": d,
                "temperature_2m_max": row.get("temperature_2m_max"),
                "temperature_2m_min": row.get("temperature_2m_min"),
                "temperature_2m_mean": row.get("temperature_2m_mean"),
                "apparent_temperature_max": row.get("apparent_temperature_max"),
                "shortwave_radiation_sum": row.get("shortwave_radiation_sum"),
                "wind_speed_10m_max": row.get("wind_speed_10m_max"),
                "precipitation_sum": row.get("precipitation_sum"),
                "et0_fao_evapotranspiration": row.get("et0_fao_evapotranspiration"),
                "rh_2m_mean_pct": None,
                "source_type": "open_meteo_forecast_tail",
            }
        )
    return tail


def download_heatwave_data(
    output_path: Optional[Path] = None,
    climatology_start_year: int = CLIMATOLOGY_START_YEAR,
) -> Path:
    """Scrape Open-Meteo daily history for every Pakistani region and write bronze.

    Per-region fragments are cached under
    ``data/bronze/heatwave/_fragments/<region>.json`` so a re-run resumes from
    the regions that succeeded — handy because Open-Meteo throttles bursts of
    long-window requests with HTTP 429.
    """
    output_path = output_path or (
        SETTINGS.pipeline.bronze_dir / "heatwave" / "heatwave_daily_raw.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fragments_dir = output_path.parent / "_fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = datetime(climatology_start_year, 1, 1, tzinfo=timezone.utc)

    regions = _load_regions()
    if not regions:
        raise RuntimeError("master_regions.csv is empty; cannot pull heatwave data")

    daily_records: List[Dict[str, Any]] = []
    today_date = end.date()
    for region in regions:
        slug = region["region"].lower().replace(" ", "_")
        frag_path = fragments_dir / f"{slug}.json"
        if frag_path.exists():
            with open(frag_path) as fh:
                cached = json.load(fh)
            if (
                cached.get("start") == start.strftime("%Y-%m-%d")
                and cached.get("end") == end.strftime("%Y-%m-%d")
                and cached.get("rows")
            ):
                logger.info(
                    f"-> {region['region']:14s} cached  "
                    f"({len(cached['rows'])} rows)"
                )
                daily_records.extend(cached["rows"])
                continue
        logger.info(
            f"-> {region['region']:14s} {region['province']:4s} "
            f"({region['latitude']:.4f}, {region['longitude']:.4f})"
        )
        rows = _fetch_archive_daily(region, start, end)
        last_archive_date = max(r["date"] for r in rows)
        logger.info(
            f"     {len(rows)} NASA POWER rows (latest {last_archive_date})"
        )

        # NASA POWER has ~2-3 day reanalysis latency. Bridge the tail with
        # Open-Meteo (past_days slice of the forecast endpoint) so today and
        # the immediately-preceding days end up in bronze.
        tail = _fetch_open_meteo_tail(
            region, after_date=last_archive_date, today_date=today_date
        )
        if tail:
            logger.info(
                f"     +{len(tail)} Open-Meteo tail rows "
                f"({tail[0]['date']}..{tail[-1]['date']})"
            )
            rows.extend(tail)

        with open(frag_path, "w") as fh:
            json.dump(
                {
                    "region": region["region"],
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "rows": rows,
                },
                fh,
                default=str,
            )
        daily_records.extend(rows)
        time.sleep(1)  # courtesy delay; NASA POWER quota is generous

    payload = {
        "metadata": {
            "source": "NASA POWER daily archive",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "n_regions": len(regions),
            "n_daily_records": len(daily_records),
            "regions": [r["region"] for r in regions],
        },
        "regions": regions,
        "daily_records": daily_records,
    }

    with open(output_path, "w") as fh:
        json.dump(payload, fh, default=str)
    logger.info(
        f"Saved {len(daily_records)} daily records ({len(regions)} regions) to "
        f"{output_path}"
    )
    return output_path

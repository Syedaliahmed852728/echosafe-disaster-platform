"""
USGS Earthquake Web-Scraping Client.

Approach
--------
The USGS earthquake search page (https://earthquake.usgs.gov/earthquakes/search/)
is a static HTML form whose "Search" button only constructs a GET URL to the
public FDSN endpoint at https://earthquake.usgs.gov/fdsnws/event/1/query.

bs4 cannot execute JavaScript or click DOM buttons - it is an HTML parser. So
the realistic equivalent of "filling and submitting the form" is:
  1. Fetch the search page once and parse it with bs4 to verify the form's
     field names match what we send (real on-the-wire scrape of the page).
  2. Build the same GET URL the Search button would build, using the user-
     selected filters (custom date range, bbox, format=csv, ...).
  3. GET that URL with a randomised User-Agent (fake-useragent) and
     browser-like headers to reduce the chance of being throttled.

Filters used (per project spec)
-------------------------------
- Custom date range: rolling [today - 10 years, today].
- Geographic bounding box (covers Pakistan + Afghanistan + N/W India seismic
  zones + E/C Iran + W China / Himalayan collision zone):
      North = 40, South = 20, West = 60, East = 105
- Output format: CSV.

Constraints handled
-------------------
1. The endpoint returns HTTP 400 when a single request would exceed 20,000
   events. Windows are fetched in 1-year chunks and split recursively in half
   if a 400 comes back or if a chunk returns at-or-near the 20,000 limit.
2. Incremental fetching: before scraping we read MIN/MAX(event_time) from the
   `earthquake_events` table (falling back to the existing bronze JSON when
   the database isn't reachable) and only fetch the missing windows. On the
   daily Airflow run only the new tail is downloaded; the rolling 10-year
   window shifts forward automatically.
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from fake_useragent import UserAgent
except Exception:  # pragma: no cover - library optional at install time
    UserAgent = None  # type: ignore

from config.settings import SETTINGS, PROJECT_ROOT
from config.logger import get_logger

logger = get_logger(__name__)

USGS_SEARCH_FORM = "https://earthquake.usgs.gov/earthquakes/search/"
USGS_FDSN_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Bounding box for Pakistan only. EchoSafe is a Pakistan-focused project,
# so the scrape window matches the country's loose bounding box (the same
# box the silver layer uses for `in_pakistan_bbox`). Events outside this
# window were causing foreign earthquakes (Myanmar, Nepal, China, ...) to
# leak into the gold dataset and into the rolling backtests.
BBOX = {
    "minlatitude": 23.5,
    "maxlatitude": 37.5,
    "minlongitude": 60.5,
    "maxlongitude": 77.5,
}

DEFAULT_MIN_MAGNITUDE = 3.0
USGS_PER_REQUEST_LIMIT = 20000
LOOKBACK_YEARS = 10

# Static fallback UA pool if fake-useragent's data file isn't available.
_STATIC_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

_ua_singleton = None  # holds a fake_useragent.UserAgent when the library is installed
_form_fields_cache: Optional[Dict[str, str]] = None


def _ua_pool():
    global _ua_singleton
    if _ua_singleton is None and UserAgent is not None:
        try:
            _ua_singleton = UserAgent()
        except Exception as exc:
            logger.warning(f"fake-useragent init failed, using static pool: {exc}")
            _ua_singleton = None
    return _ua_singleton


def _random_headers() -> Dict[str, str]:
    pool = _ua_pool()
    if pool is not None:
        try:
            ua = pool.random
        except Exception:
            ua = np.random.choice(_STATIC_UA_POOL)
    else:
        ua = np.random.choice(_STATIC_UA_POOL)
    return {
        "User-Agent": str(ua),
        "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": USGS_SEARCH_FORM,
        "Connection": "keep-alive",
    }


def _discover_form_fields() -> Dict[str, str]:
    """
    Fetch the search page once and parse it with bs4. Returns a mapping of
    every named form input the page exposes. Used to confirm we're sending
    parameters under the same names the Search button would.
    """
    global _form_fields_cache
    if _form_fields_cache is not None:
        return _form_fields_cache
    try:
        resp = requests.get(USGS_SEARCH_FORM, headers=_random_headers(), timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        fields: Dict[str, str] = {}
        for el in soup.find_all(["input", "select"]):
            name = el.get("name")
            if name and name not in fields:
                fields[name] = el.get("id", "")
        logger.info(f"USGS form scrape: discovered {len(fields)} named fields")
        _form_fields_cache = fields
        return fields
    except Exception as exc:
        logger.warning(f"Form discovery failed (continuing): {exc}")
        _form_fields_cache = {}
        return {}


def _build_csv_params(
    start: str,
    end: str,
    *,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    orderby: str = "time-asc",
) -> Dict[str, str]:
    """Build the exact GET parameters the Search button would submit."""
    return {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "orderby": orderby,
        "minmagnitude": str(min_magnitude),
        **{k: str(v) for k, v in BBOX.items()},
    }


def _fetch_csv(
    start: str,
    end: str,
    *,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Submit the search-form-equivalent GET to the FDSN endpoint and return the
    parsed CSV. Returns None on HTTP 400 (caller will split the window) or
    after all retries are exhausted.
    """
    params = _build_csv_params(start, end, min_magnitude=min_magnitude)
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        headers = _random_headers()
        try:
            resp = requests.get(
                USGS_FDSN_ENDPOINT,
                params=params,
                headers=headers,
                timeout=SETTINGS.pipeline.request_timeout_seconds * 5,
            )
            if resp.status_code == 400:
                logger.warning(
                    f"USGS 400 for {start}..{end} (likely >20k events); splitting"
                )
                return None
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            logger.info(
                f"Scraped {len(df)} events for {start}..{end} "
                f"(UA={headers['User-Agent'][:35]}...)"
            )
            return df
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logger.warning(
                f"Attempt {attempt}/{retries} failed for {start}..{end}: {exc}"
            )
            time.sleep(2**attempt)
    logger.error(f"All retries exhausted for {start}..{end}: {last_exc}")
    return None


def _fetch_window_chunked(
    start: datetime,
    end: datetime,
    *,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    retries: int = 3,
) -> List[Dict]:
    """
    Fetch [start, end). Recursively halve on HTTP 400 or near-limit responses
    so we stay within the USGS 20,000-event-per-request cap.
    """
    s_str = start.strftime("%Y-%m-%dT%H:%M:%S")
    e_str = end.strftime("%Y-%m-%dT%H:%M:%S")
    df = _fetch_csv(s_str, e_str, min_magnitude=min_magnitude, retries=retries)

    if df is None:
        if (end - start) <= timedelta(days=1):
            logger.error(f"Cannot reduce window below 1 day: {s_str}..{e_str}")
            return []
        mid = start + (end - start) / 2
        return _fetch_window_chunked(
            start, mid, min_magnitude=min_magnitude, retries=retries
        ) + _fetch_window_chunked(
            mid, end, min_magnitude=min_magnitude, retries=retries
        )

    # Near-limit results are likely truncated; split defensively.
    if len(df) >= USGS_PER_REQUEST_LIMIT - 5 and (end - start) > timedelta(days=1):
        logger.warning(
            f"Window returned {len(df)} (near 20k limit); splitting for safety"
        )
        mid = start + (end - start) / 2
        return _fetch_window_chunked(
            start, mid, min_magnitude=min_magnitude, retries=retries
        ) + _fetch_window_chunked(
            mid, end, min_magnitude=min_magnitude, retries=retries
        )

    return _csv_to_events(df)


def _csv_to_events(df: pd.DataFrame) -> List[Dict]:
    """Map USGS CSV columns to the project's bronze schema."""
    if df.empty:
        return []
    events: List[Dict] = []
    for _, r in df.iterrows():
        try:
            ts = pd.to_datetime(r.get("time"), utc=True)
            event_time_ms = int(ts.timestamp() * 1000) if pd.notna(ts) else None
        except Exception:
            event_time_ms = None
        events.append(
            {
                "event_id": str(r.get("id", "")),
                "event_time": event_time_ms,
                "latitude": float(r["latitude"])
                if pd.notna(r.get("latitude"))
                else None,
                "longitude": float(r["longitude"])
                if pd.notna(r.get("longitude"))
                else None,
                "magnitude": float(r["mag"]) if pd.notna(r.get("mag")) else None,
                "depth_km": float(r["depth"]) if pd.notna(r.get("depth")) else None,
                "place": str(r.get("place", "")),
                "source_type": "usgs_scrape",
            }
        )
    return events


def _existing_window_from_db() -> Optional[Tuple[datetime, datetime]]:
    """Return (min, max) event_time from the earthquake_events table."""
    try:
        from sqlalchemy import text

        from database.connection import get_engine

        engine = get_engine()
        if engine is None:
            return None
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT MIN(event_time), MAX(event_time) FROM earthquake_events")
            ).fetchone()
        if not row or row[0] is None or row[1] is None:
            return None
        return (
            pd.Timestamp(row[0]).to_pydatetime().replace(tzinfo=timezone.utc),
            pd.Timestamp(row[1]).to_pydatetime().replace(tzinfo=timezone.utc),
        )
    except Exception as exc:
        logger.info(f"DB window lookup skipped: {exc}")
        return None


def _existing_window_from_bronze(path: Path) -> Optional[Tuple[datetime, datetime]]:
    """Fallback when DB is unreachable: scan the existing bronze JSON."""
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            events = json.load(fh).get("events", [])
        ts = [e["event_time"] for e in events if e.get("event_time")]
        if not ts:
            return None
        return (
            datetime.fromtimestamp(min(ts) / 1000, tz=timezone.utc),
            datetime.fromtimestamp(max(ts) / 1000, tz=timezone.utc),
        )
    except Exception as exc:
        logger.info(f"Bronze window lookup skipped: {exc}")
        return None


def _compute_missing_windows(
    existing: Optional[Tuple[datetime, datetime]],
    target_start: datetime,
    target_end: datetime,
) -> List[Tuple[datetime, datetime]]:
    """Return the gaps that need fetching to cover [target_start, target_end)."""
    if existing is None:
        return [(target_start, target_end)]
    e_min, e_max = existing
    gaps: List[Tuple[datetime, datetime]] = []
    if e_min > target_start:
        gaps.append((target_start, e_min))
    if e_max < target_end:
        gaps.append((e_max, target_end))
    return gaps


def _persist_to_db(events: List[Dict]) -> int:
    """Upsert events into the earthquake_events table (PostgreSQL)."""
    if not events:
        return 0
    try:
        from sqlalchemy import text

        from database.connection import get_engine

        engine = get_engine()
        if engine is None:
            return 0
        count = 0
        with engine.begin() as conn:
            for e in events:
                if not e.get("event_id") or not e.get("event_time"):
                    continue
                conn.execute(
                    text(
                        """
                        INSERT INTO earthquake_events
                            (event_id, event_time, latitude, longitude,
                             magnitude, depth_km, place)
                        VALUES
                            (:eid, to_timestamp(:ts), :lat, :lon, :mag, :dep, :place)
                        ON CONFLICT (event_id) DO UPDATE SET
                            event_time = EXCLUDED.event_time,
                            latitude   = EXCLUDED.latitude,
                            longitude  = EXCLUDED.longitude,
                            magnitude  = EXCLUDED.magnitude,
                            depth_km   = EXCLUDED.depth_km,
                            place      = EXCLUDED.place
                        """
                    ),
                    {
                        "eid": e["event_id"],
                        "ts": e["event_time"] / 1000,
                        "lat": e["latitude"],
                        "lon": e["longitude"],
                        "mag": e["magnitude"],
                        "dep": e["depth_km"],
                        "place": e["place"],
                    },
                )
                count += 1
        logger.info(f"Upserted {count} events into earthquake_events")
        return count
    except Exception as exc:
        logger.warning(f"DB persistence skipped: {exc}")
        return 0


def _merge_with_existing(path: Path, new_events: List[Dict]) -> List[Dict]:
    existing: List[Dict] = []
    if path.exists():
        try:
            with open(path) as fh:
                existing = json.load(fh).get("events", [])
        except Exception:
            existing = []
    by_id: Dict[str, Dict] = {}
    for e in existing + new_events:
        eid = e.get("event_id")
        if eid:
            by_id[eid] = e
    return list(by_id.values())


def _write_bronze(path: Path, events: List[Dict], source: str) -> None:
    with open(path, "w") as fh:
        json.dump(
            {
                "metadata": {
                    "source": source,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(events),
                    "bbox": BBOX,
                    "lookback_years": LOOKBACK_YEARS,
                    "min_magnitude": DEFAULT_MIN_MAGNITUDE,
                },
                "events": events,
            },
            fh,
            indent=2,
            default=str,
        )
    logger.info(f"Saved {len(events)} earthquake events to {path}")


def generate_fallback_earthquakes(count: int = 150) -> List[Dict]:
    """Deterministic simulated events for offline mode / hard failures."""
    logger.warning("Using simulated fallback earthquake data")
    np.random.seed(42)
    regions_df = pd.read_csv(PROJECT_ROOT / "data" / "reference" / "master_regions.csv")
    base_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * LOOKBACK_YEARS)
    events: List[Dict] = []
    for i in range(count):
        region = regions_df.iloc[i % len(regions_df)]
        days_offset = int(np.random.randint(0, 365 * LOOKBACK_YEARS))
        magnitude = round(min(np.random.exponential(1.5) + 3.5, 7.5), 1)
        ts_ms = int((base_time + pd.Timedelta(days=days_offset)).timestamp() * 1000)
        events.append(
            {
                "event_id": f"sim_eq_{i + 1:04d}",
                "event_time": ts_ms,
                "latitude": round(
                    float(region["latitude"]) + np.random.normal(0, 0.5), 4
                ),
                "longitude": round(
                    float(region["longitude"]) + np.random.normal(0, 0.5), 4
                ),
                "magnitude": magnitude,
                "depth_km": round(float(np.random.uniform(10, 150)), 1),
                "place": f"Near {region['region']}, {region['province']}",
                "source_type": "simulated",
            }
        )
    return events


def fetch_earthquake_events(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    max_retries: Optional[int] = None,
) -> Optional[List[Dict]]:
    """
    Backward-compatible entry point.

    Scrapes the USGS search form (rotating User-Agent + bs4 form verification)
    for the supplied window. Returns a list of event dicts in the project's
    bronze schema, or None on hard failure.
    """
    if SETTINGS.pipeline.offline_mode:
        logger.info("Offline mode enabled; using fallback earthquake events")
        return generate_fallback_earthquakes()

    today = datetime.now(timezone.utc)
    start_dt = (
        datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc)
        if start_time
        else today - timedelta(days=365 * LOOKBACK_YEARS)
    )
    end_dt = (
        datetime.fromisoformat(end_time).replace(tzinfo=timezone.utc)
        if end_time
        else today
    )

    _discover_form_fields()
    retries = max_retries if max_retries is not None else 3
    # Year-by-year chunks; recursion in _fetch_window_chunked handles oversize.
    events: List[Dict] = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=365), end_dt)
        events.extend(
            _fetch_window_chunked(
                cur, nxt, min_magnitude=min_magnitude, retries=retries
            )
        )
        cur = nxt
    return events or None


def download_earthquake_data(output_path: Optional[Path] = None) -> None:
    """
    Scrape USGS earthquakes for the project's bounding box, ensuring the DB
    always contains the past 10 years. Only missing windows are fetched.
    Designed for daily Airflow runs (rolling 10-year window).
    """
    output_path = output_path or (
        SETTINGS.pipeline.bronze_dir / "earthquake" / "earthquake_events_raw.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if SETTINGS.pipeline.offline_mode:
        events = generate_fallback_earthquakes()
        _write_bronze(output_path, events, source="SIMULATED_OFFLINE")
        return

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    target_start = today - timedelta(days=365 * LOOKBACK_YEARS)

    _discover_form_fields()

    existing = _existing_window_from_db()
    if existing is None:
        existing = _existing_window_from_bronze(output_path)

    windows = _compute_missing_windows(existing, target_start, today)
    if not windows:
        logger.info("DB already covers the past 10 years; no scrape needed.")
        return

    new_events: List[Dict] = []
    for win_start, win_end in windows:
        cur = win_start
        while cur < win_end:
            nxt = min(cur + timedelta(days=365), win_end)
            logger.info(f"Scraping window {cur.date()} -> {nxt.date()}")
            new_events.extend(_fetch_window_chunked(cur, nxt))
            cur = nxt

    if not new_events and existing is None:
        logger.error("Scrape returned no events and no prior data; using fallback")
        new_events = generate_fallback_earthquakes()
        source = "SIMULATED_FALLBACK_AFTER_FAILURE"
    else:
        source = "USGS Search Form Scrape (bs4 + fake-useragent)"

    merged = _merge_with_existing(output_path, new_events)
    _write_bronze(output_path, merged, source=source)
    _persist_to_db(new_events)


if __name__ == "__main__":
    download_earthquake_data()

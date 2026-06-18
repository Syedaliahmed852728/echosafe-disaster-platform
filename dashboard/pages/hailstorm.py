"""Hailstorm Risk Page - Prediction First with Map.

Behavior:
- Prediction summary is primary.
- Hailstorm map is included.
- No fallback regional batch note under map.
- Table compares actual weather values, actual rule-based prediction, and model prediction.
- Score and confidence are removed from user-facing table/summary.
- None / Low / Medium / High / Critical signal levels supported.
- Regional Overview and Alerts navigation buttons removed.
- Fast loading: only selected city is fetched from Open-Meteo; map uses batch file if available.
"""

from __future__ import annotations

import asyncio
import html as html_escape
import textwrap
from datetime import date, datetime, timedelta
from threading import Thread
from typing import Any

import pandas as pd
import requests
import streamlit as st

from backend.config.settings import SETTINGS
from dashboard.auth.session import logout
from backend.risk_engine.hailstorm_predictor import predict_hailstorm_risk

try:
    import pydeck as pdk
except Exception:
    pdk = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
NOMINATIM_GEOCODING_URL = "https://nominatim.openstreetmap.org/search"

DEFAULT_CITY = "Islamabad"
WEATHER_TIMEZONE = "Asia/Karachi"

RISK_COLORS = {
    "None": "#64748b",
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
    "Unknown": "#94a3b8",
    "Selected": "#102118",
}

WEATHER_CODES = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Heavy drizzle", "🌧️"),
    61: ("Light rain", "🌧️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "⛈️"),
    80: ("Rain showers", "🌦️"),
    81: ("Rain showers", "🌧️"),
    82: ("Heavy rain showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "🌨️"),
    99: ("Severe thunderstorm with hail", "🌨️"),
}


class HailstormDataError(Exception):
    """Controlled hailstorm page error."""


def h(markup: str) -> None:
    cleaned = textwrap.dedent(markup).strip()
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html_escape.escape(str(value if value is not None else ""))


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def english_display_name(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    fallback = str(fallback or "").strip()

    if text and _is_ascii(text):
        return text

    if fallback and _is_ascii(fallback):
        return fallback

    return "Selected Location"


def local_today() -> date:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(WEATHER_TIMEZONE)).date()
    return date.today()


def weather_label(code: Any) -> tuple[str, str]:
    try:
        return WEATHER_CODES.get(int(code), ("Weather update", "🌦️"))
    except Exception:
        return "Weather update", "🌦️"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clean_risk_level(value: Any) -> str:
    label = str(value or "Unknown").strip().title()

    aliases = {
        "No Risk": "None",
        "No": "None",
        "Safe": "None",
        "Normal": "None",
        "Clear": "None",
        "Severe": "Critical",
        "Extreme": "Critical",
        "Very High": "Critical",
        "Moderate": "Medium",
        "Nan": "Unknown",
        "": "Unknown",
    }

    label = aliases.get(label, label)
    return label if label in RISK_COLORS else "Unknown"


def risk_color(level: str) -> str:
    return RISK_COLORS.get(clean_risk_level(level), RISK_COLORS["Unknown"])


def risk_rgba(level: str) -> list[int]:
    level = clean_risk_level(level)

    if level == "Critical":
        return [239, 68, 68, 225]
    if level == "High":
        return [249, 115, 22, 220]
    if level == "Medium":
        return [245, 158, 11, 210]
    if level == "Low":
        return [34, 197, 94, 195]
    if level == "None":
        return [100, 116, 139, 175]

    return [16, 33, 24, 245]


def signal_message(level: str) -> str:
    level = clean_risk_level(level)

    messages = {
        "None": "No hailstorm expected today.",
        "Low": "Low hailstorm signal today. Conditions appear stable, with limited storm concern.",
        "Medium": "Moderate hailstorm signal today. Stay alert for changing wind, rainfall, and storm conditions.",
        "High": "High hailstorm signal today. Localized severe storm activity may require caution.",
        "Critical": "Critical hailstorm signal today. Severe storm conditions may require immediate preparedness.",
        "Unknown": "Prediction signal is unavailable for today.",
    }

    return messages.get(level, messages["Unknown"])


def actual_hailstorm_signal(
    weather_code: Any,
    precipitation_mm: float,
    rainfall_mm: float,
    wind_gust_kmh: float,
    temperature_drop_c: float,
    storm_intensity_proxy: float,
) -> str:
    """Rule-based actual signal from observed weather values.

    This is the actual/weather-derived prediction used for comparison against
    the trained model output.
    """
    code = int(safe_float(weather_code, 0))
    precipitation = safe_float(precipitation_mm)
    rainfall = safe_float(rainfall_mm)
    gust = safe_float(wind_gust_kmh)
    temp_drop = safe_float(temperature_drop_c)
    storm_proxy = safe_float(storm_intensity_proxy)

    if code in {99} or storm_proxy >= 75 or (gust >= 75 and precipitation >= 12):
        return "Critical"

    if code in {96, 95} or storm_proxy >= 55 or (gust >= 60 and precipitation >= 8) or (
        rainfall >= 18 and gust >= 45
    ):
        return "High"

    if storm_proxy >= 38 or (precipitation >= 6 and gust >= 35) or (
        temp_drop >= 4 and rainfall >= 5
    ):
        return "Medium"

    if storm_proxy >= 20 or precipitation >= 2 or gust >= 28 or temp_drop >= 2:
        return "Low"

    return "None"


def inject_hailstorm_css() -> None:
    h(
        """
        <style>
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(0,102,51,.10), transparent 30%),
                radial-gradient(circle at 92% 12%, rgba(20,148,71,.08), transparent 34%),
                linear-gradient(135deg, #ffffff 0%, #f8fbf7 46%, #eef8f1 100%) !important;
            color: #102118;
        }

        .block-container {
            max-width: 1420px !important;
            padding-top: 1.2rem !important;
            padding-bottom: 2.5rem !important;
        }

        .top-shell {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: .85rem 1rem;
            border-radius: 24px;
            background: rgba(255,255,255,.96);
            border: 1px solid rgba(15,122,58,.14);
            box-shadow: 0 16px 44px rgba(16,33,24,.07);
            margin-bottom: 1rem;
        }

        .top-brand {
            display: flex;
            align-items: center;
            gap: .8rem;
        }

        .top-dot {
            width: .7rem;
            height: .7rem;
            border-radius: 999px;
            background: #38bdf8;
            box-shadow: 0 0 0 6px rgba(56,189,248,.14);
        }

        .top-title {
            color: #102118;
            font-size: 1rem;
            font-weight: 950;
            letter-spacing: -.025em;
        }

        .top-subtitle {
            color: #647067;
            font-size: .78rem;
            font-weight: 700;
            margin-top: .08rem;
        }

        .hail-hero {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.6rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 88% 0%, rgba(255,255,255,.20), transparent 30%),
                linear-gradient(135deg, #0f7a3a 0%, #159447 45%, #38bdf8 100%);
            border: 1px solid rgba(15,122,58,.16);
            box-shadow: 0 24px 70px rgba(15,122,58,.18);
            margin-bottom: 1.2rem;
        }

        .hail-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
            background-size: 42px 42px;
            animation: hailGrid 18s linear infinite;
            mask-image: radial-gradient(circle at 45% 40%, black, transparent 78%);
        }

        @keyframes hailGrid {
            from { background-position: 0 0, 0 0; }
            to { background-position: 42px 42px, 42px 42px; }
        }

        .hero-inner {
            position: relative;
            z-index: 2;
        }

        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: .55rem;
            padding: .48rem .78rem;
            border-radius: 999px;
            background: rgba(255,255,255,.16);
            border: 1px solid rgba(255,255,255,.22);
            color: #ffffff;
            font-size: .76rem;
            font-weight: 950;
            letter-spacing: .09em;
            text-transform: uppercase;
        }

        .pulse-dot {
            width: .62rem;
            height: .62rem;
            border-radius: 999px;
            background: #ffffff;
            animation: pulseDot 1.8s infinite;
        }

        @keyframes pulseDot {
            0% { box-shadow: 0 0 0 0 rgba(255,255,255,.46); }
            70% { box-shadow: 0 0 0 12px rgba(255,255,255,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
        }

        .hero-title {
            color: #ffffff !important;
            font-size: clamp(2rem, 4vw, 3.8rem);
            font-weight: 950;
            line-height: 1;
            letter-spacing: -.065em;
            margin: 1rem 0 .55rem 0;
        }

        .hero-subtitle {
            color: rgba(255,255,255,.88);
            max-width: 1080px;
            font-size: 1rem;
            line-height: 1.65;
            margin: 0;
        }

        .search-panel {
            background: rgba(255,255,255,.96);
            border: 1px solid rgba(15,122,58,.14);
            border-radius: 26px;
            padding: 1.2rem;
            box-shadow: 0 18px 54px rgba(16,33,24,.07);
            margin-bottom: 1.35rem;
        }

        .search-heading {
            color: #102118;
            font-size: 1.28rem;
            font-weight: 950;
            letter-spacing: -.035em;
            margin-bottom: .25rem;
        }

        .search-subcopy {
            color: #647067;
            font-size: .9rem;
            font-weight: 650;
            margin-bottom: .95rem;
        }

        .section-title {
            margin-top: 1.2rem;
            color: #102118;
            font-size: 1.28rem;
            font-weight: 950;
            letter-spacing: -.035em;
        }

        .section-subtitle {
            color: #647067;
            margin-top: .2rem;
            margin-bottom: .9rem;
            line-height: 1.55;
            font-size: .92rem;
        }

        .summary-card {
            background:
                radial-gradient(circle at 88% 9%, rgba(56,189,248,.10), transparent 30%),
                linear-gradient(135deg, #ffffff 0%, #f9fbf7 100%);
            border: 1px solid rgba(15,122,58,.16);
            border-radius: 28px;
            padding: 1.2rem;
            box-shadow: 0 22px 58px rgba(16,33,24,.08);
        }

        .prediction-card {
            background:
                radial-gradient(circle at 90% 5%, rgba(56,189,248,.16), transparent 32%),
                linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
            border: 1px solid rgba(56,189,248,.35);
            border-radius: 24px;
            padding: 1.15rem;
            box-shadow: 0 16px 36px rgba(16,33,24,.06);
        }

        .card-eyebrow {
            color: #647067;
            font-size: .7rem;
            font-weight: 950;
            letter-spacing: .09em;
            text-transform: uppercase;
            margin-bottom: .38rem;
        }

        .card-title {
            color: #102118;
            font-size: 1.22rem;
            font-weight: 950;
            letter-spacing: -.04em;
            margin: 0;
        }

        .card-value {
            font-size: 2.55rem;
            font-weight: 950;
            letter-spacing: -.06em;
            line-height: .95;
            margin-top: .55rem;
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            border-radius: 999px;
            padding: .42rem .72rem;
            font-size: .82rem;
            font-weight: 950;
            margin-top: .7rem;
        }

        .card-message {
            color: #526158;
            line-height: 1.55;
            font-size: .9rem;
            margin-top: .75rem;
        }

        .mini-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
            margin-top: .9rem;
        }

        .mini-metric {
            padding: .8rem;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid #dce8df;
            box-shadow: 0 10px 26px rgba(16,33,24,.04);
        }

        .mini-label {
            color: #647067;
            font-size: .68rem;
            font-weight: 950;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .mini-value {
            color: #102118;
            font-size: 1.08rem;
            font-weight: 950;
            margin-top: .25rem;
        }

        .legend-row {
            display: flex;
            gap: .5rem;
            flex-wrap: wrap;
            margin: .75rem 0 1rem 0;
        }

        .legend-chip {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .38rem .62rem;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #dce8df;
            color: #526158;
            font-size: .78rem;
            font-weight: 850;
        }

        .legend-dot {
            width: .66rem;
            height: .66rem;
            border-radius: 999px;
        }

        .records-wrap {
            overflow-x: auto;
            border-radius: 22px;
            border: 1px solid #dce8df;
            box-shadow: 0 16px 36px rgba(16,33,24,.06);
            background: #ffffff;
        }

        .records-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            color: #102118;
            font-size: .88rem;
        }

        .records-table th {
            background: #e8f7ee;
            color: #0b6b35;
            text-align: left;
            padding: .85rem .9rem;
            font-size: .72rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: .07em;
            border-bottom: 1px solid #cae8d3;
            white-space: nowrap;
        }

        .records-table td {
            padding: .82rem .9rem;
            border-bottom: 1px solid #edf4ef;
            color: #26372d;
            font-weight: 650;
            white-space: nowrap;
        }

        .records-table tr:last-child td {
            border-bottom: 0;
        }

        .badge {
            display: inline-flex;
            padding: .28rem .55rem;
            border-radius: 999px;
            font-size: .74rem;
            font-weight: 950;
            border: 1px solid currentColor;
        }

        [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
        }

        [data-testid="stTextInput"] label,
        [data-testid="stDateInput"] label {
            color: #102118 !important;
            font-weight: 850 !important;
            font-size: .86rem !important;
        }

        [data-testid="stTextInput"] div[data-baseweb="input"],
        [data-testid="stDateInput"] div[data-baseweb="input"] {
            background: #ffffff !important;
            border: 1.5px solid #b9d2c0 !important;
            border-radius: 15px !important;
            min-height: 50px !important;
            overflow: hidden !important;
            box-shadow: 0 10px 26px rgba(16,33,24,.045) !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input {
            color: #102118 !important;
            -webkit-text-fill-color: #102118 !important;
            background: #ffffff !important;
            font-weight: 650 !important;
            caret-color: #0f7a3a !important;
        }

        [data-testid="stFormSubmitButton"] {
            margin-top: 1.55rem !important;
        }

        [data-testid="stFormSubmitButton"] button,
        .stButton button {
            min-height: 50px !important;
            border-radius: 15px !important;
            border: 1px solid #0b6b35 !important;
            color: #ffffff !important;
            font-weight: 950 !important;
            background: linear-gradient(135deg, #0b6b35, #19a957) !important;
            box-shadow: 0 14px 30px rgba(15,122,58,.22) !important;
        }

        @media (max-width: 1050px) {
            .mini-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """
    )


async def _get_json_async(
    url: str,
    params: dict,
    timeout: int = 20,
    headers: dict | None = None,
) -> Any:
    def _request() -> Any:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    return await asyncio.to_thread(_request)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc

    thread = Thread(target=runner)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]

    return result["value"]


def _score_nominatim_result(item: dict) -> int:
    address = item.get("address", {}) or {}
    osm_class = str(item.get("class", "")).lower()
    osm_type = str(item.get("type", "")).lower()

    score = 0

    if osm_class in {"place", "boundary"}:
        score += 50

    if osm_type in {"city", "town", "village", "municipality", "administrative"}:
        score += 40

    if address.get("city") or address.get("town") or address.get("village"):
        score += 30

    if osm_class == "natural" or osm_type in {"peak", "mountain", "hill"}:
        score -= 100

    return score


async def _geocode_open_meteo(city: str) -> dict | None:
    payload = await _get_json_async(
        OPEN_METEO_GEOCODING_URL,
        params={
            "name": city,
            "count": 10,
            "language": "en",
            "format": "json",
            "countryCode": "PK",
        },
        timeout=15,
    )

    results = payload.get("results") or []

    if not results:
        return None

    ranked = sorted(
        results,
        key=lambda item: (
            1 if str(item.get("country_code", "")).upper() == "PK" else 0,
            int(item.get("population") or 0),
        ),
        reverse=True,
    )

    result = ranked[0]

    if "latitude" not in result or "longitude" not in result:
        return None

    return {
        "city": english_display_name(result.get("name"), fallback=city.title()),
        "admin1": english_display_name(result.get("admin1"), fallback=""),
        "country": english_display_name(result.get("country"), fallback="Pakistan"),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
    }


async def _geocode_nominatim(city: str) -> dict | None:
    payload = await _get_json_async(
        NOMINATIM_GEOCODING_URL,
        params={
            "q": f"{city}, Pakistan",
            "format": "jsonv2",
            "limit": 8,
            "addressdetails": 1,
            "namedetails": 1,
            "accept-language": "en",
            "countrycodes": "pk",
        },
        headers={"User-Agent": "EchoSafe-HailstormDashboard/1.0"},
        timeout=15,
    )

    if not isinstance(payload, list) or not payload:
        return None

    ranked = sorted(payload, key=_score_nominatim_result, reverse=True)
    result = ranked[0]

    if _score_nominatim_result(result) < 0:
        return None

    address = result.get("address", {}) or {}
    namedetails = result.get("namedetails", {}) or {}

    city_name = (
        namedetails.get("name:en")
        or address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or city.title()
    )

    return {
        "city": english_display_name(city_name, fallback=city.title()),
        "admin1": english_display_name(address.get("state"), fallback=""),
        "country": english_display_name(address.get("country"), fallback="Pakistan"),
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
    }


async def _geocode_city_async(city: str) -> dict:
    city = (city or DEFAULT_CITY).strip()

    location = await _geocode_open_meteo(city)
    if location:
        return location

    location = await _geocode_nominatim(city)
    if location:
        return location

    raise HailstormDataError(f"No Pakistan city or location found for: {city}")


async def _fetch_forecast_async(latitude: float, longitude: float, start: date, end: date) -> dict:
    return await _get_json_async(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "rain",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "rain_sum",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                ]
            ),
            "timezone": WEATHER_TIMEZONE,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=20,
    )


async def _fetch_archive_async(latitude: float, longitude: float, start: date, end: date) -> dict:
    return await _get_json_async(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "rain",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "rain_sum",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                ]
            ),
            "timezone": WEATHER_TIMEZONE,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        timeout=25,
    )


def _frame_from_payload(payload: dict, key: str) -> pd.DataFrame:
    section = payload.get(key, {}) or {}
    df = pd.DataFrame(section)

    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    return df


async def _fetch_weather_range_async(latitude: float, longitude: float, start: date, end: date) -> dict:
    today = local_today()
    yesterday = today - timedelta(days=1)

    daily_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []

    if start <= yesterday:
        archive_end = min(end, yesterday)
        if start <= archive_end:
            archive = await _fetch_archive_async(latitude, longitude, start, archive_end)
            daily_frames.append(_frame_from_payload(archive, "daily"))
            hourly_frames.append(_frame_from_payload(archive, "hourly"))

    if end >= today:
        forecast_start = max(start, today)
        forecast_end = end
        forecast = await _fetch_forecast_async(latitude, longitude, forecast_start, forecast_end)
        daily_frames.append(_frame_from_payload(forecast, "daily"))
        hourly_frames.append(_frame_from_payload(forecast, "hourly"))

    daily_df = (
        pd.concat([df for df in daily_frames if not df.empty], ignore_index=True)
        if daily_frames
        else pd.DataFrame()
    )

    hourly_df = (
        pd.concat([df for df in hourly_frames if not df.empty], ignore_index=True)
        if hourly_frames
        else pd.DataFrame()
    )

    if not daily_df.empty:
        daily_df = daily_df.drop_duplicates(subset=["time"]).sort_values("time")

    if not hourly_df.empty:
        hourly_df = hourly_df.drop_duplicates(subset=["time"]).sort_values("time")

    if daily_df.empty:
        raise HailstormDataError("No weather records returned for the selected date range.")

    return {
        "daily": daily_df.to_dict("list"),
        "hourly": hourly_df.to_dict("list"),
    }


async def _fetch_hailstorm_bundle_async(city: str, start: date, end: date) -> dict:
    location = await _geocode_city_async(city)
    weather = await _fetch_weather_range_async(
        latitude=location["latitude"],
        longitude=location["longitude"],
        start=start,
        end=end,
    )

    return {
        "location": location,
        "weather": weather,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_hailstorm_bundle(city: str, start: date, end: date) -> dict:
    return _run_async(_fetch_hailstorm_bundle_async(city, start, end))


def build_model_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily = daily_df.copy()

    if daily.empty:
        return daily

    daily["date"] = pd.to_datetime(daily["time"], errors="coerce").dt.date
    daily = daily.dropna(subset=["date"]).copy()

    hourly = daily.attrs.get("hourly_df", pd.DataFrame())

    if not hourly.empty:
        hourly = hourly.copy()
        hourly["time"] = pd.to_datetime(hourly["time"], errors="coerce")
        hourly = hourly.dropna(subset=["time"]).copy()
        hourly["date"] = hourly["time"].dt.date

        hourly_agg = (
            hourly.groupby("date")
            .agg(
                temperature_mean_c=("temperature_2m", "mean"),
                humidity_mean_percent=("relative_humidity_2m", "mean"),
                precipitation_hourly_sum_mm=("precipitation", "sum"),
                rain_hourly_sum_mm=("rain", "sum"),
                wind_speed_mean_kmh=("wind_speed_10m", "mean"),
                wind_gusts_max_kmh=("wind_gusts_10m", "max"),
                hourly_weather_code_max=("weather_code", "max"),
            )
            .reset_index()
        )

        daily = daily.merge(hourly_agg, on="date", how="left")
    else:
        daily["temperature_mean_c"] = None
        daily["humidity_mean_percent"] = None
        daily["precipitation_hourly_sum_mm"] = None
        daily["rain_hourly_sum_mm"] = None
        daily["wind_speed_mean_kmh"] = None
        daily["wind_gusts_max_kmh"] = None
        daily["hourly_weather_code_max"] = None

    numeric_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "temperature_mean_c",
        "humidity_mean_percent",
        "precipitation_hourly_sum_mm",
        "rain_hourly_sum_mm",
        "wind_speed_mean_kmh",
        "wind_gusts_max_kmh",
        "hourly_weather_code_max",
    ]

    for col in numeric_cols:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    if daily["temperature_mean_c"].isna().all():
        daily["temperature_mean_c"] = (
            daily["temperature_2m_max"] + daily["temperature_2m_min"]
        ) / 2

    daily["precipitation_mm"] = daily["precipitation_sum"].fillna(
        daily["precipitation_hourly_sum_mm"]
    ).fillna(0)

    daily["rainfall_mm"] = daily["rain_sum"].fillna(
        daily["rain_hourly_sum_mm"]
    ).fillna(daily["precipitation_mm"]).fillna(0)

    daily["wind_speed_mean_kmh"] = daily["wind_speed_mean_kmh"].fillna(
        daily["wind_speed_10m_max"]
    ).fillna(0)

    daily["wind_gusts_max_kmh"] = daily["wind_gusts_max_kmh"].fillna(
        daily["wind_gusts_10m_max"]
    ).fillna(daily["wind_speed_mean_kmh"]).fillna(0)

    daily["humidity_mean_percent"] = daily["humidity_mean_percent"].fillna(0)

    daily = daily.sort_values("date", ascending=True).copy()
    daily["previous_max_temp_c"] = daily["temperature_2m_max"].shift(1)
    daily["temperature_drop_1d"] = (
        daily["previous_max_temp_c"] - daily["temperature_2m_max"]
    ).clip(lower=0).fillna(0)

    daily["previous_rainfall_mm"] = daily["rainfall_mm"].shift(1).fillna(0)
    daily["rainfall_change_1d"] = (
        daily["rainfall_mm"] - daily["previous_rainfall_mm"]
    ).fillna(0)

    daily["storm_intensity_proxy"] = (
        daily["precipitation_mm"]
        + daily["rainfall_mm"]
        + (daily["wind_speed_mean_kmh"] * 0.30)
        + (daily["humidity_mean_percent"] * 0.05)
        + daily["temperature_drop_1d"].clip(lower=0)
    )

    daily["month"] = pd.to_datetime(daily["date"].astype(str)).dt.month
    daily["season_enc"] = daily["month"].map(
        lambda month: 1 if month in [2, 3, 4] else 2 if month in [5, 6, 7, 8] else 3
    )

    return daily.sort_values("date", ascending=False)


def weather_bundle_to_feature_frame(bundle: dict, start_date: date, end_date: date) -> pd.DataFrame:
    daily = pd.DataFrame(bundle["weather"].get("daily", {}) or {})
    hourly = pd.DataFrame(bundle["weather"].get("hourly", {}) or {})

    if daily.empty:
        return daily

    daily["date"] = pd.to_datetime(daily["time"], errors="coerce").dt.date
    daily = daily.dropna(subset=["date"]).copy()
    daily = daily[(daily["date"] >= start_date) & (daily["date"] <= end_date)].copy()
    daily.attrs["hourly_df"] = hourly

    return build_model_features(daily)


def normalize_model_result(result: dict) -> dict:
    level = clean_risk_level(
        result.get("risk_level")
        or result.get("severity_label")
        or result.get("label")
        or result.get("prediction")
        or "Unknown"
    )

    confidence = result.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = None

    risk_score = result.get("risk_score")
    try:
        risk_score = float(risk_score)
    except Exception:
        risk_score = 0.0

    message = result.get("message") or result.get("reason") or signal_message(level)

    return {
        "risk_level": level,
        "model_prediction": level,
        "confidence": confidence,
        "risk_score": risk_score,
        "message": message,
        "raw": result,
    }


def predict_daily_hailstorm(feature_df: pd.DataFrame, city: str) -> pd.DataFrame:
    rows = []

    for _, row in feature_df.iterrows():
        actual_signal = actual_hailstorm_signal(
            weather_code=row.get("weather_code"),
            precipitation_mm=safe_float(row.get("precipitation_mm")),
            rainfall_mm=safe_float(row.get("rainfall_mm")),
            wind_gust_kmh=safe_float(row.get("wind_gusts_max_kmh")),
            temperature_drop_c=safe_float(row.get("temperature_drop_1d")),
            storm_intensity_proxy=safe_float(row.get("storm_intensity_proxy")),
        )

        input_data = {
            "region": city,
            "temperature_mean_c": safe_float(row.get("temperature_mean_c")),
            "temperature_max_c": safe_float(row.get("temperature_2m_max")),
            "temperature_min_c": safe_float(row.get("temperature_2m_min")),
            "humidity_mean_percent": safe_float(row.get("humidity_mean_percent")),
            "wind_speed_mean_kmh": safe_float(row.get("wind_speed_mean_kmh")),
            "precipitation_mm": safe_float(row.get("precipitation_mm")),
            "rainfall_mm": safe_float(row.get("rainfall_mm")),
            "temperature_drop_1d": safe_float(row.get("temperature_drop_1d")),
            "rainfall_change_1d": safe_float(row.get("rainfall_change_1d")),
            "storm_intensity_proxy": safe_float(row.get("storm_intensity_proxy")),
            "month": int(safe_float(row.get("month"), 3)),
            "season_enc": int(safe_float(row.get("season_enc"), 1)),
        }

        try:
            model_output = predict_hailstorm_risk(input_data)
            normalized = normalize_model_result(model_output)
        except Exception as exc:
            normalized = {
                "risk_level": "Unknown",
                "model_prediction": "Unknown",
                "confidence": None,
                "risk_score": 0.0,
                "message": f"Prediction could not be generated: {exc}",
                "raw": {},
            }

        condition, _ = weather_label(row.get("weather_code"))

        rows.append(
            {
                "date": row.get("date"),
                "condition": condition,
                "precipitation_mm": input_data["precipitation_mm"],
                "rainfall_mm": input_data["rainfall_mm"],
                "humidity_percent": input_data["humidity_mean_percent"],
                "wind_speed_kmh": input_data["wind_speed_mean_kmh"],
                "wind_gust_kmh": safe_float(row.get("wind_gusts_max_kmh")),
                "temperature_drop_c": input_data["temperature_drop_1d"],
                "storm_intensity_proxy": input_data["storm_intensity_proxy"],
                "actual_prediction": actual_signal,
                "model_prediction": normalized["model_prediction"],
                "risk_level": normalized["risk_level"],
                "confidence": normalized["confidence"],
                "risk_score": normalized["risk_score"],
                "message": signal_message(normalized["risk_level"]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["date"],
        ascending=False,
    )


def summarize_prediction(prediction_df: pd.DataFrame, city: str) -> dict:
    today = local_today()

    if prediction_df.empty:
        return {
            "city": city,
            "display_date": today,
            "risk_level": "Unknown",
            "message": signal_message("Unknown"),
            "precipitation_mm": 0,
            "rainfall_mm": 0,
            "wind_gust_kmh": 0,
            "temperature_drop_c": 0,
            "storm_intensity_proxy": 0,
        }

    work = prediction_df.copy()
    today_rows = work[work["date"] == today]

    if not today_rows.empty:
        selected = today_rows.iloc[0]
    else:
        selected = work.sort_values("date", ascending=False).iloc[0]

    level = clean_risk_level(selected.get("model_prediction"))

    return {
        "city": city,
        "display_date": selected.get("date", today),
        "risk_level": level,
        "message": signal_message(level),
        "precipitation_mm": safe_float(selected.get("precipitation_mm")),
        "rainfall_mm": safe_float(selected.get("rainfall_mm")),
        "wind_gust_kmh": safe_float(selected.get("wind_gust_kmh")),
        "temperature_drop_c": safe_float(selected.get("temperature_drop_c")),
        "storm_intensity_proxy": safe_float(selected.get("storm_intensity_proxy")),
    }


def load_regions() -> pd.DataFrame:
    path = SETTINGS.project_root / "data" / "reference" / "master_regions.csv"

    if path.exists():
        df = pd.read_csv(path)
        required = {"region", "province", "latitude", "longitude"}
        missing = required.difference(set(df.columns))

        if missing:
            raise HailstormDataError(
                f"master_regions.csv is missing columns: {', '.join(sorted(missing))}"
            )

        return df

    return pd.DataFrame(
        [
            {"region": "Islamabad", "province": "ICT", "latitude": 33.6844, "longitude": 73.0479},
            {"region": "Rawalpindi", "province": "Punjab", "latitude": 33.5651, "longitude": 73.0169},
            {"region": "Lahore", "province": "Punjab", "latitude": 31.5204, "longitude": 74.3587},
            {"region": "Karachi", "province": "Sindh", "latitude": 24.8607, "longitude": 67.0011},
            {"region": "Peshawar", "province": "KPK", "latitude": 34.0151, "longitude": 71.5249},
            {"region": "Quetta", "province": "Balochistan", "latitude": 30.1798, "longitude": 66.9750},
            {"region": "Multan", "province": "Punjab", "latitude": 30.1575, "longitude": 71.5249},
            {"region": "Sukkur", "province": "Sindh", "latitude": 27.7139, "longitude": 68.8483},
            {"region": "Gilgit", "province": "GB", "latitude": 35.9208, "longitude": 74.3140},
            {"region": "Muzaffarabad", "province": "AJK", "latitude": 34.3700, "longitude": 73.4711},
        ]
    )


def load_batch_hailstorm_predictions() -> pd.DataFrame:
    path = SETTINGS.project_root / "predictions" / "batch" / "latest_disaster_predictions.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "disaster_type" in df.columns:
        df = df[df["disaster_type"].astype(str).str.contains("Hail", case=False, na=False)].copy()

    if df.empty or "region" not in df.columns:
        return pd.DataFrame()

    regions = load_regions()
    df = regions.merge(df, on="region", how="inner", suffixes=("", "_prediction"))

    if df.empty:
        return pd.DataFrame()

    if "risk_level" not in df.columns:
        df["risk_level"] = "Unknown"

    if "risk_score" not in df.columns:
        df["risk_score"] = 0

    df["risk_level"] = df["risk_level"].apply(clean_risk_level)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)

    return df


def build_hailstorm_map_points(
    selected_city: str,
    selected_lat: float,
    selected_lon: float,
    summary: dict,
) -> pd.DataFrame:
    batch_df = load_batch_hailstorm_predictions()
    rows: list[dict] = []

    if not batch_df.empty:
        for _, row in batch_df.iterrows():
            level = clean_risk_level(row.get("risk_level"))
            score = safe_float(row.get("risk_score"), 0)

            rows.append(
                {
                    "region": row.get("region", "Unknown"),
                    "province": row.get("province", ""),
                    "latitude": safe_float(row.get("latitude")),
                    "longitude": safe_float(row.get("longitude")),
                    "risk_level": level,
                    "risk_score": round(score, 1),
                    "message": row.get("message", "Hailstorm prediction available."),
                    "color": risk_rgba(level),
                    "radius": 56000 if level in {"High", "Critical"} else 40000,
                    "label": f"{row.get('region', 'Region')} {level}",
                }
            )

    selected_level = clean_risk_level(summary.get("risk_level"))

    rows.append(
        {
            "region": selected_city,
            "province": "Searched Location",
            "latitude": selected_lat,
            "longitude": selected_lon,
            "risk_level": selected_level,
            "risk_score": None,
            "message": summary.get("message", "Selected location prediction."),
            "color": risk_rgba(selected_level),
            "radius": 70000,
            "label": f"{selected_city} {selected_level}",
        }
    )

    return pd.DataFrame(rows)


def render_hailstorm_legend() -> None:
    h(
        """
        <div class="legend-row">
            <div class="legend-chip"><span class="legend-dot" style="background:#64748b;"></span>None</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#22c55e;"></span>Low</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#f59e0b;"></span>Medium</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#f97316;"></span>High</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#ef4444;"></span>Critical</div>
        </div>
        """
    )


def render_hailstorm_map(summary: dict, location: dict) -> None:
    h(
        """
        <div class="section-title">Pakistan Hailstorm Prediction Map</div>
        <p class="section-subtitle">Regional hailstorm signals are shown on the map. The searched location is highlighted with its latest prediction.</p>
        """
    )

    render_hailstorm_legend()

    map_df = build_hailstorm_map_points(
        selected_city=summary["city"],
        selected_lat=float(location["latitude"]),
        selected_lon=float(location["longitude"]),
        summary=summary,
    )

    if map_df.empty:
        st.info("No map data available.")
        return

    if pdk is not None:
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="color",
            radius_min_pixels=7,
            radius_max_pixels=30,
            pickable=True,
            opacity=0.9,
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_text="label",
            get_size=13,
            get_color=[16, 33, 24, 235],
            get_text_anchor="middle",
            get_alignment_baseline="bottom",
            get_pixel_offset=[0, -18],
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(
                    latitude=float(location["latitude"]),
                    longitude=float(location["longitude"]),
                    zoom=5.2,
                    pitch=0,
                ),
                layers=[point_layer, text_layer],
                tooltip={
                    "html": """
                    <b>{region}</b><br/>
                    {province}<br/>
                    Prediction: <b>{risk_level}</b><br/>
                    {message}
                    """,
                    "style": {
                        "backgroundColor": "white",
                        "color": "#102118",
                        "border": "1px solid #cae8d3",
                        "borderRadius": "10px",
                    },
                },
            ),
            use_container_width=True,
        )
    else:
        st.map(
            map_df.rename(columns={"latitude": "lat", "longitude": "lon"}),
            latitude="lat",
            longitude="lon",
            zoom=5,
            height=470,
        )


def go_to_page(page_name: str) -> None:
    st.session_state.active_page = page_name
    st.rerun()


def render_top_header() -> None:
    top_left, top_right = st.columns([0.84, 0.16], gap="medium")

    with top_left:
        h(
            """
            <div class="top-shell">
                <div class="top-brand">
                    <span class="top-dot"></span>
                    <div>
                        <div class="top-title">EchoSafe Hailstorm Operations</div>
                        <div class="top-subtitle">City-level hailstorm prediction and risk comparison</div>
                    </div>
                </div>
                <div class="top-subtitle">Hailstorm Risk</div>
            </div>
            """
        )

    with top_right:
        if st.button("Logout", use_container_width=True, key="hailstorm_logout"):
            logout()
            st.rerun()

    nav_items = [
        ("Home", "Home"),
        ("Heatwave Risk", "Heatwave Risk"),
        ("Hailstorm Risk", "Hailstorm Risk"),
        ("Earthquake Monitoring", "Earthquake Monitoring"),
        ("Flood", "Flood / Heavy Rainfall"),
    ]

    nav_cols = st.columns(len(nav_items), gap="small")

    for col, (display_label, page_key) in zip(nav_cols, nav_items):
        with col:
            if st.button(display_label, use_container_width=True, key=f"hailstorm_nav_{page_key}"):
                go_to_page(page_key)


def render_hero() -> None:
    h(
        """
        <div class="hail-hero">
            <div class="hero-inner">
                <div class="hero-pill">
                    <span class="pulse-dot"></span>
                    Hailstorm Operations
                </div>
                <h1 class="hero-title">Hailstorm Risk Prediction</h1>
                <p class="hero-subtitle">
                    Search a city, choose a period, and compare observed storm conditions with the predicted hailstorm risk signal.
                </p>
            </div>
        </div>
        """
    )


def render_search_panel() -> None:
    h(
        """
        <div class="search-panel">
            <div class="search-heading">Search Hailstorm Prediction</div>
            <div class="search-subcopy">Select a location and date range to compare recent storm conditions with the expected hailstorm risk.</div>
        </div>
        """
    )

    today = local_today()
    min_date = today - timedelta(days=365)
    max_date = today + timedelta(days=7)

    if "hailstorm_city" not in st.session_state:
        st.session_state["hailstorm_city"] = DEFAULT_CITY

    if "hailstorm_start_date" not in st.session_state:
        st.session_state["hailstorm_start_date"] = today - timedelta(days=6)

    if "hailstorm_end_date" not in st.session_state:
        st.session_state["hailstorm_end_date"] = today

    with st.form("hailstorm_search_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([0.40, 0.19, 0.19, 0.17], gap="medium")

        with c1:
            city_input = st.text_input(
                "Location",
                value=st.session_state["hailstorm_city"],
                placeholder="Search for a city or location",
            )

        with c2:
            start_input = st.date_input(
                "Start Date",
                value=st.session_state["hailstorm_start_date"],
                min_value=min_date,
                max_value=max_date,
            )

        with c3:
            end_input = st.date_input(
                "End Date",
                value=st.session_state["hailstorm_end_date"],
                min_value=min_date,
                max_value=max_date,
            )

        with c4:
            submitted = st.form_submit_button("Compare Prediction", use_container_width=True)

    if submitted:
        if end_input < start_input:
            st.error("End Date cannot be earlier than Start Date.")
            return

        st.session_state["hailstorm_city"] = city_input.strip() or DEFAULT_CITY
        st.session_state["hailstorm_start_date"] = start_input
        st.session_state["hailstorm_end_date"] = end_input
        st.rerun()


def render_prediction_summary(summary: dict) -> None:
    level = clean_risk_level(summary.get("risk_level"))
    color = risk_color(level)

    mini_items = [
        ("Precipitation", f"{summary['precipitation_mm']:.1f} mm"),
        ("Rainfall", f"{summary['rainfall_mm']:.1f} mm"),
        ("Wind Gust", f"{summary['wind_gust_kmh']:.1f} km/h"),
        ("Temp Drop", f"{summary['temperature_drop_c']:.1f}°C"),
    ]

    mini_html = "".join(
        f"<div class='mini-metric'>"
        f"<div class='mini-label'>{esc(label)}</div>"
        f"<div class='mini-value'>{esc(value)}</div>"
        f"</div>"
        for label, value in mini_items
    )

    h(
        f"""
        <div class="summary-card">
            <div class="prediction-card">
                <div class="card-eyebrow">Current Prediction</div>
                <h2 class="card-title">{esc(summary["city"])} Hailstorm Risk</h2>
                <div class="card-value" style="color:{color};">{esc(level)}</div>
                <div class="risk-pill" style="background:{color}18;border:1px solid {color}66;color:{color};">
                    Prediction Date: {esc(summary["display_date"])}
                </div>
                <div class="card-message">
                    {esc(summary["message"])}
                </div>
            </div>
            <div class="mini-metrics">{mini_html}</div>
        </div>
        """
    )


def render_records_table(prediction_df: pd.DataFrame, city: str, start_date: date, end_date: date) -> None:
    h(
        f"""
        <div class="section-title">Compare Actual Results and Predicted Results</div>
        <p class="section-subtitle">{esc(city)} • {esc(start_date)} to {esc(end_date)}</p>
        """
    )

    if prediction_df.empty:
        st.info("No prediction records found for the selected date range.")
        return

    table = prediction_df.sort_values("date", ascending=False).head(20).copy()
    rows = []

    for _, row in table.iterrows():
        actual_level = clean_risk_level(row.get("actual_prediction"))
        model_level = clean_risk_level(row.get("model_prediction"))

        actual_color = risk_color(actual_level)
        model_color = risk_color(model_level)

        rows.append(
            f"""
            <tr>
                <td>{esc(row.get("date", ""))}</td>
                <td>{esc(row.get("condition", ""))}</td>
                <td>{safe_float(row.get("precipitation_mm")):.1f} mm</td>
                <td>{safe_float(row.get("rainfall_mm")):.1f} mm</td>
                <td>{safe_float(row.get("wind_gust_kmh")):.1f} km/h</td>
                <td>{safe_float(row.get("temperature_drop_c")):.1f}°C</td>
                <td><span class="badge" style="color:{actual_color};background:{actual_color}16;">{esc(actual_level)}</span></td>
                <td><span class="badge" style="color:{model_color};background:{model_color}16;">{esc(model_level)}</span></td>
            </tr>
            """
        )

    h(
        f"""
        <div class="records-wrap">
            <table class="records-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Actual Condition</th>
                        <th>Actual Precipitation</th>
                        <th>Actual Rainfall</th>
                        <th>Actual Wind Gust</th>
                        <th>Actual Temp Drop</th>
                        <th>Actual Prediction</th>
                        <th>Model Prediction</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    )


def render_hailstorm_page() -> None:
    inject_hailstorm_css()

    render_top_header()
    render_hero()
    render_search_panel()

    city = st.session_state.get("hailstorm_city", DEFAULT_CITY)
    start_date = st.session_state.get("hailstorm_start_date", local_today() - timedelta(days=6))
    end_date = st.session_state.get("hailstorm_end_date", local_today())

    with st.spinner(f"Running hailstorm prediction for {city}..."):
        try:
            bundle = fetch_hailstorm_bundle(city, start_date, end_date)
            feature_df = weather_bundle_to_feature_frame(bundle, start_date, end_date)

            if feature_df.empty:
                raise HailstormDataError("No prediction data was generated for the selected date range.")

            prediction_df = predict_daily_hailstorm(
                feature_df=feature_df,
                city=bundle["location"]["city"],
            )

            summary = summarize_prediction(
                prediction_df=prediction_df,
                city=bundle["location"]["city"],
            )

        except Exception as exc:
            st.error("Hailstorm prediction could not be completed. Please check the location, date range, model files, or internet connection.")
            st.caption(str(exc))
            return

    top_left, top_right = st.columns([0.95, 1.05], gap="large")

    with top_left:
        render_prediction_summary(summary)

    with top_right:
        render_hailstorm_map(summary, bundle["location"])

    render_records_table(
        prediction_df=prediction_df,
        city=bundle["location"]["city"],
        start_date=start_date,
        end_date=end_date,
    )
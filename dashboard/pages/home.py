"""Home Page - Live Weather Dashboard.

- Real Open-Meteo current weather.
- Real Open-Meteo hourly forecast.
- Real Open-Meteo 7-day forecast.
- Default city: Islamabad.
- Search supports Pakistan cities/locations dynamically.
- Forces displayed location names to English when possible.
- Fixes raw HTML/card rendering issue.
- Fixes search input cursor alignment.
- Top navigation matches app.py PAGES keys.
- Existing helper functions preserved for other dashboard pages.
"""

from __future__ import annotations

import asyncio
import html as html_escape
import textwrap
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Iterable

import pandas as pd
import requests
import streamlit as st

from backend.config.settings import SETTINGS
from dashboard.auth.session import logout

try:
    import pydeck as pdk
except Exception:
    pdk = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from dashboard.utils import dashboard_ui as ui
except Exception:
    ui = None


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_GEOCODING_URL = "https://nominatim.openstreetmap.org/search"

DEFAULT_WEATHER_CITY = "Islamabad"
WEATHER_TIMEZONE = "Asia/Karachi"


RISK_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#fb7185",
    "Critical": "#ef4444",
    "Unknown": "#94a3b8",
    "Pending": "#94a3b8",
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
    56: ("Freezing drizzle", "🌧️"),
    57: ("Freezing drizzle", "🌧️"),
    61: ("Light rain", "🌧️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "⛈️"),
    66: ("Freezing rain", "🌧️"),
    67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"),
    73: ("Snow", "🌨️"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Rain showers", "🌦️"),
    81: ("Rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "⛈️"),
    99: ("Severe thunderstorm with hail", "⛈️"),
}


class LiveWeatherError(Exception):
    """Controlled weather error."""


def h(markup: str) -> None:
    """Render HTML safely without Streamlit treating it as a code block."""
    cleaned = textwrap.dedent(markup).strip()
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html_escape.escape(str(value if value is not None else ""))


def english_display_name(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    fallback = str(fallback or "").strip()

    if not text:
        return fallback

    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return fallback or "Selected Location"


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


def local_now_naive() -> datetime:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(WEATHER_TIMEZONE)).replace(tzinfo=None)
    return datetime.now()


def risk_level(value: object) -> str:
    label = str(value or "Unknown").strip().title()
    aliases = {
        "Severe": "Critical",
        "Very High": "Critical",
        "Extreme": "Critical",
        "Moderate": "Medium",
        "Normal": "Low",
        "None": "Unknown",
        "Nan": "Unknown",
        "": "Unknown",
    }
    label = aliases.get(label, label)
    return label if label in RISK_COLORS else "Unknown"


def risk_to_color(value: object) -> str:
    return RISK_COLORS.get(risk_level(value), "#94a3b8")


def inject_home_css() -> None:
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
            background: #0f7a3a;
            box-shadow: 0 0 0 6px rgba(15,122,58,.12);
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

        .home-nav-row {
            margin-bottom: 1.35rem;
        }

        .home-hero {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.6rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 86% 0%, rgba(255,255,255,.22), transparent 30%),
                linear-gradient(135deg, #064e2f 0%, #0f7a3a 58%, #149447 100%);
            border: 1px solid rgba(6,78,47,.25);
            box-shadow: 0 24px 70px rgba(6,78,47,.20);
            margin-bottom: 1.2rem;
        }

        .home-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
            background-size: 42px 42px;
            animation: homeGrid 18s linear infinite;
            mask-image: radial-gradient(circle at 45% 40%, black, transparent 78%);
        }

        @keyframes homeGrid {
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
            background: rgba(255,255,255,.14);
            border: 1px solid rgba(255,255,255,.20);
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
            color: rgba(255,255,255,.86);
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
            margin-top: 1.25rem;
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

        .current-card {
            background:
                radial-gradient(circle at 88% 9%, rgba(15,122,58,.15), transparent 28%),
                linear-gradient(135deg, #ffffff 0%, #f9fbf7 100%);
            border: 1px solid rgba(15,122,58,.16);
            border-radius: 30px;
            padding: 1.55rem;
            box-shadow: 0 22px 58px rgba(16,33,24,.08);
        }

        .current-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
        }

        .current-location {
            color: #102118;
            font-size: 2.15rem;
            font-weight: 950;
            letter-spacing: -.06em;
            margin: 0;
        }

        .current-subtitle {
            color: #526158;
            font-size: .92rem;
            margin-top: .3rem;
            line-height: 1.45;
        }

        .current-temp {
            color: #0f7a3a;
            font-size: 3.65rem;
            font-weight: 950;
            letter-spacing: -.075em;
            line-height: .9;
            white-space: nowrap;
        }

        .condition-chip {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            color: #0b6b35;
            background: #e8f7ee;
            border: 1px solid #cae8d3;
            border-radius: 999px;
            padding: .45rem .75rem;
            font-size: .84rem;
            font-weight: 900;
            margin-top: .8rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .85rem;
            margin-top: 1.15rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 20px;
            background: #ffffff;
            border: 1px solid #dce8df;
            box-shadow: 0 14px 34px rgba(16,33,24,.055);
        }

        .metric-label {
            color: #647067;
            font-size: .7rem;
            font-weight: 950;
            letter-spacing: .085em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }

        .metric-value {
            color: #102118;
            font-size: 1.38rem;
            font-weight: 950;
            letter-spacing: -.04em;
        }

        .metric-subtitle {
            color: #647067;
            font-size: .78rem;
            line-height: 1.45;
            margin-top: .25rem;
        }

        .hourly-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: .85rem;
        }

        .daily-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: .85rem;
        }

        .hour-card,
        .daily-card {
            padding: 1.05rem .85rem;
            border-radius: 24px;
            background: rgba(255,255,255,.98);
            border: 1px solid #dce8df;
            box-shadow: 0 16px 36px rgba(16,33,24,.065);
            text-align: center;
            transition: transform .15s ease, box-shadow .15s ease;
        }

        .hour-card:hover,
        .daily-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 44px rgba(16,33,24,.09);
        }

        .hour-time,
        .daily-date {
            color: #526158;
            font-size: .72rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: .075em;
            min-height: 1rem;
        }

        .hour-icon,
        .daily-icon {
            font-size: 2.1rem;
            margin: .55rem 0 .35rem 0;
        }

        .hour-temp,
        .daily-temp {
            color: #102118;
            font-size: 1.16rem;
            font-weight: 950;
            letter-spacing: -.03em;
        }

        .hour-meta,
        .daily-rain {
            color: #647067;
            font-size: .8rem;
            margin-top: .35rem;
            line-height: 1.45;
        }

        .map-title {
            color: #102118;
            font-size: 1.28rem;
            font-weight: 950;
            letter-spacing: -.035em;
            margin: 0;
        }

        .map-subtitle {
            color: #647067;
            font-size: .88rem;
            line-height: 1.55;
            margin: .35rem 0 .9rem 0;
        }

        .api-note {
            margin-top: .85rem;
            padding: .9rem 1rem;
            border-radius: 18px;
            background: #e8f7ee;
            border: 1px solid #cae8d3;
            color: #0f7a3a;
            font-weight: 750;
            line-height: 1.55;
            font-size: .86rem;
        }

        .warning-note {
            margin-top: .75rem;
            padding: .9rem 1rem;
            border-radius: 16px;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            color: #be123c;
            font-weight: 750;
            line-height: 1.55;
            font-size: .86rem;
        }

        [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
        }

        [data-testid="stTextInput"] label {
            display: none !important;
        }

        [data-testid="stTextInput"] div[data-baseweb="input"] {
            background: #ffffff !important;
            border: 1.5px solid #b9d2c0 !important;
            border-radius: 15px !important;
            min-height: 50px !important;
            overflow: hidden !important;
            box-shadow: 0 10px 26px rgba(16,33,24,.045) !important;
        }

        [data-testid="stTextInput"] div[data-baseweb="base-input"] {
            display: flex !important;
            align-items: center !important;
            height: 50px !important;
        }

        [data-testid="stTextInput"] input {
            height: 50px !important;
            line-height: 50px !important;
            padding: 0 1rem !important;
            color: #102118 !important;
            -webkit-text-fill-color: #102118 !important;
            background: #ffffff !important;
            font-weight: 650 !important;
            caret-color: #0f7a3a !important;
        }

        [data-testid="stTextInput"] input::placeholder {
            color: #7d8d83 !important;
            opacity: 1 !important;
        }

        [data-testid="stFormSubmitButton"] {
            margin-top: 0 !important;
        }

        [data-testid="stFormSubmitButton"] button,
        .stButton button,
        [data-testid="stLinkButton"] a {
            min-height: 50px !important;
            border-radius: 15px !important;
            border: 1px solid #0b6b35 !important;
            color: #ffffff !important;
            font-weight: 950 !important;
            background: linear-gradient(135deg, #0b6b35, #19a957) !important;
            box-shadow: 0 14px 30px rgba(15,122,58,.22) !important;
        }

        @media (max-width: 1200px) {
            .hourly-grid,
            .daily-grid {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
        }

        @media (max-width: 700px) {
            .metric-grid,
            .hourly-grid,
            .daily-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .current-row {
                flex-direction: column;
            }

            .top-shell {
                flex-direction: column;
                align-items: flex-start;
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
) -> dict:
    def _request() -> dict:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    return await asyncio.to_thread(_request)


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


async def _geocode_city_open_meteo(city: str) -> dict | None:
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

    fallback_city = city.title()

    return {
        "city": english_display_name(result.get("name"), fallback=fallback_city),
        "admin1": english_display_name(result.get("admin1"), fallback=""),
        "country": english_display_name(result.get("country"), fallback="Pakistan"),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "timezone": result.get("timezone", WEATHER_TIMEZONE),
    }


async def _geocode_city_nominatim(city: str) -> dict | None:
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
        headers={
            "User-Agent": "EchoSafe-FYP-WeatherDashboard/1.0",
        },
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
        "timezone": WEATHER_TIMEZONE,
    }


async def _geocode_city_async(city: str) -> dict:
    city = (city or DEFAULT_WEATHER_CITY).strip()

    location = await _geocode_city_open_meteo(city)

    if location:
        return location

    location = await _geocode_city_nominatim(city)

    if location:
        return location

    raise LiveWeatherError(f"No Pakistan city or location found for: {city}")


async def _fetch_weather_async(latitude: float, longitude: float) -> dict:
    payload = await _get_json_async(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                ]
            ),
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation_probability",
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
                    "wind_speed_10m_max",
                ]
            ),
            "timezone": WEATHER_TIMEZONE,
            "forecast_days": 7,
        },
        timeout=20,
    )

    if "current" not in payload:
        raise LiveWeatherError("Weather response did not include current weather.")

    return payload


async def _fetch_live_weather_bundle_async(city: str) -> dict:
    location = await _geocode_city_async(city)
    weather = await _fetch_weather_async(location["latitude"], location["longitude"])

    return {
        "location": location,
        "weather": weather,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_weather_bundle(city: str) -> dict:
    return _run_async(_fetch_live_weather_bundle_async(city))


def normalize_current_weather(bundle: dict) -> dict:
    location = bundle["location"]
    current = bundle["weather"].get("current", {}) or {}
    condition, icon = weather_label(current.get("weather_code"))

    return {
        "city": english_display_name(location.get("city"), fallback=DEFAULT_WEATHER_CITY),
        "admin1": english_display_name(location.get("admin1"), fallback=""),
        "country": english_display_name(location.get("country"), fallback="Pakistan"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "updated_at": current.get("time", bundle.get("fetched_at", "")),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "apparent_temperature": current.get("apparent_temperature"),
        "precipitation": current.get("precipitation"),
        "weather_code": current.get("weather_code"),
        "condition": condition,
        "icon": icon,
        "wind_speed": current.get("wind_speed_10m"),
        "wind_gusts": current.get("wind_gusts_10m"),
    }


def normalize_hourly_forecast(bundle: dict) -> pd.DataFrame:
    df = pd.DataFrame(bundle["weather"].get("hourly", {}) or {})

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["date"] = df["time"].dt.date
    df["display_time"] = df["time"].dt.strftime("%I %p").str.lstrip("0")

    labels = df["weather_code"].apply(weather_label)
    df["condition"] = labels.apply(lambda item: item[0])
    df["icon"] = labels.apply(lambda item: item[1])

    return df


def normalize_daily_forecast(bundle: dict) -> pd.DataFrame:
    df = pd.DataFrame(bundle["weather"].get("daily", {}) or {})

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["date_label"] = df["time"].dt.strftime("%a, %d %b")

    labels = df["weather_code"].apply(weather_label)
    df["condition"] = labels.apply(lambda item: item[0])
    df["icon"] = labels.apply(lambda item: item[1])

    return df


def hero(title: str, subtitle: str, eyebrow: str = "Live Dashboard") -> None:
    inject_home_css()
    h(
        f"""
        <div class="home-hero">
            <div class="hero-inner">
                <div class="hero-pill">
                    <span class="pulse-dot"></span>
                    {esc(eyebrow)}
                </div>
                <h1 class="hero-title">{esc(title)}</h1>
                <p class="hero-subtitle">{esc(subtitle)}</p>
            </div>
        </div>
        """
    )


def badge(label: object, color: str | None = None) -> str:
    label = risk_level(label)
    color = color or risk_to_color(label)

    return (
        f"<span style='display:inline-flex;align-items:center;gap:.35rem;"
        f"padding:.28rem .65rem;border-radius:999px;"
        f"background:{color}22;border:1px solid {color}66;color:{color};"
        f"font-size:.78rem;font-weight:900;letter-spacing:.02em;'>"
        f"{esc(label)}</span>"
    )


def section_title(title: str, subtitle: str | None = None) -> None:
    h(
        f"""
        <div class="section-title">{esc(title)}</div>
        <p class="section-subtitle">{esc(subtitle or "")}</p>
        """
    )


def setup_error(title: str, message: str, details: str | None = None) -> None:
    h(
        f"""
        <div class="warning-note">
            <b>{esc(title)}</b><br>
            {esc(message)}
        </div>
        """
    )
    if details:
        st.code(details)


def require_file(path: str | Path, label: str = "required file") -> Path:
    path = Path(path)

    if not path.exists():
        setup_error(
            f"Missing {label}",
            f"The required file does not exist: {path}",
            f"Create or copy the file here:\n{path}",
        )
        st.stop()

    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str = "dataset") -> None:
    missing = [col for col in columns if col not in df.columns]

    if missing:
        setup_error(
            f"Invalid {dataset_name}",
            "The dataset is missing required columns.",
            "Missing columns:\n" + "\n".join(missing),
        )
        st.stop()


def metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    icon_or_color: str | None = None,
    risk: str | None = None,
) -> None:
    inject_home_css()
    color = risk_to_color(risk) if risk is not None else (icon_or_color or "#0f7a3a")
    h(
        f"""
        <div class="metric-card" style="border-top:4px solid {color};">
            <div class="metric-label">{esc(title)}</div>
            <div class="metric-value">{esc(value)}</div>
            <div class="metric-subtitle">{esc(subtitle)}</div>
        </div>
        """
    )


def render_result_card(result: dict, title: str = "Prediction Result") -> None:
    level = risk_level(result.get("risk_level") or result.get("severity_label"))
    color = risk_to_color(level)
    confidence = result.get("confidence")
    risk_score = result.get("risk_score")
    message = result.get("message", "")

    meta_items = []

    if confidence is not None:
        try:
            meta_items.append(f"Confidence: {float(confidence):.2%}")
        except Exception:
            meta_items.append(f"Confidence: {confidence}")

    if risk_score is not None:
        try:
            meta_items.append(f"Risk Score: {float(risk_score):.1f}")
        except Exception:
            meta_items.append(f"Risk Score: {risk_score}")

    h(
        f"""
        <div class="current-card" style="border-left:5px solid {color};">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
                <h3 style="margin:0;color:#102118;">{esc(title)}</h3>
                {badge(level, color)}
            </div>
            <h2 style="margin:.8rem 0 .35rem 0;color:{color};font-size:2rem;">{esc(level)}</h2>
            <p style="margin:.2rem 0;color:#526158;font-weight:800;">{esc(" | ".join(meta_items))}</p>
            <p style="margin:.65rem 0 0 0;color:#526158;line-height:1.6;">{esc(message)}</p>
        </div>
        """
    )


def render_map(
    df: pd.DataFrame,
    title: str | None = None,
    tooltip_fields: list[str] | None = None,
    tooltip_cols: list[str] | None = None,
    height: int = 430,
) -> None:
    inject_home_css()

    if title:
        st.caption(title)

    if df.empty:
        st.info("No map data available.")
        return

    map_df = df.copy()

    if "latitude" not in map_df.columns or "longitude" not in map_df.columns:
        setup_error("Map columns missing", "Map rendering requires latitude and longitude columns.")
        return

    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
    map_df = map_df.dropna(subset=["latitude", "longitude"])

    if map_df.empty:
        setup_error("No valid map points", "No rows contain valid latitude and longitude values.")
        return

    st.map(
        map_df.rename(columns={"latitude": "lat", "longitude": "lon"}),
        height=height,
    )


@st.cache_data(show_spinner=False)
def load_regions() -> pd.DataFrame:
    if ui is not None and hasattr(ui, "load_regions"):
        try:
            return ui.load_regions()
        except Exception:
            pass

    path = SETTINGS.project_root / "data" / "reference" / "master_regions.csv"
    require_file(path, "master regions file")
    df = pd.read_csv(path)
    require_columns(df, ["region", "province", "latitude", "longitude"], "master_regions.csv")
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if ui is not None and hasattr(ui, "load_predictions"):
        try:
            return ui.load_predictions()
        except Exception:
            pass

    path = SETTINGS.project_root / "predictions" / "batch" / "latest_disaster_predictions.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    if ui is not None and hasattr(ui, "load_alerts"):
        try:
            return ui.load_alerts()
        except Exception:
            pass

    path = SETTINGS.project_root / "predictions" / "alerts" / "latest_alerts.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_latest_weather_like(selected_region: str) -> pd.DataFrame:
    pred_df = load_predictions()

    if pred_df.empty or "region" not in pred_df.columns:
        return pd.DataFrame()

    region_df = pred_df[pred_df["region"].astype(str) == str(selected_region)].copy()

    if region_df.empty:
        return pd.DataFrame()

    return region_df.head(1)


def merge_regions_with_predictions() -> pd.DataFrame:
    regions = load_regions().copy()
    predictions = load_predictions()

    if predictions.empty or "region" not in predictions.columns:
        regions["risk_level"] = "Pending"
        regions["disaster_type"] = "Pending prediction generation"
        regions["confidence"] = None
        regions["risk_score"] = None
        return regions

    work = predictions.copy()

    if "risk_level" in work.columns:
        work["risk_level"] = work["risk_level"].apply(risk_level)
    else:
        work["risk_level"] = "Unknown"

    if "confidence" not in work.columns:
        work["confidence"] = None

    if "risk_score" not in work.columns:
        work["risk_score"] = 0

    risk_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
    work["_rank"] = work["risk_level"].map(risk_order).fillna(0)

    latest = (
        work.sort_values(["region", "_rank"], ascending=[True, False])
        .drop_duplicates("region")
        .drop(columns=["_rank"], errors="ignore")
    )

    merged = regions.merge(
        latest,
        on="region",
        how="left",
        suffixes=("", "_prediction"),
    )

    merged["risk_level"] = merged["risk_level"].fillna("Pending")
    merged["disaster_type"] = merged.get("disaster_type", "Pending prediction generation")
    merged["confidence"] = merged.get("confidence", None)
    merged["risk_score"] = merged.get("risk_score", 0).fillna(0)

    return merged


def prediction_status_note() -> None:
    pred_path = SETTINGS.project_root / "predictions" / "batch" / "latest_disaster_predictions.csv"

    if not pred_path.exists():
        h(
            """
            <div class="api-note">
                Prediction output is not generated yet. Run the batch prediction pipeline after model artifacts are ready:
                <br><code>python pipelines/run_batch_predictions.py</code>
            </div>
            """
        )


def render_risk_legend() -> None:
    items = "".join(
        [
            f"""
            <span style="display:inline-flex;align-items:center;gap:.35rem;padding:.32rem .55rem;border-radius:999px;background:#fff;border:1px solid #dce8df;color:#526158;font-size:.76rem;font-weight:850;">
                <span style="width:.62rem;height:.62rem;border-radius:999px;background:{color};"></span>
                {esc(label)}
            </span>
            """
            for label, color in RISK_COLORS.items()
            if label != "Unknown"
        ]
    )

    h(f"<div style='display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.65rem;'>{items}</div>")


def top_region_risk_summary() -> pd.DataFrame:
    pred_df = load_predictions()

    if pred_df.empty:
        return pd.DataFrame()

    work = pred_df.copy()
    work["risk_level"] = work.get("risk_level", "Unknown")
    work["risk_level"] = work["risk_level"].apply(risk_level)

    order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
    work["_rank"] = work["risk_level"].map(order).fillna(0)

    if "risk_score" not in work.columns:
        work["risk_score"] = 0

    return work.sort_values(["_rank", "risk_score"], ascending=False).drop(columns=["_rank"])


def style_dataframe(df: pd.DataFrame):
    return df


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
                        <div class="top-title">EchoSafe Weather Home</div>
                        <div class="top-subtitle">Live Weather Intelligence</div>
                    </div>
                </div>
                <div class="top-subtitle">Dashboard Home</div>
            </div>
            """
        )

    with top_right:
        if st.button("Logout", use_container_width=True, key="home_logout"):
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
            if st.button(display_label, use_container_width=True, key=f"home_nav_{page_key}"):
                go_to_page(page_key)


def render_search_panel() -> None:
    h(
        """
        <div class="search-panel">
            <div class="search-heading">Search Location</div>
            <div class="search-subcopy">Enter a Pakistan city or location to update the live weather forecast.</div>
        </div>
        """
    )

    with st.form("weather_city_search_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([0.58, 0.17, 0.25], gap="medium")

        with c1:
            city_input = st.text_input(
                "City",
                value=st.session_state["home_weather_city"],
                placeholder="Search for a city or location",
                label_visibility="collapsed",
            )

        with c2:
            search_submitted = st.form_submit_button("Search", use_container_width=True)

        with c3:
            st.empty()

    if search_submitted:
        st.session_state["home_weather_city"] = city_input.strip() or DEFAULT_WEATHER_CITY
        st.rerun()


def render_current_weather(current_weather: dict) -> None:
    location_line = " • ".join(
        [
            str(current_weather.get("admin1") or "").strip(),
            str(current_weather.get("country") or "").strip(),
        ]
    ).strip(" •")

    temperature = current_weather.get("temperature")
    humidity = current_weather.get("humidity")
    apparent = current_weather.get("apparent_temperature")
    precipitation = current_weather.get("precipitation")
    wind = current_weather.get("wind_speed")
    gusts = current_weather.get("wind_gusts")

    temperature_text = f"{float(temperature):.0f}°C" if temperature is not None else "—"
    humidity_text = f"{float(humidity):.0f}%" if humidity is not None else "—"
    apparent_text = f"{float(apparent):.0f}°C" if apparent is not None else "—"
    precipitation_text = f"{float(precipitation):.1f} mm" if precipitation is not None else "—"
    wind_text = f"{float(wind):.1f} km/h" if wind is not None else "—"
    gusts_text = f"{float(gusts):.1f} km/h" if gusts is not None else "—"

    metric_cards = [
        (
            "Feels Like",
            apparent_text,
            "Apparent temperature",
        ),
        (
            "Humidity",
            humidity_text,
            "Relative humidity",
        ),
        (
            "Wind",
            wind_text,
            f"Gusts {gusts_text}",
        ),
        (
            "Precipitation",
            precipitation_text,
            "Current precipitation",
        ),
    ]

    metric_html = "".join(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{esc(label)}</div>"
        f"<div class='metric-value'>{esc(value)}</div>"
        f"<div class='metric-subtitle'>{esc(subtitle)}</div>"
        f"</div>"
        for label, value, subtitle in metric_cards
    )

    h(
        f"""
        <div class="current-card">
            <div class="current-row">
                <div>
                    <h2 class="current-location">{esc(current_weather["city"])}</h2>
                    <div class="current-subtitle">{esc(location_line)} • {esc(current_weather["updated_at"])}</div>
                    <div class="condition-chip">{esc(current_weather["icon"])} {esc(current_weather["condition"])}</div>
                </div>
                <div class="current-temp">{esc(temperature_text)}</div>
            </div>
            <div class="metric-grid">{metric_html}</div>
        </div>
        """
    )


def render_hourly_forecast(hourly_df: pd.DataFrame) -> None:
    if hourly_df.empty:
        st.info("No hourly forecast data returned from Open-Meteo.")
        return

    now_local = local_now_naive()
    upcoming = hourly_df[hourly_df["time"] >= now_local - pd.Timedelta(hours=1)].copy()

    if upcoming.empty:
        upcoming = hourly_df.head(24).copy()

    hourly_cards_df = upcoming.head(6)
    cards: list[str] = []

    for index, row in hourly_cards_df.iterrows():
        hour_time = "Now" if index == hourly_cards_df.index[0] else row.get("display_time", "—")
        hour_icon = row.get("icon", "🌦️")

        hour_temp = (
            f"{float(row['temperature_2m']):.0f}°C"
            if pd.notna(row.get("temperature_2m"))
            else "—"
        )

        hour_rain = (
            f"{float(row['precipitation_probability']):.0f}% rain"
            if pd.notna(row.get("precipitation_probability"))
            else "—"
        )

        hour_wind = (
            f"{float(row['wind_speed_10m']):.0f} km/h"
            if pd.notna(row.get("wind_speed_10m"))
            else "—"
        )

        cards.append(
            f"<div class='hour-card'>"
            f"<div class='hour-time'>{esc(hour_time)}</div>"
            f"<div class='hour-icon'>{esc(hour_icon)}</div>"
            f"<div class='hour-temp'>{esc(hour_temp)}</div>"
            f"<div class='hour-meta'>{esc(hour_rain)}<br>{esc(hour_wind)}</div>"
            f"</div>"
        )

    h(f"<div class='hourly-grid'>{''.join(cards)}</div>")


def render_daily_forecast(daily_df: pd.DataFrame) -> None:
    if daily_df.empty:
        st.info("No daily forecast data returned from Open-Meteo.")
        return

    cards: list[str] = []

    for _, row in daily_df.head(7).iterrows():
        day_label = row.get("date_label", "—")
        day_icon = row.get("icon", "🌦️")

        max_temp = (
            f"{float(row['temperature_2m_max']):.0f}°"
            if pd.notna(row.get("temperature_2m_max"))
            else "—"
        )

        min_temp = (
            f"{float(row['temperature_2m_min']):.0f}°"
            if pd.notna(row.get("temperature_2m_min"))
            else "—"
        )

        precipitation_sum = (
            f"{float(row['precipitation_sum']):.1f} mm"
            if pd.notna(row.get("precipitation_sum"))
            else "—"
        )

        wind_max = (
            f"{float(row['wind_speed_10m_max']):.0f} km/h"
            if pd.notna(row.get("wind_speed_10m_max"))
            else "—"
        )

        cards.append(
            f"<div class='daily-card'>"
            f"<div class='daily-date'>{esc(day_label)}</div>"
            f"<div class='daily-icon'>{esc(day_icon)}</div>"
            f"<div class='daily-temp'>{esc(max_temp)} / {esc(min_temp)}</div>"
            f"<div class='daily-rain'>{esc(precipitation_sum)} rain<br>{esc(wind_max)} wind</div>"
            f"</div>"
        )

    h(f"<div class='daily-grid'>{''.join(cards)}</div>")


def render_live_map(current_weather: dict) -> None:
    lat = float(current_weather["latitude"])
    lon = float(current_weather["longitude"])
    city = current_weather["city"]
    temp = current_weather.get("temperature")
    condition = current_weather.get("condition")
    icon = current_weather.get("icon")

    zoom_url = f"https://zoom.earth/maps/#satellite,wind,radar,overlays,labels;{lat:.4f},{lon:.4f};7z"

    h(
        """
        <p class="map-title">Weather Map</p>
        <div class="map-subtitle">
            Live map marker for the selected location, for satellite, wind, and radar overlays.
        </div>
        """
    )

    map_df = pd.DataFrame(
        [
            {
                "latitude": lat,
                "longitude": lon,
                "city": city,
                "temperature": safe_float(temp),
                "condition": condition,
                "icon": icon,
            }
        ]
    )

    if pdk is not None:
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius=45000,
            get_fill_color=[15, 122, 58, 220],
            pickable=True,
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_text="city",
            get_size=16,
            get_color=[16, 33, 24, 255],
            get_text_anchor="middle",
            get_alignment_baseline="bottom",
            get_pixel_offset=[0, -22],
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=7,
                    pitch=0,
                ),
                layers=[point_layer, text_layer],
                tooltip={
                    "html": "<b>{city}</b><br>{icon} {condition}<br>{temperature}°C",
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
            zoom=7,
            height=420,
        )

    st.link_button("Open Earth Live Map", zoom_url, use_container_width=True)

    h(
        """
        <div class="api-note">
            Live weather of selected location's geocoded coordinates. The map updates to the searched location and provides live satellite, wind, and radar layers.
        </div>
        """
    )


def render_home() -> None:
    inject_home_css()

    if "home_weather_city" not in st.session_state:
        st.session_state["home_weather_city"] = DEFAULT_WEATHER_CITY

    render_top_header()

    hero(
        "Weather Forecast",
        "Live Weather Homepage. Search a Pakistan city to update current weather, hourly forecast, 7-day forecast, wind, humidity, and precipitation.",
        "Live Weather Overview",
    )

    render_search_panel()

    with st.spinner(f"Fetching live weather for {st.session_state['home_weather_city']}..."):
        try:
            weather_bundle = fetch_live_weather_bundle(st.session_state["home_weather_city"])
            current_weather = normalize_current_weather(weather_bundle)
            hourly_forecast_df = normalize_hourly_forecast(weather_bundle)
            daily_forecast_df = normalize_daily_forecast(weather_bundle)
        except Exception as exc:
            st.error("Live weather data could not be fetched. Please check the city name or internet connection.")
            st.caption(str(exc))
            return

    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    with left_col:
        render_current_weather(current_weather)

        section_title(
            "Hourly Forecast",
            f"Live upcoming forecast for {current_weather['city']}.",
        )
        render_hourly_forecast(hourly_forecast_df)

        section_title(
            "7-Day Forecast",
            f"Live daily forecast for {current_weather['city']}.",
        )
        render_daily_forecast(daily_forecast_df)

    with right_col:
        render_live_map(current_weather)
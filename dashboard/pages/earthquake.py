"""Earthquake Monitoring Page - Professional EchoSafe UI.

Behavior:
- Same top navigation style as Home / Heatwave / Hailstorm.
- Uses local gold earthquake dataset.
- Province filter + Check Prediction button.
- Map shows earthquake epicentre severity signals inside Pakistan map area only.
- Table compares actual severity and model-predicted severity.
- Earthquake occurrence is NOT predicted.
- Prediction here means post-detection severity / magnitude assistance.
"""

from __future__ import annotations

import html as html_escape
import textwrap
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st

from backend.config.settings import SETTINGS
from dashboard.auth.session import logout
from backend.risk_engine.earthquake_magnitude_predictor import predict_earthquake_magnitude
from backend.risk_engine.earthquake_severity import classify_earthquake_severity

try:
    import pydeck as pdk
except Exception:
    pdk = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


WEATHER_TIMEZONE = "Asia/Karachi"
DATASET_PATH = (
    SETTINGS.project_root
    / "data"
    / "gold"
    / "earthquake_risk"
    / "earthquake_risk_dataset.csv"
)

PAKISTAN_CENTER_LAT = 30.3753
PAKISTAN_CENTER_LON = 69.3451
PAKISTAN_LAT_MIN = 23.0
PAKISTAN_LAT_MAX = 37.8
PAKISTAN_LON_MIN = 60.5
PAKISTAN_LON_MAX = 78.2

RISK_COLORS = {
    "None": "#64748b",
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
    "Unknown": "#94a3b8",
}

REQUIRED_COLUMNS = {
    "event_time",
    "region",
    "province",
    "latitude",
    "longitude",
    "magnitude",
    "depth_km",
}


class EarthquakeDataError(Exception):
    """Controlled earthquake page error."""


def h(markup: str) -> None:
    cleaned = textwrap.dedent(markup).strip()
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html_escape.escape(str(value if value is not None else ""))


def local_today() -> date:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(WEATHER_TIMEZONE)).date()
    return date.today()


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

    return [148, 163, 184, 190]


def severity_message(level: str) -> str:
    level = clean_risk_level(level)

    messages = {
        "None": "No meaningful earthquake severity signal is detected for this event.",
        "Low": "Low earthquake severity signal. The detected event is expected to have limited impact.",
        "Medium": "Medium earthquake severity signal. Review location, depth, and nearby exposure before response decisions.",
        "High": "High earthquake severity signal. Response teams should review the affected region and prepare for possible impact.",
        "Critical": "Critical earthquake severity signal. Immediate operational attention and response coordination may be required.",
        "Unknown": "Earthquake severity signal is unavailable for this event.",
    }

    return messages.get(level, messages["Unknown"])


def is_pakistan_map_point(latitude: Any, longitude: Any) -> bool:
    lat = safe_float(latitude, default=999)
    lon = safe_float(longitude, default=999)

    return (
        PAKISTAN_LAT_MIN <= lat <= PAKISTAN_LAT_MAX
        and PAKISTAN_LON_MIN <= lon <= PAKISTAN_LON_MAX
    )


def filter_pakistan_map_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    working["latitude"] = pd.to_numeric(working["latitude"], errors="coerce")
    working["longitude"] = pd.to_numeric(working["longitude"], errors="coerce")

    working = working.dropna(subset=["latitude", "longitude"]).copy()

    pakistan_mask = (
        working["latitude"].between(PAKISTAN_LAT_MIN, PAKISTAN_LAT_MAX)
        & working["longitude"].between(PAKISTAN_LON_MIN, PAKISTAN_LON_MAX)
    )

    working = working[pakistan_mask].copy()

    return working


def classify_actual_severity(magnitude: float, depth_km: float, region: str) -> str:
    try:
        result = classify_earthquake_severity(
            {
                "magnitude": magnitude,
                "depth_km": depth_km,
                "region": region,
            }
        )

        return clean_risk_level(
            result.get("risk_level")
            or result.get("severity_label")
            or result.get("label")
            or result.get("prediction")
        )
    except Exception:
        if magnitude >= 7.0:
            return "Critical"
        if magnitude >= 6.0:
            return "High"
        if magnitude >= 4.5:
            return "Medium"
        if magnitude > 0:
            return "Low"
        return "Unknown"


def predict_severity_from_event(row: pd.Series) -> dict:
    latitude = safe_float(row.get("latitude"))
    longitude = safe_float(row.get("longitude"))
    depth_km = safe_float(row.get("depth_km"))
    actual_magnitude = safe_float(row.get("magnitude"))
    region = str(row.get("region") or row.get("place") or "Detected Event")

    try:
        prediction = predict_earthquake_magnitude(
            {
                "latitude": latitude,
                "longitude": longitude,
                "depth_km": depth_km,
                "region": region,
            }
        )

        predicted_magnitude = safe_float(
            prediction.get("estimated_magnitude"),
            default=actual_magnitude,
        )

        predicted_level = clean_risk_level(
            prediction.get("severity_label")
            or prediction.get("risk_level")
        )

        if predicted_level == "Unknown":
            predicted_level = classify_actual_severity(
                magnitude=predicted_magnitude,
                depth_km=depth_km,
                region=region,
            )

        model_name = str(prediction.get("model_name") or "Magnitude estimator")

        return {
            "predicted_magnitude": predicted_magnitude,
            "predicted_severity": predicted_level,
            "model_name": model_name,
        }

    except Exception:
        predicted_level = classify_actual_severity(
            magnitude=actual_magnitude,
            depth_km=depth_km,
            region=region,
        )

        return {
            "predicted_magnitude": actual_magnitude,
            "predicted_severity": predicted_level,
            "model_name": "Severity classifier",
        }


def inject_earthquake_css() -> None:
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
            background: #6366f1;
            box-shadow: 0 0 0 6px rgba(99,102,241,.14);
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

        .earthquake-hero {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.6rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 88% 0%, rgba(255,255,255,.20), transparent 30%),
                linear-gradient(135deg, #0f7a3a 0%, #159447 42%, #4f46e5 100%);
            border: 1px solid rgba(15,122,58,.16);
            box-shadow: 0 24px 70px rgba(15,122,58,.18);
            margin-bottom: 1.2rem;
        }

        .earthquake-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
            background-size: 42px 42px;
            animation: quakeGrid 18s linear infinite;
            mask-image: radial-gradient(circle at 45% 40%, black, transparent 78%);
        }

        @keyframes quakeGrid {
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

        .quake-field-label {
            color: #102118;
            font-weight: 900;
            font-size: .88rem;
            margin-bottom: .38rem;
            line-height: 1.2;
        }

        .quake-button-spacer {
            height: 1.62rem;
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
                radial-gradient(circle at 88% 9%, rgba(99,102,241,.10), transparent 30%),
                linear-gradient(135deg, #ffffff 0%, #f9fbf7 100%);
            border: 1px solid rgba(15,122,58,.16);
            border-radius: 28px;
            padding: 1.2rem;
            box-shadow: 0 22px 58px rgba(16,33,24,.08);
        }

        .prediction-card {
            background:
                radial-gradient(circle at 90% 5%, rgba(99,102,241,.16), transparent 32%),
                linear-gradient(135deg, #ffffff 0%, #eef2ff 100%);
            border: 1px solid rgba(99,102,241,.26);
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

        .notice-card {
            margin-top: .85rem;
            padding: .9rem 1rem;
            border-radius: 18px;
            background: #eef2ff;
            border: 1px solid rgba(99,102,241,.25);
            color: #3730a3;
            font-weight: 750;
            line-height: 1.55;
            font-size: .86rem;
        }

        [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
        }

        [data-testid="stSelectbox"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        [data-testid="stSelectbox"] label {
            display: none !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border: 1.5px solid #b9d2c0 !important;
            border-radius: 15px !important;
            min-height: 50px !important;
            height: 50px !important;
            box-shadow: 0 10px 26px rgba(16,33,24,.045) !important;
            overflow: hidden !important;
            display: flex !important;
            align-items: center !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #102118 !important;
            min-height: 50px !important;
            display: flex !important;
            align-items: center !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stSelectbox"] div[data-baseweb="select"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] svg,
        [data-testid="stSelectbox"] div[data-baseweb="select"] * {
            color: #102118 !important;
            -webkit-text-fill-color: #102118 !important;
            fill: #102118 !important;
            font-weight: 800 !important;
            font-size: .95rem !important;
            background-color: transparent !important;
        }

        div[data-baseweb="popover"] {
            background: #ffffff !important;
            border-radius: 14px !important;
            border: 1px solid #b9d2c0 !important;
            box-shadow: 0 18px 50px rgba(16,33,24,.14) !important;
            overflow: hidden !important;
        }

        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] div[role="listbox"] {
            background: #ffffff !important;
            color: #102118 !important;
        }

        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] div[role="option"],
        div[data-baseweb="popover"] div,
        div[data-baseweb="popover"] span {
            color: #102118 !important;
            -webkit-text-fill-color: #102118 !important;
            font-weight: 800 !important;
            background: #ffffff !important;
        }

        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] div[role="option"]:hover,
        div[data-baseweb="popover"] li[aria-selected="true"],
        div[data-baseweb="popover"] div[aria-selected="true"] {
            background: #e8f7ee !important;
            color: #0b6b35 !important;
            -webkit-text-fill-color: #0b6b35 !important;
        }

        [data-testid="stFormSubmitButton"] {
            margin-top: 0 !important;
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

            .quake-button-spacer {
                height: .5rem;
            }
        }
        </style>
        """
    )


@st.cache_data(show_spinner=False)
def load_earthquake_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise EarthquakeDataError(f"Earthquake dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing = REQUIRED_COLUMNS.difference(set(df.columns))

    if missing:
        raise EarthquakeDataError(
            f"earthquake_risk_dataset.csv is missing columns: {', '.join(sorted(missing))}"
        )

    if "place" not in df.columns:
        df["place"] = df["region"]

    if "earthquake_severity_label" not in df.columns:
        df["earthquake_severity_label"] = "Unknown"

    for col in [
        "latitude",
        "longitude",
        "magnitude",
        "depth_km",
        "distance_to_region_km",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["event_time_parsed"] = pd.to_datetime(
        df["event_time"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=[
            "event_time_parsed",
            "latitude",
            "longitude",
            "magnitude",
            "depth_km",
        ]
    ).copy()

    if df.empty:
        raise EarthquakeDataError(
            "Earthquake dataset is available, but no valid rows were found after cleaning."
        )

    df["event_date"] = df["event_time_parsed"].dt.date
    df["event_time_display"] = df["event_time_parsed"].dt.strftime("%Y-%m-%d %H:%M UTC")

    df["province"] = df["province"].fillna("Unknown").astype(str)
    df["region"] = df["region"].fillna("Unknown").astype(str)
    df["place"] = df["place"].fillna(df["region"]).astype(str)

    return df.sort_values("event_time_parsed", ascending=False).reset_index(drop=True)


def prepare_prediction_frame(df: pd.DataFrame, max_rows: int = 160) -> pd.DataFrame:
    working = df.sort_values("event_time_parsed", ascending=False).head(max_rows).copy()
    rows: list[dict] = []

    for _, row in working.iterrows():
        magnitude = safe_float(row.get("magnitude"))
        depth_km = safe_float(row.get("depth_km"))
        region = str(row.get("region") or row.get("place") or "Detected Event")

        actual_severity_raw = row.get("earthquake_severity_label")
        actual_severity = clean_risk_level(actual_severity_raw)

        if actual_severity == "Unknown":
            actual_severity = classify_actual_severity(
                magnitude=magnitude,
                depth_km=depth_km,
                region=region,
            )

        predicted = predict_severity_from_event(row)
        predicted_severity = clean_risk_level(predicted["predicted_severity"])
        predicted_magnitude = safe_float(predicted["predicted_magnitude"], magnitude)

        item = row.to_dict()
        item["actual_severity"] = actual_severity
        item["predicted_severity"] = predicted_severity
        item["predicted_magnitude"] = predicted_magnitude
        item["prediction_model"] = predicted["model_name"]
        item["risk_level"] = predicted_severity
        item["color"] = risk_rgba(predicted_severity)
        item["radius"] = max(22000, min(90000, magnitude * 11500))
        item["label"] = f"{region} {predicted_severity}"

        rows.append(item)

    return pd.DataFrame(rows)


def summarize_prediction(prediction_df: pd.DataFrame, province_label: str) -> dict:
    if prediction_df.empty:
        return {
            "province_label": province_label,
            "display_date": local_today(),
            "risk_level": "Unknown",
            "message": severity_message("Unknown"),
            "event_count": 0,
            "strongest_magnitude": 0,
            "max_depth": 0,
            "shallow_count": 0,
            "latest_region": "No event",
        }

    latest = prediction_df.sort_values("event_time_parsed", ascending=False).iloc[0]
    strongest = prediction_df.sort_values("magnitude", ascending=False).iloc[0]
    deepest = prediction_df.sort_values("depth_km", ascending=False).iloc[0]

    if "is_shallow" in prediction_df.columns:
        shallow_count = int(
            pd.to_numeric(prediction_df["is_shallow"], errors="coerce")
            .fillna(0)
            .astype(int)
            .sum()
        )
    else:
        shallow_count = int((prediction_df["depth_km"] <= 70).sum())

    level = clean_risk_level(latest.get("predicted_severity"))

    return {
        "province_label": province_label,
        "display_date": latest.get("event_date", local_today()),
        "risk_level": level,
        "message": severity_message(level),
        "event_count": int(len(prediction_df)),
        "strongest_magnitude": safe_float(strongest.get("magnitude")),
        "max_depth": safe_float(deepest.get("depth_km")),
        "shallow_count": shallow_count,
        "latest_region": str(latest.get("region") or latest.get("place") or "Detected Event"),
    }


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
                        <div class="top-title">EchoSafe Earthquake Monitoring</div>
                        <div class="top-subtitle">Detected event severity monitoring and prediction comparison</div>
                    </div>
                </div>
                <div class="top-subtitle">Earthquake Monitoring</div>
            </div>
            """
        )

    with top_right:
        if st.button("Logout", use_container_width=True, key="earthquake_logout"):
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
            if st.button(display_label, use_container_width=True, key=f"earthquake_nav_{page_key}"):
                go_to_page(page_key)


def render_hero() -> None:
    h(
        """
        <div class="earthquake-hero">
            <div class="hero-inner">
                <div class="hero-pill">
                    <span class="pulse-dot"></span>
                    Earthquake Operations
                </div>
                <h1 class="hero-title">Earthquake Severity Monitoring</h1>
                <p class="hero-subtitle">
                    Review detected seismic events by province, inspect epicentre locations on the map, and compare actual severity with post-detection predicted severity.
                </p>
            </div>
        </div>
        """
    )


def render_search_panel(provinces: list[str]) -> None:
    h(
        """
        <div class="search-panel">
            <div class="search-heading">Recent Earthquake Severity Prediction</div>
            <div class="search-subcopy">
                Select a province and check detected earthquake events with magnitude, depth, latitude, longitude, actual severity, and predicted severity.
            </div>
        </div>
        """
    )

    if "earthquake_province" not in st.session_state:
        st.session_state["earthquake_province"] = "All Provinces"

    with st.form("earthquake_prediction_form", clear_on_submit=False):
        c1, c2 = st.columns([0.72, 0.22], gap="large")

        with c1:
            h("<div class='quake-field-label'>Province</div>")
            province_input = st.selectbox(
                "Province",
                options=provinces,
                index=provinces.index(st.session_state["earthquake_province"])
                if st.session_state["earthquake_province"] in provinces
                else 0,
                label_visibility="collapsed",
            )

        with c2:
            h("<div class='quake-button-spacer'></div>")
            submitted = st.form_submit_button("Check Prediction", use_container_width=True)

    if submitted:
        st.session_state["earthquake_province"] = province_input
        st.rerun()


def render_prediction_summary(summary: dict) -> None:
    level = clean_risk_level(summary["risk_level"])
    color = risk_color(level)

    mini_items = [
        ("Detected Events", f"{summary['event_count']}"),
        ("Strongest", f"M{summary['strongest_magnitude']:.1f}"),
        ("Max Depth", f"{summary['max_depth']:.1f} km"),
        ("Shallow Events", f"{summary['shallow_count']}"),
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
                <div class="card-eyebrow">Current Severity Signal</div>
                <h2 class="card-title">{esc(summary["province_label"])} Earthquake Severity</h2>
                <div class="card-value" style="color:{color};">{esc(level)}</div>
                <div class="risk-pill" style="background:{color}18;border:1px solid {color}66;color:{color};">
                    Latest Event Date: {esc(summary["display_date"])}
                </div>
                <div class="card-message">
                    {esc(summary["message"])}
                    Latest monitored region: <b>{esc(summary["latest_region"])}</b>.
                </div>
            </div>
            <div class="mini-metrics">{mini_html}</div>
        </div>
        """
    )


def render_earthquake_legend() -> None:
    h(
        """
        <div class="legend-row">
            <div class="legend-chip"><span class="legend-dot" style="background:#22c55e;"></span>Low</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#f59e0b;"></span>Medium</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#f97316;"></span>High</div>
            <div class="legend-chip"><span class="legend-dot" style="background:#ef4444;"></span>Critical</div>
        </div>
        """
    )


def render_earthquake_map(prediction_df: pd.DataFrame) -> None:
    h(
        """
        <div class="section-title">Earthquake Epicentre Severity Map</div>
        <p class="section-subtitle">
            Map markers show detected earthquake epicentres inside the Pakistan monitoring area. Marker color represents predicted severity after detection.
        </p>
        """
    )

    render_earthquake_legend()

    if prediction_df.empty:
        st.info("No earthquake map data available for the selected province.")
        return

    map_df = filter_pakistan_map_points(prediction_df)

    if map_df.empty:
        st.info("No valid Pakistan-area latitude / longitude values found for map rendering.")
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
            data=map_df.head(30),
            get_position="[longitude, latitude]",
            get_text="label",
            get_size=12,
            get_color=[16, 33, 24, 235],
            get_text_anchor="middle",
            get_alignment_baseline="bottom",
            get_pixel_offset=[0, -18],
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(
                    latitude=PAKISTAN_CENTER_LAT,
                    longitude=PAKISTAN_CENTER_LON,
                    zoom=4.75,
                    pitch=0,
                    bearing=0,
                ),
                layers=[point_layer, text_layer],
                tooltip={
                    "html": """
                    <b>{region}</b><br/>
                    Province: {province}<br/>
                    Magnitude: M{magnitude}<br/>
                    Depth: {depth_km} km<br/>
                    Actual: <b>{actual_severity}</b><br/>
                    Predicted: <b>{predicted_severity}</b>
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


def render_records_table(prediction_df: pd.DataFrame, province_label: str) -> None:
    h(
        f"""
        <div class="section-title">Compare Actual Severity and Predicted Severity</div>
        <p class="section-subtitle">{esc(province_label)} • Recent detected earthquake events</p>
        """
    )

    if prediction_df.empty:
        st.info("No earthquake records found for the selected province.")
        return

    table = prediction_df.sort_values("event_time_parsed", ascending=False).head(40).copy()
    rows = []

    for _, row in table.iterrows():
        actual_level = clean_risk_level(row.get("actual_severity"))
        predicted_level = clean_risk_level(row.get("predicted_severity"))

        actual_color = risk_color(actual_level)
        predicted_color = risk_color(predicted_level)

        event_date = row.get("event_date", "")
        province = row.get("province", "")
        region = row.get("region", "")
        depth = safe_float(row.get("depth_km"))
        lat = safe_float(row.get("latitude"))
        lon = safe_float(row.get("longitude"))
        magnitude = safe_float(row.get("magnitude"))
        predicted_magnitude = safe_float(row.get("predicted_magnitude"))

        rows.append(
            f"""
            <tr>
                <td>{esc(event_date)}</td>
                <td>{esc(province)}</td>
                <td>{esc(region)}</td>
                <td>M{magnitude:.1f}</td>
                <td>{depth:.1f} km</td>
                <td>{lat:.4f}</td>
                <td>{lon:.4f}</td>
                <td><span class="badge" style="color:{actual_color};background:{actual_color}16;">{esc(actual_level)}</span></td>
                <td>M{predicted_magnitude:.1f}</td>
                <td><span class="badge" style="color:{predicted_color};background:{predicted_color}16;">{esc(predicted_level)}</span></td>
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
                        <th>Province</th>
                        <th>Region</th>
                        <th>Actual Magnitude</th>
                        <th>Depth</th>
                        <th>Latitude</th>
                        <th>Longitude</th>
                        <th>Actual Severity</th>
                        <th>Predicted Magnitude</th>
                        <th>Predicted Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    )


def render_earthquake_page() -> None:
    inject_earthquake_css()

    render_top_header()
    render_hero()

    try:
        all_events = load_earthquake_dataset()
    except Exception as exc:
        st.error("Earthquake monitoring data could not be loaded.")
        st.caption(str(exc))
        return

    provinces = ["All Provinces"] + sorted(
        [str(x) for x in all_events["province"].dropna().unique().tolist()]
    )

    render_search_panel(provinces)

    selected_province = st.session_state.get("earthquake_province", "All Provinces")
    province_label = selected_province

    if selected_province == "All Provinces":
        filtered_events = all_events.copy()
    else:
        filtered_events = all_events[
            all_events["province"].astype(str) == str(selected_province)
        ].copy()

    if filtered_events.empty:
        st.warning("No earthquake records found for the selected province.")
        return

    with st.spinner(f"Checking earthquake severity prediction for {province_label}..."):
        try:
            prediction_df = prepare_prediction_frame(filtered_events, max_rows=160)
            summary = summarize_prediction(prediction_df, province_label)
        except Exception as exc:
            st.error("Earthquake severity prediction could not be completed.")
            st.caption(str(exc))
            return

    top_left, top_right = st.columns([0.95, 1.05], gap="large")

    with top_left:
        render_prediction_summary(summary)

        h(
            """
            <div class="notice-card">
                This dashboard monitors detected earthquake events and compares severity signals.
                It does not predict earthquake occurrence before an event happens.
            </div>
            """
        )

    with top_right:
        render_earthquake_map(prediction_df)

    render_records_table(
        prediction_df=prediction_df,
        province_label=province_label,
    )
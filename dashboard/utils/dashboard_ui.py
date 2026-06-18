"""Reusable UI utilities for the EchoSafe Streamlit dashboard.

This module keeps page files clean and consistent. It intentionally avoids
random demo coordinates or fake earthquake prediction. Weather pages can still
render before the batch prediction pipeline is run, but they clearly label the
state as a model scenario instead of pretending it is a generated batch output.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd
import streamlit as st

from backend.config.settings import SETTINGS

RISK_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#fb7185",
    "Critical": "#ef4444",
    "Unknown": "#64748b",
    "Pending": "#38bdf8",
}

RISK_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4, "Unknown": 0, "Pending": 0}

DISASTER_LABELS = {
    "Flood / Heavy Rainfall Risk": "Flood / Heavy Rainfall",
    "Heatwave Risk": "Heatwave",
    "Hailstorm Risk": "Hailstorm",
    "Earthquake Monitoring": "Earthquake",
}


# -----------------------------------------------------------------------------
# Global styling
# -----------------------------------------------------------------------------

def inject_global_css() -> None:
    """Apply the dashboard visual system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --bg0: #020617;
            --bg1: #08111f;
            --panel: rgba(15, 23, 42, 0.74);
            --panel2: rgba(30, 41, 59, 0.72);
            --line: rgba(148, 163, 184, 0.18);
            --text: #f8fafc;
            --muted: #94a3b8;
            --cyan: #38bdf8;
            --blue: #3b82f6;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .stApp {
            background:
                radial-gradient(circle at 15% 15%, rgba(56,189,248,.16), transparent 34%),
                radial-gradient(circle at 84% 12%, rgba(99,102,241,.15), transparent 30%),
                linear-gradient(120deg, #020617 0%, #07111f 48%, #020617 100%);
            color: var(--text);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(2, 6, 23, .96), rgba(8, 13, 27, .94));
            border-right: 1px solid rgba(148, 163, 184, .14);
        }
        [data-testid="stSidebarNav"] { display: none !important; }
        [data-testid="stToolbar"] { visibility: hidden; height: 0%; position: fixed; }
        [data-testid="stDecoration"] { display: none; }
        .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1480px; }

        .echo-hero {
            position: relative;
            padding: 1.45rem 1.6rem;
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 26px;
            background:
                linear-gradient(135deg, rgba(15,23,42,.90), rgba(30,41,59,.55)),
                radial-gradient(circle at top right, rgba(56,189,248,.22), transparent 35%);
            box-shadow: 0 18px 80px rgba(0,0,0,.34);
            overflow: hidden;
            margin-bottom: 1.05rem;
        }
        .echo-hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(56,189,248,.045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56,189,248,.045) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(90deg, black, transparent 92%);
        }
        .hero-inner { position: relative; z-index: 1; }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .34rem .68rem;
            border-radius: 999px;
            background: rgba(56,189,248,.12);
            border: 1px solid rgba(56,189,248,.22);
            color: #bae6fd;
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .pulse-dot {
            width: .55rem; height: .55rem; border-radius: 999px; background: #22c55e;
            box-shadow: 0 0 0 rgba(34,197,94,.62);
            animation: pulse 1.8s infinite;
        }
        @keyframes pulse { 0% {box-shadow:0 0 0 0 rgba(34,197,94,.55)} 70% {box-shadow:0 0 0 12px rgba(34,197,94,0)} 100% {box-shadow:0 0 0 0 rgba(34,197,94,0)} }
        .hero-title { margin: .82rem 0 .35rem; font-size: clamp(2rem, 4.2vw, 4.1rem); line-height: .95; font-weight: 900; letter-spacing: -.055em; }
        .hero-subtitle { color: #cbd5e1; max-width: 930px; font-size: 1.04rem; line-height: 1.75; margin: 0; }

        .glass-card {
            padding: 1.1rem;
            border-radius: 22px;
            background: rgba(15, 23, 42, .72);
            border: 1px solid rgba(148, 163, 184, .17);
            box-shadow: 0 14px 45px rgba(0,0,0,.22);
            backdrop-filter: blur(14px);
        }
        .metric-card {
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: linear-gradient(145deg, rgba(15,23,42,.88), rgba(30,41,59,.58));
            border: 1px solid rgba(148,163,184,.16);
            min-height: 118px;
            overflow: hidden;
            position: relative;
        }
        .metric-card:after {
            content:""; position:absolute; right:-20px; top:-36px; width:108px; height:108px; border-radius:999px;
            background: rgba(56,189,248,.09);
        }
        .metric-label { color: #94a3b8; font-weight: 700; font-size: .78rem; letter-spacing: .05em; text-transform: uppercase; }
        .metric-value { color: #f8fafc; font-weight: 900; font-size: 2rem; margin-top: .35rem; letter-spacing: -.04em; }
        .metric-help { color: #cbd5e1; font-size: .82rem; margin-top: .12rem; }

        .section-title { font-weight: 900; font-size: 1.35rem; letter-spacing: -.03em; margin: .5rem 0 .35rem; color: #f8fafc; }
        .section-subtitle { color: #94a3b8; margin: 0 0 1rem; }
        .soft-note {
            padding: .9rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(56,189,248,.22);
            background: rgba(14, 116, 144, .11);
            color: #dbeafe;
        }
        .warning-note {
            padding: .85rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(245,158,11,.26);
            background: rgba(245,158,11,.10);
            color: #fde68a;
        }
        .risk-pill {
            display:inline-flex; align-items:center; gap:.36rem; border-radius:999px; padding:.24rem .58rem; font-size:.78rem; font-weight:800;
            border:1px solid rgba(255,255,255,.16);
        }
        .timeline-card {
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,.16);
            background: rgba(15,23,42,.62);
            padding: .86rem;
            min-height: 120px;
        }
        .timeline-day { font-size:.75rem; color:#94a3b8; font-weight:800; letter-spacing:.04em; text-transform:uppercase; }
        .timeline-temp { font-size:1.65rem; font-weight:900; margin:.2rem 0; letter-spacing:-.05em; }
        .timeline-risk { font-size:.82rem; font-weight:800; }
        .sidebar-logo-card {
            padding: 1.05rem;
            border-radius: 24px;
            background: linear-gradient(145deg, rgba(15,23,42,.92), rgba(30,41,59,.52));
            border: 1px solid rgba(148,163,184,.18);
            margin-bottom: .9rem;
        }
        .sidebar-title { font-size: 1.35rem; font-weight: 900; letter-spacing:-.04em; margin:.22rem 0 .1rem; }
        .sidebar-sub { color:#94a3b8; font-size:.78rem; line-height:1.5; }
        div.stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,.16);
            background: rgba(15,23,42,.72);
            color: #f8fafc;
            font-weight: 800;
            transition: .18s ease;
        }
        div.stButton > button:hover {
            border-color: rgba(56,189,248,.55);
            transform: translateY(-1px);
            box-shadow: 0 10px 26px rgba(56,189,248,.10);
        }
        .active-nav button {
            background: linear-gradient(135deg, rgba(14,165,233,.92), rgba(37,99,235,.88)) !important;
            border-color: rgba(186,230,253,.45) !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: .45rem; }
        .stTabs [data-baseweb="tab"] {
            height: 42px; border-radius: 999px; padding: 0 18px; background: rgba(15,23,42,.68); border: 1px solid rgba(148,163,184,.14);
        }
        .stTabs [aria-selected="true"] { background: rgba(56,189,248,.16) !important; color: #e0f2fe !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def project_path(*parts: str) -> Path:
    return SETTINGS.project_root.joinpath(*parts)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.error(f"{dataset_name} is missing required columns: {', '.join(missing)}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_regions() -> pd.DataFrame:
    path = project_path("data", "reference", "master_regions.csv")
    if not path.exists():
        st.error(f"Missing master region file: {path}")
        st.stop()
    df = pd.read_csv(path)
    require_columns(df, ["region", "province", "latitude", "longitude"], "master_regions.csv")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    if df[["latitude", "longitude"]].isna().any().any():
        st.error("master_regions.csv contains invalid latitude/longitude values.")
        st.stop()
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return read_csv_if_exists(project_path("predictions", "batch", "latest_disaster_predictions.csv"))


@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    return read_csv_if_exists(project_path("predictions", "alerts", "latest_alerts.csv"))


@st.cache_data(show_spinner=False)
def load_earthquakes() -> pd.DataFrame:
    path = project_path("data", "gold", "earthquake_risk", "earthquake_risk_dataset.csv")
    df = read_csv_if_exists(path)
    if df.empty:
        return df
    cols = ["event_time", "region", "province", "latitude", "longitude", "magnitude", "depth_km", "place", "earthquake_severity_label"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Earthquake dataset is missing required columns: {', '.join(missing)}")
        st.stop()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["event_time", "magnitude", "depth_km", "latitude", "longitude"])
    return df.sort_values("event_time", ascending=False)


@st.cache_data(show_spinner=False)
def load_latest_weather_like(region: str | None = None) -> pd.DataFrame:
    """Try to assemble latest regional weather features from gold datasets.

    This is not fake. It only uses local gold CSVs if present. If the pipeline
    has not produced enough data yet, callers can display an empty-state note.
    """
    candidates = [
        project_path("data", "gold", "heatwave_risk", "heatwave_training_dataset_2010_2025.csv"),
        project_path("data", "gold", "flood_risk", "flood_training_dataset_2010_2025.csv"),
        project_path("data", "gold", "hailstorm_risk", "hailstorm_training_dataset_2010_2025.csv"),
    ]
    frames: list[pd.DataFrame] = []
    wanted = [
        "date", "year", "month", "region", "province", "temperature_mean_c", "temperature_max_c",
        "temperature_min_c", "humidity_mean_percent", "wind_speed_mean_kmh", "rainfall_mm",
        "precipitation_mm", "water_level_index"
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        keep = [c for c in wanted if c in df.columns]
        if "region" not in keep:
            continue
        part = df[keep].copy()
        if "date" in part.columns:
            part["_sort_date"] = pd.to_datetime(part["date"], errors="coerce")
        elif {"year", "month"}.issubset(part.columns):
            part["_sort_date"] = pd.to_datetime(part["year"].astype(str) + "-" + part["month"].astype(str) + "-01", errors="coerce")
        else:
            part["_sort_date"] = pd.NaT
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    if region:
        data = data[data["region"].astype(str).str.lower() == str(region).lower()]
    if data.empty:
        return data
    data = data.sort_values("_sort_date")
    return data.groupby("region", as_index=False).tail(1).sort_values("region")


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def hero(title: str, subtitle: str, eyebrow: str = "Pakistan Disaster Intelligence") -> None:
    st.markdown(
        f"""
        <div class="echo-hero">
          <div class="hero-inner">
            <div class="eyebrow"><span class="pulse-dot"></span>{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "", accent: str = "#38bdf8") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 3px solid {accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_pill(level: str) -> str:
    color = RISK_COLORS.get(str(level), RISK_COLORS["Unknown"])
    return f'<span class="risk-pill" style="background:{color}22;color:{color};">● {level}</span>'


def risk_to_color(level: str) -> str:
    return RISK_COLORS.get(str(level), RISK_COLORS["Unknown"])


def risk_to_rgb(level: str) -> list[int]:
    hex_color = risk_to_color(level).lstrip("#")
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)] + [190]


def style_dataframe(df: pd.DataFrame, risk_cols: list[str] | None = None):
    risk_cols = risk_cols or [c for c in df.columns if "risk" in c.lower() or "severity" in c.lower()]

    def _color(value):
        color = risk_to_color(str(value))
        return f"background-color: {color}22; color: {color}; font-weight: 800;"

    styler = df.style
    for col in risk_cols:
        if col in df.columns:
            styler = styler.map(_color, subset=[col])
    return styler


def clean_prediction_label(label: str) -> str:
    return DISASTER_LABELS.get(str(label), str(label))


def merge_regions_with_predictions(disaster_type: str | None = None) -> pd.DataFrame:
    regions = load_regions().copy()
    preds = load_predictions()
    if preds.empty:
        regions["risk_level"] = "Pending"
        regions["confidence"] = np.nan
        regions["disaster_type"] = disaster_type or "Pending"
        return regions
    if disaster_type:
        preds = preds[preds["disaster_type"].astype(str).str.contains(disaster_type, case=False, na=False)]
    if preds.empty:
        regions["risk_level"] = "Pending"
        regions["confidence"] = np.nan
        regions["disaster_type"] = disaster_type or "Pending"
        return regions
    preds = preds.copy()
    preds["_risk_rank"] = preds["risk_level"].map(RISK_ORDER).fillna(0)
    preds = preds.sort_values(["region", "_risk_rank", "confidence"], ascending=[True, False, False])
    latest = preds.groupby("region", as_index=False).head(1)
    merged = regions.merge(latest, on="region", how="left", suffixes=("", "_pred"))
    merged["risk_level"] = merged["risk_level"].fillna("Pending")
    return merged


def render_map(df: pd.DataFrame, *, lat: str = "latitude", lon: str = "longitude", risk_col: str = "risk_level", tooltip_cols: list[str] | None = None, zoom: float = 4.3, height: int = 470) -> None:
    if df.empty:
        st.markdown("<div class='warning-note'>No map records available yet.</div>", unsafe_allow_html=True)
        return
    try:
        import pydeck as pdk
    except Exception:
        st.error("Map package missing. Install with: pip install pydeck")
        return

    plot_df = df.copy()
    plot_df[lat] = pd.to_numeric(plot_df[lat], errors="coerce")
    plot_df[lon] = pd.to_numeric(plot_df[lon], errors="coerce")
    plot_df = plot_df.dropna(subset=[lat, lon])
    if plot_df.empty:
        st.error("Map cannot render because latitude/longitude values are invalid.")
        return
    plot_df["_color"] = plot_df[risk_col].fillna("Unknown").map(risk_to_rgb)
    plot_df["_radius"] = plot_df[risk_col].map(lambda x: 42000 + (RISK_ORDER.get(str(x), 0) * 16000))
    tooltip_cols = tooltip_cols or ["region", "province", risk_col]
    tooltip_html = "<br/>".join([f"<b>{c.replace('_',' ').title()}:</b> {{{c}}}" for c in tooltip_cols if c in plot_df.columns])

    center_lat = float(plot_df[lat].mean())
    center_lon = float(plot_df[lon].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position=f"[{lon}, {lat}]",
        get_fill_color="_color",
        get_line_color=[255, 255, 255, 120],
        get_radius="_radius",
        radius_min_pixels=8,
        radius_max_pixels=52,
        pickable=True,
        stroked=True,
        line_width_min_pixels=1,
    )
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=35),
        layers=[layer],
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "#020617", "color": "white", "border": "1px solid #334155", "borderRadius": "12px"}},
    )
    st.pydeck_chart(deck, use_container_width=True, height=height)


def render_risk_legend() -> None:
    parts = []
    for level in ["Low", "Medium", "High", "Critical", "Pending"]:
        parts.append(risk_pill(level))
    st.markdown(" ".join(parts), unsafe_allow_html=True)


def prediction_status_note() -> None:
    if load_predictions().empty:
        st.markdown(
            """
            <div class="warning-note">
              Batch predictions are not generated yet. The dashboard interface is ready, but run
              <b>python pipelines/run_batch_predictions.py</b> to populate regional risk outputs.
            </div>
            """,
            unsafe_allow_html=True,
        )


def scenario_7_day(base_temp: float, base_humidity: float, base_wind: float, hazard: str) -> pd.DataFrame:
    """Create a deterministic model-scenario outlook from the selected inputs.

    This is deliberately labelled as a scenario outlook in pages, not an official forecast.
    """
    rows = []
    today = datetime.now().date()
    for i in range(-3, 8):
        temp = base_temp + math.sin(i * 0.8) * 1.8 + (i * 0.25 if i > 0 else i * 0.1)
        humidity = min(100, max(5, base_humidity + math.cos(i * 0.55) * 5))
        wind = max(0, base_wind + math.sin(i * 0.45) * 4)
        if hazard == "heatwave":
            score = (temp * 1.65) - (humidity * 0.12) - (wind * 0.10)
        elif hazard == "flood":
            score = humidity * 0.45 + temp * 0.18 + wind * 0.10 + max(0, i) * 1.7
        else:
            score = humidity * 0.30 + wind * 0.55 + abs(temp - 22) * 0.35 + max(0, i) * 1.1
        level = "Low"
        if score >= 78:
            level = "Critical"
        elif score >= 64:
            level = "High"
        elif score >= 48:
            level = "Medium"
        rows.append({
            "date": today + timedelta(days=i),
            "day_label": "Today" if i == 0 else (f"D+{i}" if i > 0 else f"D{i}"),
            "temperature": round(temp, 1),
            "humidity": round(humidity),
            "wind": round(wind, 1),
            "risk_level": level,
            "score": round(score, 1),
        })
    return pd.DataFrame(rows)


def render_timeline(df: pd.DataFrame, temp_suffix: str = "°C") -> None:
    cols = st.columns(min(6, len(df)))
    for col, (_, row) in zip(cols, df.iterrows()):
        color = risk_to_color(row["risk_level"])
        with col:
            st.markdown(
                f"""
                <div class="timeline-card" style="border-top:3px solid {color};">
                    <div class="timeline-day">{row['day_label']}</div>
                    <div style="color:#cbd5e1;font-size:.75rem;">{row['date']}</div>
                    <div class="timeline-temp">{row['temperature']}{temp_suffix}</div>
                    <div class="timeline-risk" style="color:{color};">{row['risk_level']} · {row['score']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def call_predictor_safely(predictor, input_data: dict) -> dict:
    try:
        result = predictor(input_data)
        if not isinstance(result, dict):
            return {"risk_level": "Unknown", "confidence": 0.0, "risk_score": 0.0, "message": "Predictor returned an unsupported result."}
        result.setdefault("risk_level", result.get("severity_label", "Unknown"))
        result.setdefault("confidence", 0.0)
        result.setdefault("risk_score", float(result.get("confidence", 0)) * 100)
        result.setdefault("message", "Risk scenario processed.")
        return result
    except Exception as exc:
        return {"risk_level": "Unknown", "confidence": 0.0, "risk_score": 0.0, "message": f"Prediction engine error: {exc}"}


def render_result_card(title: str, result: dict) -> None:
    level = str(result.get("risk_level", result.get("severity_label", "Unknown")))
    color = risk_to_color(level)
    confidence = result.get("confidence", None)
    score = result.get("risk_score", None)
    extra = []
    if confidence is not None:
        try:
            extra.append(f"Confidence {float(confidence):.1%}")
        except Exception:
            pass
    if score is not None:
        try:
            extra.append(f"Score {float(score):.1f}")
        except Exception:
            pass
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid {color}; background: linear-gradient(135deg, {color}18, rgba(15,23,42,.74));">
            <div class="section-title" style="margin-top:0;">{title}</div>
            <div style="font-size:2.25rem;font-weight:900;color:{color};line-height:1;">{level}</div>
            <div style="color:#cbd5e1;margin-top:.55rem;font-weight:700;">{' · '.join(extra)}</div>
            <div style="color:#dbeafe;margin-top:.8rem;">{result.get('message', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_region_risk_summary() -> pd.DataFrame:
    preds = load_predictions()
    if preds.empty:
        return pd.DataFrame()
    data = preds.copy()
    data["_rank"] = data["risk_level"].map(RISK_ORDER).fillna(0)
    data = data.sort_values(["_rank", "confidence"], ascending=[False, False])
    return data.head(8)

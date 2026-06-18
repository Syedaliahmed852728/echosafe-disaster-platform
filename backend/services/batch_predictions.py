"""Batch prediction + alert generation for every region in master_regions.csv.

The dashboard reads two files from disk to populate the regional map / alerts
queue:

* ``predictions/batch/latest_disaster_predictions.csv``
* ``predictions/alerts/latest_alerts.csv``

This module produces both, by:

1. Loading the master region list.
2. Pulling today's Open-Meteo summary for each region (so the demo always
   reflects current real weather).
3. Running every risk-engine predictor on the assembled feature dict.
4. Writing the two CSVs in the exact schema the dashboard expects.

A "regions only" mode falls back to climatological inputs when the network
is unavailable, so the dashboard never has to render an empty state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS
from backend.risk_engine.earthquake_severity import classify_earthquake_severity
from backend.risk_engine.flood_predictor import predict_flood_risk
from backend.risk_engine.hailstorm_predictor import predict_hailstorm_risk
from backend.risk_engine.heatwave_predictor import predict_heatwave_risk
from backend.services.weather_fetch import fetch_regions_current_weather

logger = get_logger(__name__)

BATCH_OUTPUT = SETTINGS.project_root / "predictions" / "batch" / "latest_disaster_predictions.csv"
ALERTS_OUTPUT = SETTINGS.project_root / "predictions" / "alerts" / "latest_alerts.csv"
REGIONS_PATH = SETTINGS.project_root / "data" / "reference" / "master_regions.csv"
EARTHQUAKE_GOLD = SETTINGS.project_root / "data" / "gold" / "earthquake_risk" / "earthquake_risk_dataset.csv"

ALERT_LEVELS = {"High", "Critical"}


def _load_regions() -> pd.DataFrame:
    df = pd.read_csv(REGIONS_PATH)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def _augment_features(weather_row: Dict[str, Any]) -> Dict[str, Any]:
    """Add calendar / derived features the risk engine reads."""
    out = dict(weather_row)
    month = datetime.now(timezone.utc).month
    out["month"] = month
    out["quarter"] = (month - 1) // 3 + 1
    if month in (7, 8, 9):
        season_enc = 3
    elif month in (5, 6):
        season_enc = 2
    elif month in (3, 4):
        season_enc = 1
    else:
        season_enc = 0
    out["season_enc"] = season_enc
    out["is_monsoon"] = 1 if month in (7, 8, 9) else 0
    out.setdefault("water_level_index", 0.3)
    out.setdefault("temperature_drop_1d", 0.0)
    out.setdefault("rainfall_change_1d", 0.0)
    out.setdefault("storm_intensity_proxy", 0.4)
    return out


def _latest_earthquake_for_region(region: str) -> Optional[Dict[str, Any]]:
    if not EARTHQUAKE_GOLD.exists():
        return None
    try:
        df = pd.read_csv(EARTHQUAKE_GOLD)
    except Exception:
        return None
    if df.empty or "region" not in df.columns:
        return None
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"]).sort_values("event_time", ascending=False)
    match = df[df["region"].astype(str).str.lower() == region.lower()]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "magnitude": float(row.get("magnitude") or 0.0),
        "depth_km": float(row.get("depth_km") or 50.0),
        "region": region,
        "event_time": row["event_time"],
    }


def _row_for_disaster(
    region_row: pd.Series,
    disaster_type: str,
    result: Dict[str, Any],
    *,
    event_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    return {
        "timestamp": (event_time or datetime.now(timezone.utc)).isoformat(),
        "region": region_row["region"],
        "province": region_row["province"],
        "latitude": float(region_row["latitude"]),
        "longitude": float(region_row["longitude"]),
        "disaster_type": disaster_type,
        "risk_level": result.get("risk_level", "Unknown"),
        "confidence": result.get("confidence"),
        "risk_score": result.get("risk_score"),
        "model_name": result.get("model_name", ""),
        "message": result.get("message", ""),
    }


def generate_batch_predictions(
    *, use_live_weather: bool = True
) -> pd.DataFrame:
    """Run every predictor for every region; return a long-format DataFrame."""
    regions = _load_regions()

    if use_live_weather:
        try:
            weather = fetch_regions_current_weather(regions)
        except Exception as exc:
            logger.warning("Live Open-Meteo fetch failed; using fallback: %s", exc)
            weather = pd.DataFrame()
    else:
        weather = pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, region_row in regions.iterrows():
        if not weather.empty:
            match = weather[weather["region"] == region_row["region"]]
            base = match.iloc[0].to_dict() if not match.empty else {}
        else:
            base = {}
        # Sensible defaults so predictors never see NaN.
        base.setdefault("region", region_row["region"])
        base.setdefault("temperature_max_c", 32.0)
        base.setdefault("temperature_min_c", 22.0)
        base.setdefault("temperature_mean_c", 27.0)
        base.setdefault("humidity_mean_percent", 55.0)
        base.setdefault("wind_speed_mean_kmh", 14.0)
        base.setdefault("precipitation_mm", 0.0)
        base.setdefault("rainfall_mm", 0.0)
        features = _augment_features(base)

        heatwave = predict_heatwave_risk(features)
        hailstorm = predict_hailstorm_risk(features)
        flood = predict_flood_risk(features)
        rows.append(_row_for_disaster(region_row, "Heatwave Risk", heatwave))
        rows.append(_row_for_disaster(region_row, "Hailstorm Risk", hailstorm))
        rows.append(_row_for_disaster(region_row, "Flood / Heavy Rainfall Risk", flood))

        # Earthquake row uses the most recent recorded event for the region.
        eq = _latest_earthquake_for_region(region_row["region"])
        if eq is not None:
            severity = classify_earthquake_severity(eq)
            rows.append(
                _row_for_disaster(
                    region_row,
                    "Earthquake Monitoring",
                    severity,
                    event_time=eq["event_time"].to_pydatetime() if hasattr(eq["event_time"], "to_pydatetime") else eq["event_time"],
                )
            )

    df = pd.DataFrame(rows)
    BATCH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BATCH_OUTPUT, index=False)
    logger.info("Wrote %d batch predictions to %s", len(df), BATCH_OUTPUT)
    return df


def generate_alerts(predictions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Distil batch predictions into the high/critical alert queue."""
    if predictions is None:
        if not BATCH_OUTPUT.exists():
            predictions = generate_batch_predictions()
        else:
            predictions = pd.read_csv(BATCH_OUTPUT)
    alerts = predictions[predictions["risk_level"].isin(ALERT_LEVELS)].copy()
    alerts["status"] = "Open"
    alerts = alerts[
        [
            "timestamp",
            "region",
            "province",
            "latitude",
            "longitude",
            "disaster_type",
            "risk_level",
            "confidence",
            "risk_score",
            "message",
            "status",
        ]
    ].sort_values(["risk_level", "region"], ascending=[False, True])
    ALERTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(ALERTS_OUTPUT, index=False)
    logger.info("Wrote %d alerts to %s", len(alerts), ALERTS_OUTPUT)
    return alerts


def refresh_all(*, use_live_weather: bool = True) -> Dict[str, Path]:
    """Regenerate both files in one shot."""
    preds = generate_batch_predictions(use_live_weather=use_live_weather)
    generate_alerts(preds)
    return {"predictions": BATCH_OUTPUT, "alerts": ALERTS_OUTPUT}


if __name__ == "__main__":
    refresh_all()

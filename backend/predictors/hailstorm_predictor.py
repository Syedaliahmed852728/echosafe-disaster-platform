"""
Hailstorm Risk Predictor (real-data surface-variable build).

Loads the trained Random Forest hail-day classifier (a StandardScaler ->
RandomForestClassifier sklearn pipeline) and exposes a single ``predict``
entry-point mirroring earthquake_mag_predictor / flood_predictor.

When called with only (latitude, longitude), the predictor pulls today's
hourly surface profile from Open-Meteo (the same variables the training
pipeline used) and aggregates it the same way the silver pipeline does, so
live predictions match what the model was trained on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import pandas as pd

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS
from backend.ml_models.hailstorm.severity_classifier import classify_hail_severity
from backend.pipelines.utils.hailstorm_client import THUNDER_CODES, fetch_current_features
from backend.predictors.schemas import HailstormRiskInput, HailstormRiskOutput

logger = get_logger(__name__)

ATMOSPHERIC_KEYS = (
    "temperature_max_c",
    "temperature_min_c",
    "dew_point_mean_c",
    "rh_min_pct",
    "rh_mean_pct",
    "wind_speed_mean_ms",
    "wind_gust_max_ms",
    "surface_pressure_min_hpa",
    "surface_pressure_drop_hpa",
    "precipitation_sum_mm",
    "cloud_cover_mean_pct",
    "thunder_hours",
)


class HailstormRiskPredictor:
    """Random Forest hailstorm-day classifier with safe fallback."""

    MODEL_FILENAME = "hailstorm_risk_model.pkl"

    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self):
        model_dir = SETTINGS.model.model_dir / "hailstorm_risk"
        try:
            self.model = joblib.load(model_dir / self.MODEL_FILENAME)
            with open(model_dir / "hailstorm_risk_metadata.json") as fh:
                self.metadata = json.load(fh)
            logger.info("Hailstorm risk model loaded")
        except Exception as exc:
            logger.error(f"Failed to load hailstorm risk model: {exc}")
            self.model = None

    @staticmethod
    def _aggregate_daily(hourly: List[Dict[str, Any]]) -> Dict[str, float]:
        """Mirror the silver pipeline's daily aggregation for one location."""
        if not hourly:
            return {}
        df = pd.DataFrame(hourly)
        for c in (
            "temperature_c",
            "dew_point_c",
            "rh_pct",
            "wind_speed_ms",
            "wind_gust_ms",
            "surface_pressure_hpa",
            "precip_mm",
            "cloud_cover_pct",
            "weather_code",
        ):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        pressure_max = float(df["surface_pressure_hpa"].max(skipna=True))
        pressure_min = float(df["surface_pressure_hpa"].min(skipna=True))
        return {
            "temperature_max_c": float(df["temperature_c"].max(skipna=True)),
            "temperature_min_c": float(df["temperature_c"].min(skipna=True)),
            "dew_point_mean_c": float(df["dew_point_c"].mean(skipna=True)),
            "rh_mean_pct": float(df["rh_pct"].mean(skipna=True)),
            "rh_min_pct": float(df["rh_pct"].min(skipna=True)),
            "wind_speed_mean_ms": float(df["wind_speed_ms"].mean(skipna=True)),
            "wind_gust_max_ms": float(df["wind_gust_ms"].max(skipna=True)),
            "surface_pressure_min_hpa": pressure_min,
            "surface_pressure_drop_hpa": pressure_max - pressure_min,
            "precipitation_sum_mm": float(df["precip_mm"].sum(skipna=True)),
            "cloud_cover_mean_pct": float(df["cloud_cover_pct"].mean(skipna=True)),
            "thunder_hours": int(df["weather_code"].isin(THUNDER_CODES).sum()),
        }

    def _build_feature_row(self, data: Dict[str, Any]) -> Dict[str, float]:
        now = datetime.now(timezone.utc)
        missing = [k for k in ATMOSPHERIC_KEYS if k not in data]
        if missing:
            hourly = fetch_current_features(
                float(data["latitude"]), float(data["longitude"])
            )
            agg = self._aggregate_daily(hourly)
            for k in missing:
                data.setdefault(k, agg.get(k, 0.0))

        month = int(data.get("month", now.month))
        is_premonsoon = int(month in (3, 4, 5))
        row = {k: float(data.get(k, 0.0)) for k in ATMOSPHERIC_KEYS}
        row["month"] = month
        row["is_premonsoon"] = is_premonsoon
        return row

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        HailstormRiskInput(input_data)
        region = input_data.get("region", "Unknown")
        threshold = float(self.metadata.get("decision_threshold", 0.5))

        features = self._build_feature_row(dict(input_data))
        severity = classify_hail_severity(
            gust_ms=features["wind_gust_max_ms"],
            thunder_hours=int(features["thunder_hours"]),
            precipitation_mm=features["precipitation_sum_mm"],
        )

        if self.model is None:
            score = min(
                1.0,
                (features["wind_gust_max_ms"] / 25.0) * 0.4
                + (min(features["thunder_hours"], 6) / 6.0) * 0.4
                + (min(features["precipitation_sum_mm"], 25.0) / 25.0) * 0.2,
            )
            return HailstormRiskOutput(
                region=region,
                hail_probability=float(score),
                will_hail=bool(score >= threshold),
                severity_label=severity,
                model_name="fallback_heuristic",
                message=(
                    "Hailstorm risk model is not available; returned a "
                    "gust/thunder/precipitation heuristic."
                ),
                features_used=features,
            ).to_dict()

        feature_cols = self.metadata.get("features", list(features.keys()))
        X = pd.DataFrame([{c: features.get(c, 0.0) for c in feature_cols}])
        proba = float(self.model.predict_proba(X)[0, 1])
        will_hail = bool(proba >= threshold)

        return HailstormRiskOutput(
            region=region,
            hail_probability=proba,
            will_hail=will_hail,
            severity_label=severity,
            model_name=self.metadata.get("model_type", "LogisticRegression"),
            message=(
                f"Hail probability {proba:.2%} (severity: {severity}, "
                f"threshold={threshold:.3f}). Probability is the SMOTE-balanced "
                "LogisticRegression classifier output; severity is a rule-based "
                "summary of surface gusts / thunder hours / precipitation."
            ),
            features_used=features,
        ).to_dict()


def predict_hailstorm_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for one-shot hailstorm risk scoring."""
    return HailstormRiskPredictor().predict(input_data)


if __name__ == "__main__":
    sample = {
        "latitude": 31.5216,
        "longitude": 74.4036,
        "region": "Lahore",
    }
    result = predict_hailstorm_risk(sample)
    print(json.dumps(result, indent=2, default=str))

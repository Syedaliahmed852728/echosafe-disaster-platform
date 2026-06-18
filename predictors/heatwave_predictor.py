"""
Heatwave Risk Predictor.

Loads the trained heatwave-day binary classifier (a sklearn / imblearn
pipeline whose last step is the classifier) and exposes a single ``predict``
entry-point mirroring earthquake / flood / hailstorm predictors.

When called with only (latitude, longitude), the predictor pulls today's
+ short-range Open-Meteo forecast and engineers the same features the gold
pipeline used (Tmax anomaly against climatology, 3- and 7-day rolling means,
dry-day streak, etc.). Climatology is loaded from the gold dataset so the
operating-time anomaly matches the training-time anomaly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from config.logger import get_logger
from config.settings import SETTINGS
from ml_models.heatwave.features import EXCLUDED_REGIONS
from ml_models.heatwave.severity_classifier import classify_heatwave_severity
from pipelines.utils.heatwave_client import fetch_region_forecast
from predictors.schemas import HeatwaveRiskInput, HeatwaveRiskOutput

logger = get_logger(__name__)

DAILY_KEYS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class HeatwaveRiskPredictor:
    """Heatwave-day binary classifier with climatology-aware feature build."""

    MODEL_FILENAME = "heatwave_risk_model.pkl"

    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.climatology: pd.DataFrame | None = None  # (region, day_of_year) -> tmax_normal, tmin_normal
        self._load_model()
        self._load_climatology()

    def _load_model(self):
        model_dir = SETTINGS.model.model_dir / "heatwave_risk"
        try:
            self.model = joblib.load(model_dir / self.MODEL_FILENAME)
            with open(model_dir / "heatwave_risk_metadata.json") as fh:
                self.metadata = json.load(fh)
            logger.info("Heatwave risk model loaded")
        except Exception as exc:
            logger.error(f"Failed to load heatwave risk model: {exc}")
            self.model = None

    def _load_climatology(self):
        path = (
            SETTINGS.pipeline.gold_dir
            / "heatwave_risk"
            / "heatwave_risk_dataset.csv"
        )
        try:
            gold = pd.read_csv(
                path,
                usecols=[
                    "region",
                    "latitude",
                    "longitude",
                    "day_of_year",
                    "tmax_normal",
                    "tmin_normal",
                ],
            )
            self.climatology = (
                gold.dropna(subset=["tmax_normal"])
                .groupby(["region", "day_of_year"], as_index=False)
                .agg(
                    tmax_normal=("tmax_normal", "mean"),
                    tmin_normal=("tmin_normal", "mean"),
                    latitude=("latitude", "mean"),
                    longitude=("longitude", "mean"),
                )
            )
            logger.info(
                f"Loaded heatwave climatology: {len(self.climatology)} "
                "(region, day-of-year) rows"
            )
        except Exception as exc:
            logger.warning(f"Climatology load failed: {exc}; anomalies will be 0")
            self.climatology = None

    def _nearest_region(self, lat: float, lon: float) -> Dict[str, float] | None:
        if self.climatology is None or self.climatology.empty:
            return None
        regions = (
            self.climatology.groupby("region", as_index=False)
            .agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
        )
        distances = _haversine_km(
            lat, lon, regions["latitude"].to_numpy(), regions["longitude"].to_numpy()
        )
        idx = int(np.argmin(distances))
        return {
            "region": str(regions.iloc[idx]["region"]),
            "distance_km": round(float(distances[idx]), 2),
        }

    def _climo_for(self, region: str, doy: int) -> Dict[str, float]:
        if self.climatology is None:
            return {"tmax_normal": 0.0, "tmin_normal": 0.0}
        match = self.climatology[
            (self.climatology["region"] == region)
            & (self.climatology["day_of_year"] == doy)
        ]
        if match.empty:
            return {"tmax_normal": 0.0, "tmin_normal": 0.0}
        return {
            "tmax_normal": float(match["tmax_normal"].iloc[0]),
            "tmin_normal": float(match["tmin_normal"].iloc[0]),
        }

    def _engineer(self, region: str, daily: List[Dict[str, Any]]) -> pd.DataFrame:
        """Rebuild the gold-pipeline features for a forecast / past-days window."""
        df = pd.DataFrame(daily)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for c in DAILY_KEYS:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        df["day_of_year"] = df["date"].dt.dayofyear
        df["month"] = df["date"].dt.month
        df["is_premonsoon"] = df["month"].isin([3, 4, 5, 6]).astype(int)

        normals = df["day_of_year"].apply(lambda d: self._climo_for(region, int(d)))
        df["tmax_normal"] = normals.apply(lambda x: x["tmax_normal"])
        df["tmin_normal"] = normals.apply(lambda x: x["tmin_normal"])
        df["tmax_anomaly"] = df["temperature_2m_max"] - df["tmax_normal"]
        df["tmin_anomaly"] = df["temperature_2m_min"] - df["tmin_normal"]
        df["tmax_roll3"] = df["temperature_2m_max"].rolling(3, min_periods=1).mean()
        df["tmax_roll7"] = df["temperature_2m_max"].rolling(7, min_periods=1).mean()
        df["diurnal_range_c"] = df["temperature_2m_max"] - df["temperature_2m_min"]
        dry_streak: List[int] = []
        streak = 0
        for p in df["precipitation_sum"].fillna(0).to_numpy():
            streak = streak + 1 if p <= 0 else 0
            dry_streak.append(streak)
        df["dry_streak"] = dry_streak
        return df

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        HeatwaveRiskInput(input_data)

        lat = float(input_data["latitude"])
        lon = float(input_data["longitude"])
        nearest = self._nearest_region(lat, lon) or {"region": input_data.get("region", "Unknown")}
        region = input_data.get("region", nearest["region"])
        threshold = float(self.metadata.get("decision_threshold", 0.5))

        # Hard short-circuit for alpine regions. The PMD anomaly rule fires on
        # these from a cool baseline (e.g. Gilgit Tmax 22 deg C with +8 deg C
        # anomaly), but there is no operational heatwave risk. The training
        # gold dataset drops these regions too — see ml_models/heatwave/features.py.
        if region in EXCLUDED_REGIONS:
            return HeatwaveRiskOutput(
                region=region,
                heatwave_probability=0.0,
                will_be_heatwave=False,
                severity_label="None",
                model_name="excluded_region_rule",
                features_used={},
                forecast=[],
                message=(
                    f"{region} is excluded from the heatwave model — alpine / "
                    "high-altitude regions can trip the PMD anomaly rule on a "
                    "cool baseline but carry no operational heatwave risk."
                ),
            ).to_dict()

        # Pull the daily forecast (with 14 past days for rolling-window context).
        daily = fetch_region_forecast(
            {
                "region": region,
                "province": input_data.get("province", ""),
                "latitude": lat,
                "longitude": lon,
            }
        )
        df = self._engineer(region, daily)
        if df.empty:
            raise RuntimeError("Open-Meteo forecast returned no daily rows")

        feature_cols = self.metadata.get("features", [])
        if not feature_cols:
            raise RuntimeError("Loaded heatwave model has no feature list in metadata")

        today_utc = datetime.now(timezone.utc).date()
        future_mask = df["date"].dt.date >= today_utc
        scoring = df[future_mask].reset_index(drop=True)
        if scoring.empty:
            raise RuntimeError("Forecast window contains no future days")

        for c in feature_cols:
            if c not in scoring.columns:
                scoring[c] = 0.0

        if self.model is None:
            # Fallback heuristic: anomaly + apparent temp.
            scoring["probability"] = (
                (scoring["tmax_anomaly"].clip(lower=0) / 10.0).clip(upper=1.0) * 0.6
                + (scoring["apparent_temperature_max"].clip(lower=0) / 55.0).clip(upper=1.0)
                * 0.4
            )
            model_name = "fallback_heuristic"
        else:
            scoring["probability"] = self.model.predict_proba(scoring[feature_cols])[:, 1]
            model_name = self.metadata.get("model_type", "RandomForestClassifier")

        scoring["will_be_heatwave"] = (scoring["probability"] >= threshold).astype(int)
        scoring["severity"] = scoring.apply(
            lambda r: classify_heatwave_severity(
                tmax_anomaly=float(r["tmax_anomaly"]),
                apparent_tmax=float(r["apparent_temperature_max"]),
                is_heatwave=bool(int(r["will_be_heatwave"])),
            ),
            axis=1,
        )

        # Top-line answer = today / earliest forecast day.
        first = scoring.iloc[0]
        forecast = [
            {
                "date": str(row["date"].date()),
                "temperature_2m_max": round(float(row["temperature_2m_max"]), 2),
                "apparent_temperature_max": round(
                    float(row["apparent_temperature_max"]), 2
                ),
                "tmax_anomaly": round(float(row["tmax_anomaly"]), 2),
                "probability": round(float(row["probability"]), 4),
                "will_be_heatwave": bool(row["will_be_heatwave"]),
                "severity": row["severity"],
            }
            for _, row in scoring.iterrows()
        ]

        features_used = {c: float(first[c]) for c in feature_cols}

        return HeatwaveRiskOutput(
            region=region,
            heatwave_probability=float(first["probability"]),
            will_be_heatwave=bool(first["will_be_heatwave"]),
            severity_label=str(first["severity"]),
            model_name=model_name,
            features_used=features_used,
            forecast=forecast,
            message=(
                f"Heatwave probability {first['probability']:.2%} for "
                f"{first['date'].date()} (severity: {first['severity']}, "
                f"threshold={threshold:.3f}). PMD label = Tmax >= "
                "day-of-year normal + 5 deg C for >= 5 consecutive days."
            ),
        ).to_dict()


def predict_heatwave_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for one-shot heatwave risk scoring."""
    return HeatwaveRiskPredictor().predict(input_data)


if __name__ == "__main__":
    sample = {
        "latitude": 28.2814,
        "longitude": 68.4375,
        "region": "Sukkur",
    }
    result = predict_heatwave_risk(sample)
    print(json.dumps(result, indent=2, default=str))

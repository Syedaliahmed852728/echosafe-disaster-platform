"""
Earthquake Magnitude Predictor.

Loads the trained Linear Regression magnitude estimator (a StandardScaler ->
LinearRegression sklearn pipeline) and exposes a single `predict` entry-point
that mirrors flood_predictor / heatwave_predictor / hailstorm_predictor.

Model selection rationale: see Evidence/Earthquake_Magnitude_Model_Selection.ipynb.
Linear Regression beat SVR and Random Forest on MSE and R^2 on the EchoSafe
scraped USGS dataset.

This module does NOT predict earthquake occurrence. It estimates the magnitude
of an event that has already been detected (location + depth in).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd

from config.logger import get_logger
from config.settings import SETTINGS
from ml_models.earthquake.severity_classifier import classify_severity
from predictors.schemas import EarthquakeMagnitudeInput, MagnitudeEstimateOutput

logger = get_logger(__name__)


class EarthquakeMagnitudePredictor:
    """Linear Regression magnitude estimator with safe fallback."""

    MODEL_FILENAME = "earthquake_magnitude_model.pkl"

    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self):
        model_dir = SETTINGS.model.model_dir / "earthquake_magnitude"
        try:
            self.model = joblib.load(model_dir / self.MODEL_FILENAME)
            with open(model_dir / "earthquake_magnitude_metadata.json") as fh:
                self.metadata = json.load(fh)
            logger.info("Earthquake magnitude model loaded")
        except Exception as exc:
            logger.error(f"Failed to load magnitude model: {exc}")
            self.model = None

    def _build_feature_row(self, data: Dict[str, Any]) -> pd.DataFrame:
        feature_cols = self.metadata.get(
            "features",
            ["latitude", "longitude", "depth_km", "is_shallow", "year", "month"],
        )
        depth = float(data.get("depth_km", 50.0))
        now = datetime.now(timezone.utc)
        row = {
            "latitude": float(data.get("latitude", 0.0)),
            "longitude": float(data.get("longitude", 0.0)),
            "depth_km": depth,
            "is_shallow": int(depth <= 70.0),
            "year": int(data.get("year", now.year)),
            "month": int(data.get("month", now.month)),
        }
        return pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        EarthquakeMagnitudeInput(input_data)
        region = input_data.get("region", "Unknown")

        if self.model is None:
            # Fallback: use a simple location-blind heuristic so the API/dashboard
            # still return a sane shape until the model is trained.
            depth = float(input_data.get("depth_km", 50.0))
            est = max(3.0, 6.0 - (depth / 200.0))
            return MagnitudeEstimateOutput(
                region=region,
                estimated_magnitude=est,
                severity_label=classify_severity(est, depth),
                model_name="fallback_heuristic",
                message=(
                    "Magnitude estimator model is not available; returned a depth-only "
                    "heuristic estimate. Train the model via pipeline 14."
                ),
            ).to_dict()

        X = self._build_feature_row(input_data)
        estimated_mag = float(self.model.predict(X)[0])
        depth = float(input_data.get("depth_km", 50.0))
        severity = classify_severity(estimated_mag, depth)

        return MagnitudeEstimateOutput(
            region=region,
            estimated_magnitude=estimated_mag,
            severity_label=severity,
            model_name=self.metadata.get("model_type", "LinearRegression"),
            message=(
                f"Estimated magnitude {estimated_mag:.2f} (severity: {severity}). "
                "This is a magnitude estimate for a detected event; the module does "
                "not predict earthquake occurrence."
            ),
        ).to_dict()


def predict_earthquake_magnitude(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for one-shot magnitude estimation."""
    return EarthquakeMagnitudePredictor().predict(input_data)


if __name__ == "__main__":
    sample = {
        "latitude": 33.6844,
        "longitude": 73.0479,
        "depth_km": 30.0,
        "region": "Islamabad",
    }
    result = predict_earthquake_magnitude(sample)
    print(json.dumps(result, indent=2, default=str))

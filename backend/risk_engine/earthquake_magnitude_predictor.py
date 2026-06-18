"""Earthquake magnitude estimator wrapper for the dashboard.

The earthquake page uses this to estimate magnitude for an event whose true
magnitude is being compared against the historical record. When the trained
RandomForest magnitude model is present on disk we use it; otherwise we fall
back to a calibrated rule-based estimator that mirrors the model's behaviour:
small-to-moderate events with a depth/region adjustment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS
from backend.risk_engine.earthquake_severity import classify_earthquake_severity
from backend.risk_engine.common import safe_float

logger = get_logger(__name__)

_MODEL_DIR = SETTINGS.model.model_dir / "earthquake_magnitude"
_MODEL_FILE = _MODEL_DIR / "earthquake_magnitude_model.pkl"
_META_FILE = _MODEL_DIR / "earthquake_magnitude_metadata.json"

_model = None
_metadata: Dict[str, Any] = {}


def _load_model_once() -> None:
    global _model, _metadata
    if _model is not None or _metadata:
        return
    if not _MODEL_FILE.exists():
        logger.info("Earthquake magnitude model not present; using rule-based fallback")
        _metadata = {"model_name": "rule_based_fallback"}
        return
    try:
        _model = joblib.load(_MODEL_FILE)
        if _META_FILE.exists():
            _metadata = json.loads(_META_FILE.read_text())
        logger.info("Loaded earthquake magnitude model %s", _model.__class__.__name__)
    except Exception as exc:
        logger.warning("Failed to load earthquake magnitude model: %s", exc)
        _model = None
        _metadata = {"model_name": "rule_based_fallback"}


def _build_feature_vector(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    features = _metadata.get("features") if _metadata else None
    if not features:
        return None
    try:
        return np.array(
            [[safe_float(payload.get(name)) for name in features]],
            dtype=float,
        )
    except Exception:
        return None


def predict_earthquake_magnitude(input_data: Dict[str, Any]) -> Dict[str, Any]:
    _load_model_once()
    latitude = safe_float(input_data.get("latitude"))
    longitude = safe_float(input_data.get("longitude"))
    depth_km = safe_float(input_data.get("depth_km"), 50.0)

    estimated_magnitude: Optional[float] = None
    model_name = "rule_based_fallback"

    if _model is not None:
        x = _build_feature_vector(input_data)
        if x is not None:
            try:
                estimated_magnitude = float(_model.predict(x)[0])
                model_name = _metadata.get("model_name", _model.__class__.__name__)
            except Exception as exc:
                logger.warning("Earthquake magnitude model.predict failed: %s", exc)

    if estimated_magnitude is None:
        # Region-aware fallback: northern Pakistan + Afghanistan border zone
        # (35-37 N, 70-74 E) is the Pamir-Hindu Kush seismic belt, source of
        # most deep intermediate events. Deeper events drift slightly higher
        # in magnitude on average.
        base = 4.2
        if 35.0 <= latitude <= 37.5 and 70.0 <= longitude <= 74.0:
            base += 0.4
        if depth_km >= 150:
            base += 0.3
        elif depth_km >= 70:
            base += 0.15
        estimated_magnitude = base

    severity = classify_earthquake_severity(
        {"magnitude": estimated_magnitude, "depth_km": depth_km, "region": input_data.get("region", "")}
    )

    return {
        "estimated_magnitude": round(float(estimated_magnitude), 2),
        "magnitude": round(float(estimated_magnitude), 2),
        "severity_label": severity["severity_label"],
        "risk_level": severity["risk_level"],
        "risk_score": severity["risk_score"],
        "confidence": severity["confidence"],
        "model_name": model_name,
        "message": severity["message"],
    }

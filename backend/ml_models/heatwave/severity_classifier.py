"""
Heatwave Severity Classifier.

Rule-based severity classification driven by the Tmax anomaly (deg above the
day-of-year normal) and the apparent (feels-like) temperature. The classifier
is intentionally simple and matches the labelling convention the gold pipeline
uses for `heatwave_severity_label`.

Severity is reported even when ``is_heatwave == 0`` so that scorching but
short-duration single days still surface as Low/Medium, which is what the
public-health side of EchoSafe cares about.
"""

from typing import Dict

from backend.config.logger import get_logger

logger = get_logger(__name__)


def classify_heatwave_severity(
    tmax_anomaly: float,
    apparent_tmax: float,
    is_heatwave: bool = False,
) -> str:
    """Return one of: 'None', 'Low', 'Medium', 'High', 'Critical'."""
    if is_heatwave and (tmax_anomaly >= 11.0 or apparent_tmax >= 55.0):
        return "Critical"
    if is_heatwave and (tmax_anomaly >= 9.0 or apparent_tmax >= 50.0):
        return "High"
    if is_heatwave and (tmax_anomaly >= 7.0 or apparent_tmax >= 45.0):
        return "Medium"
    if is_heatwave:
        return "Low"
    # Single hot days that don't satisfy the run-length rule still get rated.
    if tmax_anomaly >= 9.0 or apparent_tmax >= 50.0:
        return "Medium"
    if tmax_anomaly >= 5.0 or apparent_tmax >= 45.0:
        return "Low"
    return "None"


def classify_event(event_data: Dict) -> Dict:
    anomaly = float(event_data.get("tmax_anomaly", 0))
    apparent = float(event_data.get("apparent_temperature_max", 0))
    is_hw = bool(event_data.get("is_heatwave", False))
    severity = classify_heatwave_severity(anomaly, apparent, is_heatwave=is_hw)
    logger.info(
        f"Heatwave classified: tmax_anomaly={anomaly:.1f} C, "
        f"apparent={apparent:.1f} C -> {severity}"
    )
    return {
        "disaster_type": "Heatwave Monitoring",
        "region": event_data.get("region", "Unknown"),
        "risk_level": severity,
        "tmax_anomaly": anomaly,
        "apparent_temperature_max": apparent,
        "is_heatwave": int(is_hw),
        "message": (
            f"Heatwave severity classified as {severity}. Severity is a "
            "rule-based summary of Tmax anomaly and apparent (feels-like) "
            "temperature; the binary heatwave-day predictor is a separate "
            "ML model."
        ),
    }

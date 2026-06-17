"""
Earthquake Severity Classifier
Rule-based severity classification. NO ML prediction.
This module monitors events and classifies severity only.
"""

from typing import Dict
from config.logger import get_logger

logger = get_logger(__name__)


def classify_severity(magnitude: float, depth_km: float) -> str:
    """Classify earthquake severity based on magnitude and depth."""
    if magnitude >= 7.0 or (magnitude >= 6.5 and depth_km <= 70):
        return "Critical"
    elif magnitude >= 5.5:
        return "High"
    elif magnitude >= 4.5:
        return "Medium"
    return "Low"


def classify_event(event_data: Dict) -> Dict:
    """Full event classification with metadata."""
    mag = event_data.get("magnitude", 0)
    depth = event_data.get("depth_km", 100)
    severity = classify_severity(mag, depth)

    logger.info(f"Earthquake classified: M{mag} at {depth}km -> {severity}")

    return {
        "disaster_type": "Earthquake Monitoring",
        "region": event_data.get("region", "Unknown"),
        "risk_level": severity,
        "severity_score": min(mag * 10 + max(100 - depth, 0), 100),
        "magnitude": mag,
        "depth_km": depth,
        "message": f"Earthquake event classified as {severity} severity. Monitoring only - this module does not predict earthquake occurrence.",
    }

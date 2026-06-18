"""Earthquake severity classifier wrapper for the dashboard.

Severity is rule-based on magnitude + focal depth (shallow events are more
damaging at the same magnitude). The thresholds match the
``backend.ml_models.earthquake.severity_classifier`` used by the gold pipeline
so dashboard severity labels line up with the labels in the gold dataset.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.ml_models.earthquake.severity_classifier import classify_severity
from backend.risk_engine.common import build_result, safe_float, severity_from_label


_LEVEL_FOR_SEVERITY = {
    "Minor": "Low",
    "Light": "Low",
    "Moderate": "Medium",
    "Strong": "High",
    "Major": "High",
    "Great": "Critical",
    "Catastrophic": "Critical",
}


def _level_from_severity_label(label: str) -> str:
    if not label:
        return "Low"
    return _LEVEL_FOR_SEVERITY.get(label, severity_from_label(label))


def classify_earthquake_severity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    magnitude = safe_float(input_data.get("magnitude"))
    depth_km = safe_float(input_data.get("depth_km"), 50.0)
    severity_label = classify_severity(magnitude, depth_km)
    risk_level = _level_from_severity_label(severity_label)

    if magnitude >= 7.0:
        score = 1.0
    elif magnitude >= 6.0:
        score = 0.85
    elif magnitude >= 5.0:
        score = 0.65
    elif magnitude >= 4.0:
        score = 0.40
    elif magnitude >= 3.0:
        score = 0.20
    else:
        score = 0.05

    if depth_km <= 70:  # shallow events feel stronger
        score = min(1.0, score + 0.1)

    if magnitude >= 4.5 and depth_km <= 70:
        message = (
            f"Shallow M{magnitude:.1f} event at {depth_km:.0f} km depth; "
            "strong shaking near the epicentre."
        )
    elif magnitude >= 4.5:
        message = (
            f"M{magnitude:.1f} event at {depth_km:.0f} km depth; "
            "deeper focal depth moderates surface shaking."
        )
    elif magnitude >= 3.0:
        message = f"M{magnitude:.1f} micro / light tremor; widely felt only near epicentre."
    else:
        message = f"Sub-M3 event ({magnitude:.1f}); barely felt."

    result = build_result(
        risk_level,
        score,
        message=message,
        model_name="earthquake_severity_rule",
        extra={
            "severity_label": severity_label,
            "magnitude": magnitude,
            "depth_km": depth_km,
        },
    )
    # The dashboard's normalisation prefers severity_label when present.
    result["severity_label"] = severity_label
    return result

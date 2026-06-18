"""Shared helpers for the risk-engine wrappers.

These keep the four disaster modules consistent: same risk-level ladder,
same score-to-label mapping, same response shape.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

RISK_LEVELS = ("Low", "Medium", "High", "Critical")


def risk_level_for_score(score: float) -> str:
    """Map a 0..1 normalised risk score to the canonical ladder.

    Floor is "Low" rather than "None" because the dashboard only renders
    risk pills / map colours for Low / Medium / High / Critical — anything
    else falls into a grey "Unknown" pill which looks like missing data.
    """
    if score >= 0.85:
        return "Critical"
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def severity_from_label(label: str) -> str:
    """Coerce model labels (e.g. 'Severe Heatwave') to a risk-level word."""
    if not label:
        return "Low"
    lower = label.lower()
    if "critical" in lower or "extreme" in lower or "destructive" in lower:
        return "Critical"
    if "severe" in lower or "high" in lower or "major" in lower:
        return "High"
    if "moderate" in lower or "medium" in lower:
        return "Medium"
    if "mild" in lower or "minor" in lower or "low" in lower:
        return "Low"
    return "Low"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_result(
    risk_level: str,
    risk_score: float,
    *,
    confidence: Optional[float] = None,
    message: str = "",
    model_name: str = "rule_based",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    risk_score = max(0.0, min(1.0, float(risk_score)))
    out: Dict[str, Any] = {
        "risk_level": risk_level,
        "severity_label": risk_level,
        "confidence": float(confidence) if confidence is not None else round(0.55 + 0.4 * risk_score, 3),
        "risk_score": round(risk_score, 4),
        "message": message,
        "model_name": model_name,
    }
    if extra:
        out.update(extra)
    return out

"""
Hailstorm Severity Classifier (surface-proxy version).

Rule-based severity classification driven by the surface-variable signals the
Open-Meteo archive actually returns for Pakistan: peak wind gust, thunder
hour count, and the daily precipitation total. The classifier is intentionally
simple and matches the labelling convention the gold pipeline uses for
`hail_severity_label`.
"""

from typing import Dict

from config.logger import get_logger

logger = get_logger(__name__)


def classify_hail_severity(
    gust_ms: float,
    thunder_hours: int,
    precipitation_mm: float,
    observed: bool = False,
) -> str:
    """Return one of: 'None', 'Low', 'Medium', 'High', 'Critical'."""
    has_thunder = int(thunder_hours) > 0
    if observed and gust_ms >= 22 and precipitation_mm >= 20:
        return "Critical"
    if observed and (gust_ms >= 17 or has_thunder):
        return "High"
    if gust_ms >= 17 and has_thunder:
        return "High"
    if gust_ms >= 11 and (has_thunder or precipitation_mm >= 5):
        return "Medium"
    if observed or precipitation_mm > 0 or has_thunder or gust_ms >= 8:
        return "Low"
    return "None"


def classify_event(event_data: Dict) -> Dict:
    gust = float(event_data.get("wind_gust_max_ms", 0))
    thunder = int(event_data.get("thunder_hours", 0))
    precip = float(event_data.get("precipitation_sum_mm", 0))
    observed = bool(event_data.get("hail_observed", False))
    severity = classify_hail_severity(gust, thunder, precip, observed=observed)
    logger.info(
        f"Hailstorm classified: gust={gust:.1f} m/s, thunder_hrs={thunder}, "
        f"precip={precip:.1f} mm -> {severity}"
    )
    return {
        "disaster_type": "Hailstorm Monitoring",
        "region": event_data.get("region", "Unknown"),
        "risk_level": severity,
        "wind_gust_max_ms": gust,
        "thunder_hours": thunder,
        "precipitation_sum_mm": precip,
        "message": (
            f"Hailstorm severity classified as {severity}. Severity is a "
            "rule-based summary of gusts / thunder hours / rainfall; the binary "
            "hail-day predictor is a separate ML model."
        ),
    }

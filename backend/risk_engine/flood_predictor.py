"""Flood / heavy-rainfall risk wrapper for the dashboard.

Pakistan's flood risk is dominated by monsoon convective rainfall (July-Sept),
glacial-melt riverine pulses in summer, and urban flash-flood events in Sindh
/ Punjab during heavy single-day downpours. The wrapper scores rainfall load
and the optional water-level index supplied by the caller.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.risk_engine.common import build_result, risk_level_for_score, safe_float


def predict_flood_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    tmean = safe_float(input_data.get("temperature_mean_c"))
    humidity = safe_float(input_data.get("humidity_mean_percent"))
    wind_kmh = safe_float(input_data.get("wind_speed_mean_kmh"))
    precip = safe_float(input_data.get("precipitation_mm"))
    water_level = safe_float(input_data.get("water_level_index"))
    month = int(safe_float(input_data.get("month"), 7))

    # Signal 1: rainfall intensity. > 50 mm/day is heavy in Pakistan; >100 is
    # destructive.
    if precip >= 120:
        rain_score = 1.0
    elif precip >= 80:
        rain_score = 0.85
    elif precip >= 50:
        rain_score = 0.65
    elif precip >= 25:
        rain_score = 0.40
    elif precip >= 10:
        rain_score = 0.20
    else:
        rain_score = 0.0

    # Signal 2: caller-supplied water level (river stage / catchment proxy).
    if water_level >= 0.85:
        wl_score = 1.0
    elif water_level >= 0.65:
        wl_score = 0.7
    elif water_level >= 0.40:
        wl_score = 0.4
    elif water_level >= 0.20:
        wl_score = 0.2
    else:
        wl_score = max(0.0, min(0.15, water_level))

    # Signal 3: saturated atmosphere increases flash-flood odds.
    sat_score = max(0.0, min(1.0, (humidity - 65.0) / 30.0))

    # Signal 4: monsoon climatology bias.
    if month in (7, 8):
        season_bias = 0.15
    elif month in (6, 9):
        season_bias = 0.08
    else:
        season_bias = -0.05

    risk_score = (
        0.55 * rain_score
        + 0.25 * wl_score
        + 0.10 * sat_score
        + season_bias
        + 0.05 * max(0.0, min(1.0, wind_kmh / 60.0))
    )
    risk_score = max(0.0, min(1.0, risk_score))
    level = risk_level_for_score(risk_score)

    if level in ("High", "Critical"):
        message = (
            f"Heavy rainfall event: {precip:.0f} mm forecast; water-level index "
            f"{water_level:.2f}. Flash flood / urban flooding likely; avoid low-lying roads."
        )
    elif level == "Medium":
        message = (
            f"Moderate rainfall ({precip:.0f} mm) with elevated catchment load. "
            "Watch nullahs and drains."
        )
    elif level == "Low":
        message = f"Light rainfall ({precip:.0f} mm); minor ponding possible."
    else:
        message = "Dry conditions; no flood signal."

    return build_result(
        level,
        risk_score,
        message=message,
        model_name="flood_rule_engine_v1",
        extra={
            "rain_score": round(rain_score, 3),
            "water_level_score": round(wl_score, 3),
            "saturation_score": round(sat_score, 3),
            "season_bias": round(season_bias, 3),
        },
    )

"""Heatwave risk wrapper for the dashboard.

Takes the daily feature row the dashboard already built from Open-Meteo and
returns a uniform risk dict. The score is computed against three signals:

* Tmax absolute level (PMD heatwave guideline: warning at >= 40 deg C in the
  plains, severe at >= 45 deg C).
* Apparent (feels-like) temperature, which combines temp, humidity, wind.
* Diurnal range collapse and humidity load — captures sticky / muggy days
  that don't break Tmax but still cause heat stress.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.risk_engine.common import build_result, risk_level_for_score, safe_float


def predict_heatwave_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    tmax = safe_float(input_data.get("temperature_max_c"))
    tmean = safe_float(input_data.get("temperature_mean_c"))
    tmin = safe_float(input_data.get("temperature_min_c"))
    humidity = safe_float(input_data.get("humidity_mean_percent"))
    wind_kmh = safe_float(input_data.get("wind_speed_mean_kmh"))
    month = int(safe_float(input_data.get("month"), 5))
    is_monsoon = int(safe_float(input_data.get("is_monsoon"), 0))

    # Apparent temperature (Australian BoM-style approximation in deg C).
    e = (humidity / 100.0) * 6.105 * 2.71828 ** (17.27 * tmax / (237.7 + max(tmax, -50)))
    apparent = tmax + 0.33 * e - 0.7 * (wind_kmh / 3.6) - 4.0

    # Signal 1: absolute Tmax.
    if tmax >= 48:
        tmax_score = 1.0
    elif tmax >= 45:
        tmax_score = 0.85
    elif tmax >= 42:
        tmax_score = 0.65
    elif tmax >= 38:
        tmax_score = 0.40
    elif tmax >= 35:
        tmax_score = 0.20
    else:
        tmax_score = 0.0

    # Signal 2: apparent (feels-like) load.
    apparent_score = max(0.0, min(1.0, (apparent - 35.0) / 18.0))

    # Signal 3: warm nights — Tmin >= 28 deg C means no overnight relief.
    night_score = max(0.0, min(1.0, (tmin - 26.0) / 8.0))

    # Pre-monsoon May/June bake is the worst-case in Pakistan; monsoon humidity
    # raises feels-like but moderates absolute Tmax. Bias slightly.
    seasonal_bias = 0.05 if month in (5, 6) and not is_monsoon else 0.0

    risk_score = (
        0.55 * tmax_score
        + 0.30 * apparent_score
        + 0.15 * night_score
        + seasonal_bias
    )
    risk_score = max(0.0, min(1.0, risk_score))
    level = risk_level_for_score(risk_score)

    if level in ("High", "Critical"):
        message = (
            f"Heatwave conditions: Tmax {tmax:.1f} degC, feels-like {apparent:.1f} degC, "
            f"Tmin {tmin:.1f} degC. Limit outdoor exposure 11:00-17:00 and increase fluids."
        )
    elif level == "Medium":
        message = (
            f"Elevated heat stress: Tmax {tmax:.1f} degC, feels-like {apparent:.1f} degC. "
            "Hydrate and pace outdoor activity."
        )
    elif level == "Low":
        message = (
            f"Warm day (Tmax {tmax:.1f} degC) but no heatwave threshold breached."
        )
    else:
        message = f"Comfortable conditions (Tmax {tmax:.1f} degC, humidity {humidity:.0f}%)."

    return build_result(
        level,
        risk_score,
        message=message,
        model_name="heatwave_rule_engine_v1",
        extra={
            "apparent_temperature_c": round(apparent, 2),
            "tmax_score": round(tmax_score, 3),
            "apparent_score": round(apparent_score, 3),
            "night_score": round(night_score, 3),
        },
    )

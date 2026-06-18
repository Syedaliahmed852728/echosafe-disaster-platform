"""Hailstorm risk wrapper for the dashboard.

Hail in Pakistan is overwhelmingly a pre-monsoon (March-May) phenomenon
driven by deep convection: large diurnal swing, fast cold-pool wind gusts,
and convective rainfall. The wrapper scores those proxies directly.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.risk_engine.common import build_result, risk_level_for_score, safe_float


def predict_hailstorm_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    tmean = safe_float(input_data.get("temperature_mean_c"))
    tmax = safe_float(input_data.get("temperature_max_c"))
    tmin = safe_float(input_data.get("temperature_min_c"))
    humidity = safe_float(input_data.get("humidity_mean_percent"))
    wind_kmh = safe_float(input_data.get("wind_speed_mean_kmh"))
    precip = safe_float(input_data.get("precipitation_mm"))
    rainfall = safe_float(input_data.get("rainfall_mm"))
    temp_drop = abs(safe_float(input_data.get("temperature_drop_1d")))
    rain_change = abs(safe_float(input_data.get("rainfall_change_1d")))
    storm_intensity = safe_float(input_data.get("storm_intensity_proxy"))
    month = int(safe_float(input_data.get("month"), 4))

    # Signal 1: storm-cell wind speed (m/s converted from kmh).
    wind_score = max(0.0, min(1.0, (wind_kmh - 30.0) / 50.0))

    # Signal 2: heavy convective rainfall.
    rain_score = max(0.0, min(1.0, max(precip, rainfall) / 25.0))

    # Signal 3: instability — large diurnal range + sudden temperature drop.
    diurnal = max(0.0, tmax - tmin)
    instability_score = max(
        0.0,
        min(
            1.0,
            0.6 * max(0.0, (diurnal - 12.0) / 18.0)
            + 0.4 * max(0.0, (temp_drop - 4.0) / 12.0),
        ),
    )

    # Signal 4: pre-monsoon climatology bias. Hail is essentially confined to
    # March-May plus a smaller October peak.
    if month in (3, 4, 5):
        season_bias = 0.12
    elif month == 10:
        season_bias = 0.06
    else:
        season_bias = -0.05

    # Signal 5: caller-supplied storm intensity proxy (0..1ish).
    intensity_score = max(0.0, min(1.0, storm_intensity / 100.0)) if storm_intensity > 1.5 else max(0.0, min(1.0, storm_intensity))

    risk_score = (
        0.30 * wind_score
        + 0.25 * rain_score
        + 0.25 * instability_score
        + 0.20 * intensity_score
        + season_bias
        + 0.05 * max(0.0, min(1.0, rain_change / 15.0))
    )
    risk_score = max(0.0, min(1.0, risk_score))
    level = risk_level_for_score(risk_score)

    if level in ("High", "Critical"):
        message = (
            f"Convective storm signature: gusts {wind_kmh:.0f} km/h, "
            f"rainfall {max(precip, rainfall):.1f} mm, diurnal range "
            f"{diurnal:.1f} degC. Hail likely - seek covered parking."
        )
    elif level == "Medium":
        message = (
            f"Unstable atmosphere ({diurnal:.1f} degC diurnal swing, "
            f"gusts {wind_kmh:.0f} km/h). Watch for afternoon thunderstorms."
        )
    elif level == "Low":
        message = "Light convective potential; minor risk."
    else:
        message = "Stable conditions; no hailstorm signature."

    return build_result(
        level,
        risk_score,
        message=message,
        model_name="hailstorm_rule_engine_v1",
        extra={
            "wind_score": round(wind_score, 3),
            "rain_score": round(rain_score, 3),
            "instability_score": round(instability_score, 3),
            "season_bias": round(season_bias, 3),
        },
    )

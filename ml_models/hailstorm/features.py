"""
Hailstorm feature definitions (surface-variable, real-data version).

The Open-Meteo public archive does not expose CAPE / pressure-level winds /
freezing-level height for Pakistan, so the production hail classifier uses
surface proxies that operational forecasters lean on when soundings aren't
available:

- temperature_max_c           : day-time heating, instability driver
- temperature_min_c           : diurnal range proxy
- dew_point_mean_c            : low-level moisture
- rh_min_pct                  : mid-day dryness (dry slot => stronger downdraughts)
- rh_mean_pct                 : overall moisture envelope
- wind_speed_mean_ms          : ambient surface flow
- wind_gust_max_ms            : downdraft strength proxy
- surface_pressure_min_hpa    : storm depth proxy
- surface_pressure_drop_hpa   : daily pressure fall (frontal/convective signal)
- precipitation_sum_mm        : convective rainfall total
- cloud_cover_mean_pct        : cloud envelope
- thunder_hours               : count of hours with WMO weather_code in {95,96,99}
- month                       : seasonality
- is_premonsoon               : 1 in Mar-May (Pakistan hail peak)

Target: hail_observed (binary, 1 if any METAR within the station-day reported
        GR or GS).
"""

HAIL_FEATURES = [
    "temperature_max_c",
    "temperature_min_c",
    "dew_point_mean_c",
    "rh_min_pct",
    "rh_mean_pct",
    "wind_speed_mean_ms",
    "wind_gust_max_ms",
    "surface_pressure_min_hpa",
    "surface_pressure_drop_hpa",
    "precipitation_sum_mm",
    "cloud_cover_mean_pct",
    "thunder_hours",
    "month",
    "is_premonsoon",
]

HAIL_TARGET = "hail_observed"

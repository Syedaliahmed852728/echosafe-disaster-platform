"""
Heatwave feature definitions.

Two consumers:
  - severity_classifier (rule-based, uses tmax_anomaly + apparent_temperature_max)
  - risk_trainer (binary classifier on the columns below)

Features used by the risk classifier
------------------------------------
- temperature_2m_max          : raw daily peak temperature
- temperature_2m_min          : raw daily low temperature
- temperature_2m_mean         : raw daily mean
- apparent_temperature_max    : 'feels-like' peak (humidity + wind folded in)
- shortwave_radiation_sum     : total daily insolation
- wind_speed_10m_max          : peak wind (cooling proxy when high)
- precipitation_sum           : daily rainfall
- et0_fao_evapotranspiration  : evaporative demand / dryness proxy
- tmax_anomaly                : Tmax - day-of-year normal (climatological signal)
- tmin_anomaly                : Tmin - day-of-year normal
- tmax_roll3, tmax_roll7      : 3- and 7-day rolling Tmax means
- diurnal_range_c             : Tmax - Tmin
- dry_streak                  : consecutive zero-precipitation days
- day_of_year, month          : seasonality
- is_premonsoon               : 1 in Mar - Jun (Pakistan heatwave peak)

Target: is_heatwave (binary, PMD rule: Tmax >= normal + 5 deg C for >= 5
        consecutive days; runs are also tracked in heatwave_event).
"""

HEATWAVE_FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "tmax_anomaly",
    "tmin_anomaly",
    "tmax_roll3",
    "tmax_roll7",
    "diurnal_range_c",
    "dry_streak",
    "day_of_year",
    "month",
    "is_premonsoon",
]

HEATWAVE_TARGET = "is_heatwave"

# Regions that are excluded from the heatwave model entirely. These areas
# (alpine Gilgit-Baltistan and Himalayan AJK) can trip the PMD anomaly rule
# on a small absolute temperature spike from a cool baseline, but there is
# no operational public-health heatwave risk — Gilgit's "heatwave" days look
# like Tmax ~22 deg C with a +8 deg C anomaly. Dropping them at the gold
# stage removes those misleading positives from the training set, and the
# predictor short-circuits to a constant "not a heatwave" if asked.
EXCLUDED_REGIONS = ("Gilgit", "Muzaffarabad")

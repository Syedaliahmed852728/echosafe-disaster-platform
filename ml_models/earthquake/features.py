"""
Earthquake feature definitions.

Two sub-modules consume this file:
  - severity_classifier (rule-based, no features needed)
  - magnitude_estimator (Random Forest regression on the features below)

Features used by the magnitude estimator
----------------------------------------
- latitude, longitude : epicentre coordinates (degrees)
- depth_km            : hypocentre depth in km
- is_shallow          : depth <= 70 km (binary; correlates with felt intensity)
- year, month         : seasonal/long-period structure in regional seismicity

Target: magnitude (continuous, Richter scale).
"""

MAGNITUDE_FEATURES = [
    "latitude",
    "longitude",
    "depth_km",
    "is_shallow",
    "year",
    "month",
]

MAGNITUDE_TARGET = "magnitude"

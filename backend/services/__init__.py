"""Service layer the EchoSafe dashboard talks to.

The Streamlit pages do not call ML models or read CSVs directly. They go
through this layer, which:

* loads canonical reference tables (regions, master station list, ...);
* fetches live Open-Meteo data;
* runs the rule / model based predictors in ``backend.risk_engine`` for every
  region and assembles the batch prediction + alerts files the dashboard
  expects to find on disk.

Use :func:`backend.services.batch_predictions.refresh_all` to regenerate
everything the dashboard reads.
"""

from backend.services.batch_predictions import (
    generate_alerts,
    generate_batch_predictions,
    refresh_all,
)
from backend.services.weather_fetch import (
    fetch_region_current_weather,
    fetch_regions_current_weather,
)

__all__ = [
    "generate_alerts",
    "generate_batch_predictions",
    "refresh_all",
    "fetch_region_current_weather",
    "fetch_regions_current_weather",
]

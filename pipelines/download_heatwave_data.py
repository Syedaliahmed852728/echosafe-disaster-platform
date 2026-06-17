#!/usr/bin/env python3
"""
Pipeline 04: Download Heatwave Data (Bronze).

Pulls daily ERA5-archive temperature, apparent-temperature, radiation, wind,
precipitation, and evapotranspiration for every Pakistani region from
Open-Meteo. Same variables are used for live forecasting in the predictor,
so train and predict stay aligned.

Output: data/bronze/heatwave/heatwave_daily_raw.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils.heatwave_client import download_heatwave_data
from config.logger import get_logger

logger = get_logger("pipeline.04")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 04: Download Heatwave Data")
    logger.info("Region: Pakistani master regions")
    logger.info("Source: Open-Meteo ERA5 archive (daily)")
    logger.info("=" * 60)
    download_heatwave_data()
    logger.info("Pipeline 04 completed successfully")

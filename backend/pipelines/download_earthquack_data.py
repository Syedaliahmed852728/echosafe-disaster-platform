#!/usr/bin/env python3
"""
Pipeline 02: Download Earthquake Data (Bronze)
Scrapes the USGS earthquake search form for events in the Pakistan +
neighbouring seismic zones bounding box (N=40, S=20, W=60, E=105).

Rolling window: past 10 years from today (recomputed at every run, so the
daily Airflow schedule keeps the dataset current). The client checks the
DB first and only fetches missing windows; it merges new events into the
existing bronze JSON.

Output: data/bronze/earthquake/earthquake_events_raw.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pipelines.utils.earthquake_client import download_earthquake_data
from backend.config.logger import get_logger

logger = get_logger("pipeline.02")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 02: Download Earthquake Data")
    logger.info("Region: Pakistan bounding box")
    logger.info("Source: USGS Earthquake API")
    logger.info("=" * 60)
    download_earthquake_data()
    logger.info("Pipeline 02 completed successfully")

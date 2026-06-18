#!/usr/bin/env python3
"""
Pipeline 03: Download Hailstorm Data (Bronze).

Pulls real observed hail labels from the Iowa Environmental Mesonet ASOS
network (`PK__ASOS`) plus the matching hourly atmospheric profile from
Open-Meteo's ERA5 archive for every Pakistani station. Output is one bronze
JSON containing labels + features for the past 10 years.

Output: data/bronze/hailstorm/hailstorm_events_raw.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pipelines.utils.hailstorm_client import download_hailstorm_data
from backend.config.logger import get_logger

logger = get_logger("pipeline.03")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PIPELINE 03: Download Hailstorm Data")
    logger.info("Region: Pakistan (PK__ASOS network)")
    logger.info("Sources: IEM ASOS (labels) + Open-Meteo ERA5 (features)")
    logger.info("=" * 60)
    download_hailstorm_data()
    logger.info("Pipeline 03 completed successfully")

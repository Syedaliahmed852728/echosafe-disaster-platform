#!/usr/bin/env python3
"""
DAG: Earthquake Monitoring Pipeline
Separate DAG for earthquake data ingestion and severity classification.
Schedule: Every 6 hours
"""

import sys

sys.path.insert(0, "/opt/echosafe")

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from utils.dag_defaults import DEFAULT_ARGS, PROJECT_ROOT

with DAG(
    dag_id="echosafe_earthquake_pipeline",
    default_args=DEFAULT_ARGS,
    description="Earthquake monitoring: data ingestion -> severity classification",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["echosafe", "earthquake", "monitoring"],
) as dag:
    download_earthquakes = BashOperator(
        task_id="download_earthquake_data",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/download_earthquake_data.py",
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver_earthquake",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/bronze_to_silver_earthquake.py",
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold_earthquake",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/silver_to_gold_earthquake.py",
    )

    download_earthquakes >> bronze_to_silver >> silver_to_gold

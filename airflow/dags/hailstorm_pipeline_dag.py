#!/usr/bin/env python3
"""
DAG: Hailstorm Monitoring Pipeline.

Daily run: scrape IEM hail labels + Open-Meteo features for Pakistan,
clean to silver, engineer gold features, and retrain the Random Forest
hail-day classifier.
"""

import sys

sys.path.insert(0, "/opt/echosafe")

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

from utils.dag_defaults import DEFAULT_ARGS, PROJECT_ROOT

with DAG(
    dag_id="echosafe_hailstorm_pipeline",
    default_args=DEFAULT_ARGS,
    description="Hailstorm: data ingestion -> risk classification",
    schedule_interval="0 3 * * *",  # daily at 03:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["echosafe", "hailstorm", "monitoring"],
) as dag:
    download_hailstorm = BashOperator(
        task_id="download_hailstorm_data",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/download_hailstorm_data.py",
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver_hailstorm",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/bronze_to_silver_hailstorm.py",
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold_hailstorm",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/silver_to_gold_hailstorm.py",
    )

    train_risk_model = BashOperator(
        task_id="train_hailstorm_risk_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && python -m ml_models.hailstorm.risk_trainer"
        ),
    )

    download_hailstorm >> bronze_to_silver >> silver_to_gold >> train_risk_model

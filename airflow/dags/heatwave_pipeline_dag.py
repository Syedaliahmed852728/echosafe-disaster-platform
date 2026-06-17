#!/usr/bin/env python3
"""
DAG: Heatwave Monitoring Pipeline.

Daily run: refresh Open-Meteo daily history for every Pakistani region,
clean to silver (with day-of-year climatology), engineer gold features +
PMD heatwave labels, retrain the heatwave-day classifier.
"""

import sys

sys.path.insert(0, "/opt/echosafe")

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

from utils.dag_defaults import DEFAULT_ARGS, PROJECT_ROOT

with DAG(
    dag_id="echosafe_heatwave_pipeline",
    default_args=DEFAULT_ARGS,
    description="Heatwave: data ingestion -> PMD labelling -> risk classification",
    schedule_interval="0 4 * * *",  # daily at 04:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["echosafe", "heatwave", "monitoring"],
) as dag:
    download_heatwave = BashOperator(
        task_id="download_heatwave_data",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/download_heatwave_data.py",
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver_heatwave",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/bronze_to_silver_heatwave.py",
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold_heatwave",
        bash_command=f"cd {PROJECT_ROOT} && python pipelines/silver_to_gold_heatwave.py",
    )

    train_risk_model = BashOperator(
        task_id="train_heatwave_risk_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && python -m ml_models.heatwave.risk_trainer"
        ),
    )

    download_heatwave >> bronze_to_silver >> silver_to_gold >> train_risk_model

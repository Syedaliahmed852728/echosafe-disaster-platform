"""
Shared DAG default arguments and configuration.
"""

from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "echosafe",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

PROJECT_ROOT = "/opt/echosafe"

"""
Database Connection Manager
Handles PostgreSQL connections with connection pooling.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from typing import Optional
from config.settings import SETTINGS
from config.logger import get_logger

logger = get_logger(__name__)

_engine = None


def get_engine():
    """Get or create SQLAlchemy engine with connection pooling."""
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(
                SETTINGS.database.url,
                poolclass=QueuePool,
                pool_size=SETTINGS.database.pool_size,
                max_overflow=SETTINGS.database.max_overflow,
                pool_timeout=SETTINGS.database.pool_timeout,
                echo=SETTINGS.database.echo,
            )
            logger.info("Database engine created")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            _engine = None
    return _engine


def execute_query(query: str, params: dict = None) -> Optional[pd.DataFrame]:
    """Execute a SQL query and return results as DataFrame."""
    engine = get_engine()
    if engine is None:
        logger.warning("Database not available, skipping query")
        return None
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return None


def health_check() -> bool:
    """Check if database connection is healthy."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

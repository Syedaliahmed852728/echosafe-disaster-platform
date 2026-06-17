"""Central application settings for EchoSafe.

The project is designed to run both from the repository root and from the
Docker image at /opt/echosafe. Settings are intentionally lightweight and use
only the standard library so every module can import them early.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv(PROJECT_ROOT / ".env.development")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: _int_env("POSTGRES_PORT", 5432))
    name: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "echosafe"))
    user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "echosafe_user")
    )
    password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "echosafe_password")
    )
    pool_size: int = field(default_factory=lambda: _int_env("DB_POOL_SIZE", 5))
    max_overflow: int = field(default_factory=lambda: _int_env("DB_MAX_OVERFLOW", 10))
    pool_timeout: int = field(default_factory=lambda: _int_env("DB_POOL_TIMEOUT", 30))
    echo: bool = field(default_factory=lambda: _bool_env("SQLALCHEMY_ECHO", False))

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class SecuritySettings:
    jwt_secret: str = field(
        default_factory=lambda: os.getenv("JWT_SECRET", "dev-secret-key-change-me")
    )
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = field(
        default_factory=lambda: _int_env("JWT_ACCESS_TOKEN_MINUTES", 60)
    )
    refresh_token_days: int = field(
        default_factory=lambda: _int_env("JWT_REFRESH_TOKEN_DAYS", 7)
    )
    rate_limit_requests: int = field(
        default_factory=lambda: _int_env("RATE_LIMIT_REQUESTS", 100)
    )
    rate_limit_window: int = field(
        default_factory=lambda: _int_env("RATE_LIMIT_WINDOW_SECONDS", 60)
    )
    password_iterations: int = field(
        default_factory=lambda: _int_env("PASSWORD_HASH_ITERATIONS", 210_000)
    )


@dataclass(frozen=True)
class PipelineSettings:
    offline_mode: bool = field(
        default_factory=lambda: _bool_env("ECHOSAFE_OFFLINE_MODE", False)
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: _int_env("ECHOSAFE_REQUEST_TIMEOUT_SECONDS", 12)
    )
    max_api_retries: int = field(
        default_factory=lambda: _int_env("ECHOSAFE_MAX_API_RETRIES", 1)
    )
    xgb_n_jobs: int = field(default_factory=lambda: _int_env("ECHOSAFE_XGB_N_JOBS", 1))
    bronze_dir: Path = PROJECT_ROOT / "data" / "bronze"
    silver_dir: Path = PROJECT_ROOT / "data" / "silver"
    gold_dir: Path = PROJECT_ROOT / "data" / "gold"
    predictions_dir: Path = PROJECT_ROOT / "predictions"
    reports_dir: Path = PROJECT_ROOT / "reports"
    weather_date_start: str = field(
        default_factory=lambda: os.getenv("WEATHER_DATE_START", "2010-01-01")
    )
    weather_date_end: str = field(
        default_factory=lambda: os.getenv("WEATHER_DATE_END", "2025-12-31")
    )


@dataclass(frozen=True)
class ModelSettings:
    model_dir: Path = PROJECT_ROOT / "models"
    drift_threshold: float = field(
        default_factory=lambda: _float_env("DRIFT_THRESHOLD", 0.05)
    )
    min_f1_threshold: float = field(
        default_factory=lambda: _float_env("MIN_F1_THRESHOLD", 0.65)
    )


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    env: str = field(default_factory=lambda: os.getenv("ECHOSAFE_ENV", "development"))
    debug: bool = field(default_factory=lambda: _bool_env("ECHOSAFE_DEBUG", True))
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    model: ModelSettings = field(default_factory=ModelSettings)


SETTINGS = Settings()

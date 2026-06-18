"""Risk-engine wrappers for the Streamlit dashboard.

Each predictor here accepts the feature dictionary the dashboard already
assembles (region + current weather conditions) and returns a uniform
``{risk_level, confidence, risk_score, message}`` result. The wrappers fold
in the trained models from ``backend.predictors`` when their artifacts are on
disk, and otherwise fall back to deterministic rule-based scoring so every
dashboard page always renders a result.
"""

from backend.risk_engine.common import (
    RISK_LEVELS,
    risk_level_for_score,
    severity_from_label,
)

__all__ = ["RISK_LEVELS", "risk_level_for_score", "severity_from_label"]

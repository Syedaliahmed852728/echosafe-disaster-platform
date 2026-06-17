"""
Earthquake Magnitude Trainer.

Trains a Linear Regression magnitude estimator (wrapped in a StandardScaler
pipeline so the predictor can feed it raw inputs). Selection was empirical:
on the EchoSafe scraped USGS dataset Linear Regression beat SVR and Random
Forest on both MSE and R^2 — see Evidence/Earthquake_Magnitude_Model_Selection.ipynb
for the side-by-side comparison.

This module does NOT predict earthquake occurrence. It estimates the magnitude
of an already-detected event from its epicentre + depth + temporal context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.logger import get_logger
from config.settings import SETTINGS
from ml_models.base_trainer import BaseTrainer
from ml_models.earthquake.features import MAGNITUDE_FEATURES, MAGNITUDE_TARGET

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_MODEL_DIR = SETTINGS.model.model_dir / "earthquake_magnitude"
DEFAULT_REPORT_PATH = (
    SETTINGS.project_root / "reports" / "earthquake_magnitude_evaluation.json"
)
MODEL_FILENAME = "earthquake_magnitude_model.pkl"


class EarthquakeMagnitudeTrainer(BaseTrainer):
    """Linear Regression magnitude estimator following the BaseTrainer contract."""

    def __init__(self):
        with open(CONFIG_PATH) as fh:
            config = yaml.safe_load(fh)
        super().__init__("earthquake_magnitude", config)

    def load_data(self) -> pd.DataFrame:
        path = (
            SETTINGS.pipeline.gold_dir
            / "earthquake_risk"
            / "earthquake_risk_dataset.csv"
        )
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} events from {path.name}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        return [c for c in MAGNITUDE_FEATURES if c in df.columns]

    def _time_based_split(self, df: pd.DataFrame, feature_cols: list):
        df = df.dropna(subset=feature_cols + [MAGNITUDE_TARGET]).copy()
        if "year" in df.columns:
            sorted_years = sorted(df["year"].dropna().unique())
            if len(sorted_years) >= 2:
                cutoff = sorted_years[int(len(sorted_years) * 0.8)]
                train_mask = df["year"] < cutoff
                test_mask = df["year"] >= cutoff
                if train_mask.sum() > 100 and test_mask.sum() > 20:
                    return (
                        df.loc[train_mask, feature_cols],
                        df.loc[train_mask, MAGNITUDE_TARGET],
                        df.loc[test_mask, feature_cols],
                        df.loc[test_mask, MAGNITUDE_TARGET],
                        f"time-split (test from {cutoff} onwards)",
                    )
        # Fallback: tail-based split.
        n = len(df)
        cutoff_idx = max(1, int(n * 0.8))
        return (
            df.iloc[:cutoff_idx][feature_cols],
            df.iloc[:cutoff_idx][MAGNITUDE_TARGET],
            df.iloc[cutoff_idx:][feature_cols],
            df.iloc[cutoff_idx:][MAGNITUDE_TARGET],
            "tail-split (last 20% by row order)",
        )

    def train(self) -> Dict[str, Any]:
        df = self.load_data()
        feature_cols = self.get_feature_columns(df)
        if not feature_cols:
            raise ValueError(f"None of {MAGNITUDE_FEATURES} found in gold dataset")

        X_train, y_train, X_test, y_test, split_desc = self._time_based_split(
            df, feature_cols
        )
        logger.info(
            f"Training split: {len(X_train)} train / {len(X_test)} test [{split_desc}]"
        )

        model_cfg = self.config.get("model", {})
        # StandardScaler -> LinearRegression. Matches what the notebook tested,
        # but at prediction time the predictor can pass raw features and the
        # pipeline handles scaling.
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    LinearRegression(
                        fit_intercept=model_cfg.get("fit_intercept", True)
                    ),
                ),
            ]
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        mse = float(mean_squared_error(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        # For a linear model the "importance" surface is the scaled coefficients.
        reg = self.model.named_steps["regressor"]
        coefficients = {
            feat: round(float(c), 5) for feat, c in zip(feature_cols, reg.coef_)
        }

        self.metadata = {
            "disaster": "earthquake_magnitude",
            "model_type": "LinearRegression",
            "pipeline": "StandardScaler -> LinearRegression",
            "features": feature_cols,
            "target": MAGNITUDE_TARGET,
            "split": split_desc,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mse": round(mse, 5),
            "mae": round(mae, 5),
            "r2": round(r2, 5),
            "intercept": round(float(reg.intercept_), 5),
            "coefficients_scaled": coefficients,
            "selected_over": ["SVR_rbf", "RandomForestRegressor"],
            "selection_evidence": "Evidence/Earthquake_Magnitude_Model_Selection.ipynb",
        }
        logger.info(f"Magnitude estimator metrics: {self.metadata}")
        return self.metadata

    def save(self, output_dir: str | Path | None = None) -> None:
        out = Path(output_dir or DEFAULT_MODEL_DIR)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / MODEL_FILENAME)
        with open(out / "earthquake_magnitude_metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)
        DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_REPORT_PATH, "w") as fh:
            json.dump(self.metadata, fh, indent=2)
        logger.info(f"Magnitude model saved to {out / MODEL_FILENAME}")
        logger.info(f"Evaluation report saved to {DEFAULT_REPORT_PATH}")

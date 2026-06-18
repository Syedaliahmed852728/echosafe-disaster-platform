#!/usr/bin/env python3
"""
Earthquake Rolling Backtest.

Reuses the project's bronze -> silver -> gold pipelines to assemble a
test snapshot, then walks day-by-day through a user supplied
[start_date, end_date] window. For each day in the window the model is
retrained on every event strictly before that day and used to predict the
magnitudes of the events that occurred on that day. Days with no recorded
earthquake are skipped with a message; the trained model is never persisted.

Usage
-----
    python -m test.earthquake 2023-01-01 2023-01-07

Or programmatically:
    from test.earthquake import EarthquakeRollingTester
    results = EarthquakeRollingTester().run("2023-01-01", "2023-01-07")
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config.logger import get_logger
from backend.config.settings import PROJECT_ROOT
from backend.ml_models.earthquake.features import MAGNITUDE_FEATURES, MAGNITUDE_TARGET
from backend.ml_models.earthquake.magnitude_trainer import EarthquakeMagnitudeTrainer
from backend.ml_models.earthquake.severity_classifier import classify_severity
from backend.pipelines.bronze_to_silver_earthquake import bronze_to_silver_earthquake
from backend.pipelines.download_earthquack_data import download_earthquake_data
from backend.pipelines.silver_to_gold_earthquake import engineer_earthquake_features

logger = get_logger("test.earthquake")

TEST_DATA_DIR = PROJECT_ROOT / "test_data" / "earthquake"


class EarthquakeRollingTester(EarthquakeMagnitudeTrainer):
    """Day-by-day backtester that inherits the production magnitude trainer."""

    def __init__(self, refresh_pipelines: bool = False):
        super().__init__()
        self.refresh_pipelines = refresh_pipelines
        self.snapshot_path: Optional[Path] = None

    def _ensure_pipeline_artifacts(self) -> None:
        """Run the upstream pipelines only when their outputs are missing."""
        gold_path = (
            PROJECT_ROOT
            / "data"
            / "gold"
            / "earthquake_risk"
            / "earthquake_risk_dataset.csv"
        )
        if self.refresh_pipelines or not gold_path.exists():
            logger.info("Refreshing bronze -> silver -> gold earthquake pipelines")
            download_earthquake_data()
            bronze_to_silver_earthquake()
            engineer_earthquake_features()

    def _snapshot_to_test_data(self, df: pd.DataFrame) -> Path:
        """Copy the gold snapshot into test_data/earthquake/<minute-precision>.csv."""
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path = TEST_DATA_DIR / f"earthquake_{stamp}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Test snapshot written to {path}")
        return path

    def load_data(self) -> pd.DataFrame:
        self._ensure_pipeline_artifacts()
        df = super().load_data()
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["event_time"]).reset_index(drop=True)
        if self.snapshot_path is None:
            self.snapshot_path = self._snapshot_to_test_data(df)
        return df

    def _fit_on(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Fit a fresh StandardScaler -> LinearRegression on the supplied frame."""
        train_df = train_df.dropna(subset=feature_cols + [MAGNITUDE_TARGET])
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    LinearRegression(
                        fit_intercept=self.config.get("model", {}).get(
                            "fit_intercept", True
                        )
                    ),
                ),
            ]
        )
        model.fit(train_df[feature_cols], train_df[MAGNITUDE_TARGET])
        self.model = model
        return {"n_train": int(len(train_df))}

    def _predict_day(
        self,
        day_df: pd.DataFrame,
        feature_cols: List[str],
        day: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        usable = day_df.dropna(subset=feature_cols + [MAGNITUDE_TARGET])
        if usable.empty:
            return []
        preds = self.model.predict(usable[feature_cols])
        rows: List[Dict[str, Any]] = []
        for (_, row), pred in zip(usable.iterrows(), preds):
            predicted = float(pred)
            actual = float(row[MAGNITUDE_TARGET])
            depth = (
                float(row["depth_km"])
                if "depth_km" in row and pd.notna(row["depth_km"])
                else 100.0
            )
            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "event_id": row.get("event_id"),
                    "event_time": str(row.get("event_time")),
                    "region": row.get("region"),
                    "province": row.get("province"),
                    "place": row.get("place"),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "depth_km": depth,
                    "in_pakistan_bbox": int(row.get("in_pakistan_bbox", 0))
                    if pd.notna(row.get("in_pakistan_bbox"))
                    else None,
                    "predicted_magnitude": round(predicted, 3),
                    "actual_magnitude": round(actual, 3),
                    "absolute_error": round(abs(predicted - actual), 3),
                    "predicted_severity": classify_severity(predicted, depth),
                    "actual_severity": classify_severity(actual, depth),
                }
            )
        return rows

    @staticmethod
    def _parse_date(value: str) -> pd.Timestamp:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d")).tz_localize("UTC")

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Walk daily from start_date to end_date (inclusive) and report results."""
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if end < start:
            raise ValueError("end_date must be on or after start_date")

        df = self.load_data()
        feature_cols = self.get_feature_columns(df)
        if not feature_cols:
            raise ValueError(f"None of {MAGNITUDE_FEATURES} found in gold dataset")

        predictions: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        day = start
        while day <= end:
            next_day = day + timedelta(days=1)
            train_df = df[df["event_time"] < day]
            day_df = df[(df["event_time"] >= day) & (df["event_time"] < next_day)]

            if train_df.empty or len(
                train_df.dropna(subset=feature_cols + [MAGNITUDE_TARGET])
            ) < 2:
                msg = (
                    f"{day.date()}: skipped — not enough events before this date "
                    "to train a model"
                )
                logger.info(msg)
                skipped.append({"date": day.strftime("%Y-%m-%d"), "reason": msg})
                day = next_day
                continue

            if day_df.empty:
                msg = f"{day.date()}: skipped — no earthquake events recorded on this date"
                logger.info(msg)
                skipped.append({"date": day.strftime("%Y-%m-%d"), "reason": msg})
                day = next_day
                continue

            fit_info = self._fit_on(train_df, feature_cols)
            day_rows = self._predict_day(day_df, feature_cols, day)
            if not day_rows:
                msg = (
                    f"{day.date()}: skipped — events present but feature columns are "
                    "incomplete"
                )
                logger.info(msg)
                skipped.append({"date": day.strftime("%Y-%m-%d"), "reason": msg})
                day = next_day
                continue

            logger.info(
                f"{day.date()}: trained on {fit_info['n_train']} events, "
                f"predicted {len(day_rows)} event(s)"
            )
            predictions.extend(day_rows)
            day = next_day

        summary = self._summarise(predictions)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "snapshot_csv": str(self.snapshot_path) if self.snapshot_path else None,
            "features": feature_cols,
            "predictions": predictions,
            "skipped_days": skipped,
            "summary": summary,
        }

    @staticmethod
    def _summarise(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not predictions:
            return {"n_predictions": 0}
        pred = [p["predicted_magnitude"] for p in predictions]
        actual = [p["actual_magnitude"] for p in predictions]
        return {
            "n_predictions": len(predictions),
            "mae": round(float(mean_absolute_error(actual, pred)), 4),
            "mse": round(float(mean_squared_error(actual, pred)), 4),
            "r2": round(float(r2_score(actual, pred)), 4)
            if len(predictions) > 1
            else None,
        }


def run_rolling_test(
    start_date: str,
    end_date: str,
    refresh_pipelines: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper mirroring the predictors module style."""
    return EarthquakeRollingTester(refresh_pipelines=refresh_pipelines).run(
        start_date, end_date
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m test.earthquake <start_date YYYY-MM-DD> "
              "<end_date YYYY-MM-DD> [--refresh]")
        sys.exit(1)
    refresh = "--refresh" in sys.argv[3:]
    result = run_rolling_test(sys.argv[1], sys.argv[2], refresh_pipelines=refresh)
    import json

    print(json.dumps(result, indent=2, default=str))

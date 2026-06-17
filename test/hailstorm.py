#!/usr/bin/env python3
"""
Hailstorm Rolling Backtest.

Mirrors test/earthquake.py for the hailstorm pipeline. Reuses the bronze ->
silver -> gold pipelines to assemble a snapshot, then walks day-by-day
through a user supplied [start_date, end_date] window. For each day the
Random Forest hail-day classifier is retrained on every row strictly before
that day and used to predict the station-days that occurred on that day.
Days with no station-day rows are skipped with a message; the trained model
is never persisted.

Usage
-----
    python -m test.hailstorm 2024-04-01 2024-04-05

Or programmatically:
    from test.hailstorm import HailstormRollingTester
    result = HailstormRollingTester().run("2024-04-01", "2024-04-05")
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.logger import get_logger
from config.settings import PROJECT_ROOT
from ml_models.hailstorm.features import HAIL_FEATURES, HAIL_TARGET
from ml_models.hailstorm.risk_trainer import HailstormRiskTrainer, build_pipeline
from ml_models.hailstorm.severity_classifier import classify_hail_severity
from pipelines.bronze_to_silver_hailstorm import bronze_to_silver_hailstorm
from pipelines.download_hailstorm_data import download_hailstorm_data  # noqa: F401
from pipelines.silver_to_gold_hailstorm import engineer_hailstorm_features

logger = get_logger("test.hailstorm")

TEST_DATA_DIR = PROJECT_ROOT / "test_data" / "hailstorm"


class HailstormRollingTester(HailstormRiskTrainer):
    """Day-by-day backtester that inherits the production risk trainer."""

    def __init__(self, refresh_pipelines: bool = False):
        super().__init__()
        self.refresh_pipelines = refresh_pipelines
        self.snapshot_path: Optional[Path] = None

    def _ensure_pipeline_artifacts(self) -> None:
        gold_path = (
            PROJECT_ROOT
            / "data"
            / "gold"
            / "hailstorm_risk"
            / "hailstorm_risk_dataset.csv"
        )
        if self.refresh_pipelines or not gold_path.exists():
            logger.info("Refreshing bronze -> silver -> gold hailstorm pipelines")
            from pipelines.utils.hailstorm_client import download_hailstorm_data as dl

            dl()
            bronze_to_silver_hailstorm()
            engineer_hailstorm_features()

    def _snapshot_to_test_data(self, df: pd.DataFrame) -> Path:
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path = TEST_DATA_DIR / f"hailstorm_{stamp}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Test snapshot written to {path}")
        return path

    def load_data(self) -> pd.DataFrame:
        self._ensure_pipeline_artifacts()
        df = super().load_data()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).reset_index(drop=True)
        if self.snapshot_path is None:
            self.snapshot_path = self._snapshot_to_test_data(df)
        return df

    def _fit_on(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        train_df = train_df.dropna(subset=feature_cols + [HAIL_TARGET])
        n_pos = int(train_df[HAIL_TARGET].astype(int).sum())
        if n_pos < 2:
            raise ValueError(
                f"Need at least 2 positive examples to fit SMOTE; got {n_pos}"
            )
        self.model = build_pipeline(
            self.config.get("model", {}),
            self.config.get("resampler", {}),
            n_minority=n_pos,
        )
        self.model.fit(train_df[feature_cols], train_df[HAIL_TARGET].astype(int))
        return {
            "n_train": int(len(train_df)),
            "n_train_positives": n_pos,
            "positive_rate_train": round(float(train_df[HAIL_TARGET].mean()), 6),
        }

    def _predict_day(
        self,
        day_df: pd.DataFrame,
        feature_cols: List[str],
        day: pd.Timestamp,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        usable = day_df.dropna(subset=feature_cols)
        if usable.empty:
            return []
        probs = self.model.predict_proba(usable[feature_cols])[:, 1]
        rows: List[Dict[str, Any]] = []
        for (_, row), proba in zip(usable.iterrows(), probs):
            predicted_label = int(proba >= threshold)
            actual_label = int(row.get(HAIL_TARGET, 0))
            gust = float(row.get("wind_gust_max_ms", 0.0))
            thunder = int(row.get("thunder_hours", 0))
            precip = float(row.get("precipitation_sum_mm", 0.0))
            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "event_id": row.get("event_id"),
                    "icao": row.get("icao"),
                    "station_name": row.get("station_name"),
                    "region": row.get("region"),
                    "province": row.get("province"),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "temperature_max_c": float(row.get("temperature_max_c", 0.0)),
                    "dew_point_mean_c": float(row.get("dew_point_mean_c", 0.0)),
                    "rh_min_pct": float(row.get("rh_min_pct", 0.0)),
                    "wind_gust_max_ms": gust,
                    "wind_speed_mean_ms": float(row.get("wind_speed_mean_ms", 0.0)),
                    "surface_pressure_min_hpa": float(
                        row.get("surface_pressure_min_hpa", 0.0)
                    ),
                    "surface_pressure_drop_hpa": float(
                        row.get("surface_pressure_drop_hpa", 0.0)
                    ),
                    "precipitation_sum_mm": precip,
                    "thunder_hours": thunder,
                    "predicted_probability": round(float(proba), 4),
                    "predicted_hail": bool(predicted_label),
                    "actual_hail": bool(actual_label),
                    "correct": predicted_label == actual_label,
                    "predicted_severity": classify_hail_severity(
                        gust, thunder, precip, observed=bool(predicted_label)
                    ),
                    "actual_severity": (
                        str(row.get("hail_severity_label"))
                        if pd.notna(row.get("hail_severity_label"))
                        else None
                    ),
                    "wxcodes_examples": str(row.get("wxcodes_examples", ""))
                    if pd.notna(row.get("wxcodes_examples", ""))
                    else "",
                }
            )
        return rows

    @staticmethod
    def _parse_date(value: str) -> pd.Timestamp:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d")).tz_localize("UTC")

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if end < start:
            raise ValueError("end_date must be on or after start_date")

        df = self.load_data()
        feature_cols = self.get_feature_columns(df)
        if not feature_cols:
            raise ValueError(f"None of {HAIL_FEATURES} found in gold dataset")

        # Trainer config stores `decision_threshold: auto`; the value tuned at
        # training time is stored in metadata. Fall back to fallback_threshold
        # (default 0.5) if no metadata is available.
        train_cfg = self.config.get("training", {})
        threshold_cfg = train_cfg.get("decision_threshold", 0.5)
        if isinstance(threshold_cfg, str) and threshold_cfg.lower() == "auto":
            threshold = float(
                self.metadata.get(
                    "decision_threshold", train_cfg.get("fallback_threshold", 0.5)
                )
            )
        else:
            threshold = float(threshold_cfg)

        predictions: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        day = start
        while day <= end:
            next_day = day + timedelta(days=1)
            train_df = df[df["date"] < day]
            day_df = df[(df["date"] >= day) & (df["date"] < next_day)]

            n_pos_pre = int(train_df[HAIL_TARGET].astype(int).sum()) if not train_df.empty else 0
            if train_df.empty or n_pos_pre < 2:
                msg = (
                    f"{day.date()}: skipped — only {n_pos_pre} positive day(s) before "
                    "this date; SMOTE needs at least 2"
                )
                logger.info(msg)
                skipped.append({"date": day.strftime("%Y-%m-%d"), "reason": msg})
                day = next_day
                continue

            if day_df.empty:
                msg = (
                    f"{day.date()}: skipped — no station-day rows recorded on this date"
                )
                logger.info(msg)
                skipped.append({"date": day.strftime("%Y-%m-%d"), "reason": msg})
                day = next_day
                continue

            fit_info = self._fit_on(train_df, feature_cols)
            day_rows = self._predict_day(day_df, feature_cols, day, threshold)
            if not day_rows:
                msg = (
                    f"{day.date()}: skipped — rows present but feature columns are "
                    "incomplete"
                )
                logger.info(msg)
                skipped.append({"date": day.strftime("%Y-%m-%d"), "reason": msg})
                day = next_day
                continue

            n_pos = sum(1 for r in day_rows if r["actual_hail"])
            logger.info(
                f"{day.date()}: trained on {fit_info['n_train']} rows "
                f"({fit_info['n_train_positives']} positives, "
                f"rate {fit_info['positive_rate_train']:.4%}), "
                f"predicted {len(day_rows)} station-day(s), {n_pos} actual hail"
            )
            predictions.extend(day_rows)
            day = next_day

        summary = self._summarise(predictions)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "snapshot_csv": str(self.snapshot_path) if self.snapshot_path else None,
            "features": feature_cols,
            "decision_threshold": threshold,
            "predictions": predictions,
            "skipped_days": skipped,
            "summary": summary,
        }

    @staticmethod
    def _summarise(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not predictions:
            return {"n_predictions": 0}
        actual = [int(p["actual_hail"]) for p in predictions]
        probs = [p["predicted_probability"] for p in predictions]
        preds = [int(p["predicted_hail"]) for p in predictions]
        unique_classes = len(set(actual))
        return {
            "n_predictions": len(predictions),
            "n_actual_hail": int(sum(actual)),
            "n_predicted_hail": int(sum(preds)),
            "accuracy": round(
                sum(int(p["correct"]) for p in predictions) / len(predictions),
                4,
            ),
            "roc_auc": round(float(roc_auc_score(actual, probs)), 4)
            if unique_classes > 1
            else None,
            "average_precision": round(
                float(average_precision_score(actual, probs)), 4
            )
            if unique_classes > 1
            else None,
            "f1_positive": round(float(f1_score(actual, preds, zero_division=0)), 4),
            "precision_positive": round(
                float(precision_score(actual, preds, zero_division=0)), 4
            ),
            "recall_positive": round(
                float(recall_score(actual, preds, zero_division=0)), 4
            ),
        }


def run_rolling_test(
    start_date: str,
    end_date: str,
    refresh_pipelines: bool = False,
) -> Dict[str, Any]:
    return HailstormRollingTester(refresh_pipelines=refresh_pipelines).run(
        start_date, end_date
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m test.hailstorm <start_date YYYY-MM-DD> "
              "<end_date YYYY-MM-DD> [--refresh]")
        sys.exit(1)
    refresh = "--refresh" in sys.argv[3:]
    result = run_rolling_test(sys.argv[1], sys.argv[2], refresh_pipelines=refresh)
    import json

    print(json.dumps(result, indent=2, default=str))

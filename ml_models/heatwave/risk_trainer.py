"""
Heatwave Risk Trainer.

Trains a binary heatwave-day classifier on the gold region-day dataset.
Production default: Random Forest. The trainer also supports an optional
SMOTE step (driven by `resampler.type: smote` in config.yaml) so the same
machinery can be reused if a future re-label sharpens the imbalance.

Decision threshold is auto-tuned on a chronological validation slice using
Youden's J (TPR - FPR), mirroring the hailstorm trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from config.logger import get_logger
from config.settings import SETTINGS
from ml_models.base_trainer import BaseTrainer
from ml_models.heatwave.features import HEATWAVE_FEATURES, HEATWAVE_TARGET

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_MODEL_DIR = SETTINGS.model.model_dir / "heatwave_risk"
DEFAULT_REPORT_PATH = (
    SETTINGS.project_root / "reports" / "heatwave_risk_evaluation.json"
)
MODEL_FILENAME = "heatwave_risk_model.pkl"


def _build_classifier(model_cfg: Dict[str, Any]):
    mtype = str(model_cfg.get("type", "RandomForestClassifier"))
    if mtype == "LogisticRegression":
        return LogisticRegression(
            max_iter=int(model_cfg.get("max_iter", 2000)),
            class_weight=model_cfg.get("class_weight", "balanced"),
            C=float(model_cfg.get("C", 1.0)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
    return RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 400)),
        max_depth=model_cfg.get("max_depth", None),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 4)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=int(model_cfg.get("random_state", 42)),
        n_jobs=-1,
    )


def build_pipeline(
    model_cfg: Dict[str, Any],
    resampler_cfg: Dict[str, Any],
    n_minority: int,
):
    """Construct the training pipeline. Adds SMOTE only when configured."""
    classifier = _build_classifier(model_cfg)
    steps = [("scaler", StandardScaler()), ("classifier", classifier)]
    sampler_type = str(resampler_cfg.get("type", "none")).lower()
    if sampler_type == "smote":
        k_cfg = int(resampler_cfg.get("k_neighbors", 5))
        k = max(1, min(k_cfg, n_minority - 1))
        smote = SMOTE(
            random_state=int(resampler_cfg.get("random_state", 42)),
            k_neighbors=k,
            sampling_strategy=resampler_cfg.get("sampling_strategy", "auto"),
        )
        return ImbPipeline(steps=[("smote", smote), *steps])
    return SkPipeline(steps=steps)


class HeatwaveRiskTrainer(BaseTrainer):
    """Heatwave-day classifier following the BaseTrainer contract."""

    def __init__(self):
        with open(CONFIG_PATH) as fh:
            config = yaml.safe_load(fh)
        super().__init__("heatwave_risk", config)

    def load_data(self) -> pd.DataFrame:
        path = (
            SETTINGS.pipeline.gold_dir
            / "heatwave_risk"
            / "heatwave_risk_dataset.csv"
        )
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} region-days from {path.name}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        return [c for c in HEATWAVE_FEATURES if c in df.columns]

    def _time_based_split(self, df: pd.DataFrame, feature_cols: list):
        df = df.dropna(subset=feature_cols + [HEATWAVE_TARGET]).copy()
        if "year" in df.columns:
            sorted_years = sorted(df["year"].dropna().unique())
            if len(sorted_years) >= 2:
                cutoff = sorted_years[int(len(sorted_years) * 0.8)]
                train_mask = df["year"] < cutoff
                test_mask = df["year"] >= cutoff
                if train_mask.sum() > 100 and test_mask.sum() > 20:
                    return (
                        df.loc[train_mask, feature_cols],
                        df.loc[train_mask, HEATWAVE_TARGET],
                        df.loc[test_mask, feature_cols],
                        df.loc[test_mask, HEATWAVE_TARGET],
                        f"time-split (test from {cutoff} onwards)",
                    )
        n = len(df)
        cutoff_idx = max(1, int(n * 0.8))
        return (
            df.iloc[:cutoff_idx][feature_cols],
            df.iloc[:cutoff_idx][HEATWAVE_TARGET],
            df.iloc[cutoff_idx:][feature_cols],
            df.iloc[cutoff_idx:][HEATWAVE_TARGET],
            "tail-split (last 20% by row order)",
        )

    def _tune_threshold(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        model_cfg: Dict[str, Any],
        resampler_cfg: Dict[str, Any],
    ) -> float:
        train_cfg = self.config.get("training", {})
        fallback = float(train_cfg.get("fallback_threshold", 0.5))
        val_frac = float(train_cfg.get("validation_fraction", 0.2))

        df = df.dropna(subset=feature_cols + [HEATWAVE_TARGET]).copy()
        if "year" not in df.columns or df[HEATWAVE_TARGET].sum() < 5:
            return fallback

        years = sorted(df["year"].unique())
        if len(years) < 2:
            return fallback
        val_cutoff = years[int(len(years) * (1.0 - val_frac))]
        train_inner = df[df["year"] < val_cutoff]
        val_inner = df[df["year"] >= val_cutoff]
        if train_inner[HEATWAVE_TARGET].sum() < 5 or val_inner[HEATWAVE_TARGET].sum() == 0:
            return fallback

        pipe = build_pipeline(
            model_cfg, resampler_cfg,
            n_minority=int(train_inner[HEATWAVE_TARGET].sum()),
        )
        pipe.fit(train_inner[feature_cols], train_inner[HEATWAVE_TARGET].astype(int))
        proba = pipe.predict_proba(val_inner[feature_cols])[:, 1]
        y_val = val_inner[HEATWAVE_TARGET].astype(int).values

        fpr, tpr, ts = roc_curve(y_val, proba)
        valid = (ts <= 1.0) & np.isfinite(ts)
        if not valid.any():
            return fallback
        j = tpr[valid] - fpr[valid]
        ts_v = ts[valid]
        best_i = int(np.argmax(j))
        threshold = max(float(ts_v[best_i]), 0.05)
        logger.info(
            f"Threshold tuned on validation (cutoff={val_cutoff}): "
            f"t={threshold:.4f} Youden_J={j[best_i]:.4f} "
            f"TPR={tpr[valid][best_i]:.4f} FPR={fpr[valid][best_i]:.4f}"
        )
        return threshold

    def train(self) -> Dict[str, Any]:
        df = self.load_data()
        feature_cols = self.get_feature_columns(df)
        if not feature_cols:
            raise ValueError(f"None of {HEATWAVE_FEATURES} found in gold dataset")

        X_train, y_train, X_test, y_test, split_desc = self._time_based_split(
            df, feature_cols
        )
        n_pos_train = int(y_train.astype(int).sum())
        n_pos_test = int(y_test.astype(int).sum())
        logger.info(
            f"Training split: {len(X_train)} train / {len(X_test)} test "
            f"({n_pos_train} positives in train; {n_pos_test} in test) "
            f"[{split_desc}]"
        )
        if n_pos_train < 5:
            raise ValueError(
                f"Not enough heatwave-day positives in train ({n_pos_train}); "
                "expand the climatology window or reduce the PMD threshold."
            )

        model_cfg = self.config.get("model", {})
        resampler_cfg = self.config.get("resampler", {})
        train_cfg = self.config.get("training", {})

        threshold_cfg = train_cfg.get("decision_threshold", 0.5)
        if isinstance(threshold_cfg, str) and threshold_cfg.lower() == "auto":
            df_train = df.loc[X_train.index]
            threshold = self._tune_threshold(df_train, feature_cols, model_cfg, resampler_cfg)
        else:
            threshold = float(threshold_cfg)

        self.model = build_pipeline(model_cfg, resampler_cfg, n_minority=n_pos_train)
        self.model.fit(X_train, y_train.astype(int))

        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        roc_auc = (
            float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else None
        )
        avg_prec = float(average_precision_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))

        clf = self.model.named_steps["classifier"]
        importances: Dict[str, float] = {}
        if hasattr(clf, "feature_importances_"):
            importances = {
                feat: round(float(imp), 5)
                for feat, imp in zip(feature_cols, clf.feature_importances_)
            }
        elif hasattr(clf, "coef_"):
            importances = {
                feat: round(float(c), 5)
                for feat, c in zip(feature_cols, clf.coef_[0])
            }

        self.metadata = {
            "disaster": "heatwave_risk",
            "model_type": str(model_cfg.get("type", "RandomForestClassifier")),
            "pipeline": (
                ("SMOTE -> " if str(resampler_cfg.get("type", "none")).lower() == "smote" else "")
                + "StandardScaler -> "
                + str(model_cfg.get("type", "RandomForestClassifier"))
            ),
            "resampler": resampler_cfg.get("type", "none"),
            "features": feature_cols,
            "target": HEATWAVE_TARGET,
            "split": split_desc,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate_train": round(float(y_train.astype(int).mean()), 5),
            "positive_rate_test": round(float(y_test.astype(int).mean()), 5),
            "decision_threshold": round(threshold, 4),
            "roc_auc": round(roc_auc, 5) if roc_auc is not None else None,
            "average_precision": round(avg_prec, 5),
            "f1": round(f1, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "feature_importances": importances,
            "selected_over": [
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "RandomUnderSampler",
                "NoResample",
            ],
            "selection_evidence": "Evidence/heatwave_risk_ML.ipynb",
        }
        logger.info(f"Heatwave risk metrics: {self.metadata}")
        return self.metadata

    def save(self, output_dir: str | Path | None = None) -> None:
        out = Path(output_dir or DEFAULT_MODEL_DIR)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / MODEL_FILENAME)
        with open(out / "heatwave_risk_metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)
        DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_REPORT_PATH, "w") as fh:
            json.dump(self.metadata, fh, indent=2)
        logger.info(f"Heatwave risk model saved to {out / MODEL_FILENAME}")
        logger.info(f"Evaluation report saved to {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    trainer = HeatwaveRiskTrainer()
    trainer.train()
    trainer.save()

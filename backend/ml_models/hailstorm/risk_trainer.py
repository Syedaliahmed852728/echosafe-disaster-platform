"""
Hailstorm Risk Trainer.

Trains a SMOTE-balanced Logistic Regression hail-day classifier (binary,
target ``hail_observed``) on the gold station-day dataset.

Selection rationale: see Evidence/hailstorm_risk_ML.ipynb. SMOTE-balanced
LogReg beat Random Forest and Gradient Boosting on ROC AUC and average
precision under extreme class imbalance (~0.025% positive rate).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from backend.config.logger import get_logger
from backend.config.settings import SETTINGS
from backend.ml_models.base_trainer import BaseTrainer
from backend.ml_models.hailstorm.features import HAIL_FEATURES, HAIL_TARGET

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_MODEL_DIR = SETTINGS.model.model_dir / "hailstorm_risk"
DEFAULT_REPORT_PATH = (
    SETTINGS.project_root / "reports" / "hailstorm_risk_evaluation.json"
)
MODEL_FILENAME = "hailstorm_risk_model.pkl"


def build_pipeline(model_cfg: Dict[str, Any], resampler_cfg: Dict[str, Any],
                   n_minority: int) -> ImbPipeline:
    """Construct the imblearn pipeline: SMOTE -> Scaler -> LogReg.

    `n_minority` is the number of positive samples available for fit; SMOTE
    requires k_neighbors <= n_minority - 1.
    """
    k_cfg = int(resampler_cfg.get("k_neighbors", 5))
    k = max(1, min(k_cfg, n_minority - 1))
    sampler = SMOTE(
        random_state=int(resampler_cfg.get("random_state", 42)),
        k_neighbors=k,
        sampling_strategy=resampler_cfg.get("sampling_strategy", "auto"),
    )
    clf = LogisticRegression(
        max_iter=int(model_cfg.get("max_iter", 2000)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        C=float(model_cfg.get("C", 1.0)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    return ImbPipeline(
        steps=[
            ("smote", sampler),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ]
    )


class HailstormRiskTrainer(BaseTrainer):
    """SMOTE-balanced LogReg hail-day classifier."""

    def __init__(self):
        with open(CONFIG_PATH) as fh:
            config = yaml.safe_load(fh)
        super().__init__("hailstorm_risk", config)

    def load_data(self) -> pd.DataFrame:
        path = (
            SETTINGS.pipeline.gold_dir
            / "hailstorm_risk"
            / "hailstorm_risk_dataset.csv"
        )
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} station-days from {path.name}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        return [c for c in HAIL_FEATURES if c in df.columns]

    def _time_based_split(self, df: pd.DataFrame, feature_cols: list):
        df = df.dropna(subset=feature_cols + [HAIL_TARGET]).copy()
        if "year" in df.columns:
            sorted_years = sorted(df["year"].dropna().unique())
            if len(sorted_years) >= 2:
                cutoff = sorted_years[int(len(sorted_years) * 0.8)]
                train_mask = df["year"] < cutoff
                test_mask = df["year"] >= cutoff
                if train_mask.sum() > 100 and test_mask.sum() > 20:
                    return (
                        df.loc[train_mask, feature_cols],
                        df.loc[train_mask, HAIL_TARGET],
                        df.loc[test_mask, feature_cols],
                        df.loc[test_mask, HAIL_TARGET],
                        f"time-split (test from {cutoff} onwards)",
                    )
        n = len(df)
        cutoff_idx = max(1, int(n * 0.8))
        return (
            df.iloc[:cutoff_idx][feature_cols],
            df.iloc[:cutoff_idx][HAIL_TARGET],
            df.iloc[cutoff_idx:][feature_cols],
            df.iloc[cutoff_idx:][HAIL_TARGET],
            "tail-split (last 20% by row order)",
        )

    def _tune_threshold(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        model_cfg: Dict[str, Any],
        resampler_cfg: Dict[str, Any],
    ) -> float:
        """Tune the decision threshold on a held-out validation slice.

        The training set itself is split chronologically — the trailing
        ``validation_fraction`` (by year) is scored. The threshold is picked
        to maximise **Youden's J = TPR - FPR**, which is the right target
        under extreme class imbalance: F1 collapses to t=0 when positives
        are tiny, but Youden's J is a balanced-ranking metric that picks the
        operating point where the model best separates the two classes.
        Returns the fallback threshold if the validation fold is empty.
        """
        train_cfg = self.config.get("training", {})
        fallback = float(train_cfg.get("fallback_threshold", 0.5))
        val_frac = float(train_cfg.get("validation_fraction", 0.2))

        df = df.dropna(subset=feature_cols + [HAIL_TARGET]).copy()
        if "year" not in df.columns or df[HAIL_TARGET].sum() < 2:
            return fallback

        years = sorted(df["year"].unique())
        if len(years) < 2:
            return fallback
        val_cutoff = years[int(len(years) * (1.0 - val_frac))]
        train_inner = df[df["year"] < val_cutoff]
        val_inner = df[df["year"] >= val_cutoff]
        if train_inner[HAIL_TARGET].sum() < 2 or val_inner[HAIL_TARGET].sum() == 0:
            return fallback

        pipe = build_pipeline(
            model_cfg, resampler_cfg, n_minority=int(train_inner[HAIL_TARGET].sum())
        )
        pipe.fit(train_inner[feature_cols], train_inner[HAIL_TARGET].astype(int))
        proba = pipe.predict_proba(val_inner[feature_cols])[:, 1]
        y_val = val_inner[HAIL_TARGET].astype(int).values

        fpr, tpr, ts = roc_curve(y_val, proba)
        # Skip the synthetic 0/1 endpoints that roc_curve adds and any
        # degenerate threshold > 1 it sometimes emits.
        valid = (ts <= 1.0) & np.isfinite(ts)
        if not valid.any():
            return fallback
        j = tpr[valid] - fpr[valid]
        ts_v = ts[valid]
        best_i = int(np.argmax(j))
        threshold = float(ts_v[best_i])
        # Floor at a small but non-trivial value so we never pick a
        # near-zero threshold (which would degenerate to "predict everything").
        threshold = max(threshold, 0.05)
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
            raise ValueError(f"None of {HAIL_FEATURES} found in gold dataset")

        X_train, y_train, X_test, y_test, split_desc = self._time_based_split(
            df, feature_cols
        )
        n_pos_train = int(y_train.astype(int).sum())
        logger.info(
            f"Training split: {len(X_train)} train / {len(X_test)} test "
            f"({n_pos_train} positives in train; {int(y_test.sum())} in test) "
            f"[{split_desc}]"
        )
        if n_pos_train < 2:
            raise ValueError(
                f"Not enough positives in training set (got {n_pos_train}); "
                "cannot fit SMOTE."
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
        coefficients = {
            feat: round(float(c), 5)
            for feat, c in zip(feature_cols, clf.coef_[0])
        }

        self.metadata = {
            "disaster": "hailstorm_risk",
            "model_type": "LogisticRegression",
            "pipeline": "SMOTE -> StandardScaler -> LogisticRegression",
            "resampler": "SMOTE",
            "features": feature_cols,
            "target": HAIL_TARGET,
            "split": split_desc,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate_train": round(float(y_train.astype(int).mean()), 6),
            "positive_rate_test": round(float(y_test.astype(int).mean()), 6),
            "decision_threshold": round(threshold, 4),
            "roc_auc": round(roc_auc, 5) if roc_auc is not None else None,
            "average_precision": round(avg_prec, 5),
            "f1": round(f1, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "coefficients": coefficients,
            "intercept": round(float(clf.intercept_[0]), 5),
            "selected_over": [
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "RandomOverSampler",
                "ADASYN",
                "SMOTETomek",
                "SMOTEENN",
                "RandomUnderSampler",
            ],
            "selection_evidence": "Evidence/hailstorm_risk_ML.ipynb",
        }
        logger.info(f"Hailstorm risk metrics: {self.metadata}")
        return self.metadata

    def save(self, output_dir: str | Path | None = None) -> None:
        out = Path(output_dir or DEFAULT_MODEL_DIR)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / MODEL_FILENAME)
        with open(out / "hailstorm_risk_metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)
        DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_REPORT_PATH, "w") as fh:
            json.dump(self.metadata, fh, indent=2)
        logger.info(f"Hailstorm risk model saved to {out / MODEL_FILENAME}")
        logger.info(f"Evaluation report saved to {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    trainer = HailstormRiskTrainer()
    trainer.train()
    trainer.save()

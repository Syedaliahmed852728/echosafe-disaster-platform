"""
Abstract base trainer for all disaster ML models.
Production pattern: shared training logic, disaster-specific configs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseTrainer(ABC):
    """Base class for disaster model trainers."""

    def __init__(self, disaster_type: str, config: Dict[str, Any]):
        self.disaster_type = disaster_type
        self.config = config
        self.model = None
        self.label_encoder = None
        self.metadata = {}

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load and return the Gold dataset."""
        pass

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        pass

    @abstractmethod
    def assign_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign risk labels."""
        pass

    @abstractmethod
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Return list of feature column names."""
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train the model and return metrics."""
        pass

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """Save model artifacts."""
        pass

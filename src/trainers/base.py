"""Base class for per-model trainers: build, train, and save one sklearn model."""

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Lifecycle for a single model: build, train, save."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def build(self):
        """Return a fresh, unfitted estimator. Implemented by each subclass."""
        raise NotImplementedError

    def train(self, X, y):
        logger.info(f"Training {self.model_name}...")
        self.model = self.build()
        self.model.fit(X, y)
        return self.model

    def save(self, models_dir):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"{self.model_name}.pkl"
        joblib.dump(self.model, path)
        logger.info(f"Saved {self.model_name} to {path}")
        return path

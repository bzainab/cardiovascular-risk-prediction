"""Gradient Boosting trainer."""

from sklearn.ensemble import GradientBoostingClassifier

from src.trainers.base import BaseTrainer


class GradientBoostingTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("Gradient_Boosting")

    def build(self):
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
        )

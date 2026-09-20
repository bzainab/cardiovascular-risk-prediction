"""Logistic Regression trainer (balanced class weighting)."""

from sklearn.linear_model import LogisticRegression

from src.trainers.base import BaseTrainer


class LogisticRegressionTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("Logistic_Regression")

    def build(self):
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
            penalty="l2",
        )

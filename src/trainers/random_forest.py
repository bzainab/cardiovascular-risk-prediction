"""Random Forest trainer (cost-sensitive)."""

from sklearn.ensemble import RandomForestClassifier

from src.trainers.base import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("Random_Forest")

    def build(self):
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

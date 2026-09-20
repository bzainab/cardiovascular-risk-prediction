"""SVM (RBF kernel) trainer."""

from sklearn.svm import SVC

from src.trainers.base import BaseTrainer


class SVMTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("SVM")

    def build(self):
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

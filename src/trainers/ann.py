"""Artificial Neural Network trainer (sklearn MLPClassifier)."""

from sklearn.neural_network import MLPClassifier

from src.trainers.base import BaseTrainer


class ANNTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("ANN")

    def build(self):
        return MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            batch_size=32,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )

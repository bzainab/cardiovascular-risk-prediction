import logging

import xgboost

from src.common import positive_class_weight
from src.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)


class XGBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__("XGBoost")
        self.scale_positive_weight = None

    def build(self):
        weight = self.scale_positive_weight
        if weight is None:
            weight = 1.0
        return xgboost.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=weight,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def train(self, X, y):
        self.scale_positive_weight = positive_class_weight(y)
        logger.info(
            f"{self.model_name}: scale_pos_weight = {self.scale_positive_weight:.3f}"
        )
        return super().train(X, y)

"""Clinically-Informed Stacking Ensemble (CISE).

Conventional stacking trains a meta-learner on base learner predictions only.
CISE also passes clinically validated Framingham risk factors to the meta-learner
so it can learn patient-dependent trust in each base model.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost

from src.common import CLINICAL_FEATURES_FOR_META_LEARNER, positive_class_weight

logger = logging.getLogger(__name__)


def default_base_learners(labels):
    """The three base learners used by CISE: Random Forest, Gradient Boosting, XGBoost."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    learners = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
            ),
        ),
        (
            "xgb",
            xgboost.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=positive_class_weight(labels),
                eval_metric="logloss",
                n_jobs=-1,
            ),
        ),
    ]
    return learners


class ClinicallyInformedStackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacking classifier: meta-learner sees base predictions plus clinical features."""

    def __init__(
        self,
        base_estimators=None,
        meta_learner=None,
        clinical_features=None,
        cv=5,
        random_state=42,
    ):
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.clinical_features = clinical_features
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        if self.meta_learner is None:
            meta = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
                solver="lbfgs",
            )
        else:
            meta = self.meta_learner

        if self.clinical_features is None:
            clinical = list(CLINICAL_FEATURES_FOR_META_LEARNER)
        else:
            clinical = list(self.clinical_features)

        self.feature_names = list(X.columns)
        self.resolved_clinical_features = clinical
        self.clinical_indices = []
        for feature in clinical:
            self.clinical_indices.append(self.feature_names.index(feature))

        self.base_estimator_names = [
            name for name, estimator in self.base_estimators
        ]

        feature_matrix = X.to_numpy()
        labels = np.asarray(y)

        skf = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        n_samples = feature_matrix.shape[0]
        n_base = len(self.base_estimators)
        out_of_fold_predictions = np.zeros((n_samples, n_base))

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(feature_matrix, labels), start=1
        ):
            logger.info(f"CISE: out-of-fold predictions fold {fold}/{self.cv}")
            for j, (name, estimator) in enumerate(self.base_estimators):
                fold_model = clone(estimator)
                fold_model.fit(feature_matrix[train_idx], labels[train_idx])
                positive_probs = fold_model.predict_proba(feature_matrix[val_idx])[
                    :, 1
                ]
                out_of_fold_predictions[val_idx, j] = positive_probs

        self.base_models = []
        for name, estimator in self.base_estimators:
            full_model = clone(estimator)
            full_model.fit(feature_matrix, labels)
            self.base_models.append((name, full_model))

        clinical_feature_values = feature_matrix[:, self.clinical_indices]
        meta_learner_inputs = np.hstack(
            [out_of_fold_predictions, clinical_feature_values]
        )
        self.fitted_meta_learner = clone(meta)
        self.fitted_meta_learner.fit(meta_learner_inputs, labels)
        self.classes_ = np.unique(labels)

        coefficient_log = {}
        feature_names = self.meta_learner_feature_names()
        coefficients = self.fitted_meta_learner.coef_[0]
        for i, name in enumerate(feature_names):
            coefficient_log[name] = coefficients[i]
        logger.info(f"CISE meta-learner coefficients: {coefficient_log}")

        return self

    def build_meta_learner_input(self, X):
        if isinstance(X, pd.DataFrame):
            feature_matrix = X.to_numpy()
        else:
            feature_matrix = np.asarray(X)

        base_model_probabilities = []
        for name, model in self.base_models:
            positive_prob = model.predict_proba(feature_matrix)[:, 1]
            base_model_probabilities.append(positive_prob)
        base_model_probabilities = np.column_stack(base_model_probabilities)

        clinical_feature_values = feature_matrix[:, self.clinical_indices]
        return np.hstack([base_model_probabilities, clinical_feature_values])

    def predict_proba(self, X):
        meta_input = self.build_meta_learner_input(X)
        return self.fitted_meta_learner.predict_proba(meta_input)

    def predict(self, X):
        meta_input = self.build_meta_learner_input(X)
        return self.fitted_meta_learner.predict(meta_input)

    def meta_learner_feature_names(self):
        names = list(self.base_estimator_names)
        for feature in self.resolved_clinical_features:
            names.append(feature)
        return names

    @property
    def coefficient_summary(self):
        """Eight meta-learner coefficients (three base models + five clinical features)."""
        names = self.meta_learner_feature_names()
        coefficients = self.fitted_meta_learner.coef_[0].tolist()
        summary = []
        for i, name in enumerate(names):
            summary.append((name, coefficients[i]))
        return summary

    def explain_single_prediction(self, X):
        """Per-feature contribution for one patient (logistic meta-learner)."""
        meta_input = self.build_meta_learner_input(X)
        feature_names = self.meta_learner_feature_names()
        coefficients = self.fitted_meta_learner.coef_[0]
        intercept = float(self.fitted_meta_learner.intercept_[0])

        contributions = []
        total_logit = intercept
        for i, name in enumerate(feature_names):
            contribution_to_logit = float(meta_input[0, i] * coefficients[i])
            total_logit = total_logit + contribution_to_logit
            contributions.append(
                {
                    "name": name,
                    "contribution_to_logit": contribution_to_logit,
                    "scaled_input_value": float(meta_input[0, i]),
                }
            )

        probability = 1.0 / (1.0 + np.exp(-total_logit))
        for entry in contributions:
            logit_without = total_logit - entry["contribution_to_logit"]
            probability_without = 1.0 / (1.0 + np.exp(-logit_without))
            entry["change_in_probability"] = float(
                probability - probability_without
            )

        def absolute_probability_change(contribution):
            return abs(contribution["change_in_probability"])

        contributions = sorted(
            contributions, key=absolute_probability_change, reverse=True
        )

        return {
            "probability": float(probability),
            "logit": float(total_logit),
            "intercept": intercept,
            "contributions": contributions,
        }

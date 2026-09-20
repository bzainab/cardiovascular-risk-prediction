"""SMOTE-vs-baseline comparison experiment.

Two generated artifacts:

1. Test set (smote_comparison.csv): each reference model is
   re-trained with SMOTE wrapped in an imblearn Pipeline (so SMOTE only ever
   sees training data, never the test set) and scored on the same
   imbalanced test set used by the cost-sensitive baseline run.

2. Cross-validated (smote_cv_comparison.csv): stratified 5-fold CV where
   SMOTE is applied inside each fold (training only,
   not the validation portion). This is compared fold-for-fold against
   the cost-sensitive estimators to check whether SMOTE's disadvantage is a
   a consistent outcome.

Run with:
    python -m src.experiments.smote_comparison
"""

import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.common import positive_class_weight
from src.data_preprocessing import preprocess_pipeline
from src.model_evaluation import ModelEvaluator
from src.trainers.ann import ANNTrainer
from src.trainers.gradient_boosting import GradientBoostingTrainer
from src.trainers.logistic_regression import LogisticRegressionTrainer
from src.trainers.random_forest import RandomForestTrainer
from src.trainers.svm import SVMTrainer
from src.trainers.xgboost_trainer import XGBoostTrainer

logger = logging.getLogger(__name__)

CV_FOLDS = 5
RANDOM_STATE = 42


def reference_trainers():
    """The six reference trainers, in a fixed order."""
    return [
        LogisticRegressionTrainer(),
        RandomForestTrainer(),
        GradientBoostingTrainer(),
        XGBoostTrainer(),
        SVMTrainer(),
        ANNTrainer(),
    ]


def build_smote_pipelines(y_train):
    """Wrap each reference trainer's estimator in an SMOTE pipeline."""
    pipelines = {}
    for trainer in reference_trainers():
        estimator = trainer.build()
        if trainer.model_name == "XGBoost":
            # SMOTE already balances the classes, so disable the cost-sensitive
            # reweighting to avoid double-counting the minority class.
            estimator.set_params(scale_pos_weight=1.0)
        pipeline_name = f"{trainer.model_name}_SMOTE"
        pipelines[pipeline_name] = ImbPipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                ("classifier", estimator),
            ]
        )
    return pipelines


def build_baseline_estimators(y_train):
    """Cost-sensitive reference estimators (no SMOTE), matching the main run."""
    estimators = {}
    for trainer in reference_trainers():
        if trainer.model_name == "XGBoost":
            trainer.scale_positive_weight = positive_class_weight(y_train)
        estimators[trainer.model_name] = trainer.build()
    return estimators


def cross_validated_auc(estimator, X, y):
    """Mean/std stratified 5-fold ROC-AUC. SMOTE (if present) fits per fold."""
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


def run_cv_comparison(X_train, y_train, results_dir):
    """Fold-safe CV: cost-sensitive vs SMOTE ROC-AUC per reference model."""
    baseline = build_baseline_estimators(y_train)
    smote = build_smote_pipelines(y_train)

    rows = []
    for name, estimator in baseline.items():
        base_mean, base_std = cross_validated_auc(estimator, X_train, y_train)
        smote_mean, smote_std = cross_validated_auc(
            smote[f"{name}_SMOTE"], X_train, y_train
        )
        base_mean_r = round(base_mean, 4)
        base_std_r = round(base_std, 4)
        smote_mean_r = round(smote_mean, 4)
        smote_std_r = round(smote_std, 4)
        delta_r = round(smote_mean - base_mean, 4)
        rows.append(
            {
                "Model": name,
                "CV_AUC_CostSensitive": base_mean_r,
                "CV_AUC_CostSensitive_std": base_std_r,
                "CV_AUC_SMOTE": smote_mean_r,
                "CV_AUC_SMOTE_std": smote_std_r,
                "Delta_SMOTE_minus_CostSensitive": delta_r,
            }
        )
        logger.info(
            f"{name} CV AUC  cost-sensitive={base_mean_r}±{base_std_r}  "
            f"SMOTE={smote_mean_r}±{smote_std_r}  Diff={delta_r:}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "smote_cv_comparison.csv", index=False)
    return df


def run_smote_comparison(X_train, y_train, X_test, y_test, results_dir):
    pipelines = build_smote_pipelines(y_train)

    trained_pipelines = {}
    for name, pipe in pipelines.items():
        logger.info(f"Fitting {name}...")
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(trained_pipelines, X_test, y_test)
    evaluator.compare_models()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(output_path=str(results_dir / "smote_comparison.csv"))
    return evaluator


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "Framingham_Data" / "framingham.csv"
    results_dir = project_root / "results"

    X_train, X_test, y_train, y_test = preprocess_pipeline(
        data_path=str(data_path),
        preprocessor_output_path=str(project_root / "models" / "preprocessor.pkl"),
    )
    logger.info("Test-set comparison:")
    run_smote_comparison(X_train, y_train, X_test, y_test, results_dir)
    logger.info("Cross-validated comparison (SMOTE applied inside each fold):")
    run_cv_comparison(X_train, y_train, results_dir)


if __name__ == "__main__":
    main()

"""
Main training script. Trains the six reference models and the Clinically-
Informed Stacking Ensemble, evaluates everything on the held-out test set,
saves all the model pickles plus a single results CSV that the web app reads.

Run from the project root:
    python -m src.run_pipeline
"""

import logging
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import preprocess_pipeline
from src.model_evaluation import ModelEvaluator
from src.trainers.ann import ANNTrainer
from src.trainers.gradient_boosting import GradientBoostingTrainer
from src.trainers.logistic_regression import LogisticRegressionTrainer
from src.trainers.random_forest import RandomForestTrainer
from src.trainers.svm import SVMTrainer
from src.trainers.xgboost_trainer import XGBoostTrainer
from src.ensemble.stacking import (
    ClinicallyInformedStackingClassifier,
    default_base_learners,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Framingham_Data" / "framingham.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
STACK_NAME = "Clinically_Informed_Stacking_Ensemble"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(RESULTS / "pipeline.log"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger(__name__)


def train_references(X, y):
    """Train each of the six reference models and pickle them."""
    trained = {}
    trainers = [
        LogisticRegressionTrainer(),
        RandomForestTrainer(),
        GradientBoostingTrainer(),
        XGBoostTrainer(),
        SVMTrainer(),
        ANNTrainer(),
    ]
    for trainer in trainers:
        trainer.train(X, y)
        trainer.save(MODELS)
        trained[trainer.model_name] = trainer.model
    return trained


def train_stack(X, y):
    """Build CISE, fit it, pickle it, log the meta-learner coefficients."""
    log.info("Training stacking ensemble...")
    stack = ClinicallyInformedStackingClassifier(
        base_estimators=default_base_learners(y),
        meta_learner=LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        ),
        cv=5,
        random_state=42,
    )
    stack.fit(X, y)

    pickle_path = MODELS / f"{STACK_NAME}.pkl"
    joblib.dump(stack, pickle_path)
    log.info("Saved CISE -> %s", pickle_path)
    log.info("Meta-learner coefficients: %s", stack.coefficient_summary)
    return stack


if __name__ == "__main__":
    log.info("=" * 70)
    log.info("CARDIOVASCULAR RISK PREDICTION — TRAINING PIPELINE")
    log.info("=" * 70)

    log.info("[1/4] Preprocessing")
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        data_path=str(DATA),
        preprocessor_output_path=str(MODELS / "preprocessor.pkl"),
    )

    log.info("[2/4] Training reference models")
    models = train_references(X_train, y_train)

    log.info("[3/4] Training Clinically-Informed Stacking Ensemble")
    models[STACK_NAME] = train_stack(X_train, y_train)

    log.info("[4/4] Evaluating on test set")
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(models, X_test, y_test)
    evaluator.compare_models()
    evaluator.plot_roc_curves(y_test, output_path=str(RESULTS / "roc_curves.png"))
    evaluator.plot_confusion_matrices(
        y_test, output_path=str(RESULTS / "confusion_matrices.png")
    )
    evaluator.save_results(output_path=str(RESULTS / "model_results.csv"))

    print()
    print("Done.")
    print(f"  Models  -> {MODELS}")
    print(f"  Results -> {RESULTS}")

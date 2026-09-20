"""Regenerate results/confusion_matrices.png with (a), (b), ... panel labels.

Loads the existing trained models, replays predictions on the held-out test
partition, and re-runs only the confusion-matrix plotting step.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import preprocess_pipeline
from src.model_evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA = ROOT / "Framingham_Data" / "framingham.csv"
MODELS_DIR = ROOT / "models"
RESULTS_CSV = ROOT / "results" / "model_results.csv"
OUTPUT_PNG = ROOT / "results" / "confusion_matrices.png"


def main():
    log.info("Preprocessing data with the same seed used during training...")
    _, X_test, _, y_test = preprocess_pipeline(
        str(DATA),
        test_size=0.2,
        random_state=42,
    )

    log.info("Loading deployed models from %s", MODELS_DIR)
    metrics_df = pd.read_csv(RESULTS_CSV)

    evaluator = ModelEvaluator()

    for _, row in metrics_df.iterrows():
        name = row["Model"]
        pickle_path = MODELS_DIR / f"{name}.pkl"
        model = joblib.load(pickle_path)

        y_pred = model.predict(X_test)
        predicted_probabilities = model.predict_proba(X_test)[:, 1]

        evaluator.predictions[name] = y_pred
        evaluator.predicted_positive_probabilities[name] = predicted_probabilities
        evaluator.results[name] = {
            "Accuracy": float(row["Accuracy"]),
            "Precision": float(row["Precision"]),
            "Recall": float(row["Recall"]),
            "F1-Score": float(row["F1-Score"]),
            "ROC-AUC": float(row["ROC-AUC"]),
        }

    log.info("Plotting confusion matrices with panel labels...")
    evaluator.plot_confusion_matrices(y_test, output_path=str(OUTPUT_PNG))
    log.info("Wrote %s", OUTPUT_PNG)


if __name__ == "__main__":
    main()

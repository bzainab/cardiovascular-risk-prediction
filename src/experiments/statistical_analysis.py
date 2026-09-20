"""
Statistical analysis:
  - bootstrap 95% CIs on each model's test-set ROC-AUC
  - bootstrap-based p-values for pairwise AUC differences
  - Brier scores and calibration plots

Run from the project root:
    python -m src.experiments.statistical_analysis
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.data_preprocessing import preprocess_pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Framingham_Data" / "framingham.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RNG = np.random.default_rng(seed=42)
N_BOOTSTRAP = 1000

LOGISTIC_REGRESSION = "Logistic_Regression"
CISE_NAME = "Clinically_Informed_Stacking_Ensemble"


def load_test_split():
    """Re-run the preprocessor the same way the training pipeline does."""
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        data_path=str(DATA),
        preprocessor_output_path=str(MODELS / "preprocessor.pkl"),
    )
    return X_test, y_test


def get_positive_class_probabilities(model, X):
    return model.predict_proba(X)[:, 1]


def bootstrap_auc(labels, predicted_probabilities, n=N_BOOTSTRAP):
    """Return mean, lower CI, and upper CI for AUC via percentile bootstrap."""
    labels = np.asarray(labels)
    predicted_probabilities = np.asarray(predicted_probabilities)
    auc_samples = np.empty(n)
    n_samples = len(labels)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        auc_samples[i] = roc_auc_score(labels[idx], predicted_probabilities[idx])
    return (
        auc_samples.mean(),
        np.percentile(auc_samples, 2.5),
        np.percentile(auc_samples, 97.5),
        auc_samples,
    )


def bootstrap_auc_difference_p_value(
    labels, probabilities_model_a, probabilities_model_b, n=N_BOOTSTRAP
):
    """Two-sided bootstrap p-value for the difference in AUC between two models."""
    labels = np.asarray(labels)
    probabilities_model_a = np.asarray(probabilities_model_a)
    probabilities_model_b = np.asarray(probabilities_model_b)
    observed_difference = roc_auc_score(
        labels, probabilities_model_a
    ) - roc_auc_score(labels, probabilities_model_b)

    differences = np.empty(n)
    n_samples = len(labels)
    for i in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        differences[i] = roc_auc_score(
            labels[idx], probabilities_model_a[idx]
        ) - roc_auc_score(labels[idx], probabilities_model_b[idx])

    if observed_difference >= 0:
        p_value = (differences < 0).mean() * 2
    else:
        p_value = (differences > 0).mean() * 2
    return float(min(p_value, 1.0)), float(observed_difference)


def main():
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    X_test, y_test = load_test_split()
    labels = np.asarray(y_test)

    results_table = pd.read_csv(RESULTS / "model_results.csv")
    models = {}
    for _, row in results_table.iterrows():
        name = row["Model"]
        path = MODELS / f"{name}.pkl"
        models[name] = joblib.load(path)

    rows = []
    probabilities_by_model = {}
    for name, model in models.items():
        predicted_probabilities = get_positive_class_probabilities(model, X_test)
        probabilities_by_model[name] = predicted_probabilities
        mean_auc, lower_ci, upper_ci, _ = bootstrap_auc(labels, predicted_probabilities)
        brier = brier_score_loss(labels, predicted_probabilities)
        observed_auc = roc_auc_score(labels, predicted_probabilities)
        rows.append(
            {
                "Model": name,
                "AUC": round(observed_auc, 4),
                "AUC_CI_lo": round(lower_ci, 4),
                "AUC_CI_hi": round(upper_ci, 4),
                "Brier": round(brier, 4),
            }
        )
        print(
            f"  {name:42s}  AUC={observed_auc:.4f}  "
            f"95% CI [{lower_ci:.4f}, {upper_ci:.4f}]  Brier={brier:.4f}"
        )
    auc_df = (
        pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    )
    auc_df.to_csv(RESULTS / "auc_with_ci.csv", index=False)
    print(f"\nWrote: {RESULTS / 'auc_with_ci.csv'}")

    print("\nPairwise bootstrap p-values vs Logistic Regression:")
    pairwise_rows = []
    if LOGISTIC_REGRESSION in probabilities_by_model:
        logistic_probs = probabilities_by_model[LOGISTIC_REGRESSION]
        for name, predicted_probabilities in probabilities_by_model.items():
            if name == LOGISTIC_REGRESSION:
                continue
            p_value, difference = bootstrap_auc_difference_p_value(
                labels, logistic_probs, predicted_probabilities
            )
            pairwise_rows.append(
                {
                    "ModelA": LOGISTIC_REGRESSION,
                    "ModelB": name,
                    "AUC_diff": round(difference, 4),
                    "p_value": round(p_value, 4),
                }
            )
            print(f"  LR vs {name:42s}  diff={difference:+.4f}  p={p_value:.4f}")

    if CISE_NAME in probabilities_by_model:
        print("\nPairwise bootstrap p-values vs CISE:")
        cise_probabilities = probabilities_by_model[CISE_NAME]
        for name, predicted_probabilities in probabilities_by_model.items():
            if name == CISE_NAME:
                continue
            p_value, difference = bootstrap_auc_difference_p_value(
                labels, cise_probabilities, predicted_probabilities
            )
            pairwise_rows.append(
                {
                    "ModelA": CISE_NAME,
                    "ModelB": name,
                    "AUC_diff": round(difference, 4),
                    "p_value": round(p_value, 4),
                }
            )
            print(f"  CISE vs {name:42s}  diff={difference:+.4f}  p={p_value:.4f}")

    pd.DataFrame(pairwise_rows).to_csv(RESULTS / "pairwise_tests.csv", index=False)
    print(f"\nWrote: {RESULTS / 'pairwise_tests.csv'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(auc_df))
    aucs = auc_df["AUC"].to_numpy()
    lower_bounds = auc_df["AUC_CI_lo"].to_numpy()
    upper_bounds = auc_df["AUC_CI_hi"].to_numpy()
    error_lower = aucs - lower_bounds
    error_upper = upper_bounds - aucs
    ax.bar(x, aucs, yerr=[error_lower, error_upper], capsize=5, color="#2c5282", alpha=0.85)
    ax.set_xticks(x)
    tick_labels = []
    for model_name in auc_df["Model"]:
        tick_labels.append(model_name.replace("_", "\n"))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("ROC-AUC (test set)")
    ax.set_ylim(0.5, 0.8)
    ax.axhline(
        0.5, color="grey", linestyle="--", linewidth=0.8, label="Random (AUC=0.5)"
    )
    ax.set_title("Test-set AUC with 95% bootstrap confidence intervals")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart_path = RESULTS / "auc_with_ci.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Wrote: {chart_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    palette = plt.cm.tab10.colors
    for i, (name, predicted_probabilities) in enumerate(probabilities_by_model.items()):
        observed_rate, mean_predicted = calibration_curve(
            labels, predicted_probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(
            mean_predicted,
            observed_rate,
            marker="o",
            linewidth=1.4,
            color=palette[i % 10],
            label=name,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed CHD rate")
    ax.set_title("Calibration plot (10 quantile bins)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    calibration_path = RESULTS / "calibration.png"
    plt.savefig(calibration_path, dpi=150)
    plt.close()
    print(f"Wrote: {calibration_path}")

    summary = {
        "n_bootstrap": N_BOOTSTRAP,
        "models": rows,
        "pairwise": pairwise_rows,
    }
    with open(RESULTS / "statistical_tests.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {RESULTS / 'statistical_tests.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

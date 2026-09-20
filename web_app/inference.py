"""
ML inference for the cardiovascular risk prediction web app.

Loads every model listed in results/model_results.csv.
The primary model is the one with the highest ROC-AUC.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
RESULTS_CSV = ROOT / "results" / "model_results.csv"
PREPROCESSOR = MODELS_DIR / "preprocessor.pkl"

CISE_NAME = "Clinically_Informed_Stacking_Ensemble"

DISPLAY_NAMES = {
    "Logistic_Regression": "Logistic Regression",
    "Random_Forest": "Random Forest",
    "Gradient_Boosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "SVM": "SVM",
    "ANN": "ANN",
    "Clinically_Informed_Stacking_Ensemble": "Clinically-Informed Stacking Ensemble",
}

PRETTY = {
    "rf": "Random Forest base prediction",
    "gb": "Gradient Boosting base prediction",
    "xgb": "XGBoost base prediction",
    "age": "Age",
    "sysBP": "Systolic Blood Pressure",
    "totChol": "Total Cholesterol",
    "currentSmoker": "Current Smoker",
    "diabetes": "Diabetes",
}


def display_name(model_key):
    return DISPLAY_NAMES.get(model_key, model_key.replace("_", " "))


def load_preprocessor():
    """Load imputer, scaler and feature names from preprocessor.pkl."""
    pre = joblib.load(PREPROCESSOR)
    scaler = pre.get("scaler")
    imputer = pre.get("imputer")
    feature_names = pre.get("feature_names") or [
        "male",
        "age",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
        "edu_1.0",
        "edu_2.0",
        "edu_3.0",
        "edu_4.0",
    ]
    return scaler, imputer, feature_names


def load_models():
    """Read results CSV and load every matching pickle."""
    models = {}
    metrics = {}
    if not RESULTS_CSV.exists():
        log.error(
            "Results CSV not found at %s — train the pipeline first.", RESULTS_CSV
        )
        return models, metrics

    df = pd.read_csv(RESULTS_CSV)
    for _, row in df.iterrows():
        name = row["Model"]
        path = MODELS_DIR / f"{name}.pkl"
        models[name] = joblib.load(path)
        metrics[name] = {
            "auc": float(row["ROC-AUC"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1-Score"]),
            "accuracy": float(row["Accuracy"]),
        }
        log.info("Loaded %s", name)
    return models, metrics


def primary_model_name(metrics):
    if not metrics:
        return None
    return max(metrics, key=lambda name: metrics[name]["auc"])


def risk_band(probability, threshold=0.5):
    if probability < threshold:
        return "Low", "success"
    if probability < 0.7:
        return "Moderate", "warning"
    return "High", "danger"


def make_features(req, scaler, imputer, feature_names):
    """Turn patient dict into a one-row scaled DataFrame for the models."""
    male = int(req.get("male", 0))
    age = float(req.get("age", 50))
    education = int(req.get("education", 1))
    current_smoker = int(req.get("currentSmoker", 0))
    cigs_per_day = float(req.get("cigsPerDay", 0))
    bp_meds = int(req.get("BPMeds", 0))
    prevalent_stroke = int(req.get("prevalentStroke", 0))
    prevalent_hyp = int(req.get("prevalentHyp", 0))
    diabetes = int(req.get("diabetes", 0))
    tot_chol = float(req.get("totChol", 200))
    sys_bp = float(req.get("sysBP", 120))
    dia_bp = float(req.get("diaBP", 80))
    bmi = float(req.get("BMI", 25))
    heart_rate = float(req.get("heartRate", 70))
    glucose = float(req.get("glucose", 100))

    edu = np.zeros(4)
    if 1 <= education <= 4:
        edu[education - 1] = 1.0

    raw = pd.DataFrame(
        [
            [
                male,
                age,
                current_smoker,
                cigs_per_day,
                bp_meds,
                prevalent_stroke,
                prevalent_hyp,
                diabetes,
                tot_chol,
                sys_bp,
                dia_bp,
                bmi,
                heart_rate,
                glucose,
                *edu,
            ]
        ],
        columns=feature_names,
    )

    if imputer is not None:
        raw = pd.DataFrame(imputer.transform(raw), columns=feature_names)
    scaled = pd.DataFrame(scaler.transform(raw), columns=feature_names)

    patient = {
        "male": male,
        "age": age,
        "education": education,
        "currentSmoker": current_smoker,
        "cigsPerDay": cigs_per_day,
        "BPMeds": bp_meds,
        "prevalentStroke": prevalent_stroke,
        "prevalentHyp": prevalent_hyp,
        "diabetes": diabetes,
        "totChol": tot_chol,
        "sysBP": sys_bp,
        "diaBP": dia_bp,
        "BMI": bmi,
        "heartRate": heart_rate,
        "glucose": glucose,
    }
    return scaled, patient


def predict_all(features, models):
    """Run predict_proba on every loaded model."""
    out = {}
    for name, model in models.items():
        out[name] = float(model.predict_proba(features)[0, 1])
    return out


def build_prediction_result(req, models, metrics, scaler, imputer, feature_names):
    """Build the same result dict as the former Flask /predict endpoint."""
    primary = primary_model_name(metrics)
    features, patient = make_features(req, scaler, imputer, feature_names)
    probs = predict_all(features, models)

    primary_probability = probs.get(primary) or 0.5
    primary_level, primary_color = risk_band(primary_probability)

    all_preds = []
    for name, probability in probs.items():
        level, colour = risk_band(probability)
        m = metrics.get(name, {})
        all_preds.append(
            {
                "model": display_name(name),
                "probability": round(probability * 100, 1),
                "risk_level": level,
                "risk_color": colour,
                "auc": round(m.get("auc", 0.0), 4),
                "recall": round(m.get("recall", 0.0), 3),
                "precision": round(m.get("precision", 0.0), 3),
            }
        )

    all_preds.sort(key=lambda entry: entry["probability"], reverse=True)

    explanation = None
    cise = models.get(CISE_NAME)
    if cise is not None:
        raw = cise.explain_single_prediction(features)
        contributions = []
        for contribution in raw["contributions"]:
            feature_name = contribution["name"]
            contributions.append(
                {
                    "name": feature_name,
                    "label": PRETTY.get(feature_name, feature_name),
                    "delta_pp": round(contribution["change_in_probability"] * 100, 1),
                    "logit_contribution": round(
                        contribution["contribution_to_logit"], 4
                    ),
                    "kind": (
                        "base_learner"
                        if feature_name in {"rf", "gb", "xgb"}
                        else "clinical"
                    ),
                }
            )
        explanation = {
            "model": display_name(CISE_NAME),
            "probability": round(raw["probability"] * 100, 1),
            "contributions": contributions,
        }

    return {
        "patient_data": patient,
        "primary_prediction": {
            "model": display_name(primary),
            "probability": round(primary_probability * 100, 1),
            "risk_level": primary_level,
            "risk_color": primary_color,
        },
        "all_predictions": all_preds,
        "explanation": explanation,
        "success": True,
    }


def get_model_info(models, metrics):
    """Same structure as the former Flask /api/model-info endpoint."""
    info = []
    for name in metrics:
        m = metrics[name]
        item = {
            "key": name,
            "name": display_name(name),
            "auc": round(m["auc"], 4),
            "recall": round(m["recall"], 3),
            "precision": round(m["precision"], 3),
            "f1": round(m["f1"], 3),
            "accuracy": round(m["accuracy"], 3),
        }
        model = models.get(name)
        if name == CISE_NAME and model is not None:
            item["meta_coefficients"] = [
                {"name": feature_name, "weight": round(weight, 4)}
                for feature_name, weight in model.coefficient_summary
            ]
        info.append(item)
    return info


def format_download_text(result):
    """Plain-text export matching the former browser download."""
    primary = result["primary_prediction"]
    lines = [
        "CARDIOVASCULAR RISK ASSESSMENT RESULTS",
        "=====================================",
        "",
        f"10-Year Risk: {primary['probability']}%",
        f"Risk Level: {primary['risk_level']}",
        "",
        "MODEL PREDICTIONS:",
    ]
    for pred in result["all_predictions"]:
        lines.append(f"{pred['model']}: {pred['probability']}% ({pred['risk_level']})")
    return "\n".join(lines) + "\n"

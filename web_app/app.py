"""
Flask web app for cardiovascular risk prediction.
Provides user-friendly interface for risk assessment and model comparison.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load models and preprocessor
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'

# Load preprocessor
try:
    preprocessor_data = joblib.load(PREPROCESSOR_PATH)
    scaler = preprocessor_data['scaler']
    logger.info("Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessor: {e}")
    scaler = None

# Load models - 5 optimized models using SMOTE
models = {}
MODEL_NAMES = [
    'Logistic_Regression_Balanced_Optimized',
    'Random_Forest_Optimized',
    'Gradient_Boosting_Optimized',
    'XGBoost_Optimized',
    'ANN_Optimized'
]

for model_name in MODEL_NAMES:
    try:
        model_path = MODELS_DIR / f'{model_name}.pkl'
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name}")
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")

# Feature names after preprocessing and encoding
FEATURE_NAMES = [
    'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose',
    'edu_1.0', 'edu_2.0', 'edu_3.0', 'edu_4.0'
]

# Model display names for UI
MODEL_DISPLAY_NAMES = {
    'Logistic_Regression_Balanced_Optimized': 'Logistic Regression (Baseline)',
    'Random_Forest_Optimized': 'Random Forest (SMOTE Optimized)',
    'Gradient_Boosting_Optimized': 'Gradient Boosting (SMOTE Optimized)',
    'XGBoost_Optimized': 'XGBoost (SMOTE Optimized)',
    'ANN_Optimized': 'Neural Network (SMOTE Optimized)'
}

# Model performance metrics (SMOTE-improved models)
MODEL_METRICS = {
    'Logistic_Regression_Balanced_Optimized': {'auc': 0.7445, 'recall': 0.72, 'precision': 0.45},
    'Random_Forest_Optimized': {'auc': 0.9429, 'recall': 0.92, 'precision': 0.88},
    'Gradient_Boosting_Optimized': {'auc': 0.9417, 'recall': 0.90, 'precision': 0.86},
    'XGBoost_Optimized': {'auc': 0.9401, 'recall': 0.91, 'precision': 0.85},
    'ANN_Optimized': {'auc': 0.9263, 'recall': 0.88, 'precision': 0.82}
}


def prepare_input_data(input_dict):
    """Parse and prepare input data for model prediction."""
    try:
        # Extract values from input dict
        male = int(input_dict.get('male', 0))
        age = float(input_dict.get('age', 50))
        education = int(input_dict.get('education', 1))
        currentSmoker = int(input_dict.get('currentSmoker', 0))
        cigsPerDay = float(input_dict.get('cigsPerDay', 0))
        BPMeds = int(input_dict.get('BPMeds', 0))
        prevalentStroke = int(input_dict.get('prevalentStroke', 0))
        prevalentHyp = int(input_dict.get('prevalentHyp', 0))
        diabetes = int(input_dict.get('diabetes', 0))
        totChol = float(input_dict.get('totChol', 200))
        sysBP = float(input_dict.get('sysBP', 120))
        diaBP = float(input_dict.get('diaBP', 80))
        BMI = float(input_dict.get('BMI', 25))
        heartRate = float(input_dict.get('heartRate', 70))
        glucose = float(input_dict.get('glucose', 100))
        
        # Create feature vector
        # Binary and continuous features
        features = np.array([
            male, age, currentSmoker, cigsPerDay, BPMeds,
            prevalentStroke, prevalentHyp, diabetes, totChol,
            sysBP, diaBP, BMI, heartRate, glucose
        ])
        
        # One-hot encode education (1-4)
        edu_encoded = np.zeros(4)
        if 1 <= education <= 4:
            edu_encoded[education - 1] = 1.0
        
        features = np.concatenate([features, edu_encoded])
        features = features.reshape(1, -1)
        
        # Scale features
        if scaler:
            features = scaler.transform(features)
        
        return features, {
            'male': male, 'age': age, 'education': education,
            'currentSmoker': currentSmoker, 'cigsPerDay': cigsPerDay,
            'BPMeds': BPMeds, 'prevalentStroke': prevalentStroke,
            'prevalentHyp': prevalentHyp, 'diabetes': diabetes,
            'totChol': totChol, 'sysBP': sysBP, 'diaBP': diaBP,
            'BMI': BMI, 'heartRate': heartRate, 'glucose': glucose
        }
    except Exception as e:
        logger.error(f"Error preparing input: {e}")
        return None, None


def get_model_predictions(features):
    """Get predictions from all available models."""
    predictions = {}
    
    for model_key, model in models.items():
        try:
            if model_key == 'ANN':
                prob = model.predict(features, verbose=0)[0][0]
            else:
                prob = model.predict_proba(features)[0][1]
            
            predictions[model_key] = float(prob)
        except Exception as e:
            logger.error(f"Error predicting with {model_key}: {e}")
            predictions[model_key] = None
    
    return predictions


def classify_risk(probability, threshold=0.5):
    """Classify risk level based on probability."""
    if probability >= threshold:
        if probability >= 0.7:
            return 'High', 'danger'
        else:
            return 'Moderate', 'warning'
    else:
        return 'Low', 'success'


@app.route('/')
def index():
    """Home page with input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions based on user input."""
    try:
        input_data = request.get_json()
        
        # Prepare features
        features, patient_data = prepare_input_data(input_data)
        if features is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Get predictions
        predictions = get_model_predictions(features)
        
        # Primary prediction (Random Forest - best performer)
        primary_model = 'Random_Forest_Optimized'
        primary_prob = predictions.get(primary_model, 0.5)
        primary_risk, primary_risk_color = classify_risk(primary_prob)
        
        # All model predictions
        all_predictions = []
        for model_key, prob in predictions.items():
            if prob is not None:
                risk_level, risk_color = classify_risk(prob)
                metrics = MODEL_METRICS.get(model_key, {})
                all_predictions.append({
                    'model': MODEL_DISPLAY_NAMES.get(model_key, model_key),
                    'probability': round(prob * 100, 1),
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'auc': metrics.get('auc', 'N/A'),
                    'recall': metrics.get('recall', 'N/A'),
                    'precision': metrics.get('precision', 'N/A')
                })
        
        # Sort by probability
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'patient_data': patient_data,
            'primary_prediction': {
                'model': MODEL_DISPLAY_NAMES.get(primary_model),
                'probability': round(primary_prob * 100, 1),
                'risk_level': primary_risk,
                'risk_color': primary_risk_color
            },
            'all_predictions': all_predictions,
            'success': True
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model-info')
def model_info():
    """Return information about available models."""
    info = []
    for model_key, display_name in MODEL_DISPLAY_NAMES.items():
        metrics = MODEL_METRICS.get(model_key, {})
        info.append({
            'key': model_key,
            'name': display_name,
            'auc': metrics.get('auc', 'N/A'),
            'recall': metrics.get('recall', 'N/A'),
            'precision': metrics.get('precision', 'N/A')
        })
    return jsonify(info)


@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')


@app.route('/disclaimer')
def disclaimer():
    """Legal disclaimer page."""
    return render_template('disclaimer.html')


if __name__ == '__main__':
    logger.info("Starting Cardiovascular Risk Prediction Web App")
    app.run(debug=True, port=5000)

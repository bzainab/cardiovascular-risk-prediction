# Web App Deployment Guide

## Quick Start

### Prerequisites
- Python 3.10+ with all dependencies installed (see `requirements.txt`)
- All models trained and saved in `models/`
- Preprocessor saved in `models/preprocessor.pkl`

### Running the App

1. **Navigate to the web app directory:**
   ```bash
   cd web_app
   ```

2. **Start the Flask app:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   - Navigate to `http://localhost:5000`

### Directory Structure
```
web_app/
├── app.py                          # Flask application
├── templates/
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Main prediction interface
│   ├── about.html                  # Project information
│   └── disclaimer.html             # Legal disclaimer
├── static/
│   ├── style.css                   # Styling
│   └── script.js                   # Client-side functionality
└── README.md                       # (this file)
```

## Features

### 1. **Patient Information Form**
- 15 input fields covering all Framingham features
- Input validation with realistic ranges
- Smart field dependencies (e.g., cigarettes/day enabled only for smokers)

### 2. **Risk Assessment**
- Primary prediction using Logistic Regression (Balanced)
- Comparative predictions from all 9 models
- Risk categorization: Low/Moderate/High

### 3. **Model Transparency**
- Performance metrics (ROC-AUC, Recall, Precision) displayed for each model
- Educational comparison of different algorithms
- Clear explanation of model strengths

### 4. **Comprehensive Information**
- About page: Dataset, methodology, validation approach
- Disclaimer page: Clear medical disclaimers and limitations
- Navigation between prediction, info, and disclaimer pages

## Data Flow

```
User Input (Form)
    ↓
[app.py: prepare_input_data()]
    ↓
Feature Engineering (scaling, encoding)
    ↓
[app.py: get_model_predictions()]
    ↓
All 9 models predict
    ↓
[index.html: Display results]
    ↓
User sees: primary result + model comparison table
```

## Model Integration

The app loads models from `../models/`:
- `Logistic_Regression_Balanced.pkl` (PRIMARY)
- `Logistic_Regression_Accuracy.pkl`
- `Random_Forest.pkl`
- `Extra_Trees.pkl`
- `Gradient_Boosting.pkl`
- `Hist_Gradient_Boosting.pkl`
- `SVM_RBF.pkl`
- `XGBoost.pkl`
- `ANN.pkl`

All models are evaluated using the same preprocessor and feature scaling pipeline.

## Feature Engineering

**Input Features (15 total):**
1. `male` (0/1)
2. `age` (years)
3. `education` (1-4, one-hot encoded to 4 features)
4. `currentSmoker` (0/1)
5. `cigsPerDay` (0-100)
6. `BPMeds` (0/1)
7. `prevalentStroke` (0/1)
8. `prevalentHyp` (0/1)
9. `diabetes` (0/1)
10. `totChol` (mg/dL)
11. `sysBP` (mmHg)
12. `diaBP` (mmHg)
13. `BMI` (kg/m²)
14. `heartRate` (bpm)
15. `glucose` (mg/dL)

**Preprocessing:**
- StandardScaler normalization (fit on training data)
- Education categorical encoding (4 binary features)
- No imputation (user provides all values)

## API Endpoints

### POST `/predict`
**Request:**
```json
{
  "male": 1,
  "age": 50,
  "education": 2,
  "currentSmoker": 0,
  "cigsPerDay": 0,
  "BPMeds": 0,
  "prevalentStroke": 0,
  "prevalentHyp": 0,
  "diabetes": 0,
  "totChol": 200,
  "sysBP": 120,
  "diaBP": 80,
  "BMI": 25,
  "heartRate": 70,
  "glucose": 100
}
```

**Response:**
```json
{
  "success": true,
  "patient_data": {...},
  "primary_prediction": {
    "model": "Logistic Regression (Balanced)",
    "probability": 35.2,
    "risk_level": "Moderate",
    "risk_color": "warning"
  },
  "all_predictions": [
    {
      "model": "Model Name",
      "probability": 35.2,
      "risk_level": "Moderate",
      "risk_color": "warning",
      "auc": 0.6975,
      "recall": 0.6047,
      "precision": 0.2541
    },
    ...
  ]
}
```

### GET `/api/model-info`
Returns metadata about all available models including performance metrics.

## Customization

### Change Primary Model
Edit `app.py` line ~200:
```python
primary_model = 'MODEL_NAME_HERE'  # Change this
```

### Adjust Risk Thresholds
Edit `classify_risk()` function in `app.py`:
```python
def classify_risk(probability, threshold=0.5):  # Change threshold
    if probability >= threshold:
        ...
```

### Modify Input Ranges
Edit `index.html` form fields:
```html
<input type="number" min="1" max="120" value="50">  <!-- Adjust min/max -->
```

## Important Notes

- **No data storage**: All inputs processed in-memory; nothing persisted
- **Educational only**: Not approved for clinical use
- **Disclaimer required**: Always display medical disclaimer
- **Model performance**: ROC-AUC ~0.70 - moderate discrimination
- **Class imbalance**: Model trained on 15.2% CHD prevalence

## Troubleshooting

### Models not loading
- Check `models/` directory path
- Verify all `.pkl` files exist
- Check file permissions

### Preprocessor error
- Ensure `preprocessor.pkl` is in `models/`
- Verify it was saved with same scaler type

### Input validation failing
- Check form field min/max ranges
- Verify numeric input formats

### Port already in use
```bash
python app.py --port 5001  # Try different port
```

## Performance Optimization

For production deployment:
- Use WSGI server (Gunicorn, uWSGI)
- Add caching for model predictions
- Implement rate limiting
- Add HTTPS/SSL
- Deploy behind reverse proxy (Nginx)

```bash
# Example production deployment
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Testing

Basic curl test:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 50, "male": 1, "totChol": 200, ...}'
```

## Support & Documentation

- See `PROJECT_HANDBOOK.md` for full project requirements
- See `PROPOSAL.md` for methodology overview
- See `about.html` for end-user information
- See `disclaimer.html` for legal information

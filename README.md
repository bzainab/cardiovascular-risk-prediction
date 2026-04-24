# Cardiovascular Risk Prediction System

Machine learning project predicting 10-year coronary heart disease (CHD) risk using the Framingham Heart Study dataset.

## Project Structure

```
├── Framingham Data/          # Raw dataset
├── src/                      # Core Python modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── notebooks/                # Jupyter notebooks for exploration
├── models/                   # Saved trained models
├── web_app/                  # Flask web application
│   ├── app.py
│   ├── templates/
│   └── static/
├── results/                  # Model evaluation results, plots
├── documents/                # Project documentation
└── requirements.txt
```

## Dataset

- **4,240 records** with 16 features + 1 target variable
- **Target**: `TenYearCHD` (binary: 0/1)
- **Class balance**: 84.81% negative, 15.19% positive

## Development Phases

1. **Phase 1: Data Exploration & Preprocessing**
2. **Phase 2: Model Training & Evaluation**
3. **Phase 3: Model Selection & Optimization**
4. **Phase 4: Web App Development**
5. **Phase 5: Integration & Testing**

## Quick Start

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

## Features

- Multiple ML models (Logistic Regression, Random Forest, XGBoost, ANN)
- Cross-validation with stratified splits
- Web interface for risk prediction
- Educational disclaimers and model explainability

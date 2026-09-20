# Cardiovascular Risk Prediction System

Machine learning project predicting 10-year coronary heart disease (CHD) risk using the Framingham Heart Study dataset.

## Features

- End-to-end data preprocessing, model training, evaluation, and inference workflow
- Multiple models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, ANN, and a stacking ensemble
- Stratified evaluation with ROC-AUC, precision, recall, F1-score, calibration, and confusion-matrix outputs
- SMOTE comparison experiments for class imbalance
- Streamlit web app for interactive risk prediction
- Educational disclaimer pages and user-facing model context

## Project Structure

```text
Framingham_Data/          # Raw dataset
src/                      # Core Python package
  common.py               # Shared config, paths, and helpers
  data_preprocessing.py   # Data loading and preprocessing
  model_evaluation.py     # Metrics, plots, and evaluation
  run_pipeline.py         # End-to-end training/evaluation entrypoint
  trainers/               # Per-model trainers
  ensemble/               # Stacking ensemble
  experiments/            # Additional analyses
notebooks/                # Jupyter notebooks for exploration
models/                   # Saved trained models (.pkl)
web_app/                  # Streamlit web application
  streamlit_app.py
  inference.py
  pages/                  # About and disclaimer pages
results/                  # Model evaluation results and plots
Makefile                  # Common development tasks
requirements.txt
```

## Dataset

- 4,240 records with 15 features and 1 target variable
- Target: `TenYearCHD`
- Class balance: 84.81% negative, 15.19% positive

## Quick Start

Create a virtual environment and install dependencies:

```bash
make venv
source .venv/bin/activate
make install
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the Pipeline

Train and evaluate all models end to end:

```bash
make pipeline
```

This runs:

```bash
python -m src.run_pipeline
```

Evaluation artefacts are written to `results/`, and trained model artefacts are written to `models/`.

## Run the Web App

Launch the Streamlit risk-prediction interface:

```bash
make web
```

This runs:

```bash
streamlit run web_app/streamlit_app.py
```

## Results

The saved evaluation results compare multiple models using accuracy, precision, recall, F1-score, ROC-AUC, calibration, confidence intervals, and SMOTE experiments.

| Model | ROC-AUC | Notes |
| --- | ---: | --- |
| Logistic Regression | 0.6975 | Best ROC-AUC in the saved evaluation results |
| ANN | 0.6797 | Strong calibration score in the saved results |
| Clinically Informed Stacking Ensemble | 0.6787 | Ensemble model using clinically informed features |

## Linting and Formatting

Run all linting checks:

```bash
make lint
```

Auto-format the code:

```bash
make format
```

## Disclaimer

This application is for educational demonstration only. It is not a medical diagnostic tool and should not be used to make healthcare decisions.

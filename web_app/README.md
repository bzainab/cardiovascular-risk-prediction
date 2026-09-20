# Web App

Streamlit app for cardiovascular risk prediction.

## Quick Start

### Prerequisites

- Python 3.10+ with dependencies from `requirements.txt`
- Models in `../models/` (run `python -m src.run_pipeline` from the project root)
- `results/model_results.csv` from the latest training run

### Run

From the project root:

```bash
make web
```

Open the URL shown in the terminal (default `http://localhost:8501`).

## Structure

```
web_app/
├── streamlit_app.py    # Entry point (Assessment page + navigation)
├── inference.py        # Model loading and prediction logic
├── pages/
│   ├── about.py        # Project info and live metrics
│   └── disclaimer.py   # Legal disclaimer
└── README.md
```

## Features

- **Assessment** — 15 Framingham features, all 7 models, primary = highest ROC-AUC
- **CISE explanation** — per-patient contribution breakdown
- **About** — dataset, methodology, live metrics, CISE coefficients
- **Disclaimer** — educational-use notice and limitations

## Data Flow

```
User inputs (st.form)
    → inference.make_features() → scaled DataFrame
    → predict_all() on 7 models
    → charts, tables, CISE explanation
```

No inputs are stored server-side beyond the Streamlit session.

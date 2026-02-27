# Privacy-Preserving Healthcare Machine Learning System

## What This Project Is About
This project builds a healthcare risk prediction system that protects patient privacy while still providing useful machine learning predictions. It includes:
- A standard baseline model for comparison
- A differentially private (DP) model for privacy-preserving training
- A simple frontend demo and API for risk prediction

## What Problem This Solves
Healthcare data is highly sensitive and can leak through datasets or model behavior. This project addresses that risk by:
- Training with differential privacy to reduce memorization of individual records
- Measuring privacy leakage risk using a membership inference evaluation
- Demonstrating the trade-off between model utility and privacy in a practical workflow

## How To Run Locally
From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train models:

```bash
python3 train_baseline.py --data data/sample_health.csv --target outcome
python3 train_dp.py --data data/sample_health.csv --target outcome --epsilon 2.0 --delta 1e-5
python3 evaluate_privacy.py --data data/sample_health.csv --target outcome
```

Run frontend demo:

```bash
streamlit run src/demo_frontend.py
```

Run API:

```bash
MODEL_BUNDLE=models/dp_bundle.joblib uvicorn api.server:app --reload
```

## Requirements Used To Build This Project
- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- streamlit
- joblib
- fastapi
- uvicorn
- pydantic

(Install all with `pip install -r requirements.txt`.)

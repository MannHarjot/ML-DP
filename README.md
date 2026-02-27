# Privacy-Preserving Healthcare Machine Learning System

## What The Project Is About
This project is an end-to-end healthcare ML system that predicts **diabetes risk** while protecting sensitive patient data. It includes model training, privacy evaluation, API serving, and an interactive frontend demo.

## What Problem It Solves
Healthcare organizations need ML predictions, but patient data is highly sensitive. Standard ML pipelines can leak private information through stored datasets or model behavior. This creates privacy, security, and compliance risks.

## Solution Used
We use **Differential Privacy (DP)** in model training to reduce the chance that individual patient records can be inferred from the model.

The system includes:
- A **baseline model** (non-private) for utility comparison
- A **DP model** (DP-SGD style: gradient clipping + noise)
- A simple **membership inference risk evaluation** to compare leakage risk
- A frontend where users can adjust **epsilon** and observe privacy/utility trade-offs

## What We Used To Build It
- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- streamlit
- fastapi
- uvicorn
- pydantic
- joblib

## How To Run Locally
From project root:

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

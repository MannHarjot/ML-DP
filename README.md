# Privacy-Preserving Healthcare ML MVP

This repo includes a minimal end-to-end pipeline to train disease-risk models with and without differential privacy, evaluate privacy leakage risk, and serve predictions through an API.

## Files Added for MVP
- `train_baseline.py`: trains standard logistic regression baseline
- `train_dp.py`: trains DP-SGD logistic regression model
- `evaluate_privacy.py`: runs a basic membership-inference attack
- `api/server.py`: FastAPI inference service
- `docs/privacy_report.md`: privacy/compliance report template

## Setup
```bash
pip install -r requirements.txt
```

## Train Baseline
```bash
python train_baseline.py --data /path/to/health_data.csv --target outcome
```

## Train Differentially Private Model
```bash
python train_dp.py --data /path/to/health_data.csv --target outcome --epsilon 2.0 --delta 1e-5
```

## Evaluate Privacy Leakage Risk
```bash
python evaluate_privacy.py --data /path/to/health_data.csv --target outcome
```

## Run API
```bash
MODEL_BUNDLE=models/dp_bundle.joblib uvicorn api.server:app --reload
```

Then call `POST /predict` with:

```json
{
  "features": {
    "age": 52,
    "bmi": 31.1,
    "glucose": 165,
    "blood_pressure": 88
  }
}
```

## Existing Streamlit App
The prior Streamlit app remains in `src/app.py` and can still be used independently.

## Frontend Demo (Recommended for Presentation)
Launch an interactive UI that compares baseline and DP predictions side-by-side:

```bash
streamlit run src/demo_frontend.py
```

If model bundles are missing, train first:

```bash
python3 train_baseline.py --data data/sample_health.csv --target outcome
python3 train_dp.py --data data/sample_health.csv --target outcome --epsilon 2.0 --delta 1e-5
```

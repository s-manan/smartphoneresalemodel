# Used Smartphone Resale Price Estimator

A beginner-friendly ML regression demo with a Flask web app and Dockerfile.

## Included files
- `app.py`: Flask prediction interface
- `models/resale_model.joblib`: trained scikit-learn pipeline
- `data_synthetic.csv`: 5,000 synthetic examples to make this starter project runnable immediately
- `train_model.py`: reproducible training script
- `metrics.json`: held-out evaluation metrics
- `Dockerfile`: container configuration

## Important data limitation
The included data is **synthetic**, generated from illustrative pricing rules. It is not scraped from a marketplace and the metrics are not evidence of real-world resale-price accuracy. Treat this as a runnable project scaffold, not a reliable valuation service. For a portfolio project with defensible results, replace `data_synthetic.csv` with a properly licensed real used-phone resale dataset, update the preprocessing if needed, retrain, and evaluate on a held-out test set.

## Run locally
Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://localhost:8000.

## Retrain
```bash
python train_model.py
```

## Run with Docker
```bash
docker build -t used-smartphone-resale .
docker run --rm -p 8000:8000 used-smartphone-resale
```

## Inputs
- Brand and model family
- Age in years
- Physical condition
- Storage and RAM
- Battery health percentage

## Model
RandomForestRegressor inside a scikit-learn Pipeline with one-hot encoding for categorical variables. Pipeline serialization includes preprocessing so the app uses the same transformations as training.

## Evaluation metrics from this synthetic demonstration
```json
{
  "dataset_rows": 5000,
  "test_rows": 1000,
  "MAE_INR": 3329.94,
  "RMSE_INR": 4260.53,
  "R2": 0.9153,
  "note": "Metrics are on synthetic data generated from illustrative pricing rules; they do not measure real-world resale accuracy."
}
```

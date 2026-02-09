# Financial Markets Analysis

This repo contains:
- `Financial_Markets_Analysis.ipynb`: notebook-based analysis
- `nse_sensex (1).csv`: historical daily stock data (NSE-style)
- `frontend/`: interactive dashboard (upload + validation + charts)
- `backend/`: FastAPI service for validation, analytics, and ML inference
- `ml/`: time-series-safe training and inference CLI

## Architecture (local-first)

- **Frontend** (`frontend/app.py`): Streamlit UI for upload, analytics, and prediction.
- **Backend** (`backend/app/main.py`): FastAPI endpoints for dataset validation/storage, analytics, and `/ml/predict`.
- **Local storage**:
  - datasets: `data/datasets/<dataset_id>/data.parquet` + `meta.json`
  - model runs: `artifacts/<run_id>/model.joblib` + `run.json`
- **ML** (`ml/fma_ml`): leakage-safe feature pipeline, walk-forward evaluation, saved artifacts.

## Dataset format (supported by the dashboard)

Required columns:
- `DATE` (parseable date)
- `SYMBOL` (ticker)
- `CLOSE` (numeric)

Recommended columns:
- `OPEN`, `HIGH`, `LOW`, `VOLUME`

The dashboard normalizes these to: `date`, `symbol`, `close`, `open`, `high`, `low`, `volume`.

## Run the dashboard (Windows PowerShell)

Install Python 3.10 (user scope, secure source):

```powershell
winget install --id Python.Python.3.10 -e --scope user
py -0p
```

Start the app:

```powershell
cd C:\Users\.\Financial-Markets-Analysis
.\scripts\run_frontend.ps1
```

### What a user does (local-first)

- One command (recommended):

```powershell
cd C:\Users\.\Financial-Markets-Analysis
.\scripts\run_all.ps1
```

- Run the backend in one terminal:

```powershell
cd C:\Users\.\Financial-Markets-Analysis
py -3.10 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- Run the frontend in another terminal:

```powershell
cd C:\Users\.\Financial-Markets-Analysis
.\scripts\run_frontend.ps1
```

- Open `http://localhost:8501` and:
  - explore analytics on the repo dataset or upload a CSV
  - (optional) train a model via CLI and use the **Prediction** tab

### Deployment note

This project is intentionally **local-first**.

If your machine policy blocks `pip --user`, use a single repo venv instead:

```powershell
cd C:\Users\.\Financial-Markets-Analysis
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r .\frontend\requirements.txt
.\.venv\Scripts\python -m streamlit run .\frontend\app.py
```

## ML (local, time-series-safe)

Install ML deps:

```powershell
cd C:\Users\.\Financial-Markets-Analysis
py -3.10 -m pip install -r .\ml\requirements.txt
```

Train a baseline next-day direction model using a saved dataset:

```powershell
$DATASET_ID="46648df56102493a9b366b91862fedae"
py -3.10 -m ml.fma_ml.cli train --dataset-id $DATASET_ID --symbol ASIANPAINT --model logreg --splits 5 --out artifacts
```

The command prints the run folder path (contains `model.joblib` + `run.json`).

Predict next-day direction probability using a saved run:

```powershell
$RUN_ID="85181c56309a4e62a2eb96f60ad0371e"
$DATASET_ID="46648df56102493a9b366b91862fedae"
py -3.10 -m ml.fma_ml.cli predict --run-id $RUN_ID --dataset-id $DATASET_ID --symbol ASIANPAINT
```

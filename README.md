# EchoSafe — Disaster Risk Platform for Pakistan

A small, end-to-end disaster intelligence platform that pulls live and
historical data for **earthquake**, **heatwave**, and **hailstorm**
events across Pakistan, trains ML models on the past 10 years, and
serves the results through a Streamlit dashboard.

For a deep dive into data sources and model choices, see
[`info.readme`](info.readme).

---

## What's in the repo

```
backend/         All Python source code
  config/        Settings, JWT auth, logging
  pipelines/     bronze -> silver -> gold ETL for every module
  ml_models/     Trainers (Logistic Regression + SMOTE, Linear Regression)
  predictors/    Live prediction with the trained models
  risk_engine/   Lightweight rule-based scorers used by the dashboard
  services/      Batch prediction + alert generation
dashboard/       Streamlit app (login, per-disaster pages, map, alerts)
data/            bronze / silver / gold CSV + JSON outputs
models/          Trained model artifacts (.pkl + metadata.json)
predictions/     Latest batch predictions + alerts (consumed by dashboard)
airflow/dags/    DAG definitions for daily pipeline runs
evidence/        Model-selection notebooks (.ipynb)
```

---

## Setup

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd echosafe-disaster-platform
uv sync
```

That's it. `uv sync` reads `pyproject.toml` + `uv.lock` and builds the
virtual environment.

---

## Run

**Launch the dashboard** (the main demo):

```bash
uv run streamlit run dashboard/app.py
```

Open the printed URL (usually `http://localhost:8501`).

**Default login:** `admin` / `echosafe` (also `analyst` / `echosafe`).

**Refresh today's regional risk map and alerts** (optional, ~30 s, hits
Open-Meteo for every region):

```bash
uv run python -m backend.services.batch_predictions
```

**Re-run a pipeline** (only when you want to refresh bronze/silver/gold):

```bash
uv run python -m backend.pipelines.download_heatwave_data
uv run python -m backend.pipelines.bronze_to_silver_heatwave
uv run python -m backend.pipelines.silver_to_gold_heatwave
```

Same pattern for `earthquack` (USGS scrape) and `hailstorm`. The pipelines
are **incremental** — they reuse what's already on disk and only fetch
missing date windows. Lookback is configurable:

```bash
ECHOSAFE_LOOKBACK_YEARS=5 uv run python -m backend.pipelines.download_heatwave_data
```

---

## Architecture (high level)

```
┌─────────────────────────────────────────────────────────────┐
│   Sources:  USGS  ·  NASA POWER  ·  Open-Meteo  ·  IEM ASOS │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
            ┌──────────────────────────────────┐
            │  backend/pipelines/              │
            │  download -> bronze -> silver -> │
            │     gold (incremental)           │
            └────────────────┬─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
   ┌────────────────────┐         ┌────────────────────┐
   │ backend/ml_models/ │         │ backend/predictors/│
   │   train + save     │ ──────► │   load + predict   │
   │   models/*.pkl     │         │   from forecasts   │
   └────────────────────┘         └─────────┬──────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │ backend/services/       │
                              │ batch_predictions.py    │
                              │ -> predictions/*.csv    │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ dashboard/  (Streamlit) │
                              │ Login, Maps, Alerts,    │
                              │ per-disaster pages      │
                              └─────────────────────────┘
```

### Flow in one paragraph

`backend/pipelines/` fetches raw event data from the four upstream sources
into `data/bronze/`, cleans it into `data/silver/`, and engineers
ML-ready features into `data/gold/`. `backend/ml_models/` trains a
classifier or regressor on the gold dataset and saves the artifact in
`models/`. `backend/predictors/` loads that artifact and scores live
weather. `backend/services/batch_predictions.py` runs all predictors for
every region and writes the two CSVs that the Streamlit `dashboard/`
reads to draw the regional map and the alerts queue.

### Models

| Module     | Task                       | Model                                       |
| ---------- | -------------------------- | ------------------------------------------- |
| Earthquake | Magnitude regression       | StandardScaler -> LinearRegression          |
| Heatwave   | Heatwave-day classifier    | SMOTE -> StandardScaler -> LogisticRegression |
| Hailstorm  | Hail-day classifier        | SMOTE -> StandardScaler -> LogisticRegression |

All three use a **time-based train/test split** (train on the past, test
on the most recent year). Selection rationale and per-model metrics live
in [`info.readme`](info.readme) and the notebooks in `evidence/`.

---

## Optional: Airflow

DAG definitions in `airflow/dags/` schedule the download → silver →
gold → retrain chain. They are not required for the dashboard demo; the
same scripts can be invoked manually as shown above.

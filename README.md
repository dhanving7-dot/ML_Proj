# PlacementPredict

PlacementPredict is a student placement prediction platform. It consists of a leakage-safe ML pipeline, a FastAPI service with a model registry, monitoring and controlled retraining, an M2–M6 machine-learning laboratory with 27 experiments, and a React frontend.

Every number shown in the UI comes from a real API call against real data. Nothing is mocked or simulated.

> Predictions are statistical estimates from historical data, not guarantees.

---

## 1. Quick start (Windows, from the project root)

Prerequisites: Python 3.13 and Node 24 (both verified). Microsoft Edge is optional and only used by the browser checks.

**Dataset:** the CSV is versioned with DVC and is not stored in git. Place `placement_predict_50k Dataset.csv` at `data/raw/placement_50k.csv` (or run `dvc pull` once a DVC remote is configured) before training.

```powershell
# 1. Python environment (from the project root)
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt

# 2. Train, evaluate and register the first model (~30 s on the full 50k dataset)
cd backend
..\.venv\Scripts\python -m scripts.train

# 3. Start the API (http://127.0.0.1:8000, docs at /docs)
..\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 4. In a second terminal: the frontend (http://localhost:5173)
cd frontend
npm install
npm run dev            # or: npm run build && npm run preview  (http://localhost:4173)
```

The Vite dev and preview servers proxy `/api`, `/docs`, `/redoc` and `/openapi.json` to port 8000. To point a build at another API host, set `VITE_API_URL`.

**API documentation:** <http://127.0.0.1:8000/docs> (Swagger UI) and <http://127.0.0.1:8000/redoc>.

### Optional: generate real monitoring traffic

```powershell
cd backend
..\.venv\Scripts\python -m scripts.replay_traffic --n 300 --record-outcomes   # held-out rows + their true outcomes
..\.venv\Scripts\python -m scripts.replay_traffic --n 200 --segment tier3      # a shifted population to see drift
```

Replayed requests go through the HTTP API with `X-Prediction-Source: replay`. They are stored and displayed separately from online traffic.

---

## 2. Dataset

| Item | Value |
|---|---|
| File | `data/raw/placement_50k.csv` (`placement_predict_50k Dataset.csv`, md5 `14fce27e…`) |
| Size | 50,000 rows × 32 columns; target `PlacementStatus` (65.7 % placed) |
| Versioning | DVC (`data/raw/placement_50k.csv.dvc`) |
| Provenance | Not documented. The column structure suggests a synthetic or benchmark dataset, so results describe this dataset rather than real-world placement outcomes. |

Excluded from model inputs, with the reasons recorded in `backend/ml/contract.py`:

- `StudentID`: an identifier with no predictive meaning.
- `Salary Package`: post-outcome leakage, because a salary only exists after placement. It is used as the regression target in M2/M3.
- `IsAnomaly`: a dataset-generation annotation that is not available at prediction time. It is used only to score the M4 anomaly detectors.
- `CGPA_Tier`: a binned duplicate of CGPA (cut-offs 6.5 / 8.0). It matches that rule for about 97 % of rows, and most mismatches are the rows flagged in `IsAnomaly`. Data validation reports the inconsistent rows.

Five assessment columns contain 6–10 % missing values. They are imputed with training-split medians and accompanied by missing-value indicator columns.

---

## 3. Architecture

```
backend/
  ml/                 shared ML package (imported by BOTH training and serving)
    contract.py       feature contract: names, types, ranges, enums, excluded columns
    data.py           ingestion, validation report, cleaning, stratified splits, statistics
    features.py       coercion, SGPA feature engineering, ColumnTransformer pipeline
    training.py       CV model selection → refit → calibration/threshold on validation → one test evaluation
    metrics.py        classification/regression metrics, curves, McNemar, paired bootstrap
    drift.py          PSI, KL divergence, skewness/kurtosis reference profiles
    registry.py       atomic artifact writes, SHA-256 integrity check on load
    tracking.py       MLflow logging (failures reported, never fatal)
    lab/              27 M2–M6 experiments (train split → validation split; test never used)
  app/                FastAPI service
    main.py           wiring, CORS, request-log middleware, safe error handlers
    schemas.py        StudentFeatures generated from the contract (extra fields rejected)
    services/         model, prediction, training/retraining, monitoring, datasets, lab
    routers/          system, predictions, registry (models/training/datasets), observability
  scripts/            train.py, validate_data.py, replay_traffic.py
  tests/              pytest suite
frontend/             Vite + React 19 + TypeScript + Tailwind 4 + Recharts + GSAP + Motion
  src/api             typed client, error normalisation, endpoint functions
  src/hooks           useApi (abort + stale-response guard), useAction (no double submit), counters
  src/motion          ScrollTrigger reveals, word reveals, magnetic CTAs, scroll progress
  src/components      UI kit, charts (validated palette, legends, table views), layout
  src/pages           Home, Dashboard, Predict, Lab (M2–M6), Lifecycle, Monitoring, Models, Data
dvc.yaml, params.yaml DVC pipeline (validate → train) and training parameters
```

**Training-serving consistency.** Several mechanisms keep training and serving aligned:
- The API schema is generated from the same contract that training validates against.
- The fitted pipeline (imputer medians, scaler parameters, encoder vocabularies) is persisted and loaded, never re-fitted in production.
- Each artifact stores 20 golden examples with their exact feature vectors and probabilities. These are replayed through the serving path every time a model is loaded, and the result is shown in `/api/health`.
- Every prediction stores the exact feature vector the model saw.

---

## 4. Commands

| Task | Command (from `backend/` unless noted) |
|---|---|
| Train / retrain (CLI) | `..\.venv\Scripts\python -m scripts.train [--sample-rows 8000] [--candidates logistic_regression,xgboost]` |
| Validate the dataset | `..\.venv\Scripts\python -m scripts.validate_data` |
| Reproduce the pipeline | from the project root, with the venv on `PATH`: `dvc repro`, `dvc metrics show`, `dvc dag` |
| MLflow UI | from the project root: `.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Backend tests | `..\.venv\Scripts\python -m pytest` |
| Frontend tests | `cd frontend; npm test` (Vitest) · `npm run typecheck` · `npm run build` |
| Browser checks (Edge) | `cd frontend; node scripts/browser-check.mjs http://localhost:4173` · `node scripts/e2e-check.mjs` · `node scripts/lab-check.mjs` |

Configuration uses environment variables with the `PP_` prefix (see `backend/app/config.py`). Examples: `PP_DATABASE_URL`, `PP_MODELS_DIR`, `PP_MLFLOW_ENABLED`, `PP_CORS_ORIGINS`, `PP_MAX_UPLOAD_MB`.

---

## 5. Results from the current run (held-out test split, 7,500 students)

| Candidate (5-fold CV ROC-AUC on training split) | Mean ± std |
|---|---|
| XGBoost (selected) | 0.9805 ± 0.0012 |
| Logistic Regression | 0.9803 ± 0.0009 |
| Histogram Gradient Boosting | 0.9798 ± 0.0012 |
| LightGBM | 0.9797 ± 0.0011 |
| Random Forest | 0.9787 ± 0.0008 |

XGBoost is only 0.0002 ahead, which is well inside one fold's standard deviation, so the candidates are effectively tied. M5's McNemar test also found no significant difference between logistic regression and random forest (p = 0.65).

**Active model v2 on the test split:** ROC-AUC 0.9802, F1 0.941, accuracy 92.1 %, precision 0.921, recall 0.962, Brier 0.054. The threshold is 0.45, tuned on validation only. Validation ECE is 0.008 without calibration, so production calibration is off. Platt and isotonic calibration remain available as options.

---

## 6. M1–M6 coverage

| Module | Where it is implemented |
|---|---|
| **M1** lifecycle, paradigms, serving, monitoring | Lifecycle page (live status of 9 stages and the traced path of the latest prediction); supervised learning implemented; unsupervised learning in the lab; RL, DL, GenAI and LLMs explained conceptually only |
| **M2** linear models and preprocessing | Normal equation vs scikit-learn vs gradient descent; Ridge/Lasso/Elastic Net with paths; logistic regression (sigmoid, BCE, threshold, L1/L2, multinomial salary bands); Standard/MinMax scaling; one-hot, ordinal and cross-fitted target encoding (with a leakage demo); mean/median/iterative imputation with MissingIndicator |
| **M3** trees and ensembles | Gini/entropy/information gain computed by hand; pre-pruning and cost-complexity pruning; MSE regression tree; random forest (bootstrap, √p features, OOB), AdaBoost, GBM, HistGB, XGBoost, LightGBM; impurity vs permutation importance and SHAP |
| **M4** unsupervised | K-means (elbow, silhouette); hierarchical clustering (4 linkages, dendrogram, cophenetic correlation); DBSCAN (core/border/noise, k-distance); PCA; t-SNE/UMAP with trustworthiness; Isolation Forest and One-Class SVM scored against `IsAnomaly`; autoencoder conceptual only |
| **M5** evaluation | K-fold vs stratified, nested CV; full classification metrics, ROC/PR; Platt/isotonic calibration and reliability diagrams; grid/random/Bayesian (Optuna TPE) search; learning and validation curves; McNemar; paired bootstrap on regression RMSE (RMSE/MAE/MAPE/R² also in M2/M3) |
| **M6** production | FastAPI online and batch prediction, schema checks, health/version; golden E2E skew test plus 4 real skew scenarios; point-in-time feature sets and a same-name/different-meaning demo; pickle/joblib/ONNX with measured parity; PSI/KL/skewness/kurtosis drift; concept drift from recorded outcomes; MLflow; DVC; controlled retraining with safe promotion; gRPC and Feast covered conceptually |

---

## 7. Verification performed (actually executed)

| Check | Result |
|---|---|
| `pytest` (backend: startup, DB init, validation, missing data, preprocessing consistency, training and artifact integrity, prediction validation/schema/persistence, dashboard, metrics, clustering, calibration/CV, schema-mismatch detection, monitoring events, failed and rolled-back retraining, concurrency, CORS, error leakage, MLflow) | **55 passed** |
| Vitest + React Testing Library (routing/404, active nav, health badge, form payload, result only after the API resolves, client and server validation, API failure state, stale-response guard, exact counter values, reduced motion, error mapping) | **16 passed** |
| `tsc --noEmit` / `vite build` | 0 errors / success |
| Browser check (Edge headless): 8 routes × desktop 1440, tablet 820, mobile 390 | 0 console errors, 0 failed requests, 0 px horizontal overflow |
| E2E (Edge): prediction reveal, trace, client validation, double-click protection, nav/back/forward, deep link + refresh, live lab re-run, drift check toast, scroll reveals after route changes, 404, reduced motion, API-offline error state | **12 / 12 passed** |
| All 27 lab experiments rendered in Edge with real results | **27 / 27 passed** |
| DVC `dvc repro` → `dvc status` | pipeline ran and is up to date |
| MLflow | 3 training runs logged (SQLite store) |

Bugs found and fixed during verification:
- An SQLite write lock was held across the response, which deadlocked the request-log middleware. Fixed with function-scoped DB sessions.
- An animated counter could briefly show a huge negative number because animation-frame timestamps can precede `performance.now()`.
- Mobile overflow in grid layouts.
- PSI bars were drawn on a log scale, which visually overstated drift.
- Smoothed line charts implied values between measured points.

---

## 8. Known limitations

- **Security:**
  - No authentication or per-client rate limiting. Batch size, upload size and file type are limited, but no formal security review has been performed.
  - Pickle/joblib artifacts are only loaded when their SHA-256 matches the registry. That protects integrity, but it is not a sandbox.
- **Point-in-time:** the production model uses the `full` feature set (all 8 semesters). If predictions must be made before semesters 7–8 are graded, retrain with `feature_set: pre_final_year`. The M6 lab measures the cost of doing so.
- **Model comparison during retraining:** retraining compares the candidate with the active model on the candidate's validation split. If the dataset or split differs from the active model's, the job records a warning that some of those rows may have been in the active model's training data.
- **ONNX:** the parity experiment uses a portable variant without the custom SGPA features and missing indicators, which have no ONNX converter. Tree models show rare larger differences because ONNX Runtime compares split thresholds in float32.
- **Scale:** the SQLite backend and the in-process job lock are designed for a single API process. Running several workers would need a shared job queue and a server database.
- **Heavier lab experiments** (ensembles, search, nested CV, point-in-time) take roughly 10–50 s and therefore run on demand rather than automatically.
- **Database contents:** the local database contains the predictions created while verifying the app. That includes 300 labelled replayed test rows and one deliberately invalid uploaded dataset used to test validation.
- The original `ML_Project.ipynb` (credit-card-fraud EDA) is left untouched. Credit-card fraud is outside this project's scope.

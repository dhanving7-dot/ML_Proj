# PlacementPredict → FraudGuard migration: audit, dependency map and checklist

Audit performed on commit `9b3798f` (tagged `pre-fraud-migration`) before any file was changed.
The migration is implemented on branch `migrate/fraud-detection`.

## 1. Dataset facts (measured on `creditcard.csv`, md5 `e90efcb83d69faf99fcab8b0255024de`)

| Fact | Value |
|---|---|
| Shape | 284,807 rows × 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| dtypes | 30 × float64, `Class` int64 |
| Missing values | 0 |
| Class balance | 284,315 legitimate, 492 fraud (0.1727 %) |
| Exact duplicate rows | 1,081 (19 of them fraud); 0 feature-duplicates with conflicting labels |
| `Time` | 0 – 172,792 s, sorted ascending, two days (49.2 % of rows in the second day) |
| `Amount` | 0 – 25,691.16, median 22.0, 1,825 zero amounts (27 fraud) |
| `V1`–`V28` | PCA outputs: means ≈ 0, pairwise correlation 0.0, std 0.33 – 1.96 |
| Hour of window | fraud rate varies from 0.05 % (hour 10) to 1.7 % (hour 2) |

## 2. Placement-specific assumptions found, and where they live

| Area | Files | Placement assumption |
|---|---|---|
| Feature contract | `backend/ml/contract.py` | 32 student columns, categorical enums (City, Stream…), SGPA semesters, target `PlacementStatus`, `Salary Package` leakage column, `IsAnomaly`, `CGPA_Tier`, feature sets `full` / `pre_final_year` |
| Validation | `backend/ml/data.py` | CGPA/SGPA consistency checks, `CGPA_Tier` rule, `StudentID` duplicates, class-balance warning at 5 % |
| Features | `backend/ml/features.py` | `AcademicFeatureEngineer` (SGPA trend/volatility), one-hot + ordinal encoders, median imputation + missing indicators |
| Training | `backend/ml/training.py`, `estimators.py` | 5-candidate CV selection by ROC-AUC, single classifier, F1 threshold, permutation importance on ROC-AUC |
| Drift | `backend/ml/drift.py` | categorical references (works generically) |
| API schemas | `backend/app/schemas.py` | `StudentFeatures` generated from the contract, `EXAMPLE_STUDENT`, `placed_count`, salary disclaimer |
| Prediction service | `backend/app/services/prediction_service.py`, `model_service.py` | single probability + threshold, nullable-field imputation, golden check over `StudentFeatures` |
| Prediction router | `backend/app/routers/predictions.py` | history `summary` shows CGPA/Tier/Stream; trace text mentions SGPA |
| Dashboard / lifecycle | `backend/app/routers/system.py` | `placed_rate`, `cgpa/stream/tier` in recent predictions, DVC check for `placement_50k.csv.dvc` |
| Datasets | `backend/app/services/dataset_service.py` | `StudentID`, sample rows by student |
| Monitoring | `backend/app/services/monitoring_service.py` | `placed_rate`, accuracy-based concept drift |
| Retraining | `backend/app/services/training_service.py` | promotion gate on validation ROC-AUC |
| Lab M2–M6 | `backend/ml/lab/*.py`, `app/services/lab_service.py` | salary regression, City/Tier encoders, SGPA correlation notes, `IsAnomaly` labels, semester point-in-time sets, Tier3/backlog drift segments |
| Scripts | `backend/scripts/*.py` | placement dataset path, Tier3 replay segment |
| Tests | `backend/tests/*.py` | placement sample fixture, student payloads, CGPA assertions |
| DVC / params | `dvc.yaml`, `params.yaml`, `dvc.lock` | `placement_50k.csv`, candidate list, `feature_set` |
| Frontend API | `frontend/src/api/types.ts`, `endpoints.ts` | `StudentInput`, `placed_count`, `cgpa/stream/tier` |
| Frontend pages | `Home`, `Dashboard`, `Predict`, `Models`, `Monitoring`, `Lifecycle`, `Data`, `lab/M2–M6` | student copy, semester form groups, "Placed / Not placed" encodings, CGPA calculator |
| Branding | navbar, footer, page header, `index.html`, README | "PlacementPredict" |

## 3. Dependency map (what each generic component depends on)

```
ml/contract.py ──► ml/data.py ──► ml/training.py ──► app/services/training_service.py ──► routers/registry.py
      │                │               │                       │
      │                ▼               ▼                       ▼
      │          ml/features.py ◄── ml/drift.py           ml/registry.py (generic) ─► model_service (golden check)
      ▼
app/schemas.py ──► routers/predictions.py ──► services/prediction_service.py ──► orm.Prediction ──► dashboard / monitoring / lifecycle
      │
      └──► lab_service.py ──► ml/lab/m2…m6 (each imports contract + features)
frontend/api/types.ts ◄── every page; endpoints.ts mirrors the routers 1:1
```

Generic and reusable without semantic change: `app/db.py`, `app/errors.py`, `app/main.py` (middleware, CORS, error handling),
`ml/registry.py` (atomic writes + SHA-256), `ml/tracking.py` (MLflow), `ml/drift.py` (PSI/KL), `ml/metrics.py`,
job locking / rollback in `training_service.py`, the request log, monitoring events, the lab registry/cache/concurrency
limit, and on the frontend the whole UI kit, chart kit, motion layer, `useApi`/`useAction`, toasts, modal, tabs and layout.

## 4. Decisions

| Topic | Decision | Reason |
|---|---|---|
| Duplicates | Exact duplicate rows are dropped **before** splitting (`drop_duplicates: true`, counts recorded in the validation report and model metadata) | Identical rows in both train and test would leak and inflate test metrics. Configurable; never silent. |
| Split | Stratified 70 / 15 / 15, seed 42 | Brief requirement; test used once. A time-ordered split is measured in the M6 lab. |
| `Time` | Not used raw. Encoded as `sin`/`cos` of the hour within a 24 h cycle measured from the first transaction | Raw `Time` only increases, so future transactions fall outside the training range. The clock offset of `Time = 0` is not documented, so this is "hour since the first transaction's clock time", not wall-clock time. |
| `Amount` | `log1p` then standardised | Strong right skew (max 25,691, median 22). |
| `V1`–`V28` | Standardised with training statistics; no invented meanings | Anonymised PCA outputs. |
| Logistic Regression | `class_weight="balanced"`, `C` chosen by stratified CV on the training split (average precision), prior correction of the intercept so the output is a fraud probability, threshold tuned on validation | Imbalance handling + interpretable probability. Calibration is measured (Brier, ECE), not assumed. |
| Isolation Forest | Fitted on the **training split, all rows, labels unused** (`reference: all_train`; `legitimate_train` available). Anomaly flag = top `flag_rate` of the training reference scores (label-free) | Keeps the detector unsupervised; labels only score it afterwards. |
| Score normalisation | Each score is mapped to its **rarity** in the training reference: `r = −log10(1 − F_ref(s)) / log10(n_ref + 1)` ∈ [0, 1] | Puts a probability and an anomaly score on the same unit (decades of rarity) instead of adding incompatible raw scores. |
| Fusion | `hybrid = w · r_LR + (1 − w) · r_IF`; `w` chosen on the validation split by average precision | Transparent, one parameter, tuned on validation only. |
| Decisions | `HIGH_RISK` if hybrid ≥ `t_high` (validation F1-optimal); `REVIEW` if ≥ `t_review` (validation recall target, default 0.90); else `LOW_RISK` | Configurable in `params.yaml`; evaluated once on test. |
| Database | New SQLite file `backend/fraudguard.db`; the placement database is left untouched as an archive | Prediction rows changed shape (three scores + decision); placement rows cannot be represented. A schema-version table prevents pointing the new app at the old file. |
| Artifacts | New registry directory `artifacts/fraud_models/`; `artifacts/models/` (placement v1/v2) kept | No overwrite of existing artifacts. |
| MLflow | New experiment `fraud-detection` in the same `mlflow.db` | Placement runs stay browsable. |
| DVC | `data/raw/creditcard.csv` tracked with `dvc add`; pipeline deps switched; placement `.dvc` file kept as archive | Reproducible pipeline on the new data. |

## 5. Checklist

- [x] Audit repository and dataset (this document)
- [x] Checkpoint: tag `pre-fraud-migration`, branch `migrate/fraud-detection`
- [ ] `ml/contract.py`, `ml/data.py`, `ml/features.py` for the fraud schema
- [ ] `ml/fraud_model.py` (LR + IF + fusion + decisions), `ml/training.py`, `ml/metrics.py` extensions
- [ ] DB schema, prediction/monitoring/dataset/training services, routers, evaluation explorer endpoint
- [ ] Lab M2–M6 adapted to the fraud data (see §6)
- [ ] Scripts, `params.yaml`, `dvc.yaml`, DVC tracking of the dataset, MLflow experiment
- [ ] Backend tests updated + new tests
- [ ] Frontend API client/types, pages (Overview, Analyze, Transactions, Evaluation, Models, Monitoring, Lifecycle, Data, Lab, Home)
- [ ] Frontend tests, type check, build, browser checks
- [ ] Full training run, README with measured results

## 6. Lab experiments: keep / adapt / archive

| Experiment | Action |
|---|---|
| M2 linear regression, regularisation | Adapted: target `log1p(Amount)` from `V1`–`V28` (a real continuous target in this dataset) |
| M2 logistic regression | Adapted to `Class` with class weights; multinomial part uses amount tertiles |
| M2 preprocessing | Adapted: scaling any column; one-hot / target encoding on hour-of-window; ordinal on amount band; identifier leakage demo on the row index; imputation on clearly labelled injected (MCAR) missingness because the dataset has none |
| M3 decision tree, pruning, ensembles, importance | Adapted to fraud with PR-AUC / F1 instead of accuracy; ensembles train on all training frauds + an undersampled legitimate sample |
| M3 regression tree | Adapted to `log1p(Amount)` |
| M4 k-means, hierarchical, DBSCAN, PCA, t-SNE/UMAP | Adapted to `V1`–`V28` + log amount; fraud share reported after fitting |
| M4 anomaly detection | Adapted: scored against `Class` instead of `IsAnomaly` |
| M5 all | Adapted to average precision; stratified vs plain K-fold is now a strong demonstration because folds can contain zero frauds |
| M5 statistical tests | McNemar kept; paired bootstrap now on average-precision difference |
| M6 skew | Adapted scenarios: re-fitted scaler, amount in cents, `Time` in hours, swapped columns, client-side log of `Amount` |
| M6 point-in-time | Replaced semester feature sets with random vs time-ordered splitting and a shifted `Time` origin |
| M6 ONNX, packaging, drift sandbox | Adapted (fraud segments) |
| Placement-only content (salary, CGPA_Tier, semesters, IsAnomaly) | Archived: available at tag `pre-fraud-migration` |

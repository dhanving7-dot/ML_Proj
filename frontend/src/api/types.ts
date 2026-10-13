/* Typed mirrors of the FastAPI responses (backend/app). Keep in sync with the backend schemas. */

export type FeatureGroup = "profile" | "academic" | "assessment" | "experience";

export interface NumericFieldSchema {
  name: string;
  kind: "numeric";
  group: FeatureGroup;
  description: string;
  minimum: number;
  maximum: number;
  integer: boolean;
  nullable: boolean;
  unit: string | null;
}

export interface CategoricalFieldSchema {
  name: string;
  kind: "nominal" | "ordinal";
  group: FeatureGroup;
  description: string;
  categories: string[];
  nullable: boolean;
}

export type FieldSchema = NumericFieldSchema | CategoricalFieldSchema;

export interface SchemaResponse {
  request_fields: FieldSchema[];
  model_features: string[];
  feature_set: string;
  engineered_features: string[];
  excluded_columns: Record<string, string>;
  target: string;
  class_labels: Record<string, string>;
  available_classifiers: string[];
}

export type StudentInput = Record<string, string | number | null>;

export interface ConsistencyCheck {
  passed: boolean;
  examples?: number;
  max_probability_diff?: number;
  max_feature_vector_diff?: number;
  checked_at?: string;
  reason?: string;
}

export interface Health {
  status: "ok" | "degraded";
  api_version: string;
  time: string;
  uptime_s: number;
  database: string;
  model_loaded: boolean;
  model_version: string | null;
  model_type: string | null;
  consistency_check: ConsistencyCheck | null;
  message: string | null;
}

export interface PredictionResponse {
  request_id: string;
  created_at: string;
  model_version: string;
  model_type: string;
  prediction: 0 | 1;
  label: string;
  probability: number;
  threshold: number;
  latency_ms: number;
  timings: Record<string, number>;
  imputed_fields: string[];
  disclaimer: string;
}

export interface BatchItem {
  index: number;
  request_id: string;
  prediction: 0 | 1;
  label: string;
  probability: number;
}

export interface BatchResponse {
  batch_id: string;
  model_version: string;
  count: number;
  placed_count: number;
  latency_ms: number;
  results: BatchItem[];
  errors: { index: number; errors: { field: string; message: string }[] }[];
  disclaimer: string;
}

export interface PredictionRow {
  request_id: string;
  batch_id: string | null;
  source: "online" | "batch" | "replay";
  created_at: string;
  model_version: string;
  prediction: 0 | 1;
  label: string;
  probability: number;
  threshold: number;
  latency_ms: number;
  actual_outcome: 0 | 1 | null;
  outcome_recorded_at: string | null;
  summary: Record<string, string | number | null>;
}

export interface PredictionTrace extends PredictionRow {
  inputs: StudentInput;
  feature_vector: number[];
  feature_names: string[];
  imputed_fields: string[];
  timings: Record<string, number>;
  trace: { step: string; detail: string; ms?: number | null }[];
}

export interface SampleRow {
  student_id: number;
  features: StudentInput;
  actual_outcome: 0 | 1;
}

export interface ClassMetrics {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity?: number;
  roc_auc: number | null;
  average_precision: number | null;
  log_loss: number;
  brier: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  support: number;
}

export interface MonitoringEvent {
  id?: number;
  ts: string;
  kind: string;
  severity: "info" | "warning" | "critical";
  message: string;
  details?: Record<string, unknown> | null;
}

export interface Dashboard {
  generated_at: string;
  predictions: {
    total: number;
    last_24h: number;
    placed: number;
    placed_rate: number | null;
    mean_probability: number | null;
    labelled: number;
    live_accuracy: number | null;
    trend: { date: string; predictions: number; placed: number }[];
    probability_histogram: { bin: string; count: number }[];
  };
  operations: {
    requests_24h: number;
    server_errors_24h: number;
    error_rate_24h: number | null;
    predict_latency_p50_ms: number | null;
  };
  model: null | {
    version: string;
    type: string;
    label: string;
    threshold: number;
    feature_set: string;
    loaded_at: string;
    trained_at: string | null;
    test_metrics: { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number; brier: number };
    test_support: number;
    consistency: ConsistencyCheck;
    cv_results: { model: string; label: string; roc_auc: number; roc_auc_std: number; f1: number }[];
  };
  models_registered: number;
  training: {
    runs: number;
    job_running: boolean;
    last: null | { job_id: string; status: string; decision: string | null; reason: string | null; finished_at: string | null; new_version: string | null };
  };
  monitoring: {
    last_drift_check: null | { ts: string; severity: string; message: string };
    recent_events: MonitoringEvent[];
  };
  recent_predictions: {
    request_id: string;
    created_at: string;
    source: string;
    label: string;
    prediction: 0 | 1;
    probability: number;
    model_version: string;
    cgpa: number | null;
    stream: string | null;
    tier: string | null;
  }[];
}

export type StageStatus = "complete" | "attention" | "pending";

export interface LifecycleStage {
  key: string;
  title: string;
  status: StageStatus;
  detail: string;
  facts: Record<string, unknown>;
}

export interface Lifecycle {
  stages: LifecycleStage[];
  last_prediction_request_id: string | null;
  tooling: {
    mlflow: { installed_version: string | null; enabled: boolean; tracking_uri: string; runs_logged: number; last_error: string | null };
    dvc: { installed_version: string | null; repo_initialised: boolean; dataset_tracked: boolean; pipeline_defined: boolean; lock_file: boolean };
    scikit_learn: string | null;
    optional: Record<string, string | null>;
  };
  lab_runs: number;
}

export interface MonitoringSummary {
  window_hours: number;
  generated_at: string;
  requests: {
    total: number;
    server_errors: number;
    client_errors: number;
    error_rate: number | null;
    client_error_rate: number | null;
    throughput_per_min: number;
  };
  latency_ms: { predict_p50: number | null; predict_p95: number | null; predict_p99: number | null; predict_requests: number };
  predictions: {
    total: number;
    placed_rate: number | null;
    mean_probability: number | null;
    by_source: Record<string, number>;
    probability_histogram: { bin: string; count: number }[];
  };
  series: { t: string; requests: number; predict_requests: number; errors: number; client_errors: number; mean_latency_ms: number | null }[];
}

export type DriftStatus = "stable" | "investigate" | "significant_shift" | "insufficient_data" | "no_model";

export interface DriftBin {
  bin: string;
  expected: number;
  actual: number;
}

export interface DriftFeatureDetail {
  kind: "numeric" | "categorical";
  psi: number | null;
  kl_divergence: number | null;
  status: DriftStatus;
  n: number;
  bins?: DriftBin[];
  summary?: Record<string, number | null>;
  unseen_categories?: string[];
}

export interface DriftReport {
  status: DriftStatus;
  message?: string;
  model_version?: string;
  window?: number;
  rows?: number;
  min_rows?: number;
  from?: string;
  to?: string;
  features?: { feature: string; psi: number | null; kl_divergence: number | null; status: DriftStatus; kind: string }[];
  prediction?: { psi: number | null; kl_divergence: number | null; status: DriftStatus; bins: DriftBin[] } | null;
  detail?: Record<string, DriftFeatureDetail>;
  flagged_features?: string[];
  thresholds?: { stable_below: number; significant_at_or_above: number };
  interpretation?: string;
}

export interface Performance {
  labelled: number;
  total_predictions: number;
  min_labelled: number;
  model_version?: string;
  offline_test?: { accuracy: number; f1: number; roc_auc: number; precision: number; recall: number };
  labelled_active_model: number;
  status: "insufficient_labels" | "healthy" | "degraded";
  message?: string;
  live?: ClassMetrics;
  accuracy_drop_vs_test?: number | null;
  blocks?: { block: number; rows: number; accuracy: number }[];
  interpretation?: string;
}

export interface CvResult {
  model: string;
  label: string;
  roc_auc: { mean: number; std: number; folds: number[] };
  f1: { mean: number; std: number; folds: number[] };
  accuracy: { mean: number; std: number; folds: number[] };
  neg_log_loss: { mean: number; std: number; folds: number[] };
  fit_time_s: number;
}

export interface ModelSummary {
  version: string;
  model_type: string;
  status: "active" | "archived";
  created_at: string;
  activated_at: string | null;
  feature_set: string;
  threshold: number;
  sha256: string;
  size_bytes: number;
  dataset_id: number | null;
  mlflow_run_id: string | null;
  metrics: { validation: Partial<ClassMetrics>; test: Partial<ClassMetrics>; cv: CvResult | null };
}

export interface CurvePointROC {
  fpr: number;
  tpr: number;
  threshold: number;
}
export interface CurvePointPR {
  recall: number;
  precision: number;
  threshold: number;
}
export interface CalibrationResult {
  points: { mean_predicted: number; fraction_positive: number }[];
  histogram: { bin_start: number; count: number }[];
  ece: number;
  brier: number;
}
export interface SweepPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  predicted_positive_rate: number;
}

export interface ModelDetail extends ModelSummary {
  metadata: {
    model_type: string;
    model_label: string;
    task: string;
    features: string[];
    engineered_features: string[];
    transformed_features: string[];
    excluded_columns: Record<string, string>;
    threshold: number;
    calibration: { method: string; rows: Record<string, number> };
    config: Record<string, unknown>;
    splits: {
      strategy: string;
      seed: number;
      fractions: Record<string, number>;
      train: { rows: number; positive_rate: number };
      validation: { rows: number; positive_rate: number };
      test: { rows: number; positive_rate: number };
      usage: Record<string, string>;
    };
    model_selection: { protocol: string; metric: string; selected: string; results: CvResult[] };
    metrics: { validation: ClassMetrics; test: ClassMetrics; test_at_0_5: ClassMetrics; validation_uncalibrated_roc_auc: number };
    curves: {
      test_roc: CurvePointROC[];
      test_pr: CurvePointPR[];
      test_calibration: CalibrationResult;
      validation_calibration_uncalibrated: CalibrationResult;
      validation_threshold_sweep: SweepPoint[];
    };
    feature_importance: {
      permutation_validation_roc_auc: { feature: string; importance: number; std: number }[];
      impurity_top30: { feature: string; importance: number }[] | null;
    };
    data_validation: ValidationReport;
    dataset?: { id: number; name: string; sha256: string; rows: number };
    artifact_sha256: string;
    library_versions: Record<string, string>;
    timings: Record<string, number>;
    candidates_skipped: string[];
  };
  reference_features: string[];
  consistency: ConsistencyCheck | null;
}

export interface TrainingJob {
  job_id: string;
  trigger: string;
  status: "queued" | "running" | "succeeded" | "rejected" | "failed";
  stage: string | null;
  message: string | null;
  config: Record<string, unknown>;
  dataset_id: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  candidate_model_type: string | null;
  candidate_metrics: { validation: Partial<ClassMetrics>; test: Partial<ClassMetrics>; cv: CvResult | null } | null;
  comparison: Record<string, unknown> | null;
  active_version_before: string | null;
  decision: "promoted" | "rejected" | null;
  reason: string | null;
  new_version: string | null;
  mlflow_run_id: string | null;
  mlflow_error: string | null;
  log: { ts: string; stage: string; message: string }[];
}

export interface TrainingConfigInput {
  candidates?: string[];
  feature_set?: "full" | "pre_final_year";
  engineered_features?: boolean;
  cv_folds?: number;
  selection_metric?: "roc_auc" | "f1" | "accuracy" | "neg_log_loss";
  calibration?: "none" | "sigmoid" | "isotonic";
  threshold_strategy?: "f1" | "fixed";
  sample_rows?: number | null;
  min_validation_roc_auc?: number;
}

export interface ValidationCheck {
  name: string;
  status: "pass" | "warn" | "fail";
  message: string;
  details: Record<string, unknown> | null;
}

export interface ValidationReport {
  status: "passed" | "passed_with_warnings" | "failed";
  n_rows: number;
  n_columns: number;
  checks: ValidationCheck[];
}

export interface DatasetSummary {
  id: number;
  name: string;
  source: "default" | "upload";
  sha256: string;
  n_rows: number;
  n_columns: number;
  positive_rate: number | null;
  validation_status: ValidationReport["status"];
  uploaded_at: string;
}

export interface DatasetDetail extends DatasetSummary {
  validation_report: ValidationReport;
}

export interface NumericColumnStats {
  name: string;
  kind: "numeric";
  missing: number;
  missing_rate: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  q25: number | null;
  median: number | null;
  q75: number | null;
  max: number | null;
  skewness: number | null;
  kurtosis: number | null;
  histogram: { counts: number[]; edges: number[] };
}

export interface CategoricalColumnStats {
  name: string;
  kind: "categorical";
  missing: number;
  missing_rate: number;
  unique: number;
  top: { value: string; count: number }[];
}

export interface DatasetStats {
  dataset_id: number;
  rows: number;
  columns: (NumericColumnStats | CategoricalColumnStats)[];
}

export interface LabSplitInfo {
  source: string;
  seed: number;
  train_rows_used: number;
  validation_rows_used: number;
  test_split: string;
  note: string;
}

export interface LabResponse<T> {
  module: string;
  experiment: string;
  title: string;
  params: Record<string, unknown>;
  ran_at: string;
  duration_ms: number;
  dataset_sha256: string;
  model_version: string | null;
  result: T & { split?: LabSplitInfo };
  cached: boolean;
}

export interface LabCatalogItem {
  module: string;
  key: string;
  title: string;
  topics: string[];
  needs_model: boolean;
  params_schema: Record<string, unknown>;
}

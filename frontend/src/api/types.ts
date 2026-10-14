/* Typed mirrors of the FastAPI responses (backend/app). Keep in sync with the backend schemas. */

export type FeatureGroup = "time" | "amount" | "pca";
export type Decision = "LOW_RISK" | "REVIEW" | "HIGH_RISK";
export const DECISIONS: Decision[] = ["LOW_RISK", "REVIEW", "HIGH_RISK"];

export interface FieldSchema {
  name: string;
  kind: "numeric";
  group: FeatureGroup;
  description: string;
  minimum: number | null;
  maximum: number | null;
  unit: string | null;
  integer: boolean;
  nullable: boolean;
}

export interface SchemaResponse {
  request_fields: FieldSchema[];
  model_features: string[];
  metadata_fields: string[];
  transformed_features: string[];
  feature_ranges: Record<string, [number, number]> | null;
  excluded_columns: Record<string, string>;
  target: string;
  class_labels: Record<string, string>;
  decisions: Decision[];
  time_encodings: string[];
  amount_transforms: string[];
}

/** 30 model features plus the optional transaction_id. */
export type TransactionInput = Record<string, number | string | null>;

export interface Policy {
  fusion_weight_lr: number;
  fusion_weight_if: number;
  review_threshold: number;
  high_risk_threshold: number;
  lr_threshold: number;
  if_flag_percentile: number;
  if_review_percentile: number | null;
}

export interface ConsistencyCheck {
  passed: boolean;
  examples?: number;
  max_feature_vector_diff?: number;
  max_lr_probability_diff?: number;
  max_anomaly_score_diff?: number;
  max_hybrid_score_diff?: number;
  decision_mismatches?: number;
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
  policy: Policy | null;
  consistency_check: ConsistencyCheck | null;
  message: string | null;
}

export interface Scores {
  lr_probability: number;
  anomaly_score: number;
  anomaly_percentile: number;
  lr_rarity: number;
  if_rarity: number;
  hybrid_score: number;
}

export interface Contribution {
  feature: string;
  label: string;
  standardised_value: number;
  coefficient: number;
  contribution: number;
}

export interface Explanation {
  lr_top_contributions: Contribution[];
  lr_intercept: number;
  lr_prior_correction: number;
  lr_logit: number;
  hybrid_components: { lr: number; isolation_forest: number };
  lr_exceedance: number;
  if_exceedance: number;
  anomaly_review_trigger: boolean;
  out_of_training_range: string[];
  lr_rarity?: number;
  if_rarity?: number;
}

export interface PredictionResponse {
  request_id: string;
  created_at: string;
  model_version: string;
  model_type: string;
  transaction_id: string | null;
  decision: Decision;
  scores: Scores;
  flags: { logistic_regression: boolean; isolation_forest: boolean };
  policy: Policy;
  explanation: Explanation;
  out_of_training_range: string[];
  latency_ms: number;
  timings: Record<string, number>;
  disclaimer: string;
}

export interface BatchItem {
  index: number;
  request_id: string;
  transaction_id: string | null;
  decision: Decision;
  hybrid_score: number;
  lr_probability: number;
  anomaly_percentile: number;
}

export interface BatchResponse {
  batch_id: string;
  model_version: string;
  count: number;
  decision_counts: Record<Decision, number>;
  latency_ms: number;
  results: BatchItem[];
  errors: { index: number; errors: { field: string; message: string }[] }[];
  disclaimer: string;
}

export interface PredictionRow {
  request_id: string;
  batch_id: string | null;
  transaction_id: string | null;
  source: "online" | "batch" | "replay";
  created_at: string;
  model_version: string;
  decision: Decision;
  hybrid_score: number;
  lr_probability: number;
  anomaly_score: number;
  anomaly_percentile: number;
  lr_flag: boolean;
  if_flag: boolean;
  amount: number;
  time_s: number;
  out_of_range: string[];
  latency_ms: number;
  actual_outcome: 0 | 1 | null;
  outcome_recorded_at: string | null;
}

export interface PredictionTrace extends PredictionRow {
  inputs: Record<string, number>;
  feature_vector: number[];
  feature_names: string[];
  feature_labels: string[];
  policy: Policy;
  explanation: Explanation;
  timings: Record<string, number>;
  trace: { step: string; detail: string; ms?: number | null }[];
}

export interface PredictionPage {
  total: number;
  limit: number;
  offset: number;
  decision_counts: Record<Decision, number>;
  items: PredictionRow[];
}

export interface PredictionQuery {
  limit?: number;
  offset?: number;
  source?: string;
  decision?: Decision[];
  model_version?: string;
  q?: string;
  min_score?: number;
  max_score?: number;
  min_amount?: number;
  max_amount?: number;
  labelled?: boolean;
  outcome?: 0 | 1;
  sort?: "newest" | "oldest" | "score_desc" | "score_asc" | "amount_desc";
}

export interface SampleRow {
  row_id: string;
  features: Record<string, number>;
  actual_class: 0 | 1;
}

export interface ClassMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  fraud_detection_rate: number;
  specificity: number;
  false_positive_rate: number;
  accuracy: number;
  balanced_accuracy: number;
  roc_auc: number | null;
  average_precision: number | null;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  false_positives: number;
  false_negatives: number;
  alerts: number;
  alert_rate: number;
  support: number;
  positives: number;
  positive_rate: number;
  log_loss?: number;
  brier?: number;
}

export interface DecisionRow {
  decision: Decision;
  transactions: number;
  fraud: number;
  legitimate: number;
  share_of_transactions: number;
  share_of_all_fraud: number;
  fraud_rate_in_band: number | null;
}

export type ComponentKey = "logistic_regression" | "isolation_forest" | "hybrid_high_risk" | "hybrid_review_or_high";

export interface EvaluationBlock {
  logistic_regression: ClassMetrics;
  isolation_forest: ClassMetrics;
  hybrid_high_risk: ClassMetrics;
  hybrid_review_or_high: ClassMetrics;
  decisions: DecisionRow[];
}

export interface MonitoringEvent {
  id?: number;
  ts: string;
  kind: string;
  severity: "info" | "warning" | "critical";
  message: string;
  details?: Record<string, unknown> | null;
}

export interface DashboardRow {
  request_id: string;
  created_at: string;
  source: string;
  transaction_id: string | null;
  decision: Decision;
  hybrid_score: number;
  lr_probability: number;
  anomaly_percentile: number;
  amount: number;
  model_version: string;
  actual_outcome: 0 | 1 | null;
}

export type HeadlineMetrics = Pick<ClassMetrics, "precision" | "recall" | "f1" | "average_precision" | "roc_auc" | "false_positives" | "false_negatives" | "alert_rate">;

export interface Dashboard {
  generated_at: string;
  predictions: {
    total: number;
    last_24h: number;
    by_decision: Record<Decision, number>;
    by_source: Record<string, number>;
    flagged: number;
    flagged_rate: number | null;
    high_risk_rate: number | null;
    open_review: number;
    open_high_risk: number;
    labelled: number;
    confirmed_fraud: number;
    confirmed_fraud_flagged: number;
    trend: ({ date: string } & Record<Decision, number>)[];
    hybrid_histogram: { bin: string; count: number }[];
  };
  operations: { requests_24h: number; server_errors_24h: number; error_rate_24h: number | null; predict_latency_p50_ms: number | null };
  model: null | {
    version: string;
    type: string;
    label: string;
    policy: Policy;
    loaded_at: string;
    trained_at: string | null;
    test_metrics: Record<ComponentKey, HeadlineMetrics>;
    test_decisions: DecisionRow[];
    test_support: number;
    test_fraud_rows: number;
    consistency: ConsistencyCheck;
  };
  models_registered: number;
  training: {
    runs: number;
    job_running: boolean;
    last: null | { job_id: string; status: string; decision: string | null; reason: string | null; finished_at: string | null; new_version: string | null };
  };
  monitoring: { last_drift_check: null | { ts: string; severity: string; message: string }; recent_events: MonitoringEvent[] };
  recent_alerts: DashboardRow[];
  recent_predictions: DashboardRow[];
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
    mlflow: { installed_version: string | null; enabled: boolean; tracking_uri: string; experiment: string; runs_logged: number; last_error: string | null };
    dvc: { installed_version: string | null; repo_initialised: boolean; dataset_tracked: boolean; pipeline_defined: boolean; lock_file: boolean };
    scikit_learn: string | null;
    optional: Record<string, string | null>;
  };
  lab_runs: number;
}

export interface MonitoringSummary {
  window_hours: number;
  generated_at: string;
  requests: { total: number; server_errors: number; client_errors: number; error_rate: number | null; client_error_rate: number | null; throughput_per_min: number };
  latency_ms: { predict_p50: number | null; predict_p95: number | null; predict_p99: number | null; predict_requests: number };
  predictions: {
    total: number;
    by_decision: Record<Decision, number>;
    flagged_rate: number | null;
    high_risk_rate: number | null;
    mean_hybrid_score: number | null;
    by_source: Record<string, number>;
    histograms: Record<"hybrid_score" | "lr_probability" | "anomaly_percentile", { bin: string; count: number }[]>;
  };
  series: {
    t: string;
    requests: number;
    predict_requests: number;
    errors: number;
    client_errors: number;
    mean_latency_ms: number | null;
    low_risk: number;
    review: number;
    high_risk: number;
  }[];
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
  scores?: Record<string, { psi: number | null; kl_divergence: number | null; status: DriftStatus; bins: DriftBin[]; summary?: Record<string, number | null> }>;
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
  offline_test?: { high_risk_precision: number; high_risk_recall: number; review_or_high_recall: number; average_precision: number };
  labelled_active_model: number;
  status: "insufficient_labels" | "healthy" | "degraded";
  message?: string;
  live?: { high_risk: ClassMetrics; review_or_high: ClassMetrics; ranking: { average_precision: number | null; roc_auc: number | null } };
  recall_drop_vs_test?: number | null;
  blocks?: { block: number; rows: number; frauds: number; recall_review_or_high: number | null; precision_review_or_high: number | null }[];
  interpretation?: string;
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
  points: { mean_predicted: number; fraction_positive: number; count?: number }[];
  histogram: { bin_start: number; count: number }[];
  ece: number;
  brier: number;
  mean_predicted?: number;
  observed_rate?: number;
}
export interface SweepPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  alert_rate: number;
  alerts: number;
  false_positives: number;
  accuracy?: number;
  predicted_positive_rate?: number;
}

export interface LrCvResult {
  C: number;
  average_precision: { mean: number; std: number; folds: number[] };
  roc_auc: { mean: number; std: number; folds: number[] };
  fit_time_s: number;
}

export type HeadlineBlock = Record<ComponentKey, Partial<ClassMetrics>> & { decisions?: DecisionRow[] };

export interface ModelSummary {
  version: string;
  model_type: string;
  status: "active" | "candidate" | "archived";
  created_at: string;
  activated_at: string | null;
  feature_set: string;
  threshold: number;
  policy: Policy;
  sha256: string;
  size_bytes: number;
  dataset_id: number | null;
  mlflow_run_id: string | null;
  metrics: { validation: HeadlineBlock; test: HeadlineBlock; cv: LrCvResult | null; policy: Policy };
}

export interface SplitsInfo {
  strategy: string;
  seed: number;
  fractions: Record<string, number>;
  train: { rows: number; fraud_rows: number; positive_rate: number };
  validation: { rows: number; fraud_rows: number; positive_rate: number };
  test: { rows: number; fraud_rows: number; positive_rate: number };
  cleaning: Record<string, number | string>;
  usage: Record<string, string>;
}

export interface Tuning {
  weight_source: string;
  lr_threshold_rule: string;
  high_risk_rule: string;
  review_rule: string;
  review_binding_constraint: string;
  if_flag_rule: string;
  if_review_trigger: string;
  notes: string[];
}

export interface Curves {
  test_pr: Record<"logistic_regression" | "isolation_forest" | "hybrid", CurvePointPR[]>;
  test_roc: Record<"logistic_regression" | "isolation_forest" | "hybrid", CurvePointROC[]>;
  validation_threshold_sweep: Record<"hybrid" | "logistic_regression", SweepPoint[]>;
  fusion_weight: { weight_lr: number; average_precision: number; roc_auc: number }[];
}

export interface CalibrationInfo {
  method: string;
  logit_offset: number;
  validation: CalibrationResult;
  validation_without_correction: CalibrationResult;
  test: CalibrationResult;
}

export interface ModelMetadata {
  model_type: string;
  model_label: string;
  task: string;
  components: {
    logistic_regression: { C: number; imbalance: string; class_weight: string | null; training_rows: number; training_fraud_rows: number; intercept: number; prior_correction_offset: number };
    isolation_forest: { reference: string; fit_rows: number; labels_used_for_fitting: boolean; params: Record<string, unknown>; anomaly_score: string };
    references: Record<string, { n: number; quantiles: Record<string, number> }>;
  };
  policy: Policy;
  tuning: Tuning;
  normalisation: string;
  fusion_formula: string;
  decision_rules: Record<Decision, string>;
  config: Record<string, unknown>;
  feature_set: string;
  features: string[];
  transformed_features: string[];
  feature_ranges: Record<string, [number, number]>;
  excluded_columns: Record<string, string>;
  splits: SplitsInfo;
  model_selection: { protocol: string; metric: string; selected_C: number; results: LrCvResult[] };
  metrics: { validation: EvaluationBlock; test: EvaluationBlock };
  calibration: CalibrationInfo;
  curves: Curves;
  feature_importance: {
    lr_coefficients: { feature: string; label: string; coefficient: number }[];
    lr_permutation_validation_ap: { feature: string; importance: number; std: number }[];
  };
  data_validation: ValidationReport;
  dataset?: { id: number; name: string; sha256: string; rows: number };
  artifact_sha256: string;
  library_versions: Record<string, string>;
  timings: Record<string, number>;
}

export interface ModelDetail extends ModelSummary {
  metadata: ModelMetadata;
  reference_features: string[];
  consistency: ConsistencyCheck | null;
}

export interface IntegrityReport {
  version: string;
  exists: boolean;
  matches: boolean;
  expected_sha256: string;
  actual_sha256: string | null;
  size_bytes?: number;
}

export interface EvaluationSummary {
  model_version: string;
  policy: Policy;
  tuning: Tuning;
  normalisation: string;
  fusion_formula: string;
  decision_rules: Record<Decision, string>;
  splits: SplitsInfo;
  metrics: { validation: EvaluationBlock; test: EvaluationBlock };
  calibration: CalibrationInfo;
  curves: Curves;
  model_selection: ModelMetadata["model_selection"];
  components: ModelMetadata["components"];
  methodology: string[];
}

export interface ExploreRequest {
  weight: number;
  review_threshold: number;
  high_risk_threshold: number;
  lr_threshold: number;
  if_flag_rate: number;
  if_review_percentile?: number | null;
}

export interface ExploreResponse {
  split: "validation";
  model_version: string;
  rows: number;
  fraud_rows: number;
  policy: Policy;
  active_policy: Policy;
  metrics: EvaluationBlock;
  curves: { pr: Record<"logistic_regression" | "isolation_forest" | "hybrid", CurvePointPR[]>; hybrid_threshold_sweep: SweepPoint[] };
  note: string;
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
  candidate_metrics: { validation: HeadlineBlock; test: HeadlineBlock; cv: LrCvResult | null; policy: Policy } | null;
  comparison: Record<string, unknown> | null;
  active_version_before: string | null;
  decision: "promoted" | "registered" | "rejected" | null;
  reason: string | null;
  new_version: string | null;
  mlflow_run_id: string | null;
  mlflow_error: string | null;
  log: { ts: string; stage: string; message: string }[];
}

export interface TrainingConfigInput {
  drop_duplicates?: boolean;
  sample_rows?: number | null;
  time_encoding?: "cyclical_hour" | "raw" | "none";
  amount_transform?: "log1p" | "raw";
  lr_imbalance?: "class_weight" | "undersample" | "none";
  lr_undersample_ratio?: number;
  lr_c_grid?: number[];
  lr_calibration?: "prior_correction" | "none";
  cv_folds?: number;
  if_n_estimators?: number;
  if_max_samples?: number;
  if_reference?: "all_train" | "legitimate_train";
  if_flag_rate?: number;
  fusion_weight?: number | null;
  high_risk_strategy?: "f1" | "precision_target";
  high_risk_precision_target?: number;
  review_recall_target?: number;
  max_review_rate?: number;
  if_review_percentile?: number | null;
  min_validation_average_precision?: number;
  auto_promote?: boolean;
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
  mean_legitimate?: number | null;
  mean_fraud?: number | null;
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
  train_fraud_rows: number | null;
  validation_rows_used: number;
  validation_fraud_rows: number | null;
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

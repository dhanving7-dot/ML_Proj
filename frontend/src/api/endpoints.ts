import { get, http, post, toApiError } from "./client";
import type {
  BatchResponse,
  Dashboard,
  DatasetDetail,
  DatasetStats,
  DatasetSummary,
  DriftReport,
  Health,
  LabCatalogItem,
  LabResponse,
  Lifecycle,
  ModelDetail,
  ModelSummary,
  MonitoringEvent,
  MonitoringSummary,
  Performance,
  PredictionResponse,
  PredictionRow,
  PredictionTrace,
  SampleRow,
  SchemaResponse,
  StudentInput,
  TrainingConfigInput,
  TrainingJob,
} from "./types";

type Opts = { signal?: AbortSignal };

async function upload<T>(url: string, file: File, signal?: AbortSignal): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  try {
    // Let the browser set the multipart boundary.
    return (await http.post<T>(url, form, { signal, headers: { "Content-Type": undefined }, timeout: 120_000 })).data;
  } catch (e) {
    throw toApiError(e);
  }
}

export const api = {
  health: (o: Opts = {}) => get<Health>("/api/health", { signal: o.signal, timeout: 8_000 }),
  schema: (o: Opts = {}) => get<SchemaResponse>("/api/schema", o),
  dashboard: (o: Opts = {}) => get<Dashboard>("/api/dashboard", o),
  lifecycle: (o: Opts = {}) => get<Lifecycle>("/api/lifecycle", o),

  predict: (student: StudentInput, o: Opts = {}) => post<PredictionResponse>("/api/predict", student, o),
  predictBatchCsv: (file: File, o: Opts = {}) => upload<BatchResponse>("/api/predict/batch/csv", file, o.signal),
  predictions: (params: { limit?: number; offset?: number; source?: string } = {}, o: Opts = {}) =>
    get<{ total: number; limit: number; offset: number; items: PredictionRow[] }>("/api/predictions", { params, signal: o.signal }),
  prediction: (id: string, o: Opts = {}) => get<PredictionTrace>(`/api/predictions/${encodeURIComponent(id)}`, o),
  recordOutcome: (id: string, actual: 0 | 1) =>
    post<{ request_id: string; actual_outcome: number; correct: boolean }>(`/api/predictions/${encodeURIComponent(id)}/outcome`, { actual_outcome: actual }),

  datasets: (o: Opts = {}) => get<{ items: DatasetSummary[] }>("/api/datasets", o),
  dataset: (id: number, o: Opts = {}) => get<DatasetDetail>(`/api/datasets/${id}`, o),
  datasetStats: (id: number, o: Opts = {}) => get<DatasetStats>(`/api/datasets/${id}/stats`, o),
  datasetSample: (id: number, seed?: number, o: Opts = {}) =>
    get<{ dataset_id: number; split: string; items: SampleRow[] }>(`/api/datasets/${id}/sample`, { params: { n: 1, seed }, signal: o.signal }),
  uploadDataset: (file: File, o: Opts = {}) => upload<DatasetDetail>("/api/datasets/upload", file, o.signal),

  models: (o: Opts = {}) => get<{ active_version: string | null; items: ModelSummary[] }>("/api/models", o),
  model: (version: string, o: Opts = {}) => get<ModelDetail>(`/api/models/${encodeURIComponent(version)}`, o),
  activateModel: (version: string) => post<ModelSummary>(`/api/models/${encodeURIComponent(version)}/activate`),

  startTraining: (body: { dataset_id?: number | null; config: TrainingConfigInput }) => post<{ job_id: string; status: string }>("/api/training/jobs", body),
  trainingJobs: (o: Opts = {}) => get<{ job_running: boolean; items: TrainingJob[] }>("/api/training/jobs", o),
  trainingJob: (id: string, o: Opts = {}) => get<TrainingJob>(`/api/training/jobs/${encodeURIComponent(id)}`, o),

  monitoringSummary: (hours: number, o: Opts = {}) => get<MonitoringSummary>("/api/monitoring/summary", { params: { hours }, signal: o.signal }),
  drift: (o: Opts = {}) => get<DriftReport>("/api/monitoring/drift", o),
  driftCheck: () => post<DriftReport>("/api/monitoring/drift/check"),
  performance: (o: Opts = {}) => get<Performance>("/api/monitoring/performance", o),
  events: (limit = 30, o: Opts = {}) => get<{ items: MonitoringEvent[] }>("/api/monitoring/events", { params: { limit }, signal: o.signal }),

  labCatalog: (o: Opts = {}) => get<{ items: LabCatalogItem[] }>("/api/lab/catalog", o),
  lab: <T>(module: string, key: string, params: Record<string, unknown>, o: Opts = {}) =>
    post<LabResponse<T>>(`/api/lab/${module}/${key}`, params, { signal: o.signal, timeout: 180_000 }),
};

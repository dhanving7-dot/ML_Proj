import { motion } from "motion/react";
import { Boxes, CircleCheck, History, Play, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/endpoints";
import type { ModelDetail, ModelSummary, TrainingConfigInput, TrainingJob } from "../api/types";
import { ConfusionMatrix, HBars, MetricGrid, PrChart, ReliabilityChart, RocChart, ThresholdSweep } from "../components/charts/Common";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { SelectField, Slider, Toggle } from "../components/ui/Fields";
import { Modal } from "../components/ui/Modal";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Chip, jobTone, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { useToast } from "../components/ui/Toast";
import { useHealth } from "../hooks/HealthContext";
import { useAction, useApi } from "../hooks/useApi";
import { bytes, cn, dateTime, int, modelLabel, num, pct, timeAgo } from "../lib/format";
import { DataTable } from "./lab/kit";

export default function Models() {
  const models = useApi((s) => api.models({ signal: s }), []);
  const jobs = useApi((s) => api.trainingJobs({ signal: s }), [], { intervalMs: 15_000 });
  const [selected, setSelected] = useState<string | null>(null);
  const version = selected ?? models.data?.active_version ?? models.data?.items[0]?.version ?? null;
  const health = useHealth();

  const refreshAll = () => {
    models.refetch();
    jobs.refetch();
    health.refetch();
  };

  return (
    <>
      <PageHeader
        crumb="Models"
        title="Model registry & retraining"
        description="Every registered version with its configuration, held-out metrics and integrity hash. Retraining validates the data, trains a candidate, compares it with the active model and only then promotes it."
      />
      <Section className="space-y-6">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
          <div className="space-y-6">
            <Panel>
              <PanelHeader eyebrow="Registry" title="Versions" />
              {models.error ? (
                <ErrorState error={models.error} onRetry={models.refetch} compact />
              ) : !models.data ? (
                <LoadingBlock rows={4} />
              ) : models.data.items.length === 0 ? (
                <EmptyState title="No model registered" icon={<Boxes className="size-5" aria-hidden="true" />}>
                  Start a training job below, or run <code className="font-mono">python -m scripts.train</code>.
                </EmptyState>
              ) : (
                <ul className="space-y-2" role="listbox" aria-label="Model versions">
                  {models.data.items.map((m) => (
                    <VersionRow key={m.version} m={m} selected={m.version === version} onSelect={() => setSelected(m.version)} />
                  ))}
                </ul>
              )}
            </Panel>
            <RetrainForm onFinished={refreshAll} running={!!jobs.data?.job_running} />
          </div>
          <div className="min-w-0">{version ? <ModelView key={version} version={version} activeVersion={models.data?.active_version ?? null} onActivated={refreshAll} /> : null}</div>
        </div>
        <JobHistory jobs={jobs.data?.items} error={jobs.error} />
      </Section>
    </>
  );
}

function VersionRow({ m, selected, onSelect }: { m: ModelSummary; selected: boolean; onSelect: () => void }) {
  return (
    <li>
      <button
        role="option"
        aria-selected={selected}
        onClick={onSelect}
        className={cn("w-full rounded-xl border p-3 text-left transition-colors", selected ? "border-violet/60 bg-violet/10" : "border-line hover:border-line-strong hover:bg-white/[0.02]")}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="font-display text-lg font-semibold text-ink">{m.version}</span>
          <StatusPill tone={m.status === "active" ? "good" : "neutral"}>{m.status}</StatusPill>
        </div>
        <p className="mt-1 text-sm text-ink-2">{modelLabel(m.model_type)}</p>
        <p className="mt-1 text-xs text-ink-3">
          Test ROC-AUC {num(m.metrics.test.roc_auc, 4)} · F1 {num(m.metrics.test.f1, 3)} · {timeAgo(m.created_at)}
        </p>
      </button>
    </li>
  );
}

function ModelView({ version, activeVersion, onActivated }: { version: string; activeVersion: string | null; onActivated: () => void }) {
  const q = useApi((s) => api.model(version, { signal: s }), [version]);
  const [confirm, setConfirm] = useState(false);
  const toast = useToast();
  const activate = useAction(() => api.activateModel(version));

  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} />;
  if (!q.data) return <LoadingBlock rows={10} />;
  const d: ModelDetail = q.data;
  const md = d.metadata;
  const t = md.metrics.test;
  const isActive = version === activeVersion;

  const doActivate = async () => {
    const res = await activate.run();
    setConfirm(false);
    if (res) {
      toast({ tone: "success", title: `Model ${version} activated`, body: "The API now serves this version; its artifact hash and golden examples were re-verified on load." });
      onActivated();
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <Panel luminous>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-3">{md.task}</p>
            <h2 className="mt-2 font-display text-3xl font-semibold text-ink">
              {version} · {md.model_label}
            </h2>
            <p className="mt-1 text-sm text-ink-2">
              Created {dateTime(d.created_at)} · feature set <Chip>{d.feature_set}</Chip> · threshold {num(d.threshold, 2)} · calibration {md.calibration.method}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {isActive ? (
              <StatusPill tone="good">Serving</StatusPill>
            ) : (
              <Button variant="secondary" size="sm" onClick={() => setConfirm(true)} icon={<RotateCcw className="size-3.5" aria-hidden="true" />}>
                Activate this version
              </Button>
            )}
            {d.consistency && <StatusPill tone={d.consistency.passed ? "good" : "critical"}>Golden check {d.consistency.passed ? "passed" : "failed"}</StatusPill>}
          </div>
        </div>
        <div className="mt-5">
          <MetricGrid
            items={[
              { label: "Test ROC-AUC", value: num(t.roc_auc, 4) },
              { label: "Test F1", value: num(t.f1, 4) },
              { label: "Test accuracy", value: pct(t.accuracy) },
              { label: "Avg precision", value: num(t.average_precision, 4) },
              { label: "Precision", value: num(t.precision, 4) },
              { label: "Recall", value: num(t.recall, 4) },
              { label: "Brier", value: num(t.brier, 4) },
              { label: "Test rows", value: int(t.support) },
            ]}
          />
        </div>
        {activate.error && <p className="mt-3 text-sm text-critical-text">{activate.error.message}</p>}
      </Panel>

      <Panel>
        <PanelHeader eyebrow="Data splits" title="Leakage-safe evaluation protocol" description={md.model_selection.protocol} />
        <DataTable
          columns={["Split", "Rows", "Placed rate", "Used for"]}
          numericFrom={1}
          rows={(["train", "validation", "test"] as const).map((k) => [k, int(md.splits[k].rows), pct(md.splits[k].positive_rate), md.splits.usage[k]])}
        />
      </Panel>

      <Panel>
        <PanelHeader eyebrow="Model selection" title="Cross-validated candidates (training split)" description={`Selected by mean ${md.model_selection.metric}. ${md.candidates_skipped.length ? `Skipped (not installed): ${md.candidates_skipped.join(", ")}.` : ""}`} />
        <DataTable
          columns={["Candidate", "ROC-AUC", "± std", "F1", "Accuracy", "Log loss", "Fit s / fold"]}
          rows={md.model_selection.results.map((r) => [
            <span className={cn(r.model === md.model_selection.selected && "font-medium text-ink")}>
              {r.label} {r.model === md.model_selection.selected && <CircleCheck className="ml-1 inline size-3.5 text-good-text" aria-label="selected" />}
            </span>,
            num(r.roc_auc.mean, 4),
            num(r.roc_auc.std, 4),
            num(r.f1.mean, 4),
            pct(r.accuracy.mean, 2),
            num(-r.neg_log_loss.mean, 4),
            num(r.fit_time_s, 2),
          ])}
        />
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <ConfusionMatrix cm={t.confusion_matrix} title="Test confusion matrix" subtitle={`Threshold ${num(t.threshold, 2)} (tuned on validation)`} />
        </Panel>
        <Panel>
          <ThresholdSweep data={md.curves.validation_threshold_sweep} threshold={md.threshold} title="Threshold tuning (validation split)" />
        </Panel>
        <Panel>
          <RocChart points={md.curves.test_roc} auc={t.roc_auc} title="Test ROC curve" />
        </Panel>
        <Panel>
          <PrChart points={md.curves.test_pr} ap={t.average_precision} baseline={md.splits.test.positive_rate} title="Test precision–recall" />
        </Panel>
      </div>

      <Panel>
        <ReliabilityChart
          title="Calibration"
          series={[
            { name: "Test", result: md.curves.test_calibration },
            { name: "Validation (uncalibrated)", result: md.curves.validation_calibration_uncalibrated },
          ]}
        />
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <HBars
            title="Permutation importance"
            subtitle="Drop in validation ROC-AUC when a feature is shuffled (correlated SGPA columns share importance)"
            data={md.feature_importance.permutation_validation_roc_auc.slice(0, 14).map((f) => ({ label: f.feature, value: f.importance }))}
            format={(v) => num(v, 4)}
          />
        </Panel>
        {md.feature_importance.impurity_top30 && (
          <Panel>
            <HBars title="Impurity / gain importance (encoded features)" data={md.feature_importance.impurity_top30.slice(0, 14).map((f) => ({ label: f.feature, value: f.importance }))} format={(v) => num(v, 3)} />
          </Panel>
        )}
      </div>

      <Panel>
        <PanelHeader eyebrow="Packaging" title="Artifact & schema" />
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2 text-sm text-ink-2">
            <p>
              Size {bytes(d.size_bytes)} · SHA-256 <span className="break-all font-mono text-xs text-ink">{d.sha256}</span>
            </p>
            <p>MLflow run: {d.mlflow_run_id ? <span className="font-mono text-xs text-ink">{d.mlflow_run_id}</span> : "not logged"}</p>
            <p>
              {md.features.length} contract features + engineered ({md.engineered_features.join(", ") || "none"}) → {md.transformed_features.length} model inputs.
            </p>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {Object.entries(md.library_versions).map(([k, v]) => (
                <Chip key={k}>
                  {k} {v}
                </Chip>
              ))}
            </div>
          </div>
          <div>
            <p className="mb-2 text-sm font-medium text-ink">Columns excluded from model inputs</p>
            <ul className="space-y-2 text-xs text-ink-2">
              {Object.entries(md.excluded_columns).map(([k, v]) => (
                <li key={k}>
                  <Chip>{k}</Chip> {v}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Panel>

      <Modal open={confirm} onClose={() => setConfirm(false)} title={`Activate ${version}?`}>
        <p className="text-sm text-ink-2">
          The API will switch from {activeVersion ?? "no model"} to {version} immediately. The artifact hash and golden examples are verified on load; if loading fails, the previous version stays active.
        </p>
        <div className="mt-6 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setConfirm(false)}>
            Cancel
          </Button>
          <Button onClick={doActivate} loading={activate.pending}>
            Activate {version}
          </Button>
        </div>
      </Modal>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ retraining */

const CANDIDATES = ["logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm", "decision_tree", "gradient_boosting", "adaboost"];

function RetrainForm({ onFinished, running }: { onFinished: () => void; running: boolean }) {
  const schema = useApi((s) => api.schema({ signal: s }), []);
  const available = schema.data?.available_classifiers ?? CANDIDATES;
  const [cands, setCands] = useState<string[]>(["logistic_regression", "hist_gradient_boosting", "lightgbm"]);
  const [featureSet, setFeatureSet] = useState<"full" | "pre_final_year">("full");
  const [calibration, setCalibration] = useState<"none" | "sigmoid" | "isotonic">("none");
  const [folds, setFolds] = useState(5);
  const [sample, setSample] = useState("full");
  const [minAuc, setMinAuc] = useState(0.75);
  const [engineered, setEngineered] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const toast = useToast();
  const start = useAction((cfg: TrainingConfigInput) => api.startTraining({ config: cfg }));
  const job = useApi((s) => api.trainingJob(jobId as string, { signal: s }), [jobId], { enabled: !!jobId, intervalMs: 1500 });
  const done = job.data && !["queued", "running"].includes(job.data.status);

  useEffect(() => {
    if (!done || !job.data) return;
    const j = job.data;
    toast({
      tone: j.status === "succeeded" ? "success" : j.status === "rejected" ? "info" : "error",
      title: j.status === "succeeded" ? `Promoted ${j.new_version}` : j.status === "rejected" ? "Candidate rejected; active model kept" : "Training failed; active model kept",
      body: j.reason ?? undefined,
    });
    onFinished();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

  const submit = async () => {
    const res = await start.run({
      candidates: cands,
      feature_set: featureSet,
      calibration,
      cv_folds: folds,
      engineered_features: engineered,
      min_validation_roc_auc: minAuc,
      ...(sample === "full" ? {} : { sample_rows: Number(sample) }),
    });
    if (res) setJobId(res.job_id);
  };

  const busy = running || (!!jobId && !done);
  return (
    <Panel>
      <PanelHeader eyebrow="Controlled retraining" title="Train a candidate" description="Validate → train → evaluate → compare with the active model → promote only if every gate passes." />
      <fieldset className="space-y-4" disabled={busy}>
        <div>
          <legend className="mb-2 text-[13px] font-medium text-ink-2">Candidates</legend>
          <div className="flex flex-wrap gap-1.5">
            {CANDIDATES.filter((c) => available.includes(c)).map((c) => {
              const on = cands.includes(c);
              return (
                <button
                  key={c}
                  type="button"
                  aria-pressed={on}
                  onClick={() => setCands((xs) => (on ? xs.filter((x) => x !== c) : [...xs, c]))}
                  className={cn("rounded-full border px-2.5 py-1 text-xs transition-colors", on ? "border-violet/60 bg-violet/15 text-ink" : "border-line text-ink-3 hover:text-ink")}
                >
                  {modelLabel(c)}
                </button>
              );
            })}
          </div>
        </div>
        <SelectField label="Feature set" value={featureSet} onChange={(v) => setFeatureSet(v as typeof featureSet)} options={[{ value: "full", label: "Full record (all 8 semesters)" }, { value: "pre_final_year", label: "Pre-final-year (point-in-time safe)" }]} />
        <SelectField label="Probability calibration" value={calibration} onChange={(v) => setCalibration(v as typeof calibration)} options={[{ value: "none", label: "None" }, { value: "sigmoid", label: "Platt (sigmoid)" }, { value: "isotonic", label: "Isotonic" }]} />
        <SelectField label="Training data" value={sample} onChange={setSample} options={[{ value: "full", label: "Full dataset (≈ 30–60 s)" }, { value: "20000", label: "20,000-row sample" }, { value: "8000", label: "8,000-row sample (fast)" }]} />
        <Slider label="CV folds" value={folds} min={2} max={10} step={1} onChange={setFolds} />
        <Slider label="Quality gate: min validation ROC-AUC" value={minAuc} min={0.5} max={0.999} step={0.001} onChange={setMinAuc} format={(v) => v.toFixed(3)} />
        <Toggle label="Engineered SGPA features" checked={engineered} onChange={setEngineered} hint="SGPA trend and volatility" />
      </fieldset>
      <Button className="mt-5 w-full" onClick={submit} disabled={!cands.length || busy} loading={start.pending || busy} icon={<Play className="size-3.5" aria-hidden="true" />}>
        {busy ? "Training job running" : "Start training job"}
      </Button>
      {start.error && <p className="mt-2 text-sm text-critical-text">{start.error.message}</p>}
      {jobId && job.data && <JobProgress job={job.data} />}
    </Panel>
  );
}

function JobProgress({ job }: { job: TrainingJob }) {
  return (
    <div className="mt-5 rounded-xl border border-line p-3" aria-live="polite">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-ink-3">job {job.job_id.slice(0, 8)}</span>
        <StatusPill tone={jobTone(job.status)}>{job.status}</StatusPill>
      </div>
      <ol className="mt-3 max-h-48 space-y-1 overflow-y-auto text-xs">
        {job.log.map((l, i) => (
          <li key={i} className={cn("flex gap-2", i === job.log.length - 1 ? "text-ink" : "text-ink-3")}>
            <span className="w-20 shrink-0 font-mono">{l.stage}</span>
            <span>{l.message}</span>
          </li>
        ))}
      </ol>
      {job.reason && <p className="mt-2 text-xs text-ink-2">{job.reason}</p>}
    </div>
  );
}

function JobHistory({ jobs, error }: { jobs?: TrainingJob[]; error?: Error }) {
  return (
    <Panel>
      <PanelHeader eyebrow="Audit" title="Training & retraining history" actions={<History className="size-4 text-ink-3" aria-hidden="true" />} />
      {error && !jobs ? (
        <ErrorState error={error} compact />
      ) : !jobs ? (
        <LoadingBlock rows={4} />
      ) : jobs.length === 0 ? (
        <EmptyState title="No training runs yet" />
      ) : (
        <DataTable
          numericFrom={9}
          columns={["When", "Trigger", "Status", "Candidate", "Val ROC-AUC", "Compared with", "Decision", "New version", "Reason"]}
          rows={jobs.map((j) => [
            timeAgo(j.created_at),
            j.trigger,
            <StatusPill tone={jobTone(j.status)}>{j.status}</StatusPill>,
            modelLabel(j.candidate_model_type),
            num(j.candidate_metrics?.validation?.roc_auc, 4),
            (j.comparison?.active_version as string) ?? "—",
            j.decision ?? "—",
            j.new_version ?? "—",
            <span className="block max-w-md whitespace-normal text-xs">{j.reason}</span>,
          ])}
        />
      )}
    </Panel>
  );
}

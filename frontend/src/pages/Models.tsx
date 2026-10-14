import { motion } from "motion/react";
import { Boxes, CircleCheck, Fingerprint, History, Play, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/endpoints";
import type { ModelDetail, ModelSummary, TrainingConfigInput, TrainingJob } from "../api/types";
import { ConfusionMatrix, CurveOverlay, HBars } from "../components/charts/Common";
import { COMPONENT } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { SelectField, Slider, Toggle } from "../components/ui/Fields";
import { Modal } from "../components/ui/Modal";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Chip, jobTone, modelStatusTone, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { useToast } from "../components/ui/Toast";
import { useHealth } from "../hooks/HealthContext";
import { useAction, useApi } from "../hooks/useApi";
import { bytes, cn, dateTime, int, num, pct, timeAgo } from "../lib/format";
import { ComparisonTable } from "./Evaluation";
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
        description="Every registered version with its configuration, decision policy, held-out metrics and integrity hash. Retraining validates the data, trains a candidate, compares it with the active model on validation PR-AUC and only then promotes it, or keeps it as a candidate for manual promotion."
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
          <StatusPill tone={modelStatusTone(m.status)}>{m.status}</StatusPill>
        </div>
        <p className="mt-1 text-sm text-ink-2">Hybrid · w = {num(m.policy?.fusion_weight_lr, 2)}</p>
        <p className="mt-1 text-xs text-ink-3">
          Test PR-AUC {num(m.metrics.test.hybrid_high_risk?.average_precision, 4)} · HIGH_RISK recall {num(m.metrics.test.hybrid_high_risk?.recall, 3)} · {timeAgo(m.created_at)}
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
  const integrity = useAction(() => api.modelIntegrity(version));

  if (q.error) return <ErrorState error={q.error} onRetry={q.refetch} />;
  if (!q.data) return <LoadingBlock rows={10} />;
  const d: ModelDetail = q.data;
  const md = d.metadata;
  const t = md.metrics.test;
  const isActive = version === activeVersion;
  const lr = md.components.logistic_regression;
  const iso = md.components.isolation_forest;

  const doActivate = async () => {
    const res = await activate.run();
    setConfirm(false);
    if (res) {
      toast({ tone: "success", title: `Model ${version} is now serving`, body: "Its artifact hash and golden examples were re-verified on load." });
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
              Created {dateTime(d.created_at)} · features <Chip>{d.feature_set}</Chip>
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {isActive ? (
              <StatusPill tone="good">Serving</StatusPill>
            ) : (
              <Button variant="secondary" size="sm" onClick={() => setConfirm(true)} icon={<RotateCcw className="size-3.5" aria-hidden="true" />}>
                {d.status === "candidate" ? "Promote candidate" : "Activate this version"}
              </Button>
            )}
            {d.consistency && <StatusPill tone={d.consistency.passed ? "good" : "critical"}>Golden check {d.consistency.passed ? "passed" : "failed"}</StatusPill>}
          </div>
        </div>
        <div className="mt-5">
          <ComparisonTable m={t} />
        </div>
        <p className="mt-3 text-xs text-ink-3">
          Held-out test split: {int(md.splits.test.rows)} rows, {md.splits.test.fraud_rows} fraud. Policy: w = {num(d.policy.fusion_weight_lr, 2)}, REVIEW ≥ {num(d.policy.review_threshold, 4)}, HIGH_RISK ≥{" "}
          {num(d.policy.high_risk_threshold, 4)}.
        </p>
        {activate.error && <p className="mt-3 text-sm text-critical-text">{activate.error.message}</p>}
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <PanelHeader eyebrow="Model A" title="Logistic Regression" />
          <dl className="grid grid-cols-2 gap-2 text-sm">
            {[
              ["C (by CV)", String(lr.C)],
              ["Imbalance", lr.imbalance],
              ["Training rows", `${int(lr.training_rows)} (${lr.training_fraud_rows} fraud)`],
              ["Prior correction", num(-lr.prior_correction_offset, 3)],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line px-3 py-2">
                <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">{k}</dt>
                <dd className="mt-0.5 font-mono text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-4">
            <DataTable
              columns={["C", "CV PR-AUC", "± std", "CV ROC-AUC", "Fit s / fold"]}
              rows={md.model_selection.results.map((r) => [
                <span className={cn(r.C === md.model_selection.selected_C && "font-medium text-ink")}>
                  {r.C} {r.C === md.model_selection.selected_C && <CircleCheck className="ml-1 inline size-3.5 text-good-text" aria-label="selected" />}
                </span>,
                num(r.average_precision.mean, 4),
                num(r.average_precision.std, 4),
                num(r.roc_auc.mean, 4),
                num(r.fit_time_s, 2),
              ])}
            />
            <p className="mt-2 text-xs text-ink-3">{md.model_selection.protocol}</p>
          </div>
        </Panel>
        <Panel>
          <PanelHeader eyebrow="Model B" title="Isolation Forest" />
          <dl className="grid grid-cols-2 gap-2 text-sm">
            {[
              ["Reference", iso.reference],
              ["Fit rows", int(iso.fit_rows)],
              ["Labels used to fit", iso.labels_used_for_fitting ? "legitimate rows only" : "no"],
              ["Trees", String(iso.params.n_estimators ?? "—")],
              ["max_samples", String(iso.params.max_samples ?? "—")],
              ["Flag", `top ${pct(1 - d.policy.if_flag_percentile, 2)}`],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line px-3 py-2">
                <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">{k}</dt>
                <dd className="mt-0.5 font-mono text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-ink-3">{iso.anomaly_score}</p>
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <ConfusionMatrix cm={t.hybrid_high_risk.confusion_matrix} title="Hybrid · HIGH_RISK (test)" subtitle={`Threshold ${num(d.policy.high_risk_threshold, 4)}, tuned on validation`} />
        </Panel>
        <Panel>
          <CurveOverlay
            title="Precision–recall (test)"
            series={[
              { name: "Logistic Regression", color: COMPONENT.logistic_regression, points: md.curves.test_pr.logistic_regression },
              { name: "Isolation Forest", color: COMPONENT.isolation_forest, points: md.curves.test_pr.isolation_forest },
              { name: "Hybrid", color: COMPONENT.hybrid, points: md.curves.test_pr.hybrid, dashed: true },
            ]}
            x="recall"
            y="precision"
            xLabel="Recall"
            yLabel="Precision"
            step
            height={260}
          />
        </Panel>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <HBars
            title="LR coefficients (standardised inputs)"
            subtitle="Positive pushes towards fraud. V-components are anonymised; magnitudes are comparable because inputs are standardised."
            data={md.feature_importance.lr_coefficients.slice(0, 14).map((f) => ({ label: f.label, value: f.coefficient }))}
            format={(v) => num(v, 3)}
            diverging
            valueLabel="Coefficient"
          />
        </Panel>
        <Panel>
          <HBars
            title="LR permutation importance (validation)"
            subtitle="Drop in validation PR-AUC when one raw feature is shuffled"
            data={md.feature_importance.lr_permutation_validation_ap.slice(0, 14).map((f) => ({ label: f.feature, value: f.importance }))}
            format={(v) => num(v, 4)}
            valueLabel="PR-AUC drop"
          />
        </Panel>
      </div>

      <Panel>
        <PanelHeader
          eyebrow="Packaging"
          title="Artifact integrity & reproducibility"
          actions={
            <Button size="sm" variant="secondary" loading={integrity.pending} onClick={() => integrity.run()} icon={<Fingerprint className="size-3.5" aria-hidden="true" />}>
              Re-hash artifact
            </Button>
          }
        />
        {integrity.data && (
          <p className={cn("mb-3 text-sm", integrity.data.matches ? "text-good-text" : "text-critical-text")}>
            {integrity.data.matches ? "SHA-256 matches the registry record." : integrity.data.exists ? "SHA-256 does NOT match the registry record: the artifact was modified." : "The artifact file is missing."}
          </p>
        )}
        {integrity.error && <p className="mb-3 text-sm text-critical-text">{integrity.error.message}</p>}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2 text-sm text-ink-2">
            <p>
              Size {bytes(d.size_bytes)} · SHA-256 <span className="break-all font-mono text-xs text-ink">{d.sha256}</span>
            </p>
            <p>MLflow run: {d.mlflow_run_id ? <span className="font-mono text-xs text-ink">{d.mlflow_run_id}</span> : "not logged"}</p>
            {md.dataset && (
              <p>
                Dataset <span className="font-mono text-xs text-ink">{md.dataset.name}</span> (sha256 {md.dataset.sha256.slice(0, 12)}…, {int(md.dataset.rows)} rows) · split seed {md.splits.seed}
              </p>
            )}
            <p>
              {md.features.length} contract features → {md.transformed_features.length} model inputs ({md.transformed_features.slice(-3).join(", ")} engineered).
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
            <p className="mb-2 text-sm font-medium text-ink">Training configuration</p>
            <pre className="max-h-64 overflow-auto rounded-lg bg-white/[0.03] p-3 font-mono text-[11px] text-ink-2">{JSON.stringify(md.config, null, 2)}</pre>
          </div>
        </div>
      </Panel>

      <Modal open={confirm} onClose={() => setConfirm(false)} title={`${d.status === "candidate" ? "Promote" : "Activate"} ${version}?`}>
        <p className="text-sm text-ink-2">
          The API will switch from {activeVersion ?? "no model"} to {version} immediately. The artifact hash and golden examples are verified on load; if either fails, the previous version stays active.
        </p>
        <div className="mt-6 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setConfirm(false)}>
            Cancel
          </Button>
          <Button onClick={doActivate} loading={activate.pending}>
            {d.status === "candidate" ? "Promote" : "Activate"} {version}
          </Button>
        </div>
      </Modal>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ retraining */

function RetrainForm({ onFinished, running }: { onFinished: () => void; running: boolean }) {
  const [imbalance, setImbalance] = useState<"class_weight" | "undersample" | "none">("class_weight");
  const [ifReference, setIfReference] = useState<"all_train" | "legitimate_train">("all_train");
  const [sample, setSample] = useState("full");
  const [folds, setFolds] = useState(5);
  const [recallTarget, setRecallTarget] = useState(0.9);
  const [maxReview, setMaxReview] = useState(0.01);
  const [minAp, setMinAp] = useState(0.3);
  const [fixedWeight, setFixedWeight] = useState(false);
  const [weight, setWeight] = useState(1);
  const [autoPromote, setAutoPromote] = useState(true);
  const [dedupe, setDedupe] = useState(true);
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
      title:
        j.decision === "promoted" ? `Promoted ${j.new_version}` : j.decision === "registered" ? `Registered ${j.new_version} as a candidate` : j.status === "rejected" ? "Candidate rejected; active model kept" : "Training failed; active model kept",
      body: j.reason ?? undefined,
    });
    onFinished();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

  const submit = async () => {
    const res = await start.run({
      lr_imbalance: imbalance,
      if_reference: ifReference,
      cv_folds: folds,
      review_recall_target: recallTarget,
      max_review_rate: maxReview,
      min_validation_average_precision: minAp,
      fusion_weight: fixedWeight ? weight : null,
      auto_promote: autoPromote,
      drop_duplicates: dedupe,
      ...(sample === "full" ? {} : { sample_rows: Number(sample) }),
    });
    if (res) setJobId(res.job_id);
  };

  const busy = running || (!!jobId && !done);
  return (
    <Panel>
      <PanelHeader eyebrow="Controlled retraining" title="Train a candidate" description="Validate → train LR + Isolation Forest → tune on validation → evaluate once on test → compare with the active model → promote only if every gate passes." />
      <fieldset className="space-y-4" disabled={busy}>
        <SelectField
          label="LR imbalance handling"
          value={imbalance}
          onChange={(v) => setImbalance(v as typeof imbalance)}
          options={[
            { value: "class_weight", label: "Balanced class weights" },
            { value: "undersample", label: "Undersample legitimate rows (training folds only)" },
            { value: "none", label: "None" },
          ]}
        />
        <SelectField
          label="Isolation Forest reference"
          value={ifReference}
          onChange={(v) => setIfReference(v as typeof ifReference)}
          options={[
            { value: "all_train", label: "All training rows (labels unused)" },
            { value: "legitimate_train", label: "Legitimate training rows only" },
          ]}
        />
        <SelectField
          label="Training data"
          value={sample}
          onChange={setSample}
          options={[
            { value: "full", label: "Full dataset (≈ 20–40 s)" },
            { value: "100000", label: "100,000-row stratified sample" },
            { value: "40000", label: "40,000-row stratified sample (fast)" },
          ]}
        />
        <Slider label="CV folds (LR C)" value={folds} min={2} max={10} step={1} onChange={setFolds} />
        <Slider label="REVIEW recall target (validation)" value={recallTarget} min={0.5} max={1} step={0.01} onChange={setRecallTarget} format={(v) => v.toFixed(2)} />
        <Slider label="Max REVIEW-or-above rate" value={maxReview} min={0.001} max={0.05} step={0.001} onChange={setMaxReview} format={(v) => pct(v, 1)} />
        <Slider label="Quality gate: min validation PR-AUC" value={minAp} min={0} max={0.95} step={0.01} onChange={setMinAp} format={(v) => v.toFixed(2)} />
        <Toggle label="Fix the fusion weight" checked={fixedWeight} onChange={setFixedWeight} hint="Off = choose w on the validation split" />
        {fixedWeight && <Slider label="Weight on LR (w)" value={weight} min={0} max={1} step={0.05} onChange={setWeight} format={(v) => v.toFixed(2)} />}
        <Toggle label="Drop exact duplicates before splitting" checked={dedupe} onChange={setDedupe} hint="Prevents identical rows in train and test" />
        <Toggle label="Promote automatically if every gate passes" checked={autoPromote} onChange={setAutoPromote} hint="Off = register as a candidate for manual promotion" />
      </fieldset>
      <Button className="mt-5 w-full" onClick={submit} disabled={busy} loading={start.pending || busy} icon={<Play className="size-3.5" aria-hidden="true" />}>
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
          columns={["When", "Trigger", "Status", "Val PR-AUC", "Test PR-AUC", "Compared with", "Decision", "Version", "Reason"]}
          rows={jobs.map((j) => [
            timeAgo(j.created_at),
            j.trigger,
            <StatusPill tone={jobTone(j.status)}>{j.status}</StatusPill>,
            num(j.candidate_metrics?.validation?.hybrid_high_risk?.average_precision, 4),
            num(j.candidate_metrics?.test?.hybrid_high_risk?.average_precision, 4),
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

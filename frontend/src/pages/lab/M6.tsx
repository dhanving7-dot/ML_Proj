import { CircleCheck, CircleX, ShieldAlert, ShieldCheck } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartFrame } from "../../components/charts/ChartFrame";
import { ChartTooltip } from "../../components/charts/ChartTooltip";
import { MetricGrid } from "../../components/charts/Common";
import { axisProps, CHROME, gridProps, STATUS } from "../../components/charts/theme";
import { Panel } from "../../components/ui/Panel";
import { SelectField, Slider } from "../../components/ui/Fields";
import { driftTone, StatusPill } from "../../components/ui/Status";
import { bytes, cn, int, modelLabel, ms, num, pct } from "../../lib/format";
import type { ModuleProps } from "../Lab";
import { BarsChart, Card, DataTable, ExperimentShell, Note, useExperiment } from "./kit";
import { ModuleLayout } from "./ModuleLayout";

export function M6(props: ModuleProps) {
  return (
    <ModuleLayout
      id="m6"
      {...props}
      intro={
        <>
          Production ML engineering with the <strong className="text-ink">active hybrid model</strong>: how training-serving skew happens and is caught, why point-in-time correctness matters for
          transactions ordered in time, how models are packaged, and how drift is measured. Feature stores such as Feast are covered conceptually; here the shared <code className="font-mono text-ink">ml.contract</code> /{" "}
          <code className="font-mono text-ink">ml.features</code> package plays that role.
        </>
      }
      experiments={[
        { key: "skew", label: "Training-serving skew", Component: SkewExp },
        { key: "point-in-time", label: "Point-in-time", Component: PitExp },
        { key: "drift-sandbox", label: "Drift sandbox", Component: DriftExp },
        { key: "onnx", label: "ONNX export", Component: OnnxExp },
        { key: "packaging", label: "Packaging", Component: PackagingExp },
      ]}
    />
  );
}

/* ------------------------------------------------------------------ skew */

interface Scenario {
  id: string;
  title: string;
  cause: string;
  rows_in_batch: number;
  frauds_in_batch: number;
  mean_abs_hybrid_change: number;
  decision_change_rate: number;
  fraud_flagged_correct: number;
  fraud_flagged_skewed: number;
  average_precision_correct: number | null;
  average_precision_skewed: number | null;
  caught_by_contract_validation: boolean;
  rows_with_out_of_range_warning: number;
}
interface SkewRes {
  model_version: string;
  golden_test: { rows: number; max_feature_vector_diff: number; max_hybrid_score_diff: number; decision_mismatches: number; passed: boolean };
  scenarios: Scenario[];
  prevention: string[];
}

function SkewExp() {
  const exp = useExperiment<SkewRes>("m6", "skew", {}, { auto: true });
  return (
    <ExperimentShell exp={exp} title="Training-serving skew" topics={["Golden E2E test", "Feature-vector diff", "Re-fitted scaler", "Unit mismatch", "Swapped columns", "Double transform"]} estimate="≈ 2–4 s">
      {(d) => {
        const r = d.result;
        const g = r.golden_test;
        return (
          <>
            <Panel luminous>
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-3">Golden end-to-end test · model {r.model_version}</p>
                  <h4 className="mt-2 text-xl font-semibold text-ink">Raw CSV → training path vs. JSON → API validation → serving path</h4>
                  <p className="mt-2 max-w-2xl text-sm text-ink-2">
                    {g.rows.toLocaleString()} validation transactions (every validation fraud included) were pushed through both paths and the feature vectors compared element by element, not
                    just the decisions.
                  </p>
                </div>
                <StatusPill tone={g.passed ? "good" : "critical"}>{g.passed ? "Paths identical" : "Skew detected"}</StatusPill>
              </div>
              <div className="mt-5">
                <MetricGrid
                  items={[
                    { label: "Max feature diff", value: g.max_feature_vector_diff.toExponential(1) },
                    { label: "Max hybrid diff", value: g.max_hybrid_score_diff.toExponential(1) },
                    { label: "Decision mismatches", value: String(g.decision_mismatches) },
                  ]}
                />
              </div>
            </Panel>
            <div className="grid gap-4 md:grid-cols-2">
              {r.scenarios.map((s) => (
                <Card key={s.id}>
                  <div className="flex items-start justify-between gap-3">
                    <h4 className="text-base font-semibold text-ink">{s.title}</h4>
                    {s.caught_by_contract_validation ? <StatusPill tone="good">Rejected by contract</StatusPill> : <StatusPill tone="warning">Passes validation silently</StatusPill>}
                  </div>
                  <p className="mt-2 text-sm text-ink-2">{s.cause}</p>
                  <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Decisions changed</dt>
                      <dd className="font-display text-2xl font-semibold text-ink">{pct(s.decision_change_rate, 2)}</dd>
                      <dd className="text-[11px] text-ink-3">of {s.rows_in_batch.toLocaleString()} affected rows</dd>
                    </div>
                    <div>
                      <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">PR-AUC</dt>
                      <dd className="font-display text-2xl font-semibold text-ink">
                        {num(s.average_precision_skewed, 3)} <span className="text-sm font-normal text-ink-3">vs {num(s.average_precision_correct, 3)}</span>
                      </dd>
                      <dd className="text-[11px] text-ink-3">skewed vs. correct serving</dd>
                    </div>
                    <div>
                      <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Frauds flagged</dt>
                      <dd className="font-mono text-ink">
                        {s.fraud_flagged_skewed} <span className="text-ink-3">vs {s.fraud_flagged_correct} of {s.frauds_in_batch}</span>
                      </dd>
                    </div>
                    <div>
                      <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Out-of-range warnings</dt>
                      <dd className="font-mono text-ink">{pct(s.rows_with_out_of_range_warning, 1)}</dd>
                    </div>
                  </dl>
                </Card>
              ))}
            </div>
            <Card title="How this project prevents and detects skew">
              <ul className="space-y-2 text-sm text-ink-2">
                {r.prevention.map((p) => (
                  <li key={p} className="flex gap-2">
                    <ShieldCheck className="mt-0.5 size-4 shrink-0 text-good-text" aria-hidden="true" />
                    {p}
                  </li>
                ))}
              </ul>
              <div className="mt-4">
                <Note tone="warn">
                  None of these mistakes breaks the schema: cents and hours are still non-negative numbers, swapped columns have valid names. Schema checks catch type, range and naming errors, not
                  changes in meaning; the out-of-range warning and drift monitoring (PSI on Amount and Time) are the backstop.
                </Note>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ point in time */

interface SplitScore {
  train_rows: number;
  train_frauds: number;
  eval_rows: number;
  eval_frauds: number;
  average_precision: number;
  roc_auc: number;
  train_time_range_h?: number[];
  eval_time_range_h?: number[];
}
interface PitRes {
  model: string;
  random_vs_time: { random_80_20: SplitScore; time_ordered_80_20: SplitScore; difference_time_minus_random: number };
  explanation: string;
  semantic_mismatch: {
    feature: string;
    training_definition: string;
    serving_definition: string;
    hour_of_cycle_shift_h: number;
    validation_ap_training_definition: number;
    validation_ap_serving_definition: number;
    mean_abs_probability_change: number;
    explanation: string;
  };
}

function PitExp() {
  const exp = useExperiment<PitRes>("m6", "point-in-time", { model: "logistic_regression", time_shift_hours: 6 }, { auto: false });
  return (
    <ExperimentShell
      exp={exp}
      title="Point-in-time correctness"
      topics={["Random vs time-ordered split", "Future leakage", "Same name, different meaning"]}
      estimate="≈ 5–20 s"
      controls={
        <>
          <SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={["logistic_regression", "hist_gradient_boosting"].map((m) => ({ value: m, label: modelLabel(m) }))} />
          <Slider label="Serving Time origin shift (hours)" value={Number(exp.params.time_shift_hours)} min={1} max={23} step={1} onChange={(v) => exp.setParam("time_shift_hours", v)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const rt = r.random_vs_time;
        const sm = r.semantic_mismatch;
        return (
          <>
            <Card title="Does training on 'the future' inflate the score?" description={r.explanation}>
              <DataTable
                columns={["Split of train + validation rows", "Train rows (fraud)", "Eval rows (fraud)", "PR-AUC", "ROC-AUC"]}
                numericFrom={1}
                rows={[
                  ["Random 80 / 20", `${int(rt.random_80_20.train_rows)} (${rt.random_80_20.train_frauds})`, `${int(rt.random_80_20.eval_rows)} (${rt.random_80_20.eval_frauds})`, num(rt.random_80_20.average_precision, 4), num(rt.random_80_20.roc_auc, 4)],
                  [
                    `Time-ordered (train ${rt.time_ordered_80_20.train_time_range_h?.join("–")} h, evaluate ${rt.time_ordered_80_20.eval_time_range_h?.join("–")} h)`,
                    `${int(rt.time_ordered_80_20.train_rows)} (${rt.time_ordered_80_20.train_frauds})`,
                    `${int(rt.time_ordered_80_20.eval_rows)} (${rt.time_ordered_80_20.eval_frauds})`,
                    num(rt.time_ordered_80_20.average_precision, 4),
                    num(rt.time_ordered_80_20.roc_auc, 4),
                  ],
                ]}
              />
              <p className="mt-3 text-xs text-ink-3">Measured difference (time-ordered − random): {num(rt.difference_time_minus_random, 4)} PR-AUC.</p>
            </Card>
            <Card title={`Same name, different meaning: ${sm.feature}`} description={sm.explanation}>
              <DataTable
                columns={["", "Definition", "Validation PR-AUC"]}
                numericFrom={2}
                rows={[
                  ["Training", sm.training_definition, num(sm.validation_ap_training_definition, 4)],
                  ["Serving", sm.serving_definition, num(sm.validation_ap_serving_definition, 4)],
                ]}
              />
              <p className="mt-3 text-xs text-ink-3">
                The hour-of-cycle moves by {num(sm.hour_of_cycle_shift_h, 1)} h; mean absolute change in P(fraud) is {num(sm.mean_abs_probability_change, 5)}. No validation error is raised.
              </p>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ drift sandbox */

interface SegPerf {
  rows: number;
  fraud_rows: number;
  fraud_rate: number;
  flagged_rate: number;
  recall_review_or_high: number | null;
  average_precision: number | null;
}
interface DriftRes {
  segment: string;
  title: string;
  segments: Record<string, string>;
  feature_drift: { feature: string; psi: number; kl_divergence: number; status: string }[];
  score_drift: Record<string, { psi: number; kl_divergence: number; status: string }>;
  detail: Record<string, { bins: { bin: string; expected: number; actual: number }[] }>;
  performance: { overall_validation: SegPerf; segment: SegPerf };
  thresholds: { stable_below: number; significant_at_or_above: number };
  explanation: string;
}

const statusColor = (s: string) => (s === "stable" ? STATUS.good : s === "investigate" ? STATUS.warning : STATUS.critical);

export function PsiChart({ data, thresholds, height }: { data: { feature: string; psi: number | null; status: string }[]; thresholds: { stable_below: number; significant_at_or_above: number }; height?: number }) {
  const rows = data.slice(0, 15).map((d) => ({ ...d, psi: d.psi ?? 0 }));
  return (
    <ChartFrame
      title="Population Stability Index per feature"
      subtitle={`PSI < ${thresholds.stable_below} stable · ${thresholds.stable_below}–${thresholds.significant_at_or_above} investigate · ≥ ${thresholds.significant_at_or_above} significant shift (dashed lines)`}
      legend={[
        { label: "Stable", color: STATUS.good },
        { label: "Investigate", color: STATUS.warning },
        { label: "Significant shift", color: STATUS.critical },
      ]}
      table={{ columns: ["Feature", "PSI", "Status"], rows: data.map((d) => [d.feature, num(d.psi, 4), d.status.replace("_", " ")]) }}
      height={height ?? Math.max(220, rows.length * 24 + 40)}
    >
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }} barCategoryGap={4}>
          <CartesianGrid {...gridProps} horizontal={false} vertical />
          <XAxis type="number" domain={[0, (max: number) => Math.max(0.3, Math.ceil(max * 10) / 10)]} {...axisProps} tickFormatter={(v) => num(v, 2)} />
          <YAxis type="category" dataKey="feature" {...axisProps} width={140} tick={{ fill: CHROME.ink2, fontSize: 11 }} interval={0} />
          <ReferenceLine x={thresholds.stable_below} stroke={CHROME.ink2} strokeDasharray="4 4" />
          <ReferenceLine x={thresholds.significant_at_or_above} stroke={CHROME.ink2} strokeDasharray="4 4" />
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={(v) => num(v, 4)} />} />
          <Bar dataKey="psi" name="PSI" radius={4}>
            {rows.map((r) => (
              <Cell key={r.feature} fill={statusColor(r.status)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

function DriftExp() {
  const exp = useExperiment<DriftRes>("m6", "drift-sandbox", { segment: "high_amount" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Drift sandbox"
      topics={["PSI", "KL divergence", "Score drift", "Input drift vs. performance"]}
      estimate="≈ 2–4 s"
      controls={
        <SelectField
          label="Serving population"
          value={String(exp.params.segment)}
          onChange={(v) => exp.setParam("segment", v)}
          options={[
            { value: "random", label: "Random sample (control)" },
            { value: "high_amount", label: "Top 5 % of amounts" },
            { value: "night", label: "Hours 0–5 of the cycle" },
            { value: "second_day", label: "Second day only" },
            { value: "fraud_only", label: "Fraud rows only" },
          ]}
        />
      }
    >
      {(d) => {
        const r = d.result;
        const top = r.feature_drift[0];
        const bins = r.detail[top?.feature]?.bins ?? [];
        const o = r.performance.overall_validation;
        const s = r.performance.segment;
        return (
          <>
            <Card title={r.title} description={r.explanation}>
              <DataTable
                columns={["Population", "Rows", "Fraud rows", "Fraud rate", "Flagged rate", "REVIEW+ recall", "PR-AUC"]}
                rows={[
                  ["All validation transactions", int(o.rows), o.fraud_rows, pct(o.fraud_rate, 3), pct(o.flagged_rate, 2), num(o.recall_review_or_high, 3), num(o.average_precision, 4)],
                  ["This segment", int(s.rows), s.fraud_rows, pct(s.fraud_rate, 3), pct(s.flagged_rate, 2), num(s.recall_review_or_high, 3), num(s.average_precision, 4)],
                ]}
              />
              <div className="mt-4 flex flex-wrap items-center gap-2 text-sm text-ink-2">
                Score drift:
                {Object.entries(r.score_drift).map(([k, v]) => (
                  <StatusPill key={k} tone={driftTone(v.status)}>
                    {k.replace(/_/g, " ")} PSI {num(v.psi, 3)}
                  </StatusPill>
                ))}
              </div>
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <PsiChart data={r.feature_drift} thresholds={r.thresholds} />
              </Card>
              <Card>
                <BarsChart
                  title={`Most-shifted feature: ${top?.feature}`}
                  subtitle={`Share of rows per training-decile bin · PSI ${num(top?.psi, 3)} · KL ${num(top?.kl_divergence, 3)} bits`}
                  data={bins}
                  x="bin"
                  series={[
                    { key: "expected", name: "Training reference" },
                    { key: "actual", name: "Serving segment" },
                  ]}
                  yFormat={(v) => pct(v, 0)}
                  height={280}
                />
              </Card>
            </div>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ onnx */

interface OnnxRes {
  available: boolean;
  converted?: boolean;
  error?: string;
  model?: string;
  variant_note?: string;
  opset?: number;
  inputs?: number;
  graph_nodes?: number;
  onnx_bytes?: number;
  joblib_bytes?: number;
  convert_ms?: number;
  parity?: { rows: number; max_abs_probability_diff: number; mean_abs_probability_diff: number; label_agreement: number; note: string };
  latency_ms?: { sklearn_batch: number; onnxruntime_batch: number };
}

function OnnxExp() {
  const exp = useExperiment<OnnxRes>("m6", "onnx", { model: "logistic_regression" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="ONNX portability"
      topics={["ONNX", "skl2onnx", "ONNX Runtime", "Prediction parity"]}
      estimate="≈ 1–2 s"
      controls={<SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={["logistic_regression", "hist_gradient_boosting", "random_forest"].map((m) => ({ value: m, label: modelLabel(m) }))} />}
    >
      {(d) => {
        const r = d.result;
        if (!r.available || !r.converted || !r.parity) return <Note tone="warn">ONNX experiment unavailable: {r.error}</Note>;
        const ok = r.parity.max_abs_probability_diff < 1e-4;
        return (
          <>
            <Card title={`${modelLabel(r.model)} exported to ONNX (opset ${r.opset})`} description={r.variant_note}>
              <MetricGrid
                items={[
                  { label: "Max |Δp|", value: r.parity.max_abs_probability_diff.toExponential(2), hint: `${r.parity.rows.toLocaleString()} validation rows` },
                  { label: "Mean |Δp|", value: r.parity.mean_abs_probability_diff.toExponential(2) },
                  { label: "Label agreement", value: pct(r.parity.label_agreement, 2) },
                  { label: "Graph nodes", value: String(r.graph_nodes), hint: `${r.inputs} named inputs` },
                  { label: "ONNX size", value: bytes(r.onnx_bytes) },
                  { label: "Joblib size", value: bytes(r.joblib_bytes) },
                  { label: "sklearn batch", value: ms(r.latency_ms?.sklearn_batch) },
                  { label: "ONNX Runtime batch", value: ms(r.latency_ms?.onnxruntime_batch) },
                ]}
              />
              <div className={cn("mt-4 flex items-start gap-2 rounded-xl border p-3 text-sm", ok ? "border-good/30 bg-good/[0.05]" : "border-warning/30 bg-warning/[0.05]")}>
                {ok ? <CircleCheck className="mt-0.5 size-4 shrink-0 text-good-text" aria-hidden="true" /> : <ShieldAlert className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden="true" />}
                <p className="text-ink-2">{r.parity.note}</p>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ packaging */

interface PackRes {
  model_version: string;
  artifact: { path_name: string; sha256: string; library_versions: Record<string, string>; contents: string[] };
  serialization: Record<string, { bytes: number; dump_ms: number; load_ms: number }>;
  security: string;
  grpc: string;
}

function PackagingExp() {
  const exp = useExperiment<PackRes>("m6", "packaging", {}, { auto: true });
  return (
    <ExperimentShell exp={exp} title="Model packaging" topics={["Pickle", "Joblib", "Compression", "Artifact integrity", "gRPC (conceptual)"]} estimate="< 1 s">
      {(d) => {
        const r = d.result;
        const rows = Object.entries(r.serialization).map(([k, v]) => ({ format: k.replace("joblib_compress_", "joblib c="), ...v, kb: v.bytes / 1024 }));
        return (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title={`Serialising active model ${r.model_version}`} description="Measured now on this machine.">
                <DataTable columns={["Format", "Size", "Dump", "Load"]} rows={rows.map((x) => [x.format, bytes(x.bytes), ms(x.dump_ms), ms(x.load_ms)])} />
              </Card>
              <Card>
                <BarsChart title="Artifact size by format" data={rows} x="format" series={[{ key: "kb", name: "Size (KB)" }]} yFormat={(v) => `${Math.round(v)}`} height={220} />
              </Card>
            </div>
            <Card title="What ships in a model version">
              <ul className="space-y-1.5 text-sm text-ink-2">
                {r.artifact.contents.map((c) => (
                  <li key={c}>• {c}</li>
                ))}
              </ul>
              <p className="mt-3 break-all font-mono text-xs text-ink-3">sha256 {r.artifact.sha256}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {Object.entries(r.artifact.library_versions ?? {}).map(([k, v]) => (
                  <span key={k} className="rounded-md border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2">
                    {k} {v}
                  </span>
                ))}
              </div>
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Security">
                <p className="flex gap-2 text-sm text-ink-2">
                  <CircleX className="mt-0.5 size-4 shrink-0 text-critical-text" aria-hidden="true" />
                  {r.security}
                </p>
              </Card>
              <Card title="gRPC (conceptual)">
                <p className="text-sm text-ink-2">{r.grpc}</p>
              </Card>
            </div>
          </>
        );
      }}
    </ExperimentShell>
  );
}

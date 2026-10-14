import { AnimatePresence, motion } from "motion/react";
import { Clock, Download, FileUp, Layers, ShieldAlert, ShieldCheck, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { api } from "../api/endpoints";
import type { BatchResponse, FieldSchema, PredictionResponse, SampleRow, TransactionInput } from "../api/types";
import { DECISION_COLOR } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { RiskResult, type GroundTruth } from "../components/predict/RiskResult";
import { Button } from "../components/ui/Button";
import { NumberField } from "../components/ui/Fields";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Chip } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { TabPanel, Tabs } from "../components/ui/Tabs";
import { useAction, useApi } from "../hooks/useApi";
import { cn, decisionLabel, int, num, pct, prob } from "../lib/format";

type Mode = "single" | "batch";

export default function Analyze() {
  const [mode, setMode] = useState<Mode>("single");
  return (
    <>
      <PageHeader
        crumb="Analyze"
        title="Analyze a transaction"
        description="Enter the 30 features of a transaction or replay a real one from the held-out test split. Logistic Regression, the Isolation Forest and the hybrid decision are computed by the API, never in the browser."
      />
      <Section>
        <Tabs
          idPrefix="analyze-mode"
          ariaLabel="Analysis mode"
          value={mode}
          onChange={setMode}
          items={[
            { key: "single", label: "Single transaction" },
            { key: "batch", label: "Batch (CSV)" },
          ]}
          className="mb-6 w-fit"
        />
        <TabPanel idPrefix="analyze-mode" value={mode}>
          {mode === "single" ? <SingleAnalysis /> : <BatchAnalysis />}
        </TabPanel>
      </Section>
    </>
  );
}

/* ------------------------------------------------------------------ single */

type FormValues = Record<string, string>;

function toForm(features: Record<string, number>): FormValues {
  return Object.fromEntries(Object.entries(features).map(([k, v]) => [k, String(v)]));
}

export function validateTransaction(fields: FieldSchema[], values: FormValues): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const f of fields) {
    const raw = (values[f.name] ?? "").trim();
    if (raw === "") {
      errors[f.name] = "Required";
      continue;
    }
    const v = Number(raw);
    if (!Number.isFinite(v)) errors[f.name] = "Enter a finite number";
    else if (f.minimum != null && v < f.minimum) errors[f.name] = `Must be ≥ ${f.minimum}`;
    else if (f.maximum != null && v > f.maximum) errors[f.name] = `Must be ≤ ${f.maximum}`;
  }
  const id = (values.transaction_id ?? "").trim();
  if (id && !/^[A-Za-z0-9_.:-]{1,64}$/.test(id)) errors.transaction_id = "Letters, digits and _ . : - only (max 64)";
  return errors;
}

function toPayload(fields: FieldSchema[], values: FormValues): TransactionInput {
  const out: TransactionInput = {};
  for (const f of fields) out[f.name] = Number((values[f.name] ?? "").trim());
  const id = (values.transaction_id ?? "").trim();
  if (id) out.transaction_id = id;
  return out;
}

function SingleAnalysis() {
  const schema = useApi((s) => api.schema({ signal: s }), []);
  const datasets = useApi((s) => api.datasets({ signal: s }), []);
  const defaultId = datasets.data?.items.find((d) => d.source === "default")?.id;
  const [values, setValues] = useState<FormValues>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [sample, setSample] = useState<SampleRow | null>(null);
  const [dirty, setDirty] = useState(false);
  const [result, setResult] = useState<{ res: PredictionResponse; truth?: GroundTruth } | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const predict = useAction((payload: TransactionInput) => api.predict(payload));
  const loader = useAction((id: number, label?: 0 | 1) => api.datasetSample(id, { seed: Math.floor(Math.random() * 1_000_000), label }));
  const fields = schema.data?.request_fields ?? [];
  const ranges = schema.data?.feature_ranges ?? null;

  const loadSample = async (label?: 0 | 1) => {
    if (!defaultId) return;
    const res = await loader.run(defaultId, label);
    const row = res?.items[0];
    if (!row) return;
    setValues({ ...toForm(row.features), transaction_id: `replay-${row.row_id}` });
    setSample(row);
    setDirty(false);
    setErrors({});
  };

  // Pre-fill with one real held-out transaction so the form is immediately usable.
  useEffect(() => {
    if (defaultId && !Object.keys(values).length) void loadSample(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultId]);

  const set = (name: string, v: string) => {
    setValues((s) => ({ ...s, [name]: v }));
    setDirty(true);
    if (errors[name]) setErrors(({ [name]: _, ...rest }) => rest);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const errs = validateTransaction(fields, values);
    setErrors(errs);
    if (Object.keys(errs).length) {
      document.querySelector<HTMLElement>("[aria-invalid='true']")?.focus();
      return;
    }
    const res = await predict.run(toPayload(fields, values));
    if (res) {
      setResult({ res, truth: sample && !dirty ? { rowId: sample.row_id, actual: sample.actual_class } : undefined });
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }));
    }
  };

  // Map server-side validation errors onto the fields.
  useEffect(() => {
    if (predict.error?.fieldErrors.length) {
      setErrors(Object.fromEntries(predict.error.fieldErrors.map((f) => [f.field, f.message])));
    }
  }, [predict.error]);

  const basics = useMemo(() => fields.filter((f) => f.group !== "pca"), [fields]);
  const pca = useMemo(() => fields.filter((f) => f.group === "pca"), [fields]);
  const outside = useMemo(
    () =>
      ranges
        ? fields.filter((f) => {
            const v = Number(values[f.name]);
            const r = ranges[f.name];
            return values[f.name] !== undefined && values[f.name] !== "" && Number.isFinite(v) && r && (v < r[0] || v > r[1]);
          }).length
        : 0,
    [fields, values, ranges],
  );

  if (schema.error) return <ErrorState error={schema.error} onRetry={schema.refetch} title="The feature schema could not be loaded" />;
  if (!schema.data) return <LoadingBlock rows={8} label="Loading form" />;

  const field = (f: FieldSchema) => (
    <NumberField
      key={f.name}
      name={f.name}
      label={f.name === "Time" ? "Time (s since first tx)" : f.name}
      value={values[f.name] ?? ""}
      onChange={(v) => set(f.name, v)}
      step="any"
      required
      error={errors[f.name]}
      hint={ranges?.[f.name] ? `train ${num(ranges[f.name][0], f.group === "pca" ? 1 : 0)} to ${num(ranges[f.name][1], f.group === "pca" ? 1 : 0)}` : undefined}
    />
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
      <form onSubmit={submit} noValidate className="space-y-5" aria-describedby="form-status">
        <Panel className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0 text-sm">
            {sample ? (
              <p className="text-ink-2">
                {dirty ? "Edited from" : "Loaded"} <span className="font-mono text-ink">{sample.row_id}</span>, a replayed row from the held-out test split{dirty ? " (ground truth hidden)" : ""}.
              </p>
            ) : (
              <p className="text-ink-2">Enter the features or replay a real transaction from the test split.</p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="secondary" size="sm" onClick={() => loadSample(0)} loading={loader.pending} disabled={!defaultId} icon={<ShieldCheck className="size-3.5" aria-hidden="true" />}>
              Load a legitimate one
            </Button>
            <Button type="button" variant="secondary" size="sm" onClick={() => loadSample(1)} loading={loader.pending} disabled={!defaultId} icon={<ShieldAlert className="size-3.5" aria-hidden="true" />}>
              Load a fraud
            </Button>
          </div>
          {loader.error && <p className="w-full text-xs text-critical-text">{loader.error.message}</p>}
        </Panel>

        <Panel as="section" aria-labelledby="grp-basics">
          <GroupTitle id="grp-basics" icon={Clock} title="Time, amount and reference" blurb="Time is seconds since the first transaction in the dataset; the currency of Amount is not published." />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {basics.map(field)}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="tx-id" className="text-[13px] font-medium text-ink-2">
                Transaction ID <span className="text-ink-3">(optional)</span>
              </label>
              <input
                id="tx-id"
                value={values.transaction_id ?? ""}
                onChange={(e) => set("transaction_id", e.target.value)}
                aria-invalid={!!errors.transaction_id}
                placeholder="for search only"
                className={cn(
                  "w-full rounded-xl border bg-surface-2/80 px-3.5 py-2.5 text-sm text-ink outline-none placeholder:text-ink-3 focus:border-violet focus:shadow-[0_0_0_4px_rgb(139_92_246/0.18)]",
                  errors.transaction_id ? "border-critical/70" : "border-line",
                )}
              />
              {errors.transaction_id ? <p className="text-xs text-critical-text">{errors.transaction_id}</p> : <p className="text-xs text-ink-3">Never used by the models</p>}
            </div>
          </div>
        </Panel>

        <Panel as="section" aria-labelledby="grp-pca">
          <GroupTitle id="grp-pca" icon={Layers} title="Anonymised components V1–V28" blurb="PCA outputs published with the dataset. Their meaning is not disclosed; any finite value is accepted." />
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">{pca.map(field)}</div>
        </Panel>

        <div className="sticky bottom-4 z-20">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-line-strong bg-surface/95 p-3 pl-5 shadow-2xl backdrop-blur-md">
            <p id="form-status" className="text-sm text-ink-2" aria-live="polite">
              {Object.keys(errors).length ? (
                <span className="text-critical-text">{Object.keys(errors).length} field(s) need attention</span>
              ) : predict.pending ? (
                "Scoring with both models…"
              ) : outside ? (
                <span className="text-warning">{outside} value(s) outside the training range: scores will be extrapolations</span>
              ) : (
                `${fields.length} contract fields · validated again by the API`
              )}
            </p>
            <Button type="submit" size="lg" loading={predict.pending} icon={<Sparkles className="size-4" aria-hidden="true" />}>
              {predict.pending ? "Scoring" : "Assess risk"}
            </Button>
          </div>
        </div>
      </form>

      <div ref={resultRef} className="scroll-mt-24 lg:sticky lg:top-24 lg:self-start">
        <AnimatePresence mode="wait">
          {predict.pending ? (
            <motion.div key="pending" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <Panel className="relative overflow-hidden" aria-busy="true">
                <div className="flex flex-col items-center gap-4 py-10">
                  <div className="relative size-24">
                    <span className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-r-cyan border-t-violet" />
                    <span className="absolute inset-3 animate-[spin_1.6s_linear_infinite_reverse] rounded-full border-2 border-transparent border-b-blue" />
                  </div>
                  <p className="text-sm text-ink-2">Validating, preprocessing and scoring…</p>
                </div>
              </Panel>
            </motion.div>
          ) : predict.error && !predict.error.fieldErrors.length ? (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <ErrorState error={predict.error} title="Scoring failed" />
            </motion.div>
          ) : result ? (
            <motion.div key={result.res.request_id}>
              <RiskResult result={result.res} groundTruth={result.truth} />
            </motion.div>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Panel>
                <EmptyState title="No assessment yet" icon={<Sparkles className="size-5" aria-hidden="true" />}>
                  Submit the form to get the Logistic Regression probability, the Isolation Forest anomaly percentile, the hybrid risk score and the decision.
                </EmptyState>
              </Panel>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function GroupTitle({ id, icon: Icon, title, blurb }: { id: string; icon: typeof Clock; title: string; blurb: string }) {
  return (
    <div className="mb-4 flex items-start gap-3">
      <span className="grid size-9 shrink-0 place-items-center rounded-xl border border-line bg-white/[0.03] text-cyan">
        <Icon className="size-4" aria-hidden="true" />
      </span>
      <div>
        <h2 id={id} className="text-base font-semibold text-ink">
          {title}
        </h2>
        <p className="text-xs text-ink-3">{blurb}</p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ batch */

function csvDownload(name: string, lines: string[]) {
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function BatchAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const batch = useAction((f: File) => api.predictBatchCsv(f));
  const datasets = useApi((s) => api.datasets({ signal: s }), []);
  const defaultId = datasets.data?.items.find((d) => d.source === "default")?.id;
  const example = useAction((id: number) => api.datasetSample(id, { n: 50, seed: 7 }));
  const res = batch.data;

  const pick = (f: File | undefined) => {
    if (!f) return;
    setFile(f);
    batch.reset();
  };

  const downloadExample = async () => {
    if (!defaultId) return;
    const r = await example.run(defaultId);
    if (!r?.items.length) return;
    const cols = Object.keys(r.items[0].features);
    csvDownload("heldout-test-sample.csv", [["transaction_id", ...cols].join(","), ...r.items.map((it) => [`replay-${it.row_id}`, ...cols.map((c) => it.features[c])].join(","))]);
  };

  const download = (r: BatchResponse) =>
    csvDownload(`risk-${r.batch_id.slice(0, 8)}.csv`, [
      "row_index,request_id,transaction_id,decision,hybrid_score,lr_probability,anomaly_percentile",
      ...r.results.map((x) => `${x.index},${x.request_id},${x.transaction_id ?? ""},${x.decision},${x.hybrid_score},${x.lr_probability},${x.anomaly_percentile}`),
    ]);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <Panel>
        <PanelHeader
          eyebrow="Batch scoring"
          title="Score a CSV of transactions"
          description="Up to 5,000 rows and 10 MB. Columns Time, V1–V28 and Amount are required; an optional transaction_id column is kept for search; other columns (such as Class) are ignored. Invalid rows are reported individually."
        />
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            pick(e.dataTransfer.files[0]);
          }}
          className={cn("flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed px-6 py-10 text-center transition-colors", drag ? "border-cyan bg-cyan/5" : "border-line-strong hover:border-violet/60")}
        >
          <FileUp className="size-6 text-cyan" aria-hidden="true" />
          <span className="text-sm text-ink">{file ? file.name : "Drop a CSV here or click to choose"}</span>
          {file && <span className="text-xs text-ink-3">{(file.size / 1024).toFixed(1)} KB</span>}
          <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => pick(e.target.files?.[0])} />
        </label>
        <Button className="mt-4 w-full" disabled={!file} loading={batch.pending} onClick={() => file && batch.run(file)}>
          Run batch scoring
        </Button>
        <Button className="mt-2 w-full" variant="ghost" size="sm" disabled={!defaultId} loading={example.pending} onClick={downloadExample} icon={<Download className="size-3.5" aria-hidden="true" />}>
          Download 50 held-out test rows as an example CSV
        </Button>
        {(batch.error || example.error) && (
          <div className="mt-4">
            <ErrorState error={(batch.error ?? example.error)!} compact title="Batch failed" />
          </div>
        )}
      </Panel>
      <Panel>
        {!res ? (
          <EmptyState title="No batch results yet">Results appear here with a per-row decision and scores, plus any rows that failed validation.</EmptyState>
        ) : (
          <div>
            <PanelHeader
              eyebrow={`Batch ${res.batch_id.slice(0, 8)} · model ${res.model_version}`}
              title={`${int(res.count)} transactions scored`}
              actions={
                <Button size="sm" variant="secondary" onClick={() => download(res)} icon={<Download className="size-3.5" aria-hidden="true" />}>
                  Download CSV
                </Button>
              }
            />
            <div className="mb-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              {(["HIGH_RISK", "REVIEW", "LOW_RISK"] as const).map((d) => (
                <div key={d} className="rounded-xl border border-line p-3">
                  <p className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.12em] text-ink-3">
                    <span className="size-2 rounded-full" style={{ background: DECISION_COLOR[d] }} aria-hidden="true" />
                    {decisionLabel(d)}
                  </p>
                  <p className="mt-1 font-display text-xl text-ink">{int(res.decision_counts[d])}</p>
                </div>
              ))}
              <div className="rounded-xl border border-line p-3">
                <p className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Invalid rows</p>
                <p className="mt-1 font-display text-xl text-ink">{int(res.errors.length)}</p>
              </div>
            </div>
            <div className="max-h-80 overflow-auto rounded-xl border border-line">
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 bg-surface-2 text-[11px] uppercase tracking-[0.12em] text-ink-3">
                  <tr>
                    <th className="px-3 py-2 font-medium">Row</th>
                    <th className="px-3 py-2 font-medium">Decision</th>
                    <th className="px-3 py-2 text-right font-medium">Hybrid</th>
                    <th className="px-3 py-2 text-right font-medium">LR prob.</th>
                    <th className="px-3 py-2 text-right font-medium">Anomaly pct.</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {[...res.results]
                    .sort((a, b) => b.hybrid_score - a.hybrid_score)
                    .slice(0, 200)
                    .map((r) => (
                      <tr key={r.request_id} className="border-t border-line">
                        <td className="px-3 py-1.5 text-ink-2">
                          {r.index}
                          {r.transaction_id && <span className="ml-2 font-mono text-[11px] text-ink-3">{r.transaction_id}</span>}
                        </td>
                        <td className="px-3 py-1.5">
                          <span className="inline-flex items-center gap-1.5">
                            <span className="size-2 rounded-full" style={{ background: DECISION_COLOR[r.decision] }} aria-hidden="true" />
                            {decisionLabel(r.decision)}
                          </span>
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono">{num(r.hybrid_score, 4)}</td>
                        <td className="px-3 py-1.5 text-right font-mono">{prob(r.lr_probability)}</td>
                        <td className="px-3 py-1.5 text-right font-mono">{pct(r.anomaly_percentile, 1)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-ink-3">Sorted by hybrid score. {res.results.length > 200 ? `Showing 200 of ${int(res.results.length)} rows; download the CSV for all.` : ""}</p>
            {res.errors.length > 0 && (
              <div className="mt-4">
                <p className="mb-2 text-sm font-medium text-ink">Rows rejected by validation</p>
                <ul className="max-h-40 space-y-1 overflow-auto text-xs text-ink-2">
                  {res.errors.slice(0, 50).map((e) => (
                    <li key={e.index}>
                      Row {e.index}: {e.errors.map((x) => `${x.field} – ${x.message}`).join("; ")}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <p className="mt-4 text-xs text-ink-3">{res.disclaimer}</p>
            <div className="mt-2">
              <Chip>batch_id {res.batch_id}</Chip>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}

import { AnimatePresence, motion } from "motion/react";
import { BookOpen, Briefcase, Calculator, ClipboardCheck, Download, FileUp, Shuffle, Sparkles, User } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { api } from "../api/endpoints";
import type { BatchResponse, FeatureGroup, FieldSchema, PredictionResponse, SampleRow, StudentInput } from "../api/types";
import { OUTCOME } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { PredictionResult } from "../components/predict/PredictionResult";
import { Button } from "../components/ui/Button";
import { NumberField, SelectField } from "../components/ui/Fields";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Chip } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { TabPanel, Tabs } from "../components/ui/Tabs";
import { useToast } from "../components/ui/Toast";
import { useAction, useApi } from "../hooks/useApi";
import { cn, humanize, int, num, pct } from "../lib/format";

type Mode = "single" | "batch";

export default function Predict() {
  const [mode, setMode] = useState<Mode>("single");
  return (
    <>
      <PageHeader
        crumb="Predict"
        title="Estimate a placement outcome"
        description="Enter a student profile or load a real student from the held-out test set. The estimate is produced by the active model only after the API responds."
      />
      <Section>
        <Tabs
          idPrefix="predict-mode"
          ariaLabel="Prediction mode"
          value={mode}
          onChange={setMode}
          items={[
            { key: "single", label: "Single student" },
            { key: "batch", label: "Batch (CSV)" },
          ]}
          className="mb-6 w-fit"
        />
        <TabPanel idPrefix="predict-mode" value={mode}>
          {mode === "single" ? <SinglePrediction /> : <BatchPrediction />}
        </TabPanel>
      </Section>
    </>
  );
}

/* ------------------------------------------------------------------ single */

const GROUPS: { key: FeatureGroup; title: string; icon: typeof User; blurb: string }[] = [
  { key: "profile", title: "Profile", icon: User, blurb: "Institution and programme." },
  { key: "academic", title: "Academic record", icon: BookOpen, blurb: "Semester grades, CGPA, attendance and backlogs." },
  { key: "assessment", title: "Assessments", icon: ClipboardCheck, blurb: "Optional scores; blanks are imputed with training medians." },
  { key: "experience", title: "Experience & activities", icon: Briefcase, blurb: "Internships, projects and extra-curricular work." },
];

type FormValues = Record<string, string>;

function toForm(features: StudentInput): FormValues {
  return Object.fromEntries(Object.entries(features).map(([k, v]) => [k, v == null ? "" : String(v)]));
}

function validate(fields: FieldSchema[], values: FormValues): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const f of fields) {
    const raw = (values[f.name] ?? "").trim();
    if (raw === "") {
      if (!f.nullable) errors[f.name] = "Required";
      continue;
    }
    if (f.kind === "numeric") {
      const v = Number(raw);
      if (!Number.isFinite(v)) errors[f.name] = "Enter a number";
      else if (v < f.minimum || v > f.maximum) errors[f.name] = `Must be between ${f.minimum} and ${f.maximum}`;
      else if (f.integer && !Number.isInteger(v)) errors[f.name] = "Whole number required";
    } else if (!f.categories.includes(raw)) {
      errors[f.name] = "Choose a listed option";
    }
  }
  return errors;
}

function toPayload(fields: FieldSchema[], values: FormValues): StudentInput {
  const out: StudentInput = {};
  for (const f of fields) {
    const raw = (values[f.name] ?? "").trim();
    out[f.name] = raw === "" ? null : f.kind === "numeric" ? Number(raw) : raw;
  }
  return out;
}

function SinglePrediction() {
  const schema = useApi((s) => api.schema({ signal: s }), []);
  const datasets = useApi((s) => api.datasets({ signal: s }), []);
  const defaultId = datasets.data?.items.find((d) => d.source === "default")?.id;
  const [values, setValues] = useState<FormValues>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [sample, setSample] = useState<SampleRow | null>(null);
  const [dirty, setDirty] = useState(false);
  const [result, setResult] = useState<{ res: PredictionResponse; truth?: { studentId: number; outcome: 0 | 1 } } | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const toast = useToast();
  const predict = useAction((payload: StudentInput) => api.predict(payload));
  const loader = useAction((id: number) => api.datasetSample(id, Math.floor(Math.random() * 1_000_000)));
  const fields = schema.data?.request_fields ?? [];

  const loadSample = async () => {
    if (!defaultId) return;
    const res = await loader.run(defaultId);
    const row = res?.items[0];
    if (!row) return;
    setValues(toForm(row.features));
    setSample(row);
    setDirty(false);
    setErrors({});
  };

  // Pre-fill with one real held-out student so the form is immediately usable.
  useEffect(() => {
    if (defaultId && !Object.keys(values).length) void loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultId]);

  const set = (name: string, v: string) => {
    setValues((s) => ({ ...s, [name]: v }));
    setDirty(true);
    if (errors[name]) setErrors(({ [name]: _, ...rest }) => rest);
  };

  const computeCgpa = () => {
    const sems = Array.from({ length: 8 }, (_, i) => Number(values[`SGPA_Sem${i + 1}`]));
    if (sems.some((v) => !Number.isFinite(v) || v === 0)) {
      toast({ tone: "info", title: "Fill all eight SGPA fields first" });
      return;
    }
    set("CGPA", (sems.reduce((a, b) => a + b, 0) / 8).toFixed(2));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const errs = validate(fields, values);
    setErrors(errs);
    if (Object.keys(errs).length) {
      document.querySelector<HTMLElement>("[aria-invalid='true']")?.focus();
      return;
    }
    const res = await predict.run(toPayload(fields, values));
    if (res) {
      setResult({ res, truth: sample && !dirty ? { studentId: sample.student_id, outcome: sample.actual_outcome } : undefined });
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }));
    }
  };

  // Map server-side validation errors onto the fields.
  useEffect(() => {
    if (predict.error?.fieldErrors.length) {
      setErrors(Object.fromEntries(predict.error.fieldErrors.map((f) => [f.field, f.message])));
    }
  }, [predict.error]);

  const grouped = useMemo(() => GROUPS.map((g) => ({ ...g, fields: fields.filter((f) => f.group === g.key) })), [fields]);

  if (schema.error) return <ErrorState error={schema.error} onRetry={schema.refetch} title="The feature schema could not be loaded" />;
  if (!schema.data) return <LoadingBlock rows={8} label="Loading form" />;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
      <form onSubmit={submit} noValidate className="space-y-5" aria-describedby="form-status">
        <Panel className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0 text-sm">
            {sample ? (
              <p className="text-ink-2">
                {dirty ? "Edited from" : "Loaded"} student <span className="font-mono text-ink">#{sample.student_id}</span> from the held-out test split{dirty ? " (ground truth hidden)" : ""}.
              </p>
            ) : (
              <p className="text-ink-2">Fill in the profile or load a real student from the test split.</p>
            )}
          </div>
          <Button type="button" variant="secondary" size="sm" onClick={loadSample} loading={loader.pending} disabled={!defaultId} icon={<Shuffle className="size-3.5" aria-hidden="true" />}>
            Load a real student
          </Button>
          {loader.error && <p className="w-full text-xs text-critical-text">{loader.error.message}</p>}
        </Panel>

        {grouped.map((g) => {
          const Icon = g.icon;
          return (
            <Panel key={g.key} as="section" aria-labelledby={`grp-${g.key}`}>
              <div className="mb-4 flex items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <span className="grid size-9 shrink-0 place-items-center rounded-xl border border-line bg-white/[0.03] text-cyan">
                    <Icon className="size-4" aria-hidden="true" />
                  </span>
                  <div>
                    <h2 id={`grp-${g.key}`} className="text-base font-semibold text-ink">
                      {g.title}
                    </h2>
                    <p className="text-xs text-ink-3">{g.blurb}</p>
                  </div>
                </div>
                {g.key === "academic" && (
                  <Button type="button" size="sm" variant="ghost" onClick={computeCgpa} icon={<Calculator className="size-3.5" aria-hidden="true" />}>
                    CGPA from SGPAs
                  </Button>
                )}
              </div>
              <div className={cn("grid gap-4", g.key === "academic" ? "grid-cols-2 sm:grid-cols-4" : "grid-cols-1 sm:grid-cols-2")}>
                {g.fields.map((f) =>
                  f.kind === "numeric" ? (
                    <NumberField
                      key={f.name}
                      name={f.name}
                      label={humanize(f.name).replace("Percent", "%")}
                      value={values[f.name] ?? ""}
                      onChange={(v) => set(f.name, v)}
                      min={f.minimum}
                      max={f.maximum}
                      step={f.integer ? 1 : 0.01}
                      required={!f.nullable}
                      placeholder={f.nullable ? "optional" : undefined}
                      error={errors[f.name]}
                    />
                  ) : (
                    <SelectField
                      key={f.name}
                      name={f.name}
                      label={humanize(f.name)}
                      value={values[f.name] ?? ""}
                      onChange={(v) => set(f.name, v)}
                      options={[{ value: "", label: "Select…" }, ...f.categories.map((c) => ({ value: c, label: c }))]}
                      required={!f.nullable}
                      error={errors[f.name]}
                      hint={f.kind === "ordinal" ? "Ordered category" : undefined}
                    />
                  ),
                )}
              </div>
            </Panel>
          );
        })}

        <div className="sticky bottom-4 z-20">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-line-strong bg-surface/95 p-3 pl-5 shadow-2xl backdrop-blur-md">
            <p id="form-status" className="text-sm text-ink-2" aria-live="polite">
              {Object.keys(errors).length ? (
                <span className="text-critical-text">{Object.keys(errors).length} field(s) need attention</span>
              ) : predict.pending ? (
                "Running the model…"
              ) : (
                `${fields.length} contract fields · validated again by the API`
              )}
            </p>
            <Button type="submit" size="lg" loading={predict.pending} icon={<Sparkles className="size-4" aria-hidden="true" />}>
              {predict.pending ? "Predicting" : "Predict placement"}
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
                    <span className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-violet border-r-cyan" />
                    <span className="absolute inset-3 animate-[spin_1.6s_linear_infinite_reverse] rounded-full border-2 border-transparent border-b-blue" />
                  </div>
                  <p className="text-sm text-ink-2">Validating, preprocessing and scoring…</p>
                </div>
              </Panel>
            </motion.div>
          ) : predict.error && !predict.error.fieldErrors.length ? (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <ErrorState error={predict.error} title="Prediction failed" />
            </motion.div>
          ) : result ? (
            <motion.div key={result.res.request_id}>
              <PredictionResult result={result.res} groundTruth={result.truth} />
            </motion.div>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Panel>
                <EmptyState title="No prediction yet" icon={<Sparkles className="size-5" aria-hidden="true" />}>
                  Submit the form to get the active model's estimate, the probability, the decision threshold and a full trace of the request.
                </EmptyState>
              </Panel>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ batch */

function BatchPrediction() {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const batch = useAction((f: File) => api.predictBatchCsv(f));
  const res = batch.data;

  const pick = (f: File | undefined) => {
    if (!f) return;
    setFile(f);
    batch.reset();
  };

  const download = (r: BatchResponse) => {
    const lines = ["row_index,request_id,prediction,label,probability", ...r.results.map((x) => `${x.index},${x.request_id},${x.prediction},${x.label},${x.probability}`)];
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `predictions-${r.batch_id.slice(0, 8)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <Panel>
        <PanelHeader eyebrow="Batch inference" title="Score a CSV of students" description="Up to 5,000 rows and 5 MB. Columns must use the contract names; extra columns (e.g. StudentID) are ignored and invalid rows are reported individually." />
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
          Run batch prediction
        </Button>
        {batch.error && <div className="mt-4"><ErrorState error={batch.error} compact title="Batch failed" /></div>}
      </Panel>
      <Panel>
        {!res ? (
          <EmptyState title="No batch results yet">Results appear here with a per-row label, probability and request ID, plus any rows that failed validation.</EmptyState>
        ) : (
          <div>
            <PanelHeader
              eyebrow={`Batch ${res.batch_id.slice(0, 8)} · model ${res.model_version}`}
              title={`${int(res.count)} students scored`}
              actions={
                <Button size="sm" variant="secondary" onClick={() => download(res)} icon={<Download className="size-3.5" aria-hidden="true" />}>
                  Download CSV
                </Button>
              }
            />
            <div className="mb-4 grid grid-cols-3 gap-3 text-sm">
              <div className="rounded-xl border border-line p-3">
                <p className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Predicted placed</p>
                <p className="mt-1 font-display text-xl text-ink">{pct(res.placed_count / Math.max(res.count, 1))}</p>
              </div>
              <div className="rounded-xl border border-line p-3">
                <p className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Invalid rows</p>
                <p className="mt-1 font-display text-xl text-ink">{int(res.errors.length)}</p>
              </div>
              <div className="rounded-xl border border-line p-3">
                <p className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Server time</p>
                <p className="mt-1 font-display text-xl text-ink">{num(res.latency_ms, 0)} ms</p>
              </div>
            </div>
            <div className="max-h-80 overflow-auto rounded-xl border border-line">
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 bg-surface-2 text-[11px] uppercase tracking-[0.12em] text-ink-3">
                  <tr>
                    <th className="px-3 py-2 font-medium">Row</th>
                    <th className="px-3 py-2 font-medium">Outcome</th>
                    <th className="px-3 py-2 text-right font-medium">P(placed)</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {res.results.slice(0, 200).map((r) => (
                    <tr key={r.request_id} className="border-t border-line">
                      <td className="px-3 py-1.5 text-ink-2">{r.index}</td>
                      <td className="px-3 py-1.5">
                        <span className="inline-flex items-center gap-1.5">
                          <span className="size-2 rounded-full" style={{ background: r.prediction ? OUTCOME.placed : OUTCOME.notPlaced }} aria-hidden="true" />
                          {r.label}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono">{num(r.probability, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {res.results.length > 200 && <p className="mt-2 text-xs text-ink-3">Showing 200 of {int(res.results.length)} rows; download the CSV for all.</p>}
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
            <div className="mt-2"><Chip>batch_id {res.batch_id}</Chip></div>
          </div>
        )}
      </Panel>
    </div>
  );
}

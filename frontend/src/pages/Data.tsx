import { CircleCheck, CircleX, Database, FileUp, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { api } from "../api/endpoints";
import type { DatasetDetail, NumericColumnStats, ValidationReport } from "../api/types";
import { SERIES } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Chip, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { useToast } from "../components/ui/Toast";
import { useAction, useApi } from "../hooks/useApi";
import { cn, int, num, pct, timeAgo } from "../lib/format";
import { BarsChart, DataTable } from "./lab/kit";

const statusTone = (s: string) => (s === "passed" ? "good" : s === "failed" ? "critical" : "warning");

export default function Data() {
  const list = useApi((s) => api.datasets({ signal: s }), []);
  const [selected, setSelected] = useState<number | null>(null);
  const id = selected ?? list.data?.items.find((d) => d.source === "default")?.id ?? list.data?.items[0]?.id ?? null;
  const schema = useApi((s) => api.schema({ signal: s }), []);

  return (
    <>
      <PageHeader
        crumb="Data"
        title="Datasets, validation & statistics"
        description="Every dataset is validated against the feature contract (columns, types, missing and non-finite values, ranges, target labels, imbalance, duplicates) before it can be used for training. Uploads are size-limited, parsed safely and stored under a random name."
      />
      <Section className="space-y-6">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
          <div className="space-y-6">
            <Panel>
              <PanelHeader eyebrow="Registered" title="Datasets" />
              {list.error ? (
                <ErrorState error={list.error} onRetry={list.refetch} compact />
              ) : !list.data ? (
                <LoadingBlock rows={3} />
              ) : (
                <ul className="space-y-2">
                  {list.data.items.map((d) => (
                    <li key={d.id}>
                      <button onClick={() => setSelected(d.id)} aria-pressed={d.id === id} className={cn("w-full rounded-xl border p-3 text-left transition-colors", d.id === id ? "border-violet/60 bg-violet/10" : "border-line hover:border-line-strong")}>
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate text-sm font-medium text-ink">{d.name}</span>
                          <StatusPill tone={statusTone(d.validation_status)}>{d.validation_status.replace(/_/g, " ")}</StatusPill>
                        </div>
                        <p className="mt-1 text-xs text-ink-3">
                          #{d.id} · {d.source} · {int(d.n_rows)} rows × {d.n_columns} cols · {timeAgo(d.uploaded_at)}
                        </p>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
            <Uploader
              onUploaded={(d) => {
                list.refetch();
                setSelected(d.id);
              }}
            />
            {schema.data && (
              <Panel>
                <PanelHeader eyebrow="Feature contract" title="Columns that are never model inputs" />
                <ul className="space-y-3 text-xs text-ink-2">
                  {Object.entries(schema.data.excluded_columns).map(([k, v]) => (
                    <li key={k}>
                      <Chip>{k}</Chip>
                      <p className="mt-1">{v}</p>
                    </li>
                  ))}
                </ul>
              </Panel>
            )}
          </div>
          <div className="min-w-0 space-y-6">{id != null ? <DatasetView key={id} id={id} /> : <EmptyState title="No dataset registered" icon={<Database className="size-5" aria-hidden="true" />} />}</div>
        </div>
      </Section>
    </>
  );
}

function Uploader({ onUploaded }: { onUploaded: (d: DatasetDetail) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const toast = useToast();
  const up = useAction((f: File) => api.uploadDataset(f));
  const submit = async () => {
    if (!file) return;
    const res = await up.run(file);
    if (res) {
      toast({ tone: res.validation_status === "failed" ? "error" : "success", title: `Uploaded ${res.name}`, body: `Validation ${res.validation_status.replace(/_/g, " ")}` });
      onUploaded(res);
      setFile(null);
    }
  };
  return (
    <Panel>
      <PanelHeader eyebrow="Ingestion" title="Upload a dataset" description="CSV only, up to 200 MB (the full creditcard.csv is about 144 MB). It must contain Time, V1–V28, Amount and Class to be usable for training." />
      <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed border-line-strong px-4 py-8 text-center hover:border-violet/60">
        <FileUp className="size-5 text-cyan" aria-hidden="true" />
        <span className="text-sm text-ink">{file ? file.name : "Choose a CSV file"}</span>
        <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <Button className="mt-4 w-full" disabled={!file} loading={up.pending} onClick={submit}>
        Upload & validate
      </Button>
      {up.error && <p className="mt-2 text-sm text-critical-text">{up.error.message}</p>}
    </Panel>
  );
}

function DatasetView({ id }: { id: number }) {
  const detail = useApi((s) => api.dataset(id, { signal: s }), [id]);
  const stats = useApi((s) => api.datasetStats(id, { signal: s }), [id]);
  const [col, setCol] = useState<string>("Amount");
  if (detail.error) return <ErrorState error={detail.error} onRetry={detail.refetch} />;
  if (!detail.data) return <LoadingBlock rows={8} />;
  const d = detail.data;
  const numeric = (stats.data?.columns.filter((c) => c.kind === "numeric") ?? []) as NumericColumnStats[];
  const categorical = stats.data?.columns.filter((c) => c.kind === "categorical") ?? [];
  const current = numeric.find((c) => c.name === col) ?? numeric[0];

  return (
    <>
      <Panel luminous>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-3">Dataset #{d.id}</p>
            <h2 className="mt-1 font-display text-2xl font-semibold text-ink">{d.name}</h2>
            <p className="mt-1 break-all font-mono text-xs text-ink-3">sha256 {d.sha256}</p>
          </div>
          <StatusPill tone={statusTone(d.validation_status)}>{d.validation_status.replace(/_/g, " ")}</StatusPill>
        </div>
        <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
          <div>
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Rows</dt>
            <dd className="font-display text-xl text-ink">{int(d.n_rows)}</dd>
          </div>
          <div>
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Columns</dt>
            <dd className="font-display text-xl text-ink">{d.n_columns}</dd>
          </div>
          <div>
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Fraud rate</dt>
            <dd className="font-display text-xl text-ink">{pct(d.positive_rate, 3)}</dd>
          </div>
        </dl>
      </Panel>
      <ValidationReportView report={d.validation_report} />
      <Panel>
        <PanelHeader eyebrow="Profiling" title="Numeric columns" description="Skewness > 0 means a right tail; excess kurtosis > 0 means heavier tails than a normal distribution." />
        {stats.error ? (
          <ErrorState error={stats.error} onRetry={stats.refetch} compact />
        ) : !stats.data ? (
          <LoadingBlock rows={6} />
        ) : (
          <>
            {current && (
              <div className="mb-5">
                <div className="mb-3 flex flex-wrap gap-1.5">
                  {numeric.map((c) => (
                    <button key={c.name} onClick={() => setCol(c.name)} aria-pressed={c.name === current.name} className={cn("rounded-full border px-2.5 py-0.5 text-xs", c.name === current.name ? "border-blue bg-blue/15 text-ink" : "border-line text-ink-3 hover:text-ink")}>
                      {c.name}
                    </button>
                  ))}
                </div>
                <BarsChart
                  title={`Distribution of ${current.name}`}
                  subtitle={`mean ${num(current.mean, 2)} · median ${num(current.median, 2)} · skewness ${num(current.skewness, 3)} · excess kurtosis ${num(current.kurtosis, 3)} · ${pct(current.missing_rate)} missing`}
                  data={current.histogram.counts.map((c, i) => ({ bin: num((current.histogram.edges[i] + current.histogram.edges[i + 1]) / 2, 2), count: c }))}
                  x="bin"
                  series={[{ key: "count", name: "Rows", color: SERIES[0] }]}
                  yFormat={(v) => int(v)}
                />
              </div>
            )}
            <DataTable
              columns={["Column", "Missing", "Mean", "Std", "Min", "Median", "Max", "Skewness", "Kurtosis", "Mean (legit.)", "Mean (fraud)"]}
              rows={numeric.map((c) => [c.name, pct(c.missing_rate), num(c.mean, 2), num(c.std, 2), num(c.min, 2), num(c.median, 2), num(c.max, 2), num(c.skewness, 3), num(c.kurtosis, 3), num(c.mean_legitimate, 3), num(c.mean_fraud, 3)])}
            />
          </>
        )}
      </Panel>
      {categorical.length > 0 && (
        <Panel>
          <PanelHeader eyebrow="Profiling" title="Categorical columns" />
          <DataTable
            numericFrom={9}
            columns={["Column", "Unique", "Missing", "Most frequent values"]}
            rows={categorical.map((c) => (c.kind === "categorical" ? [c.name, c.unique, pct(c.missing_rate), c.top.slice(0, 5).map((t) => `${t.value} (${int(t.count)})`).join(", ")] : []))}
          />
        </Panel>
      )}
    </>
  );
}

function ValidationReportView({ report }: { report: ValidationReport }) {
  const icon = { pass: CircleCheck, warn: TriangleAlert, fail: CircleX };
  const color = { pass: "text-good-text", warn: "text-warning", fail: "text-critical-text" };
  return (
    <Panel>
      <PanelHeader eyebrow="Data validation" title="Contract checks" description={`${report.checks.filter((c) => c.status === "pass").length} passed · ${report.checks.filter((c) => c.status === "warn").length} warnings · ${report.checks.filter((c) => c.status === "fail").length} failures`} />
      <ul className="divide-y divide-line">
        {report.checks.map((c) => {
          const Icon = icon[c.status];
          return (
            <li key={c.name} className="flex gap-3 py-3">
              <Icon className={cn("mt-0.5 size-4 shrink-0", color[c.status])} aria-label={c.status} />
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">{c.name.replace(/_/g, " ")}</p>
                <p className="text-sm text-ink-2">{c.message}</p>
                {c.details && Object.keys(c.details).length > 0 && (
                  <details className="mt-1">
                    <summary className="cursor-pointer text-xs text-ink-3 hover:text-ink">Details</summary>
                    <pre className="mt-1 max-h-40 overflow-auto rounded-lg bg-white/[0.03] p-2 font-mono text-[11px] text-ink-2">{JSON.stringify(c.details, null, 2)}</pre>
                  </details>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}

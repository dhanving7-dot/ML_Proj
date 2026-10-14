import { ChevronLeft, ChevronRight, RefreshCw, Search, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { api } from "../api/endpoints";
import type { Decision, PredictionQuery, PredictionRow } from "../api/types";
import { DECISION_COLOR } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Trace } from "../components/predict/RiskResult";
import { Button } from "../components/ui/Button";
import { SelectField } from "../components/ui/Fields";
import { Modal } from "../components/ui/Modal";
import { Panel } from "../components/ui/Panel";
import { Chip, decisionTone, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { useToast } from "../components/ui/Toast";
import { useAction, useApi } from "../hooks/useApi";
import { cn, cycleTime, dateTime, decisionLabel, int, money, num, pct, prob, timeAgo } from "../lib/format";

const ALL: Decision[] = ["HIGH_RISK", "REVIEW", "LOW_RISK"];
const PAGE_SIZES = [25, 50, 100];

export default function Transactions() {
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState(params.get("q") ?? "");
  const [debouncedQ, setDebouncedQ] = useState(q);
  const decisions = (params.getAll("decision").filter((d) => ALL.includes(d as Decision)) as Decision[]) ?? [];
  const source = params.get("source") ?? "";
  const labelled = params.get("labelled") ?? "";
  const sort = (params.get("sort") ?? "newest") as NonNullable<PredictionQuery["sort"]>;
  const minScore = params.get("min_score") ?? "";
  const minAmount = params.get("min_amount") ?? "";
  const maxAmount = params.get("max_amount") ?? "";
  const pageSize = Number(params.get("size") ?? 25);
  const page = Math.max(0, Number(params.get("page") ?? 0));
  const [open, setOpen] = useState<PredictionRow | null>(null);

  useEffect(() => {
    const t = window.setTimeout(() => setDebouncedQ(q.trim()), 300);
    return () => window.clearTimeout(t);
  }, [q]);

  const update = (patch: Record<string, string | string[] | null>, resetPage = true) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      next.delete(k);
      if (Array.isArray(v)) v.forEach((x) => next.append(k, x));
      else if (v) next.set(k, v);
    }
    if (resetPage) next.delete("page");
    setParams(next, { replace: true });
  };

  useEffect(() => {
    if ((params.get("q") ?? "") !== debouncedQ) update({ q: debouncedQ || null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQ]);

  const query: PredictionQuery = useMemo(() => {
    const out: PredictionQuery = { limit: pageSize, offset: page * pageSize, sort };
    if (debouncedQ) out.q = debouncedQ;
    if (decisions.length) out.decision = decisions;
    if (source) out.source = source;
    if (labelled === "yes") out.labelled = true;
    if (labelled === "no") out.labelled = false;
    if (labelled === "fraud") out.outcome = 1;
    if (labelled === "legitimate") out.outcome = 0;
    if (minScore !== "" && Number.isFinite(Number(minScore))) out.min_score = Number(minScore);
    if (minAmount !== "" && Number.isFinite(Number(minAmount))) out.min_amount = Number(minAmount);
    if (maxAmount !== "" && Number.isFinite(Number(maxAmount))) out.max_amount = Number(maxAmount);
    return out;
  }, [pageSize, page, sort, debouncedQ, decisions.join(","), source, labelled, minScore, minAmount, maxAmount]); // eslint-disable-line react-hooks/exhaustive-deps

  const list = useApi((s) => api.predictions(query, { signal: s }), [JSON.stringify(query)], { intervalMs: 30_000 });
  const data = list.data;
  const pages = data ? Math.max(1, Math.ceil(data.total / pageSize)) : 1;
  const filtersOn = !!(debouncedQ || decisions.length || source || labelled || minScore || minAmount || maxAmount);

  const toggleDecision = (d: Decision) => update({ decision: decisions.includes(d) ? decisions.filter((x) => x !== d) : [...decisions, d] });

  return (
    <>
      <PageHeader
        crumb="Transactions"
        title="Transaction monitoring"
        description="Every scored transaction with its decision, both model scores and any confirmed outcome. Replayed rows from the public dataset are labelled 'replay'; nothing here is simulated."
        actions={
          <Button variant="secondary" size="sm" onClick={list.refetch} loading={list.refreshing} icon={<RefreshCw className="size-3.5" aria-hidden="true" />}>
            Refresh
          </Button>
        }
      />
      <Section className="space-y-6">
        <Panel>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_repeat(3,minmax(0,1fr))]">
            <div className="flex flex-col gap-1.5">
              <label htmlFor="tx-search" className="text-[13px] font-medium text-ink-2">
                Search
              </label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-3" aria-hidden="true" />
                <input
                  id="tx-search"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="Request ID prefix or transaction ID"
                  className="w-full rounded-xl border border-line bg-surface-2/80 py-2.5 pl-9 pr-9 text-sm text-ink outline-none placeholder:text-ink-3 focus:border-violet focus:shadow-[0_0_0_4px_rgb(139_92_246/0.18)]"
                />
                {q && (
                  <button type="button" onClick={() => setQ("")} aria-label="Clear search" className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-ink-3 hover:text-ink">
                    <X className="size-3.5" />
                  </button>
                )}
              </div>
            </div>
            <SelectField
              label="Source"
              value={source}
              onChange={(v) => update({ source: v || null })}
              options={[
                { value: "", label: "All sources" },
                { value: "online", label: "Online" },
                { value: "batch", label: "Batch CSV" },
                { value: "replay", label: "Replay (dataset rows)" },
              ]}
            />
            <SelectField
              label="Outcome"
              value={labelled}
              onChange={(v) => update({ labelled: v || null })}
              options={[
                { value: "", label: "Any" },
                { value: "no", label: "No outcome yet" },
                { value: "yes", label: "Outcome recorded" },
                { value: "fraud", label: "Confirmed fraud" },
                { value: "legitimate", label: "Confirmed legitimate" },
              ]}
            />
            <SelectField
              label="Sort"
              value={sort}
              onChange={(v) => update({ sort: v })}
              options={[
                { value: "newest", label: "Newest first" },
                { value: "oldest", label: "Oldest first" },
                { value: "score_desc", label: "Highest risk first" },
                { value: "score_asc", label: "Lowest risk first" },
                { value: "amount_desc", label: "Largest amount first" },
              ]}
            />
          </div>
          <div className="mt-4 flex flex-wrap items-end gap-4">
            <div>
              <p className="mb-1.5 text-[13px] font-medium text-ink-2">Decision</p>
              <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by decision">
                {ALL.map((d) => {
                  const on = decisions.includes(d);
                  return (
                    <button
                      key={d}
                      type="button"
                      aria-pressed={on}
                      onClick={() => toggleDecision(d)}
                      className={cn("inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition-colors", on ? "border-violet/60 bg-violet/15 text-ink" : "border-line text-ink-3 hover:text-ink")}
                    >
                      <span className="size-2 rounded-full" style={{ background: DECISION_COLOR[d] }} aria-hidden="true" />
                      {decisionLabel(d)}
                      {data && <span className="font-mono text-[10px] text-ink-3">{int(data.decision_counts[d])}</span>}
                    </button>
                  );
                })}
              </div>
            </div>
            <SmallNumber label="Min hybrid score" value={minScore} onChange={(v) => update({ min_score: v || null })} placeholder="0–1" />
            <SmallNumber label="Min amount" value={minAmount} onChange={(v) => update({ min_amount: v || null })} />
            <SmallNumber label="Max amount" value={maxAmount} onChange={(v) => update({ max_amount: v || null })} />
            {filtersOn && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setQ("");
                  setParams(new URLSearchParams(), { replace: true });
                }}
              >
                Clear filters
              </Button>
            )}
          </div>
        </Panel>

        <Panel padded={false} className="overflow-hidden">
          {list.error && !data ? (
            <div className="p-5">
              <ErrorState error={list.error} onRetry={list.refetch} title="The transaction history could not be loaded" />
            </div>
          ) : !data ? (
            <div className="p-5">
              <LoadingBlock rows={8} label="Loading transactions" />
            </div>
          ) : data.items.length === 0 ? (
            <div className="p-5">
              <EmptyState title={filtersOn ? "No transactions match these filters" : "No transactions scored yet"} action={filtersOn ? undefined : <Button to="/analyze" size="sm">Analyze a transaction</Button>}>
                {filtersOn ? "Change or clear the filters." : "Scored transactions appear here as soon as the API serves them."}
              </EmptyState>
            </div>
          ) : (
            <div className={cn("overflow-x-auto transition-opacity", list.refreshing && "opacity-70")}>
              <table className="w-full min-w-[880px] text-left text-sm">
                <thead className="border-b border-line bg-white/[0.02] text-[11px] uppercase tracking-[0.12em] text-ink-3">
                  <tr>
                    <th className="px-4 py-2.5 font-medium">When</th>
                    <th className="px-4 py-2.5 font-medium">Decision</th>
                    <th className="px-4 py-2.5 text-right font-medium">Hybrid</th>
                    <th className="px-4 py-2.5 text-right font-medium">LR prob.</th>
                    <th className="px-4 py-2.5 text-right font-medium">Anomaly pct.</th>
                    <th className="px-4 py-2.5 text-right font-medium">Amount</th>
                    <th className="px-4 py-2.5 font-medium">Transaction</th>
                    <th className="px-4 py-2.5 font-medium">Outcome</th>
                    <th className="px-4 py-2.5 font-medium">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((p) => (
                    <tr
                      key={p.request_id}
                      tabIndex={0}
                      onClick={() => setOpen(p)}
                      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setOpen(p))}
                      className="cursor-pointer border-t border-line outline-none transition-colors hover:bg-white/[0.03] focus-visible:bg-white/[0.05]"
                      aria-label={`Open ${p.transaction_id ?? p.request_id.slice(0, 8)}, ${decisionLabel(p.decision)}`}
                    >
                      <td className="px-4 py-2.5 text-ink-2" title={dateTime(p.created_at)}>
                        {timeAgo(p.created_at)}
                      </td>
                      <td className="px-4 py-2.5">
                        <StatusPill tone={decisionTone(p.decision)}>{decisionLabel(p.decision)}</StatusPill>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-ink tabular">{num(p.hybrid_score, 4)}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-ink-2 tabular">{prob(p.lr_probability)}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-ink-2 tabular">{pct(p.anomaly_percentile, 1)}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-ink-2 tabular">{money(p.amount)}</td>
                      <td className="max-w-[12rem] truncate px-4 py-2.5 font-mono text-xs text-ink-2">{p.transaction_id ?? p.request_id.slice(0, 12)}</td>
                      <td className="px-4 py-2.5 text-xs">{p.actual_outcome == null ? <span className="text-ink-3">—</span> : p.actual_outcome ? <span className="text-critical-text">Fraud</span> : <span className="text-ink-2">Legitimate</span>}</td>
                      <td className="px-4 py-2.5">
                        <Chip>{p.source}</Chip>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data && data.total > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-4 py-3 text-xs text-ink-2">
              <span>
                {int(data.offset + 1)}–{int(Math.min(data.offset + data.items.length, data.total))} of {int(data.total)}
              </span>
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1.5">
                  Rows
                  <select value={pageSize} onChange={(e) => update({ size: e.target.value })} className="rounded-md border border-line bg-surface-2 px-1.5 py-1 text-ink">
                    {PAGE_SIZES.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </label>
                <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => update({ page: String(page - 1) }, false)} icon={<ChevronLeft className="size-3.5" aria-hidden="true" />}>
                  Previous
                </Button>
                <span className="font-mono">
                  {page + 1} / {pages}
                </span>
                <Button size="sm" variant="ghost" disabled={page + 1 >= pages} onClick={() => update({ page: String(page + 1) }, false)} icon={<ChevronRight className="size-3.5" aria-hidden="true" />}>
                  Next
                </Button>
              </div>
            </div>
          )}
        </Panel>
      </Section>
      <Modal open={!!open} onClose={() => setOpen(null)} title={open ? `Transaction ${open.transaction_id ?? open.request_id.slice(0, 12)}` : ""} wide>
        {open && <TransactionDetail row={open} onChanged={list.refetch} />}
      </Modal>
    </>
  );
}

function SmallNumber({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  const id = `f-${label.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <div className="flex w-32 flex-col gap-1.5">
      <label htmlFor={id} className="text-[13px] font-medium text-ink-2">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={0}
        value={local}
        placeholder={placeholder}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={() => local !== value && onChange(local)}
        onKeyDown={(e) => e.key === "Enter" && onChange(local)}
        className="w-full rounded-xl border border-line bg-surface-2/80 px-3 py-2 text-sm text-ink outline-none focus:border-violet"
      />
    </div>
  );
}

function TransactionDetail({ row, onChanged }: { row: PredictionRow; onChanged: () => void }) {
  const toast = useToast();
  const [outcome, setOutcome] = useState(row.actual_outcome);
  const feedback = useAction((v: 0 | 1) => api.recordOutcome(row.request_id, v));
  const record = async (v: 0 | 1) => {
    const res = await feedback.run(v);
    if (res) {
      setOutcome(v);
      toast({ tone: "success", title: "Outcome recorded", body: v ? "Marked as confirmed fraud." : "Marked as confirmed legitimate." });
      onChanged();
    }
  };
  const detail = useApi((s) => api.prediction(row.request_id, { signal: s }), [row.request_id]);
  const d = detail.data;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill tone={decisionTone(row.decision)}>{row.decision}</StatusPill>
        <Chip>{row.source}</Chip>
        <Chip>model {row.model_version}</Chip>
        <span className="text-xs text-ink-3">{dateTime(row.created_at)}</span>
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {[
          ["Hybrid score", num(row.hybrid_score, 4)],
          ["LR probability", prob(row.lr_probability)],
          ["Anomaly percentile", pct(row.anomaly_percentile, 2)],
          ["Amount", money(row.amount)],
          ["Time", cycleTime(row.time_s)],
          ["LR flag", row.lr_flag ? "yes" : "no"],
          ["IF flag", row.if_flag ? "yes" : "no"],
          ["Latency", `${num(row.latency_ms, 1)} ms`],
        ].map(([k, v]) => (
          <div key={k} className="rounded-xl border border-line px-3 py-2">
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">{k}</dt>
            <dd className="mt-0.5 font-mono text-ink tabular">{v}</dd>
          </div>
        ))}
      </dl>
      {row.out_of_range.length > 0 && (
        <p className="text-xs text-warning">
          Outside the training range: {row.out_of_range.join(", ")}
        </p>
      )}
      {d && (
        <div>
          <p className="mb-2 text-sm font-medium text-ink">Largest Logistic Regression contributions (log-odds)</p>
          <ul className="grid gap-1 text-xs sm:grid-cols-2">
            {d.explanation.lr_top_contributions.map((c) => (
              <li key={c.feature} className="flex justify-between gap-3 rounded bg-white/[0.03] px-2 py-1">
                <span className="text-ink-2">{c.label}</span>
                <span className={cn("font-mono", c.contribution >= 0 ? "text-critical-text" : "text-cyan")}>
                  {c.contribution >= 0 ? "+" : ""}
                  {num(c.contribution, 2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-line p-3">
        <span className="text-xs text-ink-3">Confirmed outcome: {outcome == null ? "not recorded" : outcome ? "fraud" : "legitimate"}</span>
        <Button size="sm" variant="secondary" loading={feedback.pending} onClick={() => record(1)}>
          Confirmed fraud
        </Button>
        <Button size="sm" variant="secondary" loading={feedback.pending} onClick={() => record(0)}>
          Confirmed legitimate
        </Button>
        {feedback.error && <span className="text-xs text-critical-text">{feedback.error.message}</span>}
      </div>
      {detail.error ? <ErrorState error={detail.error} onRetry={detail.refetch} compact /> : <Trace id={row.request_id} />}
    </div>
  );
}

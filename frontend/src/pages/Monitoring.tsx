import { Activity, AlertTriangle, Gauge, Radar, RefreshCw, Timer } from "lucide-react";
import { useState } from "react";
import { api } from "../api/endpoints";
import type { DriftReport, MonitoringSummary, Performance } from "../api/types";
import { MetricGrid } from "../components/charts/Common";
import { COMPONENT, DECISION_COLOR, SERIES } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Stat } from "../components/ui/Stat";
import { Chip, driftTone, severityTone, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { Tabs } from "../components/ui/Tabs";
import { useToast } from "../components/ui/Toast";
import { useAction, useApi } from "../hooks/useApi";
import { dateTime, decisionLabel, int, ms, num, pct, timeAgo } from "../lib/format";
import { BarsChart, LinesChart } from "./lab/kit";
import { PsiChart } from "./lab/M6";

const WINDOWS = [
  { key: "1", label: "1 hour" },
  { key: "24", label: "24 hours" },
  { key: "168", label: "7 days" },
] as const;

export default function Monitoring() {
  const [hours, setHours] = useState<"1" | "24" | "168">("24");
  const summary = useApi((s) => api.monitoringSummary(Number(hours), { signal: s }), [hours], { intervalMs: 20_000 });
  const drift = useApi((s) => api.drift({ signal: s }), [], { intervalMs: 60_000 });
  const perf = useApi((s) => api.performance({ signal: s }), [], { intervalMs: 60_000 });
  const events = useApi((s) => api.events(40, { signal: s }), [], { intervalMs: 20_000 });
  const toast = useToast();
  const check = useAction(() => api.driftCheck());

  const runCheck = async () => {
    const res = await check.run();
    if (res) {
      toast({
        tone: res.status === "significant_shift" ? "error" : res.status === "investigate" ? "info" : "success",
        title: `Drift check: ${res.status.replace(/_/g, " ")}`,
        body: res.message ?? `${res.rows} transactions compared with the training reference; event recorded.`,
      });
      drift.refetch();
      events.refetch();
    }
  };

  return (
    <>
      <PageHeader
        crumb="Monitoring"
        title="Operational & model monitoring"
        description="Request volume, latency and errors come from the request log; drift compares served inputs and scores with the training and validation distributions; detection quality needs confirmed outcomes (feedback)."
        actions={<Tabs idPrefix="mon-window" ariaLabel="Time window" size="sm" items={WINDOWS.map((w) => ({ key: w.key, label: w.label }))} value={hours} onChange={setHours} />}
      />
      <Section className="space-y-6">
        {summary.error && !summary.data ? <ErrorState error={summary.error} onRetry={summary.refetch} /> : !summary.data ? <LoadingBlock rows={6} /> : <Operations s={summary.data} />}
        <DriftPanel d={drift.data} error={drift.error} onRetry={drift.refetch} onCheck={runCheck} checking={check.pending} />
        <div className="grid gap-6 lg:grid-cols-2">
          <PerformancePanel p={perf.data} />
          <Panel>
            <PanelHeader eyebrow="Audit trail" title="Monitoring & lifecycle events" actions={<Button size="sm" variant="ghost" onClick={events.refetch} icon={<RefreshCw className="size-3.5" aria-hidden="true" />}>Refresh</Button>} />
            {!events.data ? (
              <LoadingBlock rows={4} />
            ) : events.data.items.length === 0 ? (
              <EmptyState title="No events yet" />
            ) : (
              <ul className="max-h-[520px] space-y-3 overflow-y-auto pr-1">
                {events.data.items.map((e) => (
                  <li key={e.id} className="rounded-xl border border-line p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <StatusPill tone={severityTone(e.severity)}>{e.kind.replace(/_/g, " ")}</StatusPill>
                      <span className="text-xs text-ink-3" title={dateTime(e.ts)}>
                        {timeAgo(e.ts)}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-ink-2">{e.message}</p>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </Section>
    </>
  );
}

function Operations({ s }: { s: MonitoringSummary }) {
  const series = s.series.map((b) => ({
    ...b,
    label: new Date(b.t).toLocaleString(undefined, s.window_hours > 24 ? { month: "short", day: "numeric" } : { hour: "2-digit", minute: "2-digit" }),
    other: b.requests - b.predict_requests,
  }));
  const by = Object.entries(s.predictions.by_source);
  return (
    <>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Panel>
          <Stat label="Requests" value={s.requests.total} format={(v) => int(v)} icon={<Activity className="size-3.5" />} hint={`${num(s.requests.throughput_per_min, 2)} per minute on average`} />
        </Panel>
        <Panel>
          <Stat label="Server error rate" value={s.requests.error_rate} format={(v) => pct(v, 2)} icon={<AlertTriangle className="size-3.5" />} hint={`${s.requests.server_errors} 5xx · ${s.requests.client_errors} 4xx (validation rejections)`} />
        </Panel>
        <Panel>
          <Stat
            label="Scoring latency p50"
            value={s.latency_ms.predict_p50}
            format={(v) => ms(v)}
            icon={<Timer className="size-3.5" />}
            hint={s.latency_ms.predict_p95 != null ? `p95 ${ms(s.latency_ms.predict_p95)} · p99 ${ms(s.latency_ms.predict_p99)}` : "No scoring requests in window"}
          />
        </Panel>
        <Panel>
          <Stat
            label="Transactions scored"
            value={s.predictions.total}
            format={(v) => int(v)}
            icon={<Gauge className="size-3.5" />}
            hint={s.predictions.flagged_rate != null ? `${pct(s.predictions.flagged_rate, 2)} flagged (REVIEW or HIGH_RISK)` : "None in window"}
          />
        </Panel>
      </div>
      <Panel>
        <BarsChart
          title="Decisions over time"
          subtitle={`Last ${s.window_hours} h, from the predictions table`}
          data={series}
          x="label"
          stacked
          series={[
            { key: "low_risk", name: "Low risk", color: DECISION_COLOR.LOW_RISK },
            { key: "review", name: "Review", color: DECISION_COLOR.REVIEW },
            { key: "high_risk", name: "High risk", color: DECISION_COLOR.HIGH_RISK },
          ]}
          yFormat={(v) => String(Math.round(v))}
        />
      </Panel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel>
          <BarsChart
            title="Request volume"
            subtitle={`Last ${s.window_hours} h, server-side request log (health checks excluded) · ${s.requests.server_errors} server errors in window`}
            data={series}
            x="label"
            stacked
            series={[
              { key: "predict_requests", name: "Scoring requests" },
              { key: "other", name: "Other API requests" },
            ]}
            yFormat={(v) => String(Math.round(v))}
          />
        </Panel>
        <Panel>
          <LinesChart
            title="Mean request latency"
            subtitle="All logged API requests per bucket"
            data={series}
            x="label"
            xType="category"
            series={[{ key: "mean_latency_ms", name: "Mean latency", color: SERIES[6], type: "linear", dot: true }]}
            yFormat={(v) => ms(v)}
            footer="Lab experiments and training jobs are logged API requests too, so they raise the mean in the buckets where they ran."
          />
        </Panel>
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <Panel className="lg:col-span-2">
          <ScoreHistograms s={s} />
        </Panel>
        <Panel>
          <PanelHeader eyebrow="Traffic" title="Sources and decisions" />
          {by.length === 0 ? (
            <EmptyState title="No transactions in this window" />
          ) : (
            <ul className="space-y-3">
              {by.map(([k, v]) => (
                <li key={k} className="flex items-center justify-between text-sm">
                  <Chip>{k}</Chip>
                  <span className="font-mono text-ink">{int(v)}</span>
                </li>
              ))}
            </ul>
          )}
          <ul className="mt-4 space-y-2 border-t border-line pt-4">
            {(["HIGH_RISK", "REVIEW", "LOW_RISK"] as const).map((d) => (
              <li key={d} className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-1.5 text-ink-2">
                  <span className="size-2 rounded-full" style={{ background: DECISION_COLOR[d] }} aria-hidden="true" />
                  {decisionLabel(d)}
                </span>
                <span className="font-mono text-ink">{int(s.predictions.by_decision[d])}</span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-ink-3">“replay” traffic is held-out rows of the public dataset sent through the API by scripts/replay_traffic.py (offline data), labelled separately from online use.</p>
        </Panel>
      </div>
    </>
  );
}

function ScoreHistograms({ s }: { s: MonitoringSummary }) {
  const [which, setWhich] = useState<"hybrid_score" | "lr_probability" | "anomaly_percentile">("hybrid_score");
  const titles = { hybrid_score: "hybrid risk score", lr_probability: "LR fraud probability", anomaly_percentile: "Isolation Forest anomaly percentile" };
  const colors = { hybrid_score: COMPONENT.hybrid, lr_probability: COMPONENT.logistic_regression, anomaly_percentile: COMPONENT.isolation_forest };
  return (
    <div>
      <div className="mb-3">
        <Tabs
          idPrefix="mon-score"
          ariaLabel="Score"
          size="sm"
          value={which}
          onChange={setWhich}
          items={[
            { key: "hybrid_score", label: "Hybrid" },
            { key: "lr_probability", label: "LR probability" },
            { key: "anomaly_percentile", label: "Anomaly percentile" },
          ]}
        />
      </div>
      <BarsChart
        title={`Served ${titles[which]} distribution`}
        subtitle="Transactions in this window per score bin (score distribution monitoring)"
        data={s.predictions.histograms[which]}
        x="bin"
        series={[{ key: "count", name: "Transactions", color: colors[which] }]}
        yFormat={(v) => String(Math.round(v))}
      />
    </div>
  );
}

function DriftPanel({ d, error, onRetry, onCheck, checking }: { d?: DriftReport; error?: Error; onRetry: () => void; onCheck: () => void; checking: boolean }) {
  return (
    <Panel luminous>
      <PanelHeader
        eyebrow="Data drift"
        title="Served transactions vs. the training distribution"
        description={d?.interpretation ?? "Population Stability Index and KL divergence per feature and per score over the most recent transactions of the active model."}
        actions={
          <>
            {d && <StatusPill tone={driftTone(d.status)}>{d.status.replace(/_/g, " ")}</StatusPill>}
            <Button size="sm" onClick={onCheck} loading={checking} icon={<Radar className="size-3.5" aria-hidden="true" />}>
              Run drift check
            </Button>
          </>
        }
      />
      {error && !d ? (
        <ErrorState error={error} onRetry={onRetry} compact />
      ) : !d ? (
        <LoadingBlock rows={4} />
      ) : d.status === "insufficient_data" || d.status === "no_model" ? (
        <EmptyState title="Not enough scored transactions yet" icon={<Radar className="size-5" aria-hidden="true" />}>
          {d.message}
        </EmptyState>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <PsiChart data={d.features ?? []} thresholds={d.thresholds!} />
          <div className="space-y-4">
            <MetricGrid
              items={[
                { label: "Window", value: `${int(d.rows)} rows`, hint: `model ${d.model_version}` },
                { label: "Flagged features", value: String(d.flagged_features?.length ?? 0) },
                { label: "Hybrid score PSI", value: num(d.scores?.hybrid_score?.psi, 3), hint: d.scores?.hybrid_score?.status.replace(/_/g, " ") },
                { label: "LR prob. PSI", value: num(d.scores?.lr_probability?.psi, 3), hint: d.scores?.lr_probability?.status.replace(/_/g, " ") },
                { label: "Anomaly pct. PSI", value: num(d.scores?.anomaly_percentile?.psi, 3), hint: d.scores?.anomaly_percentile?.status.replace(/_/g, " ") },
              ]}
            />
            <p className="text-xs text-ink-3">
              From {dateTime(d.from)} to {dateTime(d.to)}. Flagged: {d.flagged_features?.length ? d.flagged_features.join(", ") : "none"}.
            </p>
            {d.scores?.hybrid_score && (
              <BarsChart
                title="Hybrid score distribution drift"
                subtitle="Share of transactions per quantile bin of the validation-split hybrid scores"
                data={d.scores.hybrid_score.bins.map((b, i) => ({ ...b, bin_label: `B${i + 1}` }))}
                x="bin_label"
                series={[
                  { key: "expected", name: "Validation reference" },
                  { key: "actual", name: "Served" },
                ]}
                yFormat={(v) => pct(v, 0)}
                height={200}
              />
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

function PerformancePanel({ p }: { p?: Performance }) {
  return (
    <Panel>
      <PanelHeader
        eyebrow="Feedback & concept drift"
        title="Live detection quality from confirmed outcomes"
        description={p?.interpretation ?? "Outcomes are recorded per transaction on the Analyze and Transactions pages or via POST /api/predictions/{id}/outcome."}
        actions={p && p.status !== "insufficient_labels" ? <StatusPill tone={p.status === "healthy" ? "good" : "critical"}>{p.status}</StatusPill> : undefined}
      />
      {!p ? (
        <LoadingBlock rows={4} />
      ) : p.status === "insufficient_labels" ? (
        <EmptyState title={`${p.labelled_active_model} of ${p.min_labelled} labelled transactions`}>{p.message}</EmptyState>
      ) : (
        <div className="space-y-5">
          <MetricGrid
            items={[
              { label: "HIGH_RISK precision", value: num(p.live?.high_risk.precision, 3), hint: `offline test ${num(p.offline_test?.high_risk_precision, 3)}` },
              { label: "HIGH_RISK recall", value: num(p.live?.high_risk.recall, 3), hint: `offline ${num(p.offline_test?.high_risk_recall, 3)}` },
              { label: "REVIEW+ recall", value: num(p.live?.review_or_high.recall, 3), hint: `offline ${num(p.offline_test?.review_or_high_recall, 3)}` },
              { label: "Live PR-AUC", value: num(p.live?.ranking.average_precision, 3), hint: `offline ${num(p.offline_test?.average_precision, 3)}` },
              { label: "Labelled", value: int(p.labelled_active_model), hint: `${int(p.live?.review_or_high.positives)} confirmed fraud · model ${p.model_version}` },
            ]}
          />
          {p.blocks && p.blocks.length > 1 && (
            <LinesChart
              title="REVIEW-or-above recall per block of 50 labelled transactions"
              data={p.blocks}
              x="block"
              xLabel="Block (chronological)"
              series={[{ key: "recall_review_or_high", name: "Recall", dot: true, type: "linear", color: COMPONENT.hybrid }]}
              yFormat={(v) => pct(v, 0)}
              refY={p.offline_test ? [{ value: p.offline_test.review_or_high_recall, label: "offline test" }] : undefined}
              height={220}
            />
          )}
        </div>
      )}
    </Panel>
  );
}

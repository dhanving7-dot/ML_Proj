import { Activity, ArrowRight, Boxes, Gauge, RefreshCw, ShieldAlert, Siren, Timer } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/endpoints";
import type { Dashboard as DashboardData, DashboardRow } from "../api/types";
import { ChartFrame } from "../components/charts/ChartFrame";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { axisProps, COMPONENT, DECISION_COLOR, gridProps } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Stat } from "../components/ui/Stat";
import { Chip, decisionTone, jobTone, severityTone, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, Skeleton } from "../components/ui/States";
import { useApi } from "../hooks/useApi";
import { COMPONENT_LABELS, dateTime, decisionLabel, int, money, ms, num, pct, timeAgo } from "../lib/format";
import { Reveal } from "../motion/Reveal";

export default function Dashboard() {
  const q = useApi((s) => api.dashboard({ signal: s }), [], { intervalMs: 20_000 });
  const d = q.data;

  return (
    <>
      <PageHeader
        crumb="Overview"
        title="Fraud detection overview"
        description="Every number comes from the API: persisted risk assessments, the request log, the active model's one-time held-out evaluation and the monitoring log. Replayed dataset rows are counted separately from online traffic."
        actions={
          <Button variant="secondary" size="sm" onClick={q.refetch} loading={q.refreshing} icon={<RefreshCw className="size-3.5" aria-hidden="true" />}>
            Refresh
          </Button>
        }
      />
      <Section className="space-y-6">
        {q.error && !d ? (
          <ErrorState error={q.error} onRetry={q.refetch} title="Overview data could not be loaded" />
        ) : !d ? (
          <DashboardSkeleton />
        ) : (
          <div className={q.refreshing ? "opacity-80 transition-opacity" : "transition-opacity"}>
            <Kpis d={d} />
            <div className="mt-6 grid gap-6 lg:grid-cols-3">
              <ModelCard d={d} />
              <TrendCard d={d} />
            </div>
            <div className="mt-6 grid gap-6 lg:grid-cols-3">
              <AlertsCard rows={d.recent_alerts} />
              <HistogramCard d={d} />
            </div>
            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <TrainingCard d={d} />
              <EventsCard d={d} />
            </div>
            <p className="mt-4 text-right text-xs text-ink-3">Updated {timeAgo(d.generated_at)} · refreshes every 20 s</p>
          </div>
        )}
      </Section>
    </>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="Loading overview">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-32 rounded-2xl" />
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <Skeleton className="h-80 rounded-2xl" />
        <Skeleton className="h-80 rounded-2xl lg:col-span-2" />
      </div>
    </div>
  );
}

function Kpis({ d }: { d: DashboardData }) {
  const p = d.predictions;
  const items = [
    { label: "Transactions scored", value: p.total, format: (v: number) => int(v), icon: <Activity className="size-3.5" />, hint: `${int(p.last_24h)} in the last 24 h · ${int(p.by_source.replay ?? 0)} replayed` },
    { label: "High risk", value: p.by_decision.HIGH_RISK, format: (v: number) => int(v), icon: <ShieldAlert className="size-3.5" />, hint: `${int(p.open_high_risk)} without a confirmed outcome` },
    { label: "Flagged rate", value: p.flagged_rate, format: (v: number) => pct(v, 2), icon: <Siren className="size-3.5" />, hint: "REVIEW or HIGH_RISK share of scored transactions (not a fraud rate)" },
    { label: "Awaiting review", value: p.open_review, format: (v: number) => int(v), icon: <Gauge className="size-3.5" />, hint: `${int(p.by_decision.REVIEW)} REVIEW decisions in total` },
    {
      label: "Active model",
      value: d.model ? 1 : null,
      format: () => d.model?.version ?? "—",
      icon: <Boxes className="size-3.5" />,
      hint: d.model ? `${d.model.consistency?.passed ? "golden check passed" : "golden check FAILED"} · loaded ${timeAgo(d.model.loaded_at)}` : "No model loaded",
    },
  ];
  return (
    <Reveal stagger="[data-kpi]" className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
      {items.map((it, i) => (
        <div key={it.label} data-kpi>
          <Panel luminous={i === 0} className="h-full">
            <Stat label={it.label} value={it.value} format={it.format} icon={it.icon} hint={it.hint} accent={i === 0} />
          </Panel>
        </div>
      ))}
    </Reveal>
  );
}

function ModelCard({ d }: { d: DashboardData }) {
  const m = d.model;
  return (
    <Panel className="lg:col-span-1">
      <PanelHeader eyebrow="Active model · held-out test" title={m ? `${m.version} · hybrid` : "No active model"} actions={m && <StatusPill tone={m.consistency?.passed ? "good" : "critical"}>{m.consistency?.passed ? "Consistent" : "Check failed"}</StatusPill>} />
      {!m ? (
        <EmptyState title="Train a model to start scoring" action={<Button to="/models" size="sm">Open models</Button>}>
          No version is registered yet.
        </EmptyState>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[300px] text-sm">
              <thead className="text-[10px] uppercase tracking-[0.1em] text-ink-3">
                <tr>
                  <th className="pb-2 text-left font-medium">Component</th>
                  <th className="pb-2 text-right font-medium">PR-AUC</th>
                  <th className="pb-2 text-right font-medium">Prec.</th>
                  <th className="pb-2 text-right font-medium">Recall</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {(["logistic_regression", "isolation_forest", "hybrid_high_risk", "hybrid_review_or_high"] as const).map((k) => (
                  <tr key={k} className="border-t border-line">
                    <td className="py-2 text-ink-2">{COMPONENT_LABELS[k]}</td>
                    <td className="py-2 text-right font-mono text-ink">{num(m.test_metrics[k].average_precision, 3)}</td>
                    <td className="py-2 text-right font-mono text-ink">{num(m.test_metrics[k].precision, 3)}</td>
                    <td className="py-2 text-right font-mono text-ink">{num(m.test_metrics[k].recall, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-ink-3">
            {int(m.test_support)} held-out transactions ({m.test_fraud_rows} fraud), evaluated once at training time. Fusion weight w = {num(m.policy.fusion_weight_lr, 2)} (tuned on validation).
            Trained {dateTime(m.trained_at)}.
          </p>
          <Button to="/evaluation" variant="ghost" size="sm" className="-ml-3 mt-3" icon={<ArrowRight className="size-3.5" aria-hidden="true" />}>
            Evaluation lab
          </Button>
        </>
      )}
    </Panel>
  );
}

function TrendCard({ d }: { d: DashboardData }) {
  const data = d.predictions.trend.map((t) => ({ ...t, day: t.date.slice(5) }));
  const any = data.some((t) => t.LOW_RISK + t.REVIEW + t.HIGH_RISK > 0);
  return (
    <Panel className="lg:col-span-2">
      <PanelHeader eyebrow="Activity" title="Decisions per day (last 14 days)" />
      {!any ? (
        <EmptyState title="No transactions scored yet" action={<Button to="/analyze" size="sm">Analyze the first transaction</Button>}>
          Activity appears here as soon as the API scores a transaction.
        </EmptyState>
      ) : (
        <ChartFrame
          title="Daily decisions"
          legend={(["LOW_RISK", "REVIEW", "HIGH_RISK"] as const).map((k) => ({ label: decisionLabel(k), color: DECISION_COLOR[k] }))}
          table={{ columns: ["Date", "Low risk", "Review", "High risk"], rows: data.map((t) => [t.date, t.LOW_RISK, t.REVIEW, t.HIGH_RISK]) }}
          height={250}
        >
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }} barCategoryGap="22%">
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="day" {...axisProps} interval="preserveStartEnd" />
              <YAxis {...axisProps} width={40} allowDecimals={false} />
              <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={(v) => int(v)} />} />
              {(["LOW_RISK", "REVIEW", "HIGH_RISK"] as const).map((k, i) => (
                <Bar key={k} dataKey={k} name={decisionLabel(k)} stackId="a" fill={DECISION_COLOR[k]} stroke="#0b0d18" strokeWidth={2} radius={i === 2 ? [4, 4, 0, 0] : 0} isAnimationActive animationDuration={900} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      )}
    </Panel>
  );
}

function HistogramCard({ d }: { d: DashboardData }) {
  const total = d.predictions.hybrid_histogram.reduce((a, b) => a + b.count, 0);
  const pol = d.model?.policy;
  return (
    <Panel>
      <PanelHeader eyebrow="Score distribution" title="Hybrid risk scores" description="All scored transactions; most legitimate traffic sits in the lowest bins." />
      {total === 0 ? (
        <EmptyState title="No transactions yet" />
      ) : (
        <ChartFrame
          title="Hybrid score histogram"
          subtitle={pol ? `REVIEW ≥ ${num(pol.review_threshold, 3)} · HIGH_RISK ≥ ${num(pol.high_risk_threshold, 3)}` : undefined}
          table={{ columns: ["Bin", "Count"], rows: d.predictions.hybrid_histogram.map((b) => [b.bin, b.count]) }}
          height={220}
        >
          <ResponsiveContainer>
            <BarChart data={d.predictions.hybrid_histogram} margin={{ top: 6, right: 4, bottom: 0, left: 0 }} barCategoryGap={2}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="bin" {...axisProps} tickFormatter={(v: string) => v.split("–")[0]} />
              <YAxis {...axisProps} width={44} allowDecimals={false} scale="sqrt" />
              <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={(v) => int(v)} title={(l) => `Hybrid score ${l}`} />} />
              <Bar dataKey="count" name="Transactions" fill={COMPONENT.hybrid} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={900} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      )}
      <p className="mt-2 text-[11px] text-ink-3">The y-axis uses a square-root scale so small high-score bins stay visible.</p>
    </Panel>
  );
}

export function AlertsCard({ rows }: { rows: DashboardRow[] }) {
  return (
    <Panel className="lg:col-span-2">
      <PanelHeader eyebrow="Recent alerts" title="Latest REVIEW and HIGH_RISK decisions" actions={<Button to="/transactions?decision=HIGH_RISK&decision=REVIEW" size="sm" variant="secondary">All alerts</Button>} />
      {rows.length === 0 ? (
        <EmptyState title="No alerts yet">Alerts are transactions the hybrid policy routes to REVIEW or HIGH_RISK.</EmptyState>
      ) : (
        <div className="-mx-2 overflow-x-auto">
          <table className="w-full min-w-[600px] text-left text-sm">
            <thead className="text-[11px] uppercase tracking-[0.12em] text-ink-3">
              <tr>
                <th className="px-2 py-2 font-medium">When</th>
                <th className="px-2 py-2 font-medium">Decision</th>
                <th className="px-2 py-2 text-right font-medium">Hybrid</th>
                <th className="px-2 py-2 text-right font-medium">Amount</th>
                <th className="px-2 py-2 font-medium">Transaction</th>
                <th className="px-2 py-2 font-medium">Source</th>
                <th className="px-2 py-2 font-medium">Outcome</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.request_id} className="border-t border-line transition-colors hover:bg-white/[0.02]">
                  <td className="px-2 py-2.5 text-ink-2" title={dateTime(p.created_at)}>
                    {timeAgo(p.created_at)}
                  </td>
                  <td className="px-2 py-2.5">
                    <StatusPill tone={decisionTone(p.decision)}>{decisionLabel(p.decision)}</StatusPill>
                  </td>
                  <td className="px-2 py-2.5 text-right font-mono text-ink tabular">{num(p.hybrid_score, 3)}</td>
                  <td className="px-2 py-2.5 text-right font-mono text-ink-2 tabular">{money(p.amount)}</td>
                  <td className="max-w-[10rem] truncate px-2 py-2.5 font-mono text-xs text-ink-2">{p.transaction_id ?? p.request_id.slice(0, 10)}</td>
                  <td className="px-2 py-2.5">
                    <Chip>{p.source}</Chip>
                  </td>
                  <td className="px-2 py-2.5 text-xs text-ink-2">{p.actual_outcome == null ? "—" : p.actual_outcome ? "fraud" : "legitimate"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function TrainingCard({ d }: { d: DashboardData }) {
  return (
    <Panel>
      <PanelHeader
        eyebrow="Model lifecycle"
        title="Training & operations"
        actions={d.training.job_running ? <StatusPill tone="info">Job running</StatusPill> : d.training.last ? <StatusPill tone={jobTone(d.training.last.status)}>Last: {d.training.last.status}</StatusPill> : null}
      />
      <p className="text-sm text-ink-2">
        {int(d.training.runs)} training run(s), {int(d.models_registered)} registered version(s).
        {d.training.last?.reason && <span className="mt-1 block text-xs text-ink-3">{d.training.last.reason}</span>}
      </p>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl border border-line px-3 py-2">
          <dt className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.12em] text-ink-3">
            <Timer className="size-3" aria-hidden="true" /> Median latency (24 h)
          </dt>
          <dd className="mt-0.5 font-mono text-ink">{ms(d.operations.predict_latency_p50_ms)}</dd>
        </div>
        <div className="rounded-xl border border-line px-3 py-2">
          <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Server error rate (24 h)</dt>
          <dd className="mt-0.5 font-mono text-ink">{pct(d.operations.error_rate_24h, 2)}</dd>
        </div>
        <div className="rounded-xl border border-line px-3 py-2">
          <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Confirmed outcomes</dt>
          <dd className="mt-0.5 font-mono text-ink">{int(d.predictions.labelled)}</dd>
        </div>
        <div className="rounded-xl border border-line px-3 py-2">
          <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Confirmed fraud flagged</dt>
          <dd className="mt-0.5 font-mono text-ink">
            {int(d.predictions.confirmed_fraud_flagged)} / {int(d.predictions.confirmed_fraud)}
          </dd>
        </div>
      </dl>
    </Panel>
  );
}

function EventsCard({ d }: { d: DashboardData }) {
  return (
    <Panel>
      <PanelHeader
        eyebrow="Monitoring"
        title="Recent events"
        actions={
          <Button to="/monitoring" size="sm" variant="ghost" icon={<Activity className="size-3.5" aria-hidden="true" />}>
            Monitoring
          </Button>
        }
      />
      {d.monitoring.recent_events.length === 0 ? (
        <EmptyState title="No monitoring events yet" />
      ) : (
        <ul className="space-y-3">
          {d.monitoring.recent_events.map((e, i) => (
            <li key={i} className="flex gap-3">
              <StatusPill tone={severityTone(e.severity)} className="h-fit shrink-0">
                {e.kind.replace(/_/g, " ")}
              </StatusPill>
              <div className="min-w-0 text-sm">
                <p className="text-ink-2">{e.message}</p>
                <p className="text-xs text-ink-3">{timeAgo(e.ts)}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

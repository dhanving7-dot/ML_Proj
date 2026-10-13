import { Activity, ArrowRight, Gauge, RefreshCw, Sparkles, Target, Timer, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/endpoints";
import type { Dashboard as DashboardData } from "../api/types";
import { ChartFrame } from "../components/charts/ChartFrame";
import { ChartTooltip } from "../components/charts/ChartTooltip";
import { MetricGrid } from "../components/charts/Common";
import { axisProps, gridProps, OUTCOME, SERIES } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Stat } from "../components/ui/Stat";
import { Chip, jobTone, severityTone, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, Skeleton } from "../components/ui/States";
import { useApi } from "../hooks/useApi";
import { dateTime, int, modelLabel, ms, num, pct, timeAgo } from "../lib/format";
import { Reveal } from "../motion/Reveal";

export default function Dashboard() {
  const q = useApi((s) => api.dashboard({ signal: s }), [], { intervalMs: 20_000 });
  const d = q.data;

  return (
    <>
      <PageHeader
        crumb="Dashboard"
        title="Live system overview"
        description="Every number on this page comes from the API: persisted predictions, request logs, the registered model's held-out evaluation and the monitoring log."
        actions={
          <Button variant="secondary" size="sm" onClick={q.refetch} loading={q.refreshing} icon={<RefreshCw className="size-3.5" aria-hidden="true" />}>
            Refresh
          </Button>
        }
      />
      <Section className="space-y-6">
        {q.error && !d ? (
          <ErrorState error={q.error} onRetry={q.refetch} title="Dashboard data could not be loaded" />
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
              <HistogramCard d={d} />
              <RecentCard d={d} />
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
    <div className="space-y-6" role="status" aria-label="Loading dashboard">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
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
  const items = [
    { label: "Predictions served", value: d.predictions.total, format: (v: number) => int(v), icon: <Users className="size-3.5" />, hint: `${int(d.predictions.last_24h)} in the last 24 h` },
    { label: "Predicted placed", value: d.predictions.placed_rate, format: (v: number) => pct(v), icon: <Target className="size-3.5" />, hint: "Share of served predictions labelled Placed (not an accuracy)" },
    {
      label: "Live accuracy",
      value: d.predictions.live_accuracy,
      format: (v: number) => pct(v),
      icon: <Gauge className="size-3.5" />,
      hint: d.predictions.labelled ? `On ${int(d.predictions.labelled)} predictions with recorded outcomes` : "Needs recorded ground-truth outcomes",
    },
    {
      label: "Median latency",
      value: d.operations.predict_latency_p50_ms,
      format: (v: number) => ms(v),
      icon: <Timer className="size-3.5" />,
      hint: d.operations.error_rate_24h != null ? `Server error rate ${pct(d.operations.error_rate_24h, 2)} (24 h)` : "No requests in the last 24 h",
    },
  ];
  return (
    <Reveal stagger="[data-kpi]" className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
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
      <PanelHeader eyebrow="Active model" title={m ? `${m.version} · ${m.label}` : "No active model"} actions={m && <StatusPill tone={m.consistency?.passed ? "good" : "critical"}>{m.consistency?.passed ? "Consistent" : "Check failed"}</StatusPill>} />
      {!m ? (
        <EmptyState title="Train a model to start serving" action={<Button to="/models" size="sm">Open models</Button>}>
          No version is registered yet.
        </EmptyState>
      ) : (
        <>
          <MetricGrid
            items={[
              { label: "ROC-AUC", value: num(m.test_metrics.roc_auc, 3) },
              { label: "F1", value: num(m.test_metrics.f1, 3) },
              { label: "Accuracy", value: pct(m.test_metrics.accuracy) },
              { label: "Precision", value: num(m.test_metrics.precision, 3) },
              { label: "Recall", value: num(m.test_metrics.recall, 3) },
              { label: "Brier", value: num(m.test_metrics.brier, 4) },
            ]}
          />
          <p className="mt-4 text-xs leading-relaxed text-ink-3">
            Held-out test split ({int(m.test_support)} students), evaluated once at training time. Threshold {num(m.threshold, 2)} · feature set <Chip>{m.feature_set}</Chip> ·
            trained {dateTime(m.trained_at)}
          </p>
          <Button to="/models" variant="ghost" size="sm" className="mt-3 -ml-3" icon={<ArrowRight className="size-3.5" aria-hidden="true" />}>
            Model details
          </Button>
        </>
      )}
    </Panel>
  );
}

function TrendCard({ d }: { d: DashboardData }) {
  const data = d.predictions.trend.map((t) => ({ ...t, notPlaced: t.predictions - t.placed, day: t.date.slice(5) }));
  const any = data.some((t) => t.predictions > 0);
  return (
    <Panel className="lg:col-span-2">
      <PanelHeader eyebrow="Prediction activity" title="Predictions per day (last 14 days)" />
      {!any ? (
        <EmptyState title="No predictions yet" action={<Button to="/predict" size="sm">Make the first prediction</Button>}>
          Activity appears here as soon as the API serves a prediction.
        </EmptyState>
      ) : (
        <ChartFrame
          title="Daily predictions by predicted outcome"
          legend={[
            { label: "Predicted placed", color: OUTCOME.placed },
            { label: "Predicted not placed", color: OUTCOME.notPlaced },
          ]}
          table={{ columns: ["Date", "Placed", "Not placed", "Total"], rows: data.map((t) => [t.date, t.placed, t.notPlaced, t.predictions]) }}
          height={250}
        >
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }} barCategoryGap="22%">
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="day" {...axisProps} interval="preserveStartEnd" />
              <YAxis {...axisProps} width={40} allowDecimals={false} />
              <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={(v) => int(v)} />} />
              <Bar dataKey="placed" name="Predicted placed" stackId="a" fill={OUTCOME.placed} stroke="#0b0d18" strokeWidth={2} isAnimationActive animationDuration={900} />
              <Bar dataKey="notPlaced" name="Predicted not placed" stackId="a" fill={OUTCOME.notPlaced} stroke="#0b0d18" strokeWidth={2} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={900} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      )}
    </Panel>
  );
}

function HistogramCard({ d }: { d: DashboardData }) {
  const total = d.predictions.probability_histogram.reduce((a, b) => a + b.count, 0);
  return (
    <Panel>
      <PanelHeader eyebrow="Model output" title="Probability distribution" description="How confident the served predictions are (all time)." />
      {total === 0 ? (
        <EmptyState title="No predictions yet" />
      ) : (
        <ChartFrame title="P(placed) histogram" table={{ columns: ["Bin", "Count"], rows: d.predictions.probability_histogram.map((b) => [b.bin, b.count]) }} height={220}>
          <ResponsiveContainer>
            <BarChart data={d.predictions.probability_histogram} margin={{ top: 6, right: 4, bottom: 0, left: 0 }} barCategoryGap={2}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="bin" {...axisProps} tickFormatter={(v: string) => v.split("–")[0]} />
              <YAxis {...axisProps} width={36} allowDecimals={false} />
              <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={(v) => int(v)} title={(l) => `P(placed) ${l}`} />} />
              <Bar dataKey="count" name="Predictions" fill={SERIES[0]} radius={[4, 4, 0, 0]} isAnimationActive animationDuration={900} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      )}
    </Panel>
  );
}

function RecentCard({ d }: { d: DashboardData }) {
  return (
    <Panel className="lg:col-span-2">
      <PanelHeader eyebrow="Recent activity" title="Latest predictions" actions={<Button to="/predict" size="sm" variant="secondary">New prediction</Button>} />
      {d.recent_predictions.length === 0 ? (
        <EmptyState title="Nothing served yet" />
      ) : (
        <div className="-mx-2 overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead className="text-[11px] uppercase tracking-[0.12em] text-ink-3">
              <tr>
                <th className="px-2 py-2 font-medium">When</th>
                <th className="px-2 py-2 font-medium">Outcome</th>
                <th className="px-2 py-2 text-right font-medium">P(placed)</th>
                <th className="px-2 py-2 font-medium">Student</th>
                <th className="px-2 py-2 font-medium">Source</th>
                <th className="px-2 py-2 font-medium">Model</th>
              </tr>
            </thead>
            <tbody>
              {d.recent_predictions.map((p) => (
                <tr key={p.request_id} className="border-t border-line transition-colors hover:bg-white/[0.02]">
                  <td className="px-2 py-2.5 text-ink-2" title={dateTime(p.created_at)}>
                    {timeAgo(p.created_at)}
                  </td>
                  <td className="px-2 py-2.5">
                    <span className="inline-flex items-center gap-1.5 text-ink">
                      <span className="size-2 rounded-full" style={{ background: p.prediction ? OUTCOME.placed : OUTCOME.notPlaced }} aria-hidden="true" />
                      {p.label}
                    </span>
                  </td>
                  <td className="px-2 py-2.5 text-right font-mono text-ink tabular">{num(p.probability, 3)}</td>
                  <td className="px-2 py-2.5 text-ink-2">
                    CGPA {p.cgpa ?? "—"} · {p.stream} · {p.tier}
                  </td>
                  <td className="px-2 py-2.5">
                    <Chip>{p.source}</Chip>
                  </td>
                  <td className="px-2 py-2.5 font-mono text-xs text-ink-2">{p.model_version}</td>
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
  const m = d.model;
  const cv = [...(m?.cv_results ?? [])].sort((a, b) => b.roc_auc - a.roc_auc);
  return (
    <Panel>
      <PanelHeader
        eyebrow="Training"
        title="Model selection & retraining"
        actions={d.training.job_running ? <StatusPill tone="info">Job running</StatusPill> : d.training.last ? <StatusPill tone={jobTone(d.training.last.status)}>Last: {d.training.last.status}</StatusPill> : null}
      />
      <p className="text-sm text-ink-2">
        {int(d.training.runs)} training run(s), {int(d.models_registered)} registered version(s).
        {d.training.last?.reason && <span className="mt-1 block text-xs text-ink-3">{d.training.last.reason}</span>}
      </p>
      {cv.length > 0 && (
        <table className="mt-4 w-full text-sm">
          <caption className="mb-2 text-left text-xs text-ink-3">Candidates for {m?.version}: stratified 5-fold CV ROC-AUC on the training split</caption>
          <tbody>
            {cv.map((r, i) => (
              <tr key={r.model} className="border-t border-line">
                <td className="py-2 text-ink-2">
                  {modelLabel(r.model)} {i === 0 && <Sparkles className="ml-1 inline size-3.5 text-cyan" aria-label="selected" />}
                </td>
                <td className="py-2 text-right font-mono text-ink tabular">{num(r.roc_auc, 4)}</td>
                <td className="py-2 pl-2 text-right font-mono text-xs text-ink-3 tabular">± {num(r.roc_auc_std, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
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

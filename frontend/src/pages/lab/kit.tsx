import { AnimatePresence, motion } from "motion/react";
import { Clock, Database, Play, Zap } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { ApiError, toApiError } from "../../api/client";
import { api } from "../../api/endpoints";
import type { LabResponse, LabSplitInfo } from "../../api/types";
import { ChartFrame, type LegendItem } from "../../components/charts/ChartFrame";
import { ChartTooltip } from "../../components/charts/ChartTooltip";
import { axisProps, CHROME, gridProps, SERIES } from "../../components/charts/theme";
import { Button } from "../../components/ui/Button";
import { Panel } from "../../components/ui/Panel";
import { Chip } from "../../components/ui/Status";
import { ErrorState, Skeleton } from "../../components/ui/States";
import { cn, dateTime, int, num } from "../../lib/format";

/* ------------------------------------------------------------------ experiment state */

export type Params = Record<string, string | number | boolean | null>;

export interface ExperimentState<T> {
  params: Params;
  setParam: (k: string, v: Params[string]) => void;
  data: LabResponse<T> | undefined;
  error: ApiError | undefined;
  pending: boolean;
  run: () => void;
  stale: boolean;
}

/**
 * Runs one lab experiment. Newer runs abort older ones and late responses are discarded.
 * ``auto`` runs on mount; ``live`` re-runs (debounced) whenever a parameter changes.
 */
export function useExperiment<T>(module: string, key: string, defaults: Params, opts: { auto?: boolean; live?: boolean } = {}): ExperimentState<T> {
  const [params, setParams] = useState<Params>(defaults);
  const [data, setData] = useState<LabResponse<T>>();
  const [error, setError] = useState<ApiError>();
  const [pending, setPending] = useState(false);
  const [ranWith, setRanWith] = useState<string>("");
  const ctrl = useRef<AbortController | null>(null);
  const seq = useRef(0);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const run = useCallback(() => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    const id = ++seq.current;
    const p = paramsRef.current;
    setPending(true);
    setError(undefined);
    api
      .lab<T>(module, key, p, { signal: c.signal })
      .then((res) => {
        if (id !== seq.current) return;
        setData(res);
        setRanWith(JSON.stringify(p));
      })
      .catch((e) => {
        const err = toApiError(e);
        if (id !== seq.current || err.kind === "cancelled") return;
        setError(err);
      })
      .finally(() => id === seq.current && setPending(false));
  }, [module, key]);

  useEffect(() => {
    if (opts.auto) run();
    return () => {
      seq.current++;
      ctrl.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [module, key]);

  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    if (!opts.live) return;
    const t = window.setTimeout(run, 450);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(params)]);

  const setParam = useCallback((k: string, v: Params[string]) => setParams((s) => ({ ...s, [k]: v })), []);
  return { params, setParam, data, error, pending, run, stale: !!data && ranWith !== JSON.stringify(params) };
}

/* ------------------------------------------------------------------ shell */

export function ExperimentShell<T>({
  exp,
  title,
  topics,
  estimate,
  controls,
  children,
  live,
}: {
  exp: ExperimentState<T>;
  title: string;
  topics: string[];
  estimate: string;
  controls?: ReactNode;
  children: (data: LabResponse<T>) => ReactNode;
  live?: boolean;
}) {
  const { data, error, pending } = exp;
  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-4 xl:sticky xl:top-24 xl:self-start">
        <Panel>
          <h3 className="text-base font-semibold text-ink">{title}</h3>
          <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="Topics covered">
            {topics.map((t) => (
              <li key={t}>
                <Chip>{t}</Chip>
              </li>
            ))}
          </ul>
          {controls && <div className="mt-5 space-y-4 border-t border-line pt-5">{controls}</div>}
          <div className="mt-5 flex items-center gap-3">
            <Button onClick={exp.run} loading={pending} icon={<Play className="size-3.5" aria-hidden="true" />} className="flex-1">
              {data ? "Re-run" : "Run experiment"}
            </Button>
          </div>
          <p className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-3">
            <Clock className="size-3" aria-hidden="true" /> Typical run: {estimate}
            {live && <span className="ml-auto inline-flex items-center gap-1 text-cyan"><Zap className="size-3" aria-hidden="true" />live</span>}
          </p>
          {exp.stale && !pending && <p className="mt-2 text-xs text-warning">Parameters changed: results below are from the previous run.</p>}
        </Panel>
        {data && <RunMeta data={data} />}
      </div>
      <div className="relative min-w-0" aria-busy={pending}>
        {error && <div className="mb-4"><ErrorState error={error} onRetry={exp.run} title="Experiment failed" /></div>}
        {!data && !error ? (
          pending ? (
            <ResultSkeleton />
          ) : (
            <Panel>
              <div className="py-12 text-center">
                <p className="text-ink">This experiment trains models on demand.</p>
                <p className="mt-1 text-sm text-ink-3">Press “Run experiment” to compute real results ({estimate}).</p>
              </div>
            </Panel>
          )
        ) : data ? (
          <div className={cn("space-y-6 transition-opacity duration-300", pending && "opacity-50")}>{children(data)}</div>
        ) : null}
        <AnimatePresence>
          {pending && data && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="pointer-events-none absolute left-1/2 top-6 z-10 -translate-x-1/2">
              <span className="glass inline-flex items-center gap-2 rounded-full border border-line-strong px-4 py-2 text-sm text-ink">
                <span className="size-3 animate-spin rounded-full border-2 border-violet border-t-transparent" aria-hidden="true" /> Running on the server…
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="Running experiment">
      <Skeleton className="h-24 rounded-2xl" />
      <div className="grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-72 rounded-2xl" />
        <Skeleton className="h-72 rounded-2xl" />
      </div>
    </div>
  );
}

function RunMeta<T>({ data }: { data: LabResponse<T> }) {
  const split = data.result.split as LabSplitInfo | undefined;
  return (
    <Panel className="text-xs">
      <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-3">Run details</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5">
        <dt className="text-ink-3">Ran at</dt>
        <dd className="text-ink-2">{dateTime(data.ran_at)}</dd>
        <dt className="text-ink-3">Duration</dt>
        <dd className="text-ink-2">
          {num(data.duration_ms / 1000, 2)} s {data.cached && <Chip className="ml-1">cached</Chip>}
        </dd>
        {data.model_version && (
          <>
            <dt className="text-ink-3">Model</dt>
            <dd className="text-ink-2">{data.model_version}</dd>
          </>
        )}
        <dt className="text-ink-3">Dataset</dt>
        <dd className="truncate font-mono text-ink-2" title={data.dataset_sha256}>
          {data.dataset_sha256.slice(0, 12)}…
        </dd>
        {split && (
          <>
            <dt className="text-ink-3">Train rows</dt>
            <dd className="text-ink-2">{int(split.train_rows_used)}</dd>
            <dt className="text-ink-3">Validation</dt>
            <dd className="text-ink-2">{int(split.validation_rows_used)}</dd>
            <dt className="text-ink-3">Seed</dt>
            <dd className="text-ink-2">{split.seed}</dd>
          </>
        )}
      </dl>
      {split && (
        <p className="mt-3 flex gap-1.5 border-t border-line pt-3 text-ink-3">
          <Database className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
          <span>
            {split.note} Test split: {split.test_split}.
          </span>
        </p>
      )}
      <details className="mt-3 border-t border-line pt-3">
        <summary className="cursor-pointer text-ink-3 hover:text-ink">Parameters</summary>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-white/[0.03] p-2 font-mono text-[11px] text-ink-2">{JSON.stringify(data.params, null, 2)}</pre>
      </details>
    </Panel>
  );
}

/* ------------------------------------------------------------------ small building blocks */

export function Card({ title, children, className, description }: { title?: string; description?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Panel className={className}>
      {title && <h4 className="mb-1 text-sm font-semibold text-ink">{title}</h4>}
      {description && <p className="mb-4 text-xs leading-relaxed text-ink-3">{description}</p>}
      {children}
    </Panel>
  );
}

export function Note({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  return (
    <p className={cn("rounded-xl border p-3 text-xs leading-relaxed", tone === "warn" ? "border-warning/30 bg-warning/[0.05] text-ink-2" : "border-cyan/20 bg-cyan/[0.04] text-ink-2")}>{children}</p>
  );
}

export function DataTable({ columns, rows, className, numericFrom = 1 }: { columns: string[]; rows: (ReactNode | string | number | null)[][]; className?: string; numericFrom?: number }) {
  return (
    <div className={cn("overflow-x-auto rounded-xl border border-line", className)}>
      <table className="w-full min-w-max text-left text-sm">
        <thead className="bg-white/[0.02] text-[11px] uppercase tracking-[0.1em] text-ink-3">
          <tr>
            {columns.map((c, i) => (
              <th key={c} className={cn("px-3 py-2 font-medium", i >= numericFrom && "text-right")}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line hover:bg-white/[0.02]">
              {r.map((v, j) => (
                <td key={j} className={cn("px-3 py-2", j >= numericFrom ? "text-right font-mono text-[13px] text-ink" : "text-ink-2")}>
                  {v ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ generic charts */

export interface SeriesDef {
  key: string;
  name: string;
  color?: string;
  dashed?: boolean;
  width?: number;
  dot?: boolean;
  type?: "monotone" | "linear" | "stepAfter";
}

export function LinesChart({
  title,
  subtitle,
  data,
  x,
  series,
  xLabel,
  yLabel,
  xFormat = (v) => String(v),
  yFormat = (v) => num(v, 3),
  yDomain,
  xType = "number",
  xScale,
  refX,
  refY,
  height = 260,
  band,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  data: object[];
  x: string;
  series: SeriesDef[];
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  yDomain?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"];
  xType?: "number" | "category";
  xScale?: "log" | "linear";
  refX?: { value: number; label?: string }[];
  refY?: { value: number; label?: string }[];
  height?: number;
  band?: { lower: string; upper: string; color: string; name: string };
  footer?: ReactNode;
}) {
  const legend: LegendItem[] | undefined = series.length > 1 || band ? series.map((s, i) => ({ label: s.name, color: s.color ?? SERIES[i], shape: s.dashed ? "dash" : "line" })) : undefined;
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={legend}
      height={height}
      footer={footer}
      table={{ columns: [xLabel ?? x, ...series.map((s) => s.name)], rows: (data as Record<string, unknown>[]).map((d) => [String(d[x]), ...series.map((s) => (typeof d[s.key] === "number" ? yFormat(d[s.key] as number) : null))]) }}
    >
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 14, bottom: xLabel ? 18 : 0, left: 0 }}>
          <CartesianGrid {...gridProps} />
          <XAxis
            dataKey={x}
            type={xType}
            scale={xScale ?? "auto"}
            domain={xType === "number" ? ["dataMin", "dataMax"] : undefined}
            {...axisProps}
            tickFormatter={(v) => (typeof v === "number" ? xFormat(v) : v)}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10 } : undefined}
            allowDataOverflow={xScale === "log"}
          />
          <YAxis {...axisProps} width={56} domain={yDomain ?? ["auto", "auto"]} tickFormatter={(v) => yFormat(v)} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", offset: 12 } : undefined} />
          {refX?.map((r) => <ReferenceLine key={`x${r.value}`} x={r.value} stroke={CHROME.ink2} strokeDasharray="4 4" label={r.label ? { value: r.label, position: "top", fill: CHROME.muted, fontSize: 10 } : undefined} />)}
          {refY?.map((r) => <ReferenceLine key={`y${r.value}`} y={r.value} stroke={CHROME.muted} strokeDasharray="4 4" label={r.label ? { value: r.label, position: "insideTopRight", fill: CHROME.muted, fontSize: 10 } : undefined} />)}
          <Tooltip content={<ChartTooltip valueFormat={yFormat} title={(l) => `${xLabel ?? x}: ${typeof l === "number" ? xFormat(l) : l}`} />} />
          {band && <Area dataKey={(d: Record<string, number>) => [d[band.lower], d[band.upper]]} name={band.name} stroke="none" fill={band.color} fillOpacity={0.12} isAnimationActive={false} legendType="none" />}
          {series.map((s, i) => (
            <Line
              key={s.key}
              dataKey={s.key}
              name={s.name}
              stroke={s.color ?? SERIES[i]}
              strokeWidth={s.width ?? 2}
              strokeDasharray={s.dashed ? "5 4" : undefined}
              dot={s.dot ? { r: 3, strokeWidth: 2, stroke: CHROME.surface, fill: s.color ?? SERIES[i] } : false}
              activeDot={{ r: 4 }}
              type={s.type ?? "linear"}
              connectNulls
              isAnimationActive
              animationDuration={1000}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function BarsChart({
  title,
  subtitle,
  data,
  x,
  series,
  stacked,
  yFormat = (v) => num(v, 3),
  xFormat,
  height = 240,
  yDomain,
  refY,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  data: object[];
  x: string;
  series: SeriesDef[];
  stacked?: boolean;
  yFormat?: (v: number) => string;
  xFormat?: (v: string | number) => string;
  height?: number;
  yDomain?: [number | "auto", number | "auto"];
  refY?: { value: number; label?: string; color?: string }[];
  footer?: ReactNode;
}) {
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      height={height}
      footer={footer}
      legend={series.length > 1 ? series.map((s, i) => ({ label: s.name, color: s.color ?? SERIES[i] })) : undefined}
      table={{ columns: [x, ...series.map((s) => s.name)], rows: (data as Record<string, unknown>[]).map((d) => [String(d[x]), ...series.map((s) => (typeof d[s.key] === "number" ? yFormat(d[s.key] as number) : null))]) }}
    >
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 10, bottom: 0, left: 0 }} barGap={2} barCategoryGap="18%">
          <CartesianGrid {...gridProps} />
          <XAxis dataKey={x} {...axisProps} interval="preserveStartEnd" tickFormatter={xFormat} />
          <YAxis {...axisProps} width={48} domain={yDomain ?? [0, "auto"]} tickFormatter={(v) => yFormat(v)} />
          {refY?.map((r) => <ReferenceLine key={r.value} y={r.value} stroke={r.color ?? CHROME.muted} strokeDasharray="4 4" label={r.label ? { value: r.label, position: "insideTopRight", fill: CHROME.muted, fontSize: 10 } : undefined} />)}
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={yFormat} />} />
          {series.map((s, i) => (
            <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color ?? SERIES[i]} stackId={stacked ? "s" : undefined} radius={stacked ? 0 : [4, 4, 0, 0]} stroke={stacked ? CHROME.surface : undefined} strokeWidth={stacked ? 2 : 0} isAnimationActive animationDuration={900} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export interface ScatterGroup {
  key: string | number;
  label: string;
  color: string;
}

/**
 * Scatter with at most three coloured groups (the all-pairs-safe slots). With more groups, one
 * group is emphasised and the rest are drawn in neutral gray; a chip row selects the emphasis.
 */
export function ScatterGroups({
  title,
  subtitle,
  points,
  groups,
  xLabel = "x",
  yLabel = "y",
  height = 320,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  points: { x: number; y: number; g: string | number }[];
  groups: ScatterGroup[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  footer?: ReactNode;
}) {
  const many = groups.length > 3;
  const [focus, setFocus] = useState<string | number>(groups[0]?.key);
  const visibleGroups = many
    ? [
        ...groups.filter((g) => g.key !== focus).map((g) => ({ ...g, color: CHROME.neutral, label: g.label })),
        { ...groups.find((g) => g.key === focus)!, color: SERIES[0] },
      ]
    : groups;
  const legend = many ? [{ label: `${groups.find((g) => g.key === focus)?.label} (emphasised)`, color: SERIES[0] }, { label: "Other groups", color: CHROME.neutral }] : groups.map((g) => ({ label: g.label, color: g.color }));
  return (
    <div>
      {many && (
        <div className="mb-3 flex flex-wrap items-center gap-1.5" role="group" aria-label="Emphasise group">
          <span className="text-xs text-ink-3">Emphasise:</span>
          {groups.map((g) => (
            <button key={g.key} onClick={() => setFocus(g.key)} aria-pressed={focus === g.key} className={cn("rounded-full border px-2.5 py-0.5 text-xs transition-colors", focus === g.key ? "border-blue bg-blue/20 text-ink" : "border-line text-ink-3 hover:text-ink")}>
              {g.label}
            </button>
          ))}
        </div>
      )}
      <ChartFrame
        title={title}
        subtitle={subtitle}
        legend={legend}
        height={height}
        footer={footer}
        table={{ columns: ["Group", xLabel, yLabel], rows: points.slice(0, 400).map((p) => [groups.find((g) => g.key === p.g)?.label ?? String(p.g), num(p.x, 3), num(p.y, 3)]) }}
      >
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid {...gridProps} vertical />
            <XAxis type="number" dataKey="x" name={xLabel} {...axisProps} tickFormatter={(v) => num(v, 1)} label={{ value: xLabel, position: "insideBottom", offset: -10 }} domain={["auto", "auto"]} />
            <YAxis type="number" dataKey="y" name={yLabel} {...axisProps} width={44} tickFormatter={(v) => num(v, 1)} domain={["auto", "auto"]} />
            <ZAxis range={[22, 22]} />
            <Tooltip cursor={{ stroke: CHROME.axis }} content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(_, d) => groups.find((g) => g.key === (d?.g as string | number))?.label ?? ""} />} />
            {visibleGroups.map((g) => (
              <Scatter key={String(g.key)} name={g.label} data={points.filter((p) => p.g === g.key)} fill={g.color} fillOpacity={many && g.color === CHROME.neutral ? 0.35 : 0.8} isAnimationActive animationDuration={700} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </ChartFrame>
    </div>
  );
}

export const logFmt = (v: number) => {
  const e = Math.log10(v);
  return Number.isInteger(Math.round(e * 100) / 100) ? `1e${Math.round(e)}` : v < 0.01 ? v.toExponential(0) : v.toPrecision(2);
};

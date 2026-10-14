import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CalibrationResult, CurvePointPR, CurvePointROC, SweepPoint } from "../../api/types";
import { cn, int, num, pct } from "../../lib/format";
import { ChartFrame } from "./ChartFrame";
import { ChartTooltip } from "./ChartTooltip";
import { axisProps, CHROME, DIVERGING, gridProps, SERIES } from "./theme";

/* --------------------------------------------------------------- confusion matrix */

/** Sequential single-hue (blue) intensity on the dark surface: low values recede toward the surface. */
function cellColor(frac: number): string {
  const a = 0.08 + 0.72 * frac;
  return `rgba(57, 135, 229, ${a.toFixed(3)})`;
}

export function ConfusionMatrix({
  cm,
  labels = ["Legitimate", "Fraud"],
  title = "Confusion matrix",
  subtitle,
}: {
  cm: { tn: number; fp: number; fn: number; tp: number } | number[][];
  labels?: string[];
  title?: string;
  subtitle?: string;
}) {
  const matrix = Array.isArray(cm) ? cm : [[cm.tn, cm.fp], [cm.fn, cm.tp]];
  const total = matrix.flat().reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...matrix.flat(), 1);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const names2 = [["True negative", "False positive"], ["False negative", "True positive"]];
  return (
    <figure>
      <figcaption className="mb-3">
        <p className="text-sm font-medium text-ink">{title}</p>
        {subtitle && <p className="text-xs text-ink-3">{subtitle}</p>}
      </figcaption>
      <div className="flex gap-3">
        <div className="flex w-5 items-center justify-center">
          <span className="-rotate-90 whitespace-nowrap text-[11px] text-ink-3">Actual</span>
        </div>
        <div className="flex-1">
          <table className="w-full table-fixed border-separate border-spacing-[3px] text-center" aria-label={title}>
            <thead>
              <tr>
                <th className="w-24" />
                {labels.map((l) => (
                  <th key={l} scope="col" className="pb-1 text-[11px] font-normal text-ink-3">
                    Pred. {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  <th scope="row" className="pr-2 text-right text-[11px] font-normal text-ink-3">
                    {labels[i]}
                  </th>
                  {row.map((v, j) => {
                    const frac = v / max;
                    const rowTotal = row.reduce((a, b) => a + b, 0) || 1;
                    const on = hover?.[0] === i && hover?.[1] === j;
                    return (
                      <td
                        key={j}
                        tabIndex={0}
                        onMouseEnter={() => setHover([i, j])}
                        onMouseLeave={() => setHover(null)}
                        onFocus={() => setHover([i, j])}
                        onBlur={() => setHover(null)}
                        className={cn("rounded-lg py-4 transition-[box-shadow,transform] duration-200 outline-none", on && "scale-[1.03] shadow-[0_0_0_2px_#22d3ee]")}
                        style={{ background: cellColor(frac) }}
                        aria-label={`Actual ${labels[i]}, predicted ${labels[j]}: ${v} (${pct(v / rowTotal)} of row)`}
                      >
                        <div className={cn("font-display text-xl font-semibold tabular", frac > 0.55 ? "text-white" : "text-ink")}>{int(v)}</div>
                        <div className="text-[10px] text-ink-2">{pct(v / total)} of all</div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 min-h-4 text-xs text-ink-3" aria-live="polite">
            {hover && matrix.length === 2
              ? `${names2[hover[0]][hover[1]]}: ${int(matrix[hover[0]][hover[1]])} rows · ${pct(matrix[hover[0]][hover[1]] / (matrix[hover[0]].reduce((a, b) => a + b, 0) || 1))} of actual ${labels[hover[0]].toLowerCase()}`
              : "Hover or focus a cell for details."}
          </p>
        </div>
      </div>
    </figure>
  );
}

/* --------------------------------------------------------------- ROC / PR */

export function RocChart({ points, auc, title = "ROC curve", height = 260 }: { points: CurvePointROC[]; auc?: number | null; title?: string; height?: number }) {
  return (
    <ChartFrame
      title={title}
      subtitle={auc != null ? `Area under curve ${num(auc, 4)}` : undefined}
      legend={[
        { label: "Model", color: SERIES[0], shape: "line" },
        { label: "Chance", color: CHROME.muted, shape: "dash" },
      ]}
      table={{ columns: ["Threshold", "FPR", "TPR"], rows: points.filter((_, i) => i % 4 === 0).map((p) => [num(p.threshold, 3), num(p.fpr, 3), num(p.tpr, 3)]) }}
      height={height}
    >
      <ResponsiveContainer>
        <LineChart data={points} margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => num(v, 1)} label={{ value: "False positive rate", position: "insideBottom", offset: -10 }} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHROME.muted} strokeDasharray="4 4" />
          <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(_, d) => `Threshold ${num(d?.threshold as number, 3)} · FPR ${num(d?.fpr as number, 3)}`} />} />
          <Line dataKey="tpr" name="True positive rate" stroke={SERIES[0]} strokeWidth={2} dot={false} isAnimationActive animationDuration={1200} type="monotone" />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function PrChart({ points, ap, baseline, title = "Precision–recall curve", height = 260 }: { points: CurvePointPR[]; ap?: number | null; baseline?: number; title?: string; height?: number }) {
  const data = [...points].sort((a, b) => a.recall - b.recall);
  return (
    <ChartFrame
      title={title}
      subtitle={ap != null ? `Average precision ${num(ap, 4)}${baseline != null ? ` · no-skill precision ${num(baseline, 3)}` : ""}` : undefined}
      legend={[
        { label: "Model", color: SERIES[0], shape: "line" },
        ...(baseline != null ? [{ label: "No skill", color: CHROME.muted, shape: "dash" as const }] : []),
      ]}
      table={{ columns: ["Threshold", "Recall", "Precision"], rows: data.filter((_, i) => i % 4 === 0).map((p) => [num(p.threshold, 3), num(p.recall, 3), num(p.precision, 3)]) }}
      height={height}
    >
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis dataKey="recall" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => num(v, 1)} label={{ value: "Recall", position: "insideBottom", offset: -10 }} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
          {baseline != null && <ReferenceLine y={baseline} stroke={CHROME.muted} strokeDasharray="4 4" />}
          <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(_, d) => `Threshold ${num(d?.threshold as number, 3)} · recall ${num(d?.recall as number, 3)}`} />} />
          <Line dataKey="precision" name="Precision" stroke={SERIES[0]} strokeWidth={2} dot={false} type="stepAfter" isAnimationActive animationDuration={1200} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/* --------------------------------------------------------------- reliability diagram */

export function ReliabilityChart({ series, title = "Reliability diagram", height = 280 }: { series: { name: string; result: CalibrationResult }[]; title?: string; height?: number }) {
  const legend = [...series.map((s, i) => ({ label: `${s.name} (ECE ${num(s.result.ece, 3)})`, color: SERIES[i], shape: "line" as const })), { label: "Perfect calibration", color: CHROME.muted, shape: "dash" as const }];
  return (
    <ChartFrame
      title={title}
      subtitle="Mean predicted probability vs. observed fraud rate, per probability bin"
      legend={legend}
      table={{
        columns: ["Series", "Mean predicted", "Observed fraud rate"],
        rows: series.flatMap((s) => s.result.points.map((p) => [s.name, num(p.mean_predicted, 3), num(p.fraction_positive, 3)])),
      }}
      height={height}
    >
      <ResponsiveContainer>
        <LineChart margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => num(v, 1)} label={{ value: "Mean predicted probability", position: "insideBottom", offset: -10 }} allowDuplicatedCategory={false} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHROME.muted} strokeDasharray="4 4" />
          <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(_, d) => `Mean predicted ${num(d?.mean_predicted as number, 3)}`} />} />
          {series.map((s, i) => (
            <Line key={s.name} data={s.result.points} dataKey="fraction_positive" name={s.name} stroke={SERIES[i]} strokeWidth={2} dot={{ r: 3.5, strokeWidth: 2, stroke: CHROME.surface, fill: SERIES[i] }} isAnimationActive animationDuration={1000} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/* --------------------------------------------------------------- horizontal bars */

export function HBars({
  data,
  title,
  subtitle,
  format = (v) => num(v, 3),
  diverging,
  color = SERIES[0],
  height,
  valueLabel = "Value",
}: {
  data: { label: string; value: number }[];
  title: string;
  subtitle?: string;
  format?: (v: number) => string;
  diverging?: boolean;
  color?: string;
  height?: number;
  valueLabel?: string;
}) {
  const h = height ?? Math.max(160, data.length * 26 + 40);
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={diverging ? [{ label: "Positive", color: DIVERGING.positive }, { label: "Negative", color: DIVERGING.negative }] : undefined}
      table={{ columns: ["Feature", valueLabel], rows: data.map((d) => [d.label, format(d.value)]) }}
      height={h}
    >
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }} barCategoryGap={4}>
          <CartesianGrid {...gridProps} horizontal={false} vertical />
          <XAxis type="number" {...axisProps} tickFormatter={(v) => format(v)} />
          <YAxis type="category" dataKey="label" {...axisProps} width={150} tick={{ fill: CHROME.ink2, fontSize: 11 }} interval={0} />
          {diverging && <ReferenceLine x={0} stroke={CHROME.axis} />}
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={<ChartTooltip valueFormat={format} />} />
          <Bar dataKey="value" name={valueLabel} radius={4} isAnimationActive animationDuration={900}>
            {data.map((d) => (
              <Cell key={d.label} fill={diverging ? (d.value >= 0 ? DIVERGING.positive : DIVERGING.negative) : color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/* --------------------------------------------------------------- threshold sweep */

export function ThresholdSweep({ data, threshold, title = "Metrics across decision thresholds", height = 260 }: { data: SweepPoint[]; threshold?: number; title?: string; height?: number }) {
  return (
    <ChartFrame
      title={title}
      subtitle={threshold != null ? `Marker at the current threshold ${num(threshold, 2)}` : undefined}
      legend={[
        { label: "Precision", color: SERIES[0], shape: "line" },
        { label: "Recall", color: SERIES[1], shape: "line" },
        { label: "F1", color: SERIES[2], shape: "line" },
      ]}
      table={{ columns: ["Threshold", "Precision", "Recall", "F1", "Alert rate"], rows: data.map((d) => [num(d.threshold, 3), num(d.precision, 3), num(d.recall, 3), num(d.f1, 3), d.alert_rate != null ? num(d.alert_rate, 4) : "—"]) }}
      height={height}
    >
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="threshold" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => num(v, 1)} label={{ value: "Decision threshold", position: "insideBottom", offset: -10 }} />
          <YAxis domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
          {threshold != null && <ReferenceLine x={threshold} stroke={CHROME.ink2} />}
          <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(l) => `Threshold ${num(l as number, 3)}`} />} />
          <Line dataKey="precision" name="Precision" stroke={SERIES[0]} strokeWidth={2} dot={false} />
          <Line dataKey="recall" name="Recall" stroke={SERIES[1]} strokeWidth={2} dot={false} />
          <Line dataKey="f1" name="F1" stroke={SERIES[2]} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

/** Metric tiles. Columns follow the available width (auto-fill), so narrow cards never clip labels. */
export function MetricGrid({ items, minWidth = 128 }: { items: { label: string; value: string; hint?: string }[]; minWidth?: number }) {
  return (
    <dl className="grid gap-3" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${minWidth}px, 1fr))` }}>
      {items.map((it) => (
        <div key={it.label} className="rounded-xl border border-line bg-white/[0.02] px-3.5 py-3">
          <dt className="truncate text-[11px] uppercase tracking-[0.12em] text-ink-3" title={it.label}>{it.label}</dt>
          <dd className="mt-1 font-display text-lg font-semibold text-ink tabular">{it.value}</dd>
          {it.hint && <dd className="text-[11px] text-ink-3">{it.hint}</dd>}
        </div>
      ))}
    </dl>
  );
}

/* --------------------------------------------------------------- overlay of several curves */

export interface CurveSeries {
  name: string;
  color: string;
  points: object[];
  dashed?: boolean;
}

/** Several curves on shared numeric axes (e.g. PR curves of LR, Isolation Forest and the hybrid). */
export function CurveOverlay({
  title,
  subtitle,
  series,
  x,
  y,
  xLabel,
  yLabel,
  height = 280,
  step,
  baseline,
  diagonal,
}: {
  title: string;
  subtitle?: string;
  series: CurveSeries[];
  x: string;
  y: string;
  xLabel: string;
  yLabel: string;
  height?: number;
  step?: boolean;
  baseline?: { y: number; label: string };
  diagonal?: boolean;
}) {
  const val = (p: object, k: string) => (p as Record<string, number>)[k];
  const sorted = series.map((s) => ({ ...s, points: [...s.points].sort((a, b) => val(a, x) - val(b, x)) }));
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={[
        ...sorted.map((s) => ({ label: s.name, color: s.color, shape: (s.dashed ? "dash" : "line") as "dash" | "line" })),
        ...(baseline ? [{ label: baseline.label, color: CHROME.muted, shape: "dash" as const }] : []),
        ...(diagonal ? [{ label: "Chance", color: CHROME.muted, shape: "dash" as const }] : []),
      ]}
      table={{ columns: ["Series", xLabel, yLabel], rows: sorted.flatMap((s) => s.points.filter((_, i) => i % 5 === 0).map((p) => [s.name, num(val(p, x), 4), num(val(p, y), 4)])) }}
      height={height}
    >
      <ResponsiveContainer>
        <LineChart margin={{ top: 6, right: 12, bottom: 18, left: 0 }}>
          <CartesianGrid {...gridProps} vertical />
          <XAxis dataKey={x} type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v) => num(v, 1)} label={{ value: xLabel, position: "insideBottom", offset: -10 }} allowDuplicatedCategory={false} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
          {baseline && <ReferenceLine y={baseline.y} stroke={CHROME.muted} strokeDasharray="4 4" />}
          {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHROME.muted} strokeDasharray="4 4" />}
          <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(_, d) => `${xLabel} ${num(d?.[x] as number, 3)}`} />} />
          {sorted.map((s) => (
            <Line key={s.name} data={s.points} dataKey={y} name={s.name} stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} dot={false} type={step ? "stepAfter" : "linear"} isAnimationActive animationDuration={1000} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

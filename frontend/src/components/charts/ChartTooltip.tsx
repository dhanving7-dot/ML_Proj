import type { ReactNode } from "react";

interface Entry {
  name?: string | number;
  value?: number | string | (number | string)[];
  color?: string;
  dataKey?: string | number | ((obj: unknown) => unknown);
  payload?: Record<string, unknown>;
}

/**
 * Tooltip content for Recharts. Text uses text tokens; the coloured swatch carries identity.
 * ``valueFormat`` formats numbers; ``title`` derives a heading from the hovered datum.
 */
export function ChartTooltip({
  active,
  payload,
  label,
  valueFormat = (v) => String(v),
  title,
}: {
  active?: boolean;
  payload?: Entry[];
  label?: string | number;
  valueFormat?: (v: number) => string;
  title?: (label: string | number | undefined, datum: Record<string, unknown> | undefined) => ReactNode;
}) {
  if (!active || !payload?.length) return null;
  const datum = payload[0]?.payload;
  return (
    <div className="glass min-w-[140px] rounded-lg border border-line-strong px-3 py-2 text-xs shadow-xl">
      <p className="mb-1.5 font-medium text-ink">{title ? title(label, datum) : label}</p>
      <ul className="space-y-1">
        {payload.map((p, i) => (
          <li key={`${String(p.dataKey)}-${i}`} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-ink-2">
              <span className="size-2 rounded-full" style={{ background: p.color }} aria-hidden="true" />
              {p.name}
            </span>
            <span className="font-mono text-ink tabular">{typeof p.value === "number" ? valueFormat(p.value) : Array.isArray(p.value) ? p.value.join(" – ") : p.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

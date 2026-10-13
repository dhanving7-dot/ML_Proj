import { Table2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { cn } from "../../lib/format";

export interface LegendItem {
  label: string;
  color: string;
  shape?: "dot" | "line" | "dash";
}

export interface TableView {
  columns: string[];
  rows: (string | number | null)[][];
}

/**
 * Chart container: title, legend (always present for >= 2 series), a fixed plot height that
 * includes the axis band, and a table-view twin so values never depend on colour or hover.
 */
export function ChartFrame({
  title,
  subtitle,
  legend,
  table,
  height = 260,
  children,
  className,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  legend?: LegendItem[];
  table?: TableView;
  height?: number;
  children: ReactNode;
  className?: string;
  footer?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <figure className={cn("min-w-0", className)}>
      <figcaption className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-ink">{title}</p>
          {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {legend && legend.length > 0 && (
            <ul className="flex flex-wrap items-center gap-x-3 gap-y-1" aria-label="Legend">
              {legend.map((l) => (
                <li key={l.label} className="flex items-center gap-1.5 text-xs text-ink-2">
                  <span
                    aria-hidden="true"
                    className={cn("inline-block shrink-0", l.shape === "line" ? "h-0.5 w-3.5 rounded" : l.shape === "dash" ? "h-0 w-3.5 border-t-2 border-dashed" : "size-2.5 rounded-full")}
                    style={l.shape === "dash" ? { borderColor: l.color } : { background: l.color }}
                  />
                  {l.label}
                </li>
              ))}
            </ul>
          )}
          {table && (
            <button
              type="button"
              onClick={() => setShowTable((v) => !v)}
              aria-pressed={showTable}
              className="inline-flex items-center gap-1 rounded-md border border-line px-2 py-0.5 text-[11px] text-ink-3 transition-colors hover:border-line-strong hover:text-ink"
            >
              <Table2 className="size-3" aria-hidden="true" />
              {showTable ? "Chart" : "Table"}
            </button>
          )}
        </div>
      </figcaption>
      {showTable && table ? (
        <div className="overflow-auto rounded-lg border border-line" style={{ maxHeight: height }}>
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-surface-2">
              <tr>
                {table.columns.map((c) => (
                  <th key={c} className="px-3 py-2 font-medium text-ink-2">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular">
              {table.rows.map((r, i) => (
                <tr key={i} className="border-t border-line">
                  {r.map((v, j) => (
                    <td key={j} className="px-3 py-1.5 text-ink-2">
                      {v ?? "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ height }} className="w-full min-w-0">
          {children}
        </div>
      )}
      {footer && <div className="mt-2 text-xs text-ink-3">{footer}</div>}
    </figure>
  );
}

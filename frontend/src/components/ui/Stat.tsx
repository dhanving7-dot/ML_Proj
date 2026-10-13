import { useInView } from "motion/react";
import { useRef, type ReactNode } from "react";
import { useCountUp } from "../../hooks/useCountUp";
import { cn } from "../../lib/format";

interface StatProps {
  label: string;
  value: number | null | undefined;
  format: (v: number) => string;
  hint?: ReactNode;
  icon?: ReactNode;
  className?: string;
  accent?: boolean;
}

/** KPI tile. The counter starts when visible and always lands on the exact API value. */
export function Stat({ label, value, format, hint, icon, className, accent }: StatProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const shown = useCountUp(value, 1200, inView);
  return (
    <div ref={ref} className={cn("flex min-w-0 flex-col gap-2", className)}>
      <div className="flex items-center gap-2 text-[12px] font-medium uppercase tracking-[0.14em] text-ink-3">
        {icon}
        <span className="truncate">{label}</span>
      </div>
      <div className={cn("font-display text-3xl font-semibold tracking-tight sm:text-[2.1rem]", accent ? "text-gradient" : "text-ink")} aria-live="polite">
        {value == null ? <span className="text-ink-3">—</span> : format(shown ?? value)}
      </div>
      {hint && <div className="text-xs leading-relaxed text-ink-3">{hint}</div>}
    </div>
  );
}

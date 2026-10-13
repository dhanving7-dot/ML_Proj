import { CircleCheck, CircleDashed, CircleX, Info, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "../../lib/format";

export type Tone = "good" | "warning" | "serious" | "critical" | "info" | "neutral";

const toneClass: Record<Tone, string> = {
  good: "text-good-text border-good/40 bg-good/10",
  warning: "text-warning border-warning/40 bg-warning/10",
  serious: "text-serious border-serious/40 bg-serious/10",
  critical: "text-critical-text border-critical/50 bg-critical/10",
  info: "text-cyan border-cyan/30 bg-cyan/10",
  neutral: "text-ink-2 border-line-strong bg-white/[0.04]",
};

const toneIcon: Record<Tone, typeof CircleCheck> = {
  good: CircleCheck,
  warning: TriangleAlert,
  serious: TriangleAlert,
  critical: CircleX,
  info: Info,
  neutral: CircleDashed,
};

/** Status is never colour-only: every pill carries an icon and a text label. */
export function StatusPill({ tone, children, className }: { tone: Tone; children: ReactNode; className?: string }) {
  const Icon = toneIcon[tone];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium", toneClass[tone], className)}>
      <Icon className="size-3.5 shrink-0" aria-hidden="true" />
      {children}
    </span>
  );
}

export function Chip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={cn("inline-flex items-center rounded-md border border-line bg-white/[0.03] px-2 py-0.5 font-mono text-[11px] text-ink-2", className)}>
      {children}
    </span>
  );
}

export const driftTone = (s: string | undefined | null): Tone =>
  s === "stable" ? "good" : s === "investigate" ? "warning" : s === "significant_shift" ? "critical" : "neutral";

export const severityTone = (s: string): Tone => (s === "critical" ? "critical" : s === "warning" ? "warning" : "info");

export const jobTone = (s: string): Tone =>
  s === "succeeded" ? "good" : s === "rejected" ? "warning" : s === "failed" ? "critical" : "info";

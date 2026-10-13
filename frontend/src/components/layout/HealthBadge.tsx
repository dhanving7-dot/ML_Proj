import { useHealth } from "../../hooks/HealthContext";
import { cn } from "../../lib/format";

/** Real backend status from GET /api/health (polled every 15 s). Never simulated. */
export function HealthBadge({ className, detailed }: { className?: string; detailed?: boolean }) {
  const { data, error, loading } = useHealth();
  let tone: "good" | "warning" | "critical" | "neutral" = "neutral";
  let label = "Checking API…";
  if (error && !data) {
    tone = "critical";
    label = "API offline";
  } else if (data) {
    if (data.status === "ok") {
      tone = "good";
      label = detailed ? `API online · model ${data.model_version}` : `Model ${data.model_version}`;
    } else {
      tone = "warning";
      label = data.model_loaded ? "API degraded" : "No model loaded";
    }
  }
  const dot = { good: "bg-good", warning: "bg-warning", critical: "bg-critical", neutral: "bg-ink-3" }[tone];
  const title = data
    ? `API ${data.api_version} · database ${data.database} · ${data.model_loaded ? `model ${data.model_version} (${data.model_type})` : "no model"}${
        data.consistency_check ? ` · consistency check ${data.consistency_check.passed ? "passed" : "FAILED"}` : ""
      }`
    : error?.message;
  return (
    <span
      role="status"
      title={title}
      className={cn("inline-flex items-center gap-2 rounded-full border border-line bg-white/[0.03] px-3 py-1 text-xs text-ink-2", className)}
    >
      <span className="relative flex size-2" aria-hidden="true">
        {tone === "good" && !loading && <span className={cn("absolute inset-0 rounded-full animate-pulse-ring", dot)} />}
        <span className={cn("relative size-2 rounded-full", dot)} />
      </span>
      <span>{label}</span>
      {tone === "good" && <span className="sr-only">: backend healthy</span>}
    </span>
  );
}

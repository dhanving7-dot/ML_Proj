import { AnimatePresence, motion } from "motion/react";
import { ChevronDown, CircleCheck, CircleX, Info } from "lucide-react";
import { useState } from "react";
import { api } from "../../api/endpoints";
import type { PredictionResponse } from "../../api/types";
import { useAction, useApi } from "../../hooks/useApi";
import { cn, dateTime, humanize, ms, num } from "../../lib/format";
import { OUTCOME } from "../charts/theme";
import { Button } from "../ui/Button";
import { CopyButton } from "../ui/CopyButton";
import { Panel } from "../ui/Panel";
import { Chip } from "../ui/Status";
import { ErrorState, LoadingBlock } from "../ui/States";
import { useToast } from "../ui/Toast";
import { ProbabilityGauge } from "./ProbabilityGauge";

export function PredictionResult({ result, groundTruth }: { result: PredictionResponse; groundTruth?: { studentId: number; outcome: 0 | 1 } }) {
  const placed = result.prediction === 1;
  const [showTrace, setShowTrace] = useState(false);
  const toast = useToast();
  const feedback = useAction((v: 0 | 1) => api.recordOutcome(result.request_id, v));
  const t = result.timings;
  const parts = [
    { k: "preprocessing_ms", label: "Preprocessing", color: "#9085e9" },
    { k: "inference_ms", label: "Inference", color: "#3987e5" },
    { k: "persistence_ms", label: "Persistence", color: "#199e70" },
  ];
  const total = parts.reduce((a, p) => a + (t[p.k] ?? 0), 0) || 1;

  const record = async (v: 0 | 1) => {
    const res = await feedback.run(v);
    if (res) toast({ tone: "success", title: "Outcome recorded", body: `Stored as ground truth for request ${result.request_id.slice(0, 8)}…; it now counts toward live accuracy.` });
  };

  return (
    <motion.div initial={{ opacity: 0, y: 24, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}>
      <Panel luminous className="overflow-hidden">
        <div className="pointer-events-none absolute -right-20 -top-20 size-64 rounded-full blur-3xl" style={{ background: `${placed ? OUTCOME.placed : OUTCOME.notPlaced}33` }} aria-hidden="true" />
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-3">Model estimate</p>
        <div className="mt-3 flex flex-col items-center gap-2 sm:flex-row sm:items-center sm:gap-6">
          <ProbabilityGauge probability={result.probability} threshold={result.threshold} />
          <div className="text-center sm:text-left">
            <motion.p
              initial={{ opacity: 0, y: 12, filter: "blur(6px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{ delay: 0.35, duration: 0.7 }}
              className="font-display text-4xl font-semibold"
              style={{ color: placed ? OUTCOME.placed : OUTCOME.notPlaced }}
            >
              {result.label}
            </motion.p>
            <p className="mt-1 text-sm text-ink-2">
              P(placed) = <span className="font-mono text-ink">{num(result.probability, 4)}</span> {placed ? "≥" : "<"} threshold {num(result.threshold, 2)}
            </p>
          </div>
        </div>

        {groundTruth && (
          <div className="mt-5 flex items-start gap-2 rounded-xl border border-line bg-white/[0.02] p-3 text-sm">
            {groundTruth.outcome === result.prediction ? <CircleCheck className="mt-0.5 size-4 shrink-0 text-good-text" aria-hidden="true" /> : <CircleX className="mt-0.5 size-4 shrink-0 text-critical-text" aria-hidden="true" />}
            <p className="text-ink-2">
              Dataset ground truth for student #{groundTruth.studentId} (held-out test split): <strong className="text-ink">{groundTruth.outcome ? "Placed" : "Not Placed"}</strong>. The model was{" "}
              {groundTruth.outcome === result.prediction ? "correct" : "wrong"} on this student.
            </p>
          </div>
        )}

        <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Model</dt>
            <dd className="mt-0.5 text-ink">
              {result.model_version} <span className="text-ink-3">· {humanize(result.model_type)}</span>
            </dd>
          </div>
          <div>
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Latency</dt>
            <dd className="mt-0.5 font-mono text-ink">{ms(result.latency_ms)}</dd>
          </div>
          <div className="col-span-2">
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Request ID</dt>
            <dd className="mt-0.5 flex items-center gap-2">
              <span className="truncate font-mono text-xs text-ink">{result.request_id}</span>
              <CopyButton text={result.request_id} />
            </dd>
          </div>
        </dl>

        <div className="mt-4">
          <p className="mb-1.5 text-[11px] uppercase tracking-[0.12em] text-ink-3">Server time breakdown</p>
          <div className="flex h-2 w-full gap-[2px] overflow-hidden rounded-full" aria-hidden="true">
            {parts.map((p) => (
              <motion.span key={p.k} initial={{ width: 0 }} animate={{ width: `${((t[p.k] ?? 0) / total) * 100}%` }} transition={{ duration: 0.9, delay: 0.4 }} style={{ background: p.color }} className="h-full" />
            ))}
          </div>
          <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
            {parts.map((p) => (
              <li key={p.k} className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ background: p.color }} aria-hidden="true" />
                {p.label} <span className="font-mono text-ink">{ms(t[p.k])}</span>
              </li>
            ))}
          </ul>
        </div>

        {result.imputed_fields.length > 0 && (
          <p className="mt-4 flex flex-wrap items-center gap-1.5 text-xs text-ink-2">
            <Info className="size-3.5 text-cyan" aria-hidden="true" /> Imputed with training medians:
            {result.imputed_fields.map((f) => (
              <Chip key={f}>{f}</Chip>
            ))}
          </p>
        )}

        <p className="mt-5 rounded-xl border border-warning/30 bg-warning/[0.06] p-3 text-xs leading-relaxed text-ink-2">
          <strong className="text-warning">Estimate, not a guarantee.</strong> {result.disclaimer}
        </p>

        <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-line pt-4">
          <span className="text-xs text-ink-3">Know the real outcome?</span>
          <Button size="sm" variant="secondary" disabled={!!feedback.data} loading={feedback.pending} onClick={() => record(1)}>
            Was placed
          </Button>
          <Button size="sm" variant="secondary" disabled={!!feedback.data} loading={feedback.pending} onClick={() => record(0)}>
            Was not placed
          </Button>
          {feedback.error && <span className="text-xs text-critical-text">{feedback.error.message}</span>}
          <Button size="sm" variant="ghost" className="ml-auto" onClick={() => setShowTrace((v) => !v)} aria-expanded={showTrace} icon={<ChevronDown className={cn("size-3.5 transition-transform", showTrace && "rotate-180")} aria-hidden="true" />}>
            {showTrace ? "Hide" : "Show"} trace
          </Button>
        </div>
        <AnimatePresence initial={false}>{showTrace && <Trace id={result.request_id} />}</AnimatePresence>
      </Panel>
    </motion.div>
  );
}

function Trace({ id }: { id: string }) {
  const q = useApi((s) => api.prediction(id, { signal: s }), [id]);
  return (
    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
      <div className="pt-4">
        {q.error ? (
          <ErrorState error={q.error} onRetry={q.refetch} compact />
        ) : !q.data ? (
          <LoadingBlock rows={4} />
        ) : (
          <>
            <ol className="relative space-y-3 border-l border-line pl-5">
              {q.data.trace.map((s) => (
                <li key={s.step} className="relative text-sm">
                  <span className="absolute -left-[25px] top-1.5 size-2 rounded-full bg-cyan" aria-hidden="true" />
                  <p className="font-medium text-ink">
                    {s.step} {s.ms != null && <span className="font-mono text-xs text-ink-3">· {ms(s.ms)}</span>}
                  </p>
                  <p className="text-xs text-ink-2">{s.detail}</p>
                </li>
              ))}
            </ol>
            <p className="mt-4 text-xs text-ink-3">
              Stored {dateTime(q.data.created_at)} · feature vector the model saw ({q.data.feature_vector.length} values, first 12 shown):
            </p>
            <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px] sm:grid-cols-3">
              {q.data.feature_vector.slice(0, 12).map((v, i) => (
                <span key={i} className="truncate rounded bg-white/[0.03] px-2 py-1 text-ink-2" title={q.data!.feature_names[i]}>
                  {q.data!.feature_names[i] ?? `f${i}`} = <span className="text-ink">{num(v, 3)}</span>
                </span>
              ))}
            </div>
          </>
        )}
      </div>
    </motion.div>
  );
}

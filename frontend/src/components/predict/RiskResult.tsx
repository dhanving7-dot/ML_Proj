import { AnimatePresence, motion } from "motion/react";
import { ChevronDown, CircleCheck, CircleX, Info, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { api } from "../../api/endpoints";
import type { PredictionResponse } from "../../api/types";
import { useAction, useApi } from "../../hooks/useApi";
import { cn, dateTime, decisionLabel, ms, num, pct, prob } from "../../lib/format";
import { DECISION_COLOR, DIVERGING } from "../charts/theme";
import { Button } from "../ui/Button";
import { CopyButton } from "../ui/CopyButton";
import { Panel } from "../ui/Panel";
import { Chip, decisionTone, StatusPill } from "../ui/Status";
import { ErrorState, LoadingBlock } from "../ui/States";
import { useToast } from "../ui/Toast";
import { RiskGauge } from "./RiskGauge";

export interface GroundTruth {
  rowId: string;
  actual: 0 | 1;
}

/** Everything shown here comes from the API response; explanations are the models' own arithmetic. */
export function RiskResult({ result, groundTruth }: { result: PredictionResponse; groundTruth?: GroundTruth }) {
  const [showTrace, setShowTrace] = useState(false);
  const toast = useToast();
  const feedback = useAction((v: 0 | 1) => api.recordOutcome(result.request_id, v));
  const { scores: s, policy: p, explanation: e } = result;
  const flagged = result.decision !== "LOW_RISK";
  const t = result.timings;
  const parts = [
    { k: "preprocessing_ms", label: "Preprocessing", color: "#9085e9" },
    { k: "inference_ms", label: "Scoring", color: "#3987e5" },
    { k: "explanation_ms", label: "Explanation", color: "#c98500" },
    { k: "persistence_ms", label: "Persistence", color: "#199e70" },
  ];
  const total = parts.reduce((a, x) => a + (t[x.k] ?? 0), 0) || 1;
  const maxAbs = Math.max(...e.lr_top_contributions.map((c) => Math.abs(c.contribution)), 1e-9);

  const record = async (v: 0 | 1) => {
    const res = await feedback.run(v);
    if (res) toast({ tone: "success", title: "Outcome recorded", body: `Stored as ${v ? "confirmed fraud" : "confirmed legitimate"} for request ${result.request_id.slice(0, 8)}…; it now counts toward live performance.` });
  };

  return (
    <motion.div initial={{ opacity: 0, y: 24, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}>
      <Panel luminous className="overflow-hidden">
        <div className="pointer-events-none absolute -right-20 -top-20 size-64 rounded-full blur-3xl" style={{ background: `${DECISION_COLOR[result.decision]}33` }} aria-hidden="true" />
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-3">Risk assessment</p>
        <div className="mt-3 flex flex-col items-center gap-2 sm:flex-row sm:items-center sm:gap-6">
          <RiskGauge score={s.hybrid_score} review={p.review_threshold} high={p.high_risk_threshold} decision={result.decision} />
          <div className="text-center sm:text-left">
            <motion.p
              initial={{ opacity: 0, y: 12, filter: "blur(6px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              transition={{ delay: 0.35, duration: 0.7 }}
              className="font-display text-4xl font-semibold"
              style={{ color: DECISION_COLOR[result.decision] }}
            >
              {decisionLabel(result.decision)}
            </motion.p>
            <p className="mt-1 font-mono text-xs text-ink-3">{result.decision}</p>
            <p className="mt-2 text-sm text-ink-2">
              Hybrid score <span className="font-mono text-ink">{num(s.hybrid_score, 4)}</span>; REVIEW at ≥ {num(p.review_threshold, 4)}, HIGH_RISK at ≥ {num(p.high_risk_threshold, 4)}
            </p>
          </div>
        </div>

        {groundTruth && (
          <div className="mt-5 flex items-start gap-2 rounded-xl border border-line bg-white/[0.02] p-3 text-sm">
            {(groundTruth.actual === 1) === flagged ? (
              <CircleCheck className="mt-0.5 size-4 shrink-0 text-good-text" aria-hidden="true" />
            ) : (
              <CircleX className="mt-0.5 size-4 shrink-0 text-critical-text" aria-hidden="true" />
            )}
            <p className="text-ink-2">
              Replayed row <span className="font-mono text-ink">{groundTruth.rowId}</span> from the held-out test split. Its recorded class is{" "}
              <strong className="text-ink">{groundTruth.actual ? "Fraud" : "Legitimate"}</strong>; the system {flagged ? "flagged it for review or blocking" : "did not flag it"}.
            </p>
          </div>
        )}

        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          <ScoreTile
            title="Logistic Regression"
            value={prob(s.lr_probability)}
            caption="fraud probability (prior-corrected)"
            flag={result.flags.logistic_regression}
            detail={`Flags at ≥ ${prob(p.lr_threshold)}. ${pct(e.lr_exceedance, 3)} of training transactions score higher.`}
          />
          <ScoreTile
            title="Isolation Forest"
            value={pct(s.anomaly_percentile, 2)}
            caption="anomaly percentile (not a probability)"
            flag={result.flags.isolation_forest}
            detail={`Raw score ${num(s.anomaly_score, 4)}. Flags in the top ${pct(1 - p.if_flag_percentile, 2)} of training transactions.`}
          />
          <ScoreTile
            title="Hybrid"
            value={num(s.hybrid_score, 4)}
            caption={`${num(p.fusion_weight_lr, 2)} × ${num(s.lr_rarity, 3)} + ${num(p.fusion_weight_if, 2)} × ${num(s.if_rarity, 3)}`}
            detail="Weighted rarity of the two scores; the weight was tuned on validation data."
          />
        </div>

        <div className="mt-6">
          <p className="text-sm font-medium text-ink">Why the Logistic Regression scored it this way</p>
          <p className="mt-1 text-xs leading-relaxed text-ink-3">
            Exact log-odds contributions (coefficient × standardised value) of the six largest terms. V1–V28 are anonymised PCA components; their meaning is not published, so
            no real-world reason can be attached to them.
          </p>
          <ul className="mt-3 space-y-1.5">
            {e.lr_top_contributions.map((c) => (
              <li key={c.feature} className="grid grid-cols-[8.5rem_minmax(0,1fr)_4.5rem] items-center gap-2 text-xs">
                <span className="truncate text-ink-2" title={c.label}>
                  {c.label}
                </span>
                <span className="relative h-2.5 rounded bg-white/[0.04]" aria-hidden="true">
                  <span
                    className="absolute top-0 h-full rounded"
                    style={{
                      background: c.contribution >= 0 ? DIVERGING.negative : DIVERGING.positive,
                      width: `${(Math.abs(c.contribution) / maxAbs) * 50}%`,
                      left: c.contribution >= 0 ? "50%" : `${50 - (Math.abs(c.contribution) / maxAbs) * 50}%`,
                    }}
                  />
                  <span className="absolute left-1/2 top-[-2px] h-[14px] w-px bg-white/20" />
                </span>
                <span className="text-right font-mono text-ink tabular">
                  {c.contribution >= 0 ? "+" : ""}
                  {num(c.contribution, 2)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-3">
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ background: DIVERGING.negative }} aria-hidden="true" /> towards fraud (+)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ background: DIVERGING.positive }} aria-hidden="true" /> towards legitimate (−)
            </span>
            <span>
              intercept {num(e.lr_intercept, 2)} · prior correction {num(e.lr_prior_correction, 2)} · total log-odds {num(e.lr_logit, 2)}
            </span>
          </p>
          <p className="mt-3 text-xs text-ink-3">The Isolation Forest does not attribute its score to individual features, so only its percentile is reported.</p>
        </div>

        {(result.out_of_training_range.length > 0 || e.anomaly_review_trigger) && (
          <div className="mt-4 space-y-2">
            {result.out_of_training_range.length > 0 && (
              <p className="flex flex-wrap items-center gap-1.5 rounded-xl border border-warning/30 bg-warning/[0.06] p-3 text-xs text-ink-2">
                <TriangleAlert className="size-3.5 text-warning" aria-hidden="true" /> Outside the range seen in training (scores are extrapolations):
                {result.out_of_training_range.map((f) => (
                  <Chip key={f}>{f}</Chip>
                ))}
              </p>
            )}
            {e.anomaly_review_trigger && (
              <p className="flex items-center gap-1.5 text-xs text-ink-2">
                <Info className="size-3.5 text-cyan" aria-hidden="true" /> Routed to REVIEW by the anomaly trigger (anomaly percentile above {pct(p.if_review_percentile, 2)}).
              </p>
            )}
          </div>
        )}

        <dl className="mt-6 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-3">Model</dt>
            <dd className="mt-0.5 text-ink">{result.model_version}</dd>
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
            {parts.map((x) => (
              <motion.span key={x.k} initial={{ width: 0 }} animate={{ width: `${((t[x.k] ?? 0) / total) * 100}%` }} transition={{ duration: 0.9, delay: 0.4 }} style={{ background: x.color }} className="h-full" />
            ))}
          </div>
          <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
            {parts.map((x) => (
              <li key={x.k} className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ background: x.color }} aria-hidden="true" />
                {x.label} <span className="font-mono text-ink">{ms(t[x.k])}</span>
              </li>
            ))}
          </ul>
        </div>

        <p className="mt-5 rounded-xl border border-warning/30 bg-warning/[0.06] p-3 text-xs leading-relaxed text-ink-2">
          <strong className="text-warning">Decision support, not a verdict.</strong> {result.disclaimer}
        </p>

        <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-line pt-4">
          <span className="text-xs text-ink-3">Know the confirmed outcome?</span>
          <Button size="sm" variant="secondary" disabled={!!feedback.data} loading={feedback.pending} onClick={() => record(1)}>
            Confirmed fraud
          </Button>
          <Button size="sm" variant="secondary" disabled={!!feedback.data} loading={feedback.pending} onClick={() => record(0)}>
            Confirmed legitimate
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

function ScoreTile({ title, value, caption, detail, flag }: { title: string; value: string; caption: string; detail: string; flag?: boolean }) {
  return (
    <div className="rounded-xl border border-line bg-white/[0.02] p-3.5">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[11px] uppercase tracking-[0.12em] text-ink-3">{title}</p>
        {flag !== undefined && <StatusPill tone={flag ? "warning" : "neutral"}>{flag ? "flag" : "no flag"}</StatusPill>}
      </div>
      <p className="mt-1.5 font-display text-xl font-semibold text-ink tabular">{value}</p>
      <p className="text-[11px] text-ink-3">{caption}</p>
      <p className="mt-2 text-xs leading-relaxed text-ink-2">{detail}</p>
    </div>
  );
}

export function Trace({ id }: { id: string }) {
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
              Stored {dateTime(q.data.created_at)} · feature vector the models saw ({q.data.feature_vector.length} standardised values, first 12 shown):
            </p>
            <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px] sm:grid-cols-3">
              {q.data.feature_vector.slice(0, 12).map((v, i) => (
                <span key={i} className="truncate rounded bg-white/[0.03] px-2 py-1 text-ink-2" title={q.data!.feature_labels[i]}>
                  {q.data!.feature_names[i] ?? `f${i}`} = <span className="text-ink">{num(v, 3)}</span>
                </span>
              ))}
            </div>
            <div className="mt-3">
              <StatusPill tone={decisionTone(q.data.decision)}>{q.data.decision}</StatusPill>
            </div>
          </>
        )}
      </div>
    </motion.div>
  );
}

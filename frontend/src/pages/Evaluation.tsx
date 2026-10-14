import { FlaskConical, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/endpoints";
import type { DecisionRow, EvaluationBlock, EvaluationSummary, ExploreRequest, Policy } from "../api/types";
import { ConfusionMatrix, CurveOverlay, MetricGrid, ReliabilityChart, ThresholdSweep } from "../components/charts/Common";
import { COMPONENT, DECISION_COLOR } from "../components/charts/theme";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { Slider, Toggle } from "../components/ui/Fields";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { TabPanel, Tabs } from "../components/ui/Tabs";
import { useApi } from "../hooks/useApi";
import { COMPONENT_LABELS, decisionLabel, int, num, pct } from "../lib/format";
import { DataTable, LinesChart } from "./lab/kit";

type View = "test" | "explore";

export default function Evaluation() {
  const summary = useApi((s) => api.evaluationSummary({ signal: s }), []);
  const [view, setView] = useState<View>("test");
  return (
    <>
      <PageHeader
        crumb="Evaluation"
        title="Model evaluation lab"
        description="Logistic Regression, Isolation Forest and the hybrid compared on the same rows. Held-out test results are read-only (evaluated once at training time); what-if policies run live on the validation split."
      />
      <Section className="space-y-6">
        {summary.error && !summary.data ? (
          summary.error.status === 503 ? (
            <EmptyState title="No active model" action={<Button to="/models" size="sm">Train a model</Button>}>
              {summary.error.message}
            </EmptyState>
          ) : (
            <ErrorState error={summary.error} onRetry={summary.refetch} title="The evaluation could not be loaded" />
          )
        ) : !summary.data ? (
          <LoadingBlock rows={10} label="Loading evaluation" />
        ) : (
          <>
            <Methodology s={summary.data} />
            <Tabs
              idPrefix="eval-view"
              ariaLabel="Evaluation view"
              value={view}
              onChange={setView}
              items={[
                { key: "test", label: "Held-out test (read-only)" },
                { key: "explore", label: "Policy explorer (validation)" },
              ]}
              className="w-fit"
            />
            <TabPanel idPrefix="eval-view" value={view}>
              {view === "test" ? <TestResults s={summary.data} /> : <Explorer s={summary.data} />}
            </TabPanel>
          </>
        )}
      </Section>
    </>
  );
}

function Methodology({ s }: { s: EvaluationSummary }) {
  const sp = s.splits;
  return (
    <Panel luminous>
      <PanelHeader eyebrow={`Model ${s.model_version} · methodology`} title="How these numbers were produced" />
      <div className="grid gap-6 lg:grid-cols-2 [&>*]:min-w-0">
        <ol className="list-decimal space-y-1.5 pl-5 text-sm text-ink-2">
          {s.methodology.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ol>
        <div className="space-y-3 text-sm">
          <DataTable
            columns={["Split", "Rows", "Fraud", "Used for"]}
            numericFrom={1}
            rows={(["train", "validation", "test"] as const).map((k) => [k, int(sp[k].rows), int(sp[k].fraud_rows), <span className="whitespace-normal text-xs">{sp.usage[k]}</span>])}
          />
          <p className="text-xs text-ink-3">
            Cleaning: {int(Number(sp.cleaning.dropped_duplicates ?? 0))} exact duplicate rows dropped before splitting ({String(sp.cleaning.duplicate_policy ?? "")}).
          </p>
          <div className="min-w-0 overflow-x-auto break-all rounded-xl border border-line bg-white/[0.02] p-3 font-mono text-[11px] leading-relaxed text-ink-2">
            <p>{s.normalisation}</p>
            <p className="mt-1">{s.fusion_formula}</p>
            {Object.entries(s.decision_rules).map(([k, v]) => (
              <p key={k} className="mt-1">
                <span style={{ color: DECISION_COLOR[k as keyof typeof DECISION_COLOR] }}>{k}</span>: {v}
              </p>
            ))}
          </div>
          <p className="text-xs text-ink-3">
            Fusion weight: {s.tuning.weight_source}. {s.tuning.high_risk_rule}; {s.tuning.review_rule}. Anomaly review trigger: {s.tuning.if_review_trigger}.
          </p>
          {s.tuning.notes.map((n) => (
            <p key={n} className="text-xs text-warning">
              {n}
            </p>
          ))}
        </div>
      </div>
    </Panel>
  );
}

const ROWS = ["logistic_regression", "isolation_forest", "hybrid_high_risk", "hybrid_review_or_high"] as const;

export function ComparisonTable({ m }: { m: EvaluationBlock }) {
  return (
    <DataTable
      columns={["Component", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "False pos.", "False neg.", "Alert rate", "Threshold"]}
      rows={ROWS.map((k) => [
        COMPONENT_LABELS[k],
        num(m[k].precision, 4),
        num(m[k].recall, 4),
        num(m[k].f1, 4),
        num(m[k].average_precision, 4),
        num(m[k].roc_auc, 4),
        int(m[k].false_positives),
        int(m[k].false_negatives),
        pct(m[k].alert_rate, 3),
        k === "hybrid_review_or_high" ? "REVIEW+" : num(m[k].threshold, 4),
      ])}
    />
  );
}

function DecisionTable({ rows }: { rows: DecisionRow[] }) {
  return (
    <DataTable
      columns={["Decision", "Transactions", "Fraud", "Legitimate", "Share of all fraud", "Fraud rate in band"]}
      rows={rows.map((r) => [
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-full" style={{ background: DECISION_COLOR[r.decision] }} aria-hidden="true" />
          {decisionLabel(r.decision)}
        </span>,
        int(r.transactions),
        int(r.fraud),
        int(r.legitimate),
        pct(r.share_of_all_fraud, 1),
        r.fraud_rate_in_band == null ? "—" : pct(r.fraud_rate_in_band, 2),
      ])}
    />
  );
}

function TestResults({ s }: { s: EvaluationSummary }) {
  const t = s.metrics.test;
  const pr = s.curves.test_pr;
  const roc = s.curves.test_roc;
  const best = [...s.curves.fusion_weight].sort((a, b) => b.average_precision - a.average_precision)[0];
  return (
    <div className="space-y-6">
      <Panel>
        <PanelHeader
          eyebrow="Held-out test split · evaluated once"
          title="Component comparison"
          description={`${int(t.logistic_regression.support)} transactions, ${t.logistic_regression.positives} of them fraud. Accuracy is omitted on purpose: labelling everything legitimate would already score ${pct(1 - t.logistic_regression.positive_rate, 2)}.`}
        />
        <ComparisonTable m={t} />
      </Panel>
      <div className="grid gap-6 xl:grid-cols-3">
        <Panel>
          <ConfusionMatrix cm={t.logistic_regression.confusion_matrix} title="Logistic Regression" subtitle={`Threshold ${num(t.logistic_regression.threshold, 4)} (validation F1)`} />
        </Panel>
        <Panel>
          <ConfusionMatrix cm={t.isolation_forest.confusion_matrix} title="Isolation Forest" subtitle="Top anomaly percentile flag (label-free)" />
        </Panel>
        <Panel>
          <ConfusionMatrix cm={t.hybrid_high_risk.confusion_matrix} title="Hybrid · HIGH_RISK" subtitle={`Hybrid ≥ ${num(s.policy.high_risk_threshold, 4)}`} />
        </Panel>
      </div>
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <CurveOverlay
            title="Precision–recall (test)"
            subtitle={`PR-AUC: LR ${num(t.logistic_regression.average_precision, 3)} · IF ${num(t.isolation_forest.average_precision, 3)} · hybrid ${num(t.hybrid_high_risk.average_precision, 3)}`}
            series={[
              { name: "Logistic Regression", color: COMPONENT.logistic_regression, points: pr.logistic_regression },
              { name: "Isolation Forest", color: COMPONENT.isolation_forest, points: pr.isolation_forest },
              { name: "Hybrid", color: COMPONENT.hybrid, points: pr.hybrid, dashed: true },
            ]}
            x="recall"
            y="precision"
            xLabel="Recall"
            yLabel="Precision"
            step
            baseline={{ y: t.logistic_regression.positive_rate, label: "No skill" }}
          />
        </Panel>
        <Panel>
          <CurveOverlay
            title="ROC (test)"
            subtitle="ROC-AUC looks high for every model on imbalanced data; read it with the PR curve."
            series={[
              { name: "Logistic Regression", color: COMPONENT.logistic_regression, points: roc.logistic_regression },
              { name: "Isolation Forest", color: COMPONENT.isolation_forest, points: roc.isolation_forest },
              { name: "Hybrid", color: COMPONENT.hybrid, points: roc.hybrid, dashed: true },
            ]}
            x="fpr"
            y="tpr"
            xLabel="False positive rate"
            yLabel="True positive rate"
            diagonal
          />
        </Panel>
      </div>
      <Panel>
        <PanelHeader eyebrow="Three-level decisions (test)" title="What each decision band contains" />
        <DecisionTable rows={t.decisions} />
      </Panel>
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel>
          <LinesChart
            title="Fusion weight vs validation PR-AUC"
            subtitle={best ? `Best w = ${num(best.weight_lr, 2)} (PR-AUC ${num(best.average_precision, 4)}); chosen on validation, never on test` : undefined}
            data={s.curves.fusion_weight}
            x="weight_lr"
            xLabel="Weight on Logistic Regression (w)"
            series={[
              { key: "average_precision", name: "PR-AUC", color: COMPONENT.hybrid, dot: true },
              { key: "roc_auc", name: "ROC-AUC", color: COMPONENT.logistic_regression, dashed: true },
            ]}
            yDomain={[0, 1]}
            xFormat={(v) => num(v, 2)}
          />
          <p className="mt-2 text-xs text-ink-3">w = 0 uses only the Isolation Forest; w = 1 uses only Logistic Regression.</p>
        </Panel>
        <Panel>
          <ReliabilityChart
            title="Logistic Regression calibration"
            series={[
              { name: "Test (prior-corrected)", result: s.calibration.test },
              { name: "Validation (prior-corrected)", result: s.calibration.validation },
              { name: "Validation (no correction)", result: s.calibration.validation_without_correction },
            ]}
          />
          <p className="mt-2 text-xs text-ink-3">
            Brier score test {num(s.calibration.test.brier, 5)} · ECE test {num(s.calibration.test.ece, 5)} · logit offset {num(s.calibration.logit_offset, 3)} ({s.calibration.method}).
          </p>
        </Panel>
      </div>
    </div>
  );
}

function initial(p: Policy): ExploreRequest {
  return {
    weight: p.fusion_weight_lr,
    review_threshold: p.review_threshold,
    high_risk_threshold: p.high_risk_threshold,
    lr_threshold: p.lr_threshold,
    if_flag_rate: Math.round((1 - p.if_flag_percentile) * 1e6) / 1e6,
    if_review_percentile: p.if_review_percentile,
  };
}

function Explorer({ s }: { s: EvaluationSummary }) {
  const [req, setReq] = useState<ExploreRequest>(() => initial(s.policy));
  const [sent, setSent] = useState<ExploreRequest>(req);
  useEffect(() => {
    const t = window.setTimeout(() => setSent(req), 400);
    return () => window.clearTimeout(t);
  }, [req]);
  const q = useApi((sig) => api.evaluationExplore(sent, { signal: sig }), [JSON.stringify(sent)]);
  const set = (k: keyof ExploreRequest, v: number | null) =>
    setReq((r) => {
      const next = { ...r, [k]: v };
      if (k === "review_threshold" && (v as number) > next.high_risk_threshold) next.high_risk_threshold = v as number;
      if (k === "high_risk_threshold" && (v as number) < next.review_threshold) next.review_threshold = v as number;
      return next;
    });
  const d = q.data;
  const changed = JSON.stringify(req) !== JSON.stringify(initial(s.policy));

  return (
    <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-4 xl:sticky xl:top-24 xl:self-start">
        <Panel>
          <h3 className="flex items-center gap-2 text-base font-semibold text-ink">
            <FlaskConical className="size-4 text-cyan" aria-hidden="true" /> What-if policy
          </h3>
          <p className="mt-1 text-xs text-ink-3">Scored on the validation split ({int(s.splits.validation.rows)} rows, {s.splits.validation.fraud_rows} fraud). The test split is never re-scored.</p>
          <div className="mt-5 space-y-5">
            <Slider label="Fusion weight on LR (w)" value={req.weight} min={0} max={1} step={0.05} onChange={(v) => set("weight", v)} format={(v) => v.toFixed(2)} />
            <Slider label="REVIEW threshold" value={req.review_threshold} min={0} max={1} step={0.005} onChange={(v) => set("review_threshold", v)} format={(v) => v.toFixed(3)} />
            <Slider label="HIGH_RISK threshold" value={req.high_risk_threshold} min={0} max={1} step={0.005} onChange={(v) => set("high_risk_threshold", v)} format={(v) => v.toFixed(3)} />
            <Slider label="LR threshold (LR-only flag)" value={req.lr_threshold} min={0.01} max={0.999} step={0.001} onChange={(v) => set("lr_threshold", v)} format={(v) => v.toFixed(3)} />
            <Slider label="Isolation Forest flag rate" value={req.if_flag_rate} min={0.001} max={0.1} step={0.001} onChange={(v) => set("if_flag_rate", v)} format={(v) => pct(v, 1)} />
            <Toggle
              label="Anomaly review trigger"
              checked={req.if_review_percentile != null}
              onChange={(on) => set("if_review_percentile", on ? 0.999 : null)}
              hint="Also send extreme anomalies to REVIEW"
            />
            {req.if_review_percentile != null && (
              <Slider label="Trigger percentile" value={req.if_review_percentile} min={0.99} max={0.9999} step={0.0001} onChange={(v) => set("if_review_percentile", v)} format={(v) => pct(v, 2)} />
            )}
          </div>
          <Button className="mt-5 w-full" variant="secondary" size="sm" disabled={!changed} onClick={() => setReq(initial(s.policy))} icon={<RotateCcw className="size-3.5" aria-hidden="true" />}>
            Reset to the active policy
          </Button>
          <p className="mt-2 text-[11px] text-ink-3">Changing these sliders does not change the deployed model. To deploy different thresholds, retrain with new settings on the Models page.</p>
        </Panel>
      </div>
      <div className="min-w-0 space-y-6" aria-busy={q.loading || q.refreshing}>
        {q.error && <ErrorState error={q.error} onRetry={q.refetch} title="The what-if evaluation failed" />}
        {!d ? (
          <LoadingBlock rows={8} label="Scoring the validation split" />
        ) : (
          <div className={q.refreshing ? "space-y-6 opacity-70 transition-opacity" : "space-y-6 transition-opacity"}>
            <Panel>
              <PanelHeader
                eyebrow="Validation split"
                title="Component comparison under this policy"
                actions={changed ? <StatusPill tone="warning">what-if</StatusPill> : <StatusPill tone="good">active policy</StatusPill>}
              />
              <MetricGrid
                items={[
                  { label: "HIGH_RISK precision", value: num(d.metrics.hybrid_high_risk.precision, 3) },
                  { label: "HIGH_RISK recall", value: num(d.metrics.hybrid_high_risk.recall, 3) },
                  { label: "REVIEW+ recall", value: num(d.metrics.hybrid_review_or_high.recall, 3) },
                  { label: "Review workload", value: pct(d.metrics.hybrid_review_or_high.alert_rate, 2), hint: `${int(d.metrics.hybrid_review_or_high.alerts)} transactions` },
                  { label: "Hybrid PR-AUC", value: num(d.metrics.hybrid_high_risk.average_precision, 4) },
                ]}
              />
              <div className="mt-5">
                <ComparisonTable m={d.metrics} />
              </div>
            </Panel>
            <Panel>
              <PanelHeader eyebrow="Decision bands (validation)" title="Fraud captured by each band" />
              <DecisionTable rows={d.metrics.decisions} />
            </Panel>
            <div className="grid gap-6 2xl:grid-cols-2">
              <Panel>
                <CurveOverlay
                  title="Precision–recall at this weight (validation)"
                  series={[
                    { name: "Logistic Regression", color: COMPONENT.logistic_regression, points: d.curves.pr.logistic_regression },
                    { name: "Isolation Forest", color: COMPONENT.isolation_forest, points: d.curves.pr.isolation_forest },
                    { name: `Hybrid (w = ${num(d.policy.fusion_weight_lr, 2)})`, color: COMPONENT.hybrid, points: d.curves.pr.hybrid, dashed: true },
                  ]}
                  x="recall"
                  y="precision"
                  xLabel="Recall"
                  yLabel="Precision"
                  step
                />
              </Panel>
              <Panel>
                <ThresholdSweep data={d.curves.hybrid_threshold_sweep} threshold={d.policy.high_risk_threshold} title="Hybrid threshold sweep (validation)" />
              </Panel>
            </div>
            <p className="text-xs text-ink-3">{d.note}</p>
          </div>
        )}
      </div>
    </div>
  );
}

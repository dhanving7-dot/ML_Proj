import type { CalibrationResult, ClassMetrics, CurvePointPR, CurvePointROC, SweepPoint } from "../../api/types";
import { ConfusionMatrix, MetricGrid, PrChart, ReliabilityChart, RocChart, ThresholdSweep } from "../../components/charts/Common";
import { SERIES } from "../../components/charts/theme";
import { SelectField, Slider } from "../../components/ui/Fields";
import { StatusPill } from "../../components/ui/Status";
import { int, modelLabel, num, pct } from "../../lib/format";
import type { ModuleProps } from "../Lab";
import { BarsChart, Card, DataTable, ExperimentShell, LinesChart, Note, useExperiment } from "./kit";
import { ModuleLayout } from "./ModuleLayout";

export function M5(props: ModuleProps) {
  return (
    <ModuleLayout
      id="m5"
      {...props}
      intro={
        <>
          Evaluation done properly on a 0.17 % positive rate: model selection uses cross-validation on the training split, calibration and thresholds are tuned on validation data, and
          the held-out test split is never touched here. Scores are PR-AUC (average precision) and ROC-AUC, because accuracy rewards a model that never flags anything.
        </>
      }
      experiments={[
        { key: "metrics", label: "Metrics & curves", Component: MetricsExp },
        { key: "cross-validation", label: "K-fold CV", Component: CvExp },
        { key: "nested-cv", label: "Nested CV", Component: NestedExp },
        { key: "calibration", label: "Calibration", Component: CalibExp },
        { key: "search", label: "Hyperparameter search", Component: SearchExp },
        { key: "learning-curves", label: "Learning curves", Component: LearnExp },
        { key: "statistical-tests", label: "Statistical tests", Component: StatsExp },
      ]}
    />
  );
}

const MODELS5 = ["logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting", "adaboost"];

/* ------------------------------------------------------------------ metrics */

interface MetRes {
  model: string;
  label: string;
  metrics: ClassMetrics;
  roc: CurvePointROC[];
  pr: CurvePointPR[];
  no_skill_precision: number;
  threshold_sweep: SweepPoint[];
  best_f1_threshold: number;
  accuracy_baseline: number;
  note: string;
}

function MetricsExp() {
  const exp = useExperiment<MetRes>("m5", "metrics", { model: "logistic_regression", threshold: 0.5 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Classification metrics on rare fraud"
      topics={["Accuracy paradox", "Precision", "Recall", "F1", "Confusion matrix", "ROC-AUC", "PR curve", "Average precision"]}
      estimate="≈ 1–5 s"
      controls={
        <>
          <SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={MODELS5.map((m) => ({ value: m, label: modelLabel(m) }))} />
          <Slider label="Decision threshold" value={Number(exp.params.threshold)} min={0.05} max={0.95} step={0.05} onChange={(v) => exp.setParam("threshold", v)} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const m = r.metrics;
        return (
          <>
            <Card title={`${r.label} on the validation split at threshold ${num(m.threshold, 2)}`}>
              <MetricGrid
                items={[
                  { label: "Precision", value: num(m.precision, 4), hint: "TP / (TP + FP)" },
                  { label: "Recall", value: num(m.recall, 4), hint: "TP / (TP + FN)" },
                  { label: "F1", value: num(m.f1, 4), hint: `best F1 at t=${num(r.best_f1_threshold, 3)}` },
                  { label: "PR-AUC", value: num(m.average_precision, 4), hint: `no skill ${num(r.no_skill_precision, 4)}` },
                  { label: "ROC-AUC", value: num(m.roc_auc, 4) },
                  { label: "False positives", value: int(m.false_positives), hint: `${int(m.false_negatives)} false negatives` },
                  { label: "Accuracy", value: pct(m.accuracy, 2), hint: `"always legitimate" scores ${pct(r.accuracy_baseline, 2)}` },
                  { label: "Log loss", value: num(m.log_loss, 4) },
                ]}
              />
              <div className="mt-4">
                <Note tone="warn">{r.note}</Note>
              </div>
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <ConfusionMatrix cm={m.confusion_matrix} subtitle={`${m.support.toLocaleString()} validation transactions (${m.positives} fraud); move the threshold to trade false positives for false negatives`} />
              </Card>
              <Card>
                <ThresholdSweep data={r.threshold_sweep} threshold={m.threshold} />
              </Card>
            </div>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <RocChart points={r.roc} auc={m.roc_auc} />
              </Card>
              <Card>
                <PrChart points={r.pr} ap={m.average_precision} baseline={r.no_skill_precision} />
              </Card>
            </div>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ CV */

interface CvFold {
  fold: number;
  validation_rows: number;
  fraud_rows: number;
  positive_rate: number;
  average_precision: number | null;
  roc_auc: number | null;
}
interface CvSide {
  folds: CvFold[];
  average_precision_mean: number | null;
  average_precision_std: number | null;
  fraud_count_std: number;
  positive_rate_std: number;
  undefined_folds: number;
}
interface CvRes {
  model: string;
  k: number;
  overall_positive_rate: number;
  kfold: CvSide;
  stratified_kfold: CvSide;
  note: string;
}

function CvExp() {
  const exp = useExperiment<CvRes>("m5", "cross-validation", { k: 5, model: "logistic_regression", sample_size: 20000 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="K-fold vs. stratified K-fold"
      topics={["K-fold", "Stratified K-fold", "Rare positives"]}
      estimate="≈ 1–5 s"
      controls={
        <>
          <Slider label="Folds (k)" value={Number(exp.params.k)} min={2} max={10} step={1} onChange={(v) => exp.setParam("k", v)} />
          <SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={["logistic_regression", "decision_tree", "random_forest"].map((m) => ({ value: m, label: modelLabel(m) }))} />
          <SelectField
            label="Training sample"
            value={String(exp.params.sample_size)}
            onChange={(v) => exp.setParam("sample_size", Number(v))}
            options={[5000, 20000, 60000].map((v) => ({ value: String(v), label: v.toLocaleString() }))}
            hint="Smaller samples have fewer frauds per fold"
          />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const rows = r.kfold.folds.map((f, i) => ({
          fold: `Fold ${f.fold}`,
          kfold: f.fraud_rows,
          stratified: r.stratified_kfold.folds[i]?.fraud_rows,
          kfold_ap: f.average_precision,
          stratified_ap: r.stratified_kfold.folds[i]?.average_precision,
        }));
        return (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <BarsChart
                  title="Fraud rows inside each validation fold"
                  subtitle={`Overall rate ${pct(r.overall_positive_rate, 3)}. Std of fraud count: K-fold ${num(r.kfold.fraud_count_std, 2)}, stratified ${num(r.stratified_kfold.fraud_count_std, 2)}`}
                  data={rows}
                  x="fold"
                  series={[
                    { key: "kfold", name: "K-fold" },
                    { key: "stratified", name: "Stratified" },
                  ]}
                  yFormat={(v) => String(Math.round(v))}
                />
              </Card>
              <Card>
                <BarsChart
                  title="PR-AUC per fold"
                  subtitle={`Mean ± std: K-fold ${num(r.kfold.average_precision_mean, 3)} ± ${num(r.kfold.average_precision_std, 3)}, stratified ${num(r.stratified_kfold.average_precision_mean, 3)} ± ${num(r.stratified_kfold.average_precision_std, 3)}`}
                  data={rows}
                  x="fold"
                  series={[
                    { key: "kfold_ap", name: "K-fold" },
                    { key: "stratified_ap", name: "Stratified" },
                  ]}
                  yFormat={(v) => num(v, 2)}
                  yDomain={[0, 1]}
                />
              </Card>
            </div>
            {r.kfold.undefined_folds > 0 && <Note tone="warn">{r.kfold.undefined_folds} plain K-fold fold(s) contained no fraud at all, so PR-AUC is undefined there.</Note>}
            <Note>{r.note}</Note>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ nested */

interface NestedRes {
  grid: Record<string, (number | string)[]>;
  non_nested: { best_score: number; best_params: Record<string, unknown>; explanation: string };
  nested: { outer_folds: { outer_fold: number; outer_average_precision: number; inner_best_average_precision: number; best_params: Record<string, unknown> }[]; mean: number; std: number };
  optimism_gap: number;
  seconds: number;
}

function NestedExp() {
  const exp = useExperiment<NestedRes>("m5", "nested-cv", { outer_k: 5, inner_k: 3, n_legit: 8000 }, { auto: false });
  return (
    <ExperimentShell
      exp={exp}
      title="Nested cross-validation"
      topics={["Nested CV", "Selection bias", "Grid search"]}
      estimate="≈ 10–20 s"
      controls={
        <>
          <Slider label="Outer folds" value={Number(exp.params.outer_k)} min={3} max={10} step={1} onChange={(v) => exp.setParam("outer_k", v)} />
          <Slider label="Inner folds" value={Number(exp.params.inner_k)} min={2} max={5} step={1} onChange={(v) => exp.setParam("inner_k", v)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        return (
          <>
            <Card title="Decision-tree grid (PR-AUC)" description={`Grid: ${Object.entries(r.grid).map(([k, v]) => `${k} ∈ {${v.join(", ")}}`).join("; ")}. Computed in ${num(r.seconds, 1)} s.`}>
              <MetricGrid
                items={[
                  { label: "Non-nested best", value: num(r.non_nested.best_score, 4), hint: "same folds select and score" },
                  { label: "Nested estimate", value: num(r.nested.mean, 4), hint: `± ${num(r.nested.std, 4)}` },
                  { label: "Optimism gap", value: num(r.optimism_gap, 4), hint: "non-nested − nested" },
                ]}
              />
              <div className="mt-4">
                <Note>
                  {r.non_nested.explanation} The gap measured on this data is {Math.abs(r.optimism_gap) < 0.005 ? "small" : "noticeable"}; with few frauds per fold, fold-to-fold noise is also large.
                </Note>
              </div>
            </Card>
            <Card title="Outer folds: each picks its own hyperparameters on inner folds">
              <DataTable
                columns={["Outer fold", "Inner best PR-AUC", "Outer PR-AUC", "Chosen parameters"]}
                numericFrom={1}
                rows={r.nested.outer_folds.map((f) => [f.outer_fold, num(f.inner_best_average_precision, 4), num(f.outer_average_precision, 4), JSON.stringify(f.best_params)])}
              />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ calibration */

interface CalMethod {
  label: string;
  brier: number;
  ece: number;
  mean_predicted: number;
  observed_rate: number;
  log_loss: number;
  average_precision: number;
  reliability: CalibrationResult["points"];
  histogram: CalibrationResult["histogram"];
}
interface CalRes {
  label: string;
  methods: Record<"uncalibrated" | "sigmoid" | "isotonic", CalMethod>;
  note: string;
}

function CalibExp() {
  const exp = useExperiment<CalRes>("m5", "calibration", { model: "random_forest" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Probability calibration"
      topics={["Platt scaling", "Isotonic regression", "Reliability diagram", "Brier score", "ECE"]}
      estimate="≈ 2–6 s"
      controls={
        <SelectField
          label="Base model"
          value={String(exp.params.model)}
          onChange={(v) => exp.setParam("model", v)}
          options={["random_forest", "adaboost", "decision_tree", "logistic_regression", "hist_gradient_boosting"].map((m) => ({ value: m, label: modelLabel(m) }))}
        />
      }
    >
      {(d) => {
        const r = d.result;
        const keys = ["uncalibrated", "sigmoid", "isotonic"] as const;
        return (
          <>
            <Card>
              <ReliabilityChart series={keys.map((k) => ({ name: r.methods[k].label, result: { points: r.methods[k].reliability, histogram: r.methods[k].histogram, ece: r.methods[k].ece, brier: r.methods[k].brier } }))} />
            </Card>
            <Card title={`${r.label}: calibration metrics on held-back validation rows`} description={r.note}>
              <DataTable
                columns={["Method", "Mean predicted", "Observed rate", "Brier", "ECE", "Log loss", "PR-AUC"]}
                rows={keys.map((k) => [
                  r.methods[k].label,
                  pct(r.methods[k].mean_predicted, 3),
                  pct(r.methods[k].observed_rate, 3),
                  num(r.methods[k].brier, 5),
                  num(r.methods[k].ece, 5),
                  num(r.methods[k].log_loss, 4),
                  num(r.methods[k].average_precision, 4),
                ])}
              />
            </Card>
            <Card>
              <BarsChart
                title="Where the uncalibrated probabilities fall"
                data={r.methods.uncalibrated.histogram.map((h) => ({ bin: num(h.bin_start, 1), count: h.count }))}
                x="bin"
                series={[{ key: "count", name: "Rows" }]}
                yFormat={(v) => String(Math.round(v))}
              />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ search */

interface SearchOne {
  evaluations: number;
  best_cv_average_precision: number;
  best_params: Record<string, unknown>;
  validation_average_precision: number;
  seconds: number;
  trace: { iteration: number; score: number; best_so_far: number }[];
  error?: string;
}
interface SearchRes {
  budget: number;
  grid_size: number;
  space: { grid: Record<string, unknown[]>; random_and_bayesian: string };
  results: Record<"grid" | "random" | "bayesian", SearchOne>;
  note: string;
}

function SearchExp() {
  const exp = useExperiment<SearchRes>("m5", "search", { model: "hist_gradient_boosting", budget: 12 }, { auto: false });
  return (
    <ExperimentShell
      exp={exp}
      title="Grid vs. random vs. Bayesian search"
      topics={["Grid search", "Random search", "Bayesian optimisation (TPE)"]}
      estimate="≈ 20–40 s"
      controls={
        <>
          <SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={["hist_gradient_boosting", "random_forest"].map((m) => ({ value: m, label: modelLabel(m) }))} />
          <Slider label="Budget (random & Bayesian trials)" value={Number(exp.params.budget)} min={4} max={30} step={1} onChange={(v) => exp.setParam("budget", v)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const names = { grid: "Grid", random: "Random", bayesian: "Bayesian (Optuna TPE)" } as const;
        const keys = (["grid", "random", "bayesian"] as const).filter((k) => !r.results[k].error);
        const n = Math.max(...keys.map((k) => r.results[k].trace.length));
        const traces = Array.from({ length: n }, (_, i) => ({ iteration: i + 1, ...Object.fromEntries(keys.map((k) => [k, r.results[k].trace[i]?.best_so_far ?? null])) }));
        return (
          <>
            <Card>
              <LinesChart
                title="Best cross-validated PR-AUC found so far"
                subtitle={`Grid size ${r.grid_size}; random and Bayesian use ${r.budget} trials each (3-fold stratified CV)`}
                data={traces}
                x="iteration"
                xLabel="Evaluation"
                series={keys.map((k, i) => ({ key: k, name: names[k], color: SERIES[i], type: "stepAfter" as const }))}
                yFormat={(v) => num(v, 4)}
              />
            </Card>
            <Card title="Winners, re-scored on the validation split">
              <DataTable
                columns={["Strategy", "Evaluations", "Best CV PR-AUC", "Validation PR-AUC", "Seconds", "Best parameters"]}
                rows={keys.map((k) => [names[k], r.results[k].evaluations, num(r.results[k].best_cv_average_precision, 4), num(r.results[k].validation_average_precision, 4), num(r.results[k].seconds, 1), JSON.stringify(r.results[k].best_params)])}
              />
              <div className="mt-4">
                <Note>{r.note} No strategy is guaranteed to win on every run; compare the numbers above rather than assuming Bayesian search is best.</Note>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ learning curves */

interface LearnRes {
  label: string;
  learning_curve: { train_size: number; train_mean: number; train_std: number; validation_mean: number; validation_std: number }[];
  validation_curve: { parameter: string; points: { value: number; train_mean: number; validation_mean: number; gap: number; diagnosis: string }[] };
  rules: string;
}

function LearnExp() {
  const exp = useExperiment<LearnRes>("m5", "learning-curves", { model: "decision_tree" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Learning and validation curves"
      topics={["Learning curve", "Validation curve", "Underfitting", "Overfitting"]}
      estimate="≈ 3–10 s"
      controls={<SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={["decision_tree", "logistic_regression", "random_forest"].map((m) => ({ value: m, label: modelLabel(m) }))} />}
    >
      {(d) => {
        const r = d.result;
        const lc = r.learning_curve.map((p) => ({ ...p, vl: p.validation_mean - p.validation_std, vu: p.validation_mean + p.validation_std }));
        const vc = r.validation_curve;
        const isLog = vc.parameter === "C";
        const vcData = vc.points.map((p) => ({ ...p, x: isLog ? Math.log10(p.value) : p.value }));
        return (
          <>
            <Card>
              <LinesChart
                title={`${r.label}: learning curve`}
                subtitle="PR-AUC as the training set grows (5-fold stratified CV on an enriched sample; band = ±1 std of validation)"
                data={lc}
                x="train_size"
                xLabel="Training rows"
                xFormat={(v) => v.toLocaleString()}
                series={[
                  { key: "train_mean", name: "Train", dot: true },
                  { key: "validation_mean", name: "Validation", dot: true },
                ]}
                band={{ lower: "vl", upper: "vu", color: SERIES[1], name: "±1 std" }}
              />
            </Card>
            <Card>
              <LinesChart
                title={`Validation curve over ${vc.parameter}`}
                subtitle={r.rules}
                data={vcData}
                x="x"
                xLabel={isLog ? `log₁₀ ${vc.parameter}` : vc.parameter}
                xFormat={(v) => (isLog ? v.toFixed(1) : String(v))}
                series={[
                  { key: "train_mean", name: "Train", dot: true },
                  { key: "validation_mean", name: "Validation", dot: true },
                ]}
              />
              <div className="mt-4 flex flex-wrap gap-2">
                {vc.points.map((p) => (
                  <StatusPill key={p.value} tone={p.diagnosis === "good fit" ? "good" : "warning"}>
                    {vc.parameter}={p.value}: {p.diagnosis}
                  </StatusPill>
                ))}
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ statistical tests */

interface StatsRes {
  classification: {
    label_a: string;
    label_b: string;
    threshold_a: number;
    threshold_b: number;
    test_half_rows: number;
    test_half_frauds: number;
    metrics_a: ClassMetrics;
    metrics_b: ClassMetrics;
    mcnemar: { table: Record<string, number>; statistic: number; p_value: number; method: string; significant_at_0_05: boolean };
  };
  ranking: {
    average_precision_a: number;
    average_precision_b: number;
    roc_auc_a: number;
    roc_auc_b: number;
    paired_bootstrap: { observed_difference: number; ci_95: [number, number]; n_boot: number; significant: boolean; histogram: { start: number; count: number }[] };
    interpretation: string;
    score_note: string;
  };
}

function StatsExp() {
  const exp = useExperiment<StatsRes>("m5", "statistical-tests", { model_a: "logistic_regression", model_b: "random_forest" }, { auto: true });
  return (
    <ExperimentShell
      exp={exp}
      title="Is the difference real?"
      topics={["McNemar's test", "Paired bootstrap", "Significance"]}
      estimate="≈ 10–20 s"
      controls={
        <>
          <SelectField label="Classifier A" value={String(exp.params.model_a)} onChange={(v) => exp.setParam("model_a", v)} options={MODELS5.map((m) => ({ value: m, label: modelLabel(m) }))} />
          <SelectField label="Classifier B" value={String(exp.params.model_b)} onChange={(v) => exp.setParam("model_b", v)} options={MODELS5.map((m) => ({ value: m, label: modelLabel(m) }))} />
        </>
      }
    >
      {(d) => {
        const c = d.result.classification;
        const mc = c.mcnemar;
        const g = d.result.ranking;
        const b = g.paired_bootstrap;
        return (
          <>
            <Card
              title={`McNemar's test: ${c.label_a} vs. ${c.label_b}`}
              description={`Thresholds were chosen by F1 on one half of the validation split; the test uses the other half (${int(c.test_half_rows)} rows, ${c.test_half_frauds} fraud). Precision ${num(c.metrics_a.precision, 3)} vs ${num(c.metrics_b.precision, 3)}; recall ${num(c.metrics_a.recall, 3)} vs ${num(c.metrics_b.recall, 3)}.`}
            >
              <div className="grid gap-6 lg:grid-cols-2">
                <DataTable
                  columns={["", `${c.label_b} correct`, `${c.label_b} wrong`]}
                  rows={[
                    [`${c.label_a} correct`, mc.table.both_correct.toLocaleString(), mc.table.a_only_correct.toLocaleString()],
                    [`${c.label_a} wrong`, mc.table.b_only_correct.toLocaleString(), mc.table.both_wrong.toLocaleString()],
                  ]}
                />
                <div className="space-y-3">
                  <MetricGrid
                    items={[
                      { label: "Statistic", value: num(mc.statistic, 3) },
                      { label: "p-value", value: mc.p_value < 0.0001 ? mc.p_value.toExponential(2) : num(mc.p_value, 4) },
                    ]}
                  />
                  <StatusPill tone={mc.significant_at_0_05 ? "warning" : "neutral"}>{mc.significant_at_0_05 ? "Significant difference at α = 0.05" : "No significant difference at α = 0.05"}</StatusPill>
                  <p className="text-xs text-ink-3">
                    Only the discordant cells ({mc.table.a_only_correct} vs {mc.table.b_only_correct}) matter. Method: {mc.method}.
                  </p>
                </div>
              </div>
            </Card>
            <Card
              title={`Paired bootstrap of PR-AUC: ${c.label_a} vs. ${c.label_b}`}
              description={`PR-AUC ${num(g.average_precision_a, 4)} vs ${num(g.average_precision_b, 4)}; ROC-AUC ${num(g.roc_auc_a, 4)} vs ${num(g.roc_auc_b, 4)}. ${g.interpretation} ${g.score_note}`}
            >
              <BarsChart
                title={`AP(A) − AP(B) over ${b.n_boot.toLocaleString()} stratified paired resamples`}
                subtitle={`Observed ${num(b.observed_difference, 4)} · 95% CI [${num(b.ci_95[0], 4)}, ${num(b.ci_95[1], 4)}] · ${b.significant ? "the interval excludes 0" : "the interval includes 0"}`}
                data={b.histogram.map((h) => ({ diff: num(h.start, 3), count: h.count }))}
                x="diff"
                series={[{ key: "count", name: "Resamples", color: SERIES[6] }]}
                yFormat={(v) => String(Math.round(v))}
              />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

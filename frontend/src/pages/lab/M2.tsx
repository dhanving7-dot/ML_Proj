import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { ClassMetrics, SweepPoint } from "../../api/types";
import { ChartFrame } from "../../components/charts/ChartFrame";
import { ChartTooltip } from "../../components/charts/ChartTooltip";
import { ConfusionMatrix, HBars, MetricGrid, ThresholdSweep } from "../../components/charts/Common";
import { axisProps, CHROME, CLASS, gridProps, SERIES } from "../../components/charts/theme";
import { SelectField, Slider } from "../../components/ui/Fields";
import { Chip } from "../../components/ui/Status";
import { int, num, pct, sci } from "../../lib/format";
import type { ModuleProps } from "../Lab";
import { BarsChart, Card, DataTable, ExperimentShell, LinesChart, logFmt, Note, useExperiment } from "./kit";
import { ModuleLayout } from "./ModuleLayout";

type RegMetrics = { rmse: number; mae: number; r2: number; mape: number | null; support: number };

export function M2(props: ModuleProps) {
  return (
    <ModuleLayout
      id="m2"
      {...props}
      intro={
        <>
          Linear models on the transactions. The dataset has one continuous column besides Time, so the regression experiments predict{" "}
          <strong className="text-ink">log(1 + Amount)</strong> from V1–V28; classification predicts <strong className="text-ink">Class</strong> (fraud). Preprocessing is always
          fitted on the training split and reused on validation data.
        </>
      }
      experiments={[
        { key: "linear-regression", label: "Linear regression", Component: LinearRegressionExp },
        { key: "regularization", label: "Regularisation", Component: RegularizationExp },
        { key: "logistic", label: "Logistic regression", Component: LogisticExp },
        { key: "preprocessing", label: "Preprocessing", Component: PreprocessingExp },
      ]}
    />
  );
}

/* ------------------------------------------------------------------ linear regression */

interface LinRes {
  target: string;
  coefficients: { feature: string; normal_equation: number; sklearn: number; gradient_descent: number | null }[];
  intercept: Record<string, number | null>;
  agreement: { max_abs_diff_normal_eq_vs_sklearn: number; max_abs_diff_gd_vs_normal_eq: number | null; train_mse_normal_eq: number; train_mse_gd: number | null };
  gradient_descent: { learning_rate: number; iterations: number; diverged: boolean; loss_history: { iteration: number; train_mse: number | null }[]; duration_ms: number };
  normal_equation_ms: number;
  conditioning: { condition_number: number; smallest_eigenvalue: number; note: string };
  validation_metrics: RegMetrics;
  train_metrics: RegMetrics;
  predictions_sample: { actual: number; predicted: number }[];
}

const LR_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].map((v) => ({ value: String(v), label: String(v) }));

function LinearRegressionExp() {
  const exp = useExperiment<LinRes>("m2", "linear-regression", { learning_rate: 0.05, iterations: 400 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Normal equation vs. gradient descent"
      topics={["Linear regression", "Normal equation", "Gradient descent", "RMSE", "MAE", "R²"]}
      estimate="≈ 2–5 s"
      controls={
        <>
          <SelectField label="Learning rate (α)" value={String(exp.params.learning_rate)} onChange={(v) => exp.setParam("learning_rate", Number(v))} options={LR_OPTIONS} hint="Large α can diverge" />
          <Slider label="Iterations" value={Number(exp.params.iterations)} min={10} max={2000} step={10} onChange={(v) => exp.setParam("iterations", v)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const lim = [Math.min(...r.predictions_sample.map((p) => Math.min(p.actual, p.predicted))), Math.max(...r.predictions_sample.map((p) => Math.max(p.actual, p.predicted)))];
        return (
          <>
            <Card title={`Validation metrics (target: ${r.target})`}>
              <MetricGrid
                items={[
                  { label: "RMSE", value: num(r.validation_metrics.rmse, 3), hint: `train ${num(r.train_metrics.rmse, 3)}` },
                  { label: "MAE", value: num(r.validation_metrics.mae, 3) },
                  { label: "MAPE", value: r.validation_metrics.mape != null ? pct(r.validation_metrics.mape) : "n/a", hint: r.validation_metrics.mape == null ? "zero amounts" : undefined },
                  { label: "R²", value: num(r.validation_metrics.r2, 3), hint: `train ${num(r.train_metrics.r2, 3)}` },
                ]}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title="Gradient descent: training MSE by iteration"
                  subtitle={r.gradient_descent.diverged ? "Diverged: the learning rate is too large" : `Normal-equation optimum ${num(r.agreement.train_mse_normal_eq, 4)} (dashed)`}
                  data={r.gradient_descent.loss_history}
                  x="iteration"
                  xLabel="Iteration"
                  series={[{ key: "train_mse", name: "Train MSE" }]}
                  refY={[{ value: r.agreement.train_mse_normal_eq, label: "Normal equation" }]}
                  yFormat={(v) => num(v, 2)}
                />
              </Card>
              <Card>
                <ChartFrame
                  title="Predicted vs. actual log(1 + Amount) (validation sample)"
                  subtitle="Points on the dashed line are perfect predictions"
                  legend={[{ label: "Transactions", color: SERIES[0] }, { label: "y = x", color: CHROME.muted, shape: "dash" }]}
                  table={{ columns: ["Actual", "Predicted"], rows: r.predictions_sample.map((p) => [num(p.actual, 2), num(p.predicted, 2)]) }}
                  height={260}
                >
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
                      <CartesianGrid {...gridProps} vertical />
                      <XAxis type="number" dataKey="actual" name="Actual" domain={lim} {...axisProps} label={{ value: "Actual log(1 + Amount)", position: "insideBottom", offset: -10 }} tickFormatter={(v) => num(v, 0)} />
                      <YAxis type="number" dataKey="predicted" name="Predicted" domain={lim} {...axisProps} width={40} tickFormatter={(v) => num(v, 0)} />
                      <ZAxis range={[20, 20]} />
                      <ReferenceLine segment={[{ x: lim[0], y: lim[0] }, { x: lim[1], y: lim[1] }]} stroke={CHROME.muted} strokeDasharray="4 4" />
                      <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 2)} title={() => "Transaction"} />} />
                      <Scatter data={r.predictions_sample} fill={SERIES[0]} fillOpacity={0.6} name="Transaction" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartFrame>
              </Card>
            </div>
            <Card
              title="Coefficients from three solvers"
              description={`Max |normal equation − scikit-learn| = ${sci(r.agreement.max_abs_diff_normal_eq_vs_sklearn)}; max |gradient descent − normal equation| = ${r.agreement.max_abs_diff_gd_vs_normal_eq == null ? "n/a (diverged)" : sci(r.agreement.max_abs_diff_gd_vs_normal_eq)}. Inputs are standardised on the training split.`}
            >
              <DataTable
                columns={["Feature", "Normal equation", "scikit-learn", "Gradient descent"]}
                rows={[["(intercept)", num(r.intercept.normal_equation, 4), num(r.intercept.sklearn, 4), num(r.intercept.gradient_descent, 4)], ...r.coefficients.map((c) => [c.feature, num(c.normal_equation, 4), num(c.sklearn, 4), num(c.gradient_descent, 4)])]}
              />
              <div className="mt-4">
                <Note>
                  Condition number of XᵀX/n: <strong className="text-ink">{num(r.conditioning.condition_number, 2)}</strong>. {r.conditioning.note} Normal equation solved in {num(r.normal_equation_ms, 2)} ms; {r.gradient_descent.iterations} gradient steps took{" "}
                  {num(r.gradient_descent.duration_ms, 1)} ms.
                </Note>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ regularization */

interface RegRes {
  summary: { model: string; zero_coefficients: number; l2_norm: number; l1_norm: number; train: RegMetrics; validation: RegMetrics }[];
  coefficients: ({ feature: string } & Record<string, number>)[];
  path: { ridge_alpha: number; lasso_alpha: number; ridge_val_rmse: number; lasso_val_rmse: number; lasso_nonzero: number; ridge: Record<string, number>; lasso: Record<string, number> }[];
  alpha_note: string;
}

const RIDGE = [1, 100, 1000, 10000, 100000, 1000000, 10000000].map((v) => ({ value: String(v), label: logFmt(v) }));
const LASSO = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 1].map((v) => ({ value: String(v), label: String(v) }));
const NAMES: Record<string, string> = { ols: "OLS", ridge: "Ridge (L2)", lasso: "Lasso (L1)", elastic_net: "Elastic Net" };

function RegularizationExp() {
  const exp = useExperiment<RegRes>("m2", "regularization", { ridge_alpha: 1000, lasso_alpha: 0.01, enet_alpha: 0.01, l1_ratio: 0.5 }, { auto: true });
  return (
    <ExperimentShell
      exp={exp}
      title="Ridge, Lasso and Elastic Net"
      topics={["L2 regularisation", "L1 regularisation", "Elastic Net", "Coefficient paths"]}
      estimate="≈ 2–4 s"
      controls={
        <>
          <SelectField label="Ridge α" value={String(exp.params.ridge_alpha)} onChange={(v) => exp.setParam("ridge_alpha", Number(v))} options={RIDGE} />
          <SelectField label="Lasso α" value={String(exp.params.lasso_alpha)} onChange={(v) => exp.setParam("lasso_alpha", Number(v))} options={LASSO} />
          <SelectField label="Elastic Net α" value={String(exp.params.enet_alpha)} onChange={(v) => exp.setParam("enet_alpha", Number(v))} options={LASSO} />
          <Slider label="Elastic Net L1 ratio" value={Number(exp.params.l1_ratio)} min={0} max={1} step={0.05} onChange={(v) => exp.setParam("l1_ratio", v)} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const top = [...r.coefficients].sort((a, b) => Math.abs(b.ols) - Math.abs(a.ols));
        const focus = top.slice(0, 4).map((c) => c.feature);
        const pathData = [...r.path]
          .sort((a, b) => a.lasso_alpha - b.lasso_alpha)
          .map((p) => ({ log_alpha: Math.log10(p.lasso_alpha), nonzero: p.lasso_nonzero, ...Object.fromEntries(focus.map((f) => [f, p.lasso[f]])) }));
        const rmse = [...r.path.map((p) => ({ log_alpha: Math.log10(p.ridge_alpha), ridge: p.ridge_val_rmse })), ...r.path.map((p) => ({ log_alpha: Math.log10(p.lasso_alpha), lasso: p.lasso_val_rmse }))].sort(
          (a, b) => a.log_alpha - b.log_alpha,
        );
        return (
          <>
            <Card title="Model comparison on the validation split (target: log(1 + Amount))">
              <DataTable
                columns={["Model", "Zero coefs", "‖w‖₁", "‖w‖₂", "Train RMSE", "Val RMSE", "Val R²"]}
                rows={r.summary.map((s) => [NAMES[s.model], s.zero_coefficients, num(s.l1_norm, 2), num(s.l2_norm, 2), num(s.train.rmse, 4), num(s.validation.rmse, 4), num(s.validation.r2, 4)])}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title="Lasso coefficient path"
                  subtitle="The four largest OLS coefficients; L1 drives coefficients exactly to zero as α grows"
                  data={pathData}
                  x="log_alpha"
                  xLabel="log₁₀ α (Lasso)"
                  xFormat={(v) => v.toFixed(1)}
                  series={focus.map((f, i) => ({ key: f, name: f, color: SERIES[i] }))}
                  refY={[{ value: 0 }]}
                />
              </Card>
              <Card>
                <LinesChart
                  title="Validation RMSE vs. penalty strength"
                  subtitle={r.alpha_note}
                  data={rmse}
                  x="log_alpha"
                  xLabel="log₁₀ α"
                  xFormat={(v) => v.toFixed(1)}
                  series={[
                    { key: "ridge", name: "Ridge" },
                    { key: "lasso", name: "Lasso" },
                  ]}
                />
              </Card>
            </div>
            <Card title="Coefficients by model (standardised V1–V28)">
              <DataTable
                columns={["Feature", "OLS", "Ridge", "Lasso", "Elastic Net"]}
                rows={top.map((c) => [
                  c.feature,
                  num(c.ols, 3),
                  num(c.ridge, 3),
                  c.lasso === 0 ? <span className="text-ink-3">0 (dropped)</span> : num(c.lasso, 3),
                  c.elastic_net === 0 ? <span className="text-ink-3">0 (dropped)</span> : num(c.elastic_net, 3),
                ])}
              />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ logistic */

interface LogRes {
  params: { C: number; regularization: string; threshold: number; class_weight: string };
  intercept: number;
  coefficients: { feature: string; coefficient: number; odds_ratio: number }[];
  zero_coefficients: number;
  sigmoid_curve: { z: number; sigmoid: number }[];
  sigmoid_points: { z: number; probability: number; label: number }[];
  sigmoid_points_note: string;
  binary_cross_entropy: Record<string, number>;
  class_weight_note: string;
  validation_metrics: ClassMetrics;
  threshold_sweep: SweepPoint[];
  multinomial: { target: string; band_edges: number[]; classes: string[]; validation_accuracy: number; validation_macro_f1: number; majority_baseline_accuracy: number; confusion_matrix: number[][]; top_coefficients: Record<string, { feature: string; coefficient: number }[]> };
}

const C_OPTIONS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100].map((v) => ({ value: String(v), label: String(v) }));

function LogisticExp() {
  const exp = useExperiment<LogRes>("m2", "logistic", { C_value: 1, regularization: "l2", threshold: 0.5, class_weight: "balanced" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Logistic regression for fraud"
      topics={["Sigmoid", "Binary cross-entropy", "Class weights", "Decision threshold", "L1 / L2", "Multinomial"]}
      estimate="≈ 2–5 s"
      controls={
        <>
          <SelectField label="Inverse regularisation C" value={String(exp.params.C_value)} onChange={(v) => exp.setParam("C_value", Number(v))} options={C_OPTIONS} hint="Smaller C = stronger penalty" />
          <SelectField
            label="Penalty"
            value={String(exp.params.regularization)}
            onChange={(v) => exp.setParam("regularization", v)}
            options={[
              { value: "l2", label: "L2 (ridge)" },
              { value: "l1", label: "L1 (lasso)" },
              { value: "none", label: "None" },
            ]}
          />
          <SelectField
            label="Class weights"
            value={String(exp.params.class_weight)}
            onChange={(v) => exp.setParam("class_weight", v)}
            options={[
              { value: "balanced", label: "Balanced (fraud ≈ 600× weight)" },
              { value: "none", label: "None (real 0.17 % prior)" },
            ]}
          />
          <Slider label="Decision threshold" value={Number(exp.params.threshold)} min={0.05} max={0.95} step={0.05} onChange={(v) => exp.setParam("threshold", v)} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const m = r.validation_metrics;
        return (
          <>
            <Card title={`Validation metrics at threshold ${num(r.params.threshold, 2)}`} description={r.class_weight_note}>
              <MetricGrid
                items={[
                  { label: "Precision", value: num(m.precision, 3) },
                  { label: "Recall", value: num(m.recall, 3) },
                  { label: "F1", value: num(m.f1, 3) },
                  { label: "PR-AUC", value: num(m.average_precision, 4), hint: "threshold-independent" },
                  { label: "ROC-AUC", value: num(m.roc_auc, 4) },
                  { label: "False positives", value: int(m.false_positives), hint: `${int(m.false_negatives)} false negatives` },
                  { label: "Accuracy", value: pct(m.accuracy, 2), hint: "misleading here" },
                  { label: "BCE (val)", value: num(r.binary_cross_entropy.validation_manual, 5), hint: `sklearn ${num(r.binary_cross_entropy.validation_sklearn, 5)}` },
                ]}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <ChartFrame
                  title="The sigmoid maps the linear score z to a probability"
                  subtitle={`σ(z) = 1 / (1 + e⁻ᶻ). ${r.sigmoid_points_note}`}
                  legend={[{ label: "σ(z)", color: CHROME.ink2, shape: "line" }, { label: "Fraud", color: CLASS.fraud }, { label: "Legitimate", color: CLASS.legitimate }]}
                  table={{ columns: ["z", "P(fraud)", "Actual"], rows: r.sigmoid_points.map((p) => [num(p.z, 2), num(p.probability, 4), p.label ? "Fraud" : "Legitimate"]) }}
                  height={280}
                >
                  <ResponsiveContainer>
                    <ComposedChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
                      <CartesianGrid {...gridProps} vertical />
                      <XAxis type="number" dataKey="z" domain={[-12, 12]} allowDataOverflow {...axisProps} label={{ value: "Linear score z = w·x + b", position: "insideBottom", offset: -10 }} />
                      <YAxis type="number" domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
                      <ReferenceLine y={r.params.threshold} stroke={CHROME.muted} strokeDasharray="4 4" label={{ value: `threshold ${r.params.threshold}`, position: "insideTopLeft", fill: CHROME.muted, fontSize: 10 }} />
                      <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 4)} title={(_, dd) => `z = ${num(dd?.z as number, 2)}`} />} />
                      <Scatter data={r.sigmoid_points.filter((p) => p.label === 0)} dataKey="probability" name="Legitimate" fill={CLASS.legitimate} fillOpacity={0.45} />
                      <Scatter data={r.sigmoid_points.filter((p) => p.label === 1)} dataKey="probability" name="Fraud" fill={CLASS.fraud} fillOpacity={0.8} />
                      <Line data={r.sigmoid_curve} dataKey="sigmoid" name="σ(z)" stroke={CHROME.ink2} strokeWidth={2} dot={false} isAnimationActive />
                    </ComposedChart>
                  </ResponsiveContainer>
                </ChartFrame>
              </Card>
              <Card>
                <ConfusionMatrix cm={m.confusion_matrix} subtitle={`Validation split, ${m.support.toLocaleString()} transactions (${m.positives} fraud)`} />
              </Card>
            </div>
            <Card>
              <ThresholdSweep data={r.threshold_sweep} threshold={r.params.threshold} title="Precision, recall and F1 as the threshold moves" />
            </Card>
            <Card>
              <HBars
                title="Largest coefficients (log-odds per standardised unit)"
                subtitle={`Intercept ${num(r.intercept, 3)}. Positive values raise P(fraud); odds ratio = e^coefficient. V-components have no published meaning.`}
                data={r.coefficients.slice(0, 15).map((c) => ({ label: c.feature, value: c.coefficient }))}
                diverging
                valueLabel="Coefficient"
              />
            </Card>
            <Card title="Multinomial logistic regression: amount band" description={`${r.multinomial.target}. Band edges: ${r.multinomial.band_edges.map((e) => num(e, 2)).join(" / ")}.`}>
              <div className="grid gap-6 lg:grid-cols-2">
                <ConfusionMatrix cm={r.multinomial.confusion_matrix} labels={r.multinomial.classes} title="3-class confusion matrix (validation)" />
                <div className="space-y-4">
                  <MetricGrid
                    items={[
                      { label: "Accuracy", value: pct(r.multinomial.validation_accuracy) },
                      { label: "Macro F1", value: num(r.multinomial.validation_macro_f1, 3) },
                      { label: "Majority baseline", value: pct(r.multinomial.majority_baseline_accuracy) },
                    ]}
                  />
                  <DataTable
                    columns={["Class", "Strongest features (coefficient)"]}
                    numericFrom={9}
                    rows={Object.entries(r.multinomial.top_coefficients).map(([cls, xs]) => [cls, xs.slice(0, 3).map((x) => `${x.feature} (${num(x.coefficient, 2)})`).join(", ")])}
                  />
                </div>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ preprocessing */

interface PreRes {
  scaling: {
    feature: string;
    standard: Record<string, number>;
    minmax: Record<string, number>;
    histograms: Partial<Record<"raw" | "standard" | "minmax" | "log1p", { x: number; count: number }[]>>;
    note: string;
  };
  encoding: {
    one_hot: { source: string; output_columns: string[] };
    ordinal: { source: string; edges: number[]; mapping: Record<string, number>; note: string };
    target_encoding_hour: { hour: number; rows: number; naive_fraud_rate: number; cross_fitted_full_fit: number }[];
    leakage_demo: { column: string; naive: { train_roc_auc: number; validation_roc_auc: number }; cross_fitted: { train_roc_auc: number; validation_roc_auc: number }; explanation: string };
  };
  imputation: {
    simulation: string;
    note: string;
    missingness: { feature: string; missing_rate_train: number; train_mean: number; train_median: number; fraud_rate_when_missing: number; fraud_rate_when_present: number }[];
    missing_indicator_columns: string[];
    downstream_comparison: { strategy: string; validation_average_precision: number; validation_roc_auc: number; fit_ms: number }[];
    comparison_model: string;
  };
}

const FEATURES = ["Amount", "Time", "V1", "V2", "V4", "V10", "V12", "V14", "V17"];

function PreprocessingExp() {
  const exp = useExperiment<PreRes>("m2", "preprocessing", { feature: "Amount", missing_rate: 0.1 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Scaling, encoding & imputation"
      topics={["StandardScaler", "MinMaxScaler", "log transform", "One-hot", "Ordinal", "Target encoding", "Cross-fitting", "Mean / median", "Model-based imputation", "MissingIndicator"]}
      estimate="≈ 5–10 s"
      controls={
        <>
          <SelectField label="Feature to scale" value={String(exp.params.feature)} onChange={(v) => exp.setParam("feature", v)} options={FEATURES.map((f) => ({ value: f, label: f }))} />
          <Slider label="Simulated missing rate" value={Number(exp.params.missing_rate)} min={0.01} max={0.5} step={0.01} onChange={(v) => exp.setParam("missing_rate", v)} format={(v) => pct(v, 0)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const s = r.scaling;
        const leak = r.encoding.leakage_demo;
        const kinds = (["raw", "log1p", "standard", "minmax"] as const).filter((k) => s.histograms[k]);
        const titles = { raw: "Raw values", log1p: "log(1 + x)", standard: "StandardScaler (z-score)", minmax: "MinMaxScaler [0, 1]" };
        return (
          <>
            <Card title={`Scaling ${s.feature} with parameters learned on the training split`} description={s.note}>
              <div className={kinds.length === 4 ? "grid gap-4 md:grid-cols-2 xl:grid-cols-4" : "grid gap-4 md:grid-cols-3"}>
                {kinds.map((k, i) => (
                  <BarsChart
                    key={k}
                    title={titles[k]}
                    subtitle={
                      k === "raw"
                        ? "Training distribution"
                        : k === "log1p"
                          ? "What the production pipeline applies to Amount"
                          : k === "standard"
                            ? `μ=${num(s.standard.mean_learned, 2)}, σ=${num(s.standard.std_learned, 2)} · val mean ${num(s.standard.validation_mean_after, 3)}`
                            : `min=${num(s.minmax.min_learned, 1)}, max=${num(s.minmax.max_learned, 1)} · val ${num(s.minmax.validation_min_after, 3)}–${num(s.minmax.validation_max_after, 3)}`
                    }
                    data={s.histograms[k]!}
                    x="x"
                    xFormat={(v) => num(Number(v), k === "minmax" ? 2 : 1)}
                    series={[{ key: "count", name: "Rows", color: SERIES[[0, 3, 6, 2][i]] }]}
                    yFormat={(v) => String(Math.round(v))}
                    height={180}
                  />
                ))}
              </div>
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Encoders on categories derived from the data" description="The dataset itself has no categorical columns, so hour-of-cycle (from Time) and an amount band are used.">
                <p className="text-xs text-ink-3">One-hot encoding of {r.encoding.one_hot.source} ({r.encoding.one_hot.output_columns.length} columns; unseen categories become all zeros):</p>
                <div className="mt-2 flex max-h-24 flex-wrap gap-1 overflow-y-auto">
                  {r.encoding.one_hot.output_columns.map((c) => (
                    <Chip key={c}>{c}</Chip>
                  ))}
                </div>
                <p className="mt-4 text-xs text-ink-3">
                  Ordinal encoding of {r.encoding.ordinal.source} (edges {r.encoding.ordinal.edges.map((e) => num(e, 2)).join(" / ")}). {r.encoding.ordinal.note}
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(r.encoding.ordinal.mapping).map(([k, v]) => (
                    <Chip key={k}>
                      {k} → {v}
                    </Chip>
                  ))}
                </div>
                <div className="mt-5">
                  <BarsChart
                    title="Target encoding of hour-of-cycle (fraud rate per hour)"
                    subtitle="Naive training mean vs the smoothed, cross-fitted encoding"
                    data={r.encoding.target_encoding_hour}
                    x="hour"
                    series={[
                      { key: "naive_fraud_rate", name: "Naive mean" },
                      { key: "cross_fitted_full_fit", name: "Cross-fitted" },
                    ]}
                    yFormat={(v) => pct(v, 2)}
                    height={220}
                  />
                </div>
              </Card>
              <Card title="Target-encoding leakage on an identifier" description={`${leak.column}. ${leak.explanation}`}>
                <BarsChart
                  title="ROC-AUC of the encoded column alone"
                  data={[
                    { variant: "Naive", train: leak.naive.train_roc_auc, validation: leak.naive.validation_roc_auc },
                    { variant: "Cross-fitted", train: leak.cross_fitted.train_roc_auc, validation: leak.cross_fitted.validation_roc_auc },
                  ]}
                  x="variant"
                  series={[
                    { key: "train", name: "Train" },
                    { key: "validation", name: "Validation" },
                  ]}
                  yDomain={[0, 1]}
                  refY={[{ value: 0.5, label: "chance" }]}
                  height={230}
                />
              </Card>
            </div>
            <Card
              title="Missing values and imputation (simulated missingness)"
              description={`${r.imputation.simulation} Downstream model: ${r.imputation.comparison_model}. MissingIndicator adds a 0/1 column for: ${r.imputation.missing_indicator_columns.join(", ")}.`}
            >
              <div className="grid gap-6 lg:grid-cols-2">
                <DataTable
                  columns={["Feature", "Missing", "Mean", "Median", "Fraud | missing", "Fraud | present"]}
                  rows={r.imputation.missingness.map((m) => [m.feature, pct(m.missing_rate_train), num(m.train_mean, 2), num(m.train_median, 2), pct(m.fraud_rate_when_missing, 2), pct(m.fraud_rate_when_present, 2)])}
                />
                <DataTable
                  columns={["Strategy", "Val PR-AUC", "Val ROC-AUC", "Fit ms"]}
                  rows={r.imputation.downstream_comparison.map((c) => [c.strategy, num(c.validation_average_precision, 4), num(c.validation_roc_auc, 4), num(c.fit_ms, 0)])}
                />
              </div>
              <div className="mt-4">
                <Note>{r.imputation.note}</Note>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

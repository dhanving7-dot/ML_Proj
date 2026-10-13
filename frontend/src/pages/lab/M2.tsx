import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { ClassMetrics, SweepPoint } from "../../api/types";
import { ChartFrame } from "../../components/charts/ChartFrame";
import { ChartTooltip } from "../../components/charts/ChartTooltip";
import { ConfusionMatrix, HBars, MetricGrid, ThresholdSweep } from "../../components/charts/Common";
import { axisProps, CHROME, gridProps, OUTCOME, SERIES } from "../../components/charts/theme";
import { SelectField, Slider } from "../../components/ui/Fields";
import { Chip } from "../../components/ui/Status";
import { num, pct, sci } from "../../lib/format";
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
          Linear models on the placement data. Regression experiments predict <strong className="text-ink">Salary Package</strong> for placed students (a continuous
          target); classification predicts placement. Preprocessing is always fitted on the training split and reused on validation data.
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
      topics={["Linear regression", "Normal equation", "Gradient descent", "RMSE", "MAE", "MAPE", "R²"]}
      estimate="< 1 s"
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
            <Card title="Validation metrics (placed students, target: salary in LPA)">
              <MetricGrid
                items={[
                  { label: "RMSE", value: num(r.validation_metrics.rmse, 3), hint: `train ${num(r.train_metrics.rmse, 3)}` },
                  { label: "MAE", value: num(r.validation_metrics.mae, 3) },
                  { label: "MAPE", value: r.validation_metrics.mape != null ? pct(r.validation_metrics.mape) : "n/a" },
                  { label: "R²", value: num(r.validation_metrics.r2, 3), hint: `train ${num(r.train_metrics.r2, 3)}` },
                ]}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title="Gradient descent: training MSE by iteration"
                  subtitle={r.gradient_descent.diverged ? "Diverged: the learning rate is too large" : `Normal-equation optimum ${num(r.agreement.train_mse_normal_eq, 3)} (dashed)`}
                  data={r.gradient_descent.loss_history}
                  x="iteration"
                  xLabel="Iteration"
                  series={[{ key: "train_mse", name: "Train MSE" }]}
                  refY={[{ value: r.agreement.train_mse_normal_eq, label: "Normal equation" }]}
                  yFormat={(v) => num(v, 1)}
                />
              </Card>
              <Card>
                <ChartFrame
                  title="Predicted vs. actual salary (validation sample)"
                  subtitle="Points on the dashed line are perfect predictions"
                  legend={[{ label: "Students", color: SERIES[0] }, { label: "y = x", color: CHROME.muted, shape: "dash" }]}
                  table={{ columns: ["Actual", "Predicted"], rows: r.predictions_sample.map((p) => [num(p.actual, 2), num(p.predicted, 2)]) }}
                  height={260}
                >
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
                      <CartesianGrid {...gridProps} vertical />
                      <XAxis type="number" dataKey="actual" name="Actual" domain={lim} {...axisProps} label={{ value: "Actual (LPA)", position: "insideBottom", offset: -10 }} tickFormatter={(v) => num(v, 0)} />
                      <YAxis type="number" dataKey="predicted" name="Predicted" domain={lim} {...axisProps} width={40} tickFormatter={(v) => num(v, 0)} />
                      <ZAxis range={[20, 20]} />
                      <ReferenceLine segment={[{ x: lim[0], y: lim[0] }, { x: lim[1], y: lim[1] }]} stroke={CHROME.muted} strokeDasharray="4 4" />
                      <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 2)} title={() => "Student"} />} />
                      <Scatter data={r.predictions_sample} fill={SERIES[0]} fillOpacity={0.6} name="Student" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartFrame>
              </Card>
            </div>
            <Card title="Coefficients from three solvers" description={`Max |normal equation − scikit-learn| = ${sci(r.agreement.max_abs_diff_normal_eq_vs_sklearn)}; max |gradient descent − normal equation| = ${r.agreement.max_abs_diff_gd_vs_normal_eq == null ? "n/a (diverged)" : num(r.agreement.max_abs_diff_gd_vs_normal_eq, 4)}. Features are standardised on the training split.`}>
              <DataTable
                columns={["Feature", "Normal equation", "scikit-learn", "Gradient descent"]}
                rows={[["(intercept)", num(r.intercept.normal_equation, 4), num(r.intercept.sklearn, 4), num(r.intercept.gradient_descent, 4)], ...r.coefficients.map((c) => [c.feature, num(c.normal_equation, 4), num(c.sklearn, 4), num(c.gradient_descent, 4)])]}
              />
              <div className="mt-4">
                <Note>
                  Condition number of XᵀX/n: <strong className="text-ink">{num(r.conditioning.condition_number, 1)}</strong>. {r.conditioning.note} Normal equation solved in {num(r.normal_equation_ms, 2)} ms; {r.gradient_descent.iterations} gradient steps took {num(r.gradient_descent.duration_ms, 1)} ms.
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

const RIDGE = [0.01, 1, 10, 100, 1000, 10000, 100000].map((v) => ({ value: String(v), label: logFmt(v) }));
const LASSO = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3].map((v) => ({ value: String(v), label: String(v) }));
const NAMES: Record<string, string> = { ols: "OLS", ridge: "Ridge (L2)", lasso: "Lasso (L1)", elastic_net: "Elastic Net" };

function RegularizationExp() {
  const exp = useExperiment<RegRes>("m2", "regularization", { ridge_alpha: 100, lasso_alpha: 0.05, enet_alpha: 0.05, l1_ratio: 0.5 }, { auto: true });
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
        const rmse = [
          ...r.path.map((p) => ({ log_alpha: Math.log10(p.ridge_alpha), ridge: p.ridge_val_rmse })),
          ...r.path.map((p) => ({ log_alpha: Math.log10(p.lasso_alpha), lasso: p.lasso_val_rmse })),
        ].sort((a, b) => a.log_alpha - b.log_alpha);
        return (
          <>
            <Card title="Model comparison on the validation split">
              <DataTable
                columns={["Model", "Zero coefs", "‖w‖₁", "‖w‖₂", "Train RMSE", "Val RMSE", "Val R²"]}
                rows={r.summary.map((s) => [NAMES[s.model], s.zero_coefficients, num(s.l1_norm, 2), num(s.l2_norm, 2), num(s.train.rmse, 3), num(s.validation.rmse, 3), num(s.validation.r2, 3)])}
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
            <Card title="Coefficients by model (standardised features)">
              <DataTable columns={["Feature", "OLS", "Ridge", "Lasso", "Elastic Net"]} rows={top.map((c) => [c.feature, num(c.ols, 3), num(c.ridge, 3), c.lasso === 0 ? <span className="text-ink-3">0 (dropped)</span> : num(c.lasso, 3), c.elastic_net === 0 ? <span className="text-ink-3">0 (dropped)</span> : num(c.elastic_net, 3)])} />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ logistic */

interface LogRes {
  params: { C: number; regularization: string; threshold: number };
  intercept: number;
  coefficients: { feature: string; coefficient: number; odds_ratio: number }[];
  zero_coefficients: number;
  sigmoid_curve: { z: number; sigmoid: number }[];
  sigmoid_points: { z: number; probability: number; label: number }[];
  binary_cross_entropy: Record<string, number>;
  validation_metrics: ClassMetrics;
  threshold_sweep: SweepPoint[];
  multinomial: { target: string; band_edges_lpa: number[]; classes: string[]; validation_accuracy: number; validation_macro_f1: number; majority_baseline_accuracy: number; confusion_matrix: number[][]; top_coefficients: Record<string, { feature: string; coefficient: number }[]> };
}

const C_OPTIONS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100].map((v) => ({ value: String(v), label: String(v) }));

function LogisticExp() {
  const exp = useExperiment<LogRes>("m2", "logistic", { C_value: 1, regularization: "l2", threshold: 0.5 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Logistic regression"
      topics={["Sigmoid", "Binary cross-entropy", "Decision threshold", "Probabilities", "L1 / L2", "Multinomial"]}
      estimate="≈ 1 s"
      controls={
        <>
          <SelectField label="Inverse regularisation C" value={String(exp.params.C_value)} onChange={(v) => exp.setParam("C_value", Number(v))} options={C_OPTIONS} hint="Smaller C = stronger penalty" />
          <SelectField label="Penalty" value={String(exp.params.regularization)} onChange={(v) => exp.setParam("regularization", v)} options={[{ value: "l2", label: "L2 (ridge)" }, { value: "l1", label: "L1 (lasso)" }, { value: "none", label: "None" }]} />
          <Slider label="Decision threshold" value={Number(exp.params.threshold)} min={0.05} max={0.95} step={0.05} onChange={(v) => exp.setParam("threshold", v)} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const m = r.validation_metrics;
        const pts = r.sigmoid_points.map((p) => ({ ...p, jitter: p.label === 1 ? p.probability : p.probability }));
        return (
          <>
            <Card title={`Validation metrics at threshold ${num(r.params.threshold, 2)}`}>
              <MetricGrid
                items={[
                  { label: "Accuracy", value: pct(m.accuracy) },
                  { label: "Precision", value: num(m.precision, 3) },
                  { label: "Recall", value: num(m.recall, 3) },
                  { label: "F1", value: num(m.f1, 3) },
                  { label: "ROC-AUC", value: num(m.roc_auc, 4), hint: "threshold-independent" },
                  { label: "BCE (val)", value: num(r.binary_cross_entropy.validation_manual, 4), hint: `sklearn ${num(r.binary_cross_entropy.validation_sklearn, 4)}` },
                  { label: "BCE (train)", value: num(r.binary_cross_entropy.train_manual, 4) },
                  { label: "Zero coefs", value: String(r.zero_coefficients), hint: "L1 sets some to 0" },
                ]}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <ChartFrame
                  title="The sigmoid maps the linear score z to a probability"
                  subtitle="σ(z) = 1 / (1 + e⁻ᶻ); dots are validation students coloured by their true outcome"
                  legend={[{ label: "σ(z)", color: CHROME.ink2, shape: "line" }, { label: "Actually placed", color: OUTCOME.placed }, { label: "Actually not placed", color: OUTCOME.notPlaced }]}
                  table={{ columns: ["z", "P(placed)", "Actual"], rows: r.sigmoid_points.map((p) => [num(p.z, 2), num(p.probability, 3), p.label ? "Placed" : "Not placed"]) }}
                  height={280}
                >
                  <ResponsiveContainer>
                    <ComposedChart margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
                      <CartesianGrid {...gridProps} vertical />
                      <XAxis type="number" dataKey="z" domain={[-10, 10]} allowDataOverflow {...axisProps} label={{ value: "Linear score z = w·x + b", position: "insideBottom", offset: -10 }} />
                      <YAxis type="number" domain={[0, 1]} {...axisProps} width={36} tickFormatter={(v) => num(v, 1)} />
                      <ReferenceLine y={r.params.threshold} stroke={CHROME.muted} strokeDasharray="4 4" label={{ value: `threshold ${r.params.threshold}`, position: "insideTopLeft", fill: CHROME.muted, fontSize: 10 }} />
                      <Tooltip content={<ChartTooltip valueFormat={(v) => num(v, 3)} title={(_, dd) => `z = ${num(dd?.z as number, 2)}`} />} />
                      <Scatter data={pts.filter((p) => p.label === 1)} dataKey="probability" name="Actually placed" fill={OUTCOME.placed} fillOpacity={0.55} />
                      <Scatter data={pts.filter((p) => p.label === 0)} dataKey="probability" name="Actually not placed" fill={OUTCOME.notPlaced} fillOpacity={0.6} />
                      <Line data={r.sigmoid_curve} dataKey="sigmoid" name="σ(z)" stroke={CHROME.ink2} strokeWidth={2} dot={false} isAnimationActive />
                    </ComposedChart>
                  </ResponsiveContainer>
                </ChartFrame>
              </Card>
              <Card>
                <ConfusionMatrix cm={m.confusion_matrix} subtitle={`Validation split, ${m.support.toLocaleString()} students`} />
              </Card>
            </div>
            <Card>
              <ThresholdSweep data={r.threshold_sweep} threshold={r.params.threshold} title="Precision, recall and F1 as the threshold moves" />
            </Card>
            <Card>
              <HBars
                title="Largest coefficients (log-odds per standardised unit)"
                subtitle={`Intercept ${num(r.intercept, 3)}. Positive values raise P(placed); odds ratio = e^coefficient.`}
                data={r.coefficients.slice(0, 15).map((c) => ({ label: c.feature, value: c.coefficient }))}
                diverging
                valueLabel="Coefficient"
              />
            </Card>
            <Card title="Multinomial logistic regression: salary band" description={`${r.multinomial.target}. Band edges: ${r.multinomial.band_edges_lpa.map((e) => `${e} LPA`).join(" / ")}.`}>
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
    histograms: Record<"raw" | "standard" | "minmax", { x: number; count: number }[]>;
    note: string;
  };
  encoding: {
    one_hot: { source: string; output_columns: string[] };
    ordinal: { source: string; mapping: Record<string, number>; note: string };
    target_encoding_city: { city: string; naive_mean: number; cross_fitted_full_fit: number }[];
    leakage_demo: { column: string; naive: { train_roc_auc: number; validation_roc_auc: number }; cross_fitted: { train_roc_auc: number; validation_roc_auc: number }; explanation: string };
  };
  imputation: {
    missingness: { feature: string; missing_rate_train: number; train_mean: number; train_median: number; placed_rate_when_missing: number; placed_rate_when_present: number }[];
    missing_indicator_columns: string[];
    downstream_comparison: { strategy: string; validation_roc_auc: number; validation_log_loss: number; fit_ms: number }[];
    comparison_model: string;
  };
}

const NUMERIC = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "SoftSkillsRating", "Internships", "Projects", "SGPA_Sem8"];

function PreprocessingExp() {
  const exp = useExperiment<PreRes>("m2", "preprocessing", { feature: "CodingTestScore" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Scaling, encoding & imputation"
      topics={["StandardScaler", "MinMaxScaler", "One-hot", "Ordinal", "Target encoding", "Cross-fitting", "Mean / median", "Model-based imputation", "MissingIndicator"]}
      estimate="≈ 1–2 s"
      controls={<SelectField label="Feature to scale" value={String(exp.params.feature)} onChange={(v) => exp.setParam("feature", v)} options={NUMERIC.map((f) => ({ value: f, label: f }))} />}
    >
      {(d) => {
        const r = d.result;
        const s = r.scaling;
        const leak = r.encoding.leakage_demo;
        return (
          <>
            <Card title={`Scaling ${s.feature} with parameters learned on the training split`} description={s.note}>
              <div className="grid gap-4 md:grid-cols-3">
                {(["raw", "standard", "minmax"] as const).map((k, i) => (
                  <BarsChart
                    key={k}
                    title={k === "raw" ? "Raw values" : k === "standard" ? "StandardScaler (z-score)" : "MinMaxScaler [0, 1]"}
                    subtitle={
                      k === "raw"
                        ? "Training distribution"
                        : k === "standard"
                          ? `μ=${num(s.standard.mean_learned, 2)}, σ=${num(s.standard.std_learned, 2)} · val mean ${num(s.standard.validation_mean_after, 3)}`
                          : `min=${num(s.minmax.min_learned, 1)}, max=${num(s.minmax.max_learned, 1)} · val range ${num(s.minmax.validation_min_after, 3)}–${num(s.minmax.validation_max_after, 3)}`
                    }
                    data={s.histograms[k]}
                    x="x"
                    xFormat={(v) => num(Number(v), k === "minmax" ? 2 : 1)}
                    series={[{ key: "count", name: "Rows", color: SERIES[i === 0 ? 0 : i === 1 ? 6 : 2] }]}
                    yFormat={(v) => String(Math.round(v))}
                    height={180}
                  />
                ))}
              </div>
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Categorical encoders">
                <p className="text-xs text-ink-3">One-hot encoding of {r.encoding.one_hot.source} ({r.encoding.one_hot.output_columns.length} columns; unseen categories become all zeros):</p>
                <div className="mt-2 flex flex-wrap gap-1">
                  {r.encoding.one_hot.output_columns.map((c) => (
                    <Chip key={c}>{c}</Chip>
                  ))}
                </div>
                <p className="mt-4 text-xs text-ink-3">Ordinal encoding of {r.encoding.ordinal.source}. {r.encoding.ordinal.note}</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(r.encoding.ordinal.mapping).map(([k, v]) => (
                    <Chip key={k}>
                      {k} → {v}
                    </Chip>
                  ))}
                </div>
                <div className="mt-5">
                  <DataTable columns={["City", "Naive mean", "Cross-fitted (full fit)"]} rows={r.encoding.target_encoding_city.map((c) => [c.city, num(c.naive_mean, 4), num(c.cross_fitted_full_fit, 4)])} />
                </div>
              </Card>
              <Card title={`Target-encoding leakage on ${leak.column}`} description={leak.explanation}>
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
            <Card title="Missing values and imputation" description={`Downstream comparison model: ${r.imputation.comparison_model}. MissingIndicator adds a 0/1 column for: ${r.imputation.missing_indicator_columns.join(", ")}.`}>
              <div className="grid gap-6 lg:grid-cols-2">
                <DataTable
                  columns={["Feature", "Missing", "Mean", "Median", "Placed | missing", "Placed | present"]}
                  rows={r.imputation.missingness.map((m) => [m.feature, pct(m.missing_rate_train), num(m.train_mean, 2), num(m.train_median, 2), pct(m.placed_rate_when_missing), pct(m.placed_rate_when_present)])}
                />
                <DataTable columns={["Strategy", "Val ROC-AUC", "Val log loss", "Fit ms"]} rows={r.imputation.downstream_comparison.map((c) => [c.strategy, num(c.validation_roc_auc, 4), num(c.validation_log_loss, 4), num(c.fit_ms, 0)])} />
              </div>
              {r.imputation.missingness.every((m) => Math.abs(m.placed_rate_when_missing - m.placed_rate_when_present) < 0.03) && (
                <div className="mt-4">
                  <Note>The placed rate is almost the same whether a value is missing or present, so missingness carries little signal here; that is why the imputation strategies score so similarly.</Note>
                </div>
              )}
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

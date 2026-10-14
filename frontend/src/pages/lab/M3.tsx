import { HBars, MetricGrid } from "../../components/charts/Common";
import { CHROME, CLASS, SERIES } from "../../components/charts/theme";
import { SelectField, Slider, Toggle } from "../../components/ui/Fields";
import { cn, int, modelLabel, num, pct } from "../../lib/format";
import type { ModuleProps } from "../Lab";
import { BarsChart, Card, DataTable, ExperimentShell, LinesChart, Note, useExperiment } from "./kit";
import { ModuleLayout } from "./ModuleLayout";

export function M3(props: ModuleProps) {
  return (
    <ModuleLayout
      id="m3"
      {...props}
      intro={<>Tree-based models on the same preprocessing as production. Accuracy is useless at a 0.17 % fraud rate, so trees are compared with PR-AUC. Classifiers train on every training fraud plus an undersampled set of legitimate rows (stated with each run); validation keeps the real rate.</>}
      experiments={[
        { key: "decision-tree", label: "Decision tree", Component: TreeExp },
        { key: "pruning", label: "Pruning", Component: PruningExp },
        { key: "regression-tree", label: "Regression tree", Component: RegTreeExp },
        { key: "ensembles", label: "Ensembles", Component: EnsembleExp },
        { key: "importance", label: "Feature importance", Component: ImportanceExp },
      ]}
    />
  );
}

/* ------------------------------------------------------------------ decision tree */

interface TreeNode {
  id: number;
  depth: number;
  feature: string | null;
  threshold: number | null;
  impurity: number;
  samples: number;
  counts: number[];
  fraud_rate: number;
  left: number | null;
  right: number | null;
}
interface TreeRes {
  params: { criterion: string; max_depth: number | null };
  depth: number;
  leaves: number;
  fit_ms: number;
  train: { average_precision: number; roc_auc: number };
  validation: { average_precision: number; roc_auc: number; at_0_5: { precision: number; recall: number; false_positives: number } };
  nodes: TreeNode[];
  root_split: null | {
    feature: string;
    threshold: number;
    parent_counts: number[];
    left_counts: number[];
    right_counts: number[];
    gini: Record<string, number>;
    entropy: Record<string, number>;
    sklearn_parent_impurity: number;
  };
  feature_importance: { feature: string; importance: number }[];
}

const short = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function TreeDiagram({ nodes }: { nodes: TreeNode[] }) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const pos = new Map<number, { x: number; y: number }>();
  const W = 960;
  const H = 360;
  const place = (id: number, depth: number, lo: number, hi: number) => {
    const n = byId.get(id);
    if (!n) return;
    pos.set(id, { x: (lo + hi) / 2, y: 30 + depth * 100 });
    if (n.left != null && byId.has(n.left)) place(n.left, depth + 1, lo, (lo + hi) / 2);
    if (n.right != null && byId.has(n.right)) place(n.right, depth + 1, (lo + hi) / 2, hi);
  };
  place(0, 0, 0, W);
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[760px]" role="img" aria-label="Top levels of the fitted decision tree">
        {nodes.map((n) =>
          [n.left, n.right].map((c, side) => {
            const a = pos.get(n.id);
            const b = c != null ? pos.get(c) : undefined;
            if (!a || !b) return null;
            return (
              <g key={`${n.id}-${side}`}>
                <path d={`M ${a.x} ${a.y + 22} C ${a.x} ${(a.y + b.y) / 2}, ${b.x} ${(a.y + b.y) / 2}, ${b.x} ${b.y - 22}`} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth={1.5} />
                <text x={(a.x + b.x) / 2 + (side ? 8 : -8)} y={(a.y + b.y) / 2} textAnchor={side ? "start" : "end"} fill={CHROME.muted} fontSize="10">
                  {side ? "no" : "yes"}
                </text>
              </g>
            );
          }),
        )}
        {nodes.map((n) => {
          const p = pos.get(n.id);
          if (!p) return null;
          const w = n.depth >= 3 ? 104 : 150;
          return (
            <g key={n.id} transform={`translate(${p.x - w / 2}, ${p.y - 22})`}>
              <title>{`${n.feature ? `${n.feature} ≤ ${n.threshold}` : "Leaf"} · samples ${n.samples} · impurity ${n.impurity} · fraud ${(n.fraud_rate * 100).toFixed(1)}%`}</title>
              <rect width={w} height={44} rx={9} fill="#141831" stroke="rgba(255,255,255,0.14)" />
              <rect x={6} y={36} width={w - 12} height={3} rx={1.5} fill={CLASS.legitimate} />
              <rect x={6} y={36} width={(w - 12) * n.fraud_rate} height={3} rx={1.5} fill={CLASS.fraud} />
              <text x={w / 2} y={16} textAnchor="middle" fill="#eef0fb" fontSize={n.depth >= 3 ? 9 : 10.5} fontWeight={600}>
                {n.feature ? `${short(n.feature, n.depth >= 3 ? 9 : 16)} ≤ ${num(n.threshold, 2)}` : "leaf"}
              </text>
              <text x={w / 2} y={29} textAnchor="middle" fill={CHROME.ink2} fontSize="9.5">
                n={n.samples} · {(n.fraud_rate * 100).toFixed(0)}%{n.depth >= 3 ? "" : " fraud"}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function TreeExp() {
  const exp = useExperiment<TreeRes>("m3", "decision-tree", { criterion: "gini", max_depth: 4, min_samples_leaf: 20, ccp_alpha: 0 }, { auto: true, live: true });
  const unlimited = exp.params.max_depth == null;
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Decision tree"
      topics={["Gini impurity", "Entropy", "Information gain", "Pre-pruning", "Overfitting"]}
      estimate="< 1 s"
      controls={
        <>
          <SelectField label="Split criterion" value={String(exp.params.criterion)} onChange={(v) => exp.setParam("criterion", v)} options={[{ value: "gini", label: "Gini impurity" }, { value: "entropy", label: "Entropy (information gain)" }, { value: "log_loss", label: "Log loss" }]} />
          <Toggle label="Unlimited depth" checked={unlimited} onChange={(v) => exp.setParam("max_depth", v ? null : 4)} hint="Grow until leaves are pure (overfits)" />
          {!unlimited && <Slider label="max_depth" value={Number(exp.params.max_depth)} min={1} max={20} step={1} onChange={(v) => exp.setParam("max_depth", v)} />}
          <Slider label="min_samples_leaf" value={Number(exp.params.min_samples_leaf)} min={1} max={300} step={1} onChange={(v) => exp.setParam("min_samples_leaf", v)} />
          <Slider label="ccp_alpha (post-pruning)" value={Number(exp.params.ccp_alpha)} min={0} max={0.01} step={0.0005} onChange={(v) => exp.setParam("ccp_alpha", v)} format={(v) => v.toFixed(4)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const rs = r.root_split;
        return (
          <>
            <Card>
              <MetricGrid
                items={[
                  { label: "Train PR-AUC", value: num(r.train.average_precision, 4) },
                  { label: "Val PR-AUC", value: num(r.validation.average_precision, 4), hint: `gap ${num(r.train.average_precision - r.validation.average_precision, 3)}` },
                  { label: "Val ROC-AUC", value: num(r.validation.roc_auc, 4) },
                  { label: "Val precision @0.5", value: num(r.validation.at_0_5.precision, 3), hint: `recall ${num(r.validation.at_0_5.recall, 3)} · ${int(r.validation.at_0_5.false_positives)} FP` },
                  { label: "Depth / leaves", value: `${r.depth} / ${r.leaves}`, hint: `fit ${num(r.fit_ms, 0)} ms` },
                ]}
              />
            </Card>
            <Card title="Top of the fitted tree" description="Each node shows its split rule, size and share of fraud in the training sample (orange part of the bar). Thresholds are on the standardised scale (0 = training mean). Left branch = condition true. Hover a node for the full rule.">
              <TreeDiagram nodes={r.nodes} />
            </Card>
            {rs && (
              <Card title={`Root split: ${rs.feature} ≤ ${rs.threshold}`} description={`Class counts [legitimate, fraud]: parent ${rs.parent_counts.join(" / ")}, left ${rs.left_counts.join(" / ")}, right ${rs.right_counts.join(" / ")}. Recomputed by hand from the counts; scikit-learn reports parent impurity ${rs.sklearn_parent_impurity}.`}>
                <DataTable
                  columns={["Measure", "Parent", "Left", "Right", "Weighted children", "Decrease / gain"]}
                  rows={[
                    ["Gini = 1 − Σp²", num(rs.gini.parent, 4), num(rs.gini.left, 4), num(rs.gini.right, 4), num(rs.gini.weighted_children, 4), num(rs.gini.decrease, 4)],
                    ["Entropy = −Σp log₂p", num(rs.entropy.parent, 4), num(rs.entropy.left, 4), num(rs.entropy.right, 4), num(rs.entropy.weighted_children, 4), num(rs.entropy.information_gain, 4)],
                  ]}
                />
              </Card>
            )}
            <Card>
              <HBars title="Impurity-based feature importance" subtitle="Total impurity decrease contributed by each (encoded) feature" data={r.feature_importance.map((f) => ({ label: f.feature, value: f.importance }))} />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ pruning */

interface PruneRes {
  pre_pruning: { curve: { max_depth: number | string; train_average_precision: number; validation_average_precision: number; leaves: number }[]; best: { max_depth: number | string; validation_average_precision: number; leaves: number } };
  post_pruning: { alphas_in_path: number; curve: { ccp_alpha: number; train_average_precision: number; validation_average_precision: number; leaves: number; depth: number }[]; best: { ccp_alpha: number; validation_average_precision: number; leaves: number; depth: number } };
  note: string;
}

function PruningExp() {
  const exp = useExperiment<PruneRes>("m3", "pruning", { criterion: "gini" }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Pre-pruning vs. cost-complexity post-pruning"
      topics={["max_depth", "Cost-complexity pruning", "Overfitting", "Underfitting"]}
      estimate="≈ 3 s"
      controls={<SelectField label="Criterion" value={String(exp.params.criterion)} onChange={(v) => exp.setParam("criterion", v)} options={[{ value: "gini", label: "Gini" }, { value: "entropy", label: "Entropy" }]} />}
    >
      {(d) => {
        const r = d.result;
        const depth = r.pre_pruning.curve.filter((c) => c.max_depth !== "none") as { max_depth: number; train_average_precision: number; validation_average_precision: number; leaves: number }[];
        const ccp = r.post_pruning.curve.filter((c) => c.ccp_alpha > 0).map((c) => ({ ...c, log_alpha: Math.log10(c.ccp_alpha) }));
        return (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title="Pre-pruning: PR-AUC vs. max_depth"
                  subtitle={`Best validation PR-AUC ${num(r.pre_pruning.best.validation_average_precision, 3)} at depth ${r.pre_pruning.best.max_depth} (${r.pre_pruning.best.leaves} leaves)`}
                  data={depth}
                  x="max_depth"
                  xLabel="max_depth"
                  series={[{ key: "train_average_precision", name: "Train" }, { key: "validation_average_precision", name: "Validation" }]}
                  yFormat={(v) => num(v, 2)}
                  refX={[{ value: Number(r.pre_pruning.best.max_depth) }]}
                />
              </Card>
              <Card>
                <LinesChart
                  title="Post-pruning: PR-AUC vs. ccp_alpha"
                  subtitle={`${r.post_pruning.alphas_in_path} alphas in the pruning path; best ${num(r.post_pruning.best.validation_average_precision, 3)} at α=${r.post_pruning.best.ccp_alpha} (${r.post_pruning.best.leaves} leaves)`}
                  data={ccp}
                  x="log_alpha"
                  xLabel="log₁₀ ccp_alpha"
                  xFormat={(v) => v.toFixed(1)}
                  series={[{ key: "train_average_precision", name: "Train" }, { key: "validation_average_precision", name: "Validation" }]}
                  yFormat={(v) => num(v, 2)}
                />
              </Card>
            </div>
            <Card>
              <LinesChart title="Tree size shrinks as ccp_alpha grows" data={ccp} x="log_alpha" xLabel="log₁₀ ccp_alpha" xFormat={(v) => v.toFixed(1)} series={[{ key: "leaves", name: "Leaves", color: SERIES[2] }]} yFormat={(v) => String(Math.round(v))} height={200} />
              <div className="mt-3">
                <Note>{r.note}</Note>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ regression tree */

interface RegTreeRes {
  features: string[];
  root_split: { feature: string; threshold_scaled: number; parent_mse: number; left: { samples: number; mse: number; mean_target: number }; right: { samples: number; mse: number; mean_target: number }; weighted_children_mse: number; mse_reduction: number };
  validation_metrics: { rmse: number; mae: number; r2: number; mape: number | null };
  depth_curve: { max_depth: number; train_rmse: number; validation_rmse: number }[];
}

function RegTreeExp() {
  const exp = useExperiment<RegTreeRes>("m3", "regression-tree", { max_depth: 4 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Regression tree (log amount)"
      topics={["Regression trees", "MSE splitting", "Variance reduction"]}
      estimate="≈ 10–15 s"
      controls={<Slider label="max_depth" value={Number(exp.params.max_depth)} min={1} max={15} step={1} onChange={(v) => exp.setParam("max_depth", v)} />}
    >
      {(d) => {
        const r = d.result;
        const s = r.root_split;
        return (
          <>
            <Card title={`Root split on ${s.feature} (standardised threshold ${s.threshold_scaled})`} description="A regression tree chooses the split that most reduces the weighted mean squared error of the children.">
              <MetricGrid
                items={[
                  { label: "Parent MSE", value: num(s.parent_mse, 3) },
                  { label: "Left MSE", value: num(s.left.mse, 3), hint: `n=${s.left.samples}, mean ${num(s.left.mean_target, 2)}` },
                  { label: "Right MSE", value: num(s.right.mse, 3), hint: `n=${s.right.samples}, mean ${num(s.right.mean_target, 2)}` },
                  { label: "MSE reduction", value: num(s.mse_reduction, 3), hint: `weighted children ${num(s.weighted_children_mse, 3)}` },
                ]}
              />
            </Card>
            <Card>
              <LinesChart
                title="RMSE vs. depth (target log(1 + Amount))"
                subtitle={`At the chosen depth: validation RMSE ${num(r.validation_metrics.rmse, 3)}, R² ${num(r.validation_metrics.r2, 3)}`}
                data={r.depth_curve}
                x="max_depth"
                xLabel="max_depth"
                series={[{ key: "train_rmse", name: "Train RMSE" }, { key: "validation_rmse", name: "Validation RMSE" }]}
                yFormat={(v) => num(v, 2)}
              />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ ensembles */

interface EnsRes {
  protocol: string;
  comparison: { model: string; label: string; family: string; train_average_precision: number; validation_average_precision: number; validation_roc_auc: number; validation_log_loss: number; overfit_gap: number; fit_seconds: number }[];
  random_forest: { oob_average_precision: number; oob_curve: { n_estimators: number; oob_average_precision: number; validation_average_precision: number }[]; bootstrap_unique_fraction: number; expected_unique_fraction: number; features_per_split: string; n_features: number };
  gradient_boosting_stages: { stage: number; train_log_loss: number; validation_log_loss: number }[];
  adaboost: { first_estimator_weights: number[]; first_estimator_errors: number[] };
  note: string;
}

function EnsembleExp() {
  const exp = useExperiment<EnsRes>("m3", "ensembles", { n_estimators: 200, max_features: "sqrt", learning_rate: 0.1 }, { auto: false });
  return (
    <ExperimentShell
      exp={exp}
      title="Bagging vs. boosting"
      topics={["Random forest", "Bootstrap", "Feature subsampling", "OOB error", "AdaBoost", "Gradient boosting", "XGBoost", "LightGBM"]}
      estimate="≈ 30–50 s (trains 7 models)"
      controls={
        <>
          <Slider label="n_estimators" value={Number(exp.params.n_estimators)} min={20} max={500} step={10} onChange={(v) => exp.setParam("n_estimators", v)} />
          <SelectField label="Random forest max_features" value={String(exp.params.max_features)} onChange={(v) => exp.setParam("max_features", v)} options={[{ value: "sqrt", label: "√p (feature subsampling)" }, { value: "log2", label: "log₂ p" }, { value: "all", label: "All p (bagged trees)" }]} />
          <SelectField label="Boosting learning rate" value={String(exp.params.learning_rate)} onChange={(v) => exp.setParam("learning_rate", Number(v))} options={[0.01, 0.05, 0.1, 0.3, 1].map((v) => ({ value: String(v), label: String(v) }))} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const rf = r.random_forest;
        return (
          <>
            <Card title="Same data, same split, seven models" description={r.protocol}>
              <DataTable
                columns={["Model", "Family", "Val PR-AUC", "Val ROC-AUC", "Val log loss", "Train PR-AUC", "Overfit gap", "Fit s"]}
                rows={r.comparison.map((c) => [c.label, c.family, num(c.validation_average_precision, 4), num(c.validation_roc_auc, 4), num(c.validation_log_loss, 4), num(c.train_average_precision, 4), <span className={cn(c.overfit_gap > 0.1 && "text-warning")}>{num(c.overfit_gap, 3)}</span>, num(c.fit_seconds, 2)])}
              />
              <div className="mt-4">
                <Note>{r.note}</Note>
              </div>
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <BarsChart
                  title="Train − validation PR-AUC gap"
                  subtitle="A large gap signals overfitting (single unpruned tree)"
                  data={r.comparison.map((c) => ({ model: modelLabel(c.model).replace("Hist. Gradient Boosting", "HistGB").replace("Gradient Boosting", "GB"), gap: c.overfit_gap }))}
                  x="model"
                  series={[{ key: "gap", name: "Overfit gap" }]}
                  yFormat={(v) => num(v, 2)}
                />
              </Card>
              <Card>
                <LinesChart
                  title="Random forest: out-of-bag vs. validation PR-AUC"
                  subtitle={`OOB PR-AUC ${num(rf.oob_average_precision, 3)} (enriched training sample, so higher than validation) · ${rf.features_per_split} of ${rf.n_features} features per split`}
                  data={rf.oob_curve}
                  x="n_estimators"
                  xLabel="Trees"
                  series={[{ key: "oob_average_precision", name: "OOB PR-AUC" }, { key: "validation_average_precision", name: "Validation PR-AUC" }]}
                  yFormat={(v) => num(v, 2)}
                  footer={`One bootstrap sample contained ${pct(rf.bootstrap_unique_fraction)} unique rows (theory: 1 − 1/e ≈ ${pct(rf.expected_unique_fraction)}); the remaining rows are "out of bag" and act as a built-in validation set.`}
                />
              </Card>
            </div>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title="Gradient boosting: log loss by stage"
                  subtitle="Each stage fits a tree to the residual errors of the ensemble so far"
                  data={r.gradient_boosting_stages}
                  x="stage"
                  xLabel="Boosting stage"
                  series={[{ key: "train_log_loss", name: "Train" }, { key: "validation_log_loss", name: "Validation" }]}
                />
              </Card>
              <Card>
                <BarsChart
                  title="AdaBoost: estimator weights (first 25 stumps)"
                  subtitle="Weight α = learning_rate · log((1 − error)/error); later learners focus on misclassified rows"
                  data={r.adaboost.first_estimator_weights.map((w, i) => ({ stage: i + 1, weight: w, error: r.adaboost.first_estimator_errors[i] }))}
                  x="stage"
                  series={[{ key: "weight", name: "Weight", color: SERIES[6] }]}
                />
              </Card>
            </div>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ importance */

interface ImpRes {
  model: string;
  importances: { feature: string; impurity: number; permutation: number; permutation_std: number }[];
  shap: { available: boolean; error?: string; output_space?: string; rows_explained?: number; mean_abs_shap?: { feature: string; value: number }[]; example?: { index: number; actual: number; base_value: number; output_value: number; top_contributions: { feature: string; shap: number; value: number }[] } };
  limitations: { max_abs_correlation_between_pca_components: number; strongest_correlation_with_log_amount: { feature: string; abs_correlation: number }; notes: string[] };
}

function ImportanceExp() {
  const exp = useExperiment<ImpRes>("m3", "importance", { model: "random_forest", explain_index: 0 }, { auto: true });
  return (
    <ExperimentShell
      exp={exp}
      title="Impurity, permutation and SHAP"
      topics={["Impurity importance", "Permutation importance", "SHAP (introductory)", "Limitations"]}
      estimate="≈ 5–15 s"
      controls={
        <>
          <SelectField label="Model" value={String(exp.params.model)} onChange={(v) => exp.setParam("model", v)} options={["random_forest", "xgboost", "lightgbm", "gradient_boosting"].map((m) => ({ value: m, label: modelLabel(m) }))} />
          <Slider label="Transaction to explain (validation row)" value={Number(exp.params.explain_index)} min={0} max={149} step={1} onChange={(v) => exp.setParam("explain_index", v)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const top = r.importances.slice(0, 12);
        const ex = r.shap.example;
        return (
          <>
            <Card title="Two views of importance" description="Impurity importance is learned from training data; permutation importance is the drop in validation PR-AUC when one raw feature is shuffled (every validation fraud + 4,000 legitimate rows).">
              <BarsChart
                title="Top 12 features by permutation importance"
                data={top.map((t) => ({ feature: t.feature, permutation: t.permutation, impurity: t.impurity }))}
                x="feature"
                series={[{ key: "permutation", name: "Permutation (ΔPR-AUC)" }, { key: "impurity", name: "Impurity (normalised)" }]}
                yFormat={(v) => num(v, 3)}
                height={280}
              />
            </Card>
            {r.shap.available && r.shap.mean_abs_shap && ex ? (
              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <HBars title="Mean |SHAP| per feature" subtitle={`${r.shap.rows_explained} validation rows · ${r.shap.output_space} space`} data={r.shap.mean_abs_shap.slice(0, 12).map((s) => ({ label: s.feature, value: s.value }))} color={SERIES[6]} />
                </Card>
                <Card>
                  <HBars
                    title={`Why validation row #${ex.index} got this score`}
                    subtitle={`Base value ${num(ex.base_value, 3)} → output ${num(ex.output_value, 3)} (${r.shap.output_space}); actual class: ${ex.actual ? "Fraud" : "Legitimate"}`}
                    data={ex.top_contributions.map((c) => ({ label: c.feature, value: c.shap }))}
                    diverging
                    valueLabel="SHAP contribution"
                  />
                </Card>
              </div>
            ) : (
              <Note tone="warn">SHAP is unavailable{r.shap.error ? `: ${r.shap.error}` : "."}</Note>
            )}
            <Card title="Limitations of feature importance">
              <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-2">
                {r.limitations.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
                <li>
                  Measured here: the largest absolute correlation between two PCA components is {num(r.limitations.max_abs_correlation_between_pca_components, 4)}; the component most correlated with log(1 + Amount) is{" "}
                  {r.limitations.strongest_correlation_with_log_amount.feature} (|r| = {num(r.limitations.strongest_correlation_with_log_amount.abs_correlation, 3)}).
                </li>
              </ul>
              <p className="mt-3 text-xs" style={{ color: CHROME.muted }}>
                Model: {modelLabel(r.model)}
              </p>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

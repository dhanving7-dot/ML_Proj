import { MetricGrid } from "../../components/charts/Common";
import { CHROME, OUTCOME, SERIES } from "../../components/charts/theme";
import { SelectField, Slider } from "../../components/ui/Fields";
import { num, pct } from "../../lib/format";
import type { ModuleProps } from "../Lab";
import { BarsChart, Card, DataTable, ExperimentShell, LinesChart, Note, ScatterGroups, useExperiment } from "./kit";
import { ModuleLayout } from "./ModuleLayout";

export function M4(props: ModuleProps) {
  return (
    <ModuleLayout
      id="m4"
      {...props}
      intro={
        <>
          Unsupervised methods on standardised numeric features. Labels are never used to fit these models; placement and the dataset's <code className="font-mono text-ink">IsAnomaly</code> flag are only
          attached afterwards to describe the results. Clusters are exploratory and do not replace the supervised placement model. A clean-looking 2-D picture is not proof that the clusters are real.
        </>
      }
      experiments={[
        { key: "kmeans", label: "K-means", Component: KMeansExp },
        { key: "hierarchical", label: "Hierarchical", Component: HierExp },
        { key: "dbscan", label: "DBSCAN", Component: DbscanExp },
        { key: "pca", label: "PCA", Component: PcaExp },
        { key: "embedding", label: "t-SNE / UMAP", Component: EmbedExp },
        { key: "anomaly", label: "Anomaly detection", Component: AnomalyExp },
      ]}
    />
  );
}

type Pt = { x: number; y: number; cluster: number; placed: number; point_type?: string };

const clusterGroups = (n: number) => Array.from({ length: n }, (_, i) => ({ key: i, label: `Cluster ${i}`, color: SERIES[i] }));

/* ------------------------------------------------------------------ k-means */

interface KRes {
  features: string[];
  k: number;
  elbow: { k: number; inertia: number; silhouette: number | null }[];
  silhouette: number;
  clusters: { cluster: number; size: number; placed_rate: number; centroid: Record<string, number> }[];
  projection: Pt[];
  note: string;
}

function KMeansExp() {
  const exp = useExperiment<KRes>("m4", "kmeans", { k: 3, sample_size: 3000 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="K-means clustering"
      topics={["K-means", "Elbow method", "Silhouette score"]}
      estimate="≈ 2 s"
      controls={
        <>
          <Slider label="Clusters (k)" value={Number(exp.params.k)} min={2} max={10} step={1} onChange={(v) => exp.setParam("k", v)} />
          <SelectField label="Sample size" value={String(exp.params.sample_size)} onChange={(v) => exp.setParam("sample_size", Number(v))} options={[1000, 3000, 6000].map((v) => ({ value: String(v), label: v.toLocaleString() }))} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const keyFeats = ["CGPA", "CodingTestScore", "AptitudeTestScore", "Internships", "AttendancePercent"];
        return (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart title="Elbow method: inertia vs. k" subtitle="Within-cluster sum of squares always falls as k grows; look for the bend" data={r.elbow} x="k" xLabel="k" series={[{ key: "inertia", name: "Inertia", dot: true }]} yFormat={(v) => num(v, 0)} refX={[{ value: r.k, label: "chosen k" }]} />
              </Card>
              <Card>
                <LinesChart title="Silhouette score vs. k" subtitle="Higher = tighter, better-separated clusters (−1 … 1)" data={r.elbow.filter((e) => e.silhouette != null)} x="k" xLabel="k" series={[{ key: "silhouette", name: "Silhouette", color: SERIES[2], dot: true }]} refX={[{ value: r.k }]} />
              </Card>
            </div>
            <Card>
              <ScatterGroups
                title={`k = ${r.k} clusters projected onto the first two principal components`}
                subtitle={`Silhouette ${num(r.silhouette, 3)}. PCA is only used for drawing; K-means ran in all ${r.features.length} dimensions.`}
                points={r.projection.map((p) => ({ x: p.x, y: p.y, g: p.cluster }))}
                groups={clusterGroups(r.k)}
                xLabel="PC1"
                yLabel="PC2"
              />
            </Card>
            <Card title="Cluster profiles (centroids in original units)" description={r.note}>
              <DataTable
                columns={["Cluster", "Size", "Placed rate", ...keyFeats]}
                rows={r.clusters.map((c) => [`Cluster ${c.cluster}`, c.size, pct(c.placed_rate), ...keyFeats.map((f) => num(c.centroid[f], 2))])}
              />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ hierarchical */

interface HRes {
  linkage: string;
  n_clusters: number;
  comparison: { linkage: string; silhouette: number | null; cophenetic_correlation: number; largest_cluster_fraction: number; cluster_sizes: number[] }[];
  dendrogram: { icoord: number[][]; dcoord: number[][]; leaf_labels: string[]; colors: string[]; cut_height: number };
  projection: Pt[];
  placed_rate_by_cluster: { cluster: number; size: number; placed_rate: number }[];
  note: string;
}

function Dendrogram({ dg }: { dg: HRes["dendrogram"] }) {
  const xs = dg.icoord.flat();
  const ys = dg.dcoord.flat();
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMax = Math.max(...ys, dg.cut_height) * 1.05;
  const W = 900;
  const H = 300;
  const pad = { l: 40, r: 10, t: 10, b: 30 };
  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - y / yMax) * (H - pad.t - pad.b);
  // One neutral hue: SciPy's branch colours follow traversal order and need not match the scatter's cluster labels.
  const colorOf = () => SERIES[6];
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[640px]" role="img" aria-label={`Dendrogram truncated to the last ${dg.leaf_labels.length} merges; dashed line marks the cut`}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(yMax * f)} y2={sy(yMax * f)} stroke="rgba(255,255,255,0.05)" />
            <text x={pad.l - 6} y={sy(yMax * f) + 3} textAnchor="end" fontSize="10" fill={CHROME.muted}>
              {(yMax * f).toFixed(0)}
            </text>
          </g>
        ))}
        {dg.icoord.map((ic, i) => (
          <polyline key={i} points={ic.map((x, j) => `${sx(x)},${sy(dg.dcoord[i][j])}`).join(" ")} fill="none" stroke={colorOf()} strokeWidth={1.6} />
        ))}
        <line x1={pad.l} x2={W - pad.r} y1={sy(dg.cut_height)} y2={sy(dg.cut_height)} stroke="#eef0fb" strokeDasharray="5 4" strokeWidth={1} />
        <text x={W - pad.r} y={sy(dg.cut_height) - 4} textAnchor="end" fontSize="10" fill={CHROME.ink2}>
          cut
        </text>
        {dg.leaf_labels.map((l, i) => (
          <text key={i} x={sx(xMin + i * ((xMax - xMin) / Math.max(dg.leaf_labels.length - 1, 1)))} y={H - 12} textAnchor="middle" fontSize="8.5" fill={CHROME.muted}>
            {l}
          </text>
        ))}
      </svg>
    </div>
  );
}

function HierExp() {
  const exp = useExperiment<HRes>("m4", "hierarchical", { linkage: "ward", n_clusters: 3, sample_size: 500 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Agglomerative hierarchical clustering"
      topics={["Single linkage", "Complete linkage", "Average linkage", "Ward", "Dendrogram", "Cophenetic correlation"]}
      estimate="< 1 s"
      controls={
        <>
          <SelectField label="Linkage" value={String(exp.params.linkage)} onChange={(v) => exp.setParam("linkage", v)} options={["ward", "complete", "average", "single"].map((l) => ({ value: l, label: l[0].toUpperCase() + l.slice(1) }))} />
          <Slider label="Clusters (cut)" value={Number(exp.params.n_clusters)} min={2} max={10} step={1} onChange={(v) => exp.setParam("n_clusters", v)} />
          <SelectField label="Sample size" value={String(exp.params.sample_size)} onChange={(v) => exp.setParam("sample_size", Number(v))} options={[200, 500, 1000].map((v) => ({ value: String(v), label: String(v) }))} hint="O(n²) memory, so a sample is used" />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        return (
          <>
            <Card title={`Dendrogram (${r.linkage} linkage)`} description="Leaves are truncated to the last 30 merges; numbers in brackets are leaf sizes. Cutting at the dashed line gives the chosen number of clusters.">
              <Dendrogram dg={r.dendrogram} />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="All four linkages on the same sample" description={r.note}>
                <DataTable
                  columns={["Linkage", "Silhouette", "Cophenetic r", "Largest cluster", "Sizes"]}
                  rows={r.comparison.map((c) => [c.linkage, num(c.silhouette, 3), num(c.cophenetic_correlation, 3), pct(c.largest_cluster_fraction), c.cluster_sizes.join(" / ")])}
                />
              </Card>
              <Card>
                <ScatterGroups title="Clusters in PCA space" points={r.projection.map((p) => ({ x: p.x, y: p.y, g: p.cluster }))} groups={clusterGroups(r.n_clusters)} xLabel="PC1" yLabel="PC2" height={280} />
              </Card>
            </div>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ dbscan */

interface DRes {
  params: { eps: number; min_samples: number };
  n_clusters: number;
  n_core: number;
  n_border: number;
  n_noise: number;
  noise_placed_rate: number | null;
  silhouette_excluding_noise: number | null;
  clusters: { cluster: number; size: number; placed_rate: number }[];
  k_distance: { rank: number; distance: number }[];
  k_distance_hint: number;
  projection: Pt[];
  note: string;
}

function DbscanExp() {
  const exp = useExperiment<DRes>("m4", "dbscan", { eps: 1.8, min_samples: 10, sample_size: 2500 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="DBSCAN density clustering"
      topics={["DBSCAN", "Core points", "Border points", "Noise", "k-distance plot"]}
      estimate="< 1 s"
      controls={
        <>
          <Slider label="eps (neighbourhood radius)" value={Number(exp.params.eps)} min={0.4} max={5} step={0.1} onChange={(v) => exp.setParam("eps", v)} format={(v) => v.toFixed(1)} />
          <Slider label="min_samples" value={Number(exp.params.min_samples)} min={2} max={40} step={1} onChange={(v) => exp.setParam("min_samples", v)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const total = r.n_core + r.n_border + r.n_noise;
        return (
          <>
            <Card>
              <MetricGrid
                items={[
                  { label: "Clusters found", value: String(r.n_clusters) },
                  { label: "Core points", value: r.n_core.toLocaleString(), hint: pct(r.n_core / total) },
                  { label: "Border points", value: r.n_border.toLocaleString(), hint: pct(r.n_border / total) },
                  { label: "Noise points", value: r.n_noise.toLocaleString(), hint: r.noise_placed_rate != null ? `${pct(r.noise_placed_rate)} placed` : undefined },
                  { label: "Silhouette (no noise)", value: num(r.silhouette_excluding_noise, 3) },
                ]}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title={`Sorted ${r.params.min_samples}-distance plot`}
                  subtitle={`Pick eps near the knee. 90th percentile = ${num(r.k_distance_hint, 2)}; current eps dashed.`}
                  data={r.k_distance}
                  x="rank"
                  xLabel="Points sorted by distance"
                  series={[{ key: "distance", name: "k-distance" }]}
                  refY={[{ value: r.params.eps, label: `eps ${r.params.eps}` }]}
                  yFormat={(v) => num(v, 2)}
                />
              </Card>
              <Card>
                <ScatterGroups
                  title="Point types in PCA space"
                  subtitle="Core points have ≥ min_samples neighbours within eps; border points sit near a core point; noise belongs to no cluster"
                  points={r.projection.map((p) => ({ x: p.x, y: p.y, g: p.point_type ?? "core" }))}
                  groups={[
                    { key: "core", label: "Core", color: SERIES[0] },
                    { key: "border", label: "Border", color: SERIES[2] },
                    { key: "noise", label: "Noise", color: SERIES[1] },
                  ]}
                  xLabel="PC1"
                  yLabel="PC2"
                  height={280}
                />
              </Card>
            </div>
            <Card title="Clusters" description={r.note}>
              {r.clusters.length ? <DataTable columns={["Cluster", "Size", "Placed rate"]} rows={r.clusters.map((c) => [`Cluster ${c.cluster}`, c.size, pct(c.placed_rate)])} /> : <Note tone="warn">No clusters at these settings: every point is noise. Increase eps or lower min_samples.</Note>}
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ pca */

interface PcaRes {
  features: string[];
  explained_variance: { component: string; ratio: number; cumulative: number }[];
  components_for_90: number;
  components_for_95: number;
  loadings: { feature: string; pc1: number; pc2: number }[];
  projection: { x: number; y: number; placed: number }[];
}

function PcaExp() {
  const exp = useExperiment<PcaRes>("m4", "pca", { sample_size: 20000 }, { auto: true });
  return (
    <ExperimentShell exp={exp} title="Principal component analysis" topics={["PCA", "Explained variance", "Cumulative variance", "Loadings", "Projection"]} estimate="< 1 s">
      {(d) => {
        const r = d.result;
        return (
          <>
            <Card>
              <MetricGrid
                items={[
                  { label: "PC1 variance", value: pct(r.explained_variance[0].ratio) },
                  { label: "PC1+PC2", value: pct(r.explained_variance[1].cumulative) },
                  { label: "PCs for 90%", value: String(r.components_for_90) },
                  { label: "PCs for 95%", value: `${r.components_for_95} of ${r.features.length}` },
                ]}
              />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <LinesChart
                  title="Scree plot"
                  subtitle="Per-component and cumulative explained variance (same 0–1 scale)"
                  data={r.explained_variance.map((e, i) => ({ ...e, n: i + 1 }))}
                  x="n"
                  xLabel="Component"
                  series={[{ key: "ratio", name: "Explained", dot: true }, { key: "cumulative", name: "Cumulative", dot: true }]}
                  yFormat={(v) => pct(v, 0)}
                  yDomain={[0, 1]}
                  refY={[{ value: 0.9, label: "90%" }]}
                />
              </Card>
              <Card>
                <ScatterGroups
                  title="Students on PC1 vs. PC2"
                  subtitle="Coloured by placement after fitting; PCA itself never sees the label"
                  points={r.projection.map((p) => ({ x: p.x, y: p.y, g: p.placed }))}
                  groups={[{ key: 1, label: "Placed", color: OUTCOME.placed }, { key: 0, label: "Not placed", color: OUTCOME.notPlaced }]}
                  xLabel="PC1"
                  yLabel="PC2"
                  height={280}
                />
              </Card>
            </div>
            <Card title="Loadings (weight of each standardised feature)">
              <DataTable columns={["Feature", "PC1", "PC2"]} rows={r.loadings.map((l) => [l.feature, num(l.pc1, 3), num(l.pc2, 3)])} />
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ embeddings */

interface EmbRes {
  method: string;
  params: Record<string, number>;
  seconds: number;
  trustworthiness: number;
  umap_available: boolean;
  points: { x: number; y: number; placed: number; tier: string }[];
  note: string;
}

function EmbedExp() {
  const exp = useExperiment<EmbRes>("m4", "embedding", { method: "tsne", perplexity: 30, n_neighbors: 15, sample_size: 1500 }, { auto: true });
  const tsne = exp.params.method === "tsne";
  return (
    <ExperimentShell
      exp={exp}
      title="Non-linear embeddings"
      topics={["t-SNE", "UMAP", "Perplexity", "Trustworthiness"]}
      estimate="≈ 2 s (t-SNE), ≈ 3–15 s (UMAP; first run compiles)"
      controls={
        <>
          <SelectField label="Method" value={String(exp.params.method)} onChange={(v) => exp.setParam("method", v)} options={[{ value: "tsne", label: "t-SNE" }, { value: "umap", label: "UMAP" }]} />
          {tsne ? (
            <Slider label="Perplexity" value={Number(exp.params.perplexity)} min={5} max={100} step={5} onChange={(v) => exp.setParam("perplexity", v)} />
          ) : (
            <Slider label="n_neighbors" value={Number(exp.params.n_neighbors)} min={2} max={100} step={1} onChange={(v) => exp.setParam("n_neighbors", v)} />
          )}
        </>
      }
    >
      {(d) => {
        const r = d.result;
        return (
          <>
            <Card>
              <ScatterGroups
                title={`${r.method === "tsne" ? "t-SNE" : "UMAP"} embedding of ${r.points.length.toLocaleString()} students`}
                subtitle={`Trustworthiness ${num(r.trustworthiness, 3)} (1 = local neighbourhoods fully preserved) · ${num(r.seconds, 1)} s`}
                points={r.points.map((p) => ({ x: p.x, y: p.y, g: p.placed }))}
                groups={[{ key: 1, label: "Placed", color: OUTCOME.placed }, { key: 0, label: "Not placed", color: OUTCOME.notPlaced }]}
                xLabel="dim 1"
                yLabel="dim 2"
                height={380}
              />
            </Card>
            <Note tone="warn">{r.note}</Note>
          </>
        );
      }}
    </ExperimentShell>
  );
}

/* ------------------------------------------------------------------ anomaly */

interface DetRes {
  flagged: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc_of_score: number;
  score_histogram: { score: number; normal: number; labelled_anomaly: number }[];
  seconds: number;
  train_rows?: number;
}
interface AnomRes {
  params: { contamination: number; nu: number };
  labelled_anomaly_rate_validation: number;
  isolation_forest: DetRes;
  one_class_svm: DetRes;
  labelled_profile: Record<string, Record<string, number>>;
  autoencoder: string;
  note: string;
}

function AnomalyExp() {
  const exp = useExperiment<AnomRes>("m4", "anomaly", { contamination: 0.035, nu: 0.035 }, { auto: true, live: true });
  return (
    <ExperimentShell
      exp={exp}
      live
      title="Anomaly detection"
      topics={["Isolation Forest", "One-Class SVM", "Contamination", "Autoencoder (conceptual)"]}
      estimate="≈ 1–2 s"
      controls={
        <>
          <Slider label="Isolation Forest contamination" value={Number(exp.params.contamination)} min={0.005} max={0.2} step={0.005} onChange={(v) => exp.setParam("contamination", v)} format={(v) => pct(v, 1)} />
          <Slider label="One-Class SVM ν" value={Number(exp.params.nu)} min={0.005} max={0.3} step={0.005} onChange={(v) => exp.setParam("nu", v)} format={(v) => v.toFixed(3)} />
        </>
      }
    >
      {(d) => {
        const r = d.result;
        const dets = [
          { name: "Isolation Forest", r: r.isolation_forest },
          { name: `One-Class SVM (fit on ${r.one_class_svm.train_rows} rows)`, r: r.one_class_svm },
        ];
        return (
          <>
            <Card title="Detectors scored against the dataset's IsAnomaly flag (validation split)" description={`Labelled anomaly rate: ${pct(r.labelled_anomaly_rate_validation, 2)}. ROC-AUC uses the continuous anomaly score; precision/recall use the flagged rows.`}>
              <DataTable columns={["Detector", "Flagged", "Precision", "Recall", "F1", "Score ROC-AUC", "Seconds"]} rows={dets.map(({ name, r: x }) => [name, x.flagged, num(x.precision, 3), num(x.recall, 3), num(x.f1, 3), num(x.roc_auc_of_score, 3), num(x.seconds, 2)])} />
            </Card>
            <div className="grid gap-6 lg:grid-cols-2">
              {dets.map(({ name, r: x }) => (
                <Card key={name}>
                  <BarsChart
                    title={`${name.split(" (")[0]}: anomaly score distribution`}
                    subtitle="Higher score = more anomalous"
                    data={x.score_histogram}
                    x="score"
                    xFormat={(v) => num(Number(v), 2)}
                    series={[{ key: "normal", name: "Normal rows" }, { key: "labelled_anomaly", name: "Labelled anomalies" }]}
                    yFormat={(v) => String(Math.round(v))}
                    stacked
                  />
                </Card>
              ))}
            </div>
            <Card title="What the labelled anomalies look like (validation means)">
              <DataTable columns={["IsAnomaly", ...Object.keys(r.labelled_profile["0"] ?? {})]} rows={Object.entries(r.labelled_profile).map(([k, row]) => [k === "1" ? "1 (anomaly)" : "0 (normal)", ...Object.values(row).map((v) => num(v, 2))])} />
              <div className="mt-4 space-y-3">
                <Note>{r.autoencoder}</Note>
                <Note tone="warn">{r.note}</Note>
              </div>
            </Card>
          </>
        );
      }}
    </ExperimentShell>
  );
}

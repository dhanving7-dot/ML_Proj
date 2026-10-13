import { motion } from "motion/react";
import { ArrowDown, Bot, Boxes, BrainCircuit, CircleCheck, CircleDashed, Database, Gauge, GitBranch, Layers, MonitorSmartphone, Network, Radar, RefreshCw, Server, ShieldCheck, Sparkles, TriangleAlert, Workflow } from "lucide-react";
import { useRef } from "react";
import { api } from "../api/endpoints";
import type { Lifecycle, LifecycleStage } from "../api/types";
import { PageHeader, Section } from "../components/layout/PageHeader";
import { Button } from "../components/ui/Button";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Chip, StatusPill } from "../components/ui/Status";
import { EmptyState, ErrorState, LoadingBlock } from "../components/ui/States";
import { useApi } from "../hooks/useApi";
import { usePrefersReducedMotion } from "../hooks/useReducedMotion";
import { cn, dateTime, ms, num } from "../lib/format";
import { Reveal } from "../motion/Reveal";
import { gsap, useGSAP } from "../motion/gsap";

const ICONS: Record<string, typeof Database> = {
  ingestion: Database,
  validation: ShieldCheck,
  features: Layers,
  training: BrainCircuit,
  evaluation: Gauge,
  registration: Boxes,
  serving: Server,
  monitoring: Radar,
  retraining: RefreshCw,
};

export default function LifecyclePage() {
  const q = useApi((s) => api.lifecycle({ signal: s }), [], { intervalMs: 30_000 });
  return (
    <>
      <PageHeader
        crumb="Lifecycle"
        title="The end-to-end ML lifecycle"
        description="Raw data → validation → features → training → evaluation → registration → serving → monitoring → retraining. Every status below is read from the backend."
      />
      <Section className="space-y-16">
        {q.error && !q.data ? <ErrorState error={q.error} onRetry={q.refetch} /> : !q.data ? <LoadingBlock rows={8} /> : <Timeline stages={q.data.stages} />}
        <PredictionPath id={q.data?.last_prediction_request_id ?? null} loaded={!!q.data} />
        <Paradigms />
        {q.data && <Tooling t={q.data.tooling} labRuns={q.data.lab_runs} />}
      </Section>
    </>
  );
}

function stageTone(s: string) {
  return s === "complete" ? "good" : s === "attention" ? "warning" : "neutral";
}

function Timeline({ stages }: { stages: LifecycleStage[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();
  useGSAP(
    () => {
      if (reduced) return;
      gsap.fromTo("[data-rail]", { scaleY: 0 }, { scaleY: 1, ease: "none", scrollTrigger: { trigger: ref.current, start: "top 70%", end: "bottom 70%", scrub: 0.4 } });
    },
    { scope: ref, dependencies: [reduced] },
  );
  return (
    <div ref={ref} className="relative">
      <span className="absolute left-[23px] top-2 bottom-2 w-px bg-white/10" aria-hidden="true" />
      <span data-rail className="absolute left-[23px] top-2 bottom-2 w-px origin-top bg-gradient-to-b from-violet via-blue to-cyan" aria-hidden="true" />
      <ol className="space-y-5">
        {stages.map((s, i) => {
          const Icon = ICONS[s.key] ?? Workflow;
          const facts = Object.entries(s.facts).filter(([, v]) => v != null && !(Array.isArray(v) && v.length === 0) && typeof v !== "object");
          const lists = Object.entries(s.facts).filter(([, v]) => Array.isArray(v) && v.length > 0) as [string, string[]][];
          return (
            <Reveal as="li" key={s.key} variant="left" className="relative pl-16">
              <span className={cn("absolute left-0 top-1 grid size-12 place-items-center rounded-full border bg-surface", s.status === "complete" ? "border-good/50 text-good-text" : s.status === "attention" ? "border-warning/50 text-warning" : "border-line-strong text-ink-3")}>
                <Icon className="size-5" aria-hidden="true" />
              </span>
              <Panel className="!p-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="text-lg font-semibold text-ink">
                    <span className="mr-2 font-mono text-xs text-ink-3">{String(i + 1).padStart(2, "0")}</span>
                    {s.title}
                  </h2>
                  <StatusPill tone={stageTone(s.status)}>{s.status}</StatusPill>
                </div>
                <p className="mt-2 text-sm text-ink-2">{s.detail}</p>
                {(facts.length > 0 || lists.length > 0) && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {facts.map(([k, v]) => (
                      <Chip key={k}>
                        {k.replace(/_/g, " ")}: {typeof v === "string" && v.includes("T") && v.endsWith("Z") ? dateTime(v) : String(v).slice(0, 40)}
                      </Chip>
                    ))}
                    {lists.map(([k, v]) => (
                      <Chip key={k}>
                        {k.replace(/_/g, " ")}: {v.join(", ")}
                      </Chip>
                    ))}
                  </div>
                )}
              </Panel>
            </Reveal>
          );
        })}
      </ol>
    </div>
  );
}

function PredictionPath({ id, loaded }: { id: string | null; loaded: boolean }) {
  const q = useApi((s) => api.prediction(id as string, { signal: s }), [id], { enabled: !!id });
  const t = q.data;
  const steps = t
    ? [
        { icon: MonitorSmartphone, title: "Frontend request", detail: `${t.source} request with ${Object.keys(t.inputs).length} fields sent to POST /api/predict`, ms: null },
        { icon: ShieldCheck, title: "FastAPI validation", detail: "Pydantic model generated from the feature contract: types, ranges, enums, no extra fields", ms: null },
        { icon: Layers, title: "Preprocessing", detail: `Engineered SGPA features, imputation (${t.imputed_fields.length ? t.imputed_fields.join(", ") : "nothing missing"}), scaling, encoding → ${t.feature_vector.length} values`, ms: t.timings.preprocessing_ms },
        { icon: BrainCircuit, title: "Model inference", detail: `Model ${t.model_version}: P(placed) = ${num(t.probability, 4)} vs threshold ${num(t.threshold, 2)} → ${t.label}`, ms: t.timings.inference_ms },
        { icon: Network, title: "Response", detail: `Request ID ${t.request_id.slice(0, 12)}… returned at ${dateTime(t.created_at)}`, ms: t.timings.total_ms },
        { icon: Database, title: "Database persistence", detail: "Inputs, feature vector, probability and timings written to the predictions table", ms: t.timings.persistence_ms },
        { icon: Radar, title: "Monitoring", detail: `Latency logged; inputs join the PSI/KL drift window for ${t.model_version}${t.actual_outcome != null ? "; ground truth recorded" : ""}`, ms: null },
      ]
    : [];
  return (
    <section aria-labelledby="path-title">
      <PanelHeader eyebrow="M1 · serving path" title={<span id="path-title">The journey of the latest real prediction</span>} description="Read from the stored record of the most recent prediction; timings are the server's own measurements." />
      {!loaded || (id && q.loading) ? (
        <LoadingBlock rows={6} />
      ) : !id ? (
        <EmptyState title="No prediction has been served yet" action={<Button to="/predict" size="sm">Make a prediction</Button>} />
      ) : q.error ? (
        <ErrorState error={q.error} onRetry={q.refetch} />
      ) : (
        <ol className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((s, i) => {
            const Icon = s.icon;
            return (
              <motion.li key={s.title} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-40px" }} transition={{ delay: i * 0.07, duration: 0.6 }} className={cn(i === 6 && "lg:col-span-1")}>
                <Panel className="h-full !p-4">
                  <div className="flex items-center justify-between">
                    <span className="grid size-8 place-items-center rounded-lg bg-white/[0.04] text-cyan">
                      <Icon className="size-4" aria-hidden="true" />
                    </span>
                    {s.ms != null && <span className="font-mono text-xs text-ink-2">{ms(s.ms)}</span>}
                  </div>
                  <p className="mt-3 text-sm font-semibold text-ink">
                    {i + 1}. {s.title}
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-ink-2">{s.detail}</p>
                </Panel>
              </motion.li>
            );
          })}
          <li className="flex items-center justify-center text-xs text-ink-3 lg:col-span-1">
            <ArrowDown className="mr-1 size-3.5" aria-hidden="true" /> request {t?.request_id.slice(0, 8)}…
          </li>
        </ol>
      )}
    </section>
  );
}

function Paradigms() {
  const items = [
    {
      icon: CircleCheck,
      title: "Supervised learning",
      status: "Implemented",
      tone: "good" as const,
      body: "Learns a mapping from inputs to a known label. Placement prediction is binary classification (PlacementStatus); the salary experiments in M2/M3 are regression on Salary Package.",
    },
    {
      icon: Sparkles,
      title: "Unsupervised learning",
      status: "Implemented in the lab",
      tone: "good" as const,
      body: "Finds structure without labels: K-means, hierarchical clustering, DBSCAN, PCA, t-SNE/UMAP and Isolation Forest (M4). Used for exploration, not to make placement decisions.",
    },
    {
      icon: Bot,
      title: "Reinforcement learning",
      status: "Conceptual only",
      tone: "neutral" as const,
      body: "An agent learns a policy by acting in an environment and receiving rewards (e.g. which practice activity to recommend next). There is no sequential decision loop or reward signal in this dataset, so RL is not implemented.",
    },
  ];
  const beyond = [
    { title: "Deep learning", body: "Neural networks learn feature representations instead of hand-engineered ones; an autoencoder is the deep-learning counterpart of the anomaly detectors in M4." },
    { title: "Generative models", body: "Model the data distribution itself (p(x)) rather than p(label | x); the same evaluation, drift and serving practices apply to them." },
    { title: "Large language models", body: "Very large generative models trained on text. They are covered in separate subjects; the MLOps ideas here (versioning, monitoring, skew) carry over directly." },
  ];
  return (
    <section aria-labelledby="paradigms-title" className="space-y-6">
      <PanelHeader eyebrow="Concepts" title={<span id="paradigms-title">Learning paradigms in this project</span>} />
      <Reveal stagger="[data-p]" className="grid gap-4 md:grid-cols-3">
        {items.map((it) => {
          const Icon = it.icon;
          return (
            <div key={it.title} data-p>
              <Panel className="h-full">
                <Icon className="size-5 text-cyan" aria-hidden="true" />
                <h3 className="mt-3 text-lg font-semibold text-ink">{it.title}</h3>
                <div className="mt-2">
                  <StatusPill tone={it.tone}>{it.status}</StatusPill>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-ink-2">{it.body}</p>
              </Panel>
            </div>
          );
        })}
      </Reveal>
      <Panel>
        <p className="mb-4 text-sm font-medium text-ink">How this connects to other subjects (not implemented here)</p>
        <dl className="grid gap-4 md:grid-cols-3">
          {beyond.map((b) => (
            <div key={b.title}>
              <dt className="text-sm font-semibold text-ink">{b.title}</dt>
              <dd className="mt-1 text-sm text-ink-2">{b.body}</dd>
            </div>
          ))}
        </dl>
      </Panel>
    </section>
  );
}

function Tooling({ t, labRuns }: { t: Lifecycle["tooling"]; labRuns: number }) {
  const dvcOk = t.dvc.repo_initialised && t.dvc.dataset_tracked && t.dvc.pipeline_defined;
  const check = (ok: boolean, label: string) => (
    <li className="flex items-center gap-2 text-sm text-ink-2">
      {ok ? <CircleCheck className="size-4 text-good-text" aria-label="yes" /> : <CircleDashed className="size-4 text-ink-3" aria-label="no" />} {label}
    </li>
  );
  return (
    <section aria-labelledby="tooling-title" className="space-y-6">
      <PanelHeader eyebrow="MLOps tooling" title={<span id="tooling-title">Experiment tracking & data versioning</span>} />
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel>
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-ink">
              <Sparkles className="size-4 text-cyan" aria-hidden="true" /> MLflow
            </h3>
            <StatusPill tone={t.mlflow.installed_version && t.mlflow.enabled ? (t.mlflow.runs_logged ? "good" : "warning") : "neutral"}>
              {t.mlflow.installed_version ? `v${t.mlflow.installed_version}` : "not installed"}
            </StatusPill>
          </div>
          <ul className="mt-4 space-y-1.5">
            {check(!!t.mlflow.installed_version, "Installed")}
            {check(t.mlflow.enabled, "Tracking enabled for training runs")}
            {check(t.mlflow.runs_logged > 0, `${t.mlflow.runs_logged} training run(s) logged with params, metrics and the model package`)}
          </ul>
          {t.mlflow.last_error && (
            <p className="mt-3 flex gap-2 text-xs text-warning">
              <TriangleAlert className="size-3.5 shrink-0" aria-hidden="true" /> Last tracking error: {t.mlflow.last_error}
            </p>
          )}
          <p className="mt-4 text-xs text-ink-3">
            Store: {t.mlflow.tracking_uri}. Browse with <code className="font-mono text-ink-2">mlflow ui --backend-store-uri sqlite:///mlflow.db</code> from the project root.
          </p>
        </Panel>
        <Panel>
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-ink">
              <GitBranch className="size-4 text-cyan" aria-hidden="true" /> DVC
            </h3>
            <StatusPill tone={dvcOk ? "good" : "neutral"}>{t.dvc.installed_version ? `v${t.dvc.installed_version}` : "not installed"}</StatusPill>
          </div>
          <ul className="mt-4 space-y-1.5">
            {check(t.dvc.repo_initialised, "Repository initialised (.dvc/)")}
            {check(t.dvc.dataset_tracked, "Dataset tracked (data/raw/placement_50k.csv.dvc)")}
            {check(t.dvc.pipeline_defined, "Pipeline defined (dvc.yaml: validate → train)")}
            {check(t.dvc.lock_file, "Pipeline reproduced at least once (dvc.lock)")}
          </ul>
          <p className="mt-4 text-xs text-ink-3">
            Reproduce with <code className="font-mono text-ink-2">dvc repro</code>; inspect metrics with <code className="font-mono text-ink-2">dvc metrics show</code>. {labRuns} lab experiment runs are logged in the database.
          </p>
        </Panel>
      </div>
    </section>
  );
}

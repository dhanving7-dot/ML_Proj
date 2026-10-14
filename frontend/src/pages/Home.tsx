import { ArrowRight, Binary, Database, Gauge, Layers, Radar, ShieldCheck } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { api } from "../api/endpoints";
import type { Dashboard, Lifecycle, SchemaResponse } from "../api/types";
import { COMPONENT } from "../components/charts/theme";
import { Hero } from "../components/home/Hero";
import { Button } from "../components/ui/Button";
import { Panel } from "../components/ui/Panel";
import { StatusPill } from "../components/ui/Status";
import { useApi } from "../hooks/useApi";
import { usePrefersReducedMotion } from "../hooks/useReducedMotion";
import { cn, int, num } from "../lib/format";
import { Magnetic } from "../motion/Magnetic";
import { Reveal } from "../motion/Reveal";
import { SplitWords } from "../motion/SplitWords";
import { gsap, ScrollTrigger, useGSAP } from "../motion/gsap";

/** Sets html[data-ambient] while a section is in view so the background lighting shifts with the story. */
function useAmbient(ref: React.RefObject<HTMLElement | null>, tone: string) {
  useGSAP(
    () => {
      if (!ref.current) return;
      ScrollTrigger.create({
        trigger: ref.current,
        start: "top 55%",
        end: "bottom 45%",
        onToggle: (self) => {
          if (self.isActive) document.documentElement.dataset.ambient = tone;
        },
      });
    },
    { scope: ref },
  );
}

export default function Home() {
  const dash = useApi((s) => api.dashboard({ signal: s }), []);
  const life = useApi((s) => api.lifecycle({ signal: s }), []);
  const schema = useApi((s) => api.schema({ signal: s }), []);
  const datasets = useApi((s) => api.datasets({ signal: s }), []);
  const rows = datasets.data?.items.find((d) => d.source === "default")?.n_rows;

  useEffect(() => () => void delete document.documentElement.dataset.ambient, []);

  return (
    <>
      <Hero dashboard={dash.data} datasetRows={rows} />
      <PredictionStory schema={schema.data} dashboard={dash.data} />
      <LifecycleBand lifecycle={life.data} />
      <LabModules />
      <HonestPerformance dashboard={dash.data} />
      <ClosingCta />
    </>
  );
}

/* ------------------------------------------------------------------ sticky storytelling */

function PredictionStory({ schema, dashboard }: { schema?: SchemaResponse; dashboard?: Dashboard }) {
  const ref = useRef<HTMLElement>(null);
  const [active, setActive] = useState(0);
  const reduced = usePrefersReducedMotion();
  useAmbient(ref, "blue");
  const nFields = schema?.request_fields.length;
  const m = dashboard?.model;

  const steps = [
    {
      icon: ShieldCheck,
      title: "Validate the request",
      body: `FastAPI checks the ${nFields ?? "30"} input fields against the shared feature contract: finite numbers, non-negative Time and Amount, no unknown fields. Renamed or missing fields are rejected before they reach a model.`,
    },
    {
      icon: Binary,
      title: "Engineer features",
      body: "Time becomes a position on a 24-hour cycle (sine and cosine) and Amount becomes log(1 + Amount). The same code runs at training and serving time; V1–V28 stay anonymous PCA components.",
    },
    {
      icon: Layers,
      title: "Preprocess with training parameters",
      body: "Standard scaling reuses the means and deviations learned on the training split. Nothing is re-fitted in production, and values outside the training range are flagged.",
    },
    {
      icon: Gauge,
      title: "Score with two models, decide once",
      body: m
        ? `Logistic Regression gives a fraud probability, the Isolation Forest an anomaly percentile. Both become rarities and are fused with w = ${num(m.policy.fusion_weight_lr, 2)}; REVIEW starts at ${num(m.policy.review_threshold, 3)} and HIGH_RISK at ${num(m.policy.high_risk_threshold, 3)}, all tuned on validation data only.`
        : "Logistic Regression gives a fraud probability, the Isolation Forest an anomaly percentile; a validated fusion turns them into LOW_RISK, REVIEW or HIGH_RISK.",
    },
    {
      icon: Database,
      title: "Persist the evidence",
      body: "Inputs, the exact feature vector, all three scores, the decision, the policy in force and the explanation are stored for every transaction.",
    },
    {
      icon: Radar,
      title: "Monitor what happens next",
      body: "Latency and errors are logged, inputs and scores join the drift window (PSI / KL), and confirmed outcomes from investigators enable live precision and recall tracking.",
    },
  ];

  useGSAP(
    () => {
      const blocks = gsap.utils.toArray<HTMLElement>("[data-step]");
      blocks.forEach((el, i) =>
        ScrollTrigger.create({ trigger: el, start: "top 60%", end: "bottom 60%", onToggle: (self) => self.isActive && setActive(i) }),
      );
      if (!reduced) {
        gsap.fromTo("[data-progress]", { scaleY: 0 }, { scaleY: 1, ease: "none", scrollTrigger: { trigger: "[data-steps]", start: "top 60%", end: "bottom 60%", scrub: 0.4 } });
      }
    },
    { scope: ref, dependencies: [reduced] },
  );

  return (
    <section id="story" ref={ref} className="relative mx-auto max-w-7xl px-4 py-24 sm:px-6 sm:py-32" aria-labelledby="story-title">
      <div className="max-w-3xl">
        <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">Anatomy of a risk assessment</p>
        <SplitWords as="h2" text="One request, six transparent steps." className="block text-4xl font-semibold leading-tight text-ink sm:text-6xl" />
      </div>
      <div className="mt-16 grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:gap-20">
        {/* Sticky visual (CSS sticky, so native scrolling is never hijacked) */}
        <div className="hidden lg:block">
          <div className="sticky top-28">
            <Panel luminous className="relative overflow-hidden">
              <ol className="relative space-y-1" aria-label="Risk assessment pipeline">
                <span className="absolute left-[19px] top-4 bottom-4 w-px bg-white/10" aria-hidden="true" />
                <span data-progress className="absolute left-[19px] top-4 bottom-4 w-px origin-top bg-gradient-to-b from-violet via-blue to-cyan" aria-hidden="true" style={reduced ? undefined : { transform: "scaleY(0)" }} />
                {steps.map((s, i) => {
                  const Icon = s.icon;
                  const on = i === active;
                  const done = i < active;
                  return (
                    <li key={s.title} className="relative flex items-center gap-4 rounded-xl px-1 py-3" aria-current={on ? "step" : undefined}>
                      <span
                        className={cn(
                          "relative z-10 grid size-10 shrink-0 place-items-center rounded-full border transition-all duration-500",
                          on ? "scale-110 border-cyan bg-cyan/15 text-cyan shadow-[0_0_28px_-4px_rgb(34_211_238/0.7)]" : done ? "border-violet/60 bg-violet/15 text-violet-soft" : "border-line-strong bg-surface-2 text-ink-3",
                        )}
                      >
                        <Icon className="size-4" aria-hidden="true" />
                      </span>
                      <span className={cn("text-[15px] font-medium transition-colors duration-500", on ? "text-ink" : "text-ink-3")}>{s.title}</span>
                    </li>
                  );
                })}
              </ol>
            </Panel>
          </div>
        </div>
        <div data-steps className="space-y-6 lg:space-y-[28vh] lg:py-[8vh]">
          {steps.map((s, i) => {
            const Icon = s.icon;
            return (
              <Reveal key={s.title} variant="blur">
                <article data-step className={cn("rounded-2xl border p-6 transition-colors duration-500 sm:p-8", i === active ? "border-line-strong bg-white/[0.03]" : "border-line bg-transparent")}>
                  <div className="mb-4 flex items-center gap-3">
                    <span className="font-mono text-xs text-ink-3">{String(i + 1).padStart(2, "0")}</span>
                    <Icon className="size-4 text-cyan lg:hidden" aria-hidden="true" />
                  </div>
                  <h3 className="text-2xl font-semibold text-ink sm:text-3xl">{s.title}</h3>
                  <p className="mt-3 text-base leading-relaxed text-ink-2">{s.body}</p>
                </article>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ lifecycle */

function LifecycleBand({ lifecycle }: { lifecycle?: Lifecycle }) {
  const ref = useRef<HTMLElement>(null);
  const reduced = usePrefersReducedMotion();
  useAmbient(ref, "cyan");
  useGSAP(
    () => {
      if (reduced) return;
      gsap.fromTo("[data-line]", { scaleX: 0 }, { scaleX: 1, ease: "none", scrollTrigger: { trigger: ref.current, start: "top 75%", end: "center 55%", scrub: 0.5 } });
    },
    { scope: ref, dependencies: [reduced, !!lifecycle] },
  );
  const tone = (s: string) => (s === "complete" ? "good" : s === "attention" ? "warning" : "neutral");
  return (
    <section ref={ref} className="relative py-24 sm:py-32" aria-labelledby="life-title">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-3xl">
            <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">M1 · the ML lifecycle</p>
            <SplitWords as="h2" text="From raw data to retraining, live." className="block text-4xl font-semibold leading-tight text-ink sm:text-6xl" />
            <Reveal variant="blur">
              <p className="mt-4 text-ink-2">Each stage below reports its real status from the backend: what has run, what is pending and what needs attention.</p>
            </Reveal>
          </div>
          <Button to="/lifecycle" variant="secondary" icon={<ArrowRight className="size-4" aria-hidden="true" />}>
            Open lifecycle
          </Button>
        </div>
        <div className="relative mt-14">
          <div className="absolute left-0 right-0 top-[22px] hidden h-px bg-white/10 lg:block" aria-hidden="true" />
          <div data-line className="absolute left-0 right-0 top-[22px] hidden h-px origin-left bg-gradient-to-r from-violet via-blue to-cyan lg:block" aria-hidden="true" />
          <Reveal as="ol" stagger="[data-node]" className="grid gap-4 sm:grid-cols-3 lg:grid-cols-9">
            {(lifecycle?.stages ?? Array.from({ length: 9 }, () => null)).map((s, i) => (
              <li key={s?.key ?? i} data-node className="relative">
                <span className="relative z-10 mb-4 hidden size-11 place-items-center rounded-full border border-line-strong bg-surface font-mono text-xs text-ink-2 lg:grid">{i + 1}</span>
                <div className="rounded-xl border border-line bg-surface/70 p-3.5">
                  <p className="text-sm font-medium text-ink">{s?.title ?? <span className="skeleton block h-4 w-20" />}</p>
                  <div className="mt-2">{s ? <StatusPill tone={tone(s.status)}>{s.status}</StatusPill> : <span className="skeleton block h-4 w-16" />}</div>
                </div>
              </li>
            ))}
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ lab modules */

const MODULES = [
  { id: "m2", tag: "M2", title: "Linear models & preprocessing", topics: ["Normal equation", "Gradient descent", "Ridge / Lasso / Elastic Net", "Class-weighted logistic regression", "Scaling, encoding, imputation"] },
  { id: "m3", tag: "M3", title: "Trees & ensembles", topics: ["Gini & entropy", "Cost-complexity pruning", "Random forest & OOB", "AdaBoost, XGBoost, LightGBM", "Permutation & SHAP"] },
  { id: "m4", tag: "M4", title: "Unsupervised learning", topics: ["K-means & silhouette", "Hierarchical clustering", "DBSCAN", "PCA, t-SNE, UMAP", "Isolation Forest vs One-Class SVM"] },
  { id: "m5", tag: "M5", title: "Evaluation & calibration", topics: ["Stratified & nested CV", "ROC & PR curves", "Platt & isotonic", "Grid / random / Bayesian search", "McNemar & bootstrap"] },
  { id: "m6", tag: "M6", title: "Production ML engineering", topics: ["Training-serving skew", "Random vs time-ordered split", "ONNX parity", "Pickle vs joblib", "PSI & KL drift"] },
];

function TiltCard({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();
  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (reduced || e.pointerType !== "mouse" || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    ref.current.style.transform = `perspective(900px) rotateX(${(-py * 5).toFixed(2)}deg) rotateY(${(px * 6).toFixed(2)}deg)`;
  };
  const reset = () => {
    if (ref.current) ref.current.style.transform = "";
  };
  return (
    <div ref={ref} onPointerMove={onMove} onPointerLeave={reset} className={cn("transition-transform duration-300 ease-out will-change-transform", className)}>
      {children}
    </div>
  );
}

function LabModules() {
  const ref = useRef<HTMLElement>(null);
  useAmbient(ref, "magenta");
  return (
    <section ref={ref} className="mx-auto max-w-7xl px-4 py-24 sm:px-6 sm:py-32" aria-labelledby="lab-title">
      <div className="max-w-3xl">
        <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">The laboratory</p>
        <SplitWords as="h2" text="Every algorithm, run on real transactions." className="block text-4xl font-semibold leading-tight text-ink sm:text-6xl" />
        <Reveal variant="blur">
          <p className="mt-4 text-ink-2">27 experiments fit on the training split and score on the validation split, with PR-AUC instead of accuracy. Results, parameters, timestamps and split sizes are shown with every run.</p>
        </Reveal>
      </div>
      <Reveal stagger="[data-card]" className="mt-14 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {MODULES.map((m, i) => (
          <div key={m.id} data-card className={i === 0 ? "lg:col-span-1" : ""}>
            <TiltCard className="h-full">
              <Link to={`/lab?module=${m.id}`} className="block h-full rounded-2xl focus-visible:outline-offset-4">
                <Panel luminous={i === 4} className="group flex h-full flex-col">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm text-cyan">{m.tag}</span>
                    <ArrowRight className="size-4 text-ink-3 transition-transform duration-300 group-hover:translate-x-1 group-hover:text-ink" aria-hidden="true" />
                  </div>
                  <h3 className="mt-6 text-2xl font-semibold text-ink">{m.title}</h3>
                  <ul className="mt-5 flex flex-wrap gap-1.5">
                    {m.topics.map((t) => (
                      <li key={t} className="rounded-full border border-line px-2.5 py-1 text-xs text-ink-2">
                        {t}
                      </li>
                    ))}
                  </ul>
                </Panel>
              </Link>
            </TiltCard>
          </div>
        ))}
        <div data-card>
          <Panel className="flex h-full flex-col justify-between bg-gradient-to-br from-violet/10 to-cyan/5">
            <p className="text-sm leading-relaxed text-ink-2">
              Supervised learning (Logistic Regression) and unsupervised anomaly detection (Isolation Forest) both run in production. Clustering is used for
              exploration only, and reinforcement learning is covered conceptually on the lifecycle page.
            </p>
            <Button to="/lab" variant="secondary" className="mt-6 self-start" icon={<ArrowRight className="size-4" aria-hidden="true" />}>
              Enter the lab
            </Button>
          </Panel>
        </div>
      </Reveal>
    </section>
  );
}

/* ------------------------------------------------------------------ honest performance */

function HonestPerformance({ dashboard }: { dashboard?: Dashboard }) {
  const ref = useRef<HTMLElement>(null);
  const reduced = usePrefersReducedMotion();
  useAmbient(ref, "violet");
  const m = dashboard?.model;
  const rows = m
    ? [
        { key: "lr", label: "Logistic Regression", ap: m.test_metrics.logistic_regression.average_precision ?? 0, color: COMPONENT.logistic_regression },
        { key: "if", label: "Isolation Forest", ap: m.test_metrics.isolation_forest.average_precision ?? 0, color: COMPONENT.isolation_forest },
        { key: "hy", label: "Hybrid", ap: m.test_metrics.hybrid_high_risk.average_precision ?? 0, color: COMPONENT.hybrid },
      ]
    : [];
  const baseline = m ? m.test_fraud_rows / m.test_support : null;

  useGSAP(
    () => {
      if (reduced || !rows.length) return;
      gsap.from("[data-bar]", { scaleX: 0, transformOrigin: "0% 50%", stagger: 0.12, duration: 1.2, scrollTrigger: { trigger: ref.current, start: "top 70%" } });
    },
    { scope: ref, dependencies: [reduced, rows.length] },
  );

  return (
    <section ref={ref} className="mx-auto max-w-7xl px-4 py-24 sm:px-6 sm:py-32" aria-labelledby="perf-title">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-center">
        <div>
          <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">Measured honestly</p>
          <SplitWords as="h2" text="Precision and recall, not accuracy." className="block text-4xl font-semibold leading-tight text-ink sm:text-5xl" />
          <Reveal variant="blur">
            <p className="mt-5 leading-relaxed text-ink-2">
              Only about 0.17 % of transactions are fraud, so a model that never flags anything is 99.8 % accurate. Every result here is average precision (PR-AUC), precision and
              recall on a held-out test split that was scored exactly once.
              {m && (
                <>
                  {" "}
                  On this data, validation tuning chose a fusion weight of <strong className="text-ink">{num(m.policy.fusion_weight_lr, 2)}</strong> on Logistic Regression
                  {m.policy.fusion_weight_lr >= 0.999
                    ? ": the anomaly signal did not add ranking value on the labelled frauds, so decisions follow Logistic Regression while the anomaly score is still computed, stored and monitored."
                    : "."}
                </>
              )}
            </p>
          </Reveal>
        </div>
        <Panel>
          {!dashboard ? (
            <div className="space-y-4">{Array.from({ length: 3 }).map((_, i) => <span key={i} className="skeleton block h-6" />)}</div>
          ) : !rows.length ? (
            <p className="text-sm text-ink-2">No trained model yet. Train one from the Models page to see the comparison.</p>
          ) : (
            <figure>
              <figcaption className="mb-4 text-sm font-medium text-ink">Held-out test PR-AUC by component</figcaption>
              <ul className="space-y-4">
                {rows.map((r) => (
                  <li key={r.key} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_4rem] items-center gap-3 text-sm">
                    <span className="truncate text-ink-2">{r.label}</span>
                    <span className="relative h-6 rounded bg-white/[0.04]" title={`${r.label}: ${num(r.ap, 4)}`}>
                      <span data-bar className="absolute inset-y-0 left-0 rounded" style={{ width: `${r.ap * 100}%`, background: r.color }} aria-hidden="true" />
                    </span>
                    <span className="text-right font-mono text-xs text-ink tabular">{num(r.ap, 3)}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-4 text-xs text-ink-3">
                Axis from 0 to 1. A random ranking would score about {num(baseline, 4)} (the test fraud rate). {int(m?.test_support)} test transactions, {m?.test_fraud_rows} of them fraud.
              </p>
            </figure>
          )}
        </Panel>
      </div>
    </section>
  );
}

function ClosingCta() {
  return (
    <section className="mx-auto max-w-7xl px-4 pb-8 sm:px-6">
      <Reveal variant="scale">
        <div className="border-luminous relative overflow-hidden rounded-3xl bg-gradient-to-br from-violet/15 via-surface to-cyan/10 px-6 py-16 text-center sm:px-12 sm:py-20">
          <div className="grid-lines pointer-events-none absolute inset-0 opacity-60" aria-hidden="true" />
          <h2 className="relative mx-auto max-w-3xl text-4xl font-semibold leading-tight text-ink sm:text-5xl">Ready to see both models reason about a transaction?</h2>
          <p className="relative mx-auto mt-4 max-w-xl text-ink-2">Replay a real transaction from the held-out test set, fraud or legitimate, or enter your own values. The assessment comes straight from the active model.</p>
          <div className="relative mt-8 flex flex-wrap justify-center gap-3">
            <Magnetic>
              <Button to="/analyze" size="lg" icon={<ArrowRight className="size-4" aria-hidden="true" />}>
                Analyze a transaction
              </Button>
            </Magnetic>
            <Magnetic>
              <Button to="/overview" size="lg" variant="secondary">
                View the overview
              </Button>
            </Magnetic>
          </div>
        </div>
      </Reveal>
    </section>
  );
}

import { ArrowRight, Binary, Database, Gauge, Layers, Radar, ShieldCheck } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { api } from "../api/endpoints";
import type { Dashboard, Lifecycle, SchemaResponse } from "../api/types";
import { Hero } from "../components/home/Hero";
import { Button } from "../components/ui/Button";
import { Panel } from "../components/ui/Panel";
import { StatusPill } from "../components/ui/Status";
import { useApi } from "../hooks/useApi";
import { usePrefersReducedMotion } from "../hooks/useReducedMotion";
import { cn, num } from "../lib/format";
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
      body: `FastAPI checks the ${nFields ?? "contract"} input fields against the shared feature contract: types, ranges and allowed categories. Unknown or renamed fields are rejected before they reach the model.`,
    },
    {
      icon: Binary,
      title: "Engineer features",
      body: `The SGPA trend and SGPA volatility (${schema?.engineered_features.join(", ") ?? "engineered features"}) are computed by the same code at training and serving time.`,
    },
    {
      icon: Layers,
      title: "Preprocess with training parameters",
      body: "Median imputation, standard scaling and one-hot / ordinal encoding reuse the values learned on the training split. Nothing is re-fitted in production.",
    },
    {
      icon: Gauge,
      title: "Estimate a probability",
      body: m
        ? `The active ${m.label} model (${m.version}) returns P(placed). A decision threshold of ${num(m.threshold, 2)} (tuned on validation data only) turns it into a label.`
        : "The active model returns P(placed) and a decision threshold turns it into a label.",
    },
    {
      icon: Database,
      title: "Persist the evidence",
      body: "Inputs, the exact feature vector, the probability, model version, request ID and timings are stored for every prediction.",
    },
    {
      icon: Radar,
      title: "Monitor what happens next",
      body: "Latency and errors are logged, inputs join the drift window (PSI / KL vs. training data), and ground-truth outcomes enable live accuracy tracking.",
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
        <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">Anatomy of a prediction</p>
        <SplitWords as="h2" text="One request, six transparent steps." className="block text-4xl font-semibold leading-tight text-ink sm:text-6xl" />
      </div>
      <div className="mt-16 grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:gap-20">
        {/* Sticky visual (CSS sticky, so native scrolling is never hijacked) */}
        <div className="hidden lg:block">
          <div className="sticky top-28">
            <Panel luminous className="relative overflow-hidden">
              <ol className="relative space-y-1" aria-label="Prediction pipeline">
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
  { id: "m2", tag: "M2", title: "Linear models & preprocessing", topics: ["Normal equation", "Gradient descent", "Ridge / Lasso / Elastic Net", "Logistic regression", "Scaling, encoding, imputation"] },
  { id: "m3", tag: "M3", title: "Trees & ensembles", topics: ["Gini & entropy", "Cost-complexity pruning", "Random forest & OOB", "AdaBoost, XGBoost, LightGBM", "Permutation & SHAP"] },
  { id: "m4", tag: "M4", title: "Unsupervised learning", topics: ["K-means & silhouette", "Hierarchical clustering", "DBSCAN", "PCA, t-SNE, UMAP", "Isolation Forest"] },
  { id: "m5", tag: "M5", title: "Evaluation & calibration", topics: ["Stratified & nested CV", "ROC & PR curves", "Platt & isotonic", "Grid / random / Bayesian search", "McNemar & bootstrap"] },
  { id: "m6", tag: "M6", title: "Production ML engineering", topics: ["Training-serving skew", "Point-in-time correctness", "ONNX parity", "Pickle vs joblib", "PSI & KL drift"] },
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
        <SplitWords as="h2" text="Every algorithm, run on real student data." className="block text-4xl font-semibold leading-tight text-ink sm:text-6xl" />
        <Reveal variant="blur">
          <p className="mt-4 text-ink-2">27 experiments fit on the training split and score on the validation split. Results, parameters, timestamps and split sizes are shown with every run.</p>
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
              Supervised learning powers placement prediction. Clustering and anomaly detection are used for exploration only, and reinforcement learning is
              covered conceptually on the lifecycle page.
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
  const rows = [...(dashboard?.model?.cv_results ?? [])].sort((a, b) => b.roc_auc - a.roc_auc);
  const lo = rows.length ? Math.min(...rows.map((r) => r.roc_auc - r.roc_auc_std)) - 0.002 : 0;
  const hi = rows.length ? Math.max(...rows.map((r) => r.roc_auc + r.roc_auc_std)) + 0.002 : 1;
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const best = rows[0];
  const second = rows[1];
  const gap = best && second ? best.roc_auc - second.roc_auc : null;

  useGSAP(
    () => {
      if (reduced || !rows.length) return;
      gsap.from("[data-whisker]", { scaleX: 0, transformOrigin: "50% 50%", stagger: 0.08, duration: 1.2, scrollTrigger: { trigger: ref.current, start: "top 70%" } });
      gsap.from("[data-dot]", { scale: 0, stagger: 0.08, duration: 0.8, delay: 0.3, ease: "back.out(2)", scrollTrigger: { trigger: ref.current, start: "top 70%" } });
    },
    { scope: ref, dependencies: [reduced, rows.length] },
  );

  return (
    <section ref={ref} className="mx-auto max-w-7xl px-4 py-24 sm:px-6 sm:py-32" aria-labelledby="perf-title">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-center">
        <div>
          <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cyan">Model selection, measured honestly</p>
          <SplitWords as="h2" text="No winner is declared without the evidence." className="block text-4xl font-semibold leading-tight text-ink sm:text-5xl" />
          <Reveal variant="blur">
            <p className="mt-5 leading-relaxed text-ink-2">
              Candidates are compared with stratified 5-fold cross-validation on the training split only. The chart shows the mean ROC-AUC ± one standard deviation
              across folds.
              {gap != null && best && second && (
                <>
                  {" "}
                  {best.label} was selected with the highest mean, but its lead over {second.label} is <strong className="text-ink">{num(gap, 4)}</strong>,{" "}
                  {gap < best.roc_auc_std ? "smaller than one fold standard deviation, so the top candidates are effectively tied." : "larger than one fold standard deviation."}
                </>
              )}
            </p>
          </Reveal>
        </div>
        <Panel>
          {!dashboard ? (
            <div className="space-y-4">{Array.from({ length: 5 }).map((_, i) => <span key={i} className="skeleton block h-6" />)}</div>
          ) : !rows.length ? (
            <p className="text-sm text-ink-2">No trained model yet. Train one from the Models page to see the comparison.</p>
          ) : (
            <figure>
              <figcaption className="mb-4 text-sm font-medium text-ink">Cross-validated ROC-AUC by candidate (training split)</figcaption>
              <ul className="space-y-4">
                {rows.map((r) => (
                  <li key={r.model} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_4.5rem] items-center gap-3 text-sm">
                    <span className={cn("truncate", r === best ? "font-medium text-ink" : "text-ink-2")}>{r.label}</span>
                    <span className="relative h-6" title={`${r.label}: ${num(r.roc_auc, 4)} ± ${num(r.roc_auc_std, 4)}`}>
                      <span className="absolute inset-x-0 top-1/2 h-px bg-white/[0.06]" aria-hidden="true" />
                      <span data-whisker className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded bg-[#3987e5]/60" style={{ left: `${x(r.roc_auc - r.roc_auc_std)}%`, width: `${x(r.roc_auc + r.roc_auc_std) - x(r.roc_auc - r.roc_auc_std)}%` }} aria-hidden="true" />
                      <span data-dot className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface bg-[#3987e5]" style={{ left: `${x(r.roc_auc)}%` }} aria-hidden="true" />
                    </span>
                    <span className="text-right font-mono text-xs text-ink tabular">{num(r.roc_auc, 4)}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-3 grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_4.5rem] gap-3 font-mono text-[10px] text-ink-3">
                <span />
                <span className="flex justify-between tabular">
                  <span>{num(lo, 3)}</span>
                  <span>{num((lo + hi) / 2, 3)}</span>
                  <span>{num(hi, 3)}</span>
                </span>
                <span />
              </div>
              <p className="mt-3 text-xs text-ink-3">
                The axis is zoomed to {num(lo, 3)}–{num(hi, 3)} so the overlap between candidates is visible; it does not start at zero.
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
          <h2 className="relative mx-auto max-w-3xl text-4xl font-semibold leading-tight text-ink sm:text-5xl">Ready to see the model reason about a student?</h2>
          <p className="relative mx-auto mt-4 max-w-xl text-ink-2">Load a real student from the held-out test set or enter your own profile. The estimate comes straight from the active model.</p>
          <div className="relative mt-8 flex flex-wrap justify-center gap-3">
            <Magnetic>
              <Button to="/predict" size="lg" icon={<ArrowRight className="size-4" aria-hidden="true" />}>
                Make a prediction
              </Button>
            </Magnetic>
            <Magnetic>
              <Button to="/dashboard" size="lg" variant="secondary">
                View the dashboard
              </Button>
            </Magnetic>
          </div>
        </div>
      </Reveal>
    </section>
  );
}

import { ArrowRight, FlaskConical } from "lucide-react";
import { useRef } from "react";
import type { Dashboard } from "../../api/types";
import { usePrefersReducedMotion } from "../../hooks/useReducedMotion";
import { int, num } from "../../lib/format";
import { Magnetic } from "../../motion/Magnetic";
import { gsap, useGSAP } from "../../motion/gsap";
import { HealthBadge } from "../layout/HealthBadge";
import { Button } from "../ui/Button";
import { ParticleField } from "./ParticleField";

const TITLE = "PlacementPredict";
const SUB = "Will a student be placed? An end-to-end machine-learning system that shows its work.";

export function Hero({ dashboard, datasetRows }: { dashboard?: Dashboard; datasetRows?: number }) {
  const root = useRef<HTMLElement>(null);
  const reduced = usePrefersReducedMotion();

  useGSAP(
    () => {
      if (reduced) return;
      const tl = gsap.timeline({ defaults: { ease: "expo.out" } });
      tl.from("[data-hero=badge]", { y: 16, autoAlpha: 0, duration: 0.8 })
        .from("[data-char]", { yPercent: 115, rotate: 8, autoAlpha: 0, stagger: 0.035, duration: 1.3 }, "-=0.45")
        .from("[data-hero=sub] [data-word]", { yPercent: 110, autoAlpha: 0, stagger: 0.035, duration: 1 }, "-=0.95")
        .from("[data-hero=copy]", { y: 18, autoAlpha: 0, filter: "blur(8px)", duration: 1, clearProps: "filter" }, "-=0.75")
        .from("[data-hero=cta] > *", { y: 18, autoAlpha: 0, stagger: 0.1, duration: 0.9 }, "-=0.7")
        .from("[data-hero=stats] > *", { y: 14, autoAlpha: 0, stagger: 0.07, duration: 0.8 }, "-=0.6")
        .from("[data-hero=scroll]", { autoAlpha: 0, duration: 0.6 }, "-=0.2")
        .from("[data-geo]", { scale: 0.85, autoAlpha: 0, stagger: 0.12, duration: 1.8 }, 0.1);

      const st = { trigger: root.current, start: "top top", end: "bottom top", scrub: true };
      gsap.to("[data-depth='1']", { yPercent: -25, ease: "none", scrollTrigger: st });
      gsap.to("[data-depth='2']", { yPercent: -55, ease: "none", scrollTrigger: st });
      gsap.to("[data-depth='3']", { yPercent: -90, rotate: 25, ease: "none", scrollTrigger: st });
      gsap.to("[data-hero=content]", { yPercent: -12, autoAlpha: 0.15, ease: "none", scrollTrigger: { ...st, start: "top top+=1", end: "bottom top+=120" } });
    },
    { scope: root, dependencies: [reduced] },
  );

  const m = dashboard?.model;
  const stats = [
    { label: "Active model", value: m ? `${m.version} · ${m.label}` : "—" },
    { label: "Held-out ROC-AUC", value: m ? `${num(m.test_metrics.roc_auc, 3)} on ${int(m.test_support)} students` : "—" },
    { label: "Training records", value: datasetRows ? int(datasetRows) : "—" },
    { label: "Predictions served", value: dashboard ? int(dashboard.predictions.total) : "—" },
  ];

  return (
    <section ref={root} className="relative isolate flex min-h-[calc(100dvh-4rem)] items-center overflow-hidden" aria-labelledby="hero-title">
      {/* Atmosphere layers */}
      <div className="grid-lines pointer-events-none absolute inset-0 -z-10" aria-hidden="true" />
      <ParticleField className="pointer-events-none absolute inset-0 -z-10 h-full w-full opacity-80" />
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
        <div data-depth="1" data-geo className="absolute -left-32 top-10 size-[34rem] rounded-full bg-violet/20 blur-[120px]" />
        <div data-depth="2" data-geo className="absolute -right-24 top-1/3 size-[26rem] rounded-full bg-cyan/12 blur-[110px]" />
        <div data-depth="2" data-geo className="absolute right-[12%] top-[16%] hidden size-56 rounded-full border border-white/10 lg:block">
          <div className="absolute inset-6 rounded-full border border-violet/30" />
          <div className="absolute inset-0 animate-[spin_40s_linear_infinite] rounded-full bg-[conic-gradient(from_0deg,transparent_0deg,rgb(139_92_246/0.35)_60deg,transparent_120deg)] [mask:radial-gradient(circle,transparent_62%,#000_63%)]" />
        </div>
        <div data-depth="3" data-geo className="absolute bottom-[18%] left-[8%] hidden size-24 rotate-12 rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-sm md:block" />
        <div data-depth="3" data-geo className="absolute bottom-[30%] right-[22%] hidden size-3 rounded-full bg-cyan shadow-[0_0_24px_6px_rgb(34_211_238/0.5)] md:block" />
        <div data-depth="1" data-geo className="absolute left-[46%] top-[12%] hidden size-2 rounded-full bg-violet-soft shadow-[0_0_20px_4px_rgb(167_139_250/0.6)] md:block" />
      </div>

      <div data-hero="content" className="mx-auto w-full max-w-7xl px-4 py-16 sm:px-6">
        <div data-hero="badge" className="mb-8">
          <HealthBadge detailed />
        </div>
        <h1 id="hero-title" aria-label={TITLE} className="font-display text-[clamp(3rem,11vw,9.5rem)] font-semibold leading-[0.9] tracking-[-0.045em]">
          {["Placement", "Predict"].map((word, w) => (
            <span key={word} className="inline-block whitespace-nowrap">
              {word.split("").map((c, j) => {
                const i = w === 0 ? j : 9 + j;
                return (
                  <span key={i} aria-hidden="true" className="inline-block overflow-hidden pb-[0.06em] align-bottom">
                    {/* Each glyph shows its slice of one continuous gradient, so the word reads as a single sweep. */}
                    <span
                      data-char
                      className="inline-block bg-clip-text text-transparent"
                      style={{
                        backgroundImage: "linear-gradient(100deg, #ede9fe 0%, #a78bfa 22%, #6d8bff 48%, #22d3ee 74%, #c4b5fd 100%)",
                        backgroundSize: `${TITLE.length * 100}% 100%`,
                        backgroundPosition: `${(i / (TITLE.length - 1)) * 100}% 0`,
                      }}
                    >
                      {c}
                    </span>
                  </span>
                );
              })}
            </span>
          ))}
        </h1>
        <p data-hero="sub" aria-label={SUB} className="mt-6 max-w-3xl font-display text-[clamp(1.35rem,2.6vw,2.15rem)] font-medium leading-[1.2] text-ink">
          {SUB.split(" ").map((w, i) => (
            <span key={i} aria-hidden="true" className="inline-block overflow-hidden align-bottom">
              <span data-word className="inline-block">
                {w}&nbsp;
              </span>
            </span>
          ))}
        </p>
        <p data-hero="copy" className="mt-6 max-w-2xl text-base leading-relaxed text-ink-2 sm:text-lg">
          A supervised classifier estimates the probability that a student is placed from academic record, assessments and experience. Every step,
          from data validation to drift monitoring, runs for real and is shown in the open.
        </p>
        <div data-hero="cta" className="mt-10 flex flex-wrap items-center gap-3">
          <Magnetic>
            <Button to="/predict" size="lg" icon={<ArrowRight className="size-4 transition-transform duration-300 group-hover:translate-x-0.5" aria-hidden="true" />}>
              Predict a placement
            </Button>
          </Magnetic>
          <Magnetic>
            <Button to="/lab" size="lg" variant="secondary" icon={<FlaskConical className="size-4" aria-hidden="true" />}>
              Explore the ML lab
            </Button>
          </Magnetic>
        </div>
        <dl data-hero="stats" className="mt-14 grid max-w-4xl grid-cols-2 gap-x-8 gap-y-5 border-t border-line pt-6 md:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label} className="min-w-0">
              <dt className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-3">{s.label}</dt>
              <dd className="mt-1 truncate text-sm font-medium text-ink" title={s.value}>
                {s.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <a
        href="#story"
        data-hero="scroll"
        className="absolute bottom-6 left-1/2 hidden -translate-x-1/2 flex-col items-center gap-2 text-[10px] uppercase tracking-[0.25em] text-ink-3 hover:text-ink sm:flex"
        aria-label="Scroll to how a prediction is made"
      >
        <span className="relative h-9 w-5 rounded-full border border-line-strong">
          <span className="absolute left-1/2 top-1.5 h-2 w-0.5 -translate-x-1/2 rounded-full bg-ink-2 animate-scroll-dot" />
        </span>
        Scroll
      </a>
    </section>
  );
}

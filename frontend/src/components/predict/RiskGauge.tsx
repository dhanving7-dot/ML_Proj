import { animate, useInView } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { Decision } from "../../api/types";
import { usePrefersReducedMotion } from "../../hooks/useReducedMotion";
import { decisionLabel } from "../../lib/format";
import { DECISION_COLOR } from "../charts/theme";

/** Semicircular gauge of the hybrid risk score with the REVIEW and HIGH_RISK thresholds. Animates to the exact API value. */
export function RiskGauge({ score, review, high, decision }: { score: number; review: number; high: number; decision: Decision }) {
  const ref = useRef<SVGSVGElement>(null);
  const inView = useInView(ref, { once: true });
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduced ? score : 0);

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setShown(score);
      return;
    }
    const controls = animate(0, score, { duration: 1.4, ease: [0.16, 1, 0.3, 1], onUpdate: setShown, onComplete: () => setShown(score) });
    return () => controls.stop();
  }, [score, inView, reduced]);

  const r = 80;
  const c = Math.PI * r;
  const color = DECISION_COLOR[decision];
  const tick = (t: number, label: string) => {
    const a = Math.PI * (1 - t);
    return (
      <g key={label}>
        <line x1={100 + Math.cos(a) * (r + 7)} y1={100 - Math.sin(a) * (r + 7)} x2={100 + Math.cos(a) * (r - 16)} y2={100 - Math.sin(a) * (r - 16)} stroke="#eef0fb" strokeWidth="1.6" />
        <text x={100 + Math.cos(a) * (r - 27)} y={100 - Math.sin(a) * (r - 27) + 3} textAnchor="middle" fill="#a9aecb" fontSize="7.5">
          {label}
        </text>
      </g>
    );
  };

  return (
    <svg
      ref={ref}
      viewBox="0 0 200 118"
      className="w-full max-w-[280px]"
      role="img"
      aria-label={`Hybrid risk score ${score.toFixed(3)}; REVIEW from ${review.toFixed(3)}, HIGH_RISK from ${high.toFixed(3)}; decision ${decisionLabel(decision)}`}
    >
      <defs>
        <linearGradient id="risk-g" x1="0" x2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.55" />
          <stop offset="1" stopColor={color} />
        </linearGradient>
      </defs>
      <path d={`M 20 100 A ${r} ${r} 0 0 1 180 100`} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="12" strokeLinecap="round" />
      <path d={`M 20 100 A ${r} ${r} 0 0 1 180 100`} fill="none" stroke="url(#risk-g)" strokeWidth="12" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - shown)} />
      {review < high && tick(review, "review")}
      {tick(high, "high")}
      <text x="100" y="92" textAnchor="middle" fill="#eef0fb" fontSize="30" fontWeight="600" fontFamily="Space Grotesk Variable, system-ui">
        {shown.toFixed(3)}
      </text>
      <text x="100" y="110" textAnchor="middle" fill="#7d83a7" fontSize="9" letterSpacing="1.5">
        HYBRID RISK SCORE
      </text>
    </svg>
  );
}

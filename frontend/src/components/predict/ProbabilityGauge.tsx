import { animate, useInView } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../../hooks/useReducedMotion";
import { OUTCOME } from "../charts/theme";

/** Semicircular gauge of P(placed) with the decision-threshold tick. Animates to the exact API value. */
export function ProbabilityGauge({ probability, threshold }: { probability: number; threshold: number }) {
  const ref = useRef<SVGSVGElement>(null);
  const inView = useInView(ref, { once: true });
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(reduced ? probability : 0);

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setShown(probability);
      return;
    }
    const controls = animate(0, probability, { duration: 1.4, ease: [0.16, 1, 0.3, 1], onUpdate: setShown, onComplete: () => setShown(probability) });
    return () => controls.stop();
  }, [probability, inView, reduced]);

  const r = 80;
  const c = Math.PI * r;
  const placed = probability >= threshold;
  const color = placed ? OUTCOME.placed : OUTCOME.notPlaced;
  const angle = Math.PI * (1 - threshold);
  const tx = 100 + Math.cos(angle) * r;
  const ty = 100 - Math.sin(angle) * r;

  return (
    <svg ref={ref} viewBox="0 0 200 118" className="w-full max-w-[280px]" role="img" aria-label={`Estimated probability of placement ${(probability * 100).toFixed(1)} percent; decision threshold ${(threshold * 100).toFixed(0)} percent`}>
      <defs>
        <linearGradient id="gauge-g" x1="0" x2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.55" />
          <stop offset="1" stopColor={color} />
        </linearGradient>
      </defs>
      <path d={`M 20 100 A ${r} ${r} 0 0 1 180 100`} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="12" strokeLinecap="round" />
      <path d={`M 20 100 A ${r} ${r} 0 0 1 180 100`} fill="none" stroke="url(#gauge-g)" strokeWidth="12" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - shown)} />
      <line x1={tx} y1={ty} x2={100 + Math.cos(angle) * (r - 18)} y2={100 - Math.sin(angle) * (r - 18)} stroke="#eef0fb" strokeWidth="2" />
      <text x={100 + Math.cos(angle) * (r - 28)} y={100 - Math.sin(angle) * (r - 28) + 3} textAnchor="middle" fill="#a9aecb" fontSize="8">
        {`t=${threshold.toFixed(2)}`}
      </text>
      <text x="100" y="92" textAnchor="middle" fill="#eef0fb" fontSize="30" fontWeight="600" fontFamily="Space Grotesk Variable, system-ui">
        {(shown * 100).toFixed(1)}%
      </text>
      <text x="100" y="110" textAnchor="middle" fill="#7d83a7" fontSize="9" letterSpacing="1.5">
        P(PLACED)
      </text>
    </svg>
  );
}

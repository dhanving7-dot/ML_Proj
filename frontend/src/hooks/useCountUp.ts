import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "./useReducedMotion";

/**
 * Animate from the previously displayed value to ``target`` and always finish on the exact target.
 * Does not restart from zero when the same value is re-rendered; respects reduced motion.
 */
export function useCountUp(target: number | null | undefined, durationMs = 1100, start = true): number | null {
  const reduced = usePrefersReducedMotion();
  const [value, setValue] = useState<number | null>(target == null ? null : reduced ? target : 0);
  const fromRef = useRef<number>(0);
  const lastTarget = useRef<number | null | undefined>(undefined);

  useEffect(() => {
    if (target == null) {
      setValue(null);
      return;
    }
    if (!start) return;
    if (reduced || lastTarget.current === target) {
      lastTarget.current = target;
      setValue(target);
      return;
    }
    const from = lastTarget.current == null ? 0 : (fromRef.current ?? 0);
    lastTarget.current = target;
    let raf = 0;
    // The first frame's timestamp is the start time: rAF timestamps can precede a
    // performance.now() taken before scheduling, which would make progress negative.
    let t0: number | null = null;
    const tick = (now: number) => {
      if (t0 === null) t0 = now;
      const p = Math.min(1, Math.max(0, (now - t0) / durationMs));
      const eased = 1 - Math.pow(1 - p, 4);
      const v = from + (target - from) * eased;
      fromRef.current = v;
      if (p < 1) {
        setValue(v);
        raf = requestAnimationFrame(tick);
      } else {
        fromRef.current = target;
        setValue(target); // land exactly on the returned value
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs, start, reduced]);

  return value;
}

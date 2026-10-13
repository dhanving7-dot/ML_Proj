import { useRef, type ElementType, type ReactNode } from "react";
import { usePrefersReducedMotion } from "../hooks/useReducedMotion";
import { gsap, useGSAP } from "./gsap";

type Variant = "up" | "fade" | "scale" | "blur" | "clip" | "left" | "right";

const FROM: Record<Variant, gsap.TweenVars> = {
  up: { y: 36, autoAlpha: 0 },
  fade: { autoAlpha: 0 },
  scale: { scale: 0.94, autoAlpha: 0 },
  blur: { y: 18, autoAlpha: 0, filter: "blur(10px)" },
  clip: { clipPath: "inset(0 0 100% 0)", y: 24 },
  left: { x: -40, autoAlpha: 0 },
  right: { x: 40, autoAlpha: 0 },
};

interface RevealProps {
  children: ReactNode;
  as?: ElementType;
  className?: string;
  variant?: Variant;
  /** CSS selector (inside this element) whose matches animate with a stagger instead of the wrapper. */
  stagger?: string;
  delay?: number;
  duration?: number;
  start?: string;
}

/**
 * Scroll-triggered reveal. Content is visible without JavaScript and with reduced motion;
 * once revealed it stays visible when scrolling back up or revisiting. All tweens and
 * ScrollTriggers are reverted automatically on unmount (route changes).
 */
export function Reveal({ children, as: Tag = "div", className, variant = "up", stagger, delay = 0, duration = 1.1, start = "top 86%" }: RevealProps) {
  const ref = useRef<HTMLElement>(null);
  const reduced = usePrefersReducedMotion();

  useGSAP(
    () => {
      if (reduced || !ref.current) return;
      const targets = stagger ? ref.current.querySelectorAll(stagger) : ref.current;
      if (stagger && (targets as NodeListOf<Element>).length === 0) return;
      gsap.from(targets, {
        ...FROM[variant],
        duration,
        delay,
        stagger: stagger ? 0.08 : 0,
        clearProps: "filter,clipPath,transform",
        scrollTrigger: { trigger: ref.current, start, toggleActions: "play none none none" },
      });
    },
    { scope: ref, dependencies: [reduced, variant, stagger] },
  );

  return (
    <Tag ref={ref} className={className}>
      {children}
    </Tag>
  );
}

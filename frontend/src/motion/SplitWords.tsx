import { useRef, type ElementType } from "react";
import { usePrefersReducedMotion } from "../hooks/useReducedMotion";
import { gsap, useGSAP } from "./gsap";

interface SplitWordsProps {
  text: string;
  as?: ElementType;
  className?: string;
  wordClassName?: string;
  /** "mount" animates immediately (hero); "scroll" waits until the heading enters the viewport. */
  trigger?: "mount" | "scroll";
  delay?: number;
  stagger?: number;
}

/**
 * Word-by-word reveal. Screen readers get the full sentence via aria-label;
 * the animated word spans are aria-hidden.
 */
export function SplitWords({ text, as: Tag = "span", className, wordClassName, trigger = "scroll", delay = 0, stagger = 0.06 }: SplitWordsProps) {
  const ref = useRef<HTMLElement>(null);
  const reduced = usePrefersReducedMotion();
  const words = text.split(" ");

  useGSAP(
    () => {
      if (reduced || !ref.current) return;
      const inner = ref.current.querySelectorAll("[data-word]");
      gsap.from(inner, {
        yPercent: 110,
        rotate: 4,
        autoAlpha: 0,
        duration: 1.1,
        stagger,
        delay,
        clearProps: "transform",
        ...(trigger === "scroll" ? { scrollTrigger: { trigger: ref.current, start: "top 88%", toggleActions: "play none none none" } } : {}),
      });
    },
    { scope: ref, dependencies: [reduced, text] },
  );

  return (
    <Tag ref={ref} className={className} aria-label={text}>
      {words.map((w, i) => (
        <span key={`${w}-${i}`} aria-hidden="true" className="inline-block overflow-hidden pb-[0.08em] align-bottom">
          <span data-word className={`inline-block will-change-transform ${wordClassName ?? ""}`}>
            {w}
            {i < words.length - 1 ? " " : ""}
          </span>
        </span>
      ))}
    </Tag>
  );
}

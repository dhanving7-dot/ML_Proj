import { motion, useScroll, useSpring } from "motion/react";

/** Thin gradient bar showing reading progress for the current page. */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 140, damping: 28, restDelta: 0.001 });
  return (
    <motion.div
      aria-hidden="true"
      className="fixed inset-x-0 top-0 z-[60] h-[2px] origin-left bg-gradient-to-r from-violet via-blue to-cyan"
      style={{ scaleX }}
    />
  );
}

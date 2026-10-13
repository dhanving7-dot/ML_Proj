import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { Suspense, useEffect } from "react";
import { useLocation, useOutlet } from "react-router";
import { ScrollTrigger } from "../../motion/gsap";
import { ScrollProgress } from "../../motion/ScrollProgress";
import { LoadingBlock } from "../ui/States";
import { Footer } from "./Footer";
import { Navbar, NAV } from "./Navbar";

function PageFallback() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
      <LoadingBlock rows={5} label="Loading page" />
    </div>
  );
}

export function Layout() {
  const location = useLocation();
  const outlet = useOutlet();

  useEffect(() => {
    const page = NAV.find((n) => n.to === location.pathname)?.label;
    document.title = page && page !== "Home" ? `${page} · PlacementPredict` : "PlacementPredict · ML placement prediction";
  }, [location.pathname]);

  return (
    <MotionConfig reducedMotion="user">
      <div className="ambient" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />
      <ScrollProgress />
      <div className="relative z-10 flex min-h-dvh flex-col">
        <Navbar />
        <AnimatePresence
          mode="wait"
          initial={false}
          onExitComplete={() => {
            window.scrollTo({ top: 0 });
            ScrollTrigger.refresh();
          }}
        >
          <motion.main
            id="main"
            key={location.pathname}
            className="flex-1"
            initial={{ opacity: 0, y: 14, filter: "blur(6px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)", transitionEnd: { filter: "none" } }}
            exit={{ opacity: 0, y: -8, filter: "blur(4px)" }}
            transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            onAnimationComplete={() => ScrollTrigger.refresh()}
          >
            <Suspense fallback={<PageFallback />}>{outlet}</Suspense>
          </motion.main>
        </AnimatePresence>
        <Footer />
      </div>
    </MotionConfig>
  );
}

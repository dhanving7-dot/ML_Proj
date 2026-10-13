import { AnimatePresence, LayoutGroup, motion } from "motion/react";
import { Menu, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { cn } from "../../lib/format";
import { HealthBadge } from "./HealthBadge";

export const NAV = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/predict", label: "Predict" },
  { to: "/lab", label: "ML Lab" },
  { to: "/lifecycle", label: "Lifecycle" },
  { to: "/monitoring", label: "Monitoring" },
  { to: "/models", label: "Models" },
  { to: "/data", label: "Data" },
] as const;

function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2.5 rounded-lg" aria-label="PlacementPredict home">
      <svg viewBox="0 0 32 32" className="size-8" aria-hidden="true">
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#8b5cf6" />
            <stop offset="1" stopColor="#22d3ee" />
          </linearGradient>
        </defs>
        <rect width="32" height="32" rx="9" fill="#0e1120" stroke="rgba(255,255,255,0.1)" />
        <path d="M8 22 L14 14 L18 18 L24 9" fill="none" stroke="url(#lg)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="24" cy="9" r="2.4" fill="#22d3ee" />
      </svg>
      <span className="font-display text-[17px] font-semibold tracking-tight text-ink">
        Placement<span className="text-gradient">Predict</span>
      </span>
    </Link>
  );
}

export function Navbar() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => setOpen(false), [location.pathname]);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className={cn("sticky top-0 z-50 transition-[background,border-color] duration-500", scrolled || open ? "border-b border-line bg-bg/85 backdrop-blur-xl" : "border-b border-transparent")}>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-md focus:bg-violet focus:px-3 focus:py-2 focus:text-white">
        Skip to content
      </a>
      <nav aria-label="Primary" className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <Logo />
        <LayoutGroup id="nav">
          <ul className="hidden items-center gap-0.5 xl:flex">
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink to={n.to} end={n.to === "/"} className={({ isActive }) => cn("relative block rounded-full px-3.5 py-2 text-[13.5px] font-medium transition-colors duration-300", isActive ? "text-ink" : "text-ink-3 hover:text-ink")}>
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <motion.span layoutId="nav-active" className="absolute inset-0 rounded-full border border-line-strong bg-white/[0.06]" transition={{ type: "spring", stiffness: 400, damping: 34 }} />
                      )}
                      <span className="relative">{n.label}</span>
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </LayoutGroup>
        <div className="flex items-center gap-2">
          <HealthBadge className="hidden sm:inline-flex" />
          <button
            type="button"
            className="grid size-10 place-items-center rounded-full border border-line text-ink-2 hover:text-ink xl:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-4" /> : <Menu className="size-4" />}
          </button>
        </div>
      </nav>
      <AnimatePresence>
        {open && (
          <motion.div
            id="mobile-nav"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden border-t border-line xl:hidden"
          >
            <ul className="mx-auto grid max-w-7xl gap-1 px-4 py-4 sm:grid-cols-2 sm:px-6">
              {NAV.map((n, i) => (
                <motion.li key={n.to} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.03 * i }}>
                  <NavLink
                    to={n.to}
                    end={n.to === "/"}
                    className={({ isActive }) => cn("block rounded-xl px-4 py-3 text-base font-medium", isActive ? "bg-white/[0.06] text-ink" : "text-ink-2 hover:bg-white/[0.04] hover:text-ink")}
                  >
                    {n.label}
                  </NavLink>
                </motion.li>
              ))}
              <li className="pt-2 sm:hidden">
                <HealthBadge detailed />
              </li>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}

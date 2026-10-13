import { LayoutGroup, motion } from "motion/react";
import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import { cn } from "../../lib/format";

export interface TabItem<K extends string> {
  key: K;
  label: ReactNode;
  hint?: string;
}

/** Accessible tablist (arrow keys, Home/End) with an animated active indicator. */
export function Tabs<K extends string>({
  items,
  value,
  onChange,
  className,
  size = "md",
  ariaLabel,
  idPrefix,
}: {
  items: TabItem<K>[];
  value: K;
  onChange: (k: K) => void;
  className?: string;
  size?: "sm" | "md";
  ariaLabel: string;
  idPrefix?: string;
}) {
  const uid = useId();
  const group = idPrefix ?? uid;
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKey = (e: KeyboardEvent, i: number) => {
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % items.length;
    if (e.key === "ArrowLeft") next = (i - 1 + items.length) % items.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = items.length - 1;
    if (next >= 0) {
      e.preventDefault();
      onChange(items[next].key);
      refs.current[next]?.focus();
    }
  };

  return (
    <LayoutGroup id={group}>
      <div role="tablist" aria-label={ariaLabel} className={cn("flex max-w-full gap-1 overflow-x-auto rounded-full border border-line bg-white/[0.02] p-1 [scrollbar-width:none]", className)}>
        {items.map((it, i) => {
          const active = it.key === value;
          return (
            <button
              key={it.key}
              ref={(el) => {
                refs.current[i] = el;
              }}
              role="tab"
              id={`${group}-tab-${it.key}`}
              aria-selected={active}
              aria-controls={`${group}-panel`}
              tabIndex={active ? 0 : -1}
              title={it.hint}
              onClick={() => onChange(it.key)}
              onKeyDown={(e) => onKey(e, i)}
              className={cn(
                "relative shrink-0 rounded-full font-medium transition-colors duration-300",
                size === "sm" ? "px-3 py-1.5 text-[13px]" : "px-4 py-2 text-sm",
                active ? "text-white" : "text-ink-2 hover:text-ink",
              )}
            >
              {active && (
                <motion.span
                  layoutId={`${group}-indicator`}
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-violet/80 to-blue/80 shadow-[0_6px_20px_-8px_rgb(139_92_246/0.8)]"
                  transition={{ type: "spring", stiffness: 380, damping: 32 }}
                />
              )}
              <span className="relative z-10">{it.label}</span>
            </button>
          );
        })}
      </div>
    </LayoutGroup>
  );
}

/** Panel paired with a <Tabs idPrefix=...>; animates content on tab change. */
export function TabPanel({ idPrefix, value, children, className }: { idPrefix: string; value: string; children: ReactNode; className?: string }) {
  return (
    <motion.div
      key={value}
      role="tabpanel"
      id={`${idPrefix}-panel`}
      aria-labelledby={`${idPrefix}-tab-${value}`}
      initial={{ opacity: 0, y: 12, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)", transitionEnd: { filter: "none" } }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

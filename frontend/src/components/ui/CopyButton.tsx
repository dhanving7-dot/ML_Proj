import { AnimatePresence, motion } from "motion/react";
import { Check, Copy } from "lucide-react";
import { useEffect, useRef, useState } from "react";

/** Copies text to the clipboard with visible feedback; reports failure instead of pretending. */
export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [state, setState] = useState<"idle" | "done" | "failed">("idle");
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setState("done");
    } catch {
      setState("failed");
    }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState("idle"), 1800);
  };

  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1 rounded-md border border-line px-1.5 py-0.5 text-[11px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
      aria-label={state === "done" ? "Copied" : `${label}: ${text}`}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={state} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.6 }} transition={{ duration: 0.15 }}>
          {state === "done" ? <Check className="size-3 text-good-text" /> : <Copy className="size-3" />}
        </motion.span>
      </AnimatePresence>
      {state === "done" ? "Copied" : state === "failed" ? "Copy failed" : label}
    </button>
  );
}

import { AnimatePresence, motion } from "motion/react";
import { CircleCheck, CircleX, Info, X } from "lucide-react";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import { cn } from "../../lib/format";

type ToastTone = "success" | "error" | "info";
interface ToastItem {
  id: number;
  tone: ToastTone;
  title: string;
  body?: string;
}

const Ctx = createContext<((t: Omit<ToastItem, "id">) => void) | null>(null);

const icons = { success: CircleCheck, error: CircleX, info: Info };
const tones = {
  success: "border-good/40 [&_svg.tone]:text-good-text",
  error: "border-critical/50 [&_svg.tone]:text-critical-text",
  info: "border-cyan/30 [&_svg.tone]:text-cyan",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const next = useRef(1);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const push = useCallback(
    (t: Omit<ToastItem, "id">) => {
      const id = next.current++;
      setItems((xs) => [...xs.slice(-3), { ...t, id }]);
      window.setTimeout(() => dismiss(id), t.tone === "error" ? 8000 : 5000);
    },
    [dismiss],
  );

  return (
    <Ctx.Provider value={push}>
      {children}
      <div aria-live="polite" aria-atomic="false" className="pointer-events-none fixed bottom-4 right-4 z-[80] flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2">
        <AnimatePresence initial={false}>
          {items.map((t) => {
            const Icon = icons[t.tone];
            return (
              <motion.div
                key={t.id}
                layout
                initial={{ opacity: 0, y: 24, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, x: 40, transition: { duration: 0.2 } }}
                transition={{ type: "spring", stiffness: 380, damping: 30 }}
                role={t.tone === "error" ? "alert" : "status"}
                className={cn("glass pointer-events-auto flex items-start gap-3 rounded-xl border p-3.5 shadow-2xl", tones[t.tone])}
              >
                <Icon className="tone mt-0.5 size-4 shrink-0" aria-hidden="true" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-ink">{t.title}</p>
                  {t.body && <p className="mt-0.5 text-xs leading-relaxed text-ink-2">{t.body}</p>}
                </div>
                <button onClick={() => dismiss(t.id)} className="rounded-md p-0.5 text-ink-3 hover:text-ink" aria-label="Dismiss notification">
                  <X className="size-3.5" />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </Ctx.Provider>
  );
}

export function useToast() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}

import type { HTMLAttributes, PointerEvent, ReactNode } from "react";
import { cn } from "../../lib/format";

interface PanelProps extends HTMLAttributes<HTMLElement> {
  as?: "div" | "section" | "article";
  luminous?: boolean;
  padded?: boolean;
  children: ReactNode;
}

function trackPointer(e: PointerEvent<HTMLElement>) {
  const r = e.currentTarget.getBoundingClientRect();
  e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
  e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
}

/** Surface card with a cursor-reactive light; ``luminous`` adds the gradient hairline for featured panels. */
export function Panel({ as: Tag = "div", luminous, padded = true, className, children, onPointerMove, ...rest }: PanelProps) {
  return (
    <Tag
      className={cn(
        "spotlight min-w-0 rounded-2xl border border-line bg-surface/80 shadow-[0_1px_0_rgb(255_255_255/0.04)_inset,0_20px_60px_-30px_rgb(0_0_0/0.8)] transition-[border-color,transform] duration-500 hover:border-line-strong",
        luminous && "border-luminous border-transparent",
        padded && "p-5 sm:p-6",
        className,
      )}
      onPointerMove={(e) => {
        trackPointer(e);
        onPointerMove?.(e);
      }}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export function PanelHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        {eyebrow && <p className="mb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-3">{eyebrow}</p>}
        <h3 className="text-lg font-semibold text-ink">{title}</h3>
        {description && <p className="mt-1 max-w-2xl text-sm leading-relaxed text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

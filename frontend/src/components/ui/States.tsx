import { CircleX, Inbox, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import type { ApiError } from "../../api/client";
import { cn } from "../../lib/format";
import { Button } from "./Button";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} aria-hidden="true" />;
}

export function LoadingBlock({ rows = 3, className, label = "Loading" }: { rows?: number; className?: string; label?: string }) {
  return (
    <div className={cn("space-y-3", className)} role="status" aria-label={label}>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className={cn("h-4", i === 0 ? "w-2/3" : i % 2 ? "w-full" : "w-5/6")} />
      ))}
    </div>
  );
}

export function ErrorState({ error, onRetry, title = "Something went wrong", compact }: { error: ApiError | Error; onRetry?: () => void; title?: string; compact?: boolean }) {
  return (
    <div role="alert" className={cn("flex flex-col items-start gap-3 rounded-xl border border-critical/40 bg-critical/[0.07] text-sm", compact ? "p-3" : "p-5")}>
      <div className="flex items-center gap-2 font-medium text-critical-text">
        <CircleX className="size-4" aria-hidden="true" />
        {title}
      </div>
      <p className="text-ink-2">{error.message}</p>
      {onRetry && (
        <Button size="sm" variant="secondary" onClick={onRetry} icon={<RefreshCw className="size-3.5" aria-hidden="true" />}>
          Try again
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ title, children, action, icon }: { title: string; children?: ReactNode; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line-strong px-6 py-10 text-center">
      <div className="grid size-11 place-items-center rounded-full bg-white/[0.04] text-ink-3">{icon ?? <Inbox className="size-5" aria-hidden="true" />}</div>
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="max-w-md text-sm text-ink-2">{children}</div>}
      {action}
    </div>
  );
}

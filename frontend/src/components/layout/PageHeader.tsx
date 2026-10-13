import type { ReactNode } from "react";
import { Link } from "react-router";
import { Reveal } from "../../motion/Reveal";
import { SplitWords } from "../../motion/SplitWords";

/** Page title block with breadcrumbs, a word-by-word title reveal and optional actions. */
export function PageHeader({ crumb, title, description, actions }: { crumb: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mx-auto max-w-7xl px-4 pb-8 pt-12 sm:px-6 sm:pt-16">
      <nav aria-label="Breadcrumb" className="mb-4 font-mono text-[11px] uppercase tracking-[0.2em] text-ink-3">
        <ol className="flex items-center gap-2">
          <li>
            <Link to="/" className="hover:text-ink">
              PlacementPredict
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li aria-current="page" className="text-ink-2">
            {crumb}
          </li>
        </ol>
      </nav>
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-3xl">
          <SplitWords as="h1" text={title} trigger="mount" className="block text-4xl font-semibold leading-[1.05] text-ink sm:text-5xl lg:text-6xl" />
          {description && (
            <Reveal variant="blur" delay={0.25}>
              <p className="mt-4 text-base leading-relaxed text-ink-2 sm:text-lg">{description}</p>
            </Reveal>
          )}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </div>
    </header>
  );
}

export function Section({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={`mx-auto max-w-7xl px-4 sm:px-6 ${className ?? ""}`}>{children}</section>;
}

import { API_BASE } from "../../api/client";

export function Footer() {
  return (
    <footer className="relative z-10 mt-24 border-t border-line">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-8 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>
          FraudGuard · risk scores come from models trained on a public, anonymised 2-day dataset; they support, not replace, investigation.
        </p>
        <nav aria-label="Developer links" className="flex flex-wrap gap-4">
          <a className="hover:text-ink" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
            API docs (OpenAPI)
          </a>
          <a className="hover:text-ink" href={`${API_BASE}/redoc`} target="_blank" rel="noreferrer">
            ReDoc
          </a>
        </nav>
      </div>
    </footer>
  );
}

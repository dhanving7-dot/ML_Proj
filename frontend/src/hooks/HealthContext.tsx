import { createContext, useContext, type ReactNode } from "react";
import { api } from "../api/endpoints";
import type { Health } from "../api/types";
import { useApi, type QueryState } from "./useApi";

const HealthCtx = createContext<QueryState<Health> | null>(null);

/** One shared health poll (every 15 s) for the whole app: nav status, hero badge, guards. */
export function HealthProvider({ children }: { children: ReactNode }) {
  const state = useApi((signal) => api.health({ signal }), [], { intervalMs: 15_000 });
  return <HealthCtx.Provider value={state}>{children}</HealthCtx.Provider>;
}

export function useHealth(): QueryState<Health> {
  const ctx = useContext(HealthCtx);
  if (!ctx) throw new Error("useHealth must be used inside <HealthProvider>");
  return ctx;
}

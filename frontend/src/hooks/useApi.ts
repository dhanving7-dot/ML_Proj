import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, toApiError } from "../api/client";

export interface QueryState<T> {
  data: T | undefined;
  error: ApiError | undefined;
  /** First load with no data yet (show a skeleton). */
  loading: boolean;
  /** Re-fetching while previous data stays on screen (no skeleton flash). */
  refreshing: boolean;
  refetch: () => void;
}

/**
 * Fetch data with automatic cancellation and stale-response protection.
 * - A newer request aborts the previous one, and a late response can never overwrite newer data.
 * - Previous data is kept while refreshing.
 * - Optional polling via ``intervalMs`` (paused while the tab is hidden).
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  opts: { enabled?: boolean; intervalMs?: number } = {},
): QueryState<T> {
  const { enabled = true, intervalMs } = opts;
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError>();
  const [loading, setLoading] = useState<boolean>(enabled);
  const [refreshing, setRefreshing] = useState(false);
  const seq = useRef(0);
  const ctrl = useRef<AbortController | null>(null);
  const hasData = useRef(false);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(() => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    const id = ++seq.current;
    if (hasData.current) setRefreshing(true);
    else setLoading(true);
    fetcherRef
      .current(c.signal)
      .then((d) => {
        if (id !== seq.current) return;
        hasData.current = true;
        setData(d);
        setError(undefined);
      })
      .catch((e: unknown) => {
        const err = toApiError(e);
        if (id !== seq.current || err.kind === "cancelled") return;
        setError(err);
      })
      .finally(() => {
        if (id !== seq.current) return;
        setLoading(false);
        setRefreshing(false);
      });
  }, []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    run();
    return () => {
      seq.current++;
      ctrl.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, run, ...deps]);

  useEffect(() => {
    if (!enabled || !intervalMs) return;
    const t = window.setInterval(() => {
      if (document.visibilityState === "visible") run();
    }, intervalMs);
    return () => window.clearInterval(t);
  }, [enabled, intervalMs, run]);

  return { data, error, loading, refreshing, refetch: run };
}

export interface ActionState<A extends unknown[], T> {
  run: (...args: A) => Promise<T | undefined>;
  pending: boolean;
  error: ApiError | undefined;
  data: T | undefined;
  reset: () => void;
}

/**
 * Wrap a mutating request. While a call is pending, further calls are ignored
 * (prevents duplicate submissions), and results arriving after unmount are dropped.
 */
export function useAction<A extends unknown[], T>(action: (...args: A) => Promise<T>): ActionState<A, T> {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ApiError>();
  const [data, setData] = useState<T>();
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const actionRef = useRef(action);
  actionRef.current = action;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async (...args: A) => {
    if (inFlight.current) return undefined;
    inFlight.current = true;
    setPending(true);
    setError(undefined);
    try {
      const res = await actionRef.current(...args);
      if (mounted.current) setData(res);
      return res;
    } catch (e) {
      const err = toApiError(e);
      if (mounted.current && err.kind !== "cancelled") setError(err);
      return undefined;
    } finally {
      inFlight.current = false;
      if (mounted.current) setPending(false);
    }
  }, []);

  const reset = useCallback(() => {
    setError(undefined);
    setData(undefined);
  }, []);

  return { run, pending, error, data, reset };
}

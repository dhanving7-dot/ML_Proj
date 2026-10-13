import axios, { AxiosError, type AxiosRequestConfig } from "axios";

/** Base URL: relative "/api" (Vite proxies it in dev/preview); override with VITE_API_URL for other hosts. */
export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export const http = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

export interface FieldError {
  field: string;
  message: string;
}

/** Normalised error shape used everywhere in the UI. */
export class ApiError extends Error {
  status: number | null;
  fieldErrors: FieldError[];
  kind: "network" | "timeout" | "http" | "cancelled";

  constructor(message: string, opts: { status?: number | null; fieldErrors?: FieldError[]; kind: ApiError["kind"] }) {
    super(message);
    this.name = "ApiError";
    this.status = opts.status ?? null;
    this.fieldErrors = opts.fieldErrors ?? [];
    this.kind = opts.kind;
  }
}

interface FastApiValidationItem {
  loc?: (string | number)[];
  msg?: string;
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (axios.isCancel(err)) return new ApiError("Request cancelled", { kind: "cancelled" });
  const ax = err as AxiosError<{ detail?: string | FastApiValidationItem[] }>;
  if (ax?.code === "ECONNABORTED" || ax?.code === "ETIMEDOUT") {
    return new ApiError("The server took too long to respond. Please try again.", { kind: "timeout" });
  }
  if (!ax?.response) {
    return new ApiError("Cannot reach the PlacementPredict API. Is the backend running on port 8000?", { kind: "network" });
  }
  const { status, data } = ax.response;
  const detail = data?.detail;
  if (Array.isArray(detail)) {
    const fieldErrors = detail.map((d) => ({
      field: (d.loc ?? []).filter((p) => p !== "body").join("."),
      message: d.msg ?? "Invalid value",
    }));
    return new ApiError("Some inputs are invalid.", { status, fieldErrors, kind: "http" });
  }
  const fallback =
    status === 503
      ? "No model is currently available for predictions."
      : status >= 500
        ? "The server hit an unexpected error."
        : `Request failed (${status}).`;
  return new ApiError(typeof detail === "string" ? detail : fallback, { status, kind: "http" });
}

export async function get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  try {
    return (await http.get<T>(url, config)).data;
  } catch (e) {
    throw toApiError(e);
  }
}

export async function post<T>(url: string, body?: unknown, config?: AxiosRequestConfig): Promise<T> {
  try {
    return (await http.post<T>(url, body, config)).data;
  } catch (e) {
    throw toApiError(e);
  }
}

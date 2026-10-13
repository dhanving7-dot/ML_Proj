import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { AxiosError, AxiosHeaders } from "axios";
import { describe, expect, it } from "vitest";
import { ApiError, toApiError } from "../api/client";
import { ConfusionMatrix } from "../components/charts/Common";
import { Stat } from "../components/ui/Stat";
import { useApi } from "../hooks/useApi";
import { useCountUp } from "../hooks/useCountUp";
import { usePrefersReducedMotion } from "../hooks/useReducedMotion";
import { int, pct } from "../lib/format";
import { setReducedMotion } from "./setup";

function axiosErr(status: number, data: unknown) {
  const e = new AxiosError("fail", "ERR_BAD_REQUEST");
  e.response = { status, data, statusText: "", headers: {}, config: { headers: new AxiosHeaders() } };
  return e;
}

describe("toApiError", () => {
  it("maps FastAPI 422 details to field errors", () => {
    const err = toApiError(axiosErr(422, { detail: [{ loc: ["body", "CGPA"], msg: "Input should be less than or equal to 10" }] }));
    expect(err.status).toBe(422);
    expect(err.fieldErrors).toEqual([{ field: "CGPA", message: "Input should be less than or equal to 10" }]);
  });
  it("uses the server detail string and a clear network message", () => {
    expect(toApiError(axiosErr(503, { detail: "No active model" })).message).toBe("No active model");
    const net = toApiError(new AxiosError("Network Error", "ERR_NETWORK"));
    expect(net.kind).toBe("network");
    expect(net.message).toMatch(/Cannot reach/);
  });
  it("never exposes a raw 500 body", () => {
    expect(toApiError(axiosErr(500, "<html>Traceback</html>")).message).toBe("The server hit an unexpected error.");
  });
});

describe("useApi", () => {
  it("ignores a stale response that resolves after a newer one", async () => {
    const resolvers: ((v: string) => void)[] = [];
    const fetcher = () => new Promise<string>((res) => resolvers.push(res));
    const { result, rerender } = renderHook(({ dep }) => useApi(fetcher, [dep]), { initialProps: { dep: 1 } });
    rerender({ dep: 2 });
    await act(async () => {
      resolvers[1]("new");
    });
    await act(async () => {
      resolvers[0]("old");
    });
    expect(result.current.data).toBe("new");
  });
  it("surfaces errors as ApiError", async () => {
    const { result } = renderHook(() => useApi(() => Promise.reject(axiosErr(404, { detail: "Not found" })), []));
    await waitFor(() => expect(result.current.error).toBeInstanceOf(ApiError));
    expect(result.current.error?.message).toBe("Not found");
    expect(result.current.loading).toBe(false);
  });
});

describe("animated counters", () => {
  it("land exactly on the API value", async () => {
    const { result } = renderHook(() => useCountUp(311, 50, true));
    await waitFor(() => expect(result.current).toBe(311), { timeout: 2000 });
  });
  it("show the value immediately with reduced motion", () => {
    setReducedMotion(true);
    const { result } = renderHook(() => useCountUp(0.7123, 1000, true));
    expect(result.current).toBe(0.7123);
    setReducedMotion(false);
  });
  it("Stat renders the exact formatted value and a dash for null", async () => {
    const { rerender } = render(<Stat label="Predictions" value={1234} format={int} />);
    await waitFor(() => expect(screen.getByText("1,234")).toBeInTheDocument(), { timeout: 3000 });
    rerender(<Stat label="Rate" value={null} format={pct} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});

describe("reduced motion hook", () => {
  it("reflects the media query", () => {
    setReducedMotion(true);
    expect(renderHook(() => usePrefersReducedMotion()).result.current).toBe(true);
    setReducedMotion(false);
    expect(renderHook(() => usePrefersReducedMotion()).result.current).toBe(false);
  });
});

describe("ConfusionMatrix", () => {
  it("renders every count with an accessible label", () => {
    render(<ConfusionMatrix cm={{ tn: 2165, fp: 407, fn: 187, tp: 4741 }} />);
    expect(screen.getByText("4,741")).toBeInTheDocument();
    expect(screen.getByLabelText(/Actual Placed, predicted Not placed: 187/)).toBeInTheDocument();
  });
});

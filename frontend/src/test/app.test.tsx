import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { FieldSchema, PredictionResponse } from "../api/types";

const fields: FieldSchema[] = [
  { name: "Gender", kind: "nominal", group: "profile", description: "", categories: ["Male", "Female"], nullable: false },
  { name: "CGPA", kind: "numeric", group: "academic", description: "", minimum: 0, maximum: 10, integer: false, nullable: false, unit: null },
  { name: "Internships", kind: "numeric", group: "experience", description: "", minimum: 0, maximum: 20, integer: true, nullable: false, unit: null },
  { name: "CodingTestScore", kind: "numeric", group: "assessment", description: "", minimum: 0, maximum: 100, integer: false, nullable: true, unit: null },
];

const prediction: PredictionResponse = {
  request_id: "abc123def4567890",
  created_at: "2026-10-13T10:00:00Z",
  model_version: "v2",
  model_type: "xgboost",
  prediction: 1,
  label: "Placed",
  probability: 0.8731,
  threshold: 0.45,
  latency_ms: 12.5,
  timings: { preprocessing_ms: 8, inference_ms: 1, persistence_ms: 2, total_ms: 12.5 },
  imputed_fields: ["CodingTestScore"],
  disclaimer: "This is a statistical estimate, not a guarantee.",
};

const api = vi.hoisted(() => ({
  health: vi.fn(),
  schema: vi.fn(),
  datasets: vi.fn(),
  datasetSample: vi.fn(),
  predict: vi.fn(),
  dashboard: vi.fn(),
  lifecycle: vi.fn(),
  prediction: vi.fn(),
  recordOutcome: vi.fn(),
}));
vi.mock("../api/endpoints", () => ({ api }));

import { routes } from "../App";
import { ToastProvider } from "../components/ui/Toast";
import { HealthProvider } from "../hooks/HealthContext";

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <HealthProvider>
      <ToastProvider>
        <RouterProvider router={router} />
      </ToastProvider>
    </HealthProvider>,
  );
  return router;
}

beforeEach(() => {
  vi.resetAllMocks();
  api.health.mockResolvedValue({ status: "ok", api_version: "1.0.0", time: "", uptime_s: 1, database: "ok", model_loaded: true, model_version: "v2", model_type: "xgboost", consistency_check: { passed: true }, message: null });
  api.schema.mockResolvedValue({ request_fields: fields, model_features: [], feature_set: "full", engineered_features: ["sgpa_trend"], excluded_columns: {}, target: "PlacementStatus", class_labels: {}, available_classifiers: [] });
  api.datasets.mockResolvedValue({ items: [] });
  api.dashboard.mockResolvedValue(undefined);
  api.lifecycle.mockResolvedValue(undefined);
});

describe("routing", () => {
  it("renders the 404 page for unknown paths and keeps the navigation", async () => {
    renderAt("/no-such-page");
    expect(await screen.findByText("This page does not exist")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
  });

  it("marks the current route as active in the navigation", async () => {
    renderAt("/predict");
    const nav = await screen.findByRole("navigation", { name: "Primary" });
    await waitFor(() => expect(within(nav).getByRole("link", { name: "Predict" })).toHaveAttribute("aria-current", "page"));
  });

  it("shows the real health status in the navigation", async () => {
    renderAt("/predict");
    expect((await screen.findAllByText("Model v2")).length).toBeGreaterThan(0);
  });
});

describe("prediction form", () => {
  async function fill(user: ReturnType<typeof userEvent.setup>) {
    await screen.findByText("Profile");
    await user.selectOptions(screen.getByLabelText(/^Gender/), "Female");
    await user.type(screen.getByLabelText(/^CGPA/), "8.2");
    await user.type(screen.getByLabelText(/^Internships/), "2");
  }

  it("sends the contract payload and shows the result only after the API responds", async () => {
    let resolve!: (v: PredictionResponse) => void;
    api.predict.mockReturnValue(new Promise((r) => (resolve = r)));
    const user = userEvent.setup();
    renderAt("/predict");
    await fill(user);
    await user.click(screen.getByRole("button", { name: /Predict placement/ }));
    expect(api.predict).toHaveBeenCalledWith({ Gender: "Female", CGPA: 8.2, Internships: 2, CodingTestScore: null });
    expect(screen.queryByText("Model estimate")).not.toBeInTheDocument();
    expect(screen.getByText("Validating, preprocessing and scoring…")).toBeInTheDocument();
    resolve(prediction);
    expect(await screen.findByText("Model estimate")).toBeInTheDocument();
    expect(screen.getByText(prediction.request_id)).toBeInTheDocument();
    expect(screen.getByText(/Estimate, not a guarantee/)).toBeInTheDocument();
    expect(screen.getByText("CodingTestScore")).toBeInTheDocument();
  });

  it("validates fields on the client and does not call the API", async () => {
    const user = userEvent.setup();
    renderAt("/predict");
    await screen.findByText("Profile");
    await user.type(screen.getByLabelText(/^CGPA/), "11");
    await user.click(screen.getByRole("button", { name: /Predict placement/ }));
    expect(await screen.findByText("Must be between 0 and 10")).toBeInTheDocument();
    expect(screen.getAllByText("Required").length).toBeGreaterThan(0);
    expect(api.predict).not.toHaveBeenCalled();
  });

  it("maps server validation errors back onto fields and shows API failures", async () => {
    const user = userEvent.setup();
    api.predict.mockRejectedValueOnce(new ApiError("Some inputs are invalid.", { status: 422, kind: "http", fieldErrors: [{ field: "Internships", message: "Server says no" }] }));
    renderAt("/predict");
    await fill(user);
    await user.click(screen.getByRole("button", { name: /Predict placement/ }));
    expect(await screen.findByText("Server says no")).toBeInTheDocument();

    api.predict.mockRejectedValueOnce(new ApiError("No model is currently available for predictions.", { status: 503, kind: "http" }));
    await user.clear(screen.getByLabelText(/^Internships/));
    await user.type(screen.getByLabelText(/^Internships/), "1");
    await user.click(screen.getByRole("button", { name: /Predict placement/ }));
    expect(await screen.findByText("Prediction failed")).toBeInTheDocument();
    expect(screen.queryByText("Model estimate")).not.toBeInTheDocument();
  });
});

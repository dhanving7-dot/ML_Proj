import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { FieldSchema, PredictionPage, PredictionResponse } from "../api/types";

const field = (name: string, group: FieldSchema["group"], minimum: number | null): FieldSchema => ({
  name,
  kind: "numeric",
  group,
  description: "",
  minimum,
  maximum: null,
  unit: null,
  integer: false,
  nullable: false,
});
const fields: FieldSchema[] = [field("Time", "time", 0), field("V1", "pca", null), field("V14", "pca", null), field("Amount", "amount", 0)];

const policy = { fusion_weight_lr: 1, fusion_weight_if: 0, review_threshold: 0.38, high_risk_threshold: 0.53, lr_threshold: 0.99, if_flag_percentile: 0.99, if_review_percentile: null };

const prediction: PredictionResponse = {
  request_id: "abc123def4567890",
  created_at: "2026-10-13T10:00:00Z",
  model_version: "v2",
  model_type: "hybrid_lr_iforest",
  transaction_id: "tx-9",
  decision: "HIGH_RISK",
  scores: { lr_probability: 0.9998, anomaly_score: 0.57, anomaly_percentile: 0.987, lr_rarity: 0.61, if_rarity: 0.36, hybrid_score: 0.61 },
  flags: { logistic_regression: true, isolation_forest: false },
  policy,
  explanation: {
    lr_top_contributions: [
      { feature: "V14", label: "V14 (anonymised PCA component)", standardised_value: -9.1, coefficient: -0.9, contribution: 8.2 },
      { feature: "amount_log", label: "log(1 + Amount)", standardised_value: 0.3, coefficient: 0.2, contribution: -0.06 },
    ],
    lr_intercept: -1.2,
    lr_prior_correction: -6.4,
    lr_logit: 8.5,
    hybrid_components: { lr: 0.61, isolation_forest: 0 },
    lr_exceedance: 0.0004,
    if_exceedance: 0.013,
    anomaly_review_trigger: false,
    out_of_training_range: [],
  },
  out_of_training_range: [],
  latency_ms: 12.5,
  timings: { preprocessing_ms: 2, inference_ms: 3, explanation_ms: 1, persistence_ms: 2, total_ms: 12.5 },
  disclaimer: "Risk scores support, but do not replace, investigation.",
};

const page: PredictionPage = {
  total: 1,
  limit: 25,
  offset: 0,
  decision_counts: { LOW_RISK: 0, REVIEW: 0, HIGH_RISK: 1 },
  items: [
    {
      request_id: "req0000000000001",
      batch_id: null,
      transaction_id: "replay-test-42",
      source: "replay",
      created_at: "2026-10-13T10:00:00Z",
      model_version: "v2",
      decision: "HIGH_RISK",
      hybrid_score: 0.61,
      lr_probability: 0.9998,
      anomaly_score: 0.57,
      anomaly_percentile: 0.987,
      lr_flag: true,
      if_flag: false,
      amount: 1.0,
      time_s: 3600,
      out_of_range: [],
      latency_ms: 10,
      actual_outcome: null,
      outcome_recorded_at: null,
    },
  ],
};

const api = vi.hoisted(() => ({
  health: vi.fn(),
  schema: vi.fn(),
  datasets: vi.fn(),
  datasetSample: vi.fn(),
  predict: vi.fn(),
  predictions: vi.fn(),
  dashboard: vi.fn(),
  lifecycle: vi.fn(),
  prediction: vi.fn(),
  recordOutcome: vi.fn(),
  evaluationSummary: vi.fn(),
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
  api.health.mockResolvedValue({ status: "ok", api_version: "2.0.0", time: "", uptime_s: 1, database: "ok", model_loaded: true, model_version: "v2", model_type: "hybrid_lr_iforest", policy, consistency_check: { passed: true }, message: null });
  api.schema.mockResolvedValue({
    request_fields: fields,
    model_features: [],
    metadata_fields: ["transaction_id"],
    transformed_features: [],
    feature_ranges: null,
    excluded_columns: {},
    target: "Class",
    class_labels: {},
    decisions: ["LOW_RISK", "REVIEW", "HIGH_RISK"],
    time_encodings: [],
    amount_transforms: [],
  });
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
    renderAt("/analyze");
    const nav = await screen.findByRole("navigation", { name: "Primary" });
    await waitFor(() => expect(within(nav).getByRole("link", { name: "Analyze" })).toHaveAttribute("aria-current", "page"));
  });

  it("redirects the old PlacementPredict paths", async () => {
    const router = renderAt("/predict");
    await waitFor(() => expect(router.state.location.pathname).toBe("/analyze"));
  });

  it("shows the real health status in the navigation", async () => {
    renderAt("/analyze");
    expect((await screen.findAllByText("Model v2")).length).toBeGreaterThan(0);
  });
});

describe("transaction analysis form", () => {
  async function fill(user: ReturnType<typeof userEvent.setup>) {
    await screen.findByText("Time, amount and reference");
    await user.type(screen.getByLabelText(/^Time \(s since/), "3600");
    await user.type(screen.getByLabelText(/^V1\s/), "-2.5");
    await user.type(screen.getByLabelText(/^V14/), "-9.1");
    await user.type(screen.getByLabelText(/^Amount/), "1");
  }

  it("sends the contract payload and shows the decision only after the API responds", async () => {
    let resolve!: (v: PredictionResponse) => void;
    api.predict.mockReturnValue(new Promise((r) => (resolve = r)));
    const user = userEvent.setup();
    renderAt("/analyze");
    await fill(user);
    await user.click(screen.getByRole("button", { name: /Assess risk/ }));
    expect(api.predict).toHaveBeenCalledWith({ Time: 3600, V1: -2.5, V14: -9.1, Amount: 1 });
    expect(screen.queryByText("Risk assessment")).not.toBeInTheDocument();
    expect(screen.getByText("Validating, preprocessing and scoring…")).toBeInTheDocument();
    resolve(prediction);
    expect(await screen.findByText("Risk assessment")).toBeInTheDocument();
    expect(screen.getByText("High risk")).toBeInTheDocument();
    expect(screen.getByText(prediction.request_id)).toBeInTheDocument();
    expect(screen.getByText("V14 (anonymised PCA component)")).toBeInTheDocument();
    expect(screen.getByText(/Decision support, not a verdict/)).toBeInTheDocument();
  });

  it("validates fields on the client and does not call the API", async () => {
    const user = userEvent.setup();
    renderAt("/analyze");
    await screen.findByText("Time, amount and reference");
    await user.type(screen.getByLabelText(/^Amount/), "-5");
    await user.click(screen.getByRole("button", { name: /Assess risk/ }));
    expect(await screen.findByText("Must be ≥ 0")).toBeInTheDocument();
    expect(screen.getAllByText("Required").length).toBeGreaterThan(0);
    expect(api.predict).not.toHaveBeenCalled();
  });

  it("maps server validation errors back onto fields and shows API failures", async () => {
    const user = userEvent.setup();
    api.predict.mockRejectedValueOnce(new ApiError("Some inputs are invalid.", { status: 422, kind: "http", fieldErrors: [{ field: "V14", message: "Server says no" }] }));
    renderAt("/analyze");
    await fill(user);
    await user.click(screen.getByRole("button", { name: /Assess risk/ }));
    expect(await screen.findByText("Server says no")).toBeInTheDocument();

    api.predict.mockRejectedValueOnce(new ApiError("No model is currently available for predictions.", { status: 503, kind: "http" }));
    await user.clear(screen.getByLabelText(/^V14/));
    await user.type(screen.getByLabelText(/^V14/), "1");
    await user.click(screen.getByRole("button", { name: /Assess risk/ }));
    expect(await screen.findByText("Scoring failed")).toBeInTheDocument();
    expect(screen.queryByText("Risk assessment")).not.toBeInTheDocument();
  });
});

describe("transaction monitoring", () => {
  it("lists transactions and filters by decision through the API", async () => {
    api.predictions.mockResolvedValue(page);
    const user = userEvent.setup();
    renderAt("/transactions");
    expect(await screen.findByText("replay-test-42")).toBeInTheDocument();
    expect(api.predictions).toHaveBeenLastCalledWith(expect.objectContaining({ limit: 25, offset: 0, sort: "newest" }), expect.anything());
    await user.click(screen.getByRole("button", { name: /^High risk/ }));
    await waitFor(() => expect(api.predictions).toHaveBeenLastCalledWith(expect.objectContaining({ decision: ["HIGH_RISK"] }), expect.anything()));
  });

  it("shows an empty state and an error state", async () => {
    api.predictions.mockResolvedValueOnce({ ...page, total: 0, items: [] });
    renderAt("/transactions");
    expect(await screen.findByText("No transactions scored yet")).toBeInTheDocument();
  });

  it("shows API failures instead of a blank table", async () => {
    api.predictions.mockRejectedValue(new ApiError("Cannot reach the FraudGuard API.", { kind: "network" }));
    renderAt("/transactions");
    expect(await screen.findByText("The transaction history could not be loaded")).toBeInTheDocument();
  });
});

describe("evaluation lab", () => {
  it("explains that a model is needed when none is active", async () => {
    api.evaluationSummary.mockRejectedValue(new ApiError("No active model is registered.", { status: 503, kind: "http" }));
    renderAt("/evaluation");
    expect(await screen.findByText("No active model")).toBeInTheDocument();
  });
});

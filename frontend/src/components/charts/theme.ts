/**
 * Chart palette. Categorical slots are the validated dark-mode steps, checked with the
 * dataviz validator against the chart surface #0e1120 (all checks pass; worst adjacent
 * CVD ΔE 8.4, normal-vision ΔE 19.3). Assign in this fixed order and never cycle.
 * Scatter-type charts (all pairs visible together) use at most the first three slots.
 */
export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] as const;

/** Class encoding used everywhere: Legitimate = slot 1 (blue), Fraud = slot 2 (orange). */
export const CLASS = { legitimate: SERIES[0], fraud: SERIES[1] } as const;

/** Decision encoding (always paired with the decision name, never colour alone). */
export const DECISION_COLOR = { LOW_RISK: "#3987e5", REVIEW: "#c98500", HIGH_RISK: "#d03b3b" } as const;

/** Model components: LR = slot 1, Isolation Forest = slot 3 (green), hybrid = slot 7 (violet). */
export const COMPONENT = { logistic_regression: SERIES[0], isolation_forest: SERIES[2], hybrid: SERIES[6] } as const;

export const CHROME = {
  grid: "rgba(255,255,255,0.06)",
  axis: "rgba(255,255,255,0.14)",
  muted: "#7d83a7",
  ink: "#eef0fb",
  ink2: "#a9aecb",
  neutral: "#4b5070",
  surface: "#0e1120",
};

/** Diverging pair (sign of a coefficient / contribution) with a neutral gray midpoint. */
export const DIVERGING = { negative: "#e66767", positive: "#3987e5", neutral: "#4b5070" } as const;

export const STATUS = { good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b" } as const;

export const axisProps = {
  stroke: CHROME.axis,
  tickLine: false,
  tick: { fill: CHROME.muted, fontSize: 11 },
} as const;

export const gridProps = { stroke: CHROME.grid, vertical: false } as const;

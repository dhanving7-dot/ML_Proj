export const cn = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

export function num(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "—";
  return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function int(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  return Math.round(x).toLocaleString("en-US");
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function ms(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "—";
  if (x >= 1000) return `${(x / 1000).toFixed(2)} s`;
  return `${x.toFixed(x < 10 ? 2 : 1)} ms`;
}

export function bytes(x: number | null | undefined): string {
  if (x == null) return "—";
  if (x < 1024) return `${x} B`;
  if (x < 1024 ** 2) return `${(x / 1024).toFixed(1)} KB`;
  return `${(x / 1024 ** 2).toFixed(2)} MB`;
}

export function sci(x: number | null | undefined): string {
  if (x == null) return "—";
  if (x === 0) return "0";
  return Math.abs(x) < 1e-3 ? x.toExponential(1) : x.toFixed(4);
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 5) return "just now";
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function humanize(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\bSem(\d)/g, "Sem $1")
    .replace(/^./, (c) => c.toUpperCase());
}

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  adaboost: "AdaBoost",
  gradient_boosting: "Gradient Boosting",
  hist_gradient_boosting: "Hist. Gradient Boosting",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

export const modelLabel = (k: string | null | undefined) => (k ? (MODEL_LABELS[k] ?? humanize(k)) : "—");

/**
 * Opens every lab experiment in Edge, runs it if it does not auto-run, waits for the real
 * result to render and reports render/console errors.   node scripts/lab-check.mjs [baseUrl] [outDir]
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright-core";

const base = process.argv[2] ?? "http://localhost:4173";
const out = process.argv[3] ?? "lab-check";
mkdirSync(out, { recursive: true });
const EXPS = {
  m2: ["linear-regression", "regularization", "logistic", "preprocessing"],
  m3: ["decision-tree", "pruning", "regression-tree", "ensembles", "importance"],
  m4: ["kmeans", "hierarchical", "dbscan", "pca", "embedding", "anomaly"],
  m5: ["metrics", "cross-validation", "nested-cv", "calibration", "search", "learning-curves", "statistical-tests"],
  m6: ["skew", "point-in-time", "drift-sandbox", "onnx", "packaging"],
};
const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await (await browser.newContext({ viewport: { width: 1440, height: 1000 } })).newPage();
let failures = 0;
for (const [m, keys] of Object.entries(EXPS)) {
  for (const k of keys) {
    const errors = [];
    const onErr = (msg) => msg.type() === "error" && errors.push(msg.text());
    const onPageErr = (e) => errors.push(e.message);
    page.on("console", onErr);
    page.on("pageerror", onPageErr);
    const t = Date.now();
    await page.goto(`${base}/lab?module=${m}&exp=${k}`, { waitUntil: "networkidle" });
    const runBtn = page.getByRole("button", { name: "Run experiment" });
    if (await runBtn.isVisible().catch(() => false)) await runBtn.click();
    let ok = true;
    try {
      await page.getByText("Run details").waitFor({ timeout: 90_000 });
      await page.waitForTimeout(1200);
      const alert = await page.getByRole("alert").count();
      if (alert) throw new Error(await page.getByRole("alert").first().innerText());
    } catch (e) {
      ok = false;
      errors.push(String(e.message).split("\n")[0]);
    }
    await page.screenshot({ path: join(out, `${m}-${k}.png`), fullPage: true });
    page.off("console", onErr);
    page.off("pageerror", onPageErr);
    if (!ok || errors.length) failures++;
    console.log(`${ok && !errors.length ? "PASS" : "FAIL"}  ${m}/${k}  ${((Date.now() - t) / 1000).toFixed(1)}s${errors.length ? "  " + errors.join(" | ") : ""}`);
  }
}
await browser.close();
process.exit(failures ? 1 : 0);

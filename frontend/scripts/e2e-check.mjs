/**
 * Interactive end-to-end checks in Microsoft Edge (playwright-core) against the running app + API.
 *   node scripts/e2e-check.mjs [baseUrl] [outDir]
 * Exits non-zero if any check fails.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright-core";

const base = process.argv[2] ?? "http://localhost:4173";
const out = process.argv[3] ?? "e2e-check";
mkdirSync(out, { recursive: true });

const results = [];
const check = async (name, fn) => {
  try {
    const detail = await fn();
    results.push({ name, ok: true, detail });
  } catch (e) {
    results.push({ name, ok: false, detail: String(e?.message ?? e).split("\n")[0] });
  }
};

const browser = await chromium.launch({ channel: "msedge", headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(e.message));

await check("Predict: real student loads and the API prediction is revealed", async () => {
  await page.goto(`${base}/predict`, { waitUntil: "networkidle" });
  await page.getByText(/Loaded student #\d+/).waitFor({ timeout: 15000 });
  const apiResponse = page.waitForResponse((r) => r.url().endsWith("/api/predict") && r.request().method() === "POST");
  await page.getByRole("button", { name: /Predict placement/ }).click();
  const res = await apiResponse;
  const body = await res.json();
  await page.getByText("Model estimate").waitFor({ timeout: 10000 });
  await page.getByText(body.request_id).waitFor({ timeout: 5000 });
  const label = await page.getByText(body.label, { exact: true }).first().isVisible();
  await page.screenshot({ path: join(out, "predict-result.png"), fullPage: false });
  if (!label) throw new Error("label not shown");
  return `${res.status()} ${body.label} p=${body.probability} id=${body.request_id.slice(0, 8)}`;
});

await check("Predict: trace expands with server timings", async () => {
  await page.getByRole("button", { name: /Show trace/ }).click();
  await page.getByText("Persistence", { exact: false }).first().waitFor({ timeout: 8000 });
  return "trace visible";
});

await check("Predict: client-side validation blocks an out-of-range CGPA (no request sent)", async () => {
  let sent = 0;
  const onReq = (r) => r.url().endsWith("/api/predict") && r.method() === "POST" && sent++;
  page.on("request", onReq);
  await page.getByLabel(/^CGPA/).fill("12");
  await page.getByRole("button", { name: /Predict placement/ }).click();
  await page.getByText("Must be between 0 and 10").waitFor({ timeout: 3000 });
  page.off("request", onReq);
  if (sent) throw new Error("request was sent");
  return "error shown, 0 requests";
});

await check("Predict: double-click submits only once", async () => {
  await page.getByLabel(/^CGPA/).fill("7.5");
  let sent = 0;
  const onReq = (r) => r.url().endsWith("/api/predict") && r.method() === "POST" && sent++;
  page.on("request", onReq);
  await page.getByRole("button", { name: /Predict placement/ }).dblclick();
  await page.waitForTimeout(1500);
  page.off("request", onReq);
  if (sent !== 1) throw new Error(`${sent} requests`);
  return "1 request";
});

await check("Navigation: nav link, lab module tab, browser back/forward", async () => {
  await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: "ML Lab" }).click();
  await page.waitForURL(/\/lab/);
  await page.getByRole("tab", { name: /M3/ }).click();
  await page.waitForURL(/module=m3/);
  await page.getByText("Gini impurity").first().waitFor();
  await page.goBack();
  await page.waitForURL((u) => !u.search.includes("m3"));
  await page.goForward();
  await page.waitForURL(/module=m3/);
  return page.url();
});

await check("Deep link + refresh restores lab module and experiment", async () => {
  await page.goto(`${base}/lab?module=m4&exp=pca`, { waitUntil: "networkidle" });
  await page.reload({ waitUntil: "networkidle" });
  const sel = await page.getByRole("tab", { name: "PCA" }).getAttribute("aria-selected");
  await page.getByText("Scree plot").waitFor({ timeout: 15000 });
  if (sel !== "true") throw new Error("PCA tab not selected");
  return "PCA selected after reload";
});

await check("Lab: changing a live parameter re-runs the experiment", async () => {
  await page.goto(`${base}/lab?module=m3&exp=decision-tree`, { waitUntil: "networkidle" });
  await page.getByText("Top of the fitted tree").waitFor({ timeout: 15000 });
  const next = page.waitForResponse((r) => r.url().includes("/api/lab/m3/decision-tree") && r.status() === 200);
  await page.getByLabel("Split criterion").selectOption("entropy");
  const res = await next;
  const body = await res.json();
  if (body.params.criterion !== "entropy") throw new Error("criterion not sent");
  return `re-ran with ${body.params.criterion}, depth ${body.result.depth}`;
});

await check("Monitoring: drift check records an event and shows a toast", async () => {
  await page.goto(`${base}/monitoring`, { waitUntil: "networkidle" });
  const res = page.waitForResponse((r) => r.url().endsWith("/api/monitoring/drift/check"));
  await page.getByRole("button", { name: /Run drift check/ }).click();
  const body = await (await res).json();
  await page.getByText(/Drift check:/).first().waitFor({ timeout: 5000 });
  return body.status;
});

await check("Scroll reveals complete and survive repeated route changes", async () => {
  await page.goto(`${base}/`, { waitUntil: "networkidle" });
  for (let i = 0; i < 3; i++) {
    await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: "Dashboard" }).click();
    await page.waitForURL(/dashboard/);
    await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: "Home", exact: true }).click();
    await page.waitForURL((u) => u.pathname === "/");
  }
  const target = page.locator("[data-step]").last();
  await target.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1800);
  const opacity = await target.evaluate((el) => getComputedStyle(el.parentElement).opacity);
  await page.mouse.wheel(0, -4000);
  await page.waitForTimeout(800);
  const stillThere = await target.evaluate((el) => getComputedStyle(el.parentElement).opacity);
  if (opacity !== "1" || stillThere !== "1") throw new Error(`opacity ${opacity} / after scrolling back ${stillThere}`);
  return "revealed (opacity 1) and still visible after scrolling back up";
});

await check("Unknown route renders the 404 page", async () => {
  await page.goto(`${base}/does-not-exist`, { waitUntil: "networkidle" });
  await page.getByText("This page does not exist").waitFor();
  return "404 ok";
});

await check("Reduced motion: home content is visible without animation", async () => {
  const rctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, reducedMotion: "reduce" });
  const p = await rctx.newPage();
  await p.goto(`${base}/`, { waitUntil: "networkidle" });
  const op = await p.locator("[data-hero=copy]").evaluate((el) => getComputedStyle(el).opacity);
  const story = await p.getByText("One request, six transparent steps.").isVisible();
  await p.screenshot({ path: join(out, "reduced-motion-home.png") });
  await rctx.close();
  if (op !== "1" || !story) throw new Error(`opacity ${op}, story ${story}`);
  return "visible";
});

await check("API offline: dashboard shows an error state instead of fake data", async () => {
  const octx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  await octx.route("**/api/**", (r) => r.abort("connectionrefused"));
  const p = await octx.newPage();
  await p.goto(`${base}/dashboard`, { waitUntil: "domcontentloaded" });
  await p.getByText("Dashboard data could not be loaded").waitFor({ timeout: 10000 });
  const badge = await p.getByText("API offline").first().isVisible();
  await p.screenshot({ path: join(out, "offline-dashboard.png") });
  await octx.close();
  if (!badge) throw new Error("offline badge missing");
  return "error state + offline badge";
});

await browser.close();
let failed = 0;
for (const r of results) {
  console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.detail ? `  —  ${r.detail}` : ""}`);
  if (!r.ok) failed++;
}
const relevant = consoleErrors.filter((e) => !/ERR_CONNECTION_REFUSED|Failed to load resource/.test(e));
console.log(`console errors: ${relevant.length}`);
relevant.forEach((e) => console.log("   " + e));
process.exit(failed || relevant.length ? 1 : 0);

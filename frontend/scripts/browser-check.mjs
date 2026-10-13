/**
 * Browser verification with the locally installed Microsoft Edge (playwright-core, no browser download).
 * Visits routes at several viewport sizes, saves full-page screenshots, and reports console errors,
 * failed requests, uncaught exceptions and horizontal overflow.
 *
 *   node scripts/browser-check.mjs [baseUrl] [outDir] [routes,comma,separated] [--reduced-motion]
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright-core";

const base = process.argv[2] ?? "http://127.0.0.1:4173";
const out = process.argv[3] ?? "browser-check";
const routes = (process.argv[4] ?? "/,/dashboard,/predict,/lab,/lifecycle,/monitoring,/models,/data").split(",");
const reduced = process.argv.includes("--reduced-motion");
const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "tablet", width: 820, height: 1180 },
  { name: "mobile", width: 390, height: 844 },
];

mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "msedge", headless: true });
const report = [];
for (const vp of viewports) {
  const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, reducedMotion: reduced ? "reduce" : "no-preference" });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => errors.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => r.status() >= 500 && errors.push(`http ${r.status()}: ${r.url()}`));
  for (const route of routes) {
    errors.length = 0;
    await page.goto(base + route, { waitUntil: "networkidle", timeout: 60_000 });
    await page.waitForTimeout(1500);
    // Scroll through the page so scroll-triggered content is revealed before the screenshot.
    const height = await page.evaluate(() => document.documentElement.scrollHeight);
    for (let y = 0; y < height; y += Math.round(vp.height * 0.7)) {
      await page.evaluate((yy) => window.scrollTo(0, yy), y);
      await page.waitForTimeout(160);
    }
    await page.waitForTimeout(900);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(400);
    const file = join(out, `${vp.name}${route === "/" ? "-home" : route.replace(/\//g, "-")}.png`);
    await page.screenshot({ path: file, fullPage: true });
    report.push({ viewport: vp.name, route, horizontalOverflowPx: overflow, errors: [...errors] });
  }
  await ctx.close();
}
await browser.close();
writeFileSync(join(out, "report.json"), JSON.stringify(report, null, 2));
for (const r of report) {
  console.log(`${r.viewport.padEnd(8)} ${r.route.padEnd(12)} overflow=${r.horizontalOverflowPx}px errors=${r.errors.length}`);
  for (const e of r.errors) console.log("    " + e);
}

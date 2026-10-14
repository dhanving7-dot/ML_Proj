/**
 * Renders every route in Microsoft Edge (playwright-core) against the running app + API and reports
 * console errors, failed requests and horizontal overflow.
 *   node scripts/browser-check.mjs [baseUrl]
 * Exits non-zero if any page fails.
 */
import { chromium } from "playwright-core";

const base = process.argv[2] ?? "http://localhost:4173";
const routes = ["/", "/overview", "/analyze", "/transactions", "/evaluation", "/models", "/monitoring", "/lifecycle", "/data", "/lab?module=m2", "/lab?module=m6"];
const browser = await chromium.launch({ channel: "msedge", headless: true });
let failures = 0;
for (const width of [1440, 390]) {
  const ctx = await browser.newContext({ viewport: { width, height: 900 } });
  const page = await ctx.newPage();
  for (const route of routes) {
    const errors = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("requestfailed", (r) => errors.push(`request failed: ${r.url()}`));
    await page.goto(base + route, { waitUntil: "networkidle", timeout: 60000 });
    await page.waitForTimeout(1500);
    const crashed = await page.getByText("This view failed to render").count();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    const ok = !crashed && errors.length === 0 && overflow <= 1;
    if (!ok) failures++;
    console.log(`${ok ? "OK  " : "FAIL"} ${width}px ${route}${crashed ? " CRASHED" : ""}${overflow > 1 ? ` overflow ${overflow}px` : ""}${errors.length ? " " + errors.slice(0, 2).join(" | ") : ""}`);
    page.removeAllListeners("console");
    page.removeAllListeners("pageerror");
    page.removeAllListeners("requestfailed");
  }
  await ctx.close();
}
await browser.close();
process.exit(failures ? 1 : 0);

// End-to-end smoke test in a headless browser: load a sample product, check the
// dashboard, ask two questions, upload a CSV, and fail on any page error.
//
//   npm run build && npm run smoke -- dist                      # built files + local API (127.0.0.1:8000)
//   npm run smoke -- https://skywalker-666.github.io/review-analyst/   # the live site
//
// Optional: CHROME_PATH=/path/to/chrome, SHOT=out.png (full-page screenshot).
import { chromium } from "playwright-core";
import { globSync, readFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";

const SRC = process.argv[2] || "dist";
const exe = process.env.CHROME_PATH ||
  globSync(`${process.env.HOME}/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell`)[0];
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
const browser = await chromium.launch({ executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) failures++; };

let url = SRC;
if (!SRC.startsWith("http")) {
  // Serve the built files from disk; the local API answers CORS for its dev origin.
  const dist = resolve(SRC);
  url = "https://app.test/";
  await page.route("https://app.test/**", (route) => {
    let p = new URL(route.request().url()).pathname;
    if (p === "/") p = "/index.html";
    route.fulfill({ body: readFileSync(join(dist, p)), contentType: TYPES[extname(p)] || "application/octet-stream" });
  });
  await page.route("http://127.0.0.1:8000/**", async (route) => {
    const resp = await route.fetch({ headers: { ...route.request().headers(), origin: "http://127.0.0.1:5173" } });
    route.fulfill({ response: resp, headers: { ...resp.headers(), "access-control-allow-origin": "*" } });
  });
}

const rows = 'section[aria-labelledby="rev-title"] tbody tr';
await page.goto(url, { waitUntil: "networkidle", timeout: 120000 });
await page.waitForSelector('section[aria-labelledby="src-title"] .row button', { timeout: 120000 });
const status = (await page.locator(".pills").innerText()).replace(/\s+/g, " ");
console.log("status:", status);

await page.getByRole("button", { name: "Electronics" }).click();
await page.waitForSelector("#sum-title", { timeout: 120000 });
console.log("summary:", (await page.locator(".stats").innerText()).replace(/\s+/g, " "));
check((await page.locator(rows).count()) === 250, "250 sample reviews in the table");
check((await page.locator(".recharts-bar-rectangle").count()) > 0, "aspect chart drawn");
await page.selectOption("#rev-filter", "fake");
const fakeRows = await page.locator(rows).count();
check(fakeRows > 0 && fakeRows < 250, `fake filter narrows the table (${fakeRows})`);
await page.selectOption("#rev-filter", "all");

for (const q of ["What do people hate about the sound?", "Is the battery any good?"]) {
  await page.fill("#ask-input", q);
  await page.click('.chat button[type="submit"]');
  await page.waitForSelector('.chat [role="status"]', { state: "detached", timeout: 120000 }).catch(() => {});
  const m = page.locator(".msg.assistant").last();
  const text = (await m.innerText()).replace(/\s+/g, " ");
  console.log(`Q: ${q}\n   ${text.slice(0, 300)}`);
  check((await m.locator(".cite").count()) > 0, "answer has sources");
  check(!text.includes("**"), "markdown rendered");
}

const csv = "review,rating\nThe battery died after a week and support never replied.,1\n" +
  "Love the color and the fit is perfect.,5\nGreat product love it great product.,5\n";
await page.setInputFiles("#csv", { name: "mine.csv", mimeType: "text/csv", buffer: Buffer.from(csv) });
await page.waitForFunction(() => document.querySelector("#sum-title")?.textContent?.includes("mine.csv"), null,
  { timeout: 120000 });
check((await page.locator(rows).count()) === 3, "uploaded CSV analysed");

check(errors.length === 0, `no page errors${errors.length ? `: ${errors.join("; ")}` : ""}`);
if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT, fullPage: true });
await browser.close();
process.exit(failures ? 1 : 0);

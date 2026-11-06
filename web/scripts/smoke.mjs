// Smoke test: build output loads, the room renders, hotspots move the camera.
// Runs headless Chromium with software WebGL, so it checks correctness, not speed.
//
//   npm run build && npm run smoke
//   npm run smoke -- --shots shots/       also save screenshots

import { mkdirSync } from "node:fs";
import path from "node:path";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

const args = process.argv.slice(2);
const shotDir = args.includes("--shots") ? args[args.indexOf("--shots") + 1] : null;
const port = 3100 + Math.floor(Math.random() * 500);
const server = await serve("out", port);
const base = `http://localhost:${port}`;
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const failures = [];
const check = (ok, msg) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  if (!ok) failures.push(msg);
};
if (shotDir) mkdirSync(shotDir, { recursive: true });
const shot = async (page, name) => shotDir && page.screenshot({ path: path.join(shotDir, `${name}.png`) });

try {
  const index = await (await fetch(`${base}/rooms/index.json`)).json();
  check(index.rooms.length > 0, `index lists ${index.rooms.length} room(s)`);

  const ctx = await browser.newContext({ viewport: { width: 412, height: 870 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  await page.goto(`${base}/`, { waitUntil: "networkidle" });
  check((await page.locator(".room-card").count()) === index.rooms.length, "home page shows a card per room");
  await shot(page, "home");

  for (const room of index.rooms) {
    for (const format of ["sog", "rad"]) {
      const manifest = await (await fetch(`${base}/rooms/${room.id}/manifest.json`)).json();
      if (format === "rad" && !manifest.splats.rad) continue;
      await page.goto(`${base}/r/${room.id}/?stats=1&tier=low&format=${format}`, { waitUntil: "domcontentloaded" });
      const first = await page
        .waitForFunction(() => window.__roomwalk && window.__roomwalk.firstViewMs !== null, null, { timeout: 120000 })
        .then(() => page.evaluate(() => window.__roomwalk))
        .catch(() => null);
      check(!!first, `${room.id} (${format}): room becomes visible${first ? ` at ${first.firstViewMs} ms` : ""}`);
      if (!first) continue;
      await page.waitForTimeout(1500);
      await shot(page, `${room.id}-${format}-start`);
      const before = await page.screenshot();
      const chips = page.locator(".chip");
      if ((await chips.count()) > 0) {
        await chips.first().click();
        await page.waitForTimeout(2500);
        const after = await page.screenshot();
        check(!before.equals(after), `${room.id} (${format}): hotspot "${await chips.first().innerText()}" moves the view`);
        await shot(page, `${room.id}-${format}-hotspot`);
      }
    }
  }
  await page.locator("button", { hasText: "Details" }).click();
  check(await page.locator(".details").isVisible(), "details sheet opens");
  await shot(page, "details");
  const real = errors.filter((e) => !/favicon/.test(e));
  check(real.length === 0, `no page errors${real.length ? `: ${real.join(" | ")}` : ""}`);
} finally {
  await browser.close();
  server.close();
}
if (failures.length) {
  console.error(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nAll smoke checks passed");

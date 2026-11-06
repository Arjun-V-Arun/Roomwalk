// Shared helpers for scripts that drive the viewer in headless Chromium.

import { chromium } from "playwright";
import { serve } from "./serve.mjs";

// Software WebGL so scripts also run on machines without a GPU. Pass --gpu to
// use the real graphics card instead (much faster on a laptop).
export async function launch({ gpu = process.argv.includes("--gpu") } = {}) {
  const args = gpu
    ? ["--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
    : ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"];
  return chromium.launch({ args, headless: true });
}

export async function startServer(opts = {}) {
  const port = 3500 + Math.floor(Math.random() * 1000);
  const server = await serve("out", port, opts);
  return { server, base: `http://localhost:${port}` };
}

export async function openRoom(browser, base, id, { width = 960, height = 540, query = "", timeout = 180000 } = {}) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  await page.goto(`${base}/r/${id}/?ui=0${query}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__roomwalk && window.__roomwalk.firstViewMs !== null, null, { timeout });
  return page;
}

export function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

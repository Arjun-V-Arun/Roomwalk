// Benchmark: how long until a phone on mobile data sees the room, and how
// much data it costs, for each delivery format.
//
//   npm run build && npm run bench -- demo --runs 3 [--gpu] [--cpu 4]
//
// Network profiles are imitated at the server (shared bandwidth cap + latency).
// CPU slowdown uses Chromium's CPU throttling. Without --gpu, WebGL runs in
// software, which inflates decode and render time; compare formats against
// each other on the same machine rather than reading the numbers as phone
// numbers. For phone numbers, open a room on a phone with ?stats=1.

import { writeFileSync } from "node:fs";
import { arg, launch, startServer } from "./browser.mjs";

const id = process.argv[2] && !process.argv[2].startsWith("--") ? process.argv[2] : "demo";
const runs = Number(arg("runs", "3"));
const cpu = Number(arg("cpu", "4"));
const out = arg("out", "bench-results.json");
const PROFILES = [
  { name: "4G, good signal", mbit: 20, latencyMs: 40 },
  { name: "4G, congested", mbit: 4, latencyMs: 150 },
];
const FORMATS = ["sog", "rad"];
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)] : null;
};

const browser = await launch();
const results = [];
try {
  for (const profile of PROFILES) {
    const { server, base } = await startServer({ mbit: profile.mbit, latencyMs: profile.latencyMs });
    for (const format of FORMATS) {
      const samples = [];
      for (let r = 0; r < runs; r++) {
        const ctx = await browser.newContext({ viewport: { width: 412, height: 870 }, deviceScaleFactor: 1, isMobile: true });
        const page = await ctx.newPage();
        const cdp = await ctx.newCDPSession(page);
        await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
        server.stats.reset();
        const t0 = Date.now();
        await page.goto(`${base}/r/${id}/?ui=0&tier=low&format=${format}`, { waitUntil: "domcontentloaded" });
        const ok = await page
          .waitForFunction(() => window.__roomwalk && window.__roomwalk.firstViewMs !== null, null, { timeout: 240000 })
          .then(() => true)
          .catch(() => false);
        const m = ok ? await page.evaluate(() => window.__roomwalk) : null;
        const firstViewAt = t0 + (m?.firstViewMs ?? 0);
        const room = server.stats.requests.filter((q) => q.path.startsWith(`/rooms/${id}/`));
        const bytesBefore = room.filter((q) => q.doneMs !== null && q.doneMs <= firstViewAt).reduce((s, q) => s + q.bytes, 0);
        await page.waitForTimeout(8000); // let streaming settle at the low-tier budget
        const roomAfter = server.stats.requests.filter((q) => q.path.startsWith(`/rooms/${id}/`));
        const js = server.stats.requests.filter((q) => q.path.endsWith(".js")).reduce((s, q) => s + q.bytes, 0);
        samples.push({
          ok,
          firstViewMs: m?.firstViewMs ?? null,
          splatsReadyMs: m?.splatsReadyMs ?? null,
          roomBytesBeforeFirstView: bytesBefore,
          roomBytesAfter8s: roomAfter.reduce((s, q) => s + q.bytes, 0),
          roomRequests: roomAfter.length,
          jsBytes: js,
        });
        await ctx.close();
        process.stdout.write(`${profile.name} / ${format} run ${r + 1}: ${m?.firstViewMs ?? "timeout"} ms\n`);
      }
      const good = samples.filter((s) => s.ok);
      results.push({
        profile: profile.name,
        format,
        runs: samples.length,
        failed: samples.length - good.length,
        firstViewMs: median(good.map((s) => s.firstViewMs)),
        splatsReadyMs: median(good.map((s) => s.splatsReadyMs)),
        roomMbBeforeFirstView: +(median(good.map((s) => s.roomBytesBeforeFirstView)) / 1e6).toFixed(2),
        roomMbAfter8s: +(median(good.map((s) => s.roomBytesAfter8s)) / 1e6).toFixed(2),
        roomRequests: median(good.map((s) => s.roomRequests)),
        jsMb: +(median(good.map((s) => s.jsBytes)) / 1e6).toFixed(2),
        samples,
      });
    }
    server.close();
  }
} finally {
  await browser.close();
}

const env = { room: id, runs, cpuSlowdown: cpu, gpu: process.argv.includes("--gpu"), date: new Date().toISOString() };
writeFileSync(out, JSON.stringify({ env, results }, null, 2));
console.log(`\n| Network | Format | First view | Room data before first view | Room data after 8 s | Requests |`);
console.log(`| --- | --- | --- | --- | --- | --- |`);
for (const r of results) {
  const fv = r.firstViewMs === null ? "timed out" : `${(r.firstViewMs / 1000).toFixed(1)} s`;
  console.log(`| ${r.profile} | ${r.format.toUpperCase()} | ${fv} | ${r.roomMbBeforeFirstView} MB | ${r.roomMbAfter8s} MB | ${r.roomRequests} |`);
}
console.log(`\nJavaScript downloaded: ${results[0]?.jsMb} MB. Results saved to ${out}.`);

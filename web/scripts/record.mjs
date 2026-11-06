// Record a walkthrough of a published room as a video.
//
// Two uses:
// 1. A short clip to share where a 3D link won't open (an Instagram story).
// 2. Testing the capture pipeline without filming anything: render the demo
//    room the way a phone would film it, then feed the video back through
//    `roomwalk build`. Because we chose every camera position, we know what
//    structure from motion and alignment should recover.
//
//   node scripts/record.mjs demo --out walk.mp4 [--frames 180] [--fps 6] [--size 960x540] [--poses poses.json] [--gpu]
//
// Camera path: one loop near the walls looking across the room, then one loop
// near the middle looking out at the walls, the way the scan guide asks people
// to film. Positions are written to <out>.poses.json for checking.

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { arg, launch, openRoom, startServer } from "./browser.mjs";

const id = process.argv[2];
if (!id || id.startsWith("--")) {
  console.error("usage: node scripts/record.mjs <room-id> --out walk.mp4");
  process.exit(1);
}
const out = path.resolve(arg("out", `${id}-walk.mp4`));
const frames = Number(arg("frames", "180"));
const fps = Number(arg("fps", "6"));
const [width, height] = arg("size", "960x540").split("x").map(Number);
const eye = Number(arg("eye", "1.45"));

const manifest = JSON.parse(readFileSync(`public/rooms/${id}/manifest.json`, "utf8"));
const { min, max } = manifest.view.bounds;
const cx = (min[0] + max[0]) / 2;
const cz = (min[2] + max[2]) / 2;
const rx = (max[0] - min[0]) / 2;
const rz = (max[2] - min[2]) / 2;

function pathPoses(n) {
  const poses = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const outer = i < half;
    const k = outer ? i / half : (i - half) / (n - half);
    const a = 2 * Math.PI * k + (outer ? 0 : 0.6);
    const r = outer ? 0.62 : 0.22;
    const x = cx + rx * r * Math.cos(a);
    const z = cz + rz * r * Math.sin(a);
    const y = eye + 0.03 * Math.sin(7 * a);
    // Outer loop looks back across the room; inner loop looks out at the walls.
    const look = outer ? a + Math.PI + 0.5 : a;
    const pitch = -0.12 + 0.12 * Math.sin(3 * a);
    poses.push({
      position: [x, y, z],
      target: [x + Math.cos(look), y + pitch, z + Math.sin(look)],
    });
  }
  return poses;
}

const poses = pathPoses(frames);
const dir = `${out}.frames`;
rmSync(dir, { recursive: true, force: true });
mkdirSync(dir, { recursive: true });

const { server, base } = await startServer();
const browser = await launch();
const t0 = Date.now();
try {
  const page = await openRoom(browser, base, id, { width, height, query: "&tier=high&format=sog" });
  await page.waitForTimeout(3000);
  for (let i = 0; i < poses.length; i++) {
    await page.evaluate((p) => window.__roomwalkSetPose(p), poses[i]);
    await page.waitForTimeout(60);
    await page.screenshot({ path: path.join(dir, `f_${String(i).padStart(5, "0")}.png`) });
    if (i % 30 === 0) console.log(`frame ${i}/${poses.length} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  }
} finally {
  await browser.close();
  server.close();
}
writeFileSync(`${out}.poses.json`, JSON.stringify({ room: id, fps, eye, poses }, null, 1));
execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(fps), "-i", path.join(dir, "f_%05d.png"),
  "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out]);
console.log(`wrote ${out} (${poses.length} frames at ${fps} fps) and ${out}.poses.json`);

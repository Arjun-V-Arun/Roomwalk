// Tiny static file server for the exported site (out/).
//
// * Supports HTTP Range requests, which streamed RAD files depend on.
// * Can imitate mobile data: one shared bandwidth budget for all connections
//   (so parallel chunk downloads don't get extra speed) plus per-request latency.
// * Counts bytes and requests per path, because Spark downloads inside web
//   workers and the page itself cannot see those requests.
//
//   node scripts/serve.mjs out 3000            full speed
//   node scripts/serve.mjs out 3000 4 150      4 Mbit/s, 150 ms latency

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { Transform } from "node:stream";
import { fileURLToPath } from "node:url";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
  ".txt": "text/plain",
};

// Token bucket shared by every response from one server.
function makeLimiter(bytesPerSec) {
  let tokens = 0;
  let last = Date.now();
  const refill = () => {
    const now = Date.now();
    tokens = Math.min(bytesPerSec / 10, tokens + ((now - last) / 1000) * bytesPerSec);
    last = now;
  };
  return () =>
    new Transform({
      transform(chunk, _enc, done) {
        const pump = (offset) => {
          if (offset >= chunk.length) return done();
          refill();
          const n = Math.floor(Math.min(tokens, chunk.length - offset, 16384));
          if (n > 0) {
            tokens -= n;
            this.push(chunk.subarray(offset, offset + n));
          }
          setTimeout(() => pump(offset + n), n > 0 ? 0 : 10);
        };
        pump(0);
      },
    });
}

export function serve(root, port = 3000, { mbit = 0, latencyMs = 0, basePath = "" } = {}) {
  root = path.resolve(root);
  const limiter = mbit > 0 ? makeLimiter((mbit * 1e6) / 8) : null;
  const stats = { requests: [], reset() { this.requests = []; } };
  const server = createServer((req, res) => {
    let url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (basePath && url.startsWith(basePath)) url = url.slice(basePath.length) || "/";
    let file = path.join(root, url);
    if (!file.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
    if (existsSync(file) && statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!existsSync(file)) {
      res.writeHead(404, { "content-type": "text/plain" }).end("not found");
      return;
    }
    const size = statSync(file).size;
    const headers = {
      "content-type": TYPES[path.extname(file)] || "application/octet-stream",
      "accept-ranges": "bytes",
      "cache-control": "no-store",
      "access-control-allow-origin": "*",
    };
    let start = 0;
    let end = size - 1;
    let status = 200;
    const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || "");
    if (range) {
      start = range[1] ? Number(range[1]) : size - Number(range[2]);
      end = range[1] && range[2] ? Number(range[2]) : size - 1;
      end = Math.min(end, size - 1);
      status = 206;
      headers["content-range"] = `bytes ${start}-${end}/${size}`;
    }
    const length = end - start + 1;
    headers["content-length"] = length;
    const entry = { path: url, bytes: req.method === "HEAD" ? 0 : length, startedMs: Date.now(), doneMs: null };
    stats.requests.push(entry);
    res.on("finish", () => (entry.doneMs = Date.now()));
    const reply = () => {
      res.writeHead(status, headers);
      if (req.method === "HEAD") return res.end();
      const stream = createReadStream(file, { start, end });
      (limiter ? stream.pipe(limiter()) : stream).pipe(res);
    };
    latencyMs > 0 ? setTimeout(reply, latencyMs) : reply();
  });
  server.stats = stats;
  return new Promise((resolve) => server.listen(port, () => resolve(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [root = "out", port = "3000", mbit = "0", latency = "0"] = process.argv.slice(2);
  const base = process.env.NEXT_PUBLIC_BASE_PATH || "";
  await serve(root, Number(port), { mbit: Number(mbit), latencyMs: Number(latency), basePath: base });
  const cap = Number(mbit) ? ` (capped at ${mbit} Mbit/s, ${latency} ms latency)` : "";
  console.log(`Serving ${root} at http://localhost:${port}${base}/${cap}`);
}

// What we measure, and how.
//
// Time to first view: from navigation start (performance.now() = 0) to the
// first rendered frame where the room is actually on screen. "On screen" is
// checked by reading a few pixels back after rendering and comparing them with
// the clear colour, so a frame that only shows background does not count.
//
// Bytes: Spark fetches splat data from web workers, and the page's resource
// timing never sees worker requests. For SOG the download progress gives the
// byte count; for streamed RAD the benchmark script counts bytes at the server.
//
// FPS: only frames rendered back to back count. The viewer renders on demand,
// so idle seconds where nothing changes would otherwise drag the average down.

export interface MetricsSnapshot {
  delivery: string;
  tier: string;
  tierReason: string;
  gpu: string;
  viewerStartMs: number | null;
  splatsReadyMs: number | null;
  firstViewMs: number | null;
  roomBytes: number;
  scriptBytes: number;
  requests: number;
  fps: number | null;
  p95FrameMs: number | null;
  pixelRatio: number;
  lodSplatScale: number;
  adaptations: string[];
  splatCount: number;
}

export class Metrics {
  snapshot: MetricsSnapshot;
  private frameTimes: number[] = [];
  private lastFrame = 0;
  private observer: PerformanceObserver | null = null;
  private listeners = new Set<(m: MetricsSnapshot) => void>();

  constructor(private roomPrefix: string, init: Partial<MetricsSnapshot>) {
    this.snapshot = {
      delivery: "", tier: "", tierReason: "", gpu: "",
      viewerStartMs: null, splatsReadyMs: null, firstViewMs: null,
      roomBytes: 0, scriptBytes: 0, requests: 0,
      fps: null, p95FrameMs: null, pixelRatio: 1, lodSplatScale: 1,
      adaptations: [], splatCount: 0,
      ...init,
    };
    this.collectResources(performance.getEntriesByType("resource") as PerformanceResourceTiming[]);
    if (typeof PerformanceObserver !== "undefined") {
      this.observer = new PerformanceObserver((list) => {
        this.collectResources(list.getEntries() as PerformanceResourceTiming[]);
        this.emit();
      });
      this.observer.observe({ type: "resource", buffered: false });
    }
  }

  private collectResources(entries: PerformanceResourceTiming[]) {
    for (const e of entries) {
      const size = e.transferSize || e.encodedBodySize || 0;
      if (e.name.includes(this.roomPrefix)) {
        this.snapshot.requests += 1;
      } else if (e.initiatorType === "script" || /\.js(\?|$)/.test(e.name)) {
        this.snapshot.scriptBytes += size;
      }
    }
  }

  mark(name: "viewerStartMs" | "splatsReadyMs" | "firstViewMs") {
    if (this.snapshot[name] === null) {
      this.snapshot[name] = Math.round(performance.now());
      this.emit();
    }
  }

  frame(now: number) {
    const dt = now - this.lastFrame;
    this.lastFrame = now;
    // Gaps over 100 ms mean the viewer was idle, not slow.
    if (dt > 0 && dt < 100) {
      this.frameTimes.push(dt);
      if (this.frameTimes.length > 120) this.frameTimes.shift();
    }
    if (this.frameTimes.length >= 20 && this.frameTimes.length % 10 === 0) {
      const sorted = [...this.frameTimes].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      this.snapshot.fps = Math.round(1000 / median);
      this.snapshot.p95FrameMs = Math.round(sorted[Math.floor(sorted.length * 0.95)] * 10) / 10;
      this.emit();
    }
  }

  recentMedianFps(): number | null {
    if (this.frameTimes.length < 30) return null;
    const recent = this.frameTimes.slice(-60).sort((a, b) => a - b);
    return 1000 / recent[Math.floor(recent.length / 2)];
  }

  resetFrames() {
    this.frameTimes = [];
  }

  set(patch: Partial<MetricsSnapshot>) {
    Object.assign(this.snapshot, patch);
    this.emit();
  }

  subscribe(fn: (m: MetricsSnapshot) => void) {
    this.listeners.add(fn);
    fn(this.snapshot);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    const copy = { ...this.snapshot, adaptations: [...this.snapshot.adaptations] };
    (window as unknown as { __roomwalk?: unknown }).__roomwalk = copy;
    this.listeners.forEach((fn) => fn(copy));
  }

  dispose() {
    this.observer?.disconnect();
    this.listeners.clear();
  }
}

// True if any sampled pixel differs clearly from the clear colour.
export function roomVisible(gl: WebGL2RenderingContext, clear: [number, number, number]): boolean {
  const w = gl.drawingBufferWidth;
  const h = gl.drawingBufferHeight;
  const px = new Uint8Array(4);
  let hits = 0;
  for (const fx of [0.25, 0.5, 0.75]) {
    for (const fy of [0.3, 0.5, 0.7]) {
      gl.readPixels(Math.floor(w * fx), Math.floor(h * fy), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      const d = Math.abs(px[0] - clear[0]) + Math.abs(px[1] - clear[1]) + Math.abs(px[2] - clear[2]);
      if (d > 24) hits += 1;
    }
  }
  return hits >= 3;
}

// Adaptive quality: lower the pixel ratio first, then the splat budget, while
// the phone cannot hold a smooth frame rate; raise them back when it can.
//
// Pixel ratio goes first because splat cost scales with pixels covered, and a
// small drop in sharpness is less visible than missing detail. Hysteresis (a
// low bar to step down, a high bar held for several checks to step up) stops
// the viewer from flickering between two levels.

import type { QualityPreset } from "./device";

export interface Level {
  pixelRatio: number;
  lodSplatScale: number;
}

export function buildLevels(preset: QualityPreset, devicePixelRatio: number): Level[] {
  const top = Math.min(preset.maxPixelRatio, devicePixelRatio);
  const s = preset.lodSplatScale;
  const raw: Level[] = [
    { pixelRatio: top, lodSplatScale: s },
    { pixelRatio: top * 0.8, lodSplatScale: s },
    { pixelRatio: top * 0.8, lodSplatScale: s * 0.7 },
    { pixelRatio: top * 0.65, lodSplatScale: s * 0.7 },
    { pixelRatio: top * 0.65, lodSplatScale: s * 0.5 },
    { pixelRatio: top * 0.55, lodSplatScale: s * 0.35 },
  ];
  return raw.map((l) => ({
    pixelRatio: Math.max(0.5, Math.round(l.pixelRatio * 100) / 100),
    lodSplatScale: Math.max(0.15, Math.round(l.lodSplatScale * 100) / 100),
  }));
}

export interface AdaptiveOptions {
  levels: Level[];
  apply: (level: Level) => void;
  sampleFps: () => number | null;
  resetSamples: () => void;
  log: (msg: string) => void;
  lowFps?: number;
  highFps?: number;
  intervalMs?: number;
}

export class AdaptiveQuality {
  index = 0;
  private lastCheck = 0;
  private goodChecks = 0;
  private lowFps: number;
  private highFps: number;
  private interval: number;

  constructor(private o: AdaptiveOptions) {
    this.lowFps = o.lowFps ?? 26;
    this.highFps = o.highFps ?? 52;
    this.interval = o.intervalMs ?? 1500;
    o.apply(o.levels[0]);
  }

  get level(): Level {
    return this.o.levels[this.index];
  }

  // Call once per rendered frame while the user is interacting.
  tick(now: number) {
    if (now - this.lastCheck < this.interval) return;
    this.lastCheck = now;
    const fps = this.o.sampleFps();
    if (fps === null) return;
    if (fps < this.lowFps && this.index < this.o.levels.length - 1) {
      this.move(+1, fps, now);
    } else if (fps > this.highFps && this.index > 0) {
      this.goodChecks += 1;
      if (this.goodChecks >= 3) this.move(-1, fps, now);
    } else {
      this.goodChecks = 0;
    }
  }

  private move(step: number, fps: number, now: number) {
    this.index += step;
    this.goodChecks = 0;
    const l = this.level;
    this.o.apply(l);
    this.o.resetSamples();
    this.o.log(`${(now / 1000).toFixed(1)}s: ${Math.round(fps)} fps, ${step > 0 ? "lowered" : "raised"} to ${l.pixelRatio}x pixels, ${l.lodSplatScale} budget`);
  }
}

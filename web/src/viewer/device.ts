// Pick a starting quality level for the phone we are running on.
//
// Gaussian splats are mostly limited by fill rate: every splat is a blended,
// overlapping quad, so a cheap phone GPU at full pixel density chokes long
// before it runs out of memory. We start from a guess based on what the
// browser tells us, then the adaptive controller corrects the guess from
// measured frame times.

export type Tier = "low" | "mid" | "high";

export interface QualityPreset {
  tier: Tier;
  // Multiplier on Spark's platform splat budget (1M on Android, 1.5M iOS, 2.5M desktop).
  lodSplatScale: number;
  // Cap on devicePixelRatio. Rendering at 3x on a budget phone triples the blending work.
  maxPixelRatio: number;
}

export const PRESETS: Record<Tier, QualityPreset> = {
  low: { tier: "low", lodSplatScale: 0.45, maxPixelRatio: 1 },
  mid: { tier: "mid", lodSplatScale: 0.75, maxPixelRatio: 1.5 },
  high: { tier: "high", lodSplatScale: 1, maxPixelRatio: 2 },
};

export interface DeviceInfo {
  tier: Tier;
  reason: string;
  gpu: string;
  cores: number;
  memoryGb: number | null;
  mobile: boolean;
}

// GPUs common in Indian budget phones. Matching on the renderer string is crude
// but it is the only hardware hint browsers expose.
const LOW_GPU = /(mali-(t|g3|g5[0-9]\b|g52|g57)|adreno \(tm\) (3|4|50|51|60|61)\d|powervr ge|swiftshader|llvmpipe)/i;
const HIGH_GPU = /(apple gpu|adreno \(tm\) (7[3-9]\d|8\d\d)|mali-g7[1-9]|mali-g7\d\d|immortalis|nvidia|radeon|geforce|intel\(r\) (iris|arc))/i;

function gpuName(): string {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return "no webgl2";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const name = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return String(name);
  } catch {
    return "unknown";
  }
}

export function detectDevice(override?: string | null): DeviceInfo {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const cores = nav.hardwareConcurrency || 4;
  const memoryGb = typeof nav.deviceMemory === "number" ? nav.deviceMemory : null;
  const mobile = /android|iphone|ipad|mobile/i.test(nav.userAgent);
  const gpu = gpuName();
  const base = { gpu, cores, memoryGb, mobile };

  if (override === "low" || override === "mid" || override === "high") {
    return { ...base, tier: override, reason: "set by ?tier=" };
  }
  if (LOW_GPU.test(gpu)) return { ...base, tier: "low", reason: `GPU ${gpu}` };
  if ((memoryGb !== null && memoryGb <= 3) || cores <= 4) {
    return { ...base, tier: "low", reason: `${cores} cores, ${memoryGb ?? "?"} GB` };
  }
  if (HIGH_GPU.test(gpu) && (!mobile || (memoryGb ?? 6) >= 6)) {
    return { ...base, tier: "high", reason: `GPU ${gpu}` };
  }
  return { ...base, tier: mobile ? "mid" : "high", reason: mobile ? "mobile default" : "desktop default" };
}

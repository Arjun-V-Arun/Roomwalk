// The room viewer. Framework-free on purpose: the Next.js page wraps it, and
// the same module could back an embeddable widget on someone else's site.
//
// Rendering is on demand. A phone showing a still room should not burn battery
// re-drawing the same frame 60 times a second, so we draw only when the camera
// moves, a fly-to animation runs, or Spark reports new work (sorting finished,
// more LoD chunks arrived).

import * as THREE from "three";
import { SparkControls, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { AdaptiveQuality, buildLevels } from "./adaptive";
import { detectDevice, PRESETS } from "./device";
import { Metrics, roomVisible } from "./metrics";
import type { Delivery, Pose, RoomManifest, Vec3 } from "./types";

const CLEAR = new THREE.Color(0x2b2f33);
const CLEAR_RGB: [number, number, number] = [0x2b, 0x2f, 0x33];

export interface ViewerOptions {
  container: HTMLElement;
  manifest: RoomManifest;
  baseUrl: string;
  delivery?: Delivery | "auto";
  tierOverride?: string | null;
  // Benchmark mode: render every frame and turn the camera slowly.
  bench?: boolean;
  onProgress?: (loaded: number, total: number) => void;
  onFirstView?: () => void;
  onInteract?: () => void;
}

export interface Viewer {
  metrics: Metrics;
  delivery: Delivery;
  flyTo(pose: Pose): void;
  currentPose(): Pose;
  dispose(): void;
}

const v3 = (a: Vec3) => new THREE.Vector3(a[0], a[1], a[2]);
const round2 = (n: number) => Math.round(n * 100) / 100;
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export async function createViewer(o: ViewerOptions): Promise<Viewer> {
  const m = o.manifest;
  const delivery: Delivery =
    o.delivery === "sog" || o.delivery === "rad" ? o.delivery : m.splats.rad ? "rad" : "sog";
  if (delivery === "rad" && !m.splats.rad) throw new Error("This room has no streaming (RAD) version");

  const device = detectDevice(o.tierOverride);
  const preset = PRESETS[device.tier];
  const metrics = new Metrics(`/rooms/${m.id}/`, {
    delivery, tier: device.tier, tierReason: device.reason, gpu: device.gpu, splatCount: m.splats.count,
  });
  metrics.mark("viewerStartMs");

  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
  renderer.setClearColor(CLEAR);
  renderer.domElement.style.touchAction = "none";
  renderer.domElement.style.display = "block";
  o.container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(70, 1, 0.03, 80);
  let dirty = true;
  const spark = new SparkRenderer({
    renderer,
    lodSplatScale: preset.lodSplatScale,
    onDirty: () => {
      dirty = true;
    },
  });
  scene.add(spark);

  const resize = () => {
    const w = o.container.clientWidth || 1;
    const h = o.container.clientHeight || 1;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    // Portrait phones need a taller field of view, or a 3 m room feels like a corridor.
    camera.fov = w < h ? 78 : 62;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    dirty = true;
  };
  const ro = new ResizeObserver(resize);
  ro.observe(o.container);
  resize();

  const adaptive = new AdaptiveQuality({
    levels: buildLevels(preset, window.devicePixelRatio || 1),
    apply: (l) => {
      renderer.setPixelRatio(l.pixelRatio);
      spark.lodSplatScale = l.lodSplatScale;
      resize();
      metrics.set({ pixelRatio: l.pixelRatio, lodSplatScale: l.lodSplatScale });
    },
    sampleFps: () => metrics.recentMedianFps(),
    resetSamples: () => metrics.resetFrames(),
    log: (msg) => metrics.set({ adaptations: [...metrics.snapshot.adaptations, msg] }),
  });

  const file = delivery === "rad" ? m.splats.rad!.file : m.splats.sog.file;
  const mesh = new SplatMesh(
    delivery === "rad"
      ? { url: o.baseUrl + file, paged: true }
      : {
          url: o.baseUrl + file,
          lod: true,
          onProgress: (e: ProgressEvent) => {
            // Spark downloads inside a worker, which page-level resource timing
            // cannot see, so the progress callback is our byte count for SOG.
            metrics.set({ roomBytes: e.loaded });
            o.onProgress?.(e.loaded, e.total || m.splats.sog.bytes);
          },
        },
  );
  const t = m.transform;
  mesh.position.copy(v3(t.position));
  mesh.quaternion.set(t.quaternion[0], t.quaternion[1], t.quaternion[2], t.quaternion[3]);
  mesh.scale.setScalar(t.scale);
  scene.add(mesh);

  const bounds = new THREE.Box3(v3(m.view.bounds.min), v3(m.view.bounds.max));
  camera.position.copy(v3(m.view.start.position));
  camera.lookAt(v3(m.view.start.target));

  // Fly-to animation between two camera poses.
  let anim: { from: THREE.Vector3; to: THREE.Vector3; q0: THREE.Quaternion; q1: THREE.Quaternion; t0: number; ms: number } | null = null;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  const flyTo = (pose: Pose) => {
    const to = v3(pose.position).clamp(bounds.min, bounds.max);
    // Camera-style look-at: the camera's -z axis faces the target. (Object3D.lookAt
    // on a plain object would point +z at it and leave us facing backwards.)
    const facing = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(to, v3(pose.target), new THREE.Vector3(0, 1, 0)),
    );
    if (reduceMotion) {
      camera.position.copy(to);
      camera.quaternion.copy(facing);
      dirty = true;
      return;
    }
    const dist = camera.position.distanceTo(to);
    anim = {
      from: camera.position.clone(), to, q0: camera.quaternion.clone(), q1: facing,
      t0: performance.now(), ms: Math.min(1400, 650 + dist * 250),
    };
  };

  // Double tap: step toward the tapped spot, stopping short of whatever is there.
  const raycaster = new THREE.Raycaster();
  const stepToward = (px: THREE.Vector2) => {
    const ndc = new THREE.Vector2(
      (px.x / renderer.domElement.clientWidth) * 2 - 1,
      -(px.y / renderer.domElement.clientHeight) * 2 + 1,
    );
    raycaster.setFromCamera(ndc, camera);
    const dir = raycaster.ray.direction.clone();
    let reach = 1.2;
    try {
      const hits = raycaster.intersectObject(mesh, false);
      if (hits.length) reach = Math.max(0, Math.min(2.5, hits[0].distance - 0.7));
    } catch {
      // Raycasting is best effort; fall back to a fixed step.
    }
    const flat = new THREE.Vector3(dir.x, 0, dir.z).normalize();
    const to = camera.position.clone().addScaledVector(flat, reach);
    to.y = camera.position.y;
    const target = to.clone().add(new THREE.Vector3(dir.x, dir.y * 0.5, dir.z));
    flyTo({ position: [to.x, to.y, to.z], target: [target.x, target.y, target.z] });
  };

  const controls = new SparkControls({ canvas: renderer.domElement });
  const pc = controls.pointerControls;
  pc.pointerRollScale = 0;
  pc.pressMoveSpeed = 0.8;
  pc.doublePressMoveSpeed = 1.6;
  pc.triplePressMoveSpeed = 2.4;
  pc.doublePress = ({ position }) => stepToward(position);
  controls.fpsMovement.moveSpeed = 1.2;

  let interacted = false;
  const markInteract = () => {
    if (!interacted) {
      interacted = true;
      o.onInteract?.();
    }
  };
  renderer.domElement.addEventListener("pointerdown", markInteract);

  let splatsReady = false;
  let firstView = false;
  let disposed = false;
  let lastActive = 0;
  let renders = 0;
  // Hooks for scripts (benchmarks, recordings). Harmless for normal visitors.
  Object.assign(window as object, {
    __roomwalkRenders: () => renders,
    __roomwalkSetPose: (pose: Pose) => {
      camera.position.copy(v3(pose.position));
      camera.quaternion.setFromRotationMatrix(
        new THREE.Matrix4().lookAt(camera.position, v3(pose.target), new THREE.Vector3(0, 1, 0)),
      );
      dirty = true;
      return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve(renders))));
    },
  });

  mesh.initialized
    .then(() => {
      if (disposed) return;
      splatsReady = true;
      metrics.mark("splatsReadyMs");
      dirty = true;
    })
    .catch((err: unknown) => console.error("RoomWalk: could not load splats", err));

  const gl = renderer.getContext() as WebGL2RenderingContext;
  let last = performance.now();

  renderer.setAnimationLoop((now: number) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    let moved = false;
    if (anim) {
      const k = Math.min(1, (now - anim.t0) / anim.ms);
      const e = easeInOut(k);
      camera.position.lerpVectors(anim.from, anim.to, e);
      camera.quaternion.slerpQuaternions(anim.q0, anim.q1, e);
      moved = true;
      if (k >= 1) anim = null;
      controls.update(new THREE.Object3D()); // keep the controls' clock in step
    } else if (o.bench && splatsReady) {
      camera.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), dt * 0.35);
      moved = true;
    } else {
      moved = controls.update(camera, camera);
    }
    if (moved) {
      camera.position.clamp(bounds.min, bounds.max);
      lastActive = now;
    }
    const active = moved || now - lastActive < 250;
    if (!(moved || dirty || !firstView || o.bench)) return;
    dirty = false;
    renderer.render(scene, camera);
    renders += 1;
    metrics.frame(now);
    if (active && firstView) adaptive.tick(now);
    if (!firstView && splatsReady && roomVisible(gl, CLEAR_RGB)) {
      firstView = true;
      metrics.mark("firstViewMs");
      o.onFirstView?.();
    }
  });

  return {
    metrics,
    delivery,
    flyTo,
    currentPose: () => {
      const fwd = new THREE.Vector3();
      camera.getWorldDirection(fwd);
      const p = camera.position;
      const tgt = p.clone().addScaledVector(fwd, 1.5);
      return {
        position: [round2(p.x), round2(p.y), round2(p.z)],
        target: [round2(tgt.x), round2(tgt.y), round2(tgt.z)],
      };
    },
    dispose: () => {
      disposed = true;
      renderer.setAnimationLoop(null);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", markInteract);
      scene.remove(mesh);
      mesh.dispose();
      spark.dispose?.();
      renderer.dispose();
      renderer.domElement.remove();
      metrics.dispose();
    },
  };
}

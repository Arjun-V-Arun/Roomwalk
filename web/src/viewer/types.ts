// Shapes of the files the capture pipeline writes into public/rooms/.
// Keep in sync with capture/roomwalk/publish.py.

export type Vec3 = [number, number, number];

export interface Pose {
  position: Vec3;
  target: Vec3;
}

export interface Hotspot extends Pose {
  id: string;
  label: string;
}

export interface Freshness {
  status: string;
  detail: string;
  issued_at?: string | null;
  recorded_at?: string | null;
  seen_at_s?: number | null;
}

export interface Finding {
  level: "fail" | "warn" | "info";
  code: string;
  message: string;
}

export interface RoomManifest {
  schema: number;
  id: string;
  title: string;
  area?: string | null;
  rent?: number | null;
  deposit?: number | null;
  notes?: string | null;
  synthetic: boolean;
  scannedAt?: string | null;
  publishedAt: string;
  freshness?: Freshness | null;
  capture?: { verdict: string; findings: Finding[]; summary: Record<string, unknown> } | null;
  splats: {
    count: number;
    sog: { file: string; bytes: number };
    rad?: { file: string; bytes: number } | null;
  };
  poster?: string | null;
  transform: { position: Vec3; quaternion: [number, number, number, number]; scale: number };
  view: { start: Pose; bounds: { min: Vec3; max: Vec3 }; eyeHeight: number };
  hotspots: Hotspot[];
}

export interface RoomSummary {
  id: string;
  title: string;
  area?: string | null;
  rent?: number | null;
  scannedAt?: string | null;
  poster?: string | null;
  synthetic?: boolean;
  verified: boolean;
  splatCount: number;
}

export type Delivery = "sog" | "rad";

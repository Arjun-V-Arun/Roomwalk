"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { Trust } from "@/components/Stamp";
import { megabytes, roomUrl, rupees, shortDate } from "@/lib/format";
import type { MetricsSnapshot } from "@/viewer/metrics";
import type { Delivery, Hotspot, RoomManifest } from "@/viewer/types";
import type { Viewer } from "@/viewer/viewer";

// What each capture problem means for someone looking at the room.
const GAP_NOTES: Record<string, string> = {
  blurry: "Some of the video was blurry, so parts of the room may look smeared.",
  dark: "Parts of the room were dark when filmed and may look noisy.",
  blown_out: "The window was very bright, so the view outside may be washed out.",
  low_texture: "Large plain walls can come out wavy or with holes. That is the scan, not the wall.",
  coverage: "Not every corner was filmed, so you may see holes behind furniture.",
  too_fast: "The camera moved quickly in places, so some details are soft.",
  short: "The video was short, so some areas were seen from only one angle.",
};

function hasWebGL2(): boolean {
  try {
    return !!document.createElement("canvas").getContext("webgl2");
  } catch {
    return false;
  }
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "").slice(0, 32) || "spot";
}

export function RoomPage({ manifest: m }: { manifest: RoomManifest }) {
  const holder = useRef<HTMLDivElement>(null);
  const [viewer, setViewer] = useState<Viewer | null>(null);
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null);
  const [ready, setReady] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<"webgl" | "load" | null>(null);
  const [hint, setHint] = useState(true);
  const [details, setDetails] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [stats, setStats] = useState(false);
  const [edit, setEdit] = useState(false);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [spots, setSpots] = useState<Hotspot[]>(m.hotspots);
  const [copied, setCopied] = useState<string | null>(null);
  const [bare, setBare] = useState(false);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    setStats(q.has("stats") || q.has("bench"));
    setEdit(q.has("edit"));
    setBare(q.get("ui") === "0");
    if (!hasWebGL2()) {
      setError("webgl");
      return;
    }
    let cancelled = false;
    let v: Viewer | null = null;
    const slowTimer = window.setTimeout(() => setSlow(true), 20000);
    (async () => {
      try {
        // Loaded on the client only: three.js and Spark need a browser.
        const { createViewer } = await import("@/viewer/viewer");
        if (cancelled || !holder.current) return;
        const fmt = q.get("format");
        v = await createViewer({
          container: holder.current,
          manifest: m,
          baseUrl: roomUrl(m.id),
          delivery: fmt === "sog" || fmt === "rad" ? (fmt as Delivery) : "auto",
          tierOverride: q.get("tier"),
          bench: q.has("bench"),
          onProgress: (loaded, total) => setProgress({ loaded, total }),
          onFirstView: () => {
            window.clearTimeout(slowTimer);
            setReady(true);
            window.setTimeout(() => setHint(false), 7000);
          },
          onInteract: () => setHint(false),
        });
        if (cancelled) {
          v.dispose();
          return;
        }
        setViewer(v);
      } catch (err) {
        console.error(err);
        if (!cancelled) setError("load");
      }
    })();
    return () => {
      cancelled = true;
      window.clearTimeout(slowTimer);
      v?.dispose();
      setViewer(null);
    };
  }, [m]);

  useEffect(() => {
    if (!viewer || !stats) return;
    const unsubscribe = viewer.metrics.subscribe(setMetrics);
    return () => {
      unsubscribe();
    };
  }, [viewer, stats]);

  const jump = useCallback(
    (h: Hotspot) => {
      viewer?.flyTo(h);
      setActive(h.id);
      setHint(false);
    },
    [viewer],
  );

  const copy = async (label: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
    } catch {
      window.prompt("Copy this:", text);
    }
  };

  const saveSpot = () => {
    if (!viewer) return;
    const label = window.prompt("Name this spot, for example: Bathroom door");
    if (!label) return;
    let id = slug(label);
    while (spots.some((s) => s.id === id)) id += "-2";
    setSpots([...spots, { id, label, ...viewer.currentPose() }]);
  };

  const price = rupees(m.rent);
  const verified = m.freshness?.status === "verified";
  const gaps = (m.capture?.findings ?? []).filter((f) => f.level !== "info" && GAP_NOTES[f.code]);
  const posterUrl = m.poster ? roomUrl(m.id, m.poster) : null;
  const pct = progress && progress.total ? Math.min(100, (100 * progress.loaded) / progress.total) : null;

  return (
    <main className={`viewer-page${bare ? " bare" : ""}`}>
      <div ref={holder} className="viewer-canvas" onPointerDown={() => setActive(null)} />
      <div
        className={`poster${ready ? " gone" : ""}`}
        style={posterUrl ? { backgroundImage: `url(${posterUrl})` } : undefined}
        aria-hidden
      />
      {!ready && !error && (
        <>
          <div
            className={`progress${pct === null ? " indeterminate" : ""}`}
            style={{ width: `${pct ?? 0}%` }}
            role="progressbar"
            aria-label="Loading the room"
            aria-valuenow={pct === null ? undefined : Math.round(pct)}
          />
          <div className="loading" aria-live="polite">
            <p>
              {progress && progress.total
                ? `Loading the room: ${megabytes(progress.loaded)} of ${megabytes(progress.total)}`
                : "Loading the room"}
              {slow && ". This is taking a while on this connection."}
            </p>
          </div>
        </>
      )}

      {error && (
        <div className="error" role="alert">
          <h2 className="display">{error === "webgl" ? "This browser can't show 3D rooms" : "The room didn't load"}</h2>
          <p>
            {error === "webgl"
              ? "It needs WebGL 2. Open the link in Chrome or Safari, or update your browser."
              : "Check your connection and try again."}
          </p>
          {error === "load" && (
            <p>
              <button className="text-btn" onClick={() => window.location.reload()}>
                Try again
              </button>
            </p>
          )}
        </div>
      )}

      <header className="topbar">
        <Link href="/" className="icon-btn" aria-label="All rooms">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 5l-7 7 7 7" />
          </svg>
        </Link>
        <div>
          <h1 className="display">{m.title}</h1>
          {m.area && <div className="sub">{m.area}</div>}
        </div>
        <div className={verified && !m.synthetic ? "slip" : undefined}>
          <Trust verified={verified} synthetic={m.synthetic} scannedAt={m.scannedAt} />
        </div>
      </header>

      {ready && (
        <p className={`hint${hint ? "" : " gone"}`} aria-hidden={!hint}>
          Drag to look around. Pinch to move. Double-tap a spot to walk there.
        </p>
      )}

      {stats && (
        <section className="panel" aria-label="Performance">
          <h2 className="display">Performance</h2>
          {metrics ? (
            <>
              <dl>
                <dt>First view</dt>
                <dd>{metrics.firstViewMs !== null ? `${(metrics.firstViewMs / 1000).toFixed(2)} s` : "waiting"}</dd>
                <dt>Room data</dt>
                <dd>{metrics.delivery === "rad" ? "streamed in chunks" : megabytes(metrics.roomBytes)}</dd>
                <dt>Format</dt>
                <dd>{metrics.delivery === "rad" ? "RAD, streamed" : "SOG, one file"}</dd>
                <dt>Frame rate</dt>
                <dd>{metrics.fps !== null ? `${metrics.fps} fps, p95 ${metrics.p95FrameMs} ms` : "move to measure"}</dd>
                <dt>Device tier</dt>
                <dd>{metrics.tier}</dd>
                <dt>Detail</dt>
                <dd>
                  {metrics.pixelRatio}x pixels, {metrics.lodSplatScale} budget
                </dd>
                <dt>Splats</dt>
                <dd>{metrics.splatCount.toLocaleString("en-IN")}</dd>
              </dl>
              {metrics.adaptations.length > 0 && (
                <ol>
                  {metrics.adaptations.slice(-4).map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ol>
              )}
              <button
                className="text-btn"
                onClick={() =>
                  copy("stats", JSON.stringify({ ...metrics, room: m.id, userAgent: navigator.userAgent, at: new Date().toISOString() }, null, 2))
                }
              >
                {copied === "stats" ? "Copied" : "Copy numbers"}
              </button>
            </>
          ) : (
            <p>Starting</p>
          )}
        </section>
      )}

      {edit && viewer && (
        <section className="panel" aria-label="Edit spots" style={{ top: "auto", bottom: "11rem" }}>
          <h2 className="display">Spots</h2>
          <ul>
            {spots.map((s) => (
              <li key={s.id}>
                {s.label}{" "}
                <button className="text-btn" style={{ minHeight: 28, padding: "0 0.5rem" }} onClick={() => setSpots(spots.filter((x) => x.id !== s.id))}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
            <button className="text-btn" onClick={saveSpot}>
              Save this view
            </button>
            <button className="text-btn" onClick={() => copy("spots", JSON.stringify(spots, null, 2))}>
              {copied === "spots" ? "Copied" : "Copy spots"}
            </button>
            <button className="text-btn" onClick={() => copy("start", JSON.stringify(viewer.currentPose(), null, 2))}>
              {copied === "start" ? "Copied" : "Copy start view"}
            </button>
          </div>
        </section>
      )}

      <div className="dock">
        {spots.length > 0 && (
          <nav className="chips" aria-label="Jump to a spot">
            {spots.map((h) => (
              <button key={h.id} className="chip" aria-pressed={active === h.id} onClick={() => jump(h)} disabled={!viewer}>
                {h.label}
              </button>
            ))}
          </nav>
        )}
        <section className="sheet" aria-label="Room details">
          <div className="sheet-row">
            <div className="sheet-price num">
              {price ? (
                <>
                  {price} <span>a month</span>
                </>
              ) : (
                <span>Rent on request</span>
              )}
            </div>
            <button className="text-btn" aria-expanded={details} onClick={() => setDetails(!details)}>
              {details ? "Hide details" : "Details"}
            </button>
          </div>
          {details && (
            <div className="details">
              <dl>
                {m.deposit ? (
                  <>
                    <dt>Deposit</dt>
                    <dd className="money">{rupees(m.deposit)}</dd>
                  </>
                ) : null}
                {m.area && (
                  <>
                    <dt>Area</dt>
                    <dd>{m.area}</dd>
                  </>
                )}
                {m.scannedAt && !m.synthetic && (
                  <>
                    <dt>Filmed</dt>
                    <dd>{shortDate(m.scannedAt)}</dd>
                  </>
                )}
              </dl>
              {m.notes && <p>{m.notes}</p>}
              <div>
                <h2 className="display">About this scan</h2>
                <p>
                  {m.synthetic
                    ? "This room is computer-generated so you can try the viewer before real rooms are listed. It is not a real place."
                    : verified
                      ? "The tenant filmed our one-time code at the start of this video, and the phone recorded it within two days of us sending it. That shows the room looked like this recently. It does not prove the address, so visit or video-call before you pay."
                      : "This video did not include our one-time code, so we can't say when it was filmed. Ask for a live video call before you pay anything."}
                </p>
              </div>
              {gaps.length > 0 && (
                <div>
                  <h2 className="display">Known gaps in the 3D</h2>
                  <ul>
                    {gaps.map((g) => (
                      <li key={g.code}>{GAP_NOTES[g.code]}</li>
                    ))}
                  </ul>
                </div>
              )}
              <p>
                <button className="text-btn" onClick={() => setStats(!stats)}>
                  {stats ? "Hide performance" : "Show performance"}
                </button>
              </p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

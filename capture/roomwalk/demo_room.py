"""A synthetic PG room built directly out of Gaussian splats.

This exists so the viewer, the compression step and the benchmarks can run
before anyone has filmed a real room. It is not a reconstruction: every splat
is placed by hand on flat surfaces, with simple lighting baked into the colours.
The web app labels it as synthetic.

Coordinates: metres, Y up, floor at y = 0, room centred on x = z = 0.
Output: a standard 3D Gaussian Splatting PLY (SH degree 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SH_C0 = 0.28209479177387814
W, D, H = 3.0, 3.6, 2.9  # room width (x), depth (z), height (y)
X0, X1, Z0, Z1 = -W / 2, W / 2, -D / 2, D / 2
WINDOW = (-0.6, 0.6, 0.9, 2.1)  # x0, x1, y0, y1 on the back wall


@dataclass
class Builder:
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))
    pos: list = field(default_factory=list)
    col: list = field(default_factory=list)
    scl: list = field(default_factory=list)
    rot: list = field(default_factory=list)
    opa: list = field(default_factory=list)

    def rect(self, origin, u, v, color, spacing=0.016, mask=None, opacity=0.93, thickness=0.08, lit=True):
        """Fill the parallelogram origin + a*u + b*v (a, b in [0, 1]) with flat splats.

        color(a_m, b_m) gets local coordinates in metres and returns (N, 3) RGB in 0..1.
        The surface normal is u x v, so order u and v to face into the room.
        """
        origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        nu, nv = max(1, int(lu / spacing)), max(1, int(lv / spacing))
        a = (np.arange(nu) + 0.5)[:, None].repeat(nv, 1).ravel() / nu
        b = (np.arange(nv) + 0.5)[None, :].repeat(nu, 0).ravel() / nv
        a = np.clip(a + self.rng.uniform(-0.35, 0.35, a.size) / nu, 0, 1)
        b = np.clip(b + self.rng.uniform(-0.35, 0.35, b.size) / nv, 0, 1)
        am, bm = a * lu, b * lv
        if mask is not None:
            keep = mask(am, bm)
            a, b, am, bm = a[keep], b[keep], am[keep], bm[keep]
        if a.size == 0:
            return
        p = origin + a[:, None] * u + b[:, None] * v
        rgb = np.clip(np.asarray(color(am, bm), float), 0, 1)
        uh, vh = u / lu, v / lv
        n = np.cross(uh, vh)
        n /= np.linalg.norm(n)
        vh = np.cross(n, uh)  # re-orthogonalise for slanted shapes
        if lit:
            rgb = rgb * shade(p, n)[:, None]
        rgb = rgb * (1 + self.rng.normal(0, 0.012, (len(p), 1)))
        q = matrix_to_quat(np.stack([uh, vh, n], axis=1))
        sx = spacing * 0.68 * self.rng.uniform(0.9, 1.1, len(p))
        sy = spacing * 0.68 * self.rng.uniform(0.9, 1.1, len(p))
        sz = np.full(len(p), spacing * thickness)
        self.pos.append(p)
        self.col.append(rgb)
        self.scl.append(np.stack([sx, sy, sz], 1))
        self.rot.append(np.repeat(q[None], len(p), 0))
        self.opa.append(np.full(len(p), opacity))

    def box(self, lo, hi, color, spacing=0.016, skip=(), **kw):
        """Axis-aligned box. Faces point outward; name faces in skip to leave them out."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        faces = {
            "top": ((x0, y1, z1), (dx, 0, 0), (0, 0, -dz)),
            "bottom": ((x0, y0, z0), (dx, 0, 0), (0, 0, dz)),
            "front": ((x0, y0, z1), (dx, 0, 0), (0, dy, 0)),
            "back": ((x1, y0, z0), (-dx, 0, 0), (0, dy, 0)),
            "left": ((x0, y0, z0), (0, 0, dz), (0, dy, 0)),
            "right": ((x1, y0, z1), (0, 0, -dz), (0, dy, 0)),
        }
        for name, (o, u, v) in faces.items():
            if name not in skip:
                self.rect(o, u, v, color if callable(color) else solid(color), spacing, **kw)

    def arrays(self):
        return (np.concatenate(self.pos), np.concatenate(self.col), np.concatenate(self.scl),
                np.concatenate(self.rot), np.concatenate(self.opa))


def solid(rgb):
    rgb = np.asarray(rgb, float)
    return lambda a, b: np.repeat(rgb[None], len(a), 0)


def texture(img_rgb: np.ndarray, width_m: float, height_m: float, flip_v=True):
    """Colour function that samples an image stretched over a rectangle."""
    h, w = img_rgb.shape[:2]

    def f(a, b):
        x = np.clip((a / width_m * w).astype(int), 0, w - 1)
        y = (b / height_m * h).astype(int)
        y = np.clip(h - 1 - y if flip_v else y, 0, h - 1)
        return img_rgb[y, x] / 255.0

    return f


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = local axes in world) to quaternion (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def shade(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Baked lighting: daylight from the window, a tube light, and soft corner darkening."""
    win = np.array([0.0, 1.5, Z0])
    to_win = win - p
    dist = np.linalg.norm(to_win, axis=1) + 1e-6
    lam_win = np.clip((to_win / dist[:, None]) @ n, 0, 1)
    day = 0.55 * lam_win / (1 + 0.18 * dist**2)
    tube = np.array([X1 - 0.02, 2.3, 0.4])
    to_tube = tube - p
    dt = np.linalg.norm(to_tube, axis=1) + 1e-6
    lamp = 0.35 * np.clip((to_tube / dt[:, None]) @ n, 0, 1) / (1 + 0.25 * dt**2)
    # Corner and floor-edge occlusion.
    d_side = np.minimum.reduce([p[:, 0] - X0, X1 - p[:, 0], p[:, 2] - Z0, Z1 - p[:, 2]])
    d_floor, d_ceil = p[:, 1], H - p[:, 1]
    ao = 1 - 0.18 * np.exp(-np.clip(d_side, 0, None) / 0.18) - 0.12 * np.exp(-np.clip(d_floor, 0, None) / 0.12)
    ao -= 0.08 * np.exp(-np.clip(d_ceil, 0, None) / 0.2)
    return np.clip((0.62 + day + lamp) * ao, 0.25, 1.25)


def calendar_image() -> np.ndarray:
    img = np.full((520, 380, 3), 250, np.uint8)
    cv2.rectangle(img, (0, 0), (380, 230), (200, 120, 60), -1)  # photo block (BGR)
    cv2.circle(img, (110, 120), 55, (60, 190, 250), -1)
    cv2.rectangle(img, (190, 60), (340, 200), (70, 140, 60), -1)
    cv2.rectangle(img, (0, 230), (380, 280), (40, 40, 190), -1)
    cv2.putText(img, "NOVEMBER 2026", (40, 266), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    for i in range(30):
        r, c = divmod(i + 6, 7)
        x, y = 18 + c * 51, 312 + r * 42
        cv2.putText(img, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 40, 200) if c == 0 else (40, 40, 40), 2)
    return img[:, :, ::-1].copy()


def poster_image() -> np.ndarray:
    h, w = 420, 600
    y = np.linspace(0, 1, h)[:, None, None]
    sky = (1 - y) * np.array([250, 190, 120]) + y * np.array([240, 130, 90])
    img = np.broadcast_to(sky, (h, w, 3)).astype(np.uint8).copy()
    for k, (col, base) in enumerate([((90, 70, 110), 0.55), ((60, 50, 80), 0.7), ((35, 30, 50), 0.85)]):
        xs = np.arange(w)
        ridge = (base - 0.12 * np.abs(np.sin(xs / (70 + 25 * k) + k)) - 0.06 * np.sin(xs / 23.0)) * h
        for x in xs:
            img[int(ridge[x]):, x] = col
    cv2.circle(img, (450, 120), 40, (255, 240, 200), -1)
    return img


def build() -> Builder:
    bld = Builder()
    tile = np.array([0.86, 0.81, 0.72])

    def floor_col(a, b):
        x, z = a + X0, b + Z0
        tx, tz = np.floor(x / 0.6), np.floor(z / 0.6)
        var = 0.035 * np.sin(tx * 12.9898 + tz * 78.233) ** 2
        vein = 0.025 * np.sin(x * 9 + np.sin(z * 7) * 2)
        c = tile[None] * (1 - var - vein)[:, None]
        grout = (np.abs(x / 0.6 - np.round(x / 0.6)) < 0.006) | (np.abs(z / 0.6 - np.round(z / 0.6)) < 0.006)
        c[grout] = [0.55, 0.52, 0.47]
        return c

    bld.rect((X0, 0, Z1), (W, 0, 0), (0, 0, -D), floor_col)
    bld.rect((X0, H, Z0), (W, 0, 0), (0, 0, D), solid([0.94, 0.94, 0.92]))

    paint = np.array([0.79, 0.86, 0.77])

    def wall_col(a, b):
        c = np.repeat(paint[None], len(a), 0) * (1 - 0.02 * np.sin(a * 3.1 + b * 1.7))[:, None]
        skirting = b < 0.1
        c[skirting] = tile * 0.95
        c[(b >= 0.1) & (b < 0.11)] = [0.6, 0.57, 0.5]
        return c

    def not_window(a, b):
        x = a + X0
        return ~((x > WINDOW[0]) & (x < WINDOW[1]) & (b > WINDOW[2]) & (b < WINDOW[3]))

    def not_door(a, b):
        x = X1 - a
        return ~((x > 0.45) & (x < 1.35) & (b < 2.1))

    bld.rect((X0, 0, Z0), (W, 0, 0), (0, H, 0), wall_col, mask=not_window)  # back wall, window
    bld.rect((X1, 0, Z1), (-W, 0, 0), (0, H, 0), wall_col, mask=not_door)  # front wall, door
    bld.rect((X0, 0, Z1), (0, 0, -D), (0, H, 0), wall_col)  # left wall
    bld.rect((X1, 0, Z0), (0, 0, D), (0, H, 0), wall_col)  # right wall

    # Window: sky and a neighbouring building, then frame, grill and curtains.
    def view(a, b):
        y = b / (WINDOW[3] - WINDOW[2])
        c = (1 - y)[:, None] * np.array([0.88, 0.93, 1.0]) + y[:, None] * np.array([0.55, 0.74, 0.96])
        bx = (a > 0.25) & (a < 0.95) & (b < 0.55)
        c[bx] = [0.93, 0.86, 0.72]
        win_rows = bx & ((b * 10 % 1.2) < 0.4) & ((a * 10 % 1.4) < 0.6)
        c[win_rows] = [0.45, 0.5, 0.58]
        return c

    wx0, wx1, wy0, wy1 = WINDOW
    bld.rect((wx0, wy0, Z0 - 0.04), (wx1 - wx0, 0, 0), (0, wy1 - wy0, 0), view, lit=False)
    frame = [0.95, 0.95, 0.93]
    for lo, hi in [((wx0 - 0.06, wy0 - 0.06), (wx1 + 0.06, wy0)), ((wx0 - 0.06, wy1), (wx1 + 0.06, wy1 + 0.06)),
                   ((wx0 - 0.06, wy0), (wx0, wy1)), ((wx1, wy0), (wx1 + 0.06, wy1))]:
        bld.box((lo[0], lo[1], Z0), (hi[0], hi[1], Z0 + 0.05), frame, spacing=0.01, skip=("back",))
    for x in np.arange(wx0 + 0.1, wx1, 0.12):
        bld.box((x - 0.006, wy0, Z0 + 0.02), (x + 0.006, wy1, Z0 + 0.032), [0.2, 0.2, 0.22], spacing=0.006)
    for y in (wy0 + 0.4, wy0 + 0.8):
        bld.box((wx0, y - 0.006, Z0 + 0.02), (wx1, y + 0.006, Z0 + 0.032), [0.2, 0.2, 0.22], spacing=0.006)

    def curtain(a, b):
        fold = 0.78 + 0.22 * np.sin(a * 55) ** 2
        return np.array([0.82, 0.6, 0.18])[None] * fold[:, None]

    for x0 in (wx0 - 0.42, wx1 + 0.02):
        bld.rect((x0, wy0 - 0.3, Z0 + 0.09), (0.4, 0, 0), (0, wy1 - wy0 + 0.45, 0), curtain, spacing=0.012)
    bld.box((wx0 - 0.5, wy1 + 0.14, Z0 + 0.06), (wx1 + 0.5, wy1 + 0.16, Z0 + 0.1), [0.35, 0.3, 0.25], spacing=0.01)

    # Door and switchboard on the front wall.
    def door_col(a, b):
        grain = 0.9 + 0.1 * np.sin(b * 40 + np.sin(a * 13) * 3)
        c = np.array([0.48, 0.29, 0.15])[None] * grain[:, None]
        groove = ((np.abs(a - 0.12) < 0.008) | (np.abs(a - 0.78) < 0.008) | (np.abs(b - 1.05) < 0.008)) & (b > 0.15) & (b < 1.95)
        c[groove] *= 0.6
        return c

    bld.rect((X1 - 0.45, 0, Z1 - 0.01), (-0.9, 0, 0), (0, 2.1, 0), door_col, spacing=0.012)
    bld.box((X1 - 1.27, 1.0, Z1 - 0.06), (X1 - 1.21, 1.04, Z1 - 0.01), [0.75, 0.74, 0.7], spacing=0.006)
    bld.box((0.0, 1.15, Z1 - 0.02), (0.26, 1.27, Z1), [0.96, 0.96, 0.94], spacing=0.006, skip=("back",))
    for i in range(4):
        x = 0.03 + i * 0.058
        bld.box((x, 1.19, Z1 - 0.03), (x + 0.03, 1.23, Z1 - 0.02), [0.15, 0.15, 0.15], spacing=0.004)

    # Bed along the left wall, head at the front.
    wood = [0.42, 0.27, 0.15]
    bld.box((X0, 0.0, -0.15), (X0 + 0.92, 0.42, Z1 - 0.05), wood, skip=("bottom",))

    def sheet(a, b):
        check = ((np.floor(a / 0.08) + np.floor(b / 0.08)) % 2).astype(float)
        return np.array([0.88, 0.9, 0.95])[None] - 0.12 * check[:, None] * np.array([0.6, 0.4, 0.0])[None]

    bld.box((X0 + 0.02, 0.42, -0.13), (X0 + 0.9, 0.58, Z1 - 0.07), sheet, skip=("bottom",))
    bld.box((X0 + 0.12, 0.58, Z1 - 0.55), (X0 + 0.8, 0.7, Z1 - 0.12), [0.97, 0.97, 0.98], spacing=0.01, skip=("bottom",))

    def blanket(a, b):
        stripe = (np.floor(a / 0.07) % 3).astype(int)
        return np.array([[0.7, 0.16, 0.2], [0.95, 0.85, 0.55], [0.18, 0.3, 0.55]])[stripe]

    bld.box((X0 + 0.04, 0.58, -0.12), (X0 + 0.88, 0.66, 0.35), blanket, spacing=0.012, skip=("bottom",))

    # Calendar and poster above the bed.
    cal = calendar_image()
    bld.rect((X0 + 0.01, 1.35, 0.95), (0, 0, -0.38), (0, 0.52, 0), texture(cal, 0.38, 0.52), spacing=0.006)
    post = poster_image()
    bld.rect((X0 + 0.01, 1.3, 0.2), (0, 0, -0.75), (0, 0.52, 0), texture(post, 0.75, 0.52), spacing=0.008)

    # Steel almirah against the right wall.
    def steel(a, b):
        c = np.array([0.6, 0.62, 0.64])[None] * (0.93 + 0.07 * np.sin(b * 2.0))[:, None]
        return c

    bld.box((X1 - 0.5, 0.0, 0.25), (X1, 1.85, 1.15), steel, skip=("bottom", "right"))
    bld.rect((X1 - 0.505, 0.05, 0.7), (0, 0, 0.004), (0, 1.75, 0), solid([0.25, 0.25, 0.27]), spacing=0.004)
    bld.box((X1 - 0.53, 0.95, 0.62), (X1 - 0.505, 1.15, 0.66), [0.2, 0.2, 0.22], spacing=0.005)

    # Study table, laptop, books and a plastic chair near the window.
    top = [0.55, 0.38, 0.22]
    bld.box((0.82, 0.72, Z0 + 0.02), (X1 - 0.02, 0.76, -0.95), top)
    for lx, lz in [(0.85, Z0 + 0.05), (X1 - 0.07, Z0 + 0.05), (0.85, -1.0), (X1 - 0.07, -1.0)]:
        bld.box((lx, 0.0, lz), (lx + 0.04, 0.72, lz + 0.04), [0.3, 0.2, 0.12], spacing=0.01)
    bld.box((0.95, 0.76, -1.55), (1.3, 0.78, -1.3), [0.22, 0.23, 0.25], spacing=0.008)
    bld.rect((0.95, 0.78, -1.56), (0.35, 0, 0), (0, 0.22, -0.03), solid([0.12, 0.14, 0.2]), spacing=0.006)
    for i, c in enumerate([[0.8, 0.2, 0.2], [0.2, 0.45, 0.75], [0.95, 0.8, 0.3]]):
        y = 0.76 + i * 0.035
        bld.box((1.32, y, -1.2 + i * 0.01), (1.46, y + 0.035, -1.0 + i * 0.01), c, spacing=0.006)
    blue = [0.18, 0.35, 0.65]
    bld.box((0.55, 0.43, -1.4), (0.95, 0.47, -1.0), blue, spacing=0.012)
    bld.box((0.55, 0.47, -1.0), (0.95, 0.95, -0.96), blue, spacing=0.012)
    for lx, lz in [(0.57, -1.38), (0.9, -1.38), (0.57, -1.02), (0.9, -1.02)]:
        bld.box((lx, 0, lz), (lx + 0.03, 0.43, lz + 0.03), blue, spacing=0.01)

    # Rug, tube light, wall clock and ceiling fan.
    def rug(a, b):
        ring = np.minimum.reduce([a, 0.9 - a, b, 1.2 - b])
        band = (np.floor(ring / 0.06) % 2).astype(bool)
        c = np.where(band[:, None], [[0.62, 0.18, 0.2]], [[0.92, 0.82, 0.62]])
        return c

    bld.rect((-0.55, 0.004, 1.35), (0.9, 0, 0), (0, 0, -1.2), rug, spacing=0.012)
    bld.box((X1 - 0.04, 2.28, -0.2), (X1, 2.32, 1.0), [1.0, 1.0, 0.97], spacing=0.008, lit=False)

    def clock(a, b):
        r = np.hypot(a - 0.15, b - 0.15)
        c = np.where((r > 0.13)[:, None], [[0.15, 0.15, 0.18]], [[0.97, 0.97, 0.95]])
        hand = (np.abs(a - 0.15) < 0.006) & (b > 0.15) & (b < 0.27)
        hand |= (np.abs(b - 0.15) < 0.006) & (a > 0.15) & (a < 0.24)
        c[hand] = [0.1, 0.1, 0.1]
        return c

    bld.rect((-0.15, 2.3, Z1 - 0.01), (0.3, 0, 0), (0, 0.3, 0), clock, spacing=0.005,
             mask=lambda a, b: np.hypot(a - 0.15, b - 0.15) < 0.15)
    bld.rect((0.15, 2.3, Z1 - 0.01), (-0.3, 0, 0), (0, 0.3, 0), clock, spacing=0.005,
             mask=lambda a, b: np.hypot(a - 0.15, b - 0.15) < 0.15)
    fan = [0.45, 0.32, 0.2]
    bld.box((-0.02, 2.5, -0.02), (0.02, H, 0.02), [0.9, 0.9, 0.88], spacing=0.008)
    bld.box((-0.12, 2.43, -0.12), (0.12, 2.52, 0.12), [0.88, 0.86, 0.82], spacing=0.008)
    for k in range(3):
        ang = k * 2 * np.pi / 3 + 0.3
        d = np.array([np.cos(ang), 0, np.sin(ang)])
        side = np.array([-np.sin(ang), 0, np.cos(ang)]) * 0.11
        start = np.array([0, 2.46, 0]) + d * 0.1 - side / 2
        bld.rect(start, side, d * 0.62, solid(fan), spacing=0.01)
        bld.rect(start + d * 0.62, side, -d * 0.62, solid(fan), spacing=0.01)
    return bld


def write_ply(path: str | Path, pos, rgb, scales, quats, opacity) -> Path:
    n = len(pos)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=[(k, "<f4") for k in names])
    data["x"], data["y"], data["z"] = pos.T
    dc = (rgb - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = dc.T
    o = np.clip(opacity, 1e-4, 1 - 1e-4)
    data["opacity"] = np.log(o / (1 - o))
    ls = np.log(scales)
    data["scale_0"], data["scale_1"], data["scale_2"] = ls.T
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = quats.T
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())
    return path


def make_demo_room(out_ply: str | Path) -> tuple[Path, int]:
    pos, rgb, scl, rot, opa = build().arrays()
    return write_ply(out_ply, pos, rgb, scl, rot, opa), len(pos)


if __name__ == "__main__":
    import sys

    path, n = make_demo_room(sys.argv[1] if len(sys.argv) > 1 else "demo-room.ply")
    print(f"wrote {n:,} splats to {path}")

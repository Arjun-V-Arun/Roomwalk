"""Freshness codes: proof that a scan was filmed after we asked for it.

The platform issues a signed token for one room. The tenant shows it as a QR
code at the start of the video. When the video is checked we decode the QR,
verify the signature and check that the recording happened inside the
validity window.

What this proves: the video was made after the code was issued, by someone we
sent the code to. What it does not prove: that this room is at the address on
the listing. That needs an in-person or document check.

Token format (ASCII, fits a small QR):  RW1:<room-id>:<issued-unix-seconds>:<signature>
Signature: first 20 hex chars of HMAC-SHA256(secret, "RW1:<room-id>:<issued>").
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREFIX = "RW1"
SIG_HEX_CHARS = 20
ROOM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
DEFAULT_MAX_AGE_H = 48.0
# Phones and laptops disagree about the time by a few minutes; tolerate that.
CLOCK_SKEW = timedelta(minutes=10)


def check_room_id(room_id: str) -> str:
    if not ROOM_ID_RE.match(room_id):
        raise ValueError(
            f"Room id {room_id!r} must be lowercase letters, digits and dashes, 1 to 48 characters"
        )
    return room_id


def secret_path() -> Path:
    return Path(os.environ.get("ROOMWALK_HOME", Path.home() / ".roomwalk")) / "secret.key"


def load_secret(create: bool = True) -> bytes:
    """Read the signing key from ROOMWALK_SECRET, or from ~/.roomwalk/secret.key."""
    env = os.environ.get("ROOMWALK_SECRET")
    if env:
        return env.encode()
    path = secret_path()
    if path.exists():
        return path.read_bytes().strip()
    if not create:
        raise FileNotFoundError(f"No signing key at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32).encode()
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows ignores POSIX modes.
    return key


def _sign(room_id: str, issued: int, secret: bytes) -> str:
    msg = f"{PREFIX}:{room_id}:{issued}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()[:SIG_HEX_CHARS]


@dataclass
class Token:
    room_id: str
    issued_at: int
    signature: str

    @property
    def text(self) -> str:
        return f"{PREFIX}:{self.room_id}:{self.issued_at}:{self.signature}"

    def issued_dt(self) -> datetime:
        return datetime.fromtimestamp(self.issued_at, tz=timezone.utc)


def issue(room_id: str, secret: bytes, now: datetime | None = None) -> Token:
    check_room_id(room_id)
    issued = int((now or datetime.now(timezone.utc)).timestamp())
    return Token(room_id, issued, _sign(room_id, issued, secret))


def parse(text: str) -> Token:
    parts = text.strip().split(":")
    if len(parts) != 4 or parts[0] != PREFIX:
        raise ValueError("not a RoomWalk code")
    _, room_id, issued, sig = parts
    check_room_id(room_id)
    if not issued.isdigit() or not re.fullmatch(r"[0-9a-f]{%d}" % SIG_HEX_CHARS, sig):
        raise ValueError("malformed RoomWalk code")
    return Token(room_id, int(issued), sig)


@dataclass
class FreshnessResult:
    # verified | not_found | not_checked | malformed | bad_signature | wrong_room | expired | before_issue
    status: str
    detail: str
    issued_at: str | None = None
    recorded_at: str | None = None
    seen_at_s: float | None = None

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict:
        return asdict(self)


def verify(
    text: str,
    room_id: str,
    secret: bytes,
    recorded_at: datetime,
    max_age_h: float = DEFAULT_MAX_AGE_H,
    seen_at_s: float | None = None,
) -> FreshnessResult:
    rec_iso = recorded_at.astimezone(timezone.utc).isoformat()
    try:
        tok = parse(text)
    except ValueError as exc:
        return FreshnessResult("malformed", str(exc), recorded_at=rec_iso, seen_at_s=seen_at_s)
    issued_iso = tok.issued_dt().isoformat()
    expected = _sign(tok.room_id, tok.issued_at, secret)
    # Constant-time compare so the check does not leak how many characters matched.
    if not hmac.compare_digest(expected, tok.signature):
        return FreshnessResult("bad_signature", "Code was not issued by this platform", issued_iso, rec_iso, seen_at_s)
    if tok.room_id != room_id:
        return FreshnessResult(
            "wrong_room", f"Code was issued for room {tok.room_id!r}, not {room_id!r}", issued_iso, rec_iso, seen_at_s
        )
    issued = tok.issued_dt()
    if recorded_at + CLOCK_SKEW < issued:
        return FreshnessResult("before_issue", "Video was recorded before the code existed", issued_iso, rec_iso, seen_at_s)
    if recorded_at - issued > timedelta(hours=max_age_h):
        return FreshnessResult(
            "expired", f"Video was recorded more than {max_age_h:g} h after the code was issued", issued_iso, rec_iso, seen_at_s
        )
    return FreshnessResult("verified", "Code is valid for this room and recording time", issued_iso, rec_iso, seen_at_s)


def render_qr(token: Token, out_png: str | Path, max_age_h: float = DEFAULT_MAX_AGE_H) -> Path:
    """Write a PNG the tenant can open on a second screen or print."""
    import qrcode
    from PIL import Image, ImageDraw, ImageFont

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=16, border=4)
    qr.add_data(token.text)
    qr.make(fit=True)
    code_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    w, h = code_img.size
    pad = 150
    canvas = Image.new("RGB", (w, h + pad), "white")
    canvas.paste(code_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=30)
        small = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = small = ImageFont.load_default()
    until = (token.issued_dt() + timedelta(hours=max_age_h)).strftime("%d %b %Y, %H:%M UTC")
    draw.text((w // 2, h + 20), f"RoomWalk scan code: {token.room_id}", fill="black", font=font, anchor="mt")
    draw.text((w // 2, h + 70), f"Film this first, up close. Valid until {until}.", fill="black", font=small, anchor="mt")
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def make_detector():
    import cv2

    # The ArUco-based detector copes better with glare from phone screens.
    if hasattr(cv2, "QRCodeDetectorAruco"):
        return cv2.QRCodeDetectorAruco()
    return cv2.QRCodeDetector()


def decode_qr(detector, image) -> str | None:
    try:
        text, _points, _ = detector.detectAndDecode(image)
    except Exception:  # OpenCV raises on some degenerate inputs
        return None
    return text or None

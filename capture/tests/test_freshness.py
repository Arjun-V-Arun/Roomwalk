from datetime import datetime, timedelta, timezone

import cv2
import pytest

from roomwalk import freshness as fr

NOW = datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc)
KEY = b"test-key"


def test_round_trip_verifies():
    tok = fr.issue("pg-101", KEY, now=NOW)
    res = fr.verify(tok.text, "pg-101", KEY, recorded_at=NOW + timedelta(hours=2))
    assert res.status == "verified"


@pytest.mark.parametrize(
    "mutate, room, recorded, status",
    [
        (lambda t: t[:-1] + ("0" if t[-1] != "0" else "1"), "pg-101", timedelta(hours=1), "bad_signature"),
        (lambda t: t, "pg-202", timedelta(hours=1), "wrong_room"),
        (lambda t: t, "pg-101", timedelta(hours=49), "expired"),
        (lambda t: t, "pg-101", -timedelta(hours=1), "before_issue"),
        (lambda t: "hello", "pg-101", timedelta(hours=1), "malformed"),
    ],
)
def test_rejections(mutate, room, recorded, status):
    tok = fr.issue("pg-101", KEY, now=NOW)
    res = fr.verify(mutate(tok.text), room, KEY, recorded_at=NOW + recorded)
    assert res.status == status


def test_small_clock_skew_is_tolerated():
    tok = fr.issue("pg-101", KEY, now=NOW)
    assert fr.verify(tok.text, "pg-101", KEY, recorded_at=NOW - timedelta(minutes=5)).verified


def test_other_key_fails():
    tok = fr.issue("pg-101", KEY, now=NOW)
    assert fr.verify(tok.text, "pg-101", b"other", recorded_at=NOW).status == "bad_signature"


@pytest.mark.parametrize("bad", ["", "UPPER", "has space", "-leading", "a" * 49, "x/y"])
def test_room_ids_are_validated(bad):
    with pytest.raises(ValueError):
        fr.check_room_id(bad)


def test_secret_is_created_once():
    a = fr.load_secret()
    b = fr.load_secret()
    assert a == b and len(a) == 64


def test_qr_png_decodes(tmp_path):
    tok = fr.issue("pg-101", KEY, now=NOW)
    png = fr.render_qr(tok, tmp_path / "code.png")
    img = cv2.imread(str(png))
    assert fr.decode_qr(fr.make_detector(), img) == tok.text

from datetime import datetime, timedelta, timezone

from roomwalk import freshness as fr
from roomwalk.quality import check_video


def codes(report):
    return {(f["level"], f["code"]) for f in report.findings}


def test_good_scan_passes_and_estimates_turn(tmp_video):
    r = check_video(str(tmp_video(seconds=40, turn_deg=380)))
    assert r.verdict == "PASS", r.findings
    assert r.summary["blurry_fraction"] < 0.05
    # Panning a known distance should read back as roughly the same turn.
    assert 320 <= r.summary["estimated_turn_deg"] <= 440


def test_blurry_scan_fails(tmp_video):
    r = check_video(str(tmp_video(seconds=40, blur=[(5, 35)])))
    assert ("fail", "blurry") in codes(r)
    assert r.verdict == "FAIL"


def test_dark_scan_fails(tmp_video):
    r = check_video(str(tmp_video(seconds=40, dark=[(0, 40)])))
    assert ("fail", "dark") in codes(r)


def test_plain_walls_fail(tmp_video):
    r = check_video(str(tmp_video(seconds=40, plain=True)))
    assert ("fail", "low_texture") in codes(r)


def test_short_and_partial_scan_warns(tmp_video):
    r = check_video(str(tmp_video(seconds=20, turn_deg=120)))
    found = codes(r)
    assert ("warn", "short") in found and ("warn", "coverage") in found


def test_freshness_code_in_video_is_verified(tmp_video):
    key = fr.load_secret()
    tok = fr.issue("pg-101", key, now=datetime.now(timezone.utc) - timedelta(minutes=3))
    r = check_video(str(tmp_video(seconds=40, qr_text=tok.text)), room_id="pg-101", secret=key)
    assert r.freshness["status"] == "verified"
    a, b = r.qr_segment
    assert a == 0.0 and 1.5 <= b <= 3.0
    # Seconds spent on the code are not judged as room footage.
    assert ("warn", "low_texture") not in codes(r)


def test_code_for_another_room_fails(tmp_video):
    key = fr.load_secret()
    tok = fr.issue("pg-999", key)
    r = check_video(str(tmp_video(seconds=40, qr_text=tok.text)), room_id="pg-101", secret=key)
    assert r.freshness["status"] == "wrong_room"
    assert ("fail", "freshness") in codes(r)


def test_missing_code_only_warns(tmp_video):
    r = check_video(str(tmp_video(seconds=40)), room_id="pg-101", secret=fr.load_secret())
    assert r.freshness["status"] == "not_found"
    assert ("warn", "freshness") in codes(r)

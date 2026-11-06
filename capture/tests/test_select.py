import cv2

from roomwalk.select import Scored, choose, export, score_frames


def make(ts, sharp):
    return [Scored(i, t, s, 120.0) for i, (t, s) in enumerate(zip(ts, sharp))]


def test_picks_sharpest_frame_per_window():
    frames = make([0, 1, 2, 3, 4, 5], [10, 90, 30, 200, 50, 40])
    sel = choose(frames, target=3, blur_abs=0)
    assert [s.index for s in sel.frames] == [1, 3, 4]


def test_drops_windows_that_are_all_blurry():
    frames = make(range(10), [100, 100, 100, 100, 1, 1, 100, 100, 100, 100])
    sel = choose(frames, target=5, blur_abs=0, blur_rel=0.4)
    assert sel.dropped_blurry == 1
    assert len(sel.frames) == 4


def test_excluded_ranges_are_skipped():
    frames = make(range(10), [500] * 3 + [100] * 7)
    sel = choose(frames, target=10, exclude=[(0, 2.5)], blur_abs=0)
    assert all(s.t > 2.5 for s in sel.frames)
    assert sel.dropped_excluded == 3


def test_export_writes_frames_and_poster(tmp_video, tmp_path):
    video = tmp_video(seconds=30, blur=[(10, 12)])
    scored = score_frames(str(video))
    sel = choose(scored, target=40)
    assert all(not (10 <= s.t <= 12) for s in sel.frames)
    paths = export(str(video), sel, tmp_path / "images", max_side=800, poster_path=tmp_path / "poster.jpg")
    assert len(paths) == len(sel.frames) >= 30
    img = cv2.imread(str(paths[0]))
    assert max(img.shape[:2]) == 800
    assert (tmp_path / "poster.jpg").exists()
    assert paths[0].name == "frame_00000.jpg"

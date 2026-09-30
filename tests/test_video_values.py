"""Placeholder values from the claims ledger and run directories; timeline auto-durations; text compositing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metagross.eval import plot_style
from tests.test_video_replay import _make_run
from video import collect_values, draw, style
from video.render_timeline import fill, load_values, run_paths, unfilled
from video.replay import RunReplay
from video.timeline import MIN_SCENE_S, NARRATION_PAD_S, WORDS_PER_S_ESTIMATE, Timeline


def _claims(root: Path) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    idx = {"kitti_pooled_t_err_pct": {"id": "kitti_pooled_t_err_pct", "value": "1.87", "unit": "%", "label": "Tested",
                                      "source": "results/kitti_summary.json#pooled.t_err_pct", "note": ""}}
    (root / "results" / "claims_index.json").write_text(json.dumps(idx), encoding="utf-8")


def test_collect_values_from_ledger_and_run(tmp_path: Path) -> None:
    _claims(tmp_path)
    run = _make_run(tmp_path)  # debug ticks: r_cert 6.0, q 0.9, NOMINAL for t < 1 s then CAUTION
    spec = {"runs": {"run_a": "run"},
            "values": {"kitti": {"claim": "kitti_pooled_t_err_pct", "fmt": "{:.1f}"},
                       "rmin": {"run": "{run_a}", "field": "r_cert_m", "stat": "min", "fmt": "{:.1f}"},
                       "qmin": {"run": "{run_a}", "field": "q", "stat": "min", "t0": 0.5, "fmt": "{:.2f}"},
                       "worst": {"run": "{run_a}", "field": "mode", "stat": "worst"},
                       "early": {"run": "{run_a}", "field": "mode", "stat": "worst", "t1": 0.5},
                       "pkt": {"run": "{run_a}", "field": "packet_bytes", "stat": "median", "fmt": "{:.0f}"},
                       "seed": {"run": "{run_a}", "json": "result.json", "key": "seed"},
                       "absent": {"claim": "does_not_exist"}}}
    spec["values"] = fill(spec["values"], run_paths(spec, []))
    res = collect_values.collect(spec, tmp_path)
    v = res["values"]
    assert v["kitti"]["value"] == "1.9" and v["kitti"]["label"] == "Tested"
    assert v["rmin"]["value"] == "6.0" and v["rmin"]["label"] == "Simulated"
    assert v["qmin"]["value"] == "0.90"
    assert v["worst"]["value"] == "caution" and v["early"]["value"] == "nominal"
    assert int(v["pkt"]["value"]) > 0 and v["seed"]["value"] == "101"
    assert "absent" in res["missing"]
    ids = {c["id"] for c in res["claims"]}
    assert "video_rmin" in ids and "video_kitti" not in ids  # ledger numbers are not re-registered
    out = tmp_path / "values.json"
    out.write_text(json.dumps(res), encoding="utf-8")
    assert load_values([str(out)])["rmin"] == "6.0"


def test_run_overrides_fill_paths(tmp_path: Path) -> None:
    spec = {"runs": {"run_a": "x/y", "t0_a": 3}, "scenes": [{"run": "{run_a}", "t0": "{t0_a}"}]}
    filled = fill(spec, run_paths(spec, ["run_a=z/w"]))
    assert filled["scenes"][0] == {"run": "z/w", "t0": "3"} and not unfilled(filled["scenes"])
    with pytest.raises(SystemExit):
        run_paths(spec, ["no_equals_sign"])


def test_auto_durations_follow_the_narration_estimate(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    words = "one two three four five six seven eight nine ten " * 3
    spec = {"scenes": [{"type": "card", "card": "title", "title": "T", "duration": "auto", "narration": words},
                       {"type": "card", "card": "title", "title": "T", "narration": ""},
                       {"type": "dashboard", "run": str(run), "t0": 1.0, "t1": "auto", "speed": 2.0, "narration": words}]}
    tl = Timeline.from_dicts(spec, root=tmp_path)
    tl.plan(tmp_path / "work", tts=False)
    est = 30 / WORDS_PER_S_ESTIMATE + NARRATION_PAD_S
    assert tl.scenes[0].duration == pytest.approx(est)
    assert tl.scenes[1].duration == pytest.approx(MIN_SCENE_S)
    assert tl.scenes[2].run_t(30, 30.0) == pytest.approx(3.0)  # 1 video second at 2x from t0 = 1 s
    frames = list(tl.scene_frames(tl.scenes[2], 0.0, only=[0, 5]))
    assert len(frames) == 2 and frames[0].shape == (1080, 1920, 3)


def test_draft_watermark_is_drawn_on_every_frame(tmp_path: Path) -> None:
    spec = {"scenes": [{"type": "card", "card": "title", "title": "T", "duration": 0.2}]}
    red = np.array(style.hex_rgb(style.MODE_HEX["STOP_AND_LOOK"]))
    for mark, expect in (("DRAFT", True), ("", False)):
        tl = Timeline.from_dicts(spec, root=tmp_path, watermark=mark)
        tl.plan(tmp_path / "w", tts=False)
        frames = list(tl.frames())
        top = frames[-1][:64, 900:1620].astype(int)
        assert bool((np.abs(top - red).max(axis=-1) <= 10).any()) == expect


def test_run_inspection_tools(tmp_path: Path) -> None:
    from video import find_runs, inspect_run

    run = _make_run(tmp_path)
    rows = inspect_run.summary_rows(run, every_s=1.0)
    assert rows[0]["t"] == 0.0 and rows[5]["dist_m"] == pytest.approx(5.0, abs=0.05)  # 1 m/s synthetic run
    assert rows[2]["mode"] == "CAUTION"
    found = find_runs.run_rows(tmp_path, count_images=True)
    assert len(found) == 1 and found[0]["seed"] == 101 and found[0]["n_debug"] == 30 and found[0]["n_left"] == 0


def test_style_tokens_mirror_plot_style() -> None:
    assert style.TOKENS == plot_style.TOKENS
    assert style.HONESTY_COLORS == plot_style.HONESTY_COLORS
    for k, v in plot_style.MODE_COLORS.items():
        assert style.MODE_HEX[k] == v


def test_text_is_composited_and_clipped() -> None:
    img = np.full((40, 120, 3), 255, np.uint8)
    draw.draw_text(img, 4, 20, "Hello", 20, (0, 0, 0), "regular", "lm")
    assert img.min() < 60 and img[:, 100:].min() == 255  # ink where the text is, none far right
    edge = np.full((20, 20, 3), 255, np.uint8)
    draw.draw_text(edge, -30, 10, "Clipped text", 16, (0, 0, 0), "bold", "lm")  # partly off-canvas: no error
    draw.draw_text(edge, 500, 500, "Gone", 16, (0, 0, 0))
    m1 = draw.text_mask("abc", 14, "regular", "mm")
    assert draw.text_mask("abc", 14, "regular", "mm") is m1  # cached


def test_link_stats_are_measured_from_telemetry(tmp_path: Path) -> None:
    rp = RunReplay(_make_run(tmp_path))  # telemetry every 0.5 s
    rate, kbps = rp.link_stats_at(10.0)
    assert rate == pytest.approx(2.0, abs=0.21)
    pkt = [d["packet_bytes"] for d in rp.telemetry if 5.0 < d["t"] <= 10.0]
    assert kbps == pytest.approx(8 * sum(pkt) / 5.0 / 1000.0)
    assert rp.link_stats_at(-1.0) == (0.0, 0.0)
    pd = rp.frame_at(10.2)
    assert pd.packet_age_s == pytest.approx(0.2, abs=1e-6)

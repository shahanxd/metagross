"""Build the DEMO_FAKE run directory and render the video-toolchain previews.

Everything produced here is SYNTHETIC and marked as such (``demo_fake``): it exists so the
compositor, encoder and operator console can be built and reviewed before real closed-loop
runs are available. It is never a result.

Outputs
-------
* ``data/demo_fake/run_seed7_demo_fake/``   fake run dir (runner + autonomy-process formats)
* ``deck_assets/dashboard_preview.png``     one 1920x1080 dashboard frame
* ``deck_assets/card_preview.png``          one evidence card (deck style)
* ``video/out/test_clip.mp4`` (+ .srt)      10 s dashboard clip via the timeline
* ``deck_assets/console_preview.png``       operator console screenshot (``--console``; needs a
                                            browser, run from PowerShell: sockets are sandboxed in Bash)

Usage::

    .venv\\Scripts\\python.exe scripts\\make_demo_data.py [--force] [--tts] [--console] [--no-clip]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from video import cards, layout  # noqa: E402
from video.demo_fake import DemoChaseRenderer, build_demo_run  # noqa: E402
from video.replay import RunReplay  # noqa: E402
from video.timeline import Timeline  # noqa: E402

log = logging.getLogger("make_demo_data")

RUN_DIR = REPO / "data" / "demo_fake" / "run_seed7_demo_fake"
STILL_T = 9.0  # s: ditch band visible, CAUTION
CLIP_T = (3.0, 13.0)  # s: NOMINAL -> CAUTION with the band turning magenta then red


def render_still(run_dir: Path, out_png: Path, t: float = STILL_T) -> Path:
    rp = RunReplay(run_dir, chase_renderer=DemoChaseRenderer(), chase_size=layout.CHASE_SIZE)
    dash = layout.Dashboard(gt_at=lambda tt: rp.gt_pose_at(tt)[:3])
    img = dash.compose(rp.frame_at(t), caption=layout.Caption("Missing ground, not an object",
                                                              "The trench is a grey band, then magenta, then red"),
                       subtitle="It drives only on ground it has seen.")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_png)
    log.info("still -> %s", out_png)
    return out_png


def render_card_preview(out_png: Path) -> Path:
    chart = REPO / "deck_assets" / "kitti_07_traj.png"
    img = cards.evidence_card("{kitti_drift_1km}", "%", "Visual-odometry drift per km on KITTI (placeholder, filled from results/)",
                              "Tested", chart if chart.exists() else None, kicker="Evidence · Localisation",
                              bullets=["Stereo VO + integrity monitor, CPU only", "Numbers are placeholders in this preview"],
                              source="results/kitti_vo.csv")
    Image.fromarray(img).save(out_png)
    log.info("card -> %s", out_png)
    return out_png


def render_clip(run_dir: Path, out_mp4: Path, tts: bool) -> dict:
    spec = {"fps": 30, "scenes": [{
        "type": "dashboard", "run": str(run_dir.relative_to(REPO)), "t0": CLIP_T[0], "t1": CLIP_T[1], "chase": "demo_fake",
        "caption": {"title": "Missing ground, not an object", "subtitle": "SYNTHETIC DEMO DATA · toolchain test"},
        "narration": "The trench is missing ground. A grey band. Then magenta. Then red.",
    }]}
    tl = Timeline.from_dicts(spec, root=REPO, chase_renderers={"demo_fake": DemoChaseRenderer})
    return tl.render(out_mp4, tts=tts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild the fake run even if it exists")
    ap.add_argument("--tts", action="store_true", help="synthesise narration with edge-tts (needs network)")
    ap.add_argument("--console", action="store_true", help="also capture the operator console (Playwright, Chrome)")
    ap.add_argument("--no-clip", action="store_true", help="skip the 10 s MP4")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cv2.setNumThreads(2)
    t0 = time.perf_counter()
    if args.force or not (RUN_DIR / "result.json").exists():
        build_demo_run(RUN_DIR)
    render_still(RUN_DIR, REPO / "deck_assets" / "dashboard_preview.png")
    render_card_preview(REPO / "deck_assets" / "card_preview.png")
    if not args.no_clip:
        summary = render_clip(RUN_DIR, REPO / "video" / "out" / "test_clip.mp4", tts=args.tts)
        log.info("clip: %s", summary)
    if args.console:
        from video.console_capture import capture_console

        capture_console(RUN_DIR, REPO / "deck_assets" / "console_preview.png", at_t=STILL_T)
    log.info("done in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()

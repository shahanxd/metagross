"""CLI: render a timeline YAML to a narrated MP4.

    python -m video.render_timeline video/scenes/golden_demo.yaml video/out/golden_demo.mp4 \
        --values video/out/values.json [--no-tts] [--allow-placeholders] [--chase-fps 15]

``{placeholders}`` in narration / captions / card text are filled from ``--values`` (a flat JSON
object whose numbers come from ``results/`` or run directories, see ``video.collect_values``).
Unfilled placeholders abort the render unless ``--allow-placeholders`` is given (draft renders),
so no hand-typed number can slip into the final video.

``--plan-only`` synthesises (or loads cached) narration and prints the per-scene durations and
the total length without rendering frames; ``--stills DIR`` writes one PNG per scene (its middle
frame) for layout checks. Chase frames are cached as JPEG under ``--chase-cache`` so a second
render after a narration / layout edit does not touch the browser.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import yaml

from video import chase
from video.timeline import Timeline

log = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
DEFAULT_CHASE_CACHE = Path("video/out/cache/chase")


def fill(obj: Any, values: dict[str, Any]) -> Any:
    """Recursively replace ``{name}`` in strings with ``values[name]`` (missing names are kept)."""
    if isinstance(obj, str):
        return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), obj)
    if isinstance(obj, list):
        return [fill(v, values) for v in obj]
    if isinstance(obj, dict):
        return {k: fill(v, values) for k, v in obj.items()}
    return obj


def unfilled(obj: Any) -> set[str]:
    """Placeholder names still present anywhere in ``obj``."""
    if isinstance(obj, str):
        return set(PLACEHOLDER.findall(obj))
    if isinstance(obj, list):
        return set().union(*(unfilled(v) for v in obj)) if obj else set()
    if isinstance(obj, dict):
        return set().union(*(unfilled(v) for v in obj.values())) if obj else set()
    return set()


def default_chase_renderers() -> dict:
    """Names usable as ``chase:`` in a timeline (``three``, ``demo_fake``)."""
    return chase.registry()


def load_values(paths: list[str]) -> dict[str, Any]:
    """Merge value files (later files win): a flat ``{name: value}`` JSON, or the
    ``video.collect_values`` output ``{"values": {name: {"value": ...}}, ...}``. Keys starting with
    '_' are metadata."""
    values: dict[str, Any] = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if isinstance(d.get("values"), dict):
            d = d["values"]
        values.update({k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in d.items() if not k.startswith("_")})
    return values


def run_paths(spec: dict, overrides: list[str]) -> dict[str, Any]:
    """The timeline's ``runs:`` block (run directories and scene start times used as
    ``{placeholders}``) with ``NAME=VALUE`` command-line overrides applied."""
    runs = dict(spec.get("runs") or {})
    for o in overrides:
        k, sep, v = o.partition("=")
        if not sep:
            raise SystemExit(f"--run expects NAME=PATH, got {o!r}")
        if k not in runs:
            log.warning("--run %s: not in the timeline's runs block (added anyway)", k)
        runs[k] = v
    return runs


def write_stills(tl: Timeline, out_dir: Path) -> list[Path]:
    """Middle frame of every scene -> ``out_dir/scene_XX.png`` (needs :meth:`Timeline.plan` first)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, t = [], 0.0
    for i, sc in enumerate(tl.scenes):
        mid = int(round(sc.duration * tl.fps / 2))
        for f in tl.scene_frames(sc, t, only=[mid]):
            p = out_dir / f"scene_{i:02d}_{sc.type}.png"
            cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            paths.append(p)
        t += sc.duration
    tl.close()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("timeline")
    ap.add_argument("out_mp4")
    ap.add_argument("--values", action="append", default=[], help="JSON of placeholder values (repeatable; later wins)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                    help="override an entry of the timeline's 'runs:' block (repeatable)")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--allow-placeholders", action="store_true", help="draft render: keep {placeholders} visible")
    ap.add_argument("--root", default=".", help="base directory for run / image paths")
    ap.add_argument("--chase-cache", default=str(DEFAULT_CHASE_CACHE), help="JPEG cache for chase frames ('' = off)")
    ap.add_argument("--chase-fps", type=float, default=30.0, help="distinct chase frames per run second (15 = fast draft)")
    ap.add_argument("--voice", help="edge-tts voice, e.g. en-IN-NeerjaNeural (default: timeline 'voice' or en-IN-PrabhatNeural)")
    ap.add_argument("--preset", default="medium", help="libx264 preset: medium (final) or veryfast (draft)")
    ap.add_argument("--watermark", default="", help='text drawn on every frame, e.g. "DRAFT · non-golden runs"')
    ap.add_argument("--scenes", help="render only these scene indices, comma-separated, 0-based as printed by --plan-only")
    ap.add_argument("--plan-only", action="store_true", help="synthesise narration, print durations, no frames")
    ap.add_argument("--stills", help="write the middle frame of each scene as PNG into this directory, no MP4")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cv2.setNumThreads(2)
    spec = yaml.safe_load(Path(a.timeline).read_text(encoding="utf-8"))
    spec = fill(spec, {**run_paths(spec, a.run), **load_values(a.values)})
    if a.scenes:
        keep = [int(s) for s in a.scenes.split(",")]
        spec["scenes"] = [spec["scenes"][i] for i in keep]
    missing = unfilled(spec)
    if missing and not a.allow_placeholders:
        raise SystemExit(f"unfilled placeholders: {sorted(missing)} (pass --values or --allow-placeholders)")
    kw: dict[str, Any] = {"chase_cache": Path(a.chase_cache) if a.chase_cache else None, "chase_fps": a.chase_fps,
                          "watermark": a.watermark}
    if a.voice:
        kw["voices"] = (a.voice,)
    tl = Timeline.from_dicts(spec, root=Path(a.root).resolve(), chase_renderers=default_chase_renderers(), **kw)
    out = Path(a.out_mp4)
    if a.plan_only or a.stills:
        total = tl.plan(out.parent / (out.stem + "_work"), tts=not a.no_tts)
        rows = [{"scene": i, "type": sc.type, "speech_s": round(sc.speech_s, 2), "duration_s": round(sc.duration, 2)}
                for i, sc in enumerate(tl.scenes)]
        for r in rows:
            log.info("scene %(scene)02d %(type)-9s speech %(speech_s)6.2f s  -> %(duration_s)6.2f s", r)
        log.info("total %.1f s (%d:%04.1f)", total, int(total // 60), total % 60)
        plan_path = out.parent / (out.stem + "_plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps({"total_s": round(total, 2), "scenes": rows}, indent=1), encoding="utf-8")
        if a.stills:
            for p in write_stills(tl, Path(a.stills)):
                log.info("still %s", p)
        return
    summary = tl.render(a.out_mp4, tts=not a.no_tts, preset=a.preset)
    out.with_name(out.stem + "_render.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("%s", summary)


if __name__ == "__main__":
    main()

"""Chase-view plumbing for the video timeline: renderer registry, frame-rate hold and disk cache.

* :class:`CachedChase` wraps any ``(state, w, h) -> rgb`` chase renderer (``video.replay.ChaseRenderer``):
  - **hold**: renders at most ``chase_fps`` distinct frames per second of run time and repeats the
    last one in between (the chase camera follows the vehicle, so 15 fps is a cheap draft setting;
    the default holds nothing at 30 fps);
  - **disk cache**: every rendered frame is stored as a JPEG under ``cache_dir`` keyed by the
    scenario digest, the output size and a hash of the renderer state, so re-rendering a timeline
    after a layout or narration change does not touch the browser again.
* :func:`registry` returns the chase-renderer names a timeline YAML may use: ``three`` (the
  Three.js simulator scene, :class:`metagross.sim.render.chase_replay.ThreeChaseFactory`) and
  ``demo_fake`` (synthetic ground for DEMO_FAKE runs).

A registry value is either a *per-run factory* (an object with ``for_run(run_dir) -> renderer``)
or a zero-argument callable returning a renderer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

JPEG_QUALITY = 94  # cached chase frames: visually lossless before the final H.264 pass
STATE_DECIMALS = 4  # pose / time rounding in the cache key (0.1 mm, 0.1 ms: far below one pixel)
DEFAULT_CHASE_FPS = 30.0


def state_key(state: dict, width: int, height: int, scene_id: str) -> str:
    """Stable hex key of a renderer state (WORLD pose, t, dynamic actors) at an output size."""
    def rnd(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, STATE_DECIMALS)
        if isinstance(v, (list, tuple)):
            return [rnd(x) for x in v]
        if isinstance(v, dict):
            return {k: rnd(x) for k, x in sorted(v.items())}
        return v

    body = {"s": scene_id, "w": int(width), "h": int(height), "pose": rnd(list(state.get("pose", []))),
            "t": rnd(float(state.get("t", 0.0))), "dyn": rnd(state.get("dynamic", [])),
            "light": rnd(state.get("lighting", {}))}
    if state.get("chase"):  # optional chase-camera block (absent keeps the keys of earlier caches valid)
        body["chase"] = rnd(state["chase"])
    payload = json.dumps(body, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class CachedChase:
    """Hold + disk cache around a chase renderer (see module docstring).

    Parameters
    ----------
    renderer: ``(state, w, h) -> (h, w, 3) uint8 RGB``.
    scene_id: identifies the scene (e.g. the scenario sha256); part of the cache key.
    cache_dir: JPEG cache directory, or None for no disk cache.
    chase_fps: distinct chase frames per second of run time (>= 30 means every video frame).
    cache_tag: identifies state the renderer adds itself (e.g. the planned chase camera of a run,
        ``ChaseReplay.cache_tag``); part of the cache key, so a new camera model never reuses old frames.
    """

    def __init__(self, renderer: Callable[[dict, int, int], np.ndarray], scene_id: str = "",
                 cache_dir: Optional[str | Path] = None, chase_fps: float = DEFAULT_CHASE_FPS,
                 cache_tag: str = "") -> None:
        self.renderer = renderer
        self.scene_id = scene_id
        self.cache_tag = cache_tag
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.chase_fps = float(chase_fps)
        self._last: Optional[tuple[int, tuple[int, int], np.ndarray]] = None
        self.hits = 0
        self.renders = 0
        self.failed = False

    def _bucket(self, t: float) -> int:
        return int(np.floor(t * self.chase_fps + 1e-6))

    def __call__(self, state: dict, width: int, height: int) -> Optional[np.ndarray]:
        """Chase frame, or None once the renderer has failed (the dashboard then draws its labelled
        top-down fallback; the failure is logged once as an ERROR)."""
        if self.failed:
            return None
        b = self._bucket(float(state.get("t", 0.0)))
        if self._last is not None and self._last[0] == b and self._last[1] == (width, height):
            return self._last[2]
        try:
            img = self._cached_or_render(state, width, height)
        except Exception as exc:  # noqa: BLE001 - browser / WebGL failures must not kill a long render
            log.error("chase renderer failed (%s: %s); using the top-down fallback for this run", type(exc).__name__, exc)
            self.failed = True
            return None
        self._last = (b, (width, height), img)
        return img

    def _cached_or_render(self, state: dict, width: int, height: int) -> np.ndarray:
        path = None
        if self.cache_dir is not None:
            key = state_key(state, width, height, f"{self.scene_id}|{self.cache_tag}" if self.cache_tag else self.scene_id)
            path = self.cache_dir / (self.scene_id[:12] or "scene") / f"{width}x{height}" / f"{key}.jpg"
            if path.exists():
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is not None and bgr.shape[:2] == (height, width):
                    self.hits += 1
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(self.renderer(state, width, height))
        self.renders += 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return img


def make_chase(factory: Any, run_dir: Path, cache_dir: Optional[Path], chase_fps: float) -> CachedChase:
    """Registry value + run directory -> cached chase renderer for that run."""
    if hasattr(factory, "for_run"):
        renderer = factory.for_run(run_dir)
        scene_id = str(getattr(renderer, "scenario_sha", "") or run_dir.name)
    else:
        renderer = factory()
        scene_id = getattr(factory, "__name__", type(factory).__name__)
    return CachedChase(renderer, scene_id=scene_id, cache_dir=cache_dir, chase_fps=chase_fps,
                       cache_tag=str(getattr(renderer, "cache_tag", "") or ""))


def registry() -> dict[str, Any]:
    """Chase renderers available to timeline YAML files (``chase: three`` / ``chase: demo_fake``)."""
    from metagross.sim.render.chase_replay import ThreeChaseFactory
    from video.demo_fake import DemoChaseRenderer

    return {"three": ThreeChaseFactory(), "demo_fake": DemoChaseRenderer}

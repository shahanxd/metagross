"""Playwright bridge to the Three.js stereo renderer (``web/``).

``ThreeRenderer`` implements :class:`metagross.contracts.interfaces.RendererProto`.
It drives a real Chrome/Edge (channel ``chrome`` or ``msedge``) with a hardware
WebGL2 context (ANGLE / Direct3D 11) through Playwright's *sync* API; every
render is a single synchronous ``page.evaluate`` call (no animation frames), so
the simulation stays in lock-step with the renderer.

Why a headed browser: Chrome no longer falls back silently to SwiftShader, and
headless Chrome without a GPU may either fail to create WebGL or run it on the
CPU. The window is parked off-screen, background throttling is disabled, and the
bridge refuses to run if the unmasked WebGL renderer is SwiftShader.

Web assets are served from disk through ``page.route`` on a fake origin
(``http://metagross.local/``) because ES modules cannot be imported from
``file://`` URLs. No socket is opened on this machine.

Image layout returned to Python (OpenCV conventions, rows top-first):
* ``left_rgb``  (H, W, 3) uint8 RGB, ``right_gray`` (H, W) uint8.
* GT ``depth`` (H, W) float32 metres along the left optical axis, ``inf`` for sky.
* GT ``semantic`` (H, W) uint8 5-class ids (``interfaces.SEM_CLASSES``).
"""

from __future__ import annotations

import base64
import json
import logging
import math
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from metagross.config import defaults
from metagross.contracts.messages import StereoCalibration, VehicleSpec
from metagross.contracts.scenario import MATERIAL_TO_SEM, OBJECT_SEM, SKY_SEM
from metagross.sim.render.camera_model import calib_to_js

log = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).resolve().parent / "web"
ORIGIN = "http://metagross.local"
# Environment overrides (Linux / headless GPU boxes; defaults keep the Windows behaviour):
#   MG_ANGLE=<backend>       ANGLE backend (default d3d11 on Windows, Chrome's choice elsewhere; e.g. vulkan,
#                            gl-egl on an NVIDIA Linux box, swiftshader for a CPU-only smoke test)
#   MG_RENDER_HEADLESS=1     headless browser (Linux servers have no display)
#   MG_BROWSER_PATH=<exe>    browser executable, used when no chrome/msedge channel is installed
#                            (default: Playwright's bundled Chromium)
#   MG_ALLOW_SOFTWARE_GL=1   accept a software WebGL renderer (slow, ~seconds per frame; smoke tests only)
_ANGLE = os.environ.get("MG_ANGLE") or ("d3d11" if sys.platform == "win32" else "")
CHROME_ARGS = (
    *((f"--use-angle={_ANGLE}",) if _ANGLE else ()),
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-features=CalculateNativeWinOcclusion",
    "--window-position=-2400,-2400",
    "--window-size=320,240",
)
SOFTWARE_RENDERERS = ("swiftshader", "llvmpipe", "microsoft basic render")
READY_TIMEOUT_MS = 60_000

_CONTENT_TYPES = {".js": "text/javascript", ".html": "text/html", ".json": "application/json", ".txt": "text/plain"}


class RendererUnavailable(RuntimeError):
    """No suitable browser / hardware WebGL2 context could be created."""


def yuv420_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Full-range BT.601 (JFIF) YUV 4:2:0 -> RGB uint8 (H, W, 3).

    ``y`` (H, W); ``u``, ``v`` (H/2, W/2) are 2x2 block means, upsampled bilinearly
    with centred siting. Matches the post shader's forward transform exactly up to
    8-bit rounding in luma.
    """
    H, W = y.shape
    uu = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128.0
    vv = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128.0
    yy = y.astype(np.float32)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[..., 0] = yy + 1.402 * vv
    rgb[..., 1] = yy - 0.344136 * uu - 0.714136 * vv
    rgb[..., 2] = yy + 1.772 * uu
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and tuples) to plain JSON types, recursively."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class ThreeRenderer:
    """Stereo / ground-truth / chase renderer backed by Three.js in Chrome.

    Parameters
    ----------
    calib:
        Stereo calibration; defaults to ``defaults.stereo_calibration()``.
    vehicle:
        Vehicle spec used only for the chase-view model.
    channel:
        Preferred browser channel; ``"msedge"`` is tried if it fails.
    shadows:
        Sun shadow maps (first quality feature to cut if frame time is short).
    msaa:
        MSAA samples for the stereo HDR targets (0 = off; edges are then aliased
        like a real global-shutter sensor without an optical low-pass filter).
    color_transport:
        ``"yuv420"`` (default): the left image crosses the browser boundary as
        full-range BT.601 YUV 4:2:0 (what USB stereo cameras emit), 2.5 B/px for the
        pair instead of 4 B/px; luma is exact, chroma is 2x2-averaged. ``"rgb"``:
        exact RGB, ~35 % more transfer time.
    """

    def __init__(
        self,
        calib: Optional[StereoCalibration] = None,
        vehicle: Optional[VehicleSpec] = None,
        *,
        channel: str = "chrome",
        shadows: bool = True,
        msaa: int = 0,
        headless: bool = False,
        color_transport: str = "yuv420",
    ) -> None:
        self.calib = calib or defaults.stereo_calibration()
        self.vehicle = vehicle or defaults.VEHICLE
        if color_transport not in ("yuv420", "rgb"):
            raise ValueError(f"color_transport must be 'yuv420' or 'rgb', got {color_transport!r}")
        if color_transport == "yuv420" and (self.calib.width % 2 or self.calib.height % 2):
            color_transport = "rgb"  # 4:2:0 needs even image dimensions
        self.color_transport = color_transport
        self._timings: dict[str, list[float]] = {"stereo": [], "stereo_js": [], "gt": [], "chase": [], "load": []}
        self.last_js: dict[str, Any] = {}
        self._pw = None
        self._browser = None
        self._page = None
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RendererUnavailable(f"playwright not importable: {e}") from e
        self._pw = sync_playwright().start()
        headless = headless or os.environ.get("MG_RENDER_HEADLESS") == "1"
        errors: list[str] = []
        for ch in dict.fromkeys((channel, "chrome", "msedge")):
            try:
                self._browser = self._pw.chromium.launch(channel=ch, headless=headless, args=list(CHROME_ARGS))
                self.channel = ch
                break
            except Exception as e:  # noqa: BLE001 - try the next channel
                errors.append(f"{ch}: {e}")
        if self._browser is None:  # no branded browser (e.g. Linux server): bundled / given Chromium
            exe = os.environ.get("MG_BROWSER_PATH") or None
            try:
                self._browser = self._pw.chromium.launch(executable_path=exe, headless=headless, args=list(CHROME_ARGS))
                self.channel = "chromium"
            except Exception as e:  # noqa: BLE001
                errors.append(f"chromium ({exe or 'bundled'}): {e}")
        if self._browser is None:
            self._pw.stop()
            raise RendererUnavailable("could not launch chrome/msedge: " + " | ".join(errors))
        try:
            self._open_page(shadows, msaa)
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ setup
    def _open_page(self, shadows: bool, msaa: int) -> None:
        ctx = self._browser.new_context(viewport={"width": 320, "height": 240})
        page = ctx.new_page()
        root = WEB_ROOT.resolve()

        def serve(route) -> None:
            path = route.request.url[len(ORIGIN):].split("?", 1)[0].lstrip("/") or "index.html"
            f = (root / path).resolve()
            if root not in f.parents and f != root or not f.is_file():
                route.fulfill(status=404, body="not found")
                return
            ctype = _CONTENT_TYPES.get(f.suffix) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            route.fulfill(status=200, body=f.read_bytes(), headers={"Content-Type": ctype, "Cache-Control": "no-store"})

        page.route(f"{ORIGIN}/**", serve)
        page.on("console", lambda m: log.debug("[browser %s] %s", m.type, m.text))
        page.on("pageerror", lambda e: log.error("[browser pageerror] %s", e))
        page.add_init_script(f"window.MG_OPTIONS = {json.dumps({'shadows': bool(shadows), 'msaa': int(msaa)})};")
        page.goto(f"{ORIGIN}/index.html")
        page.wait_for_function("window.__mgReady === true || window.__mgError !== undefined", timeout=READY_TIMEOUT_MS)
        err = page.evaluate("window.__mgError")
        if err:
            raise RendererUnavailable(f"renderer page failed to load: {err}")
        self._page = page
        self.info: dict[str, Any] = page.evaluate("rendererInfo()")
        rs = str(self.info.get("renderer", "")).lower()
        if any(s in rs for s in SOFTWARE_RENDERERS):
            if os.environ.get("MG_ALLOW_SOFTWARE_GL") != "1":
                raise RendererUnavailable(f"software WebGL renderer refused: {self.info.get('renderer')}")
            log.warning("ThreeRenderer: software WebGL (%s) accepted by MG_ALLOW_SOFTWARE_GL=1: expect seconds per frame",
                        self.info.get("renderer"))
        if not self.info.get("colorBufferFloat"):
            raise RendererUnavailable("EXT_color_buffer_float missing: HDR render targets unsupported")
        log.info("ThreeRenderer: %s via %s (three r%s)", self.info.get("renderer"), self.channel, self.info.get("three"))
        cfg = calib_to_js(self.calib)
        v = self.vehicle
        cfg["vehicle"] = {
            "length_m": v.length_m, "width_m": v.width_m, "wheel_radius_m": v.wheel_radius_m, "track_width_m": v.track_width_m,
        }
        page.evaluate("c => configureCameras(c)", cfg)

    @property
    def renderer_string(self) -> str:
        """Unmasked WebGL renderer string (e.g. 'ANGLE (Intel, ... Direct3D11 ...)')."""
        return str(self.info.get("renderer", ""))

    # ------------------------------------------------------------------ RendererProto
    def load_scenario(self, scenario: dict) -> None:
        """Build the scene from a ``metagross.scenario/1`` dict (see contracts/scenario.py)."""
        t0 = time.perf_counter()
        sc = _jsonable(scenario)
        sc["__sem"] = {"material_to_sem": {str(k): int(v) for k, v in MATERIAL_TO_SEM.items()}, "object": OBJECT_SEM, "sky": SKY_SEM}
        res = self._page.evaluate("s => loadScenario(s)", sc)
        dt = (time.perf_counter() - t0) * 1e3
        self._timings["load"].append(dt)
        log.info("scenario loaded in %.0f ms (js %.0f ms): %s", dt, res.get("ms", float("nan")), res.get("terrain"))
        self.last_load = res

    def _collect(self, r: dict) -> bytearray:
        """Fetch the remaining base64 slices of a chunked result and decode them.
        Returns a (writable) bytearray so the numpy views handed out are writable."""
        parts = [r["first"]] + [self._page.evaluate("i => mgChunk(i)", i) for i in range(1, int(r["n"]))]
        return bytearray(base64.b64decode("".join(parts)))

    def render_stereo(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Render the rectified stereo pair for ``state`` (pose, t, dynamic, lighting).

        Advances the auto-exposure state by ``state['t']`` minus the previous render time.
        Returns ``(left_rgb (H,W,3) uint8, right_gray (H,W) uint8)``.
        """
        t0 = time.perf_counter()
        yuv = self.color_transport == "yuv420"
        r = self._page.evaluate("([s, o]) => renderStereo(s, o)", [_jsonable(state), {"chunked": True, "yuv420": yuv}])
        H, W = self.calib.height, self.calib.width
        buf = np.frombuffer(self._collect(r), dtype=np.uint8)
        if yuv:
            n, n4 = H * W, (H // 2) * (W // 2)
            left = yuv420_to_rgb(buf[:n].reshape(H, W), buf[n:n + n4].reshape(H // 2, W // 2), buf[n + n4:n + 2 * n4].reshape(H // 2, W // 2))
            right = buf[n + 2 * n4:].reshape(H, W)
        else:
            left = buf[: H * W * 3].reshape(H, W, 3)
            right = buf[H * W * 3:].reshape(H, W)
        self._timings["stereo"].append((time.perf_counter() - t0) * 1e3)
        self._timings["stereo_js"].append(float(r["ms"]["total"]))
        self.last_js = {"ms": r["ms"], "exposure": r["exposure"]}
        return left, right

    def render_gt(self, state: dict) -> dict[str, np.ndarray]:
        """Evaluator-only passes for the LEFT camera.

        ``depth``: (H,W) float32 metres along the optical axis (camera z), ``inf`` = sky.
        ``semantic``: (H,W) uint8 5-class ids (terrain by material, objects = OBJECT_SEM, sky = SKY_SEM).
        """
        t0 = time.perf_counter()
        r = self._page.evaluate("s => renderGT(s, {chunked: true})", _jsonable(state))
        H, W = self.calib.height, self.calib.width
        raw = self._collect(r)
        depth = np.frombuffer(raw, dtype="<f4", count=H * W).reshape(H, W).astype(np.float32, copy=True)
        depth[~(depth > 0)] = np.inf
        sem = np.frombuffer(raw, dtype=np.uint8, offset=H * W * 4).reshape(H, W).copy()
        self._timings["gt"].append((time.perf_counter() - t0) * 1e3)
        return {"depth": depth, "semantic": sem}

    def render_chase(self, state: dict, width: int, height: int) -> np.ndarray:
        """Third-person view ~6 m behind / 3 m above the vehicle, with a vehicle model.
        Does not advance auto-exposure. Returns (height, width, 3) uint8 RGB."""
        t0 = time.perf_counter()
        r = self._page.evaluate("([s, w, h]) => renderChase(s, w, h, {chunked: true})", [_jsonable(state), int(width), int(height)])
        img = np.frombuffer(self._collect(r), dtype=np.uint8).reshape(int(height), int(width), 3)
        self._timings["chase"].append((time.perf_counter() - t0) * 1e3)
        return img

    def project_points(self, pose: list[float], points_world: np.ndarray, eye: str = "left") -> np.ndarray:
        """Project world points with the renderer's own GL matrices (test hook). Returns (N,2) px."""
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3).tolist()
        uv = self._page.evaluate("([p, q, e]) => projectPoints(p, q, e)", [list(map(float, pose)), pts, 1 if eye == "right" else 0])
        return np.asarray(uv, dtype=np.float64)

    def close(self) -> None:
        """Shut the browser and Playwright down (idempotent)."""
        for obj, meth in ((self._browser, "close"), (self._pw, "stop")):
            if obj is not None:
                try:
                    getattr(obj, meth)()
                except Exception as e:  # noqa: BLE001 - best effort on shutdown
                    log.debug("close: %s", e)
        self._browser = None
        self._pw = None
        self._page = None

    # ------------------------------------------------------------------ misc
    def stats(self) -> dict[str, dict[str, float]]:
        """Wall-clock timing summary per call type (ms): n, mean, p50, p95, max."""
        out: dict[str, dict[str, float]] = {}
        for k, v in self._timings.items():
            if v:
                a = np.asarray(v)
                out[k] = {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                          "p95": float(np.percentile(a, 95)), "max": float(a.max())}
        return out

    def reset_stats(self) -> None:
        for v in self._timings.values():
            v.clear()

    def __enter__(self) -> "ThreeRenderer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC timing dependent
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

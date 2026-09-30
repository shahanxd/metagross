"""Drive the static operator console (``operator_ui/``) with Playwright and screenshot it.

The console is served from disk through ``page.route`` on a fake origin (no local socket is
opened, same approach as :mod:`metagross.sim.render.bridge`), loaded with
``?deterministic=1`` (no wall clock, no timers), fed ``mission.json`` + ``telemetry.jsonl``
of a run directory through ``window.consoleReset`` / ``window.consoleReplay`` and seeked to a
mission time with ``window.consoleSeek`` so the screenshot is reproducible.

Needs Chrome (``channel='chrome'``) or Edge; run from PowerShell (the Bash sandbox blocks the
browser's local pipe/socket).

    .venv\\Scripts\\python.exe -m video.console_capture <run_dir> <out.png> [--t 9.0] [--size 1920x1080]
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
UI_ROOT = REPO / "operator_ui"
ORIGIN = "http://metagross.local"
BROWSER_ARGS = ("--window-position=-2400,-2400", "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows")


def capture_console(run_dir: str | Path, out_png: str | Path, at_t: float = 9.0, size: tuple[int, int] = (1920, 1080),
                    channel: str = "chrome", headless: bool = False) -> Path:
    """Screenshot the console showing ``run_dir`` telemetry up to mission time ``at_t`` (s)."""
    from playwright.sync_api import sync_playwright

    run = Path(run_dir)
    mission = json.loads((run / "mission.json").read_text(encoding="utf-8")) if (run / "mission.json").exists() else None
    lines = (run / "autonomy" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    def handle(route, request) -> None:  # serve operator_ui/ from disk
        rel = request.url.split(ORIGIN, 1)[-1].split("?", 1)[0].lstrip("/") or "index.html"
        f = (UI_ROOT / rel).resolve()
        if UI_ROOT.resolve() not in f.parents or not f.exists():
            route.fulfill(status=404, body="not found")
            return
        route.fulfill(status=200, body=f.read_bytes(), content_type=mimetypes.guess_type(f.name)[0] or "application/octet-stream")

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel=channel, headless=headless, args=list(BROWSER_ARGS))
        except Exception:  # noqa: BLE001 - fall back to Edge, then Playwright's bundled Chromium
            log.warning("channel %s unavailable, trying msedge", channel)
            browser = pw.chromium.launch(channel="msedge", headless=headless, args=list(BROWSER_ARGS))
        ctx = browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1)
        page = ctx.new_page()
        page.route(f"{ORIGIN}/**", handle)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{ORIGIN}/index.html?deterministic=1")
        page.wait_for_function("window.__consoleReady === true", timeout=15000)
        page.evaluate("m => window.consoleReset(m)", mission)
        page.evaluate("ls => window.consoleReplay(ls, {realtime: false})", lines)
        page.evaluate("t => window.consoleSeek(t)", at_t)
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(out))
        browser.close()
    if errors:
        raise RuntimeError(f"console page errors: {errors}")
    log.info("console screenshot -> %s (%dx%d, t=%.1f s)", out, size[0], size[1], at_t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Screenshot the operator console for a run directory.")
    ap.add_argument("run_dir")
    ap.add_argument("out_png")
    ap.add_argument("--t", type=float, default=9.0)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--channel", default="chrome")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    w, h = (int(v) for v in a.size.lower().split("x"))
    capture_console(a.run_dir, a.out_png, a.t, (w, h), a.channel)


if __name__ == "__main__":
    main()

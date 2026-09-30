"""Declarative scene list -> frames -> narrated MP4.

A timeline is a list of scenes (YAML file or Python dicts)::

    fps: 30
    scenes:
      - type: card            # title | evidence | end card (video.cards)
        card: title
        duration: 4
        title: "Vision-based navigation without GNSS"
        subtitle: "..."
        narration: "GNSS is jammed. ..."
      - type: dashboard       # one run through the compositor (video.layout.Dashboard)
        run: data/demo_fake/run_seed7_demo_fake
        t0: 3.0
        t1: 13.0
        chase: demo_fake      # name of a registered chase renderer, or omit for the placeholder
        caption: {title: "...", subtitle: "..."}
        narration: "..."
      - type: image           # full-screen still, e.g. deck_assets/console_preview.png
        path: deck_assets/console_preview.png
        duration: 6
        caption: {title: "...", subtitle: "..."}
      - type: split           # two runs side by side, same seed (video.layout.compose_split)
        run_a: runs/.../typical
        run_b: runs/.../metagross
        label_a: "TYPICAL STACK · unknown = free"
        label_b: "METAGROSS · unknown is never free"
        t0: 0
        t1: 20

Narration: each scene's ``narration`` is synthesised with edge-tts (:mod:`video.encode`); a
scene lasts ``max(duration, narration + NARRATION_PAD_S)``. ``duration: auto`` (or no
duration) on card / image scenes makes the scene exactly as long as its narration; ``t1: auto``
on dashboard / split scenes plays the run from ``t0`` for as long as the narration lasts.
Dashboard / split scenes whose narration outlasts the run segment hold their last frame.
``speed: 2`` plays a run segment faster (the top bar then shows "PLAYBACK 2x").
Captions are burned in (footer line) and written as a sidecar SRT; the final MP4 also carries
them as a soft subtitle track. Rendering is deterministic: the same timeline and run
directories give the same frames.

Chase views (``chase: three``) come from :mod:`video.chase`: one shared Three.js renderer for the
whole video, optionally held at ``chase_fps`` and cached on disk (``chase_cache``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np
import yaml
from PIL import Image

from metagross.contracts.messages import DriveMode
from video import cards, encode, layout, style
from video.chase import DEFAULT_CHASE_FPS, make_chase
from video.draw import Painter, text_width
from video.replay import RunReplay

log = logging.getLogger(__name__)

NARRATION_PAD_S = 0.8  # silence after each scene's narration, s
NARRATION_LEAD_S = 0.3  # narration starts this long after the scene cut, s
FADE_FRAMES = 8  # fade-in from the background colour at every scene start
AUTO = "auto"
WORDS_PER_S_ESTIMATE = 2.4  # rough speech rate for 'auto' scenes when TTS is off; with TTS the real length is used
MIN_SCENE_S = 1.5  # shortest 'auto' scene, s
WATERMARK_PX = 16  # draft watermark chip text size, px
WATERMARK_Y = 32  # draft watermark chip centre (top bar line; free space on dashboards and cards), px
WATERMARK_CX = 1260  # between the top-bar chips and the SIMULATED chip, px


@dataclass
class Scene:
    """One timeline entry (see module docstring for the keys)."""

    type: str
    params: dict[str, Any]
    narration: str = ""
    duration: float = 0.0  # resolved duration, s
    speech_s: float = 0.0  # synthesised narration length, s (0 = silent)
    audio: Optional[Path] = None
    cues: list[encode.Cue] = field(default_factory=list)

    @property
    def speed(self) -> float:
        """Run-time seconds per video second (dashboard / split scenes)."""
        return float(self.params.get("speed", 1.0))

    @property
    def is_auto(self) -> bool:
        if self.type in ("dashboard", "split"):
            return str(self.params.get("t1", AUTO)) == AUTO
        return str(self.params.get("duration", AUTO)) == AUTO

    def estimated_speech_s(self) -> float:
        return len(self.narration.split()) / WORDS_PER_S_ESTIMATE if self.narration else 0.0

    @property
    def nominal_s(self) -> float:
        """Scene length before narration is known (video seconds); 'auto' scenes use a speech estimate."""
        if self.is_auto:
            return max(MIN_SCENE_S, self.estimated_speech_s() + NARRATION_PAD_S)
        if self.type in ("dashboard", "split"):
            return (float(self.params["t1"]) - float(self.params["t0"])) / self.speed
        return float(self.params["duration"])

    def run_t(self, k: int, fps: float) -> float:
        """Run-clock time of the k-th frame of a dashboard / split scene (held at t1 if set)."""
        t = float(self.params["t0"]) + k / fps * self.speed
        if not self.is_auto:
            t = min(t, float(self.params["t1"]))
        return t


class Timeline:
    """Scenes + renderer registry. ``chase_renderers`` maps names used in the YAML to per-run
    factories (``for_run(run_dir) -> renderer``, e.g. ``ThreeChaseFactory``) or to zero-argument
    callables returning a ``(state, w, h) -> rgb`` renderer (see :func:`video.chase.make_chase`)."""

    def __init__(self, scenes: list[Scene], fps: int = encode.FPS, root: Path | None = None,
                 chase_renderers: Optional[dict[str, Any]] = None, chase_cache: Optional[Path] = None,
                 chase_fps: float = DEFAULT_CHASE_FPS, voices: Optional[tuple[str, ...]] = None,
                 tts_rate: str = encode.TTS_RATE, watermark: str = "") -> None:
        self.scenes = scenes
        self.watermark = watermark  # e.g. "DRAFT · non-golden runs": drawn on every frame when set
        self.fps = fps
        self.voices = tuple(voices) if voices else encode.VOICES
        self.tts_rate = tts_rate
        self.root = root or Path.cwd()
        self.chase_renderers = dict(chase_renderers or {})
        self.chase_cache = chase_cache
        self.chase_fps = chase_fps
        self._replays: dict[tuple, RunReplay] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dicts(cls, spec: dict, root: Path | None = None, **kw: Any) -> "Timeline":
        scenes = []
        for d in spec["scenes"]:
            d = dict(d)
            typ = d.pop("type")
            scenes.append(Scene(type=typ, narration=str(d.pop("narration", "") or "").strip(), params=d))
        if spec.get("voice") and "voices" not in kw:  # timeline-level voice choice, e.g. en-IN-NeerjaNeural
            v = spec["voice"]
            kw["voices"] = (v,) if isinstance(v, str) else tuple(v)
        if spec.get("tts_rate") and "tts_rate" not in kw:
            kw["tts_rate"] = str(spec["tts_rate"])
        return cls(scenes, fps=int(spec.get("fps", encode.FPS)), root=root, **kw)

    @classmethod
    def from_yaml(cls, path: str | Path, root: Path | None = None, **kw: Any) -> "Timeline":
        p = Path(path)
        return cls.from_dicts(yaml.safe_load(p.read_text(encoding="utf-8")), root=root or Path.cwd(), **kw)

    # ------------------------------------------------------------------ planning
    def plan(self, work_dir: Path, tts: bool = True) -> float:
        """Synthesise narration (if ``tts``), resolve durations and caption cues. Returns total seconds."""
        work_dir.mkdir(parents=True, exist_ok=True)
        t = 0.0
        for sc in self.scenes:
            speech_s = 0.0
            marks: list[encode.SpeechMark] = []
            if sc.narration and tts:
                sc.audio, marks = encode.synthesize_speech_cached(sc.narration, work_dir / "tts", self.voices, self.tts_rate)
                if sc.audio is not None:
                    speech_s = encode.audio_duration_s(sc.audio)
            if speech_s > 0:  # narrated: the scene lasts at least as long as its speech ('auto': exactly)
                floor = MIN_SCENE_S if sc.is_auto else sc.nominal_s
                sc.duration = max(floor, speech_s + NARRATION_PAD_S)
                sc.speech_s = speech_s
                cue_s = speech_s
            else:  # silent (no TTS / no network): keep the nominal length, spread captions over it
                sc.duration = sc.nominal_s
                cue_s = max(sc.nominal_s - NARRATION_PAD_S, 0.5)
            if marks:  # exact sentence timing from the TTS service
                sc.cues = encode.cues_from_marks(marks, t + NARRATION_LEAD_S, t + NARRATION_LEAD_S + speech_s)
            else:
                sc.cues = encode.timed_cues(sc.narration, t + NARRATION_LEAD_S, cue_s) if sc.narration else []
            t += sc.duration
        return t

    # ------------------------------------------------------------------ frames
    def _replay(self, run: str, chase: Optional[str], size: tuple[int, int]) -> RunReplay:
        key = (run, chase, size)
        if key not in self._replays:
            factory = self.chase_renderers.get(chase) if chase else None
            if chase and factory is None:
                log.warning("chase renderer %r not registered; using the top-down placeholder", chase)
            run_dir = self.root / run
            renderer = make_chase(factory, run_dir, self.chase_cache, self.chase_fps) if factory is not None else None
            self._replays[key] = RunReplay(run_dir, chase_renderer=renderer, chase_size=size)
        return self._replays[key]

    def prepare(self) -> None:
        """Open every run (and resolve its chase scenario) up front, so a wrong path or scenario fails
        before minutes of frames have been rendered."""
        for sc in self.scenes:
            P = sc.params
            if sc.type == "dashboard":
                self._replay(P["run"], P.get("chase"), layout.CHASE_SIZE)
            elif sc.type == "split":
                size = (layout.SPLIT_W, layout.SPLIT_CHASE_H)
                self._replay(P["run_a"], P.get("chase_a", P.get("chase")), size)
                self._replay(P["run_b"], P.get("chase_b", P.get("chase")), size)

    def close(self) -> None:
        """Release chase renderers that hold resources (e.g. the browser of ``chase: three``)."""
        for f in self.chase_renderers.values():
            if hasattr(f, "close"):
                try:
                    f.close()
                except Exception as exc:  # noqa: BLE001 - best effort on shutdown
                    log.warning("closing chase renderer failed: %s", exc)

    def scene_frames(self, sc: Scene, t_scene0: float, only: Optional[list[int]] = None) -> Iterator[np.ndarray]:
        """Frames of one scene (``sc.duration`` seconds at ``self.fps``); ``only`` renders just those
        frame indices (stills / previews) without composing the frames in between."""
        n = int(round(sc.duration * self.fps))
        ks = range(n) if only is None else [k for k in only if 0 <= k < n]
        P = sc.params

        def subtitle_at(k: int) -> str:
            tt = t_scene0 + k / self.fps
            return next((c.text for c in sc.cues if c.start_s <= tt < c.end_s), "")

        if sc.type == "card":
            kind = P.get("card", "title")
            P = {**P, **{k: str(self.root / P[k]) for k in ("path", "chart") if P.get(k)}}  # files relative to root
            cache: dict[str, np.ndarray] = {}
            for k in ks:
                sub = subtitle_at(k)
                if sub not in cache:
                    cache.clear()
                    cache[sub] = self._card(kind, P, sub)
                yield cache[sub]
        elif sc.type == "dashboard":
            rp = self._replay(P["run"], P.get("chase"), layout.CHASE_SIZE)
            dash = layout.Dashboard(config_label=P.get("config_label"), gt_at=lambda t, rp=rp: rp.gt_pose_at(t)[:3],
                                    playback=sc.speed)
            cap = layout.Caption(**P["caption"]) if P.get("caption") else None
            for k in ks:
                yield dash.compose(rp.frame_at(min(sc.run_t(k, self.fps), rp.t_end)), caption=cap, subtitle=subtitle_at(k))
        elif sc.type == "image":  # full-screen still (e.g. an operator-console capture), letterboxed
            base = self._image_frame(self.root / P["path"])
            cap = layout.Caption(**P["caption"]) if P.get("caption") else None
            cache_img: dict[str, np.ndarray] = {}
            for k in ks:
                sub = subtitle_at(k)
                if sub not in cache_img:
                    cache_img.clear()
                    img = base.copy()
                    p = Painter(img)
                    if cap is not None:
                        layout.lower_third(p, cap, 48, layout.H - 96, 1100)
                    if sub:
                        plate = img[layout.H - 64:layout.H - 16, 0:layout.W]
                        plate[...] = (plate.astype(np.float32) * 0.15 + np.array(style.BG, np.float32) * 0.85).astype(np.uint8)
                        p.text(layout.W / 2, layout.H - 40, sub, 24, style.TEXT, "regular", "mm")
                    cache_img[sub] = p.flush()
                yield cache_img[sub]
        elif sc.type == "split":
            size = (layout.SPLIT_W, layout.SPLIT_CHASE_H)
            ra = self._replay(P["run_a"], P.get("chase_a", P.get("chase")), size)
            rb = self._replay(P["run_b"], P.get("chase_b", P.get("chase")), size)
            da = layout.Dashboard(config_label=P.get("config_a"), gt_at=lambda t, r=ra: r.gt_pose_at(t)[:3], playback=sc.speed)
            db = layout.Dashboard(config_label=P.get("config_b"), gt_at=lambda t, r=rb: r.gt_pose_at(t)[:3], playback=sc.speed)
            cap = layout.Caption(**P["caption"]) if P.get("caption") else None
            for k in ks:
                tt = sc.run_t(k, self.fps)
                yield layout.compose_split(da, ra.frame_at(min(tt, ra.t_end)), db, rb.frame_at(min(tt, rb.t_end)),
                                           P.get("label_a", "A"), P.get("label_b", "B"), caption=cap, subtitle=subtitle_at(k))
        else:
            raise ValueError(f"unknown scene type {sc.type!r}")

    @staticmethod
    def _image_frame(path: Path) -> np.ndarray:
        """PNG/JPG -> 1920x1080 RGB, aspect preserved, letterboxed on the console background."""
        img = np.empty((layout.H, layout.W, 3), np.uint8)
        img[...] = style.BG
        if not path.exists():
            log.warning("image scene: %s missing; showing a blank frame", path)
            return img
        src = np.asarray(Image.open(path).convert("RGB"))
        s = min(layout.W / src.shape[1], layout.H / src.shape[0])
        w, h = int(round(src.shape[1] * s)), int(round(src.shape[0] * s))
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        y0, x0 = (layout.H - h) // 2, (layout.W - w) // 2
        img[y0:y0 + h, x0:x0 + w] = src
        return img

    @staticmethod
    def _card(kind: str, P: dict, sub: str) -> np.ndarray:
        if kind == "title":
            kw = {"kicker": P["kicker"]} if P.get("kicker") else {}
            return cards.title_card(P.get("title", ""), P.get("subtitle", ""), subtitle_text=sub, **kw)
        if kind == "evidence":
            return cards.evidence_card(str(P.get("headline", "")), P.get("unit", ""), P.get("label", ""),
                                       P.get("claim_label", "Simulated"), P.get("chart"), kicker=P.get("kicker", "Evidence"),
                                       bullets=P.get("bullets", ()), source=P.get("source", ""), subtitle_text=sub)
        if kind == "end":
            kw = {"kicker": P["kicker"]} if P.get("kicker") else {}
            return cards.end_card(P.get("title", "Unknown is never free."), P.get("lines", ()), subtitle_text=sub, **kw)
        if kind == "image":
            return cards.image_card(P["path"], P.get("title", ""), P.get("kicker", ""), subtitle_text=sub)
        raise ValueError(f"unknown card kind {kind!r}")

    def frames(self) -> Iterator[np.ndarray]:
        """All frames of the (planned) timeline, with a short fade-in at each scene start."""
        t = 0.0
        for sc in self.scenes:
            bg = np.array(style.DECK_BG if sc.type == "card" else style.BG, np.float32)
            for k, f in enumerate(self.scene_frames(sc, t)):
                if k < FADE_FRAMES:
                    a = (k + 1) / (FADE_FRAMES + 1)
                    f = np.clip(f.astype(np.float32) * a + bg * (1 - a) + 0.5, 0, 255).astype(np.uint8)
                if self.watermark:
                    f = f.copy()  # card frames are shared between frames
                    p = Painter(f)
                    layout.outlined_chip(p, WATERMARK_CX - text_width(self.watermark, WATERMARK_PX, "bold") / 2 - 12,
                                         WATERMARK_Y, self.watermark, style.MODE_COLORS[DriveMode.STOP_AND_LOOK], WATERMARK_PX)
                    p.flush()
                yield f
            t += sc.duration

    # ------------------------------------------------------------------ output
    def render(self, out_mp4: str | Path, work_dir: Optional[Path] = None, tts: bool = True, preset: str = encode.PRESET) -> dict:
        """Plan, render, encode, mix narration and mux. Returns a summary dict."""
        out = Path(out_mp4)
        work = work_dir or out.parent / (out.stem + "_work")
        total = self.plan(work, tts=tts)
        self.prepare()
        silent = work / "video_silent.mp4"
        t_wall = time.perf_counter()
        try:
            n = encode.write_mp4(silent, self.frames(), fps=self.fps, preset=preset)
        finally:
            self.close()
        cues = [c for sc in self.scenes for c in sc.cues]
        srt = encode.write_srt(cues, out.with_suffix(".srt"))
        clips, t = [], 0.0
        for sc in self.scenes:
            if sc.audio is not None:
                clips.append((t + NARRATION_LEAD_S, sc.audio))
            t += sc.duration
        wav = encode.mix_narration(clips, total, work / "narration.wav") if clips else None
        encode.mux(silent, wav, out, srt=srt)
        wall = time.perf_counter() - t_wall
        log.info("timeline -> %s: %d frames, %.1f s, %d narrated scenes, %.0f s wall (frames + encode)", out, n, total,
                 len(clips), wall)
        return {"path": str(out), "frames": n, "seconds": round(total, 2), "narrated_scenes": len(clips),
                "srt": str(srt), "audio": wav is not None, "wall_s": round(wall, 1),
                "scenes": [{"type": sc.type, "speech_s": round(sc.speech_s, 2), "duration_s": round(sc.duration, 2)}
                           for sc in self.scenes]}


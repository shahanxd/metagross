"""Frames -> H.264 MP4, narration text-to-speech, audio mux and SRT captions.

* :class:`Mp4Writer` streams RGB frames to ffmpeg (``imageio-ffmpeg``'s bundled binary):
  libx264, yuv420p, CRF 18, 30 fps, ``+faststart``.
* :func:`synthesize_speech` renders narration with edge-tts (voices tried in order
  :data:`VOICES`: en-IN-PrabhatNeural, en-IN-NeerjaNeural). edge-tts needs network access; if
  it fails the caller gets ``None`` and should fall back to silence.
  :func:`synthesize_speech_cached` keeps the MP3 + sentence marks by content hash.
* :func:`mix_narration` places decoded clips at their scene start times into one mono WAV
  (numpy mixing, deterministic), :func:`mux` combines video + WAV (+ optional soft subtitle
  track) without re-encoding the video.
* :func:`write_srt` writes sidecar captions; burned-in captions are drawn by the compositor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import imageio_ffmpeg
import numpy as np

log = logging.getLogger(__name__)

FPS = 30
CRF = 18
ENCODE_THREADS = 2  # libx264 threads: the render machine is shared (at most 2 busy cores per job)
PRESET = "medium"  # final quality; "veryfast" for drafts
AUDIO_RATE_HZ = 24000
AUDIO_BITRATE = "160k"
VOICES: tuple[str, ...] = ("en-IN-PrabhatNeural", "en-IN-NeerjaNeural")  # Indian English, male then female
TTS_RATE = "-4%"  # slightly slower than default: short sentences, room to breathe


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


class Mp4Writer:
    """Context manager: ``with Mp4Writer(path, (w, h)) as w: w.write(frame)``; frames are (h, w, 3) uint8 RGB."""

    def __init__(self, path: str | Path, size: tuple[int, int], fps: int = FPS, crf: int = CRF, preset: str = PRESET,
                 threads: int = ENCODE_THREADS) -> None:
        self.path = Path(path)
        self.size = size
        self.fps = fps
        self.n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._gen = imageio_ffmpeg.write_frames(
            str(self.path), size, fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=None,
            macro_block_size=8, ffmpeg_log_level="error",
            output_params=["-crf", str(crf), "-preset", preset, "-threads", str(threads), "-movflags", "+faststart"],
        )
        self._gen.send(None)

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise ValueError(f"frame must be ({h}, {w}, 3) uint8, got {frame.shape} {frame.dtype}")
        self._gen.send(np.ascontiguousarray(frame))
        self.n += 1

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None
            log.info("wrote %s (%d frames, %.1f s)", self.path, self.n, self.n / self.fps)

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_mp4(path: str | Path, frames: Iterable[np.ndarray], fps: int = FPS, crf: int = CRF, preset: str = PRESET) -> int:
    """Encode an iterable of equally sized RGB frames; returns the number of frames written."""
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    with Mp4Writer(path, (w, h), fps, crf, preset) as wr:
        wr.write(first)
        for f in it:
            wr.write(f)
        return wr.n


def probe(path: str | Path) -> dict:
    """Open an MP4 with imageio's ffmpeg reader: {'n_frames', 'fps', 'duration', 'size', 'first_frame_mean'}."""
    import imageio.v2 as iio

    r = iio.get_reader(str(path), format="ffmpeg")
    try:
        meta = r.get_meta_data()
        first = r.get_data(0)
        n = r.count_frames()
    finally:
        r.close()
    return {"n_frames": int(n), "fps": float(meta.get("fps", 0.0)), "duration": float(meta.get("duration", 0.0)),
            "size": tuple(meta.get("size", (0, 0))), "first_frame_mean": float(np.mean(first))}


# ----------------------------------------------------------------------------- narration
@dataclass(frozen=True)
class SpeechMark:
    """Sentence boundary reported by the TTS service (seconds from the clip start)."""

    offset_s: float
    duration_s: float
    text: str


async def _tts_stream(text: str, voice: str, rate: str, out: Path) -> list[SpeechMark]:
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="SentenceBoundary")
    marks: list[SpeechMark] = []
    with out.open("wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("SentenceBoundary", "WordBoundary"):
                marks.append(SpeechMark(chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]))  # 100 ns units
    return marks


def synthesize_speech_timed(text: str, out_path: str | Path, voices: Sequence[str] = VOICES,
                            rate: str = TTS_RATE) -> tuple[Optional[Path], list[SpeechMark]]:
    """Render ``text`` to an MP3 with edge-tts (voices tried in order) and return sentence marks.
    Returns (None, []) if every voice fails (no network, service change) or edge-tts is missing."""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        log.warning("edge-tts not installed; narration will be silent")
        return None, []
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for voice in voices:
        try:
            marks = asyncio.run(_tts_stream(text, voice, rate, out))
            if out.exists() and out.stat().st_size > 0:
                log.info("tts %s -> %s (%d sentence marks)", voice, out.name, len(marks))
                return out, marks
        except Exception as exc:  # noqa: BLE001 - network / service errors: try the next voice
            log.warning("edge-tts voice %s failed: %s", voice, exc)
    return None, []


def synthesize_speech_cached(text: str, cache_dir: str | Path, voices: Sequence[str] = VOICES,
                             rate: str = TTS_RATE) -> tuple[Optional[Path], list[SpeechMark]]:
    """:func:`synthesize_speech_timed` with a content-addressed cache (text, voices, rate) in ``cache_dir``:
    ``<sha1>.mp3`` + ``<sha1>.json`` (sentence marks). Re-renders after a layout change need no network."""
    key = hashlib.sha1(json.dumps([text, list(voices), rate]).encode("utf-8")).hexdigest()[:16]
    d = Path(cache_dir)
    mp3, meta = d / f"tts_{key}.mp3", d / f"tts_{key}.json"
    if mp3.exists() and meta.exists() and mp3.stat().st_size > 0:
        marks = [SpeechMark(**m) for m in json.loads(meta.read_text(encoding="utf-8"))]
        return mp3, marks
    path, marks = synthesize_speech_timed(text, mp3, voices, rate)
    if path is not None:
        meta.write_text(json.dumps([m.__dict__ for m in marks]), encoding="utf-8")
    return path, marks


def synthesize_speech(text: str, out_path: str | Path, voices: Sequence[str] = VOICES, rate: str = TTS_RATE) -> Optional[Path]:
    """Render ``text`` to an MP3 with edge-tts; tries ``voices`` in order. Returns the path or None."""
    return synthesize_speech_timed(text, out_path, voices, rate)[0]


def cues_from_marks(marks: Sequence[SpeechMark], start_s: float, end_s: float) -> list[Cue]:
    """Sentence marks -> caption cues; each cue lasts until the next sentence starts (last one until end_s)."""
    cues = []
    for i, m in enumerate(marks):
        a = start_s + m.offset_s
        b = start_s + marks[i + 1].offset_s if i + 1 < len(marks) else max(end_s, a + m.duration_s)
        cues.append(Cue(a, b, m.text.strip()))
    return cues


def decode_audio(path: str | Path, rate: int = AUDIO_RATE_HZ) -> np.ndarray:
    """Any audio file -> mono float32 samples in [-1, 1] at ``rate`` Hz (via ffmpeg)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def audio_duration_s(path: str | Path, rate: int = AUDIO_RATE_HZ) -> float:
    return len(decode_audio(path, rate)) / rate


def mix_narration(clips: Sequence[tuple[float, str | Path]], total_s: float, out_wav: str | Path,
                  rate: int = AUDIO_RATE_HZ) -> Path:
    """Place each (start_s, audio_file) into a silent track of ``total_s`` seconds; writes 16-bit mono WAV."""
    buf = np.zeros(int(round(total_s * rate)), np.float32)
    for start, f in clips:
        a = decode_audio(f, rate)
        i0 = int(round(start * rate))
        n = max(0, min(len(a), len(buf) - i0))
        buf[i0:i0 + n] += a[:n]
    pcm = (np.clip(buf, -1.0, 1.0) * 32767.0).astype(np.int16)
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out


def mux(video: str | Path, audio_wav: Optional[str | Path], out_path: str | Path, srt: Optional[str | Path] = None) -> Path:
    """Copy the H.264 stream, add AAC audio (and a mov_text subtitle track if ``srt``)."""
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", str(video)]
    maps = ["-map", "0:v:0"]
    if audio_wav is not None:
        cmd += ["-i", str(audio_wav)]
        maps += ["-map", f"{1}:a:0"]
    if srt is not None:
        cmd += ["-i", str(srt)]
        maps += ["-map", f"{2 if audio_wav is not None else 1}:s:0"]
    cmd += maps + ["-c:v", "copy"]
    if audio_wav is not None:
        cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE]
    if srt is not None:
        cmd += ["-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(out_path)


# ----------------------------------------------------------------------------- captions
@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str


def _srt_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: Sequence[Cue], path: str | Path) -> Path:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{_srt_time(c.start_s)} --> {_srt_time(c.end_s)}", c.text, ""]
    p = Path(path)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def split_sentences(text: str) -> list[str]:
    """Narration text -> caption lines (one sentence each)."""
    out, cur = [], ""
    for ch in text.strip():
        cur += ch
        if ch in ".?!" and len(cur.strip()) > 1:
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def timed_cues(text: str, start_s: float, dur_s: float) -> list[Cue]:
    """Spread the sentences of ``text`` over [start, start + dur] proportionally to their length."""
    sents = split_sentences(text)
    if not sents:
        return []
    w = np.array([max(len(s), 8) for s in sents], np.float64)
    edges = start_s + np.concatenate([[0.0], np.cumsum(w / w.sum() * dur_s)])
    return [Cue(float(edges[i]), float(edges[i + 1]), s) for i, s in enumerate(sents)]

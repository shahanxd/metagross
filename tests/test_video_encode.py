"""Encoder: playable H.264 MP4, audio mixing / mux, SRT, timeline rendering without TTS."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from video import encode
from video.timeline import Timeline


def _frames(n: int, w: int = 320, h: int = 240):
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, : (i + 1) * w // n] = (59, 130, 246)
        yield f


def _sine_wav(path: Path, seconds: float, rate: int = encode.AUDIO_RATE_HZ) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def test_write_mp4_is_playable(tmp_path: Path) -> None:
    out = tmp_path / "a.mp4"
    assert encode.write_mp4(out, _frames(15)) == 15
    info = encode.probe(out)
    assert info["n_frames"] == 15
    assert info["size"] == (320, 240)
    assert info["fps"] == pytest.approx(30.0)
    assert info["first_frame_mean"] > 0.0


def test_writer_rejects_wrong_frame_shape(tmp_path: Path) -> None:
    with encode.Mp4Writer(tmp_path / "b.mp4", (320, 240)) as w:
        with pytest.raises(ValueError):
            w.write(np.zeros((100, 100, 3), np.uint8))
        w.write(np.zeros((240, 320, 3), np.uint8))


def test_mix_and_mux_adds_audio_and_subtitles(tmp_path: Path) -> None:
    vid = tmp_path / "v.mp4"
    encode.write_mp4(vid, _frames(30))
    clip = _sine_wav(tmp_path / "s.wav", 0.4)
    wav = encode.mix_narration([(0.2, clip)], 1.0, tmp_path / "mix.wav")
    a = encode.decode_audio(wav)
    assert len(a) == encode.AUDIO_RATE_HZ
    assert np.abs(a[: int(0.19 * encode.AUDIO_RATE_HZ)]).max() < 1e-3  # silence before the clip
    assert np.abs(a[int(0.25 * encode.AUDIO_RATE_HZ):int(0.55 * encode.AUDIO_RATE_HZ)]).max() > 0.2
    srt = encode.write_srt([encode.Cue(0.0, 0.9, "Hello.")], tmp_path / "c.srt")
    out = encode.mux(vid, wav, tmp_path / "out.mp4", srt=srt)
    err = subprocess.run([encode.ffmpeg_exe(), "-i", str(out)], capture_output=True, text=True).stderr
    assert "Video: h264" in err and "Audio: aac" in err and "Subtitle: mov_text" in err


def test_srt_and_cue_timing(tmp_path: Path) -> None:
    cues = encode.timed_cues("First sentence here. Second one! Third?", 1.0, 6.0)
    assert [c.text for c in cues] == ["First sentence here.", "Second one!", "Third?"]
    assert cues[0].start_s == pytest.approx(1.0) and cues[-1].end_s == pytest.approx(7.0)
    text = encode.write_srt(cues, tmp_path / "x.srt").read_text(encoding="utf-8")
    assert "00:00:01,000 --> " in text and text.startswith("1\n")


def test_timeline_card_scene_renders_without_tts(tmp_path: Path) -> None:
    spec = {"fps": 30, "scenes": [
        {"type": "card", "card": "title", "duration": 0.5, "title": "T", "narration": "One. Two."},
        {"type": "card", "card": "end", "duration": 0.3},
    ]}
    tl = Timeline.from_dicts(spec, root=tmp_path)
    summary = tl.render(tmp_path / "tl.mp4", tts=False)
    assert summary["frames"] == 24 and not summary["audio"]
    info = encode.probe(tmp_path / "tl.mp4")
    assert info["n_frames"] == 24 and info["size"] == (1920, 1080)
    assert (tmp_path / "tl.srt").read_text(encoding="utf-8").count("-->") == 2


def test_placeholder_fill_and_guard() -> None:
    from video.render_timeline import fill, unfilled

    spec = {"scenes": [{"narration": "Drift {kitti_drift_1km} percent. R {rcert_min_m} m.", "lines": ["{miou}"]}]}
    out = fill(spec, {"kitti_drift_1km": 1.2})
    assert out["scenes"][0]["narration"].startswith("Drift 1.2 percent.")
    assert unfilled(out) == {"rcert_min_m", "miou"}
    assert unfilled(fill(out, {"rcert_min_m": 3, "miou": 0.7})) == set()


def test_cues_from_speech_marks() -> None:
    marks = [encode.SpeechMark(0.1, 1.0, "One."), encode.SpeechMark(1.5, 0.8, "Two.")]
    cues = encode.cues_from_marks(marks, 10.0, 12.5)
    assert [c.text for c in cues] == ["One.", "Two."]
    assert [c.start_s for c in cues] == pytest.approx([10.1, 11.5]) and [c.end_s for c in cues] == pytest.approx([11.5, 12.5])


def test_image_scene_letterboxes(tmp_path: Path) -> None:
    import cv2

    cv2.imwrite(str(tmp_path / "still.png"), np.full((720, 1280, 3), 200, np.uint8))
    tl = Timeline.from_dicts({"scenes": [{"type": "image", "path": "still.png", "duration": 0.2,
                                          "caption": {"title": "Console"}}]}, root=tmp_path)
    tl.plan(tmp_path / "w", tts=False)
    frames = list(tl.scene_frames(tl.scenes[0], 0.0))
    assert len(frames) == 6 and frames[0].shape == (1080, 1920, 3)
    assert frames[0][540, 1800].mean() > 150  # scaled still fills the frame width

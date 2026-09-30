"""METAGROSS demo-video toolchain.

Modules
-------
* :mod:`video.style`    colour / typography tokens shared by the compositor and the cards
                        (console theme for dashboards, deck theme for cards).
* :mod:`video.replay`   run directory -> per-frame panel data at 30 fps.
* :mod:`video.layout`   1920 x 1080 dashboard compositor (single run and split screen).
* :mod:`video.cards`    full-screen title / evidence / end cards in the deck style.
* :mod:`video.timeline` declarative scene list (YAML or Python) -> frames + captions.
* :mod:`video.encode`   frames -> H.264 MP4, narration TTS, audio mux, SRT captions.

Everything here is offline tooling: it reads run logs written by the simulator and the
autonomy process and never feeds anything back into either.
"""

from __future__ import annotations

__all__ = ["style", "replay", "layout", "cards", "timeline", "encode"]

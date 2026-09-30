"""Deterministic emulator of the narrow-band operator radio link.

Models, per packet: serialisation at ``bandwidth_bps`` behind a FIFO (the channel is
busy while a packet is on air), random loss with probability ``loss``, and a fixed
propagation/processing ``latency_s``. Packets that would wait longer than
``max_queue_s`` for the channel are dropped (tail drop). Seeded -> reproducible.
Defaults come from ``LINK_KBPS``, ``LINK_LOSS`` and ``LINK_LATENCY_S``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from metagross.config.defaults import LINK_KBPS, LINK_LATENCY_S, LINK_LOSS


@dataclass(slots=True)
class LinkStats:
    sent: int = 0
    delivered: int = 0
    dropped_loss: int = 0
    dropped_queue: int = 0
    bytes_offered: int = 0
    bytes_delivered: int = 0
    latencies_s: list[float] = field(default_factory=list)


class LinkEmulator:
    """``send(t, payload)`` then ``poll(t)`` to receive packets that have arrived by time t."""

    def __init__(self, bandwidth_bps: float = LINK_KBPS * 1000.0, loss: float = LINK_LOSS, latency_s: float = LINK_LATENCY_S,
                 seed: int = 0, max_queue_s: float = 5.0) -> None:
        if bandwidth_bps <= 0:
            raise ValueError("bandwidth must be positive")
        self.bps = float(bandwidth_bps)
        self.loss = float(loss)
        self.latency_s = float(latency_s)
        self.max_queue_s = float(max_queue_s)
        self.rng = np.random.default_rng(seed)
        self._busy_until = 0.0
        self._heap: list[tuple[float, int, float, bytes]] = []
        self._n = 0
        self.stats = LinkStats()

    def send(self, t: float, payload: bytes) -> bool:
        """Offer a packet at time t [s]. Returns False if it was dropped (queue or loss)."""
        self.stats.sent += 1
        self.stats.bytes_offered += len(payload)
        start = max(t, self._busy_until)
        if start - t > self.max_queue_s:
            self.stats.dropped_queue += 1
            return False
        end = start + 8.0 * len(payload) / self.bps
        self._busy_until = end
        if self.rng.random() < self.loss:  # lost on air: still used the channel
            self.stats.dropped_loss += 1
            return False
        heapq.heappush(self._heap, (end + self.latency_s, self._n, t, payload))
        self._n += 1
        return True

    def poll(self, t: float) -> list[tuple[float, bytes]]:
        """Packets with arrival time <= t, in arrival order, as (arrival_t, payload)."""
        out = []
        while self._heap and self._heap[0][0] <= t:
            arr, _, t_sent, payload = heapq.heappop(self._heap)
            self.stats.delivered += 1
            self.stats.bytes_delivered += len(payload)
            self.stats.latencies_s.append(arr - t_sent)
            out.append((arr, payload))
        return out

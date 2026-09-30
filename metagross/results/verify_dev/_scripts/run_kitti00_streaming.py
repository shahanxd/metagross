"""Run metagross.eval.kitti_vo.evaluate_sequence('00') while the right images are still downloading.

Identical code path to ``python -m metagross.eval.kitti_vo --seqs 00 --threads 2``: the only change is a
KittiSequence subclass whose frame iterator blocks (outside every timed region of run_vo) until the next
stereo pair exists on disk and decodes. image_0 must be complete (4541 files) before this starts.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

cv2.setNumThreads(2)

from metagross.eval import kitti_vo  # noqa: E402

EXPECTED = 4541
WAIT_TIMEOUT_S = 1800.0
LOG = logging.getLogger("stream00")


class WaitingFrames(list):
    """List of (left, right) paths whose iteration waits until each pair is on disk and decodable."""

    def __getitem__(self, idx):  # keep the waiting behaviour through run_vo's [:None] slice
        out = super().__getitem__(idx)
        return WaitingFrames(out) if isinstance(idx, slice) else out

    def __iter__(self):
        for lp, rp in list.__iter__(self):
            t0 = time.time()
            while True:
                if lp.exists() and rp.exists():
                    if cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE) is not None:
                        break
                if time.time() - t0 > WAIT_TIMEOUT_S:
                    raise TimeoutError(f"timed out waiting for {rp}")
                time.sleep(0.5)
            yield lp, rp


class StreamingSequence(kitti_vo.KittiSequence):
    def available(self) -> bool:
        n0 = len(list((self.dir / "image_0").glob("*.png")))
        return n0 == EXPECTED and self.gt_path.exists() and (self.dir / "calib.txt").exists()

    def frames(self):
        return WaitingFrames(super().frames())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seq = StreamingSequence("00")
    while not seq.available():
        LOG.info("waiting for image_0 to complete ...")
        time.sleep(10)
    kitti_vo.KittiSequence = StreamingSequence  # evaluate_sequence constructs KittiSequence(seq_id)
    kitti_vo.evaluate_sequence("00", None, 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

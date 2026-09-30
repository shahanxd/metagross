p = r"D:\Downloads\sih again\metagross\metagross\autonomy\perception\pipeline.py"
s = open(p, encoding="utf-8").read()
reps = [
("""frame of a Perception instance is not filtered.
\"\"\"
""", """frame of a Perception instance is not filtered.

Latency: the segmenter (ONNX Runtime, releases the GIL) runs on a worker thread that is started
as soon as the frame is known (:meth:`Perception.compute_disparity` or the top of
:meth:`Perception.process`), so it overlaps SGBM / geometry instead of adding to them
(``config['async_semantics']``, default True; outputs are identical either way).
\"\"\"
"""),
("""import logging
import time
from typing import Any, Optional
""", """import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional
"""),
("""    "ditch_persistence": True,  # confirm ditch candidates across frames in the odometry frame (persistence.py)
""", """    "ditch_persistence": True,  # confirm ditch candidates across frames in the odometry frame (persistence.py)
    "async_semantics": True,  # run the segmenter on a worker thread, overlapping SGBM (same outputs)
"""),
("""        self._disp_cache: Optional[tuple[int, float, np.ndarray, float]] = None  # (seq, t, disparity, ms)
""", """        self._disp_cache: Optional[tuple[int, float, np.ndarray, float]] = None  # (seq, t, disparity, ms)
        self._seg_pool: Optional[ThreadPoolExecutor] = None
        if segmenter is not None and self.config.get("async_semantics", True):
            self._seg_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception-seg")
        self._seg_job: Optional[tuple[int, float, Future]] = None  # (seq, t, future of segmenter(left_rgb))

    def _start_semantics(self, frame: SensorFrame) -> None:
        \"\"\"Submit the segmenter for this frame to the worker thread (once per frame).\"\"\"
        if self._seg_pool is None or frame.left_rgb is None or not self.config.get("use_semantics", True):
            return
        if self._seg_job is not None and self._seg_job[0] == frame.seq and self._seg_job[1] == frame.t:
            return
        self._seg_job = (frame.seq, frame.t, self._seg_pool.submit(self.segmenter, frame.left_rgb))

    def _semantics_result(self, frame: SensorFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        \"\"\"(class ids, entropy) of this frame: from the worker if submitted, else computed inline.\"\"\"
        job, self._seg_job = self._seg_job, None
        if job is not None and job[0] == frame.seq and job[1] == frame.t:
            return job[2].result()
        return self.segmenter(frame.left_rgb)
"""),
("""        disp, ms = disparity_for_frame(frame, self.calib, self.matcher)
        self._disp_cache = (frame.seq, frame.t, disp, ms)
        return disp
""", """        self._start_semantics(frame)
        disp, ms = disparity_for_frame(frame, self.calib, self.matcher)
        self._disp_cache = (frame.seq, frame.t, disp, ms)
        return disp
"""),
("""        geom, spec = self.geom, self.spec

        cached, self._disp_cache = self._disp_cache, None
""", """        geom, spec = self.geom, self.spec
        self._start_semantics(frame)  # no-op if compute_disparity() already started it

        cached, self._disp_cache = self._disp_cache, None
"""),
("""            sem_mask, entropy = self.segmenter(frame.left_rgb)
""", """            sem_mask, entropy = self._semantics_result(frame)
"""),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")

"""World <-> autonomy process protocol (multiprocessing.Pipe, spawn context).

The runner (metagross.sim.runner) starts ``metagross.autonomy.process.autonomy_main``
in a separate OS process and exchanges only these tuples:

    runner -> autonomy
        ("reset", MissionSpec, StereoCalibration, VehicleSpec, config: dict)
        ("frame", SensorFrame)
        ("operator", OperatorCmd)
        ("close",)

    autonomy -> runner
        ("ready",)                              after reset
        ("cmd", WheelCmd, Telemetry | None)     one per frame, in order
        ("error", str)                          fatal error text

The autonomy process writes its own logs (debug bundles, telemetry.jsonl,
timings) under ``<run_dir>/autonomy/``; the runner never reads them during a run.
The runner writes ground truth under ``<run_dir>/gt/``; the autonomy process is
forbidden (by a runtime audit hook) from opening anything outside its allow-list.
"""

from __future__ import annotations

MSG_RESET = "reset"
MSG_FRAME = "frame"
MSG_OPERATOR = "operator"
MSG_CLOSE = "close"
MSG_READY = "ready"
MSG_CMD = "cmd"
MSG_ERROR = "error"

# Ablation / configuration switches understood by metagross.autonomy.node.AutonomyStack.
DEFAULT_AUTONOMY_CONFIG: dict = {
    "name": "FULL",
    "unknown_is_free": False,  # True = typical stack: unseen cells cost nothing
    "use_negobs": True,  # missing-ground detector
    "use_governor": True,  # seen-distance speed governor
    "use_health": True,  # integrity monitor gates VO + speed
    "use_semantics": True,  # segmentation cost layer
    "use_wheel_odom": True,  # False = camera-only localisation
    "fixed_speed_mps": None,  # typical stack: constant cruise speed (float) or None
    "speed_cap_mps": None,  # T2 speed-matched control: hard cap (float) or None
    "debug_every_n": 1,  # write DebugBundle every n ticks (0 = never)
    # --- additions (integration, build phase II) ---
    "seg_model_path": None,  # segmentation ONNX (abs or repo-relative); None = first of models/lraspp_offroad5_*.onnx, lraspp_smoke.onnx
    "seg_threads": 2,  # ONNX Runtime intra-op threads of the segmenter
    "launch_apron_m": 2.0,  # radius (m) certified as GROUND at launch (camera near blind zone)
    "seed": 0,  # MPPI sampling / RANSAC seed
    "fixed_latency_s": None,  # governor reaction-time latency override (tests / replays only); None = measured
    "stub_perception": False,  # force the blind stub perception (plumbing tests)
    "stub_localizer": False,  # force the wheel+gyro stub localiser (plumbing tests)
    "debug_image_every_n": 2,  # DebugBundle.extras['left_rgb'] (320x200 RGB) every n ticks (0 = never; stereo mode only)
}

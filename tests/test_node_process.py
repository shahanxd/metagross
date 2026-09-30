"""Autonomy process: runtime file guard (audit hook) and the ipc protocol.

The audit hook cannot be removed once installed, so every test that installs it runs in
a fresh interpreter (subprocess / multiprocessing spawn)."""

from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from metagross.config.defaults import VEHICLE, stereo_calibration
from metagross.contracts import ipc
from metagross.contracts.messages import MissionSpec, OperatorAction, OperatorCmd, SensorFrame

REPO = Path(__file__).resolve().parents[1]


def test_audit_hook_blocks_forbidden_open(tmp_path):
    run_dir = tmp_path / "run"
    gt = run_dir / "gt"
    gt.mkdir(parents=True)
    (gt / "truth.json").write_text("{}")
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(REPO)!r})
        from metagross.autonomy.process import install_file_guard
        run_dir = Path({str(run_dir)!r})
        (run_dir / "autonomy").mkdir(parents=True, exist_ok=True)
        install_file_guard(run_dir)
        with open(run_dir / "autonomy" / "ok.txt", "w") as f:  # allowed
            f.write("ok")
        import metagross.autonomy.planning.mppi  # imports from the allowed package dirs still work
        for bad in [run_dir / "gt" / "truth.json", Path({str(REPO / 'metagross' / 'sim' / '__init__.py')!r}),
                    run_dir / "autonomy" / ".." / "gt" / "truth.json"]:
            try:
                open(bad).close()
            except PermissionError:
                print("BLOCKED", bad.name)
            else:
                print("OPENED", bad.name)
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.split()
    assert lines.count("BLOCKED") == 3 and "OPENED" not in lines, out.stdout
    assert (run_dir / "autonomy" / "ok.txt").read_text() == "ok"


def _frame(k: int) -> SensorFrame:
    return SensorFrame(t=0.2 * k, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0,
                       gyro_z_rps=0.0, sensor_mode="tier0_disparity", disparity=np.zeros((40, 64), np.float32))


def test_process_protocol_with_stubs(tmp_path):
    from metagross.autonomy.process import autonomy_main

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    cfg = {"stub_perception": True, "stub_localizer": True, "debug_every_n": 2}
    proc = ctx.Process(target=autonomy_main, args=(child, str(tmp_path), cfg), daemon=True)
    proc.start()
    try:
        parent.send((ipc.MSG_RESET, MissionSpec("t", (10.0, 0.0), 2.0, 60.0), stereo_calibration(), VEHICLE, {}))
        assert parent.poll(120), "no ready"
        assert parent.recv() == (ipc.MSG_READY,)
        cmds = []
        for k in range(6):
            parent.send((ipc.MSG_FRAME, _frame(k)))
            assert parent.poll(60)
            msg = parent.recv()
            assert msg[0] == ipc.MSG_CMD, msg
            cmds.append(msg)
        parent.send((ipc.MSG_OPERATOR, OperatorCmd(1.3, OperatorAction.HOLD)))
        parent.send((ipc.MSG_FRAME, _frame(6)))
        assert parent.poll(60)
        held = parent.recv()
        parent.send((ipc.MSG_CLOSE,))
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.kill()
    assert proc.exitcode == 0
    assert [c[1].seq for c in cmds] == list(range(6))
    lim = VEHICLE.max_wheel_rad_s + 1e-9
    assert all(np.isfinite(c[1].omega_l_rad_s) and abs(c[1].omega_l_rad_s) <= lim and abs(c[1].omega_r_rad_s) <= lim for c in cmds)
    assert all(c[1].compute_ms > 0 for c in cmds)
    assert held[1].mode.value == "HOLD"
    out = tmp_path / "autonomy"
    tel = [json.loads(line) for line in (out / "telemetry.jsonl").read_text().splitlines()]
    assert len(tel) >= 2 and all(r["packet_bytes"] > 20 for r in tel)
    rows = (out / "timings.csv").read_text().splitlines()
    assert rows[0].startswith("tick,t,compute_ms") and len(rows) == 1 + 7
    npz = sorted((out / "debug").glob("tick_*.npz"))
    assert len(npz) == 4
    with np.load(npz[0]) as z:
        assert "cell_state_local" in z and "plan_xy" in z and json.loads(str(z["extras_json"]))["impl"]["perception"] == "StubBlindPerception"


def test_process_reports_errors(tmp_path):
    from metagross.autonomy.process import autonomy_main

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=autonomy_main, args=(child, str(tmp_path), {}), daemon=True)
    proc.start()
    try:
        parent.send((ipc.MSG_FRAME, _frame(0)))  # frame before reset -> error
        assert parent.poll(120)
        msg = parent.recv()
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.kill()
    assert msg[0] == ipc.MSG_ERROR and "before reset" in msg[1]
    assert proc.exitcode == 0

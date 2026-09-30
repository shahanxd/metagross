"""Closed-loop episode runner: world (this process) <-> autonomy (separate OS process).

Protocol: :mod:`metagross.contracts.ipc` over a ``multiprocessing`` Pipe with the *spawn* context.
The autonomy entry point is called as ``target(conn, run_dir)`` (the convention of
``metagross.autonomy.process.autonomy_main(conn, run_dir, config=None)``): the autonomy writes only
under ``<run_dir>/autonomy/`` and its runtime file guard blocks everything else, including
``<run_dir>/gt/`` where the runner writes ground truth.

Lockstep loop
-------------
1. ``("reset", MissionSpec, StereoCalibration, VehicleSpec, config)`` -> wait for ``("ready",)``;
   then ``("operator", OperatorCmd(GO))``.
2. Every ``round(PHYSICS_HZ / camera_hz)`` physics steps: build a SensorFrame at the current state,
   send ``("frame", frame)``, block until ``("cmd", WheelCmd, Telemetry|None)``. Sim time is frozen
   while the autonomy computes; its reported ``compute_ms`` is re-injected as actuation latency
   (the command takes effect ``ceil(compute_ms / 20 ms)`` physics steps after the frame).
3. Physics sub-steps run between camera ticks; the referee ends the episode.

Outputs: ``run_dir/gt/states.npz`` (t,x,y,z,roll,pitch,yaw,v,omega,wl,wr [+ dyn_xy]),
``run_dir/gt/cmds.npz``, ``run_dir/gt/events.json``, ``run_dir/result.json`` and the operator-side
``run_dir/mission.json`` ({mission_id, goal_xy_a, success_radius_m, timeout_s, goal_sigma_m,
goal_entry: {range_m, bearing_deg}}, see :func:`mission_record`).

The child is started under :func:`neutral_main`, so spawn does not re-import the parent's
``__main__`` (e.g. ``metagross.sim.batch``) into it; ``autonomy_main`` refuses to run if any
``metagross.sim*`` / ``metagross.eval*`` module is resident. The ``mission_id`` sent at reset
is opaque (scenario sha256 prefix, see :func:`metagross.sim.world.opaque_mission_id`).

``sensor_mode='stereo'`` renders images with ``renderer_factory()`` (default: the Three.js
:class:`~metagross.sim.render.bridge.ThreeRenderer`); the world side stops the wheels when no new
command arrived for ``defaults.WHEEL_CMD_TIMEOUT_S`` (see :meth:`World.step`).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from metagross.config import defaults
from metagross.contracts import ipc
from metagross.contracts.interfaces import RendererProto
from metagross.contracts.messages import OperatorAction, OperatorCmd, WheelCmd
from metagross.sim.gridplan import CORRIDOR_WIDTH_M, plan_path
from metagross.sim.referee import count_false_stops
from metagross.sim.scenario import load_scenario
from metagross.sim.world import World

log = logging.getLogger(__name__)

DEFAULT_AUTONOMY_TARGET = "metagross.autonomy.process:autonomy_main"
READY_TIMEOUT_S = 180.0  # model loading may take a while on first start
RECV_TIMEOUT_S = 60.0  # per-frame reply timeout
JOIN_TIMEOUT_S = 10.0
GOAL_SIGMA_BASE_M = 0.5  # goal-entry uncertainty floor (m) before the heading-error term


def resolve_target(target: str | Callable | None) -> Callable:
    """'pkg.module:function' (or a callable) -> callable, imported in the parent so that the
    spawned child unpickles it by reference and imports only that module."""
    target = target or DEFAULT_AUTONOMY_TARGET
    if callable(target):
        return target
    mod, _, fn = target.partition(":")
    return getattr(importlib.import_module(mod), fn or "autonomy_main")


def machine_info() -> dict[str, Any]:
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__}


def _json_num(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def default_renderer_factory() -> RendererProto:
    """The Three.js stereo renderer (headed Chrome/Edge via Playwright) at the default calibration.

    The world calls ``load_scenario`` and feeds ``World.render_state()`` (body pose
    ``[x, y, z, roll, pitch, yaw]``, REP-103, plus ``T_world_cam``) at every camera tick."""
    from metagross.sim.render.bridge import ThreeRenderer

    return ThreeRenderer(defaults.stereo_calibration(), defaults.VEHICLE)


def mission_record(world: World, scn: dict) -> dict[str, Any]:
    """Operator-side mission record written to ``run_dir/mission.json`` (for console / video).

    ``goal_sigma_m = GOAL_SIGMA_BASE_M + |heading_init_err| (rad) * range`` is the 1-sigma goal
    position uncertainty implied by the launch-heading error; ``goal_entry`` is the goal as the
    operator entered it (range and bearing from A, bearing CCW from the launch heading, degrees)."""
    m = world.mission_spec()
    rng = math.hypot(*m.goal_xy_a)
    err = math.radians(float(scn["mission"].get("heading_init_err_deg", 0.0)))
    return {"mission_id": m.mission_id, "goal_xy_a": [round(m.goal_xy_a[0], 4), round(m.goal_xy_a[1], 4)],
            "success_radius_m": m.success_radius_m, "timeout_s": m.timeout_s,
            "goal_sigma_m": round(GOAL_SIGMA_BASE_M + abs(err) * rng, 4),
            "goal_entry": {"range_m": round(rng, 4), "bearing_deg": round(math.degrees(math.atan2(m.goal_xy_a[1], m.goal_xy_a[0])), 4)}}


_SPAWN_LOCK = threading.Lock()


@contextlib.contextmanager
def neutral_main() -> Iterator[None]:
    """Hide the parent's ``__main__`` from ``multiprocessing`` spawn for the duration of ``start()``.

    spawn re-runs the parent's main module in the child (``init_main_from_name`` /
    ``init_main_from_path``) before unpickling the target. When the parent is
    ``python -m metagross.sim.batch`` that re-imports ``metagross.sim.*`` into the autonomy
    process (GT firewall layer 2). A bare module without ``__spec__`` / ``__file__`` makes
    ``multiprocessing.spawn.get_preparation_data`` skip main re-import entirely, so the child
    imports only the target's own module (``metagross.autonomy.process``)."""
    with _SPAWN_LOCK:
        real = sys.modules.get("__main__")
        sys.modules["__main__"] = types.ModuleType("__main__", "neutral spawn main (metagross.sim.runner)")
        try:
            yield
        finally:
            if real is not None:
                sys.modules["__main__"] = real


class _AutonomyLink:
    """Parent side of the Pipe with timeouts."""

    def __init__(self, target: Callable, run_dir: Path):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=target, args=(child, str(run_dir)), name="metagross-autonomy", daemon=True)
        if getattr(target, "__module__", None) == "__main__":
            # a target defined in the parent's main script can only be unpickled by re-running that script
            log.warning("autonomy target %r lives in __main__: the child will re-import the parent's main module", target)
            self.proc.start()
        else:
            with neutral_main():
                self.proc.start()
        child.close()

    def send(self, msg: tuple) -> None:
        self.conn.send(msg)

    def recv(self, timeout_s: float) -> tuple:
        deadline = time.perf_counter() + timeout_s
        while True:
            if self.conn.poll(0.05):
                return self.conn.recv()
            if not self.proc.is_alive():
                raise RuntimeError(f"autonomy process died (exitcode {self.proc.exitcode})")
            if time.perf_counter() > deadline:
                raise TimeoutError(f"no reply from autonomy within {timeout_s:.0f} s")

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self.conn.send((ipc.MSG_CLOSE,))
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(JOIN_TIMEOUT_S)
        if self.proc.is_alive():
            log.warning("autonomy did not exit after close; terminating")
            self.proc.terminate()
            self.proc.join(JOIN_TIMEOUT_S)
        self.conn.close()


def run_episode(scenario_path: str | Path, config: dict | None, run_dir: str | Path, sensor_mode: str = "tier0",
                camera_hz: float = defaults.CAMERA_HZ_BATCH, renderer_factory: Callable[[], RendererProto] | None = None,
                max_wall_s: float | None = None, *, autonomy_target: str | Callable | None = None, noise_seed: int = 0,
                max_sim_s: float | None = None, send_go: bool = True, recv_timeout_s: float = RECV_TIMEOUT_S,
                ready_timeout_s: float = READY_TIMEOUT_S) -> dict:
    """Run one closed-loop episode and write GT logs + ``result.json`` into ``run_dir``.

    ``config`` is merged over ``ipc.DEFAULT_AUTONOMY_CONFIG`` and sent to the autonomy at reset.
    ``max_sim_s`` shortens the mission timeout (tests / smoke runs); ``max_wall_s`` aborts with
    ``failure_type='wall_timeout'``. Returns the result dict.
    """
    wall0 = time.perf_counter()
    run_dir = Path(run_dir)
    (run_dir / "gt").mkdir(parents=True, exist_ok=True)
    auto_dir = run_dir / "autonomy"
    auto_dir.mkdir(parents=True, exist_ok=True)
    scn = load_scenario(scenario_path)
    cfg = {**ipc.DEFAULT_AUTONOMY_CONFIG, **(config or {})}
    renderer = None
    if sensor_mode == "stereo":
        renderer = (renderer_factory or default_renderer_factory)()
    try:
        world = World(scn, sensor_mode=sensor_mode, renderer=renderer, noise_seed=noise_seed, timeout_s=max_sim_s)
    except Exception:
        if renderer is not None:
            renderer.close()
        raise
    mission = world.mission_spec()
    (run_dir / "mission.json").write_text(json.dumps(mission_record(world, scn), indent=1), encoding="utf-8")
    steps_per_frame = max(1, int(round(defaults.PHYSICS_HZ / camera_hz)))
    cmds: dict[str, list] = {k: [] for k in ("t", "seq", "omega_l", "omega_r", "compute_ms", "apply_t", "mode")}
    n_telemetry = 0
    error_text: str | None = None
    link = _AutonomyLink(resolve_target(autonomy_target), run_dir)
    try:
        link.send((ipc.MSG_RESET, mission, world.calibration(), defaults.VEHICLE, cfg))
        msg = link.recv(ready_timeout_s)
        if msg[0] == ipc.MSG_ERROR:
            raise RuntimeError(f"autonomy reset error: {msg[1]}")
        if msg[0] != ipc.MSG_READY:
            raise RuntimeError(f"expected 'ready', got {msg[0]!r}")
        if send_go:
            link.send((ipc.MSG_OPERATOR, OperatorCmd(t=0.0, action=OperatorAction.GO)))
        seq = 0
        while not world.done:
            if world.step_index % steps_per_frame == 0:
                frame = world.make_sensor_frame(world.t, seq)
                link.send((ipc.MSG_FRAME, frame))
                msg = link.recv(recv_timeout_s)
                if msg[0] == ipc.MSG_ERROR:
                    raise RuntimeError(f"autonomy error: {msg[1]}")
                if msg[0] != ipc.MSG_CMD:
                    raise RuntimeError(f"expected 'cmd', got {msg[0]!r}")
                cmd: WheelCmd = msg[1]
                n_telemetry += msg[2] is not None
                apply_step = world.queue_command(cmd)
                for k, v in (("t", world.t), ("seq", cmd.seq), ("omega_l", cmd.omega_l_rad_s), ("omega_r", cmd.omega_r_rad_s),
                             ("compute_ms", cmd.compute_ms), ("apply_t", apply_step * world.dt),
                             ("mode", getattr(cmd.mode, "value", str(cmd.mode)))):
                    cmds[k].append(v)
                seq += 1
            world.step()
            if max_wall_s is not None and time.perf_counter() - wall0 > max_wall_s:
                world.referee.force_end(world.t, "wall_timeout")
    except (RuntimeError, TimeoutError, EOFError, BrokenPipeError, OSError) as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        log.error("episode aborted: %s", error_text)
        world.referee.force_end(world.t, "autonomy_error")
    finally:
        link.close()
        if renderer is not None:
            renderer.close()
    result = _finalize(world, scn, cfg, run_dir, cmds, sensor_mode, camera_hz, noise_seed, wall0, error_text, n_telemetry)
    return result


def _finalize(world: World, scn: dict, cfg: dict, run_dir: Path, cmds: dict, sensor_mode: str, camera_hz: float,
              noise_seed: int, wall0: float, error_text: str | None, n_telemetry: int) -> dict:
    ref = world.referee
    gt = world.gt_log()
    np.savez_compressed(run_dir / "gt" / "states.npz", **gt)
    np.savez_compressed(run_dir / "gt" / "cmds.npz", **{k: np.asarray(v) for k, v in cmds.items()})
    (run_dir / "gt" / "events.json").write_text(json.dumps([e.to_dict() for e in ref.events], indent=1), encoding="utf-8")
    opt = plan_path(world.hazards.grid, world.hazards.hazard, tuple(world.start_xy), tuple(world.goal_xy), CORRIDOR_WIDTH_M)
    L = world.path_length_m
    spl = (opt.length_m / max(L, opt.length_m)) if (ref.success and opt.ok) else 0.0
    t_end = ref.t_end if ref.t_end is not None else world.t
    s = world.state
    n_false, stops = count_false_stops(gt["t"], gt["x"], gt["y"], gt["yaw"], gt["v"], world.hazards, tuple(world.goal_xy),
                                       float(scn["mission"]["success_radius_m"]), gt.get("dyn_xy"),
                                       scn.get("lighting", {}).get("events", []))
    comp = np.asarray(cmds["compute_ms"], float)
    result = {
        "success": bool(ref.success),
        "failure_type": ref.failure_type,
        "time": round(float(t_end), 3),
        "path_length": round(L, 3),
        "spl": round(float(spl), 4),
        "optimal_path_length": _json_num(opt.length_m),
        "min_clearance": _json_num(ref.min_clearance),
        "final_error": round(float(math.hypot(s.x - world.goal_xy[0], s.y - world.goal_xy[1])), 3),
        "ditch_entries": ref.ditch_entries,
        "collisions": ref.collisions,
        "water_entries": ref.water_entries,
        "mean_speed": round(L / t_end, 4) if t_end > 0 else 0.0,
        "false_stops": n_false,
        "stops": stops,
        "config": cfg,
        "seed": int(scn["seed"]),
        "family": scn["family"],
        "split": scn["split"],
        "sha256": scn["sha256"],
        "sensor_mode": sensor_mode,
        "camera_hz": camera_hz,
        "noise_seed": noise_seed,
        "n_frames": len(cmds["t"]),
        "n_telemetry": n_telemetry,
        "n_cmd_timeouts": int(world.n_cmd_timeouts),
        "compute_ms_mean": _json_num(float(comp.mean())) if len(comp) else None,
        "compute_ms_p95": _json_num(float(np.percentile(comp, 95))) if len(comp) else None,
        "sensor_ms_mean": _json_num(float(np.mean(world.sensor_ms))) if world.sensor_ms else None,
        "physics_ms_per_step": _json_num(float(np.mean(world.physics_ms))) if world.physics_ms else None,
        "sensor_breakdown_ms": {k: _json_num(float(np.mean([b[k] for b in world.sensor_breakdown])))
                                for k in ("terrain", "objects", "post", "output", "total")} if world.sensor_breakdown else None,
        "wall_time_s": round(time.perf_counter() - wall0, 3),
        "error": error_text,
        "machine": machine_info(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info("seed %s %s: success=%s failure=%s t=%.1f s L=%.1f m SPL=%.3f wall=%.1f s", result["seed"], result["family"],
             result["success"], result["failure_type"], result["time"], L, spl, result["wall_time_s"])
    return result

"""Autonomy OS-process entry point (see :mod:`metagross.contracts.ipc` for the protocol).

``autonomy_main(conn, run_dir, config)`` is started by the runner in a separate
process. Before touching anything else it

1. checks that no ``metagross.sim*`` / ``metagross.eval*`` / ``metagross.train*`` module is
   resident when it runs as a spawned child (:func:`assert_no_gt_modules`; the runner starts it
   under a neutral ``__main__`` so spawn does not re-import the parent's main module), and
2. installs a ``sys.addaudithook`` guard (layer 4 of the ground-truth firewall):

   * ``open``, ``os.listdir`` and ``os.scandir`` are checked against an allow-list and anything
     else raises ``PermissionError``. Readable (recursively):

     - the Python installation and the venv (``sys.prefix``, ``sys.base_prefix``, ...),
     - the ``metagross/autonomy``, ``metagross/contracts`` and ``metagross/config`` package
       dirs (plus ``metagross/__init__.py`` and its bytecode),
     - the repository ``models`` dir (network weights),
     - ``<run_dir>/autonomy/`` (the autonomy's own logs).

     Listable only (file *names*, needed by the import system's directory cache): each
     ``sys.path`` entry and each ancestor of the package dirs up to the repository root.
   * process creation (``subprocess.Popen``, ``os.system``, ``os.exec*``, ``os.spawn*``,
     ``os.posix_spawn``, ``os.startfile``) and network use (``socket.connect``,
     ``socket.sendto``, ``socket.sendmsg``, ``socket.bind``) are always refused.

   ``os.stat`` / ``os.path.exists`` raise no audit event, so the guard cannot hide the
   *existence* of a path; contents and directory listings are what it protects.

Then it serves the pipe: ``reset`` -> ``ready``; ``frame`` -> ``cmd``; ``operator``;
``close``. The command is sent *before* logs are written so logging never adds
control latency. Logs under ``<run_dir>/autonomy/``:

* ``telemetry.jsonl`` — every telemetry packet, encoded then *decoded* (exactly what
  the operator would receive), plus its size in bytes;
* ``timings.csv`` — per-tick per-module milliseconds;
* ``debug/tick_XXXXXX.npz`` — compressed DebugBundle every ``debug_every_n`` ticks;
* ``autonomy.log``.

Any exception sends ``("error", traceback)`` and returns cleanly.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

LOG = logging.getLogger("metagross.autonomy.process")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "metagross"
_guard_state = threading.local()


def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


_PKG_DIRS = ("autonomy", "contracts", "config")  # the only metagross packages the autonomy may load
FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = ("metagross.sim", "metagross.eval", "metagross.train")
PATH_EVENTS = frozenset({"open", "os.listdir", "os.scandir"})  # audit events whose args[0] is a path
LIST_EVENTS = frozenset({"os.listdir", "os.scandir"})
DENIED_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile",
    "socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind",
})


def allowed_roots(run_dir: Path) -> tuple[list[str], list[str]]:
    """(allowed directory prefixes, allowed individual files), normalised."""
    dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
            *(str(_PKG / p) for p in _PKG_DIRS), str(_REPO / "models"), str(Path(run_dir) / "autonomy")}
    files = [str(_PKG / "__init__.py")]
    pyc = _PKG / "__pycache__"
    files += [str(pyc / f"__init__.{sys.implementation.cache_tag}.pyc"), str(pyc / f"__init__.{sys.implementation.cache_tag}.opt-1.pyc")]
    return sorted({_norm(d) for d in dirs}), [_norm(f) for f in files]


def listable_dirs(sys_path: Iterable[str] | None = None) -> list[str]:
    """Directories whose *names* may be listed (not recursive): ``sys.path`` entries and the
    ancestors of the allowed package dirs up to the repository root (import-system caches)."""
    out = {_norm(p or os.getcwd()) for p in (sys.path if sys_path is None else sys_path)}
    for p in _PKG_DIRS:
        d = _PKG / p
        while True:
            d = d.parent
            out.add(_norm(str(d)))
            if d == _REPO or d == d.parent:
                break
    return sorted(out)


def _is_allowed(path: str, dirs: Iterable[str], files: Iterable[str]) -> bool:
    n = _norm(path)
    if n in files:
        return True
    for d in dirs:
        if n == d or n.startswith(d.rstrip(os.sep) + os.sep):
            return True
    return False


def resident_forbidden_modules(modules: Iterable[str] | None = None) -> list[str]:
    """Names of loaded modules the autonomy must never hold (``metagross.sim*`` / ``eval*`` / ``train*``)."""
    names = sys.modules if modules is None else modules
    return sorted(m for m in names if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_MODULE_PREFIXES))


def assert_no_gt_modules() -> None:
    """Raise ``RuntimeError`` if a ground-truth-side module is resident in this process."""
    bad = resident_forbidden_modules()
    if bad:
        raise RuntimeError(f"GT firewall: ground-truth modules resident in the autonomy process: {bad}")


def install_file_guard(run_dir: Path) -> None:
    """Install the (irremovable) audit hook: path allow-list for ``open`` / ``os.listdir`` /
    ``os.scandir``; process creation and sockets are refused (see module docstring)."""
    dirs, files = allowed_roots(run_dir)
    listable = frozenset(listable_dirs())

    def hook(event: str, args: tuple) -> None:
        if event in DENIED_EVENTS:
            LOG.error("GUARD: blocked %s", event)
            raise PermissionError(f"autonomy guard: {event} is not allowed")
        if event not in PATH_EVENTS or getattr(_guard_state, "busy", False):
            return
        path = args[0] if args else None
        if isinstance(path, int):
            return  # file descriptors (pipes, dup'ed handles)
        if path is None:
            if event == "open":
                return
            path = os.curdir  # os.listdir() / os.scandir() default to the cwd
        _guard_state.busy = True
        try:
            p = os.fsdecode(path) if isinstance(path, (bytes, os.PathLike)) else str(path)
            ok = _is_allowed(p, dirs, files) or (event in LIST_EVENTS and _norm(p) in listable)
            if not ok:
                LOG.error("GUARD: blocked %s(%r)", event, p)
                raise PermissionError(f"autonomy guard: {event} of {p!r} is not allowed")
        finally:
            _guard_state.busy = False

    sys.addaudithook(hook)
    LOG.info("guard installed; allowed dirs=%s; listable=%d dirs; denied events=%s", dirs, len(listable), sorted(DENIED_EVENTS))


def _setup_logging(out: Path) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(out / "autonomy.log") for h in root.handlers):
        fh = logging.FileHandler(out / "autonomy.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    root.setLevel(logging.INFO)


def _bundle_arrays(dbg: Any) -> dict[str, np.ndarray]:
    """Flatten a DebugBundle into npz-friendly arrays (dicts go in as JSON strings)."""
    out: dict[str, np.ndarray] = {"t": np.float64(dbg.t), "pose_xy_yaw": np.asarray(dbg.pose_xy_yaw, np.float64),
                                  "v_cap_mps": np.float64(dbg.v_cap_mps), "r_cert_m": np.float64(dbg.r_cert_m)}
    for name in ("cell_state_local", "cost_local", "semantic_mask", "missing_ground_mask", "disparity", "rollouts_xy", "plan_xy", "global_path_xy"):
        val = getattr(dbg, name)
        if val is not None:
            out[name] = np.asarray(val)
    extras_json: dict[str, Any] = {}
    for k, v in dbg.extras.items():
        if isinstance(v, np.ndarray):
            out[f"extra_{k}"] = v
        else:
            extras_json[k] = v
    out["health_json"] = np.array(json.dumps(dbg.health, default=float))
    out["timings_json"] = np.array(json.dumps(dbg.timings_ms, default=float))
    out["extras_json"] = np.array(json.dumps(extras_json, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return out


class _Logs:
    """Owns the autonomy's log files under ``<run_dir>/autonomy``."""

    def __init__(self, out: Path, debug_every_n: int) -> None:
        from metagross.autonomy.node import TIMING_KEYS

        self.out = out
        self.debug_every_n = int(debug_every_n or 0)
        (out / "debug").mkdir(parents=True, exist_ok=True)
        self.tel = open(out / "telemetry.jsonl", "a", encoding="utf-8")
        new = not (out / "timings.csv").exists()
        self.tim_f = open(out / "timings.csv", "a", newline="", encoding="utf-8")
        self.cols = ["tick", "t", "compute_ms", *TIMING_KEYS, "telemetry", "total"]
        self.tim = csv.writer(self.tim_f)
        if new:
            self.tim.writerow(self.cols)
        self.tick = 0

    def write(self, cmd: Any, tel: Any, dbg: Any) -> None:
        from metagross.autonomy.link import codec

        if tel is not None:
            pkt, info = codec.encode_with_info(tel)
            self.tel.write(json.dumps(codec.to_jsonable(codec.decode(pkt), len(pkt), info)) + "\n")
        tm = dbg.timings_ms
        self.tim.writerow([self.tick, f"{cmd.t:.3f}", f"{cmd.compute_ms:.3f}"] + [f"{tm.get(k, 0.0):.3f}" for k in self.cols[3:]])
        if self.debug_every_n > 0 and self.tick % self.debug_every_n == 0:
            np.savez_compressed(self.out / "debug" / f"tick_{self.tick:06d}.npz", **_bundle_arrays(dbg))
        self.tick += 1

    def close(self) -> None:
        for f in (self.tel, self.tim_f):
            try:
                f.close()
            except OSError:
                pass


def autonomy_main(conn: Any, run_dir: str | os.PathLike, config: Optional[dict] = None) -> None:
    """Process entry point. ``conn`` is a multiprocessing Connection (see module docstring)."""
    from metagross.contracts import ipc

    run_dir = Path(run_dir)
    out = run_dir / "autonomy"
    logs: Optional[_Logs] = None
    try:
        out.mkdir(parents=True, exist_ok=True)
        _setup_logging(out)
        if mp.parent_process() is not None:  # spawned child: nothing from the GT side may be resident
            assert_no_gt_modules()
        install_file_guard(run_dir)
        from metagross.autonomy.node import AutonomyStack

        stack: Optional[AutonomyStack] = None
        base_cfg = dict(config or {})
        while True:
            msg = conn.recv()
            kind = msg[0]
            if kind == ipc.MSG_RESET:
                _, mission, calib, vehicle, cfg = msg
                merged = {**ipc.DEFAULT_AUTONOMY_CONFIG, **base_cfg, **(cfg or {})}
                stack = AutonomyStack()
                stack.reset(mission, calib, vehicle, merged)
                if logs is not None:
                    logs.close()
                logs = _Logs(out, merged.get("debug_every_n", 1))
                conn.send((ipc.MSG_READY,))
            elif kind == ipc.MSG_FRAME:
                if stack is None or logs is None:
                    raise RuntimeError("frame received before reset")
                cmd, tel, dbg = stack.step(msg[1])
                conn.send((ipc.MSG_CMD, cmd, tel))
                logs.write(cmd, tel, dbg)
            elif kind == ipc.MSG_OPERATOR:
                if stack is not None:
                    stack.operator(msg[1])
            elif kind == ipc.MSG_CLOSE:
                LOG.info("close received")
                break
            else:
                raise ValueError(f"unknown message {kind!r}")
    except (EOFError, BrokenPipeError):
        LOG.warning("runner pipe closed")
    except Exception:  # noqa: BLE001 - report every failure to the runner, then exit cleanly
        tb = traceback.format_exc()
        LOG.error("autonomy failed:\n%s", tb)
        try:
            conn.send((ipc.MSG_ERROR, tb))
        except (OSError, EOFError, BrokenPipeError):
            pass
    finally:
        if logs is not None:
            logs.close()
        logging.shutdown()

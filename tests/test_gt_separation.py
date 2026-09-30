"""The onboard stack must never see simulator ground truth.

Layer 1 of the ground-truth firewall: static import scan of metagross/autonomy. It resolves
``import x``, ``from x import y``, *relative* imports (``from ..sim import x``) and string
literals passed to ``importlib.import_module`` / ``__import__``; dynamic (non-literal) import
calls are only allowed in reviewed files. (Layers 2-6: separate OS process with a neutral spawn
main and a resident-module check, slot whitelist + opaque mission id, runtime audit-hook guard,
seed pre-registration, post-hoc evaluation; see docs/ARCHITECTURE.md.)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUTONOMY = REPO / "metagross" / "autonomy"
FORBIDDEN = ("metagross.sim", "metagross.eval", "metagross.train")
IMPORT_FUNCS = {"import_module", "__import__"}
# Files allowed to call importlib with a non-literal module name (reviewed; the runtime guard and the
# resident-module check in autonomy/process.py cover what such a call could load).
DYNAMIC_IMPORT_ALLOWLIST = {"node.py"}


def _package_of(path: Path) -> str:
    """Dotted package of a module file, e.g. metagross/autonomy/planning/mppi.py -> metagross.autonomy.planning."""
    rel = path.relative_to(REPO).with_suffix("")
    parts = list(rel.parts)
    return ".".join(parts[:-1])


def resolve_relative(package: str, module: str | None, level: int) -> str:
    """PEP 328 resolution of ``from <'.' * level><module> import ...`` inside ``package``."""
    if level == 0:
        return module or ""
    base = package.split(".")
    if level - 1 > len(base):
        raise ValueError(f"relative import beyond top-level package: level {level} in {package}")
    base = base[: len(base) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def scan_imports(source: str, package: str) -> tuple[list[str], list[int]]:
    """(resolved imported module names, line numbers of dynamic import calls) of one module's source."""
    tree = ast.parse(source)
    names: list[str] = []
    dynamic: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = resolve_relative(package, node.module, node.level)
            names.append(mod)
            names += [f"{mod}.{a.name}" for a in node.names if a.name != "*"]  # from metagross import sim
        elif isinstance(node, ast.Call) and _call_name(node) in IMPORT_FUNCS:
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                target = arg.value
                if target.startswith("."):
                    pkg_kw = next((k.value for k in node.keywords if k.arg == "package"), None)
                    pkg = pkg_kw.value if isinstance(pkg_kw, ast.Constant) and isinstance(pkg_kw.value, str) else package
                    level = len(target) - len(target.lstrip("."))
                    target = resolve_relative(pkg, target.lstrip(".") or None, level)
                names.append(target)
            else:
                dynamic.append(node.lineno)
    return names, dynamic


def _forbidden(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def test_autonomy_never_imports_ground_truth_modules():
    offenders = []
    for py in AUTONOMY.rglob("*.py"):
        names, _ = scan_imports(py.read_text(encoding="utf-8"), _package_of(py))
        offenders += [f"{py.relative_to(AUTONOMY)} imports {n}" for n in names if _forbidden(n)]
    assert not offenders, "GT firewall breached:\n" + "\n".join(offenders)


def test_dynamic_imports_only_in_reviewed_files():
    found = []
    for py in AUTONOMY.rglob("*.py"):
        _, dyn = scan_imports(py.read_text(encoding="utf-8"), _package_of(py))
        if dyn and py.name not in DYNAMIC_IMPORT_ALLOWLIST:
            found.append(f"{py.relative_to(AUTONOMY)}:{dyn}")
    assert not found, "unreviewed dynamic import calls in metagross/autonomy:\n" + "\n".join(found)


@pytest.mark.parametrize("package,src", [
    ("metagross.autonomy", "from ..sim import world"),
    ("metagross.autonomy", "from .. import sim"),
    ("metagross.autonomy", "from ..eval.claims import x"),
    ("metagross.autonomy.planning", "from ...sim.world import World"),
    ("metagross.autonomy.planning", "import importlib\nimportlib.import_module('metagross.sim.world')"),
    ("metagross.autonomy.planning", "from importlib import import_module\nimport_module('..sim', package='metagross.autonomy')"),
    ("metagross.autonomy.planning", "__import__('metagross.eval')"),
    ("metagross.autonomy.planning", "import metagross.train.export_onnx"),
])
def test_scanner_catches_indirect_forms(package, src):
    names, _ = scan_imports(src, package)
    assert any(_forbidden(n) for n in names), names


def test_scanner_resolves_benign_relative_imports_and_flags_dynamic_calls():
    names, dyn = scan_imports("from . import mppi\nfrom ..contracts import messages\nimport importlib\nimportlib.import_module(name)",
                              "metagross.autonomy.planning")
    assert "metagross.autonomy.planning" in names and "metagross.autonomy.contracts" in names
    assert not any(_forbidden(n) for n in names) and dyn == [4]


def test_sensor_frame_whitelist():
    from metagross.contracts.messages import SENSOR_FRAME_FIELDS

    allowed = {"t", "seq", "left_rgb", "right_gray", "wheel_angle_l_rad", "wheel_angle_r_rad",
               "gyro_z_rps", "sensor_mode", "disparity"}
    assert SENSOR_FRAME_FIELDS == allowed, "SensorFrame gained a field; review for GT leakage"

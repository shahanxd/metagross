"""Red-team probe: which metagross.sim / metagross.eval modules are already resident in the
autonomy OS process when its entry point starts (before the file guard is installed)?

Used as ``--autonomy-target probe_target:probe_main``; it records sys.modules, then hands over
to the real ``metagross.autonomy.process.autonomy_main`` unchanged.
"""
import json
import os
import sys
from pathlib import Path


def probe_main(conn, run_dir, config=None):
    resident = sorted(m for m in sys.modules if m.startswith(("metagross.sim", "metagross.eval", "metagross.train")))
    out = Path(os.environ["REDTEAM_PROBE_OUT"])
    out.write_text(json.dumps({"main": getattr(sys.modules.get("__main__"), "__spec__", None) and sys.modules["__main__"].__spec__.name,
                               "resident": resident}, indent=1), encoding="utf-8")
    from metagross.autonomy.process import autonomy_main
    if config is None:
        return autonomy_main(conn, run_dir)
    return autonomy_main(conn, run_dir, config)

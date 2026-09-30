"""Generate the pre-registered scenario set.

Writes ``<out>/{split}/{seed}.json`` and ``<out>/manifest.json`` (seed, family, split, sha256, path).
Existing files are kept (and verified) unless ``--force``.

    python scripts/gen_scenarios.py --split all --workers 2
    python scripts/gen_scenarios.py --split dev --seeds 100 101
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metagross.sim.scenario import load_scenario, make_scenario, scenario_seeds, split_of, write_scenario  # noqa: E402

log = logging.getLogger("gen_scenarios")


def _one(seed: int, out: str, force: bool) -> dict:
    if not logging.getLogger().handlers:  # spawned worker: make generator INFO lines (attempt counts) visible
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    split = split_of(seed)
    path = Path(out) / split / f"{seed}.json"
    t0 = time.perf_counter()
    if path.exists() and not force:
        scn = load_scenario(path, verify=True)
    else:
        scn = make_scenario(seed, split)
        write_scenario(scn, path)
    return {"seed": seed, "family": scn["family"], "split": split, "sha256": scn["sha256"],
            "path": path.relative_to(Path(out)).as_posix(), "seconds": round(time.perf_counter() - t0, 2)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("dev", "eval", "all"), default="all")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "scenarios")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.seeds:
        seeds = list(a.seeds)
    elif a.split == "all":
        seeds = list(scenario_seeds("dev")) + list(scenario_seeds("eval"))
    else:
        seeds = list(scenario_seeds(a.split))
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=get_context("spawn")) as ex:
        entries = list(ex.map(_one, seeds, [str(a.out)] * len(seeds), [a.force] * len(seeds)))
    man_path = a.out / "manifest.json"
    manifest = {}
    if man_path.exists():
        manifest = {e["seed"]: e for e in json.loads(man_path.read_text(encoding="utf-8"))["scenarios"]}
    for e in entries:
        e = dict(e)
        e.pop("seconds")
        manifest[e["seed"]] = e
        log.info("seed %3d %-20s %s", e["seed"], e["family"], e["sha256"][:12])
    man = {"schema": "metagross.scenario/1", "scenarios": [manifest[k] for k in sorted(manifest)]}
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    log.info("wrote %d scenarios, manifest %s (mean %.1f s/scenario)", len(entries), man_path,
             sum(e["seconds"] for e in entries) / max(1, len(entries)))


if __name__ == "__main__":
    main()

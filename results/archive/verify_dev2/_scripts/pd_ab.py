"""Verifier: perception before/after on identical cached DEV stereo frames (sim/eval side).

Runs metagross.eval.perception_dev.evaluate() in stereo mode on the cached rendered sequences in
results/raw/perception/stereo_<seed>_<kind>.npz for two Perception implementations:
  * 'r2'  = working tree (metagross.autonomy.perception.pipeline)
  * 'r1'  = frozen copy of commit fee1da9's perception package (perc_fee1da9.pipeline, imports rewritten;
            directory passed as argv[1] and put on sys.path)
Writes results/verify_dev2/perception_ab.json. DEV seeds only. Metrics as defined in perception_dev.
Usage: python pd_ab.py <dir containing perc_fee1da9>
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, r"D:\Downloads\sih again\metagross")

from metagross.eval import perception_dev as pd  # noqa: E402

OUT = Path(r"D:\Downloads\sih again\metagross\results\verify_dev2\perception_ab.json")
PLAN = {"stereo": {"drive": (102, 106, 120, 125, 101), "approach": (103, 121, 122)}}
MODULES = {"r1_fee1da9": "perc_fee1da9.pipeline", "r2_tree": "metagross.autonomy.perception.pipeline"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out: dict = {"what": "perception_dev stereo metrics on cached DEV frames, fee1da9 perception vs working tree, run back to back",
                 "plan": PLAN, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name, mod in MODULES.items():
        pd.PERCEPTION_MODULE = mod
        sec = pd.evaluate(["stereo"], semantics=True, plans=PLAN)
        out[name] = sec
        pl = sec["stereo"]["pooled"]
        logging.info("%s pooled: cand_off %.4f lethal_off %.4f pos_off %.4f cert_env %s timing %s", name, pl["cand_rate_off"],
                     pl["lethal_rate_off"], pl.get("positive_rate_off", float("nan")), pl["certified_hazard_in_envelope"], pl["timing_ms"])
        for r in sec["stereo"]["sequences"]:
            logging.info("  %s %s %s: cand_off %.4f lethal_off %.4f det %s cert_env %s", name, r["seed"], r["kind"], r["cand_rate_off"],
                         r["lethal_rate_off"], r.get("detection"), r.get("certified_hazard_in_envelope"))
    OUT.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

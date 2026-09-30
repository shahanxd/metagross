"""Verifier: which round-2 perception change certifies the hidden F3 trench on DEV 122? (cached stereo frames)

Runs perception_dev.evaluate on the cached stereo_122_approach sequence with the working-tree Perception and
one round-2 change switched off at a time (module constant / DEFAULT_PERCEPTION_CONFIG key, patched in this
process only; no file is edited). Reports certified_hazard_in_envelope, ditch_path_certified and detection.
Writes results/verify_dev2/perception_122_ablation.json. DEV seed only.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, r"D:\Downloads\sih again\metagross")

from metagross.autonomy.perception import negobs, pipeline  # noqa: E402
from metagross.eval import perception_dev as pd  # noqa: E402

OUT = Path(r"D:\Downloads\sih again\metagross\results\verify_dev2\perception_122_ablation.json")
PLAN = {"stereo": {"approach": (122,)}}
BASE_CFG = dict(pipeline.DEFAULT_PERCEPTION_CONFIG)
BASE_KTAU = negobs.K_TAU_SIGMA
VARIANTS = {
    "tree_default": {},
    "no_k_tau_sigma_relax": {"K_TAU_SIGMA": 0.0},
    "no_min_obstacle_size": {"min_obstacle_size": False},
    "no_noise_aware_step": {"noise_aware_step": False},
    "no_positive_persistence": {"positive_persistence": False},
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out: dict = {"what": "stereo DEV 122 approach (cached frames), tree perception with one round-2 change disabled per variant"}
    for name, var in VARIANTS.items():
        pipeline.DEFAULT_PERCEPTION_CONFIG.clear()
        pipeline.DEFAULT_PERCEPTION_CONFIG.update(BASE_CFG)
        negobs.K_TAU_SIGMA = BASE_KTAU
        for k, v in var.items():
            if k == "K_TAU_SIGMA":
                negobs.K_TAU_SIGMA = v
            else:
                pipeline.DEFAULT_PERCEPTION_CONFIG[k] = v
        r = pd.evaluate(["stereo"], semantics=True, plans=PLAN)["stereo"]["sequences"][0]
        out[name] = {k: r.get(k) for k in ("certified_hazard_in_envelope", "certified_hazard_in_envelope_ditch_cells",
                                            "ditch_path_certified", "detection", "cand_rate_off", "lethal_rate_off")}
        logging.info("%s: cert_env %s ditch_path_cert %s det %s", name, r.get("certified_hazard_in_envelope"),
                     r.get("ditch_path_certified"), {k: (r.get("detection") or {}).get(k) for k in ("r_first_m", "r_stable_m")})
    OUT.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

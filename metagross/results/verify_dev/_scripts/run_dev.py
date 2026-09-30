"""DEV closed-loop batch (tier0 or stereo) through the real runner + autonomy process."""
import os, sys, json, logging
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))


def main():
    from metagross.sim.batch import Job, run_batch
    from metagross.contracts import ipc
    mode = sys.argv[1]
    out = Path(sys.argv[2])
    workers = int(sys.argv[3])
    max_sim = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "none" else None
    full_seeds = [int(s) for s in sys.argv[5].split(",")]
    typ_seeds = [int(s) for s in sys.argv[6].split(",")] if len(sys.argv) > 6 and sys.argv[6] else []
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    full = {**ipc.DEFAULT_AUTONOMY_CONFIG, "name": "FULL"}
    typ = {**ipc.DEFAULT_AUTONOMY_CONFIG, "name": "TYPICAL", "unknown_is_free": True, "use_negobs": False,
           "use_governor": False, "fixed_speed_mps": 1.5}
    jobs = [Job(s, dict(full)) for s in full_seeds] + [Job(s, dict(typ)) for s in typ_seeds]
    csv = run_batch(jobs, REPO / "data" / "scenarios", out, workers, sensor_mode=mode, max_sim_s=max_sim)
    print("CSV", csv)


if __name__ == "__main__":
    main()

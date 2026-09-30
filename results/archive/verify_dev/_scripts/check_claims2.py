"""Red-team: resolve the sampled claims that have no dotted source path."""
import json
from pathlib import Path

R = Path(r"D:\Downloads\sih again\metagross\results")
seg = json.loads((R / "seg_cpu.json").read_text(encoding="utf-8"))
print("seg_cpu top keys:", list(seg)[:30])
print("training:", json.dumps(seg.get("training", {}))[:600])
comp = seg.get("comparison", {})
print("comparison keys:", list(comp)[:20])
for split in ("val", "test"):
    blk = seg.get(split) or seg.get("splits", {}).get(split)
    if isinstance(blk, dict):
        print(split, {k: blk[k] for k in list(blk)[:25] if not isinstance(blk[k], (dict, list))})
        for k in ("per_class_confusion", "false_safe", "hazard_to_stable", "water_to_stable_rate", "obstacle_to_stable_rate"):
            if k in blk:
                print("  ", k, json.dumps(blk[k])[:300])
print("comparison:", json.dumps(comp)[:1500])
zs = json.loads((R / "seg_zeroshot.json").read_text(encoding="utf-8"))
print("zeroshot keys:", list(zs)[:20])
for split in ("val", "test"):
    blk = zs.get(split) or zs.get("splits", {}).get(split)
    if isinstance(blk, dict):
        print(split, "miou", blk.get("miou"), blk.get("n_images"))
pd = json.loads((R / "perception_dev.json").read_text(encoding="utf-8"))
for sec in ("before", "after"):
    seqs = pd[sec]["tier0"]["sequences"]
    items = seqs.items() if isinstance(seqs, dict) else enumerate(seqs)
    for k, s in items:
        det = s.get("detection") if isinstance(s, dict) else None
        if det:
            print(sec, k, s.get("seed"), s.get("kind", ""), {x: det.get(x) for x in ("r_stable_m", "r_first_m", "theory_r_det_m", "width_m")})
cl = json.loads((R / "closed_loop_dev.json").read_text(encoding="utf-8"))
a = cl["stereo"]["aggregate"]["FULL"]["all"]
print("stereo FULL all:", {k: a[k] for k in a if "compute" in k or "success" in k})

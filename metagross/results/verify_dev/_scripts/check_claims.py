"""Red-team: sample 10 claims (fixed seed) and resolve each against its source file."""
import csv
import json
import random
from pathlib import Path

REPO = Path(r"D:\Downloads\sih again\metagross")
rows = list(csv.DictReader(open(REPO / "results" / "claims.csv", encoding="utf-8")))
rng = random.Random(20260930)
sample = rng.sample(rows, 10)


def dig(obj, dotted):
    cur = obj
    for k in dotted.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
            cur = cur[int(k)]
        else:
            return "<MISSING>"
    return cur


for r in sample:
    src = r["source"]
    f, _, path = src.partition("#")
    p = REPO / f
    out = "<no file>"
    if p.exists() and p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        out = dig(d, path) if path else "(no path)"
        if isinstance(out, (dict, list)):
            out = json.dumps(out)[:300]
    print(f"== {r['id']} | {r['value']} {r['unit']} | {r['label']}\n   src={src}\n   resolved={out}\n   note={r['note'][:200]}")

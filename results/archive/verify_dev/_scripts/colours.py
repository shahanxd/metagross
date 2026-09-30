import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))
from metagross.eval import plot_style as ps  # noqa: E402

allowed = {c.upper() for d in (ps.TOKENS, ps.HONESTY_COLORS, ps.MODE_COLORS) for c in d.values()}
allowed |= {c.upper() for c in ps.CELL_HEX.values()} | {"#DBE6FE"}
for f in ("s2_how_it_works", "s3_missing_ground", "s3_models"):
    t = (REPO / "deck_assets" / "slots" / f"{f}.svg").read_text(encoding="utf-8")
    ET.fromstring(t)
    stray = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}\b", t)} - allowed
    print(f, "stray colours:", stray or "none")

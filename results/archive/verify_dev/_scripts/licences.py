"""Print name, version and licence metadata of every installed distribution."""
import importlib.metadata as md

rows = []
for d in md.distributions():
    m = d.metadata
    lic = (m.get("License-Expression") or m.get("License") or "").strip().replace("\n", " ")
    if len(lic) > 70:
        lic = lic[:70] + "..."
    cls = [c.split("::")[-1].strip() for c in (m.get_all("Classifier") or []) if c.startswith("License")]
    rows.append((m["Name"], d.version, lic, "; ".join(cls)))
for r in sorted(rows, key=lambda r: r[0].lower()):
    print(" | ".join(r))

p = r"D:\Downloads\sih again\metagross\metagross\eval\perception_dev.py"
s = open(p, encoding="utf-8").read()
rep = [
("""    haz_any: int  # GT lethal cells in the grid (observed or not)
    haz_any_cert: int
""", """    haz_any: int  # GT lethal cells in the grid (observed or not)
    haz_any_cert: int
    haz_env_cert_ditch: int  # certified envelope hazard cells that are GT ditch cells
    ditch_path_cert: int  # GT path cells of the tracked ditch certified as ground (any range)
"""),
("""                      int(haz.sum()), int((haz & cert).sum()), int(ditch_path.sum()), int((ditch_path & flagged).sum()), lip,
""", """                      int(haz.sum()), int((haz & cert).sum()), int((env & haz & cert & (gt_cells["ditch_id"] >= 0)).sum()),
                      int((ditch_path & cert).sum()), int(ditch_path.sum()), int((ditch_path & flagged).sum()), lip,
"""),
("""        "certified_hazard_any": {"cells": sum(f.haz_any_cert for f in scores), "of": sum(f.haz_any for f in scores)},
""", """        "certified_hazard_in_envelope_ditch_cells": sum(f.haz_env_cert_ditch for f in scores),
        "certified_hazard_any": {"cells": sum(f.haz_any_cert for f in scores), "of": sum(f.haz_any for f in scores)},
        "ditch_path_certified": {"cells": sum(f.ditch_path_cert for f in scores), "of": sum(f.ditch_path for f in scores)},
"""),
]
for a, b in rep:
    assert a in s, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")

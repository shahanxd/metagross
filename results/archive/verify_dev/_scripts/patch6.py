"""Revert the per-column noise estimate (it counts the ditch's own smeared lip as noise:
synthetic SGBM trench 12 -> 0 cells) back to the per-frame estimate."""
import re

p = r"D:\Downloads\sih again\metagross\metagross\autonomy\perception\negobs.py"
s = open(p, encoding="utf-8").read()
reps = [
("""        raw_res = np.where(meas, Dm - DG, np.nan)
        sig_frame = band_noise_px(raw_res, lab)
        sig_col = column_noise_px(raw_res, lab, sig_frame)  # (C,) local noise: textured / low-texture bands differ
""", """        sig_d = band_noise_px(np.where(meas, Dm - DG, np.nan), lab)
"""),
("""            sig_d = sig_col[c]
            sig_h = MEDIAN_EFF""", """            sig_h = MEDIAN_EFF"""),
(""""sig_d": sig_d, "sig_frame": sig_frame}""", """"sig_d": sig_d}"""),
("""NOISE_COL_HALF = 4  # per-column noise pools +-this many neighbouring bands (column_noise_px)
NOISE_MIN_COL_SAMPLES = 30  # ... and falls back to the frame estimate below this many GROUND pairs
""", ""),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
i0 = s.index("def column_noise_px(")
i1 = s.index("def _lateral_support(")
s = s[:i0] + s[i1:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok", "column_noise_px" in s)

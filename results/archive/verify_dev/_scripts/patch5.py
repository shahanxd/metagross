p = r"D:\Downloads\sih again\metagross\metagross\autonomy\perception\negobs.py"
s = open(p, encoding="utf-8").read()
reps = [
("MIN_DITCH_LATERAL_M = 0.6  #", "MIN_DITCH_LATERAL_M = 0.4  #"),
("""        sig_d = band_noise_px(np.where(meas, Dm - DG, np.nan), lab)
""", """        raw_res = np.where(meas, Dm - DG, np.nan)
        sig_frame = band_noise_px(raw_res, lab)
        sig_col = column_noise_px(raw_res, lab, sig_frame)  # (C,) local noise: textured / low-texture bands differ
"""),
("""            sig_h = MEDIAN_EFF * r_lip * h_cam * sig_d / geom.fxb / np.sqrt(np.maximum(n_pts, 1))
""", """            sig_d = sig_col[c]
            sig_h = MEDIAN_EFF * r_lip * h_cam * sig_d / geom.fxb / np.sqrt(np.maximum(n_pts, 1))
"""),
(""""sig_d": sig_d}""", """"sig_d": sig_d, "sig_frame": sig_frame}"""),
("""def _lateral_support(""", """def column_noise_px(raw_res: np.ndarray, lab: np.ndarray, sig_frame: float, lag: int = NOISE_LAG_ROWS,
                    half_cols: int = NOISE_COL_HALF) -> np.ndarray:
    \"\"\"Per-column-band noise (px), (C,): mean absolute lag-``lag`` residual difference on GROUND
    rows (x 1.2533 / sqrt(2) = Gaussian sigma), pooled over +-``half_cols`` neighbouring bands.
    Never below the frame estimate ``sig_frame`` (a quiet band must not look more certain than
    the frame), clipped to SIGMA_D_MAX_PX; bands with too few samples take ``sig_frame``.\"\"\"
    ok = (lab[lag:] == LAB_GROUND) & (lab[:-lag] == LAB_GROUND)
    dd = np.abs(raw_res[lag:] - raw_res[:-lag])
    ok &= np.isfinite(dd)
    s1 = np.where(ok, dd, 0.0).sum(0)
    n = ok.sum(0).astype(np.float64)
    k = np.ones(2 * half_cols + 1)
    s1 = np.convolve(s1, k, mode="same")
    n = np.convolve(n, k, mode="same")
    sig = np.where(n >= NOISE_MIN_COL_SAMPLES, MEDIAN_EFF * s1 / np.maximum(n, 1.0) / math.sqrt(2.0), sig_frame)
    return np.clip(np.maximum(sig, sig_frame), SIGMA_D_MIN_PX, SIGMA_D_MAX_PX)


def _lateral_support("""),
("""NOISE_MIN_SAMPLES = 200  # fewer GROUND pairs than this -> fallback SIGMA_D_PX
""", """NOISE_MIN_SAMPLES = 200  # fewer GROUND pairs than this -> fallback SIGMA_D_PX
NOISE_COL_HALF = 4  # per-column noise pools +-this many neighbouring bands (column_noise_px)
NOISE_MIN_COL_SAMPLES = 30  # ... and falls back to the frame estimate below this many GROUND pairs
"""),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")

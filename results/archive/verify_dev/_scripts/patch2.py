p = r"D:\Downloads\sih again\metagross\metagross\autonomy\perception\negobs.py"
s = open(p, encoding="utf-8").read()
reps = [
("""                _, xr_all, zr_all, _ = gather(rs, N_REAPPEAR_ROWS)
""", """                dr_all, xr_all, zr_all, _ = gather(rs, N_REAPPEAR_ROWS)
                d_rep = np.nanmedian(np.where(dr_all > 0, dr_all, np.nan), 0)
"""),
("""            wide = width_est >= MIN_DITCH_WIDTH_M
""", """            wide = width_est >= MIN_DITCH_WIDTH_M
            # A trench's far wall stands at its far lip, so the ground that reappears after the
            # gap is not CLOSER than the wall (d_rep <~ d_wall). A stereo mismatch hole on
            # repetitive texture reads farther than the ground that continues right after it.
            sig_rep = sig_d * np.sqrt(1.0 / N_REAPPEAR_ROWS + 1.0 / np.maximum(n_gap, 1))
            rep_px = np.where(np.isfinite(d_rep), d_rep - d_gap, 0.0)
            rep_ok = rep_px <= K_REP_SIGMA * sig_rep
"""),
("""            is_ditch = reap & below & plateau & big_jump & significant & wide
""", """            is_ditch = reap & below & plateau & big_jump & significant & wide & (rep_ok | (not USE_REAPPEAR_TEST))
"""),
(""""step_px": step_px, "sig_step": sig_step, "width_est": width_est, "sig_h": sig_h}""",
 """"step_px": step_px, "sig_step": sig_step, "width_est": width_est, "sig_h": sig_h,
                         "rep_z": rep_px / sig_rep}"""),
("""MIN_DITCH_LATERAL_M = 0.4""", """K_REP_SIGMA = 2.0  # reappearing ground may read at most this many sigmas closer than the far wall
USE_REAPPEAR_TEST = True  # switch for the reappearance-vs-wall test (diagnostics)
MIN_DITCH_LATERAL_M = 0.4"""),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")

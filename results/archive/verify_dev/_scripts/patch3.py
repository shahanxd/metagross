p = r"D:\Downloads\sih again\metagross\metagross\autonomy\perception\negobs.py"
s = open(p, encoding="utf-8").read()
# The reappearance-vs-wall test was measured on DEV stereo (results/raw/perception/diag_gaps_stereo.npy):
# it is not discriminative (false / true rep_z medians -0.16 / -0.46) and costs recall -> removed.
reps = [
("""                dr_all, xr_all, zr_all, _ = gather(rs, N_REAPPEAR_ROWS)
                d_rep = np.nanmedian(np.where(dr_all > 0, dr_all, np.nan), 0)
""", """                _, xr_all, zr_all, _ = gather(rs, N_REAPPEAR_ROWS)
"""),
("""            # A trench's far wall stands at its far lip, so the ground that reappears after the
            # gap is not CLOSER than the wall (d_rep <~ d_wall). A stereo mismatch hole on
            # repetitive texture reads farther than the ground that continues right after it.
            sig_rep = sig_d * np.sqrt(1.0 / N_REAPPEAR_ROWS + 1.0 / np.maximum(n_gap, 1))
            rep_px = np.where(np.isfinite(d_rep), d_rep - d_gap, 0.0)
            rep_ok = rep_px <= K_REP_SIGMA * sig_rep
""", ""),
("""            is_ditch = reap & below & plateau & big_jump & significant & wide & (rep_ok | (not USE_REAPPEAR_TEST))
""", """            is_ditch = reap & below & plateau & big_jump & significant & wide
"""),
(""""step_px": step_px, "sig_step": sig_step, "width_est": width_est, "sig_h": sig_h,
                         "rep_z": rep_px / sig_rep}""", """"step_px": step_px, "sig_step": sig_step, "width_est": width_est, "sig_h": sig_h,
                         "sig_d": sig_d}"""),
("""K_REP_SIGMA = 2.0  # reappearing ground may read at most this many sigmas closer than the far wall
USE_REAPPEAR_TEST = True  # switch for the reappearance-vs-wall test (diagnostics)
""", ""),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = r"D:\Downloads\sih again\metagross\metagross\autonomy\perception\pipeline.py"
s = open(p, encoding="utf-8").read()
a = """    "unknown_is_free": False,
    "seed": 0,"""
b = """    "unknown_is_free": False,
    "ditch_persistence": True,  # confirm ditch candidates across frames in the odometry frame (persistence.py)
    "seed": 0,"""
assert a in s
s = s.replace(a, b)
a = """``config['use_negobs'] = False`` disables gap classification (no DITCH / CREST / OCCLUDED:
missing ground simply stays UNSEEN). ``config['use_semantics'] = False`` skips the segmenter.
"""
b = """``config['use_negobs'] = False`` disables gap classification (no DITCH / CREST / OCCLUDED:
missing ground simply stays UNSEEN). ``config['use_semantics'] = False`` skips the segmenter.
``config['ditch_persistence']`` (default True): a DITCH_CANDIDATE must be supported by a raw
candidate of one of the last 2 frames within 0.25 m in the odometry frame (``pose_xy_yaw``, the
localiser pose); unconfirmed candidates are output as UNSEEN (:mod:`~.persistence`). The first
frame of a Perception instance is not filtered.
"""
assert a in s
s = s.replace(a, b)
a = '''        """One perception tick. ``pose_xy_yaw`` is accepted for the interface; all outputs are
        egocentric (body frame), so it is only echoed back."""'''
b = '''        """One perception tick. All outputs are egocentric (body frame, see the module docstring,
        including ``certified_local`` / ``r_det_m``). ``pose_xy_yaw`` (odometry frame: m, m, rad)
        is used only to confirm ditch candidates across frames (``ditch_persistence``)."""'''
assert a in s
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")

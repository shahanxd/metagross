import sys, json
sys.path.insert(0, r"D:\Downloads\sih again\metagross"); sys.path.insert(0, r"D:\Downloads\sih again\metagross\tests")
import numpy as np, cv2
cv2.setNumThreads(2)
from test_node_closed_loop import ToyWorld, OracleBevPerception, PerfectOdomLocalizer
from metagross.autonomy.node import AutonomyStack, TIMING_KEYS
from metagross.autonomy.link import codec
from metagross.config.defaults import VEHICLE, stereo_calibration
from metagross.contracts.messages import MissionSpec, SensorFrame, CellState
world = ToyWorld(); world.add_disc(7.0, 0.0, 0.8); world.add_disc(11.0, -1.5, 0.5); world.add_band_x(16.0, 16.6, CellState.DITCH_CANDIDATE, gap_y=(-4.0, -1.5))
st = AutonomyStack(perception=OracleBevPerception(world), localizer=PerfectOdomLocalizer(world))
st.reset(MissionSpec("m", (22.0, 0.0), 2.0, 60), stereo_calibration(), VEHICLE, {})
T, sizes = [], []
for k in range(300):
    s = world.s
    cmd, tel, dbg = st.step(SensorFrame(t=0.2*k, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=s.wl, wheel_angle_r_rad=s.wr, gyro_z_rps=s.w))
    world.step(cmd.omega_l_rad_s, cmd.omega_r_rad_s, 0.2)
    T.append(dbg.timings_ms)
    if tel is not None:
        pkt = codec.encode(tel); sizes.append(len(pkt)); assert codec.decode(pkt).seq == tel.seq
    if cmd.mode.value in ("ARRIVED", "SAFE_STOP"): break
print("ticks", k + 1, "final", cmd.mode.value, "pos", round(world.s.x, 2), round(world.s.y, 2), "t", round(0.2*k, 1))
out = {}
for kk in list(TIMING_KEYS) + ["telemetry", "compute", "total"]:
    a = np.array([t.get(kk, 0.0) for t in T[2:]])
    out[kk] = (round(float(np.median(a)), 2), round(float(np.percentile(a, 95)), 2))
print(json.dumps(out))
print("packets", len(sizes), "bytes median", int(np.median(sizes)), "min", min(sizes), "max", max(sizes))

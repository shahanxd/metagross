import sys, zlib, lzma, bz2
sys.path.insert(0, r"D:\Downloads\sih again\metagross"); sys.path.insert(0, r"D:\Downloads\sih again\metagross\tests")
import numpy as np, cv2
cv2.setNumThreads(2)
from test_node_closed_loop import ToyWorld, OracleBevPerception, PerfectOdomLocalizer
from metagross.autonomy.node import AutonomyStack
from metagross.autonomy.link import codec
from metagross.config.defaults import VEHICLE, stereo_calibration
from metagross.contracts.messages import MissionSpec, SensorFrame, CellState
world = ToyWorld(); world.add_disc(7.0, 0.0, 0.8); world.add_disc(11.0, -1.5, 0.5); world.add_band_x(16.0, 16.6, CellState.DITCH_CANDIDATE, gap_y=(-4.0, -1.5))
st = AutonomyStack(perception=OracleBevPerception(world), localizer=PerfectOdomLocalizer(world))
st.reset(MissionSpec("m", (22.0, 0.0), 2.0, 60), stereo_calibration(), VEHICLE, {"fixed_latency_s": 0.05})
maps = []
for k in range(300):
    s = world.s
    cmd, tel, dbg = st.step(SensorFrame(t=0.2*k, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=s.wl, wheel_angle_r_rad=s.wr, gyro_z_rps=s.w))
    world.step(cmd.omega_l_rad_s, cmd.omega_r_rad_s, 0.2)
    if tel is not None: maps.append(tel.costmap_u4)
    if cmd.mode.value in ("ARRIVED", "SAFE_STOP"): break
def up(a):
    b = a.copy(); b[1:] ^= a[:-1]; return b
enc = {
 "packed_zlib9": lambda a: zlib.compress(codec.pack_u4(a), 9),
 "bytes_zlib9": lambda a: zlib.compress(a.tobytes(), 9),
 "up_bytes_zlib9": lambda a: zlib.compress(up(a).tobytes(), 9),
 "up_packed_zlib9": lambda a: zlib.compress(codec.pack_u4(up(a)), 9),
 "packed_lzma": lambda a: lzma.compress(codec.pack_u4(a), format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}]),
 "packed_bz2": lambda a: bz2.compress(codec.pack_u4(a), 9),
}
for name, f in enc.items():
    s = [len(f(m)) for m in maps]
    print(f"{name:18s} median={int(np.median(s)):4d} max={max(s):4d}")
print("n maps", len(maps), "distinct codes", np.unique(np.concatenate([m.ravel() for m in maps])))

import sys, logging
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
logging.basicConfig(level=logging.WARNING)
import numpy as np, cv2
cv2.setNumThreads(2)
from metagross.autonomy.node import AutonomyStack
from metagross.config.defaults import VEHICLE, stereo_calibration, IMG_H, IMG_W
from metagross.contracts.messages import MissionSpec, SensorFrame
st = AutonomyStack()
st.reset(MissionSpec("i", (10.0, 0.0), 2.0, 60.0), stereo_calibration(), VEHICLE, {})
print("impl", st.impl)
rng = np.random.default_rng(0)
for k in range(5):
    img = rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
    fr = SensorFrame(t=0.2*k, seq=k, left_rgb=img, right_gray=img[..., 0].copy(), wheel_angle_l_rad=0.0, wheel_angle_r_rad=0.0, gyro_z_rps=0.0)
    cmd, tel, dbg = st.step(fr)
    print(k, cmd.mode.value, round(cmd.compute_ms, 1), dbg.extras["reason"], {k2: round(v, 1) for k2, v in dbg.timings_ms.items() if k2 in ("localizer", "perception", "mppi", "compute")})

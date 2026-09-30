import sys
sys.path.insert(0, r"D:\Downloads\sih again\metagross"); sys.path.insert(0, r"D:\Downloads\sih again\metagross\tests")
import numpy as np, cv2
cv2.setNumThreads(2)
from test_node_closed_loop import ToyWorld, run_loop
from metagross.contracts.messages import CellState
world = ToyWorld()
world.add_band_x(6.0, 6.6, CellState.DITCH_CANDIDATE); world.add_band_y(3.0, 3.4); world.add_band_y(-3.4, -3.0); world.add_band_x(-2.4, -2.0, CellState.POSITIVE)
stack, log = run_loop(world, (14.0, 0.0), t_max=60.0)
print("transitions", stack.supervisor.transitions)
xy = np.array(log["xy"]); v = np.array(log["v"])
print("n", len(xy), "front max", (xy[:,0]+0.4).max(), "final v", v[-1])
for k in range(0, len(xy), 10): print(k, np.round(xy[k], 2), round(v[k], 2), log["mode"][k].value, log["reason"][k])

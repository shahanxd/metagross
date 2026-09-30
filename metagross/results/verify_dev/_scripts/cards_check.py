import sys
from pathlib import Path
import cv2
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from video import cards
R = Path(r"D:\Downloads\sih again\metagross")
out = Path(r"C:\Users\Asus\AppData\Local\Temp\claude\D--Downloads-sih-again\4eb046a2-5c84-43b5-91e3-06a29db230da\scratchpad")
a = cards.evidence_card("1.9", "%", "Translation drift of camera-only stereo visual odometry, KITTI odometry (real driving data)", "Tested",
                        R / "deck_assets/kitti_07_traj.png", kicker="What we measured · localisation", source="results/claims_index.json · kitti_pooled_t_err_pct")
b = cards.evidence_card("12 / 30", "runs", "Simulated development runs that reached the goal, METAGROSS full stack", "Simulated", None,
                        kicker="What we measured · closed loop", bullets=["Ditch entries: METAGROSS 0, typical stack 5"], subtitle_text="Example subtitle line.")
c = cards.end_card("What this is not, yet.", ["Every drive in this video is simulated.", "Visual odometry and segmentation are tested on real images; the closed loop is not."], kicker="Honest limits")
for n, im in (("card_a", a), ("card_b", b), ("card_c", c)):
    cv2.imwrite(str(out / f"{n}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

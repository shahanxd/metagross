# METAGROSS demo video: narration and scene list (target 2:30-3:00)

The machine-readable version (the one that is rendered) is `video/scenes/golden_demo.yaml`. Keep the two in
sync: when a line changes here, change it there. How to render: `video/MAKE_VIDEO.md`.

Voice: edge-tts `en-IN-PrabhatNeural` (alternative `en-IN-NeerjaNeural`, set `voice:` in the YAML or pass
`--voice`), rate `+12 %`. Short sentences, plain words, no hype. Each scene lasts exactly as long as its narration
plus 0.8 s (`duration: auto`, `t1: auto`), so the cards follow the voice. Captions are burned into the footer and
shipped as a sidecar `.srt` and a soft subtitle track.

Honesty: every `{placeholder}` is filled by `python -m video.collect_values` from `results/claims_index.json`
(the claims ledger) or from a run directory (labelled Simulated). No number is typed by hand. Every simulated
frame shows the outlined **SIMULATED** chip (top bar) and `SIMULATED · seed N · config X` (footer), and the chase
view is labelled "SIMULATOR VIEW · chase camera, ground-truth replay" (it is not a robot sensor).

Measured length (edge-tts `en-IN-PrabhatNeural`, 2026-09-30; per-scene numbers in `video/out/<name>_plan.json`
or `<name>_render.json`): the first script at rate -4 % ran 3:58; the tighter script at +8 % ran 3:04; this script at
**+12 % runs 2:57.6** (177.6 s, `video/out/draft_v0_render.json`, closed-loop placeholders read as the word
"pending"). Real numbers are about as long as "pending", so the final should land within a few seconds of 2:58.

---

## Scene list

| # | Scene | Type | Source | On screen |
|---|-------|------|--------|-----------|
| 1 | Hook | card · title | — | "Two cameras. No GPS. No map." |
| 2 | Problem | card · title | — | "A ditch looks like nothing at all." |
| 3 | Thesis | card · title | — | "Unknown is never free." + the thesis line |
| 4 | How it works | card · image | `deck_assets/diagrams/figma/01_how_it_works.png` | six-stage pipeline diagram |
| 5 | Demo 1: ditch approach | split | `run_ditch_typical` vs `run_ditch_full` (same seed) | left: TYPICAL keeps speed; right: grey band, magenta then red, R_cert and v_cap drop |
| 6 | Demo 2: glare | dashboard | `run_glare` (F5 lighting) | mode chip NOMINAL -> CAUTION/DEGRADED, q tile, v_cap follows |
| 7 | Operator link | dashboard | `run_nominal` | the OPERATOR LINK panel: 4-bit costmap packet, bytes, rate, link load |
| 8 | Results: localisation | card · evidence | claim `kitti_pooled_t_err_pct` (Tested) | KITTI drift + trajectory |
| 9 | Results: terrain | card · evidence | claim `seg_smoke_test_miou` (Tested) | mIoU + confusion matrix |
| 10 | Results: closed loop | card · evidence | claims `closed_loop_*` (Simulated) | goal reached k / n, ditch entries both stacks |
| 11 | Limits | card · end | — | three plain lines |
| 12 | Close | card · end | — | thesis |

---

## Narration

### 1 · Hook
A small robot, in rough open country. No GPS, no map, no lidar. Just two cameras.
So how does it know where it is safe to drive?

### 2 · Problem
Most navigation stacks treat what they cannot see as free space.
But a ditch is not something sticking up. It is ground that is missing. To a camera, it looks like nothing at all.

### 3 · Thesis
So METAGROSS follows one rule. Unknown is never free.
It drives only on ground it has seen, and only as fast as it can see.

### 4 · How it works
The stereo cameras measure the ground ahead. Every patch gets a state: seen, unseen, or missing,
which is how a ditch shows up. Visual odometry tracks position, and an integrity monitor decides how far
to trust it. A speed governor then picks the fastest speed that can still stop on ground already seen.

### 5 · Demo 1: ditch approach, TYPICAL and FULL side by side
Now a ditch, hidden behind a crest. Same start, two stacks.
On the left, a typical stack that treats unknown as free. On the right, METAGROSS.
Beyond the crest the ground stays grey: not seen, so not trusted.
METAGROSS slows as its certified range shrinks to {split_rcert_min_m} metres.
The missing ground turns magenta, then red. The typical stack keeps going.

*Story check before the final render:* watch the split. If TYPICAL does not enter the ditch, replace the last
sentence with "The typical stack does not slow down." If FULL does not show magenta then red, cut that sentence.
Never narrate something the frames do not show.

### 6 · Demo 2: glare and degraded localisation
Next, glare. The image washes out and visual odometry loses its features.
The integrity score drops to {glare_q_min}, the mode changes to {glare_worst_mode},
and the speed limit comes down with it. When the image clears, the robot speeds up again.

*Story check:* `glare_worst_mode` is computed from the run (worst mode after `t0_glare`); if it is STOP_AND_LOOK
because of a stall rather than glare, move `t0_glare` or re-word.

### 7 · Operator link
The operator never gets video. Twice a second, the robot sends one small packet:
position, health mode, planned path, and a coarse map in the same colours.
About {packet_bytes} bytes. The operator can hold, resume or stop it at any time.

### 8 · Honest results
What have we measured? On real KITTI driving data,
our camera-only visual odometry drifts {kitti_t_err_pct} percent of the distance travelled.

Terrain segmentation scores {seg_miou} mean IoU on held-out off-road images.

In simulation, METAGROSS reached the goal in {success_full} of {n_closed_loop_runs} development runs,
with {ditch_entries_full} ditch entries. The typical stack had {ditch_entries_typical}.

### 9 · Limits
The limits: every drive you saw is simulated, the robot is still slower than we want,
and the full system has not yet run on a real vehicle. That is our next step.

### 10 · Close
Unknown is never free. METAGROSS drives only on ground it has seen, and only as fast as it can see.

---

## Placeholder register

All come from `values:` in `video/scenes/golden_demo.yaml`.

| Placeholder | Meaning | Source | Label |
|---|---|---|---|
| `{kitti_t_err_pct}` | KITTI protocol translation error, pooled 00/05/07 | claim `kitti_pooled_t_err_pct` | Tested |
| `{seg_miou}` | 5-class segmentation mIoU, test split | claim `seg_smoke_test_miou` | Tested |
| `{n_closed_loop_runs}`, `{success_full}` | closed-loop FULL runs and successes (DEV or EVAL, as registered) | claims `closed_loop_n_runs_full`, `closed_loop_success_full` (to be registered by the eval owner) | Simulated |
| `{ditch_entries_full}`, `{ditch_entries_typical}` | referee ditch entries per config | claims `closed_loop_ditch_entries_full`, `closed_loop_ditch_entries_typical` (to be registered) | Simulated |
| `{split_rcert_min_m}` | minimum certified range in the FULL ditch run after `t0_ditch` | `run_ditch_full` debug bundles | Simulated |
| `{glare_q_min}`, `{glare_worst_mode}` | minimum integrity q / most severe mode after `t0_glare` | `run_glare` debug bundles | Simulated |
| `{packet_bytes}` | median telemetry packet size | `run_nominal/autonomy/telemetry.jsonl` | Simulated |

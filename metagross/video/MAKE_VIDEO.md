# Making the METAGROSS demo video

From golden run directories to a narrated 1080p MP4 in under 40 minutes. All commands are PowerShell from the
repo root (`D:\Downloads\sih again\metagross`). The path has a space; keep the quotes.

```powershell
cd "D:\Downloads\sih again\metagross"
$env:OMP_NUM_THREADS = 2
$py = ".venv\Scripts\python.exe"
```

What you get: `video/out/metagross_demo.mp4` (H.264 CRF 18, 1920x1080, 30 fps, AAC narration, soft subtitle track),
`video/out/metagross_demo.srt`, `video/out/metagross_demo_render.json` (per-scene durations, wall time).

Files involved:

| File | Role |
|---|---|
| `video/scenes/golden_demo.yaml` | the timeline: `runs:` (run dirs + scene start times), `values:` (where every number comes from), `scenes:` (cards, split, dashboards, narration) |
| `video/NARRATION.md` | human-readable script, scene list and placeholder register (keep in sync with the YAML) |
| `video/collect_values.py` | fills `{placeholders}` from `results/claims_index.json` and from run directories |
| `video/render_timeline.py` | renders the timeline (TTS, frames, encode, mux) |
| `video/find_runs.py` | one row per run directory (seed, family, config, sensor, success, ditch entries) to pick golden runs |
| `video/inspect_run.py` | per-second table of a run (mode, v_cap, R_cert, q, referee events) to pick `t0_*` |
| `metagross/sim/render/chase_replay.py` | re-renders the third-person SIMULATOR VIEW from a run's GT log (Three.js, one shared Chrome) |

## 0. Inputs (from the closed-loop owner)

Four run directories written by `metagross.sim.runner.run_episode(..., sensor_mode="stereo")`, each with
`result.json`, `mission.json`, `gt/`, `autonomy/debug/tick_*.npz` (with `extra_left_rgb`) and
`autonomy/telemetry.jsonl`:

| `runs:` key | What | Must be |
|---|---|---|
| `run_ditch_full` | F2/F3 ditch seed, config FULL | stereo, a DEV seed (100-129) |
| `run_ditch_typical` | the **same seed**, config TYPICAL | stereo (tier0 works but the top bar then says SENSOR TIER0) |
| `run_glare` | F5 lighting seed, config FULL | stereo, a glare event in the window |
| `run_nominal` | a clean FULL drive (F1 trail) | stereo |

The chase view needs the scenario JSON the run used: `data/scenarios/<split>/<seed>.json` (or a
`scenario.json` copied into the run dir). Its sha256 is checked against `result.json`; a mismatch aborts.

Closed-loop numbers for the results card come from the claims ledger. Ask the eval owner to register (or edit
the `values:` block to point at the ids they chose): `closed_loop_n_runs_full`, `closed_loop_success_full`,
`closed_loop_ditch_entries_full`, `closed_loop_ditch_entries_typical`, then run `python -m metagross.eval.claims`.

## 1. Pick runs and scene start times (3 min)

```powershell
& $py -m video.find_runs results --stereo-only            # one row per run: seed, family, config, success, ditch, path
& $py -m video.find_runs results --family F3 --images      # also counts debug bundles that carry the onboard image
& $py -m video.inspect_run results/golden/stereo/FULL/ditch      # find the crest approach, e.g. t = 6 s
& $py -m video.inspect_run results/golden/stereo/TYPICAL/ditch   # the ditch entry shows as <ditch_entry>
& $py -m video.inspect_run results/golden/stereo/FULL/glare      # <lighting_on> marks the glare event
```

Choose `t0_ditch` a few seconds before the crest, `t0_glare` 2-3 s before `lighting_on`, `t0_link` anywhere
the robot is driving. `t1_ditch` / `t1_glare` only bound the windows the spoken numbers are computed over
(`split_rcert_min_m` = minimum R_cert in [t0_ditch, t1_ditch]; `glare_q_min`, `glare_worst_mode` in
[t0_glare, t1_glare]): end them when the event is over, so a later stall cannot set the number.
Each dashboard / split scene plays the run from `t0` for as long as its narration lasts (about 20-30 s; see the
plan in step 3). If a run is too slow to show the event in that time, add `speed: 2` to
that scene: the top bar then shows `PLAYBACK 2x`.

## 2. Fill the numbers (1 min)

Pass the run directories either by editing `runs:` in `golden_demo.yaml` or with `--run` (same flags for both tools):

```powershell
$RUNS = @("--run", "run_nominal=results/golden/stereo/FULL/nominal",
          "--run", "run_ditch_full=results/golden/stereo/FULL/ditch",
          "--run", "run_ditch_typical=results/golden/stereo/TYPICAL/ditch",
          "--run", "run_glare=results/golden/stereo/FULL/glare",
          "--run", "t0_ditch=6", "--run", "t1_ditch=20", "--run", "t0_glare=10", "--run", "t1_glare=25",
          "--run", "t0_link=5")
& $py -m video.collect_values video/scenes/golden_demo.yaml --out video/out/values_golden.json @RUNS
```

It logs every value with its label and source, and lists missing ones. `values_golden.json` also carries a
`claims` list (ledger row shape) for every run-derived number: copy it to `results/video_values.json` and run
`python -m metagross.eval.claims` so the numbers spoken in the video are in `results/claims.csv` (Simulated).
The render refuses to start while any `{placeholder}` is unfilled (unless `--allow-placeholders`).

## 3. Check length and story (3 min + watching)

```powershell
& $py -m video.render_timeline video/scenes/golden_demo.yaml video/out/metagross_demo.mp4 --values video/out/values_golden.json @RUNS --plan-only
& $py -m video.render_timeline video/scenes/golden_demo.yaml video/out/metagross_demo.mp4 --values video/out/values_golden.json @RUNS --stills video/out/stills
```

`--plan-only` synthesises the narration (edge-tts, needs network; cached under `video/out/metagross_demo_work/tts/`)
and prints each scene's length and the total (target 2:30-3:00). `--stills` writes the middle frame of every scene
to `video/out/stills/` - open them and check the layout.

**Story check (do not skip).** The narration of scenes 5 and 6 describes what should happen. Confirm it in the
stills / `inspect_run` tables, and edit the sentence in both `golden_demo.yaml` and `NARRATION.md` if the run
shows something else:

* split: FULL shows grey beyond the crest, then magenta, then red cells, and R_cert drops; TYPICAL "keeps going"
  (if it enters the ditch you may say "and drives in" - only if the referee logged `ditch_entry`);
* glare: the mode actually changes after `t0_glare` and `glare_worst_mode` is caused by glare (not a stall);
* operator link: `packet_bytes` is plausible and the OPERATOR LINK panel shows a packet.

If the total is over 3:00: shorten a sentence, or raise `tts_rate` (`+12%` now) a little; never cut the honesty
lines. If under 2:30, lower `tts_rate`.

## 4. Render (the long step)

```powershell
& $py -m video.render_timeline video/scenes/golden_demo.yaml video/out/metagross_demo.mp4 --values video/out/values_golden.json @RUNS
```

Options: `--chase-fps 15` halves the chase-camera renders (the chase view then updates at 15 fps; fine for a
review copy), `--preset veryfast` for a quick encode (`fast` is a good compromise if time is short; the
encoder is limited to 2 threads), `--voice en-IN-NeerjaNeural` for the female voice,
`--no-tts` for a silent cut, `--watermark "REVIEW COPY"` to stamp every frame (used for `draft_v0`).
Chase frames are cached as JPEG in `video/out/cache/chase/` keyed by scenario, size and state, so a re-render
after a narration or layout change never touches the browser for frames it has seen.

Expected time: see "Measured" below. The render uses one Python process (plus one headed Chrome for the chase
view) and libx264 with 2 threads, so it can run next to other jobs.

## 5. Verify (2 min)

```powershell
$ff = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
& $ff -v error -i video/out/metagross_demo.mp4 -vf "select='not(mod(n\,900))'" -vsync vfr video/out/check_%02d.png
& $py -c "from video.encode import probe; print(probe('video/out/metagross_demo.mp4'))"
```

Open a few `check_*.png`: the SIMULATED chip must be visible top right on every dashboard frame and the footer
must read `SIMULATED · seed N · config X`. Listen to the whole video once.

## Only the side-by-side segment

The FULL vs TYPICAL split is scene index 4 of the timeline (0-based, as `--plan-only` prints it). To render
it alone, with its narration (e.g. for the deck or a quick review):

```powershell
& $py -m video.render_timeline video/scenes/golden_demo.yaml video/out/split_full_vs_typical.mp4 --values video/out/values_golden.json @RUNS --scenes 4
```

`--scenes` takes a comma-separated list (e.g. `--scenes 4,5,6` for the three simulation scenes).

The split scene's keys: `run_a` (left, TYPICAL), `run_b` (right, FULL), `label_a` / `label_b`, `config_a` /
`config_b`, `t0`, `t1` (`auto` = as long as the narration), optional `speed`, `chase` (`three`), `caption`.
Both runs are sampled at the same run time, so they must share the seed and start pose.

## Troubleshooting

* **"unfilled placeholders"**: a value is missing; `collect_values` logged why (claim id not in the ledger,
  wrong run path, empty window). Fix the source; for a review copy only, add `--allow-placeholders`.
* **edge-tts fails** (no network): the render continues silently with estimated scene lengths; cached narration
  is reused on the next run.
* **Chrome / WebGL problems**: if the chase renderer fails, the render logs an ERROR once and the SIMULATOR VIEW
  panel falls back to a labelled top-down view for the rest of the video. Check the log before publishing.
* **`ScenarioNotFound`**: the run's `result.json` sha256 does not match `data/scenarios/<split>/<seed>.json`;
  copy the exact scenario into the run dir as `scenario.json`.
* The chase camera sits 6 m behind and 3 m above the vehicle (fixed in `web/main.js`); near trees it can clip
  a trunk. Pick a `t0` that avoids it.

## Measured (2026-09-30, this laptop, shared with 5 other jobs; CPU at 100 % during the renders)

| What | Value | How |
|---|---|---|
| chase frame, 1152 x 648 (single view) | 159 ms mean, 152 ms median (20 frames) | `render_chase` wall time incl. transfer, seed 102, first-frame browser start + scene load 8.1 s |
| chase frame, 936 x 526 (split view) | 115 ms mean (10 frames) | same |
| dashboard composition without chase | 66 ms / frame (30 frames) | `Dashboard.compose` + `RunReplay.frame_at`, run 102, chase cached |
| draft_v0, first full render | 5328 frames, 2:57.6, 844 s wall (14.1 min) | about half of the chase frames came from the cache of an earlier stills pass |
| draft_v0, re-render after layout fixes | 5328 frames, 614 s wall (10.2 min) | every chase frame from the cache: this is composition + encode alone (`video/out/draft_v0_render.json`) |
| narration synthesis | not timed separately (well under the frame time); cached after the first run | edge-tts, needs network |

Budget for the final (golden runs, nothing cached): 10.2 min composition + encode, plus the chase renders
(1,151 single-view frames x 155 ms + 858 split frames x 2 x 115 ms, about 6.5 min, plus ~10 s browser start and
scene loads) = **about 17 min for step 4** on the loaded machine, plus ~10 min for steps 1-3 and 5: inside 40 min.
`--chase-fps 15` saves ~3 min, `--preset fast` saves part of the encode.

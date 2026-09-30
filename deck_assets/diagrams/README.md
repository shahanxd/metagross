# Deck diagrams (SIH26126, METAGROSS)

Every diagram exists in two editable forms:

1. **Figma (primary, editable).** The file is "METAGROSS — SIH26126 deck diagrams":
   https://www.figma.com/design/4Sbgr7uJgmCuuymKd6Ocoo (page "Deck diagrams"). Each frame has PNG @2x and SVG
   export settings, so select the frame and click Export.
2. **Code (reproducible).** `build_diagrams.py` writes SVG and renders 2x PNG through headless Chromium:
   `& ".venv\Scripts\python.exe" deck_assets\diagrams\build_diagrams.py` (run it from PowerShell, because it launches a browser).

| # | Diagram | Figma node | Figma export (2x) | Code version (SVG + 2x PNG) |
|---|---|---|---|---|
| i | HOW IT WORKS (slide-2 hero): 6 stages, LINK branch, health band | `1:2` (auto-layout) | `figma/01_how_it_works.png` | `how_it_works.svg/.png` |
| ii | SYSTEM ARCHITECTURE: world, firewall (6 layers), autonomy, operator, evaluator | `2:2` (auto-layout) | `figma/02_system_architecture.png` | `system_architecture.svg/.png` |
| iii | MODEL + PIPELINE: segmentation model, integrity monitor, MPPI loop | `3:2` (auto-layout; the rollout fan is vectors) | `figma/03_model_pipeline.png` | `model_pipeline.svg/.png` |
| iv | MISSING GROUND concept: side view + column range profile | `4:2` (imported SVG: vectors + text layers, not auto-layout) | `figma/04_missing_ground.png` | `missing_ground.svg/.png` |
| – | Colour key strip (cell states) | `3:136` | – | `legend_cell_states.png`, `legend_cell_states_core6.png` |
| – | Honesty chips | `3:161` (chips `3:162`, `3:164`, `3:166`, `3:168`, `3:170`) | – | `chips/chip_<label>.png` (transparent background) |

The Figma and code versions carry the same content. Their layouts differ slightly: the code version of (i) has small glyphs
(cameras, mini map, skid-steer), and the Figma version uses editable text in their place.

Facts on the diagrams, and the files they come from:

* Camera, rates, link budget and map size come from `metagross/config/defaults.py`.
* MPPI K = 512 and T = 30 × 0.1 s come from `autonomy/planning/mppi.py`.
* Mode thresholds 0.7 / 0.4 / 0.2 come from `autonomy/safety/supervisor.py`.
* The 12 integrity features come from `autonomy/localization/health.py`.
* The 3.22 M parameters and 320×416 input come from `docs/MODEL_CARD.md`.
* The 6 firewall layers come from `tests/test_gt_separation.py`.
* The packet layout comes from `autonomy/link/codec.py`.
* No measured performance number appears on any diagram.

Caveat: the OFFROAD5 composition (RUGD + RELLIS-3D + GOOSE) is marked [UNVERIFIED] in `docs/REFERENCES.md`.
Keep that caveat in the speaker notes.

Colours and type are listed in `docs/DESIGN_TOKENS.md`.

# Third-party software, models and datasets

This file lists everything METAGROSS depends on, with its licence, and where it is used. Package licences were read
from the installed package metadata in `.venv` on 2026-09-30 (Python 3.10, `importlib.metadata`); dataset licences
come from `docs/REFERENCES.md`, section F, where each was checked against the official page or the Hugging Face
card. Anything not confirmed is marked **[UNVERIFIED]**. This is an engineering summary, not legal advice.

The METAGROSS repository itself has no licence file yet. That decision belongs to the team and BEL.

## 1. Policy

* **Onboard runtime (the autonomy process) uses only permissive licences**: BSD, MIT, Apache-2.0. An LGPL library
  may be present only if it is dynamically loaded and replaceable.
* **No GPL or AGPL code in the onboard process.** GPL-3.0 requires that a distributed work that includes the code be
  released under GPL-3.0 with its source. AGPL-3.0 adds the same obligation for software used over a network. A
  product for BEL built on such code would have to be released under those terms, or would need a separately
  negotiated licence from the copyright holders. We therefore do not use:
  * **ORB-SLAM3** (GPL-3.0): replaced by our own frame-to-frame stereo VO (`metagross/autonomy/localization/vo.py`),
    built on OpenCV (Apache-2.0). See `docs/QA.md`, question 9.
  * **Ultralytics YOLO** (AGPL-3.0): not needed; obstacles come from stereo geometry, and terrain classes from an
    LR-ASPP MobileNetV3 model built with torchvision (BSD-3-Clause).
* **Non-commercial datasets and weights are for research and evaluation only.** Any model trained on them (see
  section 4) must be retrained on data BEL owns or licenses before it is fielded.

## 2. Python packages

Versions are those installed in the development venv. "Onboard" = imported by `metagross/autonomy`,
`metagross/contracts` or `metagross/config`.

| Package | Version | Licence | Used by |
|---|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause | onboard, everything |
| opencv-python | 5.0.0.93 | Apache-2.0; the wheel also bundles an FFmpeg DLL under LGPL-2.1 (per the wheel's `LICENSE-3RD-PARTY.txt`), which the stack does not use | onboard (SGBM, KLT, PnP, CLAHE), sim, eval, video |
| scipy | 1.15.3 | BSD-3-Clause | onboard, sim, eval |
| scikit-image | 0.25.2 | BSD-3-Clause | onboard (global planner, `skimage.graph.MCP_Geometric`) |
| onnxruntime | 1.23.2 | MIT | onboard (segmenter), eval |
| onnx | 1.23.1 | Apache-2.0 | train (export check), tests |
| matplotlib | 3.10.9 | Matplotlib licence (PSF-based, permissive) | eval figures, deck assets |
| pillow | 12.3.0 | MIT-CMU (HPND) | video, deck assets, scripts |
| scikit-learn | 1.7.2 | BSD-3-Clause | eval (integrity monitor training, AUROC) |
| pandas | 2.3.3 | BSD-3-Clause | installed; analysis convenience |
| torch | 2.14.0 | BSD-3-Clause style (metadata lists Apache-2.0, BSD-2/3-Clause and LLVM-exception components) | train only |
| torchvision | 0.29.0 | BSD-3-Clause | train only (LR-ASPP MobileNetV3 model code) |
| playwright | 1.63.0 | Apache-2.0 | sim renderer bridge, console capture, diagrams |
| huggingface_hub | 2.0.0 | Apache-2.0 | dataset download scripts |
| requests | 2.34.2 | Apache-2.0 | download scripts |
| imageio | 2.38.0 | BSD-2-Clause | video |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause (Python wrapper). The bundled `ffmpeg-win-x86_64-v7.1.exe` is a **GPL-3.0 build** (`-version` reports `--enable-gpl --enable-version3 --enable-libx264`) | video encoding only, called as a separate executable; never part of the onboard stack. Redistributing the video tool would carry the GPL obligations for that binary |
| edge-tts | 7.2.8 | LGPL-3.0 | video narration only; it calls Microsoft's online text-to-speech service, whose terms apply to the audio [UNVERIFIED terms] |
| PyYAML | 6.0.3 | MIT | video timelines |
| pytest | 9.1.1 | MIT | tests |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | installed dependency |
| certifi | 2026.7.22 | MPL-2.0 | installed dependency |

Other installed packages are transitive dependencies with permissive licences according to their metadata
(MIT, BSD, Apache-2.0, PSF-2.0): aiohappyeyeballs, aiohttp, aiosignal, anyio, async-timeout, attrs,
charset-normalizer, click, cloudpickle, colorama, coloredlogs, contourpy, cycler, exceptiongroup, filelock,
flatbuffers, fonttools, frozenlist, fsspec, greenlet, h11, hf-xet, httpcore2, httpx2, humanfriendly, idna,
iniconfig, Jinja2, joblib, kiwisolver, lazy-loader, MarkupSafe, ml_dtypes, mpmath, multidict, networkx, packaging,
pip, pluggy, propcache, protobuf, pyee, Pygments, pyparsing, pyreadline3, python-dateutil, pytz, setuptools, six,
sympy, tabulate, threadpoolctl, tifffile, tomli, truststore, typing_extensions, tzdata, urllib3, yarl.

GPU box only (`aws/requirements-gpu.txt`, not in the laptop venv): `opencv-python-headless` (Apache-2.0),
`transformers` and `timm` (both Apache-2.0 [UNVERIFIED here: not installed locally]), used for the optional
SegFormer training and DINOv2 teacher comparisons, never at run time.

## 3. Vendored code and other software

| Component | Licence | Where |
|---|---|---|
| three.js r186 (`three.module.js`, `three.core.js`, `Sky.js`, `BufferGeometryUtils.js`, copied verbatim) | MIT (`LICENSE-three.txt`) | `metagross/sim/render/web/vendor/` |
| kitti-odom-eval (Huangying-Zhan), KITTI devkit protocol, ported | MIT (per `metagross/eval/traj_metrics.py`) | `metagross/eval/traj_metrics.py`; GT pose files downloaded from that repository |
| Google Chrome or Microsoft Edge | proprietary, installed separately, not redistributed | renderer and console capture (Playwright `channel="chrome"`) |
| Fonts: Inter, JetBrains Mono (if installed); Segoe UI, Consolas, Arial (Windows system fonts) | Inter and JetBrains Mono: SIL OFL 1.1; Windows fonts: Microsoft licence | figure and video text only; no font file is redistributed by the repository |

## 4. Models and weights

| Model | Licence of the weights | Status |
|---|---|---|
| `models/lraspp_smoke.onnx` (our LR-ASPP MobileNetV3, smoke run) | code BSD-3-Clause (torchvision); trained on RUGD-5Labels (CC BY-NC-SA 3.0) from an ImageNet-1k pretrained backbone, so **research use only** | SMOKE pipeline check, not a trained model |
| `models/lraspp_offroad5_{clean,robust}.onnx` (planned deploy model) | trained on OFFROAD5 (CC BY-NC-SA 3.0), so **research use only** | not trained yet |
| `models/segformer_b0_ade.onnx` (zero-shot baseline, Xenova ONNX export of `nvidia/segformer-b0-finetuned-ade-512-512`) | **NVIDIA Source Code License for SegFormer: non-commercial, research and evaluation only** (recorded in `models/segformer_b0_ade.json`) | evaluation baseline; never deployed, never selected automatically |
| `models/integrity.json` (our logistic VO-integrity weights) | our work, fitted on features of KITTI frames (CC BY-NC-SA 3.0) | used onboard; for a product, refit on owned data |
| ImageNet-1k pretrained MobileNetV3 backbone (torchvision) | torchvision code BSD-3-Clause; ImageNet terms restrict the images to non-commercial research [UNVERIFIED for the weights] | training initialisation |
| DINOv2-S, FiT3D (optional teacher rows, `aws/teacher_dinov2.py`) | DINOv2 released by Meta under Apache-2.0 [UNVERIFIED here]; FiT3D weights: licence not checked [UNVERIFIED] | optional research comparison; not run yet |

## 5. Datasets

| Dataset | Licence | Commercial use | How we use it |
|---|---|---|---|
| KITTI odometry (images, calibration, GT poses) | CC BY-NC-SA 3.0 | **No (non-commercial)** | VO evaluation (00, 05, 07), integrity-monitor training and test |
| KITTI mirror on Hugging Face (`yujie2696/kitti_odometry_XX`) | mirror card licence not checked [UNVERIFIED]; the KITTI licence above applies to the content | No | download source for images (frames 0-1100 of its 00 are a different drive; see `results/kitti_summary.json`) |
| RUGD | no licence stated on rugd.vision, which asks users to cite the paper [UNVERIFIED] | treat as **No** | via RUGD-5Labels and OFFROAD5 |
| RUGD-5Labels-resized (GAIA-URJC, Hugging Face) | CC BY-NC-SA 3.0 (HF tag) | **No** | segmentation evaluation, smoke training |
| RELLIS-3D | CC BY-NC-SA 3.0 | **No** | via OFFROAD5 (planned training) |
| GOOSE / GOOSE-Ex | CC BY-SA 4.0 for the data (official repository) | yes with share-alike, but we receive it through OFFROAD5, whose tag is non-commercial | via OFFROAD5 (planned training) |
| OFFROAD5 (GAIA-URJC, Hugging Face; RUGD + RELLIS-3D + GOOSE in 5 labels) | CC BY-NC-SA 3.0 (HF tag); composition [UNVERIFIED], see `docs/REFERENCES.md` | **No** | planned deploy-model training |
| ADE20K | training data of the SegFormer-B0 baseline weights; terms not checked [UNVERIFIED] | - | indirectly, through the baseline weights only |
| TartanAir | CC BY 4.0 (official page) | yes, with attribution | cited as a possible VO stress test; not downloaded or used |

**Consequence.** Every learned component we have today (the smoke segmenter, the planned OFFROAD5 model, the
integrity weights) derives from non-commercial data. The code, the simulator, the analytic models and the
geometric perception do not. Before any fielded use, the learned parts need retraining on data with suitable terms,
for example terrain imagery collected by BEL on its own vehicles.

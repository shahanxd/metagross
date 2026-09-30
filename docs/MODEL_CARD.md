# Model card: METAGROSS terrain segmenter ("Perception AI")

**Task.** Per-pixel five-class terrain segmentation of the rectified left camera image, for PS SIH26126
(vision-based UGV navigation in GPS-denied outdoor terrain). The output is a cost layer on top of the geometric
seen-ground map. It never replaces geometry: stereo decides what ground has been *seen*; semantics adds *how
costly* that seen ground is (water, mud, vegetation) and flags hazards that geometry alone reads as flat, such as
a water surface.

| | |
|---|---|
| Deploy architecture | torchvision `lraspp_mobilenet_v3_large` (LR-ASPP head, MobileNetV3-Large backbone), 3.22 M parameters (counted by `metagross.train.models.count_params`) |
| Runtime | ONNX (opset 17, static `1x3xHxW`), ONNX Runtime CPU, 2 intra-op threads by default; the perception node calls it at 1/3 of the camera rate |
| Input | RGB uint8 -> resized to the model input (deploy 320x416 HxW; smoke model 256x320) -> `(x/255 - mean)/std`, ImageNet mean/std |
| Output | `class_ids (H, W) uint8` at full image resolution (nearest upsampling) and `entropy (H, W) float32`, the Shannon entropy of the 5-class posterior divided by ln 5 (range 0 to 1) |
| Interface | `metagross.autonomy.perception.semantics.Segmenter` implements `SegmenterProto` |
| Zero-shot baseline | SegFormer-B0 fine-tuned on ADE20K (research comparison only), mapped to the 5 classes |

## Labels

The ids are identical to the GAIA-URJC OFFROAD5 / RUGD-5Labels ids and to `SEM_CLASSES` in `metagross.contracts.interfaces`.

| id | class | typical content (RUGD/OFFROAD5) | planner cost `SEM_COST` | colour (RGB) |
|---|---|---|---|---|
| 0 | background / sky | sky, far background | 0.0 (never on the ground plane) | (203, 208, 214), same as UNSEEN |
| 1 | obstacle | trees, bushes, tall vegetation, rocks and rock beds, buildings, fences, poles, vehicles, people | 0.8 | (220, 50, 47), same as POSITIVE |
| 2 | water / mud | water, puddles, mud | 0.95 | (20, 140, 190), same as WATER |
| 3 | unstable | grass, dirt, gravel, sand, mulch | 0.2 | (217, 180, 58), ochre |
| 4 | stable | asphalt, concrete, paved path | 0.0 | (34, 160, 90), same as GROUND |

Colour note: the frozen POSITIVE red and GROUND green are hard to tell apart under deuteranopia (palette
validator: deutan ΔE 5.5), so every figure also carries a text legend.

## Data

| Dataset | Content | Split sizes (indexed by `data_index.list_samples`) | Licence |
|---|---|---|---|
| `GAIA-URJC/RUGD-5Labels-resized` | RUGD off-road video frames (US trails, creek, park, village), 400x320, re-labelled to the 5 classes | train 4779, val 1924, test 733. The splits are scene-disjoint: val = creek, park-1, trail-7, trail-13; test = park-8, trail-5 | CC BY-NC-SA 3.0 |
| `GAIA-URJC/OFFROAD5` | RUGD + RELLIS-3D + GOOSE/GOOSE-Ex merged into the same 5 labels, 320 px tall | val 5275 (from the zip listing); train and test are downloaded on the GPU box | CC BY-NC-SA 3.0 |

- **Per-source tagging.** GOOSE masks carry the `_group5b` suffix, RELLIS-3D frames are named `frame*`, and everything else is RUGD. All metrics can therefore be split by source.
- **Determinism.** File lists are sorted. Subsets are drawn with a seeded RNG, and the SHA-256 of each list is stored in every metrics JSON. The augmentation for a sample is a pure function of `(seed, epoch, index)`.
- **Class imbalance.** Water/mud covers roughly 1 % of pixels or less, and "stable" is also rare in RUGD. The loss therefore uses ENet weights `1/ln(1.02 + f_k)` (normalised to mean 1), with class frequencies measured on 400 seeded training masks and stored in `metrics.json`.

## Training recipe (`python -m metagross.train.train_seg`)

- **Initialisation.** ImageNet-1k MobileNetV3-Large backbone (torchvision weights). The LR-ASPP head is new (5 classes).
- **Loss.** Class-weighted cross-entropy plus 0.5 × soft multi-class Dice. Ignore label: 255.
- **Optimiser.** AdamW (lr 6e-4, weight decay 1e-4 on weights only), linear warm-up, then poly (0.9) or cosine decay to 1 % of the base lr. Gradient-norm clip 5.
- **GPU recipe** (`aws/setup_and_train.sh`). OFFROAD5 train, 40 epochs, batch 32, crop 320x416, AMP (bf16 where supported). Two parallel runs, CLEAN vs ROBUST augmentation. The best checkpoint is chosen by val mIoU; resume comes from `last.pt`.
- **Augmentation.**
  - CLEAN: horizontal flip; scale 0.75 to 1.25 with aspect jitter 0.8 to 1.25; random crop, padding with ignore; light colour jitter. The aspect jitter covers the squash from the 640x400 camera to the 416x320 input.
  - ROBUST: CLEAN plus gamma 0.4 to 2.5 (p 0.5), exposure clipping (over- or under-exposure, p 0.3), haze (Koschmieder model, p 0.2), synthetic sun flare with ghosts (p 0.2), motion blur (5 to 17 px, p 0.25), sensor noise (read noise plus shot noise, p 0.35) and JPEG at quality 12 to 60 (p 0.3).
- **Export** (`python -m metagross.train.export_onnx`). ONNX opset 17, static shape. The ORT output is checked against torch (max |Δlogit| and argmax agreement, recorded in the sidecar JSON), and the export fails if the difference exceeds 1e-2.

## Evaluation protocol (`python -m metagross.eval.seg_eval`)

The segmenter runs exactly as it does onboard: the image is resized to the model input, and the predicted class ids
are upsampled back to the label resolution with nearest-neighbour interpolation.

**Metrics.**
- **Per-class IoU and mIoU.** mIoU averages over classes with a non-empty union (the mmseg convention). `miou_present` averages only over classes that occur in the ground truth.
- **Pixel accuracy.**
- **False-safe rate.** The share of GT obstacle or water pixels predicted as traversable (class 3 or 4). This is the safety metric, reported overall, per hazard class and per brightness tercile.
- **False-hazard rate.** Traversable pixels predicted as a hazard. This costs availability, not safety.
- **Brightness-stratified mIoU.** Images are split into terciles of mean Rec.601 luma.
- **Per-source split.**
- **Mean entropy on correct vs wrong pixels.**
- **Latency** of the full `Segmenter.__call__` at 2 and 4 threads.

**Zero-shot mapping (SegFormer-B0 ADE20K to 5 classes).** The ADE20K posteriors are summed per group (`ADE20K_TO_SEM5`), and each id was verified against the checkpoint's `id2label`:
- road 6, sidewalk 11, path 52, runway 54, floor 3 map to **stable**;
- grass 9, earth 13, field 29, sand 46, dirt track 91, land 94 map to **unstable**;
- water 21, sea 26, river 60, swimming pool 109, waterfall 113, lake 128 map to **water**;
- sky 2, mountain 16, hill 68 map to **background**;
- tree, plant, palm, rock, fence, wall, building, pole, person, vehicles, animal, signboard, bench and **every unlisted ADE class** map to **obstacle**. This conservative default follows the thesis "unknown is never free".

## Results

<!-- METRICS:BEGIN -->
All numbers below are generated by `python -m metagross.train.model_card` from the listed files (label **Tested** = measured on real images; SMOKE = pipeline check, not a trained model). Latency is the median wall-clock time of `Segmenter.__call__` (resize + normalise + ORT + argmax/entropy + upsample) on one 640x400 camera frame on the development laptop (4 cores / 8 threads, no GPU, CPU shared with other jobs during measurement; host details in each JSON); n/m = not measured.

| Model | Eval set | n img | mIoU % | pixel acc % | false-safe % | FS obstacle % | FS water % | ms @2 thr | ms @4 thr | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | rugd5 val | 1924 | 55.1 | 78.9 | 27.3 | 26.3 | 94.2 | 31 | 27 | `results/seg_smoke.json` |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | rugd5 test | 733 | 63.3 | 92.2 | 4.3 | 4.2 | 90.0 | 31 | 27 | `results/seg_smoke.json` |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | rugd5 val | 1924 | 54.3 | 80.9 | 16.9 | 16.2 | 56.7 | 223 | 118 | `results/seg_zeroshot.json` |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | rugd5 test | 733 | 54.4 | 87.6 | 2.6 | 2.5 | 79.0 | 223 | 118 | `results/seg_zeroshot.json` |

Per-class IoU (%):

| Model | Eval set | background/sky | obstacle | water/mud | unstable (grass/dirt) | stable (asphalt/concrete/gravel path) |
|---|---|---|---|---|---|---|
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | val | 68.6 | 68.1 | 0.0 | 61.5 | 77.3 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | test | 55.2 | 85.6 | 0.0 | 91.4 | 84.2 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | val | 74.7 | 74.9 | 21.7 | 58.8 | 41.2 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | test | 55.0 | 83.8 | 1.3 | 79.8 | 52.0 |

Brightness-stratified mIoU % (images split into terciles of mean luma; dark / mid / bright):

| Model | Eval set | dark | mid | bright | false-safe dark % | false-safe bright % |
|---|---|---|---|---|---|---|
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | val | 53.5 | 55.8 | 53.8 | 23.4 | 30.8 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | test | 62.7 | 63.9 | 62.3 | 3.6 | 5.6 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | val | 52.6 | 57.5 | 50.5 | 10.1 | 22.6 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | test | 57.7 | 52.1 | 50.8 | 2.1 | 3.3 |

False-safe by scene (top 6 scenes per split by false-safe pixels; last column = share of all false-safe pixels in that split):

| Model | Eval set | scene | n img | mIoU % | false-safe % | share of false-safe px % |
|---|---|---|---|---|---|---|
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | val | creek | 836 | 30.3 | 42.2 | 89.9 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | val | park-1 | 627 | 63.6 | 6.9 | 6.1 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | val | trail-7 | 289 | 46.8 | 7.6 | 3.3 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | val | trail-13 | 172 | 52.5 | 3.3 | 0.7 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | test | trail-5 | 376 | 45.6 | 5.3 | 64.6 |
| SMOKE LR-ASPP (300 CPU iters, 800 RUGD-5L imgs) | test | park-8 | 357 | 63.8 | 3.3 | 35.4 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | val | creek | 836 | 38.5 | 24.2 | 83.6 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | val | park-1 | 627 | 66.1 | 7.9 | 11.3 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | val | trail-7 | 289 | 39.2 | 5.3 | 3.8 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | val | trail-13 | 172 | 50.2 | 3.8 | 1.3 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | test | trail-5 | 376 | 40.4 | 3.2 | 63.7 |
| Zero-shot SegFormer-B0 (ADE20K -> 5 classes) | test | park-8 | 357 | 65.4 | 2.0 | 36.3 |
<!-- METRICS:END -->

**Reading the results.** This section is interpretation only; every number is in the tables above and in the JSON files.
- **The smoke model does not cover water.** It never predicts water/mud: the 800-image training subset contains almost no water pixels, and 300 iterations are not enough to learn the class. Its false-safe rate on water is therefore close to total. The full OFFROAD5 run with ENet weights is what addresses this.
- **Val false-safe comes from creek scenes.** Most of the val false-safe pixels come from the creek sequence, the largest scene in RUGD val (see the *false-safe by scene* table). RUGD labels rocky creek beds as *obstacle*, and the models call them unstable ground (grid figures, first rows). Val false-safe is therefore a pessimistic, scene-dominated number; read it together with the per-scene rows.
- **Zero-shot vs smoke.** The zero-shot SegFormer is safer on val (lower false-safe rate) but much slower. The smoke LR-ASPP is more accurate on the paved-vs-unstable split and fits the CPU budget. Neither is the deploy model; the GPU runs replace both.
- **Entropy is informative but uncalibrated.** Mean entropy is clearly higher on wrong pixels than on correct ones for both models, so it can gate how much weight the semantic cost gets.

## Licences

| Component | Licence | Deployment status |
|---|---|---|
| LR-ASPP / MobileNetV3 code (torchvision) | BSD-3-Clause | OK |
| ImageNet backbone weights (torchvision) | BSD-3 distribution; ImageNet data terms apply | Used as initialisation |
| RUGD-5L / OFFROAD5 mirrors (RUGD, RELLIS-3D, GOOSE) | CC BY-NC-SA 3.0 | Research/evaluation use. Weights trained on them inherit the non-commercial terms |
| SegFormer-B0 ADE20K (NVIDIA) + Xenova ONNX export | NVIDIA Source Code License (non-commercial) | Research baseline only, never deployed |
| DINOv2 ViT-S/14 (Meta) | Apache-2.0 | Teacher experiment only |
| FiT3D weights (Yue et al., ECCV 2024) | See the authors' release | Teacher experiment only |
| ONNX Runtime, OpenCV, numpy | MIT / Apache-2.0 / BSD | OK |

No GPL or AGPL code is in the runtime path. A production BEL deployment would retrain with the same pipeline on
commercially licensed or self-collected, self-labelled data. Only the data and weights change; the code does not.

## Intended use and limitations

- **Intended use.** A cost layer and water/mud detector for a slow UGV (≤ 2 m/s) driving on terrain it has already seen. It is not a standalone obstacle detector: geometric POSITIVE, DEPRESSION and missing-ground states stay authoritative.
- **Domain gap.** RUGD, RELLIS-3D and GOOSE were recorded in US and German forests, parks and fields, in daylight. There are no Indian terrain types (laterite, black cotton soil, desert scrub), no night-time frames, no rain on the lens and no dust storms. ROBUST augmentation only partly covers these conditions. Expect a lower mIoU on the sim renderer and on Indian test sites until the model is fine-tuned on local data.
- **The label "obstacle" includes vegetation.** Tall grass and bushes that a UGV could push through are labelled obstacle. This is conservative, and it can make the planner refuse passable vegetation.
- **Rare classes.** Water/mud is rare in training and evaluation, so its IoU has high variance. Read the per-class numbers together with the pixel share shown in the confusion-matrix figure.
- **Smoke model.** `models/lraspp_smoke.onnx` is a pipeline check: a few hundred CPU iterations on an 800-image RUGD subset at 256x320. It is labelled SMOKE everywhere and is not the trained model. It exists so the node can run semantics before the GPU run finishes.
- **Uncertainty.** `entropy` is reported as-is and is not calibrated. The eval JSON gives the mean entropy on correct vs wrong pixels as a first check that it is informative.
- **Aspect ratio.** The camera frame (640x400) is squashed to the model input. Training uses aspect jitter to cover this squash.

## Reproduce

```bash
python scripts/download_seg_data.py --dataset rugd5 --zero-shot-model          # data/rugd5 + models/segformer_b0_ade.onnx
python -m metagross.eval.seg_eval --model models/segformer_b0_ade.onnx --data-root data/rugd5 --splits val test \
    --out results/seg_zeroshot.json --fig-prefix deck_assets/seg_zeroshot --label "Zero-shot SegFormer-B0 (ADE20K -> 5 classes)"
python -m metagross.train.train_seg --model lraspp --data rugd5 --aug robust --device cpu --threads 3 --workers 1 \
    --img 256x320 --bs 8 --epochs 3 --train-subset 800 --val-subset 240 --lr 1e-3 --warmup-iters 20 --out runs/seg/lraspp_smoke
python -m metagross.train.export_onnx --ckpt runs/seg/lraspp_smoke/best.pt --out models/lraspp_smoke.onnx --data-root data/rugd5
python -m metagross.eval.seg_eval --model models/lraspp_smoke.onnx --data-root data/rugd5 --splits val test --smoke \
    --out results/seg_smoke.json --fig-prefix deck_assets/seg_smoke --label "LR-ASPP SMOKE (300 CPU iters, 800 RUGD imgs)"
python -m metagross.train.model_card                                              # regenerates the Results section
bash aws/setup_and_train.sh                                                        # full GPU training (see aws/README.md)
```

# Training, experiments & packaging (state as of 2026-08-04)

The pipeline is a **distillation loop**: SAM 3 (open-vocabulary teacher, gated
weights, see [preannotation/README.md](../preannotation/README.md)) drafts
boxes for every store frame; small EfficientDet students train on those
drafts; humans grade and correct rather than annotate from scratch.

```
unified store (data/images, 117k frames, 39 garages)
   └─ SAM 3 drafts (per-sensor preannotations.coco.json, 724k boxes)
        ├─ CVAT audit subset (~4k frames) ── human gold: eval yardstick +
        │                                    blind-spot hunting (exports in
        │                                    data/cvat_exports/ OVERRIDE drafts)
        └─ training (run_training_from_config.py / run_sweep.py)
             └─ auto: per-class metrics, dashboard row, TFLite variants,
                      sensor drop-in packages (dynamic/f32/int8 @ native res)
```

## Experiment machinery

- **`run_sweep.py`** — two modes:
  - size sweep (`--sizes 6000,12000,…,full [--arm-b]`): nested, log-spaced
    train subsets, one frozen val set, constant gradient-step budget
  - capacity ladder (`--models tf_efficientdet_lite1,…,tf_efficientdet_d2`):
    one full-store run per architecture
- **Frozen split** (`split_by: sensor-hash` in
  [config/train_sam3_full.yaml](../config/train_sam3_full.yaml)): val
  membership = hash(salt: garage/sensor) — never changes as the store grows;
  never change `split_salt`. Whole sensors on one side (no camera leakage);
  subsets are nested prefixes.
- **Tracking**: TensorBoard scalars per run (`train/<ts>/tb/`), per-class +
  size-banded `coco_metrics.json` (run_metrics.py), `run.json` provenance
  manifest, and a self-contained dashboard —
  `experiments/falcon-vision-effdet/report.html` — regenerated after every run.
- **Eval harnesses**: `validate.py` (PyTorch), `eval_tflite_coco.py` (any
  TFLite incl. the prod baseline; 4th arg `ours` for our class indices),
  `preannotation/render_drafts.py` (visual browse of drafts).
- **Reading metrics**: garage data is ~94% car — ALWAYS read per-class AP
  (car, person), never the 6-class mean (motorcycle/bus have ~no support and
  drag the mean to meaninglessness). Val labels are SAM3's: sweep numbers are
  distillation fidelity; honest accuracy comes from audited gold.

## Capacity ladder & deployment targets (2026-08-05)

Six architectures, identical data (94,628 train / frozen 5k val), identical
step budget — visualization: `experiments/falcon-vision-effdet/ladder.html`.
Car AP as %: lite0 47.6 · lite1 53.9 · **lite2 59.3** · lite3 60.0 ·
lite4 60.8 · **d1 62.7** (d2 pending). Person %: 2.1/3.2/5.4/5.3/7.3/6.1 vs
the prod-baseline gate of 3.8. Findings: knee at lite2; d1 beats lite4 with
half the params (d-series architecture > scaled lite; provisional FVS2/NPU
pick); person AP is capacity/resolution-bound, clears the gate from lite2 up.

**Deployment targets**: PRIMARY = lite1 `dropin-384-int8` (3.1 s CM3 bench,
2× production's car accuracy); speed fallback = lite0 `dropin-320-int8`
(1.6 s). lite2 (59.3%) runs 5.4 s — over the ≤4 s budget — pending one
experiment (plain-sum BiFPN retrain) before conceding it to FVS2.
Known issue: lite4/d1 clean-int8 packaging currently fails (d-series SiLU vs
TF2.8 converter suspected; checkpoints verified fine via .ts.pt).

## Earlier results (frozen 5k val, SAM3 labels)

- **Data curve (lite0)**: car AP flat ~0.463 from 6k→50k train images —
  **lite0 saturates at ~6k**; full-store 0.4756 (small cooldown-step caveat).
  Fixed-camera fleets: diversity is the currency, not volume.
- **COCO replay (arm B, 25% cap)**: car AP −1.2, person +0.004 → rejected for
  lite0. Person AP (~0.02 at every size) is resolution/capacity-bound, not
  data-bound — the capacity ladder's main question.
- **Round 0** = arm A full-store lite0 (train/20260803-182412): car AP 0.4756
  / AP-large 0.8508. vs prod baseline on the same yardstick: ~0.26 car AP.
- **Human annotation strategy** (revised): ~3–5k audited frames as eval gold +
  condition-slice blind-spot checks; no volume box-drawing. Audited CVAT
  exports automatically override drafts in every training build.

## Packaging, artifact naming & on-device latency (CM3, bench-measured)

**Export path (default since 2026-08-05): the CLEAN converter** —
torch → ONNX → onnx2tf → quantize ([clean_convert.py](../clean_convert.py);
venvs via `setup_convert_venvs.sh`; `--legacy-export` = old litert-torch
path, ~15-20% slower on-device from transpose/boundary-op pollution).

**Artifact naming — size and quantization are ALWAYS explicit:**

| file | what it is |
|---|---|
| `<run>.dropin-<size>-f32.tflite` | sensor-ready, float32 (most accurate) |
| `<run>.dropin-<size>-dyn.tflite` | sensor-ready, dynamic-range (int8 weights/f32 act) |
| `<run>.dropin-<size>-int8.tflite` | sensor-ready, full-int8 (fastest on CM3 CPU) |
| `<run>.raw-{f32,dyn,int8}.tflite` | raw network (pre-NMS heads) — dev/eval only, NOT for sensors |
| `<run>.ts.pt` | TorchScript (desktop inference) |

One `package_dropin.py` call builds and validates all three dropins at the
model's **native input size**. Firmware reads input dims from the file
(never ship a 448 build of a 320 model: exactly 2× latency — that mistake is
why sizes live in the filename now).

| build (clean export) | on-device |
|---|---|
| off-shelf lite2@448 int8 (prod) | ~3.0 s |
| lite0-320: dyn / f32 / **int8** | 3.0 / 2.6 / **1.6 s** |
| lite1-384 int8 | **3.1 s** |
| lite2-448 int8 (legacy / AEQ / TFQ) | 6.6 / 5.4 / 5.6 s |

- The 2.6 FW runtime DOES run new-style int8 (bench-verified 2026-08-03).
- **dynamic-range is SLOWER than f32** on 2.6 (hybrid-conv reference-kernel
  penalty) — dyn is a compatibility variant, never the speed pick.
- 2.6-era XNNPACK accelerates **float only** (why int8 tests never showed
  gains). With the clean graph, f32+XNNPACK (FW rebuild) is an untested lever.
- Clean export bought 6–19% on-device and matched Google's export op-for-op
  on the conv chain; the residual lite2 gap vs the off-the-shelf is
  **architectural** (weighted-BiFPN SUM chains + unfused RELU6 = elementwise
  feature-map passes the A53 pays for and desktop hides) — remaining lever:
  retrain with `weight_method='sum'`.
- Desktop timing lost latency-prediction authority for boundary-op effects
  (ties on desktop, 1.8× apart on device) — only the sensor bench decides.
- Decision policy: accuracy first while OD latency ≤ ~4 s (OD is the fusion
  confirm voice at 40% duty; prod ran 3 s for years).

## Queued next

1. lite4/d1/d2 clean-int8 packaging failure — diagnose (SiLU vs TF2.8?)
2. d2 eval → complete the ladder (+ ladder.html)
3. Plain-sum lite2 retrain (the ≤4 s challenger)
4. Round 1: photometric augmentation (night/glare/WB) + person-AP fix;
   gold re-score of all checkpoints as audited exports accumulate
5. Spot-occupancy evaluator vs portal validations (the business metric)
6. Greg-side: XNNPACK FW rebuild test with the clean f32 dropin
7. FVS2 spec: native-res training-image capture + modern runtime (long-lead)

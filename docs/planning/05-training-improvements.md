# 05 — Training Improvements (not started)

Ordered by expected value per effort.

1. [ ] **Fine-tune from the COCO detector checkpoint** instead of scratch: current best (mAP 0.231) trained with `pretrained: false`. The wrapper already wires `--pretrained`; flip config and rerun lite0. Cheapest accuracy win available.
2. [ ] **Anchor tuning** from our data: generate bbox width/height/area/aspect distributions per class from CVAT exports; parking viewpoints are unusual (elevated, close-range, fixed cameras), so COCO-default anchors (`aspect_ratios [1, 2, 0.5]`) likely misfit. Roadmap has details.
3. [ ] **Class set review**: 6 COCO classes today (person, bicycle, car, motorcycle, bus, truck); consider collapsing to vehicle/person(/two-wheeler) — fewer classes helps small models; keep COCO-id mapping for external data merges.
4. [ ] **Per-tier model scaling** once TFLite path works: lite0@320 (CM3+), lite2+/higher res for the NPU tiers; train from one codebase, sweep variant × input size against on-device latency.
5. [ ] Dataset composition: currently garage data + class-filtered full COCO (~71.5k images, COCO-dominated). Revisit ratio/sampling weights once portal data scales up — our domain images should dominate late training.
6. [ ] Eval hygiene: val split is COCO-heavy, so eval mAP under-represents garage performance; add a garage-only val metric.
7. [ ] Later (roadmap "Change Training Loop"): LR finder + cyclic LR, stronger augmentation (mixup/cutmix already exist in upstream train.py flags), QAT if PTQ int8 costs accuracy.

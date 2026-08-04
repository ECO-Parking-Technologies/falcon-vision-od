#!/usr/bin/env python3
"""Learning-curve sweep: train at nested, log-spaced dataset sizes to find
where accuracy plateaus vs training-set size.

Runs ascending sizes so the final (full) point IS the round-0 model. Every
point shares the same frozen val set (split_by: sensor-hash) and the same
gradient-step budget (epochs: auto), so points compare fairly. The dashboard
(report.html) picks each run up automatically.

    python3 run_sweep.py --config config/train_sam3_full.yaml
    python3 run_sweep.py --config ... --sizes 6000,25000,full
    python3 run_sweep.py --config ... --arm-b     # one extra full-size run
                                                  # with capped COCO mixed in

Sizes are image-count caps on the train side (whole sensors, nested subsets);
"full" = no cap. --arm-b answers "does COCO replay fix person AP / world
knowledge?" — same data otherwise, include_coco flipped on.
"""
import argparse
import copy
from pathlib import Path

import yaml

from run_training_from_config import run_training

DEFAULT_SIZES = ["6000", "12000", "25000", "50000", "full"]


def one_point(base_cfg, scratch, tag, overrides):
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)
    cfg["label_source"] = f'{base_cfg.get("label_source", "sam3")}-{tag}'
    p = scratch / f"sweep_{tag}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    print(f"\n{'='*70}\n[sweep] point: {tag}  {overrides}\n{'='*70}")
    run_training(str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/train_sam3_full.yaml")
    ap.add_argument("--sizes", default=",".join(DEFAULT_SIZES),
                    help='comma list of train-image caps, "full" = no cap')
    ap.add_argument("--arm-b", action="store_true",
                    help="add one full-size run with capped COCO mixed in")
    ap.add_argument("--models", default=None,
                    help="capacity ladder instead of a size sweep: comma list "
                         "of effdet model names, each trained once on the full "
                         "store (e.g. tf_efficientdet_lite1,tf_efficientdet_d2)")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    scratch = Path(base_cfg["output_dir"]) / "sweep_configs"
    scratch.mkdir(parents=True, exist_ok=True)

    if args.models:
        # capacity ladder: one full-store point per architecture, all exported
        BS = {"tf_efficientdet_lite4": 6, "tf_efficientdet_d1": 6,
              "tf_efficientdet_d2": 4}  # keep the 3090 out of OOM at 640/768
        for name in [m.strip() for m in args.models.split(",") if m.strip()]:
            one_point(base_cfg, scratch, name.replace("tf_efficientdet_", ""),
                      {"model": name, "export_after_training": True,
                       "batch_size": BS.get(name, base_cfg["batch_size"])})
        print("\n[sweep] ladder done — open the dashboard: "
              f"{base_cfg['output_dir']}/report.html")
        return

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    for i, s in enumerate(sizes):
        last = i == len(sizes) - 1  # full point exports even if arm B follows
        over = {"export_after_training": last}
        if s != "full":
            over["max_train_images"] = int(s)
        one_point(base_cfg, scratch, s, over)

    if args.arm_b:
        one_point(base_cfg, scratch, "full-coco",
                  {"include_coco": True, "export_after_training": True})

    print("\n[sweep] done — open the dashboard: "
          f"{base_cfg['output_dir']}/report.html")


if __name__ == "__main__":
    main()

import hashlib
import json
import logging
import os
import random
import runpy
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

import train
from config.label_loader import *


def load_audited_overrides(config):
    """Human-audited CVAT exports (data/cvat_exports/<garage>.json, COCO 1.0)
    outrank SAM3 drafts: returns {garage: {image_basename: [(orig_cat_id,
    bbox), ...]}} for every frame a human has audited."""
    root = Path(config.get("audited_exports_dir", "data/cvat_exports"))
    if not root.is_dir():
        return {}
    name_to_orig = {v: k for k, v in load_label_map().items()}
    out = {}
    for f in sorted(root.glob("*.json")):
        d = json.loads(f.read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        per_img = {}
        for a in d["annotations"]:
            per_img.setdefault(a["image_id"], []).append(a)
        g = out.setdefault(f.stem, {})
        for im in d["images"]:
            g[os.path.basename(im["file_name"])] = [
                (name_to_orig[cats[a["category_id"]]], a["bbox"])
                for a in per_img.get(im["id"], [])
                if cats[a["category_id"]] in name_to_orig]
    if out:
        n = sum(len(g) for g in out.values())
        print(f"[INFO] audited overrides: {n} frames across {len(out)} garages "
              "(human labels replace drafts on these)")
    return out


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def save_coco_annotations(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_coco_dataset(base_data_path: Path):
    coco_dir = base_data_path / "coco_dataset"
    if coco_dir.exists():
        logging.info(f"Found COCO dataset at {coco_dir}")
        return coco_dir

    # Warning about download size
    logging.warning(
        "COCO dataset not found. This will download ~20 GB of data—make sure you have the space and bandwidth!"
    )

    coco_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
    # Download
    for fname, url in urls.items():
        logging.info(f"Downloading {fname} …")
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(coco_dir / fname), url],
            check=True,
        )

    # Unzip
    for fname in urls:
        logging.info(f"Unpacking {fname} …")
        subprocess.run(
            ["unzip", "-q", str(coco_dir / fname), "-d", str(coco_dir)], check=True
        )

    # Clean up
    for fname in urls:
        (coco_dir / fname).unlink()

    logging.info(f"COCO dataset is ready at {coco_dir}")
    return coco_dir


def prepare_and_filter_coco_dataset(cfg, split_dir):
    """
    - Ensures COCO is available (cfg coco_root points at an existing archive,
      e.g. the read-only legacy one; otherwise downloaded to base_data_path).
    - Loads label_map, filters COCO JSONs to those IDs.
    - Optionally caps the mixed-in fraction (coco_max_frac) so COCO seasons
      the training instead of dominating it.
    - By default merges into the TRAIN side only (coco_train_only) — the val
      set stays pure garage frames so the yardstick never moves.
    - Symlinks the filtered images and merges the JSONs.
    """
    if cfg.get("coco_root"):
        coco_root = Path(cfg["coco_root"])  # read-only source is fine
    else:
        coco_root = ensure_coco_dataset(Path(cfg["base_data_path"]))
    label_map = load_label_map()  # {cat_id: name}
    keep_ids = set(label_map.keys())

    splits = ("train",) if cfg.get("coco_train_only", True) else ("train", "val")
    for split in splits:
        # 1) Load your existing split
        split_json = split_dir / f"{split}.json"
        data = load_coco_annotations(split_json)
        next_img_id = max(im["id"] for im in data["images"]) + 1
        next_ann_id = max((ann["id"] for ann in data["annotations"]), default=0) + 1

        # 2) Load & filter the COCO JSON
        coco_ann_file = coco_root / "annotations" / f"instances_{split}2017.json"
        coco_data = load_coco_annotations(coco_ann_file)
        # keep only our classes
        coco_data["categories"] = [
            c for c in coco_data["categories"] if c["id"] in keep_ids
        ]
        coco_anns = [
            a for a in coco_data["annotations"] if a["category_id"] in keep_ids
        ]
        valid_img_ids = {a["image_id"] for a in coco_anns}
        coco_imgs = [im for im in coco_data["images"] if im["id"] in valid_img_ids]

        # ratio cap: keep COCO at <= coco_max_frac of the mixed train set
        frac = cfg.get("coco_max_frac")
        if split == "train" and frac:
            n_ours = len(data["images"])
            cap = int(n_ours * frac / (1 - frac))
            if len(coco_imgs) > cap:
                rng = random.Random(cfg.get("seed", 42))
                coco_imgs = rng.sample(sorted(coco_imgs, key=lambda i: i["id"]), cap)
                kept = {im["id"] for im in coco_imgs}
                coco_anns = [a for a in coco_anns if a["image_id"] in kept]
                print(f"[INFO] coco capped to {cap} imgs "
                      f"({int(frac*100)}% of mixed train set)")

        # 3) Symlink & merge each COCO image/annotation
        img_id_map = {}
        img_dir = split_dir / f"{split}2017"
        for im in coco_imgs:
            orig_id = im["id"]
            filename = Path(im["file_name"]).name  # e.g. 000000123456.jpg
            new_name = f"coco_{orig_id:012d}_{filename}"
            src = coco_root / f"{split}2017" / filename
            dst = img_dir / new_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(src, dst)

            # append to your split JSON
            data["images"].append(
                {
                    "id": next_img_id,
                    "file_name": new_name,
                    "width": im.get("width", 640),
                    "height": im.get("height", 480),
                }
            )
            img_id_map[orig_id] = next_img_id
            next_img_id += 1

        for ann in coco_anns:
            data["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": img_id_map[ann["image_id"]],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann_id += 1

        # 4) Rewrite your split JSON
        save_coco_annotations(data, split_json)

        # 5) Update COCO‐style instances_*.json for evaluator
        ann_dir = split_dir / "annotations"
        shutil.copy(split_json, ann_dir / f"instances_{split}2017.json")

    logging.info(f"[INFO] Merged & filtered COCO into splits at {split_dir}")
    return coco_root


def merge_and_split_datasets(config):
    # absolute: symlink targets must not depend on the split dir's location
    base_path = Path(config["base_data_path"]).resolve()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = f"split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    split_dir = output_dir / split_name

    # COCO‐style train/val image folders
    train_img_dir = split_dir / "train2017"
    val_img_dir = split_dir / "val2017"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    train_json = {"images": [], "annotations": [], "categories": None}
    val_json = {"images": [], "annotations": [], "categories": None}

    next_img_id = 1
    next_ann_id = 1

    rng = random.Random(config.get("seed", 42))
    annotated_files = config["annotated_files"]
    if annotated_files == "auto":
        # unified store layout: <garage>/<sensor>/preannotations.coco.json
        annotated_files = sorted(
            str(p.relative_to(base_path))
            for p in base_path.glob("*/*/preannotations.coco.json"))
        print(f"[INFO] auto-discovered {len(annotated_files)} annotation files "
              f"under {base_path}")
    split_by = config.get("split_by", "frame")
    train_frac = config.get("train_val_split", 0.8)
    max_train = config.get("max_train_images")
    val_max = config.get("val_max_images")
    audited = load_audited_overrides(config)
    n_overridden = 0

    if split_by == "sensor-hash":
        # frozen split: membership depends only on (salt, garage/sensor), so
        # the val set never changes as the store grows; ordering by hash gives
        # a stable shuffle, so max_train_images subsets are NESTED prefixes
        salt = str(config.get("split_salt", "falcon-v1"))
        entries = sorted(
            ((rp, int.from_bytes(
                hashlib.md5(f"{salt}:{Path(rp).parent}".encode()).digest()[:8],
                "big")) for rp in annotated_files),
            key=lambda e: e[1])
    else:
        entries = [(rp, None) for rp in annotated_files]

    train_count = val_count = 0
    for rel_path, sensor_hash in entries:
        rel_path = Path(rel_path)
        ann_path = base_path / rel_path
        parts = rel_path.parts
        # legacy: <garage>/training_images/<sensor>/...; store: <garage>/<sensor>/...
        garage = parts[0]
        g_over = audited.get(garage, {})

        if split_by == "sensor-hash":
            is_val = (sensor_hash % 10_000) < round(10_000 * (1 - train_frac))
            if is_val and val_max and val_count >= val_max:
                continue
            if not is_val and max_train and train_count >= max_train:
                continue

        with open(ann_path) as f:
            coco = json.load(f)

        if train_json["categories"] is None:
            train_json["categories"] = coco["categories"]
            val_json["categories"] = coco["categories"]

        images = coco["images"]
        annotations = coco["annotations"]
        if split_by == "sensor-hash":
            train_imgs, val_imgs = ([], images) if is_val else (images, [])
        elif split_by == "sensor":
            # whole sensor goes to one side: no same-camera train/val leakage
            if rng.random() < train_frac:
                train_imgs, val_imgs = images, []
            else:
                train_imgs, val_imgs = [], images
        else:
            rng.shuffle(images)
            split_idx = int(len(images) * train_frac)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
        train_count += len(train_imgs)
        val_count += len(val_imgs)

        def process(img_list, target_json, img_dir):
            nonlocal next_img_id, next_ann_id, n_overridden
            # local map: annotations for THIS side's images only (the old
            # shared map leaked every file's train annotations into val.json)
            img_id_map = {}
            overridden = {}  # orig image id -> audited [(orig_cat, bbox), ...]
            for img in img_list:
                orig_id = img["id"]
                fname = os.path.basename(img["file_name"])
                new_name = f"{garage}_{fname}"
                src = base_path / img["file_name"]
                dst = img_dir / new_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(src, dst)

                target_json["images"].append(
                    {
                        "id": next_img_id,
                        "file_name": new_name,
                        "width": img.get("width", 640),
                        "height": img.get("height", 480),
                    }
                )
                img_id_map[orig_id] = next_img_id
                next_img_id += 1
                if fname in g_over:
                    overridden[orig_id] = g_over[fname]

            for ann in annotations:
                if ann["image_id"] in img_id_map and ann["image_id"] not in overridden:
                    # compute area from bbox
                    x, y, w, h = ann["bbox"]
                    area = w * h

                    target_json["annotations"].append(
                        {
                            "id": next_ann_id,
                            "image_id": img_id_map[ann["image_id"]],
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": area,
                            "iscrowd": ann.get("iscrowd", 0),
                        }
                    )
                    next_ann_id += 1

            # human-audited frames: drafts dropped above, audited boxes in
            for orig_id, audited_anns in overridden.items():
                n_overridden += 1
                for cat_id, bbox in audited_anns:
                    target_json["annotations"].append(
                        {
                            "id": next_ann_id,
                            "image_id": img_id_map[orig_id],
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )
                    next_ann_id += 1

        process(train_imgs, train_json, train_img_dir)
        process(val_imgs, val_json, val_img_dir)

    if audited:
        print(f"[INFO] {n_overridden} frames use human-audited labels "
              "(drafts overridden)")

    # Write out the split JSON manifests
    with open(split_dir / "train.json", "w") as f:
        json.dump(train_json, f, indent=2)
    with open(split_dir / "val.json", "w") as f:
        json.dump(val_json, f, indent=2)

    # Move them into COCO annotations/
    ann_dir = split_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(split_dir / "train.json", ann_dir / "instances_train2017.json")
    shutil.copy(split_dir / "val.json", ann_dir / "instances_val2017.json")

    print(f"[INFO] Output written to: {split_dir}")
    return {
        "train": split_dir / "train.json",
        "val": split_dir / "val.json",
        "train_images": train_img_dir,
        "val_images": val_img_dir,
    }


def run_training(cfg_path):
    cfg = load_config(cfg_path)
    out = merge_and_split_datasets(cfg)
    split_dir = out["train"].parent

    # 1) optionally mix in filtered COCO 2017 (skip for pure garage runs —
    #    e.g. SAM3 distillation, where 60k COCO images would drown the domain)
    if cfg.get("include_coco", True):
        prepare_and_filter_coco_dataset(cfg, split_dir)

    # 2) build & remap your label map → contiguous IDs
    orig_map = load_label_map()  # {orig_id: name}
    new_label_map, id_map = remap_label_map(orig_map)  # new_id→name, orig→new
    num_classes = len(new_label_map)

    # 3) rewrite the JSON files on‐disk so the dataset loader sees [0..num_classes)
    remap_split_jsons(split_dir, id_map, new_label_map)

    n_train = len(load_coco_annotations(split_dir / "train.json")["images"])
    n_val = len(load_coco_annotations(split_dir / "val.json")["images"])

    # epochs: auto = constant gradient-step budget, so sweep points at
    # different dataset sizes cost the same wall-clock and compare fairly
    if cfg.get("epochs") == "auto":
        budget = int(cfg.get("train_steps_budget", 25000))
        steps_per_epoch = max(1, n_train // int(cfg["batch_size"]))
        cfg["epochs"] = max(int(cfg.get("min_epochs", 2)),
                            round(budget / steps_per_epoch))
        print(f"[INFO] epochs=auto -> {cfg['epochs']} "
              f"({budget} step budget / {steps_per_epoch} steps per epoch)")

    # 4) now build your CLI args with the correct num_classes
    cli_args = [
        "--dataset",
        cfg.get("dataset", "coco"),
        "--model",
        cfg["model"],
        "--num-classes",
        str(num_classes),
        "--batch-size",
        str(cfg["batch_size"]),
        "--epochs",
        str(cfg["epochs"]),
        "--output",
        cfg["output_dir"],
        str(split_dir),  # this root now has remapped JSONs + symlinks
    ]
    if cfg.get("pretrained"):
        cli_args.append("--pretrained")
    if not cfg.get("pretrained_backbone"):
        cli_args.append("--no-pretrained-backbone")
    if cfg.get("amp", False):
        cli_args.append("--amp")
    if cfg.get("extra_args"):
        cli_args += cfg["extra_args"].split()

    # 5) run manifest: what went into this run (provenance for later comparison)
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "config": str(cfg_path),
        "model": cfg["model"],
        "num_classes": num_classes,
        "split_dir": str(split_dir),
        "split_by": cfg.get("split_by", "frame"),
        "seed": cfg.get("seed", 42),
        "train_images": n_train,
        "val_images": n_val,
        "include_coco": cfg.get("include_coco", True),
        "max_train_images": cfg.get("max_train_images"),
        "label_source": cfg.get("label_source", "unspecified"),
        "git_commit": git_rev,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
    }
    save_coco_annotations(manifest, split_dir / "run.json")
    print(f"[INFO] run manifest: {split_dir / 'run.json'} "
          f"({n_train} train / {n_val} val images)")

    # 6) invoke train
    args_ns = train.parser.parse_args(cli_args)
    train.main(args_ns)

    # 7) per-class/size metrics + final run.json into the new run dir
    runs = sorted(Path(cfg["output_dir"]).glob("train/*/summary.csv"),
                  key=lambda p: p.stat().st_mtime)
    if runs:
        import run_metrics
        try:
            run_metrics.compute(runs[-1].parent)
        except Exception as e:
            print(f"[metrics] FAILED ({e}) — backfill later with "
                  f"run_metrics.py {runs[-1].parent}")

    # 8) refresh the static dashboard
    try:
        import build_report
        subprocess.run([sys.executable, str(Path(__file__).parent / "build_report.py"),
                        "--output-dir", cfg["output_dir"]], check=False)
    except Exception as e:
        print(f"[report] skipped ({e})")

    # 9) auto-export artifacts from this run's best checkpoint
    if cfg.get("export_after_training", True):
        export_artifacts(cfg)


def export_artifacts(cfg):
    """Run all exporters against the newest best checkpoint so the full
    artifact set (TorchScript, f32/dynamic/int8 TFLite, drop-in package)
    is ready to pull without manual steps."""
    import subprocess

    out_dir = Path(cfg["output_dir"])
    ckpts = sorted(out_dir.glob("train/*/model_best.pth.tar"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print("[export] no best checkpoint found, skipping export")
        return
    ckpt = str(ckpts[-1])
    repo = Path(__file__).parent
    export_py = repo / "falcon-vision-od-export-venv" / "bin" / "python"
    env = {"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}
    # every tool gets --model explicitly: the ladder trains many architectures,
    # and the packager's config-file default is lite0 (weights would not load)
    model = cfg["model"]
    # native input size per architecture: the dropin is built at native res
    # (FW reads dims from the file; a 448 build on a 320 model = 2x latency)
    NATIVE = {"tf_efficientdet_lite0": 320, "tf_efficientdet_lite1": 384,
              "tf_efficientdet_lite2": 448, "tf_efficientdet_lite3": 512,
              "tf_efficientdet_lite3x": 640, "tf_efficientdet_lite4": 640,
              "tf_efficientdet_d0": 512, "tf_efficientdet_d1": 640,
              "tf_efficientdet_d2": 768}
    native = str(NATIVE.get(model, 448))
    jobs = [
        [sys.executable, str(repo / "generate_model_files.py"),
         "--checkpoint", ckpt, "--model", model],
    ]
    if export_py.exists():
        jobs.append([str(export_py), str(repo / "export_tflite.py"),
                     "--checkpoint", ckpt, "--model", model])
        if (repo / "package_dropin.py").exists():
            # no --validate flag: the packager builds then validates by default
            # (argparse would prefix-match --validate to --validate-only).
            # native-size dropins in BOTH quantizations: dynamic+f32 pair, then
            # full-int8 (garage-calibrated) — the complete bench test matrix
            jobs.append([str(export_py), str(repo / "package_dropin.py"),
                         "--checkpoint", ckpt, "--model", model,
                         "--input-size", native])
            jobs.append([str(export_py), str(repo / "package_dropin.py"),
                         "--checkpoint", ckpt, "--model", model,
                         "--input-size", native, "--quant", "int8",
                         "--calib-count", "256"])
    else:
        print("[export] export venv missing (run setup_export_venv.sh) — TFLite export skipped")
    for job in jobs:
        print("[export] running:", " ".join(job[1:2] + job[2:]))
        r = subprocess.run(job, cwd=repo, env=env)
        if r.returncode != 0:
            print(f"[export] FAILED (rc={r.returncode}): {job[1]} — continuing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_training(args.config)

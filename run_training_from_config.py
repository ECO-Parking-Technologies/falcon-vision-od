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
    - Ensures COCO is at base_data_path/coco_dataset.
    - Loads label_map, filters COCO JSONs to those IDs.
    - Symlinks the filtered images into split_dir/train2017/ and val2017/.
    - Merges them into split_dir/train.json & val.json (and updates annotations/instances_*.json).
    """
    base = Path(cfg["base_data_path"])
    coco_root = ensure_coco_dataset(base)
    label_map = load_label_map()  # {cat_id: name}
    keep_ids = set(label_map.keys())

    for split in ("train", "val"):
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
    base_path = Path(config["base_data_path"])
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

    for rel_path in config["annotated_files"]:
        rel_path = Path(rel_path)
        ann_path = base_path / rel_path
        garage, _, sensor = rel_path.parts[:3]

        with open(ann_path) as f:
            coco = json.load(f)

        if train_json["categories"] is None:
            train_json["categories"] = coco["categories"]
            val_json["categories"] = coco["categories"]

        images = coco["images"]
        annotations = coco["annotations"]
        random.shuffle(images)
        split_idx = int(len(images) * config.get("train_split", 0.8))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        img_id_map = {}

        def process(img_list, target_json, img_dir):
            nonlocal next_img_id, next_ann_id
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

            for ann in annotations:
                if ann["image_id"] in img_id_map:
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

        process(train_imgs, train_json, train_img_dir)
        process(val_imgs, val_json, val_img_dir)

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

    # 1) pull in & merge COCO
    prepare_and_filter_coco_dataset(cfg, split_dir)

    # 2) build & remap your label map → contiguous IDs
    orig_map = load_label_map()  # {orig_id: name}
    new_label_map, id_map = remap_label_map(orig_map)  # new_id→name, orig→new
    num_classes = len(new_label_map)

    # 3) rewrite the JSON files on‐disk so the dataset loader sees [0..num_classes)
    remap_split_jsons(split_dir, id_map, new_label_map)

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
    if cfg.get("amp", False):
        cli_args.append("--amp")
    if cfg.get("extra_args"):
        cli_args += cfg["extra_args"].split()

    # 5) invoke train
    args_ns = train.parser.parse_args(cli_args)
    train.main(args_ns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_training(args.config)

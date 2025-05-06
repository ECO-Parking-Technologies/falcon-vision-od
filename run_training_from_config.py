import json
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


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def save_coco_annotations(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


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

    # write out COCO JSON
    with open(split_dir / "train.json", "w") as f:
        json.dump(train_json, f, indent=2)
    with open(split_dir / "val.json", "w") as f:
        json.dump(val_json, f, indent=2)

    # now create the COCO‐style annotations folder
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
    out = merge_and_split_datasets(
        cfg
    )  # returns dict with 'train', 'val', 'train_images', 'val_images'
    split_dir = out[
        "train"
    ].parent  # this is the folder containing train.json & val.json

    # 1) Build the CLI args exactly how train.py expects them:
    cli_args = [
        "--dataset",
        cfg.get("dataset", "coco"),  # optional, defaults to coco
        "--model",
        cfg["model"],
        "--num-classes",
        str(cfg["num_classes"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--epochs",
        str(cfg["epochs"]),
        "--output",
        cfg["output_dir"],  # where to write checkpoints/logs
        # finally, the positional DIR where train.py will find train.json & val.json
        str(split_dir),
    ]
    if cfg.get("amp", False):
        cli_args.append("--amp")
    if cfg.get("extra_args"):
        cli_args += cfg["extra_args"].split()

    # 2) Parse into an argparse.Namespace
    args_ns = train.parser.parse_args(cli_args)

    # 3) Call the entry‐point
    train.main(args_ns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_training(args.config)

import json
import shutil
from pathlib import Path

import yaml


def load_label_map(config_path=None):
    """
    Loads a label map from config/label_map.yaml by default.

    Returns:
        label_map (dict): Maps int class ID to class name.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "label_map.yaml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return {int(k): v for k, v in data["label_map"].items()}


def remap_label_map(orig_label_map: dict) -> tuple[dict[int, str], dict[int, int]]:
    """
    Given an original mapping {orig_id: class_name}, return:
      - new_label_map: {new_id: class_name}  # new_id runs 0..N-1
      - id_map:         {orig_id: new_id}     # to translate annotations
    """
    # sort original IDs to give a stable new ordering
    sorted_orig = sorted(orig_label_map.keys())
    # build orig->new map
    id_map = {orig: new for new, orig in enumerate(sorted_orig)}
    # build new new_id->name map
    new_label_map = {new: orig_label_map[orig] for orig, new in id_map.items()}
    return new_label_map, id_map


def remap_split_jsons(
    split_dir: Path, id_map: dict[int, int], new_label_map: dict[int, str]
):
    """
    Rewrite split_dir/{train.json,val.json} and annotations/instances_*.json
    so that:
      - categories only include IDs in id_map
      - each annotation's category_id is remapped via id_map
    """
    for split in ("train", "val"):
        path = split_dir / f"{split}.json"
        data = json.loads(path.read_text())

        # 0) drop any annotations with unknown IDs
        data["annotations"] = [
            ann for ann in data["annotations"] if ann["category_id"] in id_map
        ]

        # 1) rewrite categories to only those in id_map
        data["categories"] = [
            {"id": new_id, "name": new_label_map[new_id]}
            for new_id in sorted(new_label_map)
        ]

        # 2) remap each annotation's ID
        for ann in data["annotations"]:
            ann["category_id"] = id_map[ann["category_id"]]

        # 3) save back out
        path.write_text(json.dumps(data, indent=2))

        # 4) update the coco‐style instances_*.json too
        ann_dir = split_dir / "annotations"
        shutil.copy(path, ann_dir / f"instances_{split}2017.json")

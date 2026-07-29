import json
import os
import random
import sys

# Ensure the root of the repo is in the path (so we can import from config/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from config.label_loader import load_label_map


def convert_detections_to_coco(
    label_map, garage_name, image_dir, detections, output_path, images_subset=None
):
    # Invert label map: name -> id
    name_to_id = {v: k for k, v in label_map.items() if v != "unlabeled"}

    images = []
    annotations = []
    ann_id = 1
    image_id = 1
    file_to_id = {}

    from pathlib import Path as _P
    sensor_dir = _P(image_dir)
    # legacy layout keeps 'training_images' in the CVAT-visible path; the
    # store layout is <garage>/<sensor>/<YYYY>/<MM>/<file>
    legacy = sensor_dir.parent.name == "training_images"
    prefix = os.path.join(garage_name, "training_images") if legacy else garage_name
    files = images_subset if images_subset is not None else sorted(
        str(p.relative_to(sensor_dir)) for p in sensor_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for fname in files:
        rel_path = os.path.join(prefix, sensor_dir.name, fname)
        file_to_id[fname] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": rel_path,
            }
        )
        image_id += 1

    for det in detections:
        file_name = det["image_file"]
        label_name = det["label"]
        label_idx = name_to_id.get(label_name)

        if label_idx is None or file_name not in file_to_id:
            print(
                f"[WARN] Unknown label {label_name} or file {file_name} not found, skipping"
            )
            continue

        x, y, w, h = det["bbox"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": file_to_id[file_name],
                "category_id": label_idx,
                "bbox": [x, y, w, h],
                "area": float(w) * float(h),
                "iscrowd": 0,
                "segmentation": [],
                "score": det.get("score", 1.0),
            }
        )
        ann_id += 1

    categories = [
        {"id": cid, "name": name}
        for cid, name in label_map.items()
        if cid != 0 and name != "unlabeled"
    ]

    output = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def generate_random_color():
    """Generate a random hex color string."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# image-level condition tags needing human judgment (roadmap Data Preparation);
# garage/sensor/time-of-day are NOT tags — they derive from filenames/manifest
CONDITION_TAGS = ["glare", "low-light", "rain", "snow", "dirty-lens", "obstruction"]


def _checkbox(name):
    return {"name": name, "mutable": True, "input_type": "checkbox",
            "default_value": "false", "values": ["false", "true"]}


def label_attributes(name):
    """Roadmap annotation spec (docs/road-map/Data Preparation.md):
    Occluded on everything; InEcoParkingSpot + InMotion on vehicles only."""
    attrs = [_checkbox("Occluded")]
    if name in VEHICLE_CLASSES:
        attrs = [_checkbox("InEcoParkingSpot"), _checkbox("InMotion")] + attrs
    return attrs


def export_cvat_labels(label_map, output_path):
    existing_labels = {}

    # Load existing file if it exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                existing_data = json.load(f)
                for label in existing_data:
                    existing_labels[label["name"]] = label
            except json.JSONDecodeError:
                print("[WARN] Existing labels file is invalid JSON, regenerating.")

    new_labels = []
    for idx, name in label_map.items():
        if name == "unlabeled":
            continue  # Skip unlabeled

        if name in existing_labels:
            # Reuse existing color
            color = existing_labels[name].get("color", generate_random_color())
        else:
            # New label, assign random color
            color = generate_random_color()

        new_labels.append(
            {
                "name": name,
                "id": idx,
                "color": color,
                "type": "rectangle",
                "attributes": label_attributes(name),
            }
        )

    for tag in CONDITION_TAGS:
        color = existing_labels.get(tag, {}).get("color", generate_random_color())
        new_labels.append({"name": tag, "id": None, "color": color,
                           "type": "tag", "attributes": []})

    # Only overwrite file if new_labels are different
    if existing_labels:
        unchanged = all(
            existing_labels.get(l["name"], {}).get("attributes") == l["attributes"]
            for l in new_labels
        ) and set(existing_labels) == {l["name"] for l in new_labels}
        if unchanged:
            print("[INFO] CVAT labels file already up-to-date.")
            return

    with open(output_path, "w") as f:
        json.dump(new_labels, f, indent=2)
        print(f"[INFO] Exported CVAT labels to {output_path}")

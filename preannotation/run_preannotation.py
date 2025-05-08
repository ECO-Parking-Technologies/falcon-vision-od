#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import torch
import yaml
from convert_to_cvat import *
from efficient_det_model import EfficientDetModel
from tqdm import tqdm
from utils import convert_detections, extract_sensor_and_camera

from config.label_loader import load_label_map, remap_label_map


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_and_convert_model(model_url, model_file):
    if os.path.exists(model_file):
        print(f"[INFO] Model already exists: {model_file}")
        return model_file

    print(f"[INFO] Downloading model from {model_url} …")
    temp_pth = model_url.split("/")[-1]
    urllib.request.urlretrieve(model_url, temp_pth)

    print(f"[INFO] Converting {temp_pth} → {model_file}")
    checkpoint = torch.load(temp_pth, map_location="cpu")
    torch.save(checkpoint, model_file)
    os.remove(temp_pth)
    return model_file


def run_inference_on_sensor(
    model,
    sensor_path,
    input_w,
    input_h,
    label_map,
    allowed_labels,
    threshold,
    visualize=None,
):
    # build a new→orig ID map for remapping
    _, id_map = remap_label_map(label_map)
    inv_id_map = {new: orig for orig, new in id_map.items()}
    total_time = 0.0
    img_count = 0

    images = sorted(f for f in os.listdir(sensor_path) if f.lower().endswith(".png"))
    all_detections = []

    for image_file in images:
        image_path = sensor_path / image_file
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not load image: {image_path}")
            continue

        orig_h, orig_w = image.shape[:2]
        inference_time, raw_dets = model.infer(image, (input_w, input_h))

        # running avg time
        total_time += inference_time
        img_count += 1
        avg_time = total_time / img_count

        # overlay avg inference time
        cv2.putText(
            image,
            f"Avg inf: {avg_time:.3f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        # remap model’s class IDs → original COCO IDs
        if raw_dets.size:
            remapped = raw_dets.copy()
            for i in range(remapped.shape[0]):
                cls_new = int(remapped[i, 5])
                remapped[i, 5] = inv_id_map.get(cls_new, -1)
            raw_dets = remapped

        # filter by label & threshold
        detections = convert_detections(
            raw_dets,
            label_map=label_map,
            allowed_labels=allowed_labels,
            threshold=threshold,
        )

        # scale boxes and collect
        scale_x = orig_w / float(input_w)
        scale_y = orig_h / float(input_h)
        for det in detections:
            x, y, w_box, h_box = det["bbox"]
            x1, y1 = x * scale_x, y * scale_y
            w1, h1 = w_box * scale_x, h_box * scale_y
            det["bbox"] = [x1, y1, w1, h1]
            det["image_file"] = image_file
            all_detections.append(det)

            if visualize:
                cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x1 + w1), int(y1 + h1)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"{det['label']} {det['score']:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        if visualize:
            cv2.imshow("annotated", image)
            if cv2.waitKey(visualize) == 27:
                break

    if visualize:
        cv2.destroyAllWindows()

    return all_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--visualize", type=int, default=0, help="Visualization delay in ms (0 to disable)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run inference without writing preannotations files"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_pretrained = cfg.get("use_pretrained_model", False)
    model_type = cfg["model_type"]
    input_w, input_h = cfg["efficientdet_models"][model_type]["input_size"]

    # decide where to get weights from
    if use_pretrained:
        model_url = cfg["efficientdet_models"][model_type].get("url")
        if not model_url:
            sys.exit(f"❌ No URL specified for pretrained model of type '{model_type}'")
        model_path = Path(download_and_convert_model(model_url, f"{model_type}.pt"))

        # let EfficientDetModel default to its own COCO class count (90)
        num_classes = None
        print(f"[INFO] Using pretrained COCO weights ({model_type}), num_classes=90")
    else:
        model_file_cfg = cfg.get("model_file", None)
        if not model_file_cfg:
            sys.exit("❌ 'model_file' must be specified when use_pretrained_model is False")
        model_path = Path(model_file_cfg)
        if not model_path.exists():
            sys.exit(f"❌ model_file not found: {model_file_cfg}")

        # use your custom class count (allowed_labels + background)
        allowed_labels = cfg.get("allowed_labels", list(load_label_map().keys()))
        num_classes = len(allowed_labels) + 1
        print(f"[INFO] Using custom model '{model_path.name}', num_classes={num_classes}")

    print(f"[INFO] Using model path: {model_path}")

    # load label map & pick your classes
    label_map = load_label_map()
    allowed_labels = cfg.get("allowed_labels", list(label_map.keys()))
    threshold = cfg.get("threshold", 0.25)

    # export CVAT label config
    labels_output_path = Path(cfg["base_data_path"]) / "cvat_labels.json"
    export_cvat_labels(label_map, labels_output_path)

    # build the model
    model = EfficientDetModel(
        model_path=str(model_path),
        model_name=model_type,
        num_classes=num_classes,
    )

    # iterate garages/sensors
    for garage in cfg["garages"]:
        garage_dir = Path(cfg["base_data_path"]) / garage / "training_images"
        if not garage_dir.exists():
            continue

        sensor_list = sorted(p for p in garage_dir.iterdir() if p.is_dir())
        with tqdm(total=len(sensor_list), desc=f"Garage: {garage}", unit="sensor") as pbar:
            for sensor_dir in sensor_list:
                output_json = sensor_dir / "preannotations.coco.json"
                detections = run_inference_on_sensor(
                    model,
                    sensor_dir,
                    input_w,
                    input_h,
                    label_map,
                    allowed_labels,
                    threshold,
                    visualize=args.visualize,
                )

                if args.dry_run:
                    print(f"[DRY-RUN] Skipping write of {output_json} ({len(detections)} detections)")
                else:
                    convert_detections_to_coco(
                        label_map, garage, sensor_dir, detections, str(output_json)
                    )

                pbar.update(1)


if __name__ == "__main__":
    main()

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import torch
import yaml
from convert_to_cvat import *
from efficient_det_model import EfficientDetModel
from tqdm import tqdm
from utils import convert_detections, extract_sensor_and_camera

from config.label_loader import load_label_map


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_and_convert_model(model_url, model_file):
    if os.path.exists(model_file):
        print(f"[INFO] Model already exists: {model_file}")
        return model_file

    print(f"[INFO] Downloading model from {model_url} ...")
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
    images = sorted([f for f in os.listdir(sensor_path) if f.lower().endswith(".png")])
    all_detections = []

    for image_file in images:
        image_path = os.path.join(sensor_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Could not load image: {image_path}")
            continue

        orig_h, orig_w = image.shape[:2]
        inference_time, raw_detections = model.infer(image, (input_w, input_h))
        detections = convert_detections(
            raw_detections,
            label_map=label_map,
            allowed_labels=allowed_labels,
            threshold=threshold,
        )

        scale_x = orig_w / float(input_w)
        scale_y = orig_h / float(input_h)

        for det in detections:
            if not isinstance(det, dict) or "bbox" not in det:
                continue

            x, y, w_box, h_box = det["bbox"]
            x *= scale_x
            w_box *= scale_x
            y *= scale_y
            h_box *= scale_y

            det["bbox"] = [x, y, w_box, h_box]
            det["image_file"] = image_file
            all_detections.append(det)

            if visualize:
                cv2.rectangle(
                    image,
                    (int(x), int(y)),
                    (int(x + w_box), int(y + h_box)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"{det['label']} {det['score']:.2f}",
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
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
    parser.add_argument("--visualize", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    label_map = load_label_map()
    allowed_labels = cfg.get("allowed_labels", label_map.values())
    threshold = cfg.get("threshold", 0.25)
    model_type = cfg["model_type"]
    model_url = cfg["efficientdet_models"][model_type]["url"]
    input_w, input_h = cfg["efficientdet_models"][model_type]["input_size"]
    model_file = f"{model_type}.pt"

    # Export CVAT labels file at the base path
    labels_output_path = os.path.join(cfg["base_data_path"], "cvat_labels.json")
    export_cvat_labels(label_map, labels_output_path)

    model_file = download_and_convert_model(model_url, model_file)
    model = EfficientDetModel(model_file)

    for garage in cfg["garages"]:
        garage_dir = Path(cfg["base_data_path"]) / garage / "training_images"
        if not garage_dir.exists():
            continue

        sensor_list = sorted(os.listdir(garage_dir))
        with tqdm(
            total=len(sensor_list), desc=f"Garage: {garage}", unit="sensor"
        ) as pbar:
            for sensor in sensor_list:
                sensor_dir = garage_dir / sensor
                if not sensor_dir.is_dir():
                    pbar.update(1)
                    continue

                output_json = os.path.join(sensor_dir, "preannotations.coco.json")
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
                convert_detections_to_coco(
                    label_map, garage, sensor_dir, detections, output_json
                )

                pbar.update(1)


if __name__ == "__main__":
    main()

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
from utils import convert_detections, extract_sensor_and_camera

from config.label_loader import load_label_map, remap_label_map

import logging
import time as _time
from rich import box as _box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()
log = logging.getLogger("preann")


def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file)], force=True)


class RunStats:
    def __init__(self):
        self.rows = {}   # garage -> dict
        self.current = ""
        self.t0 = _time.time()

    def row(self, g):
        return self.rows.setdefault(g, dict(sensors=0, done=0, frames=0,
                                            boxes=0, skipped=0))

    def table(self):
        t = Table(title="Grounding DINO preannotation (one row per garage)",
                  box=_box.SIMPLE_HEAD)
        for col, style in (("garage", "cyan"), ("sensors", "cyan"),
                           ("frames", "green"), ("boxes", "green"),
                           ("skipped", "dim")):
            t.add_column(col, style=style,
                         justify="left" if col == "garage" else "right")
        tot = dict(sensors=0, done=0, frames=0, boxes=0, skipped=0)
        for g in sorted(self.rows):
            r = self.rows[g]
            pct = 100 * (r["done"] + r["skipped"]) // r["sensors"] if r["sensors"] else 0
            t.add_row(g, f"{r['done'] + r['skipped']}/{r['sensors']} ({pct}%)",
                      str(r["frames"]), str(r["boxes"]), str(r["skipped"]))
            for k in tot:
                tot[k] += r[k]
        if self.rows:
            t.add_section()
            tpct = (100 * (tot["done"] + tot["skipped"]) // tot["sensors"]
                    if tot["sensors"] else 0)
            t.add_row("OVERALL", f"{tot['done'] + tot['skipped']}/{tot['sensors']} ({tpct}%)",
                      str(tot["frames"]), str(tot["boxes"]), str(tot["skipped"]))
        rate = tot["frames"] / max(1e-9, _time.time() - self.t0)
        cap = self.current
        if tot["frames"]:
            cap += f"   ·   {rate:.1f} frames/s"
        if cap:
            t.caption = cap
        return t


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_and_convert_model(model_url, model_file):
    if os.path.exists(model_file):
        log.info("model already exists: %s", model_file)
        return model_file

    log.info("downloading model from %s", model_url)
    temp_pth = model_url.split("/")[-1]
    urllib.request.urlretrieve(model_url, temp_pth)

    log.info("converting %s -> %s", temp_pth, model_file)
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
    crop_cfg=None,
    pretrained_coco=False,
    garage=None,
    queue=None,
):
    # Model detections carry 1-based class ids (background = 0), which for
    # custom models match the 1-based contiguous ids from remap_label_map.
    # Pretrained COCO-90 models already emit original COCO category ids.
    if pretrained_coco:
        inv_id_map = None
    else:
        _, id_map = remap_label_map(label_map)
        inv_id_map = {new: orig for orig, new in id_map.items()}
    total_time = 0.0
    img_count = 0

    # recursive: supports both the legacy flat layout and the store's
    # <sensor>/<YYYY>/<MM>/ nesting; jpg + png
    images = sorted(
        str(p.relative_to(sensor_path)) for p in Path(sensor_path).rglob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if queue is not None:
        sensor = Path(sensor_path).name
        images = [f for f in images if f"{garage}/{sensor}/{f}" in queue]
        if not images:
            return [], []
    all_detections = []

    for image_file in images:
        full_img = cv2.imread(str(sensor_path / image_file))
        if full_img is None:
            log.warning("could not load image: %s", image_file)
            continue

        # 1) Crop the frame if requested
        if crop_cfg:
            x0 = crop_cfg.get("x", 0)
            y0 = crop_cfg.get("y", 0)
            cw = crop_cfg.get("width", full_img.shape[1] - x0)
            ch = crop_cfg.get("height", full_img.shape[0] - y0)
            img = full_img[y0 : y0 + ch, x0 : x0 + cw]
        else:
            img = full_img
            x0 = y0 = 0
            cw, ch = img.shape[1], img.shape[0]

        # 2) Run inference
        inference_time, raw_dets = model.infer(img, (input_w, input_h))

        # 3) Compute running average time
        total_time += inference_time
        img_count += 1
        avg_time = total_time / img_count

        # 4) Overlay timing on the cropped view
        disp = img.copy()
        cv2.putText(
            disp,
            f"Avg inf: {avg_time:.3f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        # 5) Remap the model’s class IDs back to your original COCO IDs
        if inv_id_map is not None and raw_dets.size:
            rem = raw_dets.copy()
            for i in range(rem.shape[0]):
                rem[i, 5] = inv_id_map.get(int(rem[i, 5]), -1)
            raw_dets = rem

        # print(f"[DEBUG] top scores:", sorted(raw_dets[:, 4], reverse=True)[:5])

        # 6) Filter by score + allowed labels
        dets = convert_detections(
            raw_dets,
            label_map=label_map,
            allowed_labels=allowed_labels,
            threshold=threshold,
        )

        # 7) Rescale and reproject boxes into full-frame coords
        sx = cw / float(input_w)
        sy = ch / float(input_h)
        for det in dets:
            x, y, w_box, h_box = det["bbox"]
            x1 = x * sx + x0
            y1 = y * sy + y0
            w1 = w_box * sx
            h1 = h_box * sy
            det["bbox"] = [x1, y1, w1, h1]
            det["image_file"] = image_file
            all_detections.append(det)

            if visualize:
                # draw on the cropped display
                cv2.rectangle(
                    disp,
                    (int(x1 - x0), int(y1 - y0)),
                    (int(x1 - x0 + w1), int(y1 - y0 + h1)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    disp,
                    f"{det['label']} {det['score']:.2f}",
                    (int(x1 - x0), int(y1 - y0) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        if visualize:
            cv2.imshow("annotated", disp)
            if cv2.waitKey(visualize) == 27:
                break

    if visualize:
        cv2.destroyAllWindows()

    return all_detections, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Visualization delay in ms (0 to disable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference without writing preannotations files",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="ignore queue_file: preannotate every frame in the store",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="incremental resume: only process frames not already in a sensor's "
             "preannotations.coco.json, merging new results in",
    )
    parser.add_argument(
        "--no-class-filtering",
        action="store_true",
        help="Disable filtering of allowed labels (use all labels in label_map)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # resolve relative config paths against the repo root (script runs from preannotation/)
    _root = Path(__file__).resolve().parent.parent
    for _k in ("base_data_path", "queue_file", "model_file"):
        if cfg.get(_k) and not Path(cfg[_k]).is_absolute():
            cfg[_k] = str(_root / cfg[_k])
    log_file = _root / "data" / "preannotation.log"
    setup_logging(log_file)
    console.print(Panel(f"Detailed log: [bold]{log_file}[/bold]",
                        title="Grounding DINO preannotation", border_style="cyan"))

    use_pretrained = cfg.get("use_pretrained_model", False)
    model_type = cfg["model_type"]
    input_w, input_h = cfg["efficientdet_models"][model_type]["input_size"]

    # decide where to get weights from
    if model_type == "grounding_dino":
        pass  # open-vocabulary backend, no local weights file
    elif use_pretrained:
        model_url = cfg["efficientdet_models"][model_type].get("url")
        if not model_url:
            sys.exit(f"❌ No URL specified for pretrained model of type '{model_type}'")
        model_path = Path(download_and_convert_model(model_url, f"{model_type}.pt"))

        # let EfficientDetModel default to its own COCO class count (90)
        num_classes = None
        log.info("using pretrained COCO weights (%s)", model_type)
    else:
        model_file_cfg = cfg.get("model_file", None)
        if not model_file_cfg:
            sys.exit(
                "❌ 'model_file' must be specified when use_pretrained_model is False"
            )
        model_path = Path(model_file_cfg)
        if not model_path.exists():
            sys.exit(f"❌ model_file not found: {model_file_cfg}")

        # class count must match the trained model: all classes in the label map
        # (allowed_labels only filters the export, it does not change the model)
        num_classes = len(load_label_map())
        log.info("using custom model %s num_classes=%s", model_path.name, num_classes)

    if model_type == "grounding_dino":
        num_classes = None
        console.print("backend: [cyan]Grounding DINO[/cyan] (open-vocabulary)")
        log.info("backend: grounding dino")
    else:
        log.info("model path: %s", model_path)

    # load label map & pick your classes
    label_map = load_label_map()
    if args.no_class_filtering:
        allowed_labels = None
    else:
        allowed_labels = cfg.get("allowed_labels", list(label_map.keys()))
    threshold = cfg.get("threshold", 0.25)

    # export CVAT label config
    # tracked in-repo: the CVAT label spec is configuration, not data
    labels_output_path = _root / "config" / "cvat_labels.json"
    export_cvat_labels(label_map, labels_output_path)

    # build the model
    if model_type == "grounding_dino":
        from grounding_dino_model import GroundingDinoModel
        g = cfg.get("grounding_dino", {})
        model = GroundingDinoModel(
            model_id=g.get("model_id", "IDEA-Research/grounding-dino-base"),
            box_threshold=g.get("box_threshold", 0.25),
            text_threshold=g.get("text_threshold", 0.20),
        )
    else:
        model = EfficientDetModel(
            model_path=str(model_path),
            model_name=model_type,
            num_classes=num_classes,
            pretrained_backbone=cfg.get("pretrained_backbone", True),
        )

    # iterate garages/sensors
    garages = cfg.get("garages") or "all"
    if garages == "all":  # auto-discover from the store layout
        base = Path(cfg["base_data_path"])
        garages = sorted(p.name for p in base.iterdir() if p.is_dir())
        log.info("auto-discovered %d garages under %s", len(garages), base)

    # optional annotation queue: only process listed frames
    queue = None
    qf = None if args.all_frames else cfg.get("queue_file")
    if qf:
        import json as _json
        qdata = _json.loads(Path(qf).read_text())
        queue = {p for paths in qdata.values() for p in paths}
        console.print(f"queue: [green]{len(queue)}[/green] frames")
        log.info("queue: %d frames from %s", len(queue), qf)

    stats = RunStats()
    # pre-populate rows so every garage is visible with its sensor total
    garage_dirs = {}
    for garage in garages:
        gd = Path(cfg["base_data_path"]) / garage / "training_images"
        if not gd.exists():
            gd = Path(cfg["base_data_path"]) / garage
        if not gd.exists():
            continue
        garage_dirs[garage] = gd
        stats.row(garage)["sensors"] = sum(1 for p in gd.iterdir() if p.is_dir())

    with Live(stats.table(), console=console, refresh_per_second=2) as live:
        for garage, garage_dir in garage_dirs.items():
            sensor_list = sorted(p for p in garage_dir.iterdir() if p.is_dir())
            for sensor_dir in sensor_list:
                stats.current = f"{garage}/{sensor_dir.name}"
                live.update(stats.table())
                output_json = sensor_dir / "preannotations.coco.json"
                sensor_queue = queue
                existing_coco = None
                if args.skip_existing and output_json.exists():
                    import json as _json
                    existing_coco = _json.loads(output_json.read_text())
                    done = {im["file_name"] for im in existing_coco["images"]}
                    prefix = f"{garage}/{sensor_dir.name}/"
                    all_frames = {
                        f"{prefix}{p.relative_to(sensor_dir)}"
                        for p in sensor_dir.rglob("*")
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                    }
                    new = {f for f in all_frames if f not in done}
                    if queue is not None:
                        new &= queue
                    if not new:
                        stats.row(garage)["skipped"] += 1
                        live.update(stats.table())
                        continue
                    sensor_queue = new
                detections, processed = run_inference_on_sensor(
                    model,
                    sensor_dir,
                    input_w,
                    input_h,
                    label_map,
                    allowed_labels,
                    threshold,
                    visualize=args.visualize,
                    crop_cfg=cfg.get("crop", None),
                    pretrained_coco=use_pretrained or model_type == "grounding_dino",
                    garage=garage,
                    queue=sensor_queue,
                )

                r = stats.row(garage)
                if not processed:
                    r["skipped"] += 1
                    live.update(stats.table())
                    continue
                r["done"] += 1
                r["frames"] += len(processed)
                r["boxes"] += len(detections)
                log.info("%s/%s: %d frames %d boxes", garage, sensor_dir.name,
                         len(processed), len(detections))
                if args.dry_run:
                    log.info("dry-run: skipping write of %s", output_json)
                else:
                    convert_detections_to_coco(
                        label_map, garage, sensor_dir, detections, str(output_json),
                        images_subset=processed,
                    )
                    if existing_coco is not None:
                        import json as _json
                        fresh = _json.loads(Path(output_json).read_text())
                        img_off = max((im["id"] for im in existing_coco["images"]),
                                      default=0)
                        ann_off = max((a["id"] for a in existing_coco["annotations"]),
                                      default=0)
                        for im in fresh["images"]:
                            im["id"] += img_off
                        for a in fresh["annotations"]:
                            a["id"] += ann_off
                            a["image_id"] += img_off
                        existing_coco["images"] += fresh["images"]
                        existing_coco["annotations"] += fresh["annotations"]
                        Path(output_json).write_text(_json.dumps(existing_coco, indent=2))
                live.update(stats.table())

    tot_frames = sum(r["frames"] for r in stats.rows.values())
    tot_boxes = sum(r["boxes"] for r in stats.rows.values())
    console.print(Panel(f"[green]{tot_frames}[/green] frames drafted · "
                        f"{tot_boxes} boxes\nlog: {log_file}",
                        title="Done", border_style="green"))
    log.info("run complete: %d frames %d boxes", tot_frames, tot_boxes)


if __name__ == "__main__":
    main()

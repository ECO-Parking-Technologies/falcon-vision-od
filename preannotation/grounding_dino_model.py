#!/usr/bin/env python3
"""Grounding DINO (HuggingFace transformers) preannotation backend.

Open-vocabulary detector prompted with our class names; drop-in alternative to
EfficientDetModel for run_preannotation.py. Detections carry ORIGINAL COCO ids
(1=person, 2=bicycle, 3=car, 4=motorcycle, 6=bus, 8=truck) — use the runner's
pretrained_coco=True path (no contiguous-id remap).

Proven recipe from the gsam2-sandbox PoC: ~260 ms/frame on the RTX 3090,
weights ~892 MB (Apache-2.0), downloaded/cached on first run.
"""
import time

import cv2
import numpy as np
import torch

MODEL_ID = "IDEA-Research/grounding-dino-base"
PROMPT = "a car. a truck. a bus. a motorcycle. a bicycle. a person."
# order matters: match specific words before "car" (which appears in no other)
KEYWORD_TO_CAT = [("motorcycle", 4), ("bicycle", 2), ("person", 1),
                  ("truck", 8), ("bus", 6), ("car", 3)]
MAX_BOX_AREA_FRAC = 0.80  # drop near-full-frame "the scene is a car" boxes
NMS_IOU = 0.5


class GroundingDinoModel:
    def __init__(self, model_id: str = MODEL_ID, prompt: str = PROMPT,
                 box_threshold: float = 0.25, text_threshold: float = 0.20):
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device).eval()
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @staticmethod
    def _label_to_cat(text_label: str):
        for kw, cat in KEYWORD_TO_CAT:
            if kw in text_label:
                return cat
        return -1

    def infer(self, image: np.ndarray, input_size=(640, 640), debug=False):
        """Run at native resolution; return (elapsed, dets [N,6]) with boxes
        scaled into input_size space (the runner rescales to crop dims)."""
        from PIL import Image
        from torchvision.ops import nms

        h, w = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.prompt,
                                return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(h, w)])[0]
        elapsed = time.time() - t0

        rows = []
        for box, score, label in zip(res["boxes"], res["scores"],
                                     res.get("text_labels", res.get("labels", []))):
            cat = self._label_to_cat(str(label))
            x1, y1, x2, y2 = [float(v) for v in box]
            if cat < 0 or (x2 - x1) * (y2 - y1) > MAX_BOX_AREA_FRAC * w * h:
                continue
            rows.append([x1, y1, x2, y2, float(score), float(cat)])

        if rows:
            t = torch.tensor(rows)
            keep = nms(t[:, :4], t[:, 4], NMS_IOU)
            rows = t[keep].tolist()
            # scale native-pixel boxes into input_size space for the runner
            sx, sy = input_size[0] / w, input_size[1] / h
            for r in rows:
                r[0] *= sx; r[2] *= sx; r[1] *= sy; r[3] *= sy
        dets = np.array(rows, dtype=np.float32).reshape(-1, 6)
        if debug:
            for x1, y1, x2, y2, s, c in dets[:10]:
                print(f"  gdino cat={int(c)} score={s:.2f} "
                      f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        return elapsed, dets

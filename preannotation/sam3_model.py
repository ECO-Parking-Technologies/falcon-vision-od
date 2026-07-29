#!/usr/bin/env python3
"""SAM 3 (facebook/sam3) preannotation backend.

Open-vocabulary concept detector; drop-in sibling of GroundingDinoModel with
the same infer() contract. Detections carry ORIGINAL COCO ids (1=person,
2=bicycle, 3=car, 4=motorcycle, 6=bus, 8=truck) — use the runner's
pretrained_coco=True path (no contiguous-id remap).

Sandbox verdict vs Grounding DINO (52 frames, 13 garages): +123 confident
boxes GDINO missed (mostly distant cars, median ~29x29 px), zero
wheel->motorcycle false positives, truck/car labels match our policy better.
~1.4 s/frame on the RTX 3090.

Weights are GATED: request access (accept the SAM license) at
huggingface.co/facebook/sam3 first — approval comes by email. On the first
run this backend prompts for a HF read token interactively; the token lives
ONLY in this process's memory (never written to disk, logs, or shell
history). Once the weights are cached no token is needed again.
Requires transformers >= 5 (in requirements.txt).
"""
import getpass
import time

import cv2
import numpy as np
import torch

MODEL_ID = "facebook/sam3"
# one text concept per class -> original COCO category id
PROMPTS = [("person", 1), ("bicycle", 2), ("car", 3),
           ("motorcycle", 4), ("bus", 6), ("truck", 8)]
MAX_BOX_AREA_FRAC = 0.80  # drop near-full-frame boxes (safety, rare)
NMS_IOU = 0.6  # per-class dedup: SAM3 occasionally doubles boxes in fog/glare


class Sam3DraftModel:
    def __init__(self, model_id: str = MODEL_ID, score_threshold: float = 0.5):
        from transformers import Sam3Model, Sam3Processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._token = None
        self.processor = self._from_pretrained(Sam3Processor, model_id)
        self.model = self._from_pretrained(
            Sam3Model, model_id, dtype=torch.bfloat16).to(self.device).eval()
        self.score_threshold = score_threshold

    def _from_pretrained(self, cls, model_id, **kw):
        """Cache first, then gated download with an interactively prompted
        read token (kept in RAM only, per project credential policy)."""
        try:
            return cls.from_pretrained(model_id, local_files_only=True, **kw)
        except Exception:
            pass
        try:
            return cls.from_pretrained(model_id, token=self._token, **kw)
        except Exception as e:
            if self._token is not None:
                raise
            print(f"{model_id} is gated and not cached yet ({type(e).__name__}).\n"
                  f"Request access once at https://huggingface.co/{model_id}, then\n"
                  "paste a HF read token (input hidden, kept in RAM only):")
            self._token = getpass.getpass("HF token: ")
            return cls.from_pretrained(model_id, token=self._token, **kw)

    def infer(self, image: np.ndarray, input_size=(640, 640), debug=False):
        """Run all class prompts as one batch at native resolution; return
        (elapsed, dets [N,6]) with boxes scaled into input_size space (the
        runner rescales to crop dims)."""
        from PIL import Image
        from torchvision.ops import nms

        h, w = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        texts = [p for p, _ in PROMPTS]
        inputs = self.processor(images=[pil] * len(texts), text=texts,
                                return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold,
            target_sizes=inputs.get("original_sizes").tolist())
        elapsed = time.time() - t0

        rows = []
        for (_, cat), res in zip(PROMPTS, results):
            cls_rows = []
            for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
                x1, y1, x2, y2 = (float(v) for v in box)
                if (x2 - x1) * (y2 - y1) > MAX_BOX_AREA_FRAC * w * h:
                    continue
                cls_rows.append([x1, y1, x2, y2, float(score), float(cat)])
            if cls_rows:
                t = torch.tensor(cls_rows)
                keep = nms(t[:, :4], t[:, 4], NMS_IOU)
                rows += t[keep].tolist()

        sx, sy = input_size[0] / w, input_size[1] / h
        for r in rows:
            r[0] *= sx; r[2] *= sx; r[1] *= sy; r[3] *= sy
        dets = np.array(rows, dtype=np.float32).reshape(-1, 6)
        if debug:
            for x1, y1, x2, y2, s, c in dets[:10]:
                print(f"  sam3 cat={int(c)} score={s:.2f} "
                      f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        return elapsed, dets

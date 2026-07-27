#!/usr/bin/env python3
import time
from pathlib import Path
import cv2
import numpy as np
import torch

from config.label_loader import load_label_map
from effdet import DetBenchPredict, create_model

class EfficientDetModel:
    def __init__(
        self,
        model_path: str,
        model_name: str = None,
        num_classes: int = None,
        pretrained_backbone: bool = True,
    ):
        """
        EfficientDet inference wrapper.

        :param model_path: Path to a TorchScript trace (.pt) or a checkpoint/state_dict
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If it's a TorchScript file (.pt) load directly
        if self.model_path.suffix == '.pt':
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model = self.model.eval()
        else:
            # Otherwise load via create_model + state_dict
            if model_name is None:
                model_name = self.model_path.stem
            if num_classes is None:
                num_classes = len(load_label_map())
            base = create_model(
                model_name,
                bench_task="predict",
                num_classes=num_classes,
                pretrained_backbone=pretrained_backbone,
                pretrained=False,
            )
            sd = torch.load(self.model_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            sd = {k.replace("model.", ""): v for k, v in sd.items()}
            missing, unexpected = base.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")
            self.model = DetBenchPredict(base).to(self.device).eval()

    def preprocess(self, image: np.ndarray, input_size=(640, 640)):
        img = cv2.resize(image, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return t.unsqueeze(0).to(self.device)

    def infer(self, image: np.ndarray, input_size=(640, 640), debug=True):
        inp = self.preprocess(image, input_size)
        t0 = time.time()

        with torch.no_grad():
            out = self.model(inp)
        dets = out[0].cpu().numpy()

        elapsed = time.time() - t0

        if debug:
            # 1) Print overall stats
            print(f"[DEBUG] Raw detections: {dets.shape[0]} boxes")

            # 2) Sort descending by score
            order = dets[:, 4].argsort()[::-1]
            labels = load_label_map()

            # 3) Dump top-10
            for i in order[:10]:
                y1, x1, y2, x2, score, cls = dets[int(i)]
                cls_name = labels[int(cls)] if int(cls) < len(labels) else str(int(cls))
                print(f"  • #{i:02d}: {cls_name:<10s} score={score:.3f} "
                      f"box=[{int(x1)},{int(y1)}→{int(x2)},{int(y2)}]")

        return elapsed, dets

    def annotate(self, image: np.ndarray, detections, threshold: float = 0.25):
        img = image.copy()
        for ymin, xmin, ymax, xmax, score, cls in detections:
            if score < threshold:
                continue
            x, y, w, h = int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{int(cls)}:{score:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return img

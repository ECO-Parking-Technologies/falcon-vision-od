#!/usr/bin/env python3
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config.label_loader import load_label_map, remap_label_map
from effdet import DetBenchPredict, create_model

# effdet DetBenchPredict output rows: [x_min, y_min, x_max, y_max, score, class]
# in network-input pixel coords, where class is 1-based (background = 0).

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

        # A .pt may be a TorchScript trace or a plain checkpoint/state_dict;
        # try TorchScript first and fall back to state-dict loading.
        loaded = None
        if self.model_path.suffix == '.pt':
            try:
                loaded = torch.jit.load(str(self.model_path), map_location=self.device)
            except RuntimeError:
                loaded = None
        if loaded is not None:
            self.model = loaded.eval()
        else:
            # Otherwise load via create_model + state_dict
            if model_name is None:
                model_name = self.model_path.stem
            if num_classes is None:
                num_classes = len(load_label_map())
            # create_model(bench_task="predict") already returns a DetBenchPredict
            # whose raw network lives at .model
            bench = create_model(
                model_name,
                bench_task="predict",
                num_classes=num_classes,
                pretrained_backbone=pretrained_backbone,
                pretrained=False,
            )
            # weights_only=False: training checkpoints embed argparse.Namespace (trusted, local)
            sd = torch.load(self.model_path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            # training checkpoints are bench-level ("model."-prefixed); release
            # checkpoints are raw-network keys — normalize to raw and load there
            sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
            sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
            missing, unexpected = bench.model.load_state_dict(sd, strict=False)
            if missing:
                raise RuntimeError(
                    f"checkpoint is missing {len(missing)} network keys "
                    f"(e.g. {missing[:3]}) — wrong model/num_classes?"
                )
            if unexpected:
                print(f"[INFO] ignored {len(unexpected)} non-network checkpoint keys "
                      f"(e.g. {unexpected[:3]})")
            self.model = bench.to(self.device).eval()

    def preprocess(self, image: np.ndarray, input_size=(640, 640)):
        img = cv2.resize(image, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Same normalization as the training dataloader (effdet PrefetchLoader):
        # ImageNet mean/std applied to 0-255 pixel values.
        mean = torch.tensor([m * 255 for m in IMAGENET_DEFAULT_MEAN]).view(3, 1, 1)
        std = torch.tensor([s * 255 for s in IMAGENET_DEFAULT_STD]).view(3, 1, 1)
        t = torch.from_numpy(img).permute(2, 0, 1).float().sub_(mean).div_(std)
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
            contiguous_labels, _ = remap_label_map(load_label_map())

            # 3) Dump top-10 (class is 1-based; only valid for custom-trained models)
            for i in order[:10]:
                x1, y1, x2, y2, score, cls = dets[int(i)]
                cls_name = contiguous_labels.get(int(cls), str(int(cls)))
                print(f"  • #{i:02d}: {cls_name:<10s} score={score:.3f} "
                      f"box=[{int(x1)},{int(y1)}→{int(x2)},{int(y2)}]")

        return elapsed, dets

    def annotate(self, image: np.ndarray, detections, threshold: float = 0.25):
        img = image.copy()
        for xmin, ymin, xmax, ymax, score, cls in detections:
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

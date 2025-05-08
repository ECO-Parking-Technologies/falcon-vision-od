import time
from pathlib import Path

import cv2
import numpy as np
import torch

from effdet import DetBenchPredict, create_model


class EfficientDetModel:
    def __init__(
        self,
        model_path: str,
        model_name: str = None,
        num_classes: int = None,
        framework: str = "pytorch",
    ):
        """
        :param model_path: Path to the .pt/.pth file with EfficientDet weights.
        :param model_name: e.g. "tf_efficientdet_lite0". If None, will be inferred
                           from Path(model_path).stem.
        :param num_classes: number of object classes (not counting background);
                            if None, defaults to 90 (COCO).
        """
        self.model_path = model_path
        self.framework = framework.lower()
        if self.framework != "pytorch":
            raise ValueError("Only 'pytorch' framework is supported right now.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # figure out model_name
        if model_name is None:
            model_name = Path(model_path).stem
        self.model_name = model_name

        # figure out num_classes
        if num_classes is None:
            num_classes = 90
        self.num_classes = num_classes

        # build the base network
        base_model = create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            bench_task=None,  # unwrapped; we'll wrap below
        )

        # load your weights
        state_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] Mismatched keys when loading weights:")
            if missing:
                print(f"  - Missing keys: {len(missing)}")
            if unexpected:
                print(f"  - Unexpected keys: {len(unexpected)}")

        # wrap for inference
        self.model = DetBenchPredict(base_model)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def load_pytorch_model(model_path: str, model_name: str, num_classes: int):
        """
        Alternate loader if you just want a bare model → DetBenchPredict
        """
        m = create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            bench_task=None,
        )
        sd = torch.load(model_path, map_location="cpu")
        m.load_state_dict(sd)
        m = DetBenchPredict(m)
        m.eval()
        return m

    def preprocess(self, image: np.ndarray, input_size=(640, 640)):
        img = cv2.resize(image, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def infer(self, image: np.ndarray, input_size=(640, 640)):
        x = self.preprocess(image, input_size)
        t0 = time.time()
        with torch.no_grad():
            out = self.model(x)
        dt = time.time() - t0
        # out[0] is an (N,6) array: [ymin,xmin,ymax,xmax,score,class]
        dets = out[0].cpu().numpy()
        return dt, dets

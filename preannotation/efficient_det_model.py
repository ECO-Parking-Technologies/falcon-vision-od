import time
from pathlib import Path

import cv2
import numpy as np
import torch

from effdet import DetBenchPredict, create_model


class EfficientDetModel:
    def __init__(self, model_path, framework="pytorch"):
        """
        Initializes the EfficientDet model.
        :param model_path: Path to the .pt file with EfficientDet weights.
        """
        self.model_path = model_path
        self.framework = framework.lower()

        if self.framework != "pytorch":
            raise ValueError("Only 'pytorch' framework is supported right now.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = Path(model_path).stem
        # Set num_classes to 90 for pretrained EfficientDet COCO weights
        num_classes = 90

        base_model = create_model(
            model_name, pretrained=False, num_classes=num_classes, bench_task=None
        )

        state_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)

        if missing or unexpected:
            print(f"[WARN] Mismatched keys when loading weights:")
            if missing:
                print(f" - Missing keys: {len(missing)}")
            if unexpected:
                print(f" - Unexpected keys: {len(unexpected)}")

        self.model = DetBenchPredict(base_model)
        self.model.eval()
        self.model.to(self.device)

    def load_pytorch_model(
        model_path, model_name="tf_efficientdet_d3_ap", num_classes=90
    ):
        # Create base model first (no bench wrapper)
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            bench_task=None,  # important!
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # Wrap in DetBenchPredict after weights are loaded
        model = DetBenchPredict(model)
        model.eval()
        return model

    def preprocess(self, image, input_size=(640, 640)):
        image_resized = cv2.resize(image, input_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

    def infer(self, image, input_size=(640, 640)):
        image_tensor = self.preprocess(image, input_size)
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(image_tensor)

        end_time = time.time()
        inference_time = end_time - start_time

        # You might need to adapt this part based on your model's output format
        # Expected: list of detections per image, each as [ymin, xmin, ymax, xmax, score, class]
        detections = outputs[0].cpu().numpy()  # assuming a list of length 1

        return inference_time, detections

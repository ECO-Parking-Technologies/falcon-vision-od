#!/usr/bin/env python3
"""Score the off-the-shelf EfficientDet-Lite2 TFLite (production baseline) on
a COCO-format eval root (annotations/instances_val2017.json + val2017/). Emits COCO-format predictions
with the split's remapped category ids (1..6)."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MODEL = sys.argv[3] if len(sys.argv) > 3 else "baseline/efficientdet_lite2.tflite"
ROOT = Path(sys.argv[1])          # fullset root (annotations/ + val2017/)
OUT = Path(sys.argv[2])
# 4th arg "ours": model emits 0-based OUR-class indices (our dropins) instead
# of 0-based COCO-90 (the off-the-shelf baseline)
OUR_CLASSES = len(sys.argv) > 4 and sys.argv[4] == "ours"

# baseline emits 0-based COCO-90 class indices -> +1 = original COCO ids;
# split jsons are remapped: orig {1,2,3,4,6,8} -> {1,2,3,4,5,6}
ORIG_TO_REMAP = ({i: i for i in range(1, 7)} if OUR_CLASSES
                 else {1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 8: 6})

interp = Interpreter(model_path=MODEL, num_threads=8)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outs = interp.get_output_details()
ih, iw = inp["shape"][1], inp["shape"][2]
print(f"input {iw}x{ih} dtype={inp['dtype'].__name__}, {len(outs)} outputs")

coco = json.load(open(ROOT / "annotations/instances_val2017.json"))
preds, t0 = [], time.time()
for n, im in enumerate(coco["images"], 1):
    img = cv2.imread(str(ROOT / "val2017" / im["file_name"]))
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img, (iw, ih)), cv2.COLOR_BGR2RGB)
    interp.set_tensor(inp["index"], rgb[None].astype(inp["dtype"]))
    interp.invoke()
    boxes = interp.get_tensor(outs[0]["index"])[0]   # [N,4] ymin,xmin,ymax,xmax (norm)
    classes = interp.get_tensor(outs[1]["index"])[0]
    scores = interp.get_tensor(outs[2]["index"])[0]
    count = int(interp.get_tensor(outs[3]["index"])[0])
    for i in range(count):
        cat = ORIG_TO_REMAP.get(int(classes[i]) + 1)
        if cat is None or scores[i] < 0.01:
            continue
        y1, x1, y2, x2 = boxes[i]
        preds.append(dict(image_id=im["id"], category_id=cat,
                          bbox=[float(x1 * w), float(y1 * h),
                                float((x2 - x1) * w), float((y2 - y1) * h)],
                          score=float(scores[i])))
    if n % 500 == 0:
        print(f"{n}/{len(coco['images'])} ({n/(time.time()-t0):.1f} img/s)")
OUT.write_text(json.dumps(preds))
print(f"wrote {len(preds)} predictions in {time.time()-t0:.0f}s -> {OUT}")

"""FW-faithful ROI cropping for training data.

Mirrors the sensor firmware's OD input geometry
(AnalyzerParkingObjectDetectEngine::AdjustRoi): per camera, the crop is the
union bounding box of that camera's spot polygons, padded, then expanded
toward the model's tensor size exactly like the firmware does. Training on
these crops matches the deployed input distribution: an in-spot car fills a
large fraction of every sample regardless of across-lane vs down-lane
sensor geometry.

Calibration-drift-aware: polygons are matched per frame (exact snapshot-run
id from the filename, else nearest-in-time), same as label_inspot.py.
"""
import json
import re
from pathlib import Path

from preannotation.label_inspot import frame_month, frame_run8, spaces_for

# AnalyzerParkingEngineConfig.h defaults (normalized image units)
FW_PAD_LEFT, FW_PAD_RIGHT = 0.0, 0.0
FW_PAD_TOP, FW_PAD_BOTTOM = 0.1, 0.1


def _clamp(v):
    return min(1.0, max(0.0, v))


def fw_roi(spaces, width, height, tensor_w, tensor_h):
    """Firmware AdjustRoi in normalized coords -> pixel box (x0, y0, x1, y1).

    Steps (defaults: expandWidth=true, expandBottom=true):
      union bbox of polygons -> pad -> if narrower than the tensor, widen to
      tensor width centered (shifted to stay in-frame) -> if shorter than the
      tensor, grow the bottom edge (only downward) toward tensor height.
    """
    xs = [p[0] for sp in spaces for p in sp["points"]]
    ys = [p[1] for sp in spaces for p in sp["points"]]
    x0, y0 = _clamp(min(xs) - FW_PAD_LEFT), _clamp(min(ys) - FW_PAD_TOP)
    x1, y1 = _clamp(max(xs) + FW_PAD_RIGHT), _clamp(max(ys) + FW_PAD_BOTTOM)

    if (x1 - x0) * width < tensor_w:
        cx, half = (x0 + x1) / 2.0, (tensor_w / 2.0) / width
        x0, x1 = cx - half, cx + half
        if x0 < 0.0:
            x1, x0 = x1 - x0, 0.0
        elif x1 > 1.0:
            x0, x1 = x0 - (x1 - 1.0), 1.0
        x0, x1 = _clamp(x0), _clamp(x1)

    if (y1 - y0) * height < tensor_h:
        y1 = _clamp(y1 + (tensor_h - (y1 - y0) * height) / height)

    px0, py0 = int(round(x0 * width)), int(round(y0 * height))
    px1, py1 = int(round(x1 * width)), int(round(y1 * height))
    return px0, py0, max(px1, px0 + 1), max(py1, py0 + 1)


class RoiCropper:
    """Per-frame crop boxes from the spot-polygon snapshot history."""

    def __init__(self, polygons_path, tensor_w, tensor_h):
        self.polys = json.loads(Path(polygons_path).read_text())
        self.tw, self.th = tensor_w, tensor_h
        self.match_counts = {"exact": 0, "timeline": 0, "latest": 0, "none": 0}

    def crop_for(self, garage, sensor, file_name, width, height):
        """-> (x0, y0, x1, y1) pixels, or None when no calibration exists."""
        cam = re.search(r"camera(\d+)", Path(file_name).name)
        key = f"{sensor}|camera{cam.group(1)}" if cam else ""
        spaces, how = spaces_for(self.polys.get(garage, {}), key,
                                 frame_run8(file_name), frame_month(file_name))
        self.match_counts[how] += 1
        if not spaces:
            return None
        return fw_roi(spaces, width, height, self.tw, self.th)


def crop_annotation(bbox, crop, min_frac=0.25, min_px=2):
    """Shift a COCO xywh bbox into crop space, clipped; None if what remains
    is under min_frac of the original area (matches the identifiability rule:
    a sliver at the crop edge is not a labelable vehicle)."""
    cx0, cy0, cx1, cy1 = crop
    x, y, w, h = bbox
    nx0, ny0 = max(x - cx0, 0.0), max(y - cy0, 0.0)
    nx1 = min(x + w - cx0, cx1 - cx0)
    ny1 = min(y + h - cy0, cy1 - cy0)
    nw, nh = nx1 - nx0, ny1 - ny0
    if nw < min_px or nh < min_px or nw * nh < min_frac * w * h:
        return None
    return [nx0, ny0, nw, nh]

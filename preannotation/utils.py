import os


def extract_sensor_and_camera(filename):
    """
    Extracts the sensor ID and camera ID from filenames like: fv6b1216-camera1-20211103-000401-01.png
    """
    parts = filename.split("-")
    sensor = parts[0]
    camera = parts[1] if len(parts) > 1 else "unknown"
    return sensor, camera


def convert_detections(detections, label_map, allowed_labels, threshold=0.25):
    converted = []

    if isinstance(detections, list) and len(detections) == 1:
        detections = detections[0]

    for d in detections:
        if len(d) != 6:
            continue

        xmin, ymin, xmax, ymax, score, cls = d
        if score < threshold:
            continue

        label_id = int(cls)
        label_name = label_map.get(label_id, "unknown")

        if label_id not in allowed_labels:
            continue

        x = float(xmin)
        y = float(ymin)
        w = float(xmax) - float(xmin)
        h = float(ymax) - float(ymin)

        converted.append(
            {
                "bbox": [x, y, w, h],
                "score": float(score),
                "label": label_name,  # use label name instead of id
            }
        )

    return converted

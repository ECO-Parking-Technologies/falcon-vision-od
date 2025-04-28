import yaml
from pathlib import Path

def load_label_map(config_path=None):
    """
    Loads a label map from config/label_map.yaml by default.

    Returns:
        label_map (dict): Maps int class ID to class name.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "label_map.yaml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return {int(k): v for k, v in data["label_map"].items()}

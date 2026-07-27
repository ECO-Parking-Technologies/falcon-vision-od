"""Shared artifact layout for exported models.

Layout (under the training output_dir):

    artifacts/<model_name>/<run_name>/
        model.ts.pt        TorchScript trace (float, for preannotation)
        model.f32.tflite   float32 TFLite
        model.int8.tflite  full-int8 TFLite (static PTQ)
        manifest.json      what/how/from-which-checkpoint + parity metrics
    artifacts/<model_name>/latest -> <run_name>/   (stable path for configs)

run_name is the training run directory name (e.g. 20250506-142806-tf_efficientdet_lite0)
or "adhoc-<checkpoint stem>" for checkpoints outside the train/ tree.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_name_for_checkpoint(ckpt: Path) -> str:
    parent = ckpt.parent
    if parent.parent.name == "train":
        return parent.name
    return f"adhoc-{ckpt.stem}"


def artifact_dir(output_dir: Path, model_name: str, ckpt: Path) -> Path:
    d = Path(output_dir) / "artifacts" / model_name / run_name_for_checkpoint(ckpt)
    d.mkdir(parents=True, exist_ok=True)
    return d


def update_latest_symlink(art_dir: Path) -> None:
    latest = art_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(art_dir.name)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def update_manifest(art_dir: Path, section: str, data: dict) -> Path:
    """Merge `data` under `section` into manifest.json (each exporter owns a section)."""
    path = art_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    data = dict(data)
    data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["git_commit"] = git_commit()
    manifest[section] = data
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path

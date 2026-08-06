"""Shared artifact layout for exported models.

Layout (under the training output_dir) — run-first so `ls` sorts
chronologically and matches the train/ tree one-to-one:

    artifacts/<run_name>/
        <run>.dropin-<size>-{f32,dyn,int8}.tflite   sensor-ready packages
        <run>.raw-{f32,dyn,int8}.tflite             raw exports (dev/eval)
        <run>.ts.pt                                 TorchScript trace
        manifest.json                               provenance + validation
    artifacts/latest-<model_name> -> <run_name>/    (stable path for configs)

run_name is the training run directory name (e.g. 20250506-142806-tf_efficientdet_lite0)
or "adhoc-<checkpoint stem>" for checkpoints outside the train/ tree.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_name_for_checkpoint(ckpt: Path) -> str:
    """<session-dt>-<level>[-roi] file prefix. Session layout:
    <output>/<session-dt>/<level>/train/model_best.pth.tar

    The level name usually carries its variant tags (lite1-roi, lite1-a20);
    if the run manifest says roi_crop but the level name doesn't say so,
    -roi is appended — artifact names must never hide the input geometry."""
    train_dir = ckpt.parent
    if train_dir.name != "train":
        return f"adhoc-{ckpt.stem}"
    level_dir = train_dir.parent
    name = f"{level_dir.parent.name}-{level_dir.name}"
    manifest = level_dir / "run.json"
    if "roi" not in level_dir.name and manifest.exists():
        try:
            if json.loads(manifest.read_text()).get("roi_crop"):
                name += "-roi"
        except Exception:
            pass
    return name


def artifact_dir(output_dir: Path, model_name: str, ckpt: Path) -> Path:
    """export/ sibling of the checkpoint's train/ dir; adhoc checkpoints get
    <output>/adhoc-<stem>/export."""
    train_dir = ckpt.parent
    if train_dir.name == "train":
        d = train_dir.parent / "export"
    else:
        d = Path(output_dir) / f"adhoc-{ckpt.stem}" / "export"
    d.mkdir(parents=True, exist_ok=True)
    return d


def update_latest_symlink(art_dir: Path, model_name: str = None) -> None:
    """<output>/latest-<model> -> <session>/<level>/export (relative)."""
    level_dir = art_dir.parent
    root = level_dir.parent.parent
    name = model_name or level_dir.name.split("-")[0]
    latest = root / f"latest-{name}"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(art_dir.relative_to(root))


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

#!/usr/bin/env python3
"""Automate CVAT task creation: one task per garage from data/cvat_tasks/.

For every garage bundle produced by preannotation/export_cvat_tasks.py this
creates a task under the given project, attaches the images from the CVAT
share, and uploads the Grounding DINO preannotations (COCO 1.0). Idempotent:
garages that already have a task of the same name are skipped, so re-running
tops up only what's missing.

Credentials are prompted (RAM only, never stored):

    python3 cvat/create_tasks.py --host http://192.168.20.66:8085 \
        --project "Falcon Vision v2"
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich import box

try:
    from cvat_sdk.core.proxies.tasks import ResourceType
except ImportError:
    sys.exit("cvat_sdk not found — activate the MAIN venv first:\n"
             "  source falcon-vision-od-venv/bin/activate\n"
             "(it's in requirements.txt; the export venv doesn't have it)")

from cvat_access import open_client

console = Console()


def ensure_labels(client, project):
    """Create the project's labels from config/cvat_labels.json if missing."""
    import json
    have = {l.name for l in project.get_labels()}
    spec = json.loads(Path("config/cvat_labels.json").read_text())
    missing = [l for l in spec if l["name"] not in have]
    if not missing:
        return
    for l in missing:
        l.pop("id", None)  # CVAT assigns its own ids
    console.print(f"project is missing {len(missing)} labels — creating them…")
    from cvat_sdk.api_client import models
    label_models = []
    for l in missing:
        attrs = [models.AttributeRequest(
                     name=a["name"], mutable=a["mutable"],
                     input_type=models.InputTypeEnum(a["input_type"]),
                     default_value=a["default_value"], values=a["values"])
                 for a in l.get("attributes", [])]
        label_models.append(models.PatchedLabelRequest(
            name=l["name"], color=l.get("color"),
            type=l.get("type", "rectangle"), attributes=attrs))
    client.api_client.projects_api.partial_update(
        project.id,
        patched_project_write_request=models.PatchedProjectWriteRequest(
            labels=label_models))
    console.print(f"[green]created {len(label_models)} labels[/green]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8085")
    ap.add_argument("--project", default="Falcon Vision v2")
    ap.add_argument("--tasks-dir", type=Path, default=Path("data/cvat_tasks"))
    ap.add_argument("--org", default=None,
                    help="organization slug (needed if the project lives in one)")
    ap.add_argument("--garages", help="comma-separated subset (default: all bundles)")
    ap.add_argument("--cf-access", action="store_true",
                    help="authenticate through Cloudflare Access with a "
                         "service token (prompted, RAM only)")
    ap.add_argument("--host-header", default=None,
                    help="Host header override: reach the LAN IP while "
                         "traefik routes by the canonical CVAT_HOST name")
    args = ap.parse_args()

    user = Prompt.ask("[cyan]CVAT username[/cyan]", console=console)
    password = Prompt.ask("[cyan]CVAT password[/cyan]", password=True, console=console)

    bundles = sorted(d for d in args.tasks_dir.iterdir() if d.is_dir())
    if args.garages:
        wanted = {g.strip() for g in args.garages.split(",")}
        bundles = [b for b in bundles if b.name in wanted]
    if not bundles:
        sys.exit(f"no task bundles under {args.tasks_dir}")

    with open_client(args.host, user, password, args.cf_access,
                     args.host_header) as client:
        if args.org:
            client.organization_slug = args.org
        projects = [p for p in client.projects.list() if p.name == args.project]
        if not projects:
            sys.exit(f"project {args.project!r} not found — create it (with labels) first")
        project = projects[0]
        ensure_labels(client, project)
        existing = {t.name: t for t in client.tasks.list()}

        t = Table(title=f"CVAT tasks → project '{args.project}'", box=box.SIMPLE_HEAD)
        for col in ("garage", "images", "boxes", "status"):
            t.add_column(col, justify="right" if col in ("images", "boxes") else "left")

        created = skipped = failed = 0
        for b in bundles:
            imgs = sorted(p.name for p in b.glob("*.jpg"))
            ann = b / "preannotations.coco.json"
            import json
            n_anns = len(json.loads(ann.read_text())["annotations"]) if ann.exists() else 0
            if b.name in existing:
                task = existing[b.name]
                if n_anns and task.get_annotations().version == 0 and not task.get_annotations().shapes:
                    try:
                        console.print(f"importing annotations into existing [cyan]{b.name}[/cyan]…")
                        task.import_annotations("COCO 1.0", str(ann))
                        t.add_row(b.name, str(len(imgs)), str(n_anns),
                                  "[green]annotations imported[/green]")
                        created += 1
                    except Exception as e:
                        t.add_row(b.name, str(len(imgs)), str(n_anns),
                                  f"[red]import FAILED[/red] {str(e)[:60]}")
                        failed += 1
                else:
                    t.add_row(b.name, str(len(imgs)), str(n_anns), "[dim]exists, skipped[/dim]")
                    skipped += 1
                continue
            try:
                console.print(f"creating [cyan]{b.name}[/cyan] ({len(imgs)} images)…")
                client.tasks.create_from_data(
                    spec={"name": b.name, "project_id": project.id},
                    resource_type=ResourceType.SHARE,
                    resources=[f"cvat_tasks/{b.name}/{img}" for img in imgs],
                    annotation_path=str(ann) if n_anns else None,
                    annotation_format="COCO 1.0" if n_anns else None,
                )
                t.add_row(b.name, str(len(imgs)), str(n_anns), "[green]created[/green]")
                created += 1
            except Exception as e:
                t.add_row(b.name, str(len(imgs)), str(n_anns), f"[red]FAILED[/red] {str(e)[:60]}")
                failed += 1

        console.print(t)
        console.print(f"\n[green]{created} created[/green] · {skipped} skipped · "
                      f"{'[red]' if failed else ''}{failed} failed{'[/red]' if failed else ''}")


if __name__ == "__main__":
    main()

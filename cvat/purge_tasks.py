#!/usr/bin/env python3
"""Delete ALL tasks (and their annotations) in a CVAT project.

Keeps the project, its labels, and all user accounts — use before re-importing
better preannotation drafts via create_tasks.py. Asks for confirmation with
the task list shown first. Credentials are prompted (RAM only, never stored).

    python3 cvat/purge_tasks.py --host https://<cvat-host> --project "Falcon Vision v2"
"""
import argparse
import sys

from rich.console import Console
from rich.prompt import Confirm, Prompt

try:
    from cvat_sdk import make_client
except ImportError:
    sys.exit("cvat_sdk not found — activate the MAIN venv first:\n"
             "  source falcon-vision-od-venv/bin/activate")

console = Console()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8085")
    ap.add_argument("--project", default="Falcon Vision v2")
    ap.add_argument("--org", default=None)
    args = ap.parse_args()

    user = Prompt.ask("[cyan]CVAT username[/cyan]", console=console)
    password = Prompt.ask("[cyan]CVAT password[/cyan]", password=True,
                          console=console)

    with make_client(args.host, credentials=(user, password)) as client:
        if args.org:
            client.organization_slug = args.org
        projects = [p for p in client.projects.list() if p.name == args.project]
        if not projects:
            sys.exit(f"project {args.project!r} not found")
        project = projects[0]
        tasks = [t for t in client.tasks.list() if t.project_id == project.id]
        if not tasks:
            console.print("[green]no tasks to purge[/green]")
            return
        for t in sorted(tasks, key=lambda t: t.name):
            console.print(f"  {t.name}")
        if not Confirm.ask(
                f"[red]Delete these {len(tasks)} tasks and ALL their "
                f"annotations from '{args.project}'?[/red]", console=console):
            sys.exit("aborted")
        for t in tasks:
            t.remove()
            console.print(f"deleted [red]{t.name}[/red]")
        console.print(f"[green]purged {len(tasks)} tasks[/green] — project, "
                      "labels and users kept")


if __name__ == "__main__":
    main()

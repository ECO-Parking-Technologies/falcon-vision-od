# Annotation SOP — Falcon Vision (for annotators)

You're reviewing photos from parking garages and making sure every vehicle and
person is correctly boxed. A computer already drew first-draft boxes — your job
is to **check and fix**, not to start from scratch. No technical background
needed.

## One-time setup

1. Open the CVAT link you were given and **Create an account** (remember your
   username — tasks get assigned to it).
2. Tell the admin your username so they can assign you tasks.

## Daily loop

1. Top menu → **Tasks**. Open a task assigned to you (each task = one garage,
   about 100 photos).
2. Click the **Job** inside it. You're now in the editor.
3. For each photo (use `F` = next, `D` = back):
   - **Check the drawn boxes.** Wrong label (a car marked "truck")? Change it
     in the right sidebar. Box on nothing / doubled box? Delete it (`Del`).
   - **Box anything missed.** Press `N`, drag a tight box around the vehicle,
     pick the label. Box every vehicle you can clearly identify — even far
     away ones. If you honestly can't tell whether a dark blob is a car,
     skip it and move on.
   - **Tick the checkboxes** on each vehicle box (right sidebar → DETAILS):
     - `InEcoParkingSpot` — it's parked in one of the sensor-monitored spots
     - `InMotion` — it's driving, not parked
     - `Occluded` — a big part of it is hidden (pillar, another car, edge)
   - **Tag the photo** (tag tool in the left bar) only if it clearly applies:
     glare / low-light / rain / snow / dirty-lens / obstruction.
4. **Save often** (`Ctrl+S`). CVAT does not auto-save.
5. Finished every photo in the job? Menu (top-left) → set the job state to
   **completed**. Then open your next task.

## Rules of thumb

- Tight boxes, no padding, around the **visible** part only.
- A missed vehicle is worse than an extra minute — box everything identifiable.
- Torn about something for more than 10 seconds? Skip it, move on.
- Full decision table + examples: [annotation-guidelines.md](annotation-guidelines.md).
- Stuck, or something looks broken? Stop and message the admin — don't guess.

---

# Admin notes (Emilio)

- **Cloudflare tunnel**: point it at `localhost:8085`. Then Traefik MUST know
  the public hostname: `export CVAT_HOST=<tunnel-hostname>` +
  `docker compose up -d` in the cvat clone, or all tunnel traffic 404s.
  Single-host routing: after switching, use the tunnel URL yourself too.
- **Accounts**: annotators self-register at the URL. Assign tasks: open task →
  Assignee field. Set stage `annotation`.
- **Progress**: Tasks page shows per-task completion; Analytics has per-user
  stats. Spot-check early work from each new annotator against the guidelines
  (especially box tightness and the eco-spot flag) before they do volume.
- **When a garage's task is completed**: export it (task → Menu → Export
  annotations → COCO 1.0) into `data/cvat_exports/`. ~8–10 garages is enough
  to run the first training round.

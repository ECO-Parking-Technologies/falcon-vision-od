# Annotation Guidelines — Falcon Vision OD (clean-slate v2)

The one rule everything follows from: **an unlabeled visible vehicle teaches the
model "this is not a vehicle."** Missing boxes are worse than extra effort.
Consistency beats perfection — the model learns whatever policy we apply, so
apply one policy everywhere.

## What to box

| Situation | Do this |
|---|---|
| Clearly identifiable vehicle, any distance | **Box it** — even small, far, in another aisle |
| Vehicle behind fence/cables/pillar, still recognizable | Box the **visible part**, tick `Occluded` |
| Vehicle cut off at frame edge | Box the visible part, tick `Occluded` |
| Dark smudge that *might* be a car (can't tell at normal zoom) | **Skip** — if a human can't tell, don't guess |
| Reflection of a car (glass, wet floor) | Skip |
| Object smaller than ~12 px in its short dimension / unrecognizable | Skip |
| Row of cars where you can separate individuals | One box per car |
| Mass where individuals truly can't be separated | Skip the indistinguishable part (rare in garages) |
| Person anywhere in frame | Box (Occluded if partial) |

**Boxes are tight, no padding**, around the visible pixels only (don't guess the
hidden extent of an occluded car).

## Attributes (vehicle boxes)

- **InEcoParkingSpot** — the vehicle occupies a sensor-monitored spot. This is
  the ONLY place "does this car matter" lives. Background/distant cars get boxed
  *without* it. When unsure whether a spot is monitored, leave unticked.
- **InMotion** — driving through, not parked (lane traffic, mid-maneuver).
- **Occluded** — meaningfully cut off by frame edge, pillar, fence, or another
  vehicle (rule of thumb: >~25% hidden).

## Image tags (whole-frame conditions, only when clearly present)

`glare` · `low-light` (scene dark enough that vehicles, especially dark ones,
are genuinely hard to make out) · `rain` · `snow` · `dirty-lens` · `obstruction`
(something blocking the lens view). Garage/sensor/time-of-day are derived
automatically — never tag them.

## Auditing the pre-annotations (Grounding DINO drew first drafts)

Expect and fix these known patterns:
- **Missed vehicles** — especially distant/dark ones (the drafts used a 0.25
  confidence cutoff). Drawing missed boxes is the main manual work.
- **car ↔ truck mixups** (SUVs/pickups) — fix the class from the sidebar.
- **Duplicate overlapping boxes** on the same object — delete the worse one.
- **Wheel → motorcycle**, **cart → truck** false positives — delete.
- Class definitions: pickup/box-truck/van = `truck`; SUV/sedan/crossover =
  `car`; shuttle = `bus`.

Deleting a wrong box is cheap; a wrong box left in is expensive. When genuinely
torn about an object for >10 seconds, skip it and move on — consistency over
agonizing.

## Workflow

Hotkeys: `F`/`D` next/prev frame · `N` draw box · `Del` delete · `Ctrl+S` save
often. Work a full garage task in one sitting where possible (the scenes repeat,
you get fast). Suggested order: audit ~8–10 diverse garages first, export, train
(validates the policy cheaply), then continue.

## Why background cars still get boxed

At deployment the OD model looks at crops around monitored spots — but it trains
on these full frames. Every visible unboxed car in training data is a lesson in
ignoring cars. Box them all; the `InEcoParkingSpot` flag keeps the operational
distinction clean.

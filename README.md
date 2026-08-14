# KINESIS · Fight Lab

The fourth member of the KINESIS ALPHA family. One handheld 832×480 clip of a
taekwondo bout — **Y. Xu (CHN, chung, blue)** against **L.S. Gallo (MEX, hong,
red)**, Chuncheon 2024 — read completely by machines: SAM 3 segmentation,
SAM 3D Body reconstruction, joint-angle and velocity analytics, and a
load-screening ledger, in the family's coal/bone/amber darkroom language.

## The views

- **The bout** (`#/bout`) — the same seconds read three ways, on one
  transport: SAM 3 masks burned over the footage, cleaned SAM 3D Body
  projections, and the interactive 3D reconstruction. Beneath them the
  **fight view**: both athletes alone on black, per-fighter toggle between
  the raw skeleton and a low-poly body, velocity arrows that flare on
  strike highlights, live joint-angle arcs in 3D, ankle trails, and the
  exchange reel — every strike over threshold, clickable.
- **Load & technique** (`#/analysis`) — knee/hip strips (left solid, right
  dashed), foot speed, knee angular velocity, the knee phase portrait, and
  the screening ledger: terminal extensions, high-ω events, trunk
  compensation. Mechanical threshold events, not medical findings.
- **Method** (`#/method`) — what ran, on what, and the honest edges.

## Run it

```
./flab serve          # http://127.0.0.1:5199 — stdlib server, range requests
python3 tests/selftest.py
node tests/shot.js    # headless-Chrome visual QA -> work/shots/
```

The store ships prebuilt; the web app renders artifacts and computes nothing.

## The pipeline

Heavy inference ran on a Mac Studio (M3 Ultra, MPS) over SSH; export and
rendering on a Mac mini (M4). Every stage is chunk-restartable.

| stage | what | where |
|---|---|---|
| 01 | VFR source (~31 fps true) → CFR 30 master | mini |
| 02 | RetinaNet + greedy-IoU tracklets; identity = hogu colour (hong red, chung blue, referee white); clinches leave gaps, not guesses | studio |
| 03 | per-frame box-prompted SAM 3 (`Sam3Tracker`, bf16) — mask k *is* box k | studio |
| 04 | SAM 3D Body (MHR rig) per fighter per frame, 30 Hz | studio |
| 05 | skeleton cleaning, shared RANSAC floor, world re-referencing, angles, velocities, screening events | mini |
| 06 | family-palette overlays and kick micro-clips (x264) | mini |

Key numbers live in `store/bout.json`; the store contract in
`docs/STORE.md`; the cleaning rules in `pipeline/skelclean.py` (shared with
the performance lab — drop, never guess).

## The honest edges

Depth is model-inferred monocular; the shared fight space is camera-relative
with both fighters re-referenced to one RANSAC floor. Sampling floors the
peaks — speeds and angular velocities are lower bounds. Identity is clothing
plus the broadcast scoreboard, not faces. The screening ledger counts
mechanical threshold events worth reviewing on film; it diagnoses nothing.

## Family

`kinesis-alpha` · `kinesis-alpha-performance-lab` ·
`kinesis-alpha-decision-lab` · **`kinesis-alpha-fight-lab`** — same law:
two typefaces, 1 px rules, zero radius, serif tabular numerals against mono
tracked labels, absences stated and never filled.

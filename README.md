# KINESIS · Fight Lab

The fourth member of the KINESIS ALPHA family. One handheld 832×480 clip of a
taekwondo bout — **Y. Xu (CHN, chung, blue)** against **L.S. Gallo (MEX, hong,
red)**, Chuncheon 2024 — read completely by machines: SAM 3.1 segmentation,
SAM 3D Body reconstruction, joint-angle and velocity analytics, and a
load-screening ledger, in the family's coal/bone/amber darkroom language.

## The views

- **The bout** (`#/bout`) — the same seconds read three ways, on one
  transport: temporally identified SAM 3.1 video masks, cleaned SAM 3D Body
  projections, and the actual predicted MHR surface meshes. Beneath them the
  **fight view**: both athletes alone on black, per-fighter toggle between
  the raw skeleton and a clearly labelled low-poly volume proxy, velocity arrows that flare on
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

Heavy inference runs across a Mac mini (M4, MLX) and a Mac Studio (M3 Ultra,
MPS) over SSH; export and rendering run on the mini. Every stage is
chunk-restartable.

| stage | what | where |
|---|---|---|
| 01 | VFR source (~31 fps true) → CFR 30 master | mini |
| 02 | RetinaNet prompt tracks; fighter identity = hogu colour (hong red, chung blue); clinches leave gaps, not guesses | studio |
| 03 | SAM 3.1 (`mlx-community/sam3.1-bf16`) person masks refreshed per analysis frame; mask-level hogu evidence assigns hong/chung while temporal boxes supply spatial proposals, then single-instance and conservative size/on-mat gates reject disconnected referee fragments, spectators, and mat fragments | Apple silicon |
| 04 | mask-conditioned SAM 3D Body (MHR rig): MHR-70 joints plus the predicted body surface for each fighter | studio |
| 05 | skeleton cleaning, shared RANSAC floor, world re-referencing, angles, velocities, screening events | mini |
| 06 | synchronized mask, pose, and true MHR mesh videos plus kick micro-clips (x264) | mini |

Stage 03 uses a small, ignored MLX environment so the current SAM 3.1 model
does not disturb the PyTorch/SAM 3D Body environment:

```sh
python3 -m venv work/sam31-env
work/sam31-env/bin/pip install 'mlx-vlm==0.4.3' opencv-python
work/sam31-env/bin/hf download mlx-community/sam3.1-bf16 \
  --local-dir work/models/sam3.1-bf16
work/sam31-env/bin/python pipeline/03_masks.py
```

On a 16 GB machine, a local 4-bit conversion avoids MLX Metal paging while
keeping the same SAM 3.1 architecture.  Set the explicit path when using it:

```sh
work/sam31-env/bin/mlx_vlm.convert \
  --hf-path work/models/sam3.1-bf16 \
  --mlx-path work/models/sam3.1-4bit \
  --quantize --q-bits 4 --q-group-size 64
FIGHTLAB_SAM31=work/models/sam3.1-4bit \
  work/sam31-env/bin/python pipeline/03_masks.py
```

Key numbers live in `store/bout.json`; the store contract in
`docs/STORE.md`; the cleaning rules in `pipeline/skelclean.py` (shared with
the performance lab — drop, never guess).

## The honest edges

Depth is model-inferred monocular; the shared fight space is camera-relative
with both fighters re-referenced to one RANSAC floor. Unsupported camera-depth
jumps are withheld before world velocity is computed. Sampling floors the
peaks — speeds and angular velocities are lower bounds. Identity is clothing
plus the broadcast scoreboard, not faces. The screening ledger counts
mechanical threshold events worth reviewing on film; it diagnoses nothing.

## Family

`kinesis-alpha` · `kinesis-alpha-performance-lab` ·
`kinesis-alpha-decision-lab` · **`kinesis-alpha-fight-lab`** — same law:
two typefaces, 1 px rules, zero radius, serif tabular numerals against mono
tracked labels, absences stated and never filled.

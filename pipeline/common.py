"""Shared paths, constants, and helpers for the fight-lab pipeline.

Repo-shaped: every stage resolves paths from this file so the same code runs
on the Mac Studio (inference, models under ~/kinesis/models) and on the Mac
mini (export + render, models under PerformanceLabVideo/models). The video is
the CFR-30 master cut by 01 (the source file is VFR ~31 fps; a constant-rate
master is the only honest shared timeline).

Timeline law: processed frame j maps to master frame i = j * STRIDE and to
time t = i / FPS_SRC seconds. Every artifact stores master frame indices.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
STORE = ROOT / "store"

FPS_SRC = 30.0        # the CFR master
# Ten mask/pose samples per second is a practical floor for this
# 5.5 minute clip on Apple silicon.  It keeps kicks readable while avoiding a
# multi-day run of the SAM 3.1 video tracker + SAM 3D Body.  Override
# for a denser export with FIGHTLAB_STRIDE=2 (15 Hz) or =1 (30 Hz).
STRIDE = int(os.environ.get("FIGHTLAB_STRIDE", "3"))
FPS_PROC = FPS_SRC / STRIDE

ROLES = ("red", "blue", "ref")


def video_path():
    v = os.environ.get("FIGHTLAB_VIDEO")
    if v:
        return Path(v)
    return WORK / "fight_cfr30.mp4"


def models_dir():
    m = os.environ.get("FIGHTLAB_MODELS")
    if m:
        return Path(m)
    for cand in (Path.home() / "kinesis" / "models",
                 Path("/Users/hive/Claude Code/PerformanceLabVideo/models")):
        if cand.exists():
            return cand
    raise SystemExit("no models dir; set FIGHTLAB_MODELS")


def jnum(x):
    """np scalars -> plain python, recursively (json.dump chokes on np types)"""
    if isinstance(x, dict):
        return {k: jnum(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jnum(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def rle_encode(mask):
    """binary mask -> uint32 run lengths, first run counts zeros (may be 0)"""
    flat = np.asarray(mask, bool).ravel()
    if flat.size == 0:
        return np.zeros(0, np.uint32)
    change = np.nonzero(np.diff(flat))[0] + 1
    runs = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat[0]:
        runs = np.concatenate(([0], runs))
    return runs.astype(np.uint32)


def rle_decode(runs, shape):
    out = np.zeros(int(np.prod(shape)), bool)
    pos, val = 0, False
    for r in runs:
        if val:
            out[pos:pos + int(r)] = True
        pos += int(r)
        val = not val
    return out.reshape(shape)


def iter_frames(src, stride=STRIDE, f0=0, f1=None):
    """yield (master_frame_index, bgr) every `stride` frames of the master"""
    import cv2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    i = f0
    while f1 is None or i < f1:
        ret, fr = cap.read()
        if not ret:
            break
        if (i - f0) % stride == 0:
            yield i, fr
        i += 1
    cap.release()


def probe(src):
    import cv2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    meta = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "wh": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
            "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return meta


def load_fighters():
    f = WORK / "track" / "fighters.json"
    if not f.exists():
        raise SystemExit("run 02_track.py first")
    d = json.load(open(f))
    boxes = {r: {int(i): b for i, b in d["boxes"][r].items()} for r in d["boxes"]}
    return d, boxes


def mask_chunks():
    """Return only temporally identified SAM 3.1 video-mask chunks.

    Older builds wrote ``chunk_*.npz`` from independent image inference.  A
    distinct prefix makes it impossible for the pipeline to silently reuse
    those incompatible artifacts after the video-model upgrade.
    """
    folder = WORK / "masks"
    index = folder / "masks_index.json"
    if index.exists() and not os.environ.get("FIGHTLAB_IGNORE_INDEX"):
        try:
            meta = json.load(open(index))
            if int(meta.get("schema", 0)) >= 3:
                files = [folder / name for name in meta.get("chunks", [])]
                if files and all(path.exists() for path in files):
                    return files
        except (OSError, ValueError, TypeError):
            pass
    return sorted(folder.glob("video_chunk_*.npz"))


def pose_chunks():
    """Return exactly the SAM 3D chunks named by the completed stage index."""
    folder = WORK / "pose3d"
    index = folder / "pose_index.json"
    if index.exists():
        try:
            meta = json.load(open(index))
            files = [folder / name for name in meta.get("chunks", [])]
            if int(meta.get("schema", 0)) >= 2 and files and all(
                    path.exists() for path in files):
                return files
        except (OSError, ValueError, TypeError):
            pass
    return sorted(folder.glob("chunk_*.npz"))

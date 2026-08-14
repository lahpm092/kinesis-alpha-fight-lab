"""SAM 3D Body (MHR rig) per processed frame for both fighters.

The estimator gets the frame plus the red and blue boxes batched; a batched
failure retries per box; a per-box failure records ok=False for that frame -
the reconstruction is absent, not guessed. Depth is model-inferred monocular
depth; every downstream artifact says so.

Chunked + restartable: work/pose3d/chunk_<j0>.npz holds, per role,
F (master frame), K3 (70,3 f16 camera-space m), K2 (70,2 f16 source px),
CT (3 f32 camera translation m), ok (bool).

Usage: 04_pose3d.py [--limit N] [--chunk 400] [--dense]
  --dense reconstructs every master frame (30 Hz instead of 15), with the
  role box linearly interpolated across the 1-frame gaps between tracked
  frames; kick-speed peaks alias much less at 30 Hz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FIGHTERS = ("red", "blue")

_est = None


def estimator():
    global _est
    if _est is None:
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        mdl = common.models_dir() / "sam-3d-body"
        src = common.models_dir() / "sam-3d-body-src"
        if src.exists():
            sys.path.insert(0, str(src))
        else:
            sys.path.insert(0, "/Users/hive/Claude Code/PerformanceLabVideo/sam-3d-body-src")
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        import sam_3d_body.sam_3d_body_estimator as est_mod
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model, cfg = load_sam_3d_body(f"{mdl}/model.ckpt", device=dev,
                                      mhr_path=f"{mdl}/assets/mhr_model.pt")
        orig = est_mod.recursive_to
        est_mod.recursive_to = lambda x, d: orig(x, dev if d == "cuda" else d)
        _est = SAM3DBodyEstimator(model, cfg)
    return _est


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="processed frames")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args()
    src = common.video_path()
    d, boxes = common.load_fighters()
    out_dir = common.WORK / "pose3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dense:
        for r in FIGHTERS:
            idxs = sorted(boxes[r])
            for lo, hi in zip(idxs[:-1], idxs[1:]):
                if hi - lo == common.STRIDE:
                    for q in range(lo + 1, hi):
                        t = (q - lo) / (hi - lo)
                        boxes[r][q] = [a + (b - a) * t for a, b in
                                       zip(boxes[r][lo], boxes[r][hi])]
    frames = sorted(set(boxes["red"]) | set(boxes["blue"]))
    if args.limit:
        frames = frames[:args.limit]
    chunks = [frames[i:i + args.chunk] for i in range(0, len(frames), args.chunk)]
    print(f"{len(frames)} frames with a fighter box, {len(chunks)} chunks",
          flush=True)
    est = None

    for ci, ch in enumerate(chunks):
        cf = out_dir / f"chunk_{ch[0]:06d}.npz"
        if cf.exists():
            print(f"  chunk {ci + 1}/{len(chunks)} cached", flush=True)
            continue
        if est is None:
            est = estimator()
        t0 = time.time()
        acc = {r: {"F": [], "K3": [], "K2": [], "CT": [], "ok": []}
               for r in FIGHTERS}
        want = set(ch)
        cap = cv2.VideoCapture(str(src))
        cap.set(cv2.CAP_PROP_POS_FRAMES, ch[0])
        i = ch[0]
        n_done = 0
        while i <= ch[-1]:
            ret, fr = cap.read()
            if not ret:
                break
            if i in want:
                roles = [r for r in FIGHTERS if i in boxes[r]]
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                bxs = np.array([boxes[r][i] for r in roles], np.float32)
                try:
                    res = est.process_one_image(rgb, bboxes=bxs,
                                                inference_type="body")
                except Exception as e:
                    print(f"  frame {i}: batched {type(e).__name__}; per box",
                          flush=True)
                    res = []
                    for b in bxs:
                        try:
                            r1 = est.process_one_image(rgb, bboxes=b.reshape(1, 4),
                                                       inference_type="body")
                            res.append(r1[0] if r1 else None)
                        except Exception as e2:
                            print(f"  frame {i}: {type(e2).__name__} {e2}",
                                  flush=True)
                            res.append(None)
                for j, r in enumerate(roles):
                    o = res[j] if j < len(res) else None
                    a = acc[r]
                    a["F"].append(i)
                    if o is not None:
                        a["K3"].append(o["pred_keypoints_3d"][:70].astype(np.float16))
                        a["K2"].append(o["pred_keypoints_2d"][:70].astype(np.float16))
                        a["CT"].append(o["pred_cam_t"].astype(np.float32))
                        a["ok"].append(True)
                    else:
                        a["K3"].append(np.zeros((70, 3), np.float16))
                        a["K2"].append(np.zeros((70, 2), np.float16))
                        a["CT"].append(np.zeros(3, np.float32))
                        a["ok"].append(False)
                n_done += 1
                if n_done % 25 == 0:
                    print(f"    {n_done}/{len(ch)} "
                          f"({(time.time() - t0) / n_done:.2f}s/frame)", flush=True)
            i += 1
        cap.release()
        z = {}
        for r in FIGHTERS:
            a = acc[r]
            z[f"{r}_F"] = np.array(a["F"], np.int32)
            z[f"{r}_K3"] = np.array(a["K3"], np.float16)
            z[f"{r}_K2"] = np.array(a["K2"], np.float16)
            z[f"{r}_CT"] = np.array(a["CT"], np.float32)
            z[f"{r}_ok"] = np.array(a["ok"], bool)
        np.savez_compressed(cf, **z)
        print(f"  chunk {ci + 1}/{len(chunks)}: {n_done} frames in "
              f"{time.time() - t0:.0f}s", flush=True)
    print("pose3d done", flush=True)


if __name__ == "__main__":
    main()

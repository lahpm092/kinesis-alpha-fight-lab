"""Per-frame box-prompted SAM 3 masks for red, blue, and the referee.

Sam3TrackerModel from the sam3-hf checkpoint (bf16, MPS), prompted with the
02 role boxes: mask k belongs to prompted box k, so mask identity is exactly
box identity. Frames where a role has no box get no mask - absent, not
guessed. ~8x faster than SAM 3 video propagation on Apple silicon, and
identity-exact, which propagation is not (kinesis family measurement).

Chunked + restartable: work/masks/chunk_<j0>.npz holds RLE runs per frame
per role (uint32, common.rle_encode), px counts and mask centroids. A chunk
file that exists is trusted and skipped.

Usage: 03_masks.py [--limit N] [--chunk 400]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

_sam = None


def sam_tracker():
    global _sam
    if _sam is None:
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        mdir = common.models_dir() / "sam3-tracker-bf16"
        if not mdir.exists():
            mdir = common.models_dir() / "sam3-hf"
        # the processor's small files live with the original checkpoint
        pdir = common.models_dir() / "sam3-hf"
        if not (pdir / "processor_config.json").exists():
            pdir = mdir
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m = Sam3TrackerModel.from_pretrained(mdir, dtype=torch.bfloat16).to(dev).eval()
        p = Sam3TrackerProcessor.from_pretrained(pdir)
        _sam = (m, p, dev)
    return _sam


def masks_for_boxes(rgb, boxes):
    """SAM 3 tracker masks at frame size; mask k belongs to prompted box k"""
    import torch
    model, proc, dev = sam_tracker()
    inp = proc(images=rgb, input_boxes=[[list(map(float, b)) for b in boxes]],
               return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = model(**{k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
                       for k, v in inp.items()}, multimask_output=False)
    return proc.post_process_masks(out.pred_masks.float().cpu(),
                                   inp["original_sizes"].cpu())[0][:, 0].numpy() > 0


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="processed frames")
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args()
    src = common.video_path()
    d, boxes = common.load_fighters()
    out_dir = common.WORK / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(set().union(*[set(boxes[r]) for r in common.ROLES]))
    if args.limit:
        frames = frames[:args.limit]
    chunks = [frames[i:i + args.chunk] for i in range(0, len(frames), args.chunk)]
    print(f"{len(frames)} frames with a box, {len(chunks)} chunks", flush=True)

    for ci, ch in enumerate(chunks):
        cf = out_dir / f"chunk_{ch[0]:06d}.npz"
        if cf.exists():
            print(f"  chunk {ci + 1}/{len(chunks)} cached", flush=True)
            continue
        t0 = time.time()
        data = {}
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
                roles = [r for r in common.ROLES if i in boxes[r]]
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                ms = masks_for_boxes(rgb, [boxes[r][i] for r in roles])
                for j, r in enumerate(roles):
                    m = ms[j]
                    data[f"f{i}_{r}"] = common.rle_encode(m)
                    ys, xs = np.nonzero(m)
                    data[f"s{i}_{r}"] = np.array(
                        [m.sum(), xs.mean() if len(xs) else -1,
                         ys.mean() if len(ys) else -1], np.float32)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"    {n_done}/{len(ch)} "
                          f"({(time.time() - t0) / n_done:.2f}s/frame)", flush=True)
            i += 1
        cap.release()
        np.savez_compressed(cf, **data)
        print(f"  chunk {ci + 1}/{len(chunks)}: {n_done} frames in "
              f"{time.time() - t0:.0f}s", flush=True)

    json.dump({"wh": d["wh"], "n_frames": len(frames),
               "chunks": sorted(f.name for f in out_dir.glob("chunk_*.npz"))},
              open(out_dir / "masks_index.json", "w"))
    print("masks done", flush=True)


if __name__ == "__main__":
    main()

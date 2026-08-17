"""Mask-conditioned SAM 3D Body inference for both taekwondo fighters.

The old stage sent loose detector boxes from the full broadcast frame to SAM
3D Body and discarded its mesh.  This stage consumes the temporally tracked
SAM 3.1 masks, derives tight padded crops, conditions SAM 3D Body on the masks,
and stores both MHR-70 joints and a topology-preserving reduced MHR surface.
Every record uses the same master-frame index as the mask/video artifacts.

Chunk output ``work/pose3d/chunk_<master-frame>.npz`` contains, per role:
F, K3, K2, CT, FL, V and ok.  V is a ~1.7k-vertex reduction of the actual
18,439-vertex MHR prediction; ``mesh_topology.npz`` stores its triangle faces.

Usage: 04_pose3d.py [--limit N] [--chunk 40] [--chunk-start N]
                    [--chunk-stop N] [--reverse] [--redo]
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FIGHTERS = ("red", "blue")
SCHEMA = 2
_est = None


def estimator():
    global _est
    if _est is None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        mdl = common.models_dir() / "sam-3d-body"
        src = common.models_dir() / "sam-3d-body-src"
        if not src.exists():
            src = Path("/Users/hive/Claude Code/PerformanceLabVideo/sam-3d-body-src")
        sys.path.insert(0, str(src))
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        import sam_3d_body.sam_3d_body_estimator as est_mod

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model, cfg = load_sam_3d_body(
            f"{mdl}/model.ckpt", device=dev,
            mhr_path=f"{mdl}/assets/mhr_model.pt")
        # The released estimator hardcodes "cuda" while recursively moving
        # its batch.  Redirect only that requested device on Apple silicon.
        orig = est_mod.recursive_to
        est_mod.recursive_to = lambda x, d: orig(x, dev if d == "cuda" else d)
        _est = SAM3DBodyEstimator(model, cfg)
    return _est


def load_masks():
    files = common.mask_chunks()
    if not files:
        raise SystemExit("no SAM 3.1 video chunks; run 03_masks.py first")
    try:
        mask_index = json.load(open(common.WORK / "masks" / "masks_index.json"))
        expected_quality = mask_index["quality_version"]
        expected_frames = [int(v) for v in mask_index["F"]]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit("mask index is missing/incomplete; rerun 03_masks.py") from exc
    table = {}
    all_frames = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if (int(z["schema"]) != 3 or
                    str(z["quality_version"]) != expected_quality):
                raise SystemExit(f"mask chunk quality mismatch: {path.name}")
            all_frames.extend(int(v) for v in z["chunk_F"])
            for k in z.files:
                if not k.startswith("f"):
                    continue
                f, role = k[1:].split("_")
                if role in FIGHTERS:
                    table.setdefault(int(f), {})[role] = np.asarray(z[k])
    all_frames = sorted(set(all_frames))
    if all_frames != expected_frames:
        raise SystemExit("mask chunks are legacy/incomplete; rerun 03_masks.py")
    return table, all_frames


def padded_mask_box(mask, pad=.08):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    d = pad * max(x1 - x0, y1 - y0)
    return [max(0.0, x0 - d), max(0.0, y0 - d),
            min(float(w), x1 + d), min(float(h), y1 + d)]


def mesh_reducer(est, out_dir, grid=48):
    """Cluster the fixed MHR rest topology and preserve oriented triangles."""
    rest = (est.model.head_pose.mhr.character_torch.mesh.rest_vertices
            .detach().float().cpu().numpy())
    faces = est.faces.astype(np.int64)
    lo = rest.min(axis=0)
    scale = (grid - 1) / max(float(np.ptp(rest, axis=0).max()), 1e-9)
    vox = np.floor((rest - lo) * scale + .5).astype(np.int32)
    _, labels = np.unique(vox, axis=0, return_inverse=True)
    counts = np.bincount(labels).astype(np.float32)
    reduced_faces = labels[faces]
    keep = ((reduced_faces[:, 0] != reduced_faces[:, 1]) &
            (reduced_faces[:, 1] != reduced_faces[:, 2]) &
            (reduced_faces[:, 0] != reduced_faces[:, 2]))
    reduced_faces = reduced_faces[keep]
    # De-duplicate geometrically identical triangles without changing the
    # winding of the first occurrence (needed by the shaded mesh renderer).
    _, first = np.unique(np.sort(reduced_faces, axis=1), axis=0,
                         return_index=True)
    reduced_faces = reduced_faces[np.sort(first)].astype(np.int32)
    np.savez_compressed(out_dir / "mesh_topology.npz",
                        schema=np.array(SCHEMA, np.int16),
                        faces=reduced_faces,
                        n_vertices=np.array(len(counts), np.int32),
                        source_vertices=np.array(len(rest), np.int32))
    print(f"MHR mesh: {len(rest)} -> {len(counts)} vertices, "
          f"{len(reduced_faces)} faces", flush=True)

    def reduce(vertices):
        out = np.zeros((len(counts), 3), np.float32)
        np.add.at(out, labels, np.asarray(vertices, np.float32))
        out /= counts[:, None]
        return out

    return reduce, len(counts)


def empty(shape, dtype):
    return np.zeros(shape, dtype=dtype)


def mask_digest(masks, master_frames):
    """Bind a pose cache to the exact SAM masks that conditioned it."""
    digest = hashlib.sha256()
    for frame in master_frames:
        digest.update(np.asarray([frame], np.int32).tobytes())
        for role in FIGHTERS:
            runs = masks.get(frame, {}).get(role)
            digest.update(role.encode("ascii") + b"\0")
            if runs is None:
                digest.update(b"absent\0")
            else:
                arr = np.asarray(runs, np.uint32)
                digest.update(np.asarray([len(arr)], np.int32).tobytes())
                digest.update(arr.tobytes())
    return digest.hexdigest()


def cache_ok(path, master_frames, n_vertices, expected_digest):
    if not path.exists():
        return False
    try:
        z = np.load(path)
        return (int(z["schema"]) == SCHEMA and
                np.array_equal(z["red_F"], np.asarray(master_frames, np.int32)) and
                z["red_V"].shape[1:] == (n_vertices, 3) and
                str(z["mask_digest"]) == expected_digest)
    except Exception:
        return False


def main():
    import cv2
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="analysis frames")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--chunk-start", type=int, default=1,
                    help="first 1-based pose chunk to process")
    ap.add_argument("--chunk-stop", type=int, default=0,
                    help="last 1-based pose chunk to process (default: final)")
    ap.add_argument("--reverse", action="store_true",
                    help="process the selected chunk range from the end")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    src = common.video_path()
    meta = common.probe(src)
    W, H = meta["wh"]
    masks, frames = load_masks()
    if args.limit:
        frames = frames[:args.limit]
    chunks = [frames[i:i + args.chunk] for i in range(0, len(frames), args.chunk)]
    first = max(0, args.chunk_start - 1)
    stop = min(len(chunks), args.chunk_stop or len(chunks))
    if first >= stop:
        raise SystemExit(f"empty chunk range {args.chunk_start}..{args.chunk_stop}")
    selected = list(enumerate(chunks))[first:stop]
    if args.reverse:
        selected.reverse()
    out_dir = common.WORK / "pose3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    est = estimator()
    reduce_mesh, n_vertices = mesh_reducer(est, out_dir)
    print(f"{len(frames)} mask-aligned frames, {len(chunks)} pose chunks", flush=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    t_all = time.time()
    written = []
    for ci, ch in selected:
        cf = out_dir / f"chunk_{ch[0]:06d}.npz"
        written.append(cf.name)
        digest = mask_digest(masks, ch)
        if not args.redo and cache_ok(cf, ch, n_vertices, digest):
            print(f"  chunk {ci + 1}/{len(chunks)} cached", flush=True)
            continue
        acc = {r: {"F": [], "K3": [], "K2": [], "CT": [], "FL": [],
                   "V": [], "ok": []} for r in FIGHTERS}
        cap.set(cv2.CAP_PROP_POS_FRAMES, ch[0])
        wanted = set(ch)
        i = ch[0]
        t0 = time.time()
        while i <= ch[-1]:
            got, fr = cap.read()
            if not got:
                break
            if i not in wanted:
                i += 1
                continue
            role_masks = {}
            role_boxes = {}
            for role, runs in masks.get(i, {}).items():
                m = common.rle_decode(runs, (H, W))
                b = padded_mask_box(m)
                if b is not None and int(m.sum()) >= 300:
                    role_masks[role] = m
                    role_boxes[role] = b
            roles = [r for r in FIGHTERS if r in role_masks]
            results = []
            if roles:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                bxs = np.asarray([role_boxes[r] for r in roles], np.float32)
                ms = np.asarray([role_masks[r] for r in roles], bool)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        results = est.process_one_image(
                            rgb, bboxes=bxs, masks=ms, use_mask=True,
                            inference_type="body")
                except Exception as exc:
                    print(f"  frame {i}: batch {type(exc).__name__}; retry singly",
                          flush=True)
                    results = []
                    for b, m in zip(bxs, ms):
                        try:
                            with contextlib.redirect_stdout(io.StringIO()):
                                one = est.process_one_image(
                                    rgb, bboxes=b.reshape(1, 4),
                                    masks=m.reshape(1, H, W), use_mask=True,
                                    inference_type="body")
                            results.append(one[0] if one else None)
                        except Exception as exc2:
                            print(f"  frame {i}: {type(exc2).__name__}: {exc2}",
                                  flush=True)
                            results.append(None)
            by_role = {r: (results[q] if q < len(results) else None)
                       for q, r in enumerate(roles)}
            for role in FIGHTERS:
                a = acc[role]
                a["F"].append(i)
                out = by_role.get(role)
                if out is None:
                    a["K3"].append(empty((70, 3), np.float16))
                    a["K2"].append(empty((70, 2), np.float16))
                    a["CT"].append(empty((3,), np.float32))
                    a["FL"].append(0.0)
                    a["V"].append(empty((n_vertices, 3), np.float16))
                    a["ok"].append(False)
                else:
                    a["K3"].append(out["pred_keypoints_3d"][:70].astype(np.float16))
                    a["K2"].append(out["pred_keypoints_2d"][:70].astype(np.float16))
                    a["CT"].append(out["pred_cam_t"].astype(np.float32))
                    a["FL"].append(float(out["focal_length"]))
                    a["V"].append(reduce_mesh(out["pred_vertices"]).astype(np.float16))
                    a["ok"].append(True)
            if len(acc["red"]["F"]) % 10 == 0:
                print(f"    {len(acc['red']['F'])}/{len(ch)} frames "
                      f"({(time.time() - t0) / len(acc['red']['F']):.2f}s/frame)",
                      flush=True)
            i += 1

        payload = {"schema": np.array(SCHEMA, np.int16),
                   "mask_digest": np.array(digest)}
        for role in FIGHTERS:
            a = acc[role]
            payload[f"{role}_F"] = np.asarray(a["F"], np.int32)
            payload[f"{role}_K3"] = np.asarray(a["K3"], np.float16).reshape(-1, 70, 3)
            payload[f"{role}_K2"] = np.asarray(a["K2"], np.float16).reshape(-1, 70, 2)
            payload[f"{role}_CT"] = np.asarray(a["CT"], np.float32).reshape(-1, 3)
            payload[f"{role}_FL"] = np.asarray(a["FL"], np.float32)
            payload[f"{role}_V"] = np.asarray(a["V"], np.float16).reshape(-1, n_vertices, 3)
            payload[f"{role}_ok"] = np.asarray(a["ok"], bool)
        np.savez_compressed(cf, **payload)
        print(f"  chunk {ci + 1}/{len(chunks)}: {len(acc['red']['F'])} frames "
              f"in {time.time() - t0:.0f}s", flush=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    cap.release()
    if len(selected) != len(chunks) or args.reverse:
        print(f"SAM 3D Body bounded worker done in "
              f"{(time.time() - t_all) / 60:.1f} min", flush=True)
        return
    written = [f"chunk_{ch[0]:06d}.npz" for ch in chunks]
    json.dump({"schema": SCHEMA, "model": "SAM 3D Body MHR",
               "conditioning": "SAM 3.1 mask-level hogu-identified fighter masks + tight crops",
               "cache_binding": "SHA-256 of exact per-chunk fighter-mask RLE",
               "stride": common.STRIDE, "fps": common.FPS_PROC,
               "F": frames, "mesh_vertices": n_vertices, "chunks": written},
              open(out_dir / "pose_index.json", "w"), indent=1)
    print(f"SAM 3D Body done in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Detect + track the two fighters and the referee by what they wear.

A taekwondo bout hands identity to us: the chung fighter wears a blue hogu,
the hong fighter red, the referee white. So identity here is not an IoU
accident - tracklets (detector + greedy IoU continuity) are classified by
the colour of the torso region across their whole life, and the per-role
timeline keeps, at each processed frame, the box of the highest-scoring
tracklet claiming that role. A clinch that merges both fighters into one
ambiguous box classifies as neither and leaves a gap: absent, not guessed.

Output work/track/fighters.json:
  { fps_src, stride, wh, boxes: {red|blue|ref: {master_frame: [x0,y0,x1,y1]}},
    report: {...}, tracklets: [...] }
Plus work/track/qa/*.jpg contact sheets to eyeball the role assignment.

Usage: 02_track.py [--limit N]   (N processed frames, for a quick pass)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DET_CONF = 0.35
IOU_MIN = 0.30
COAST = 12          # processed frames a tracklet survives unmatched (0.8 s)
MIN_LEN = 15        # processed frames (1 s) for a tracklet to be considered
BRIDGE = 12         # per-role box holes bridged linearly (continuity both sides)
COLOR_SAMPLES = 60  # frames sampled per tracklet for colour scores

_det = None


def detector():
    global _det
    if _det is None:
        import torch
        import torchvision
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _det = (torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
                .eval().to(dev), dev)
    return _det


def detect_persons(fr):
    import cv2
    import torch
    model, dev = detector()
    t_ = (torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
          .permute(2, 0, 1).float().div(255).to(dev))
    with torch.no_grad():
        out = model([t_])[0]
    b = out["boxes"].cpu().numpy()
    keep = (out["labels"].cpu().numpy() == 1) & (out["scores"].cpu().numpy() > DET_CONF)
    return b[keep]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def torso_scores(fr, box):
    """(red, blue, white) pixel fractions of the hogu region of one box.

    The region is the middle 60% width, 22%..58% height of the box - chest
    protector territory, above the (blue) mat that fills the box bottom.
    """
    import cv2
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = fr.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    rx0 = max(0, x0 + int(0.20 * bw)); rx1 = min(w, x1 - int(0.20 * bw))
    ry0 = max(0, y0 + int(0.22 * bh)); ry1 = min(h, y0 + int(0.58 * bh))
    if rx1 - rx0 < 3 or ry1 - ry0 < 3:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(fr[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    n = H.size
    red = (((H <= 8) | (H >= 168)) & (S > 80) & (V > 45)).sum() / n
    # the chung hogu is a dark navy under the dome light - low value, mid sat
    blue = ((H >= 95) & (H <= 135) & (S > 50) & (V > 30)).sum() / n
    white = ((S < 60) & (V > 110)).sum() / n
    return float(red), float(blue), float(white)


def classify(scores):
    """median (red, blue, white) fractions -> role or None"""
    r, b, w = scores
    if r > 0.08 and r > 1.8 * b:
        return "red"
    if b > 0.08 and b > 1.8 * r:
        return "blue"
    if w > 0.18 and r < 0.08 and b < 0.08:
        return "ref"
    return None


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    src = common.video_path()
    meta = common.probe(src)
    W, H = meta["wh"]
    out_dir = common.WORK / "track"
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    f1 = args.limit * common.STRIDE if args.limit else None
    active, done = [], []
    frames_bgr = {}          # sampled frames kept for colour scoring + QA
    n_proc = 0
    t0 = time.time()
    for i, fr in common.iter_frames(src, f1=f1):
        dets = detect_persons(fr)
        n_old = len(active)
        pairs = sorted(((iou(active[a]["last"], dets[b]), a, b)
                        for a in range(n_old) for b in range(len(dets))
                        if iou(active[a]["last"], dets[b]) >= IOU_MIN),
                       reverse=True)
        used_a, used_b = set(), set()
        for v, a, b in pairs:
            if a in used_a or b in used_b:
                continue
            used_a.add(a); used_b.add(b)
            tr = active[a]
            tr["boxes"][i] = [float(x) for x in dets[b]]
            tr["sc"].append(torso_scores(fr, dets[b]))
            tr["last"] = dets[b]
            tr["miss"] = 0
        survivors = []
        for a in range(n_old):
            tr = active[a]
            if a in used_a:
                survivors.append(tr)
            else:
                tr["miss"] += 1
                (survivors if tr["miss"] <= COAST else done).append(tr)
        for b in range(len(dets)):
            if b not in used_b:
                survivors.append({"boxes": {i: [float(x) for x in dets[b]]},
                                  "sc": [torso_scores(fr, dets[b])],
                                  "last": dets[b], "miss": 0})
        active = survivors
        if n_proc % 25 == 0:
            frames_bgr[i] = fr.copy()
        n_proc += 1
        if n_proc % 250 == 0:
            print(f"  detect {n_proc} frames, {len(active)} active, "
                  f"{(time.time() - t0) / n_proc:.2f}s/frame", flush=True)
    done += active

    # ---- classify tracklets by the torso colours collected while tracking
    tracklets = []
    for tr in done:
        idxs = sorted(tr["boxes"])
        if len(idxs) < MIN_LEN:
            continue
        sc = tr["sc"]
        if not sc:
            continue
        med = tuple(float(np.median([s[k] for s in sc])) for k in range(3))
        hs = [tr["boxes"][q][3] - tr["boxes"][q][1] for q in idxs]
        y1s = [tr["boxes"][q][3] for q in idxs]
        y1_med = float(np.median(y1s))
        # only people standing on the mat can hold a role; the crowd behind
        # the barrier never reaches the lower half of the frame
        role = classify(med) if y1_med >= 0.42 * H else None
        tracklets.append({"f0": idxs[0], "f1": idxs[-1], "n": len(idxs),
                          "role": role, "scores": [round(v, 3) for v in med],
                          "h_med": float(np.median(hs)),
                          "y1_med": round(y1_med, 1), "boxes": tr["boxes"]})

    # ---- per-role timeline: highest colour score among claimants per frame
    score_i = {"red": 0, "blue": 1, "ref": 2}
    boxes = {r: {} for r in common.ROLES}
    claim = {r: {} for r in common.ROLES}
    for tk in tracklets:
        if tk["role"] is None:
            continue
        r = tk["role"]
        s = tk["scores"][score_i[r]]
        for q, b in tk["boxes"].items():
            q = int(q)
            if q not in claim[r] or s > claim[r][q]:
                claim[r][q] = s
                boxes[r][q] = b

    # ---- bridge holes <= BRIDGE processed frames (continuity both sides)
    st = common.STRIDE
    bridged = {r: 0 for r in common.ROLES}
    for r in common.ROLES:
        idxs = sorted(boxes[r])
        for lo, hi in zip(idxs[:-1], idxs[1:]):
            gap_p = (hi - lo) // st - 1
            if 0 < gap_p <= BRIDGE:
                for q in range(lo + st, hi, st):
                    t = (q - lo) / (hi - lo)
                    boxes[r][q] = [a + (b - a) * t
                                   for a, b in zip(boxes[r][lo], boxes[r][hi])]
                    bridged[r] += 1

    # ---- QA contact sheets
    col = {"red": (50, 62, 194), "blue": (150, 139, 124), "ref": (203, 228, 239)}
    for j, (i, fr) in enumerate(sorted(frames_bgr.items())[::4][:12]):
        im = fr.copy()
        for r in common.ROLES:
            if i in boxes[r]:
                x0, y0, x1, y1 = [int(v) for v in boxes[r][i]]
                cv2.rectangle(im, (x0, y0), (x1, y1), col[r], 2)
                cv2.putText(im, r, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col[r], 1, cv2.LINE_AA)
        cv2.imwrite(str(qa_dir / f"f{i:05d}.jpg"), im)

    roles_found = {r: len([t for t in tracklets if t["role"] == r])
                   for r in common.ROLES}
    report = {"n_proc_frames": n_proc,
              "n_tracklets": len(tracklets),
              "tracklets_by_role": roles_found,
              "unclassified": len([t for t in tracklets if t["role"] is None]),
              "frames_with_box": {r: len(boxes[r]) for r in common.ROLES},
              "bridged": bridged,
              "coverage": {r: round(len(boxes[r]) / max(n_proc, 1), 3)
                           for r in common.ROLES}}
    obj = {"fps_src": common.FPS_SRC, "stride": common.STRIDE, "wh": [W, H],
           "boxes": {r: {str(i): [round(float(v), 1) for v in b]
                         for i, b in sorted(boxes[r].items())}
                     for r in common.ROLES},
           "report": report,
           "tracklets": [{k: v for k, v in t.items() if k != "boxes"}
                         for t in tracklets]}
    json.dump(common.jnum(obj), open(out_dir / "fighters.json", "w"))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()

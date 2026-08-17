"""Store selftest: the artifacts the web app trusts, checked before shipping.

Checks are counts and ranges, not vibes; a failure prints what broke and the
script exits nonzero. Run after 05/06: python3 tests/selftest.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "store"
sys.path.insert(0, str(ROOT / "pipeline"))
import common  # noqa: E402

checks = 0
fails = []


def ok(cond, what):
    global checks
    checks += 1
    if not cond:
        fails.append(what)


def main():
    bout = json.load(open(STORE / "bout.json"))
    ok(bout["fighters"]["red"]["label"] == "L.S. GALLO", "red label")
    ok(bout["fighters"]["blue"]["label"] == "Y. XU", "blue label")
    dur = bout["source"]["duration_s"]
    ok(300 < dur < 360, f"duration {dur}")
    ok("SAM 3.1 concept segmentation" in bout["models"]["segmentation"],
       "segmentation provenance is not SAM 3.1 video segmentation")
    ok("mask-conditioned" in bout["models"]["pose3d"],
       "pose provenance is not mask-conditioned SAM 3D Body")

    for role in ("red", "blue"):
        s = json.load(open(STORE / "skel3d" / f"{role}.json"))
        n = s["n"]
        ok(n == len(s["frames"]) == len(s["root_w"]) == len(s["F"]),
           f"{role}: n/frames/root_w/F mismatch")
        ok(len(s["names"]) == 24 and len(s["bones"]) == 24, f"{role}: joint subset")
        ok(s["fps"] >= 10, f"{role}: fps {s['fps']}")
        ok("synthetic" not in s.get("source", "").lower(),
           f"{role}: synthetic skeleton fixture shipped")
        ok(s["F"][-1] / 30.0 > dur - 1.0,
           f"{role}: pose ends at {s['F'][-1] / 30.0:.1f}s of {dur:.1f}s")
        present = [f for f in s["frames"] if f is not None]
        ok(len(present) / n > 0.5, f"{role}: only {len(present)}/{n} frames present")
        mx = 0
        for f in present[:: max(1, len(present) // 200)]:
            for j in f:
                if j is not None:
                    mx = max(mx, abs(j[0]), abs(j[1]), abs(j[2]))
        ok(mx < 3000, f"{role}: joint {mx} mm from root")
        roots = [r[1] for r in s["root_w"] if r is not None]
        med = sorted(roots)[len(roots) // 2]
        ok(300 < med < 1500, f"{role}: median root height {med} mm")

        a = json.load(open(STORE / f"angles_{role}.json"))
        ok(len(a["t"]) == n, f"{role}: angles length")
        for k, v in a["deg"].items():
            ok(len(v) == n, f"{role}: deg.{k} length")
            vals = [x for x in v if x is not None]
            ok(all(0 <= x <= 185 for x in vals), f"{role}: deg.{k} range")
            ok(len(vals) > 0.4 * n, f"{role}: deg.{k} coverage {len(vals)}/{n}")

        vv = json.load(open(STORE / f"vectors_{role}.json"))
        ok(len(vv["t"]) == n, f"{role}: vectors length")
        sp = [x for x in vv["speed_mm_s"]["l_ank"] if x is not None]
        ok(sp and max(sp) < 30000, f"{role}: ankle speed max {max(sp) if sp else 0}")

    hl = json.load(open(STORE / "highlights.json"))
    ts = [e["t"] for e in hl["events"]]
    ok(ts == sorted(ts), "highlights sorted")
    ok(all(0 <= t <= dur for t in ts), "highlight times in bout")
    kicks = [e for e in hl["events"] if e["type"] == "kick"]
    ok(len(kicks) >= 4, f"only {len(kicks)} kicks found")
    for e in hl["events"]:
        ok(e["fighter"] in ("red", "blue"), "event fighter")

    risk = json.load(open(STORE / "risk.json"))
    ok(set(risk["fighters"]) == {"red", "blue"}, "risk fighters")
    ok(len(risk["method"]) >= 3, "risk method strings")

    tl = json.load(open(STORE / "timeline.json"))
    nf = len(tl["F"])
    ok(tl["F"][-1] / 30.0 > dur - 1.0,
       f"timeline ends at {tl['F'][-1] / 30.0:.1f}s of {dur:.1f}s")
    for r in ("red", "blue", "ref"):
        ok(len(tl["px"][r]) == nf, f"timeline px {r}")
    for r in ("red", "blue"):
        vals = [v for v in tl["px"][r] if v is not None]
        ok(len(set(vals)) > 50, f"timeline {r} mask areas look synthetic")
        ok(len(vals) > 0.65 * nf,
           f"timeline {r} mask coverage only {len(vals)}/{nf}")
        last = max(f for f, v in zip(tl["F"], tl["px"][r]) if v is not None)
        ok(last / 30.0 > dur - 1.0,
           f"timeline {r} ends at {last / 30.0:.1f}s of {dur:.1f}s")

    # Geometry gates catch a text-conditioned "person" candidate that is
    # actually a broad piece of the blue mat.  Coverage/count checks alone do
    # not distinguish that failure from a fighter silhouette.
    _, fighter_boxes = common.load_fighters()
    mask_index = json.load(open(ROOT / "work" / "masks" / "masks_index.json"))
    ok(mask_index.get("quality_version") == "mask-hogu-cc-v5",
       "fighter masks do not use connected-instance quality gate")
    W, H = mask_index["wh"]
    bad_masks = []
    frame_masks = {}
    for path in common.mask_chunks():
        z = np.load(path)
        for key in z.files:
            if not key.startswith("f") or not key.endswith(("_red", "_blue")):
                continue
            frame = int(key[1:].split("_")[0])
            role = key.rsplit("_", 1)[-1]
            mask = common.rle_decode(z[key], (H, W))
            frame_masks.setdefault(frame, {})[role] = mask
            ys, xs = np.nonzero(mask)
            if not len(xs):
                bad_masks.append((frame, role, "empty"))
                continue
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            plausible = (ys.max() + 1 >= .58 * H and bh >= .20 * H and
                         bw <= .55 * W)
            if frame not in fighter_boxes[role]:
                compactness = float(mask.sum()) / float(bw * bh)
                plausible = (plausible and bh >= .25 * H and
                             ys.max() + 1 >= .60 * H and bw <= 1.8 * bh and
                             .14 * W <= .5 * (xs.min() + xs.max() + 1) <= .86 * W and
                             compactness >= .18)
                if role == "blue":
                    plausible = plausible and ys.min() <= .62 * H
            colour_key = f"c{frame}_{role}"
            if colour_key not in z:
                plausible = False
            else:
                own, rival, own_px = [float(v) for v in z[colour_key]]
                plausible = (plausible and own >= .045 and
                             own_px >= max(20, .01 * mask.sum()) and
                             own >= 1.2 * rival)
            if not plausible:
                bad_masks.append((frame, role, "shape"))
    ok(not bad_masks, f"implausible fighter masks: {bad_masks[:8]}")
    duplicate_roles = []
    for frame, masks in frame_masks.items():
        if set(masks) != {"red", "blue"}:
            continue
        inter = int((masks["red"] & masks["blue"]).sum())
        union = int((masks["red"] | masks["blue"]).sum())
        if union and inter / union > .75:
            duplicate_roles.append((frame, round(inter / union, 3)))
    ok(not duplicate_roles,
       f"red/blue masks collapse to one instance: {duplicate_roles[:8]}")

    for m, floor in (("bout.mp4", 3e6), ("seg.mp4", 3e6), ("pose.mp4", 1e6),
                     ("mesh.mp4", 1e6), ("seg_poster.jpg", 5e3),
                     ("pose_poster.jpg", 5e3), ("mesh_poster.jpg", 5e3)):
        f = STORE / "media" / m
        ok(f.exists() and f.stat().st_size > floor, f"media/{m}")

    for w in ("index.html", "styles/main.css", "src/app.js", "src/fightview.js",
              "src/views/bout.js", "src/views/analysis.js", "src/views/method.js",
              "vendor/three.module.js", "vendor/three.core.js", "vendor/OrbitControls.js"):
        ok((ROOT / "web" / w).exists(), f"web/{w}")

    print(f"{checks} checks, {len(fails)} failed")
    for f in fails:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

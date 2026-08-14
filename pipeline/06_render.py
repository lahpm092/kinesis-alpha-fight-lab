"""Render the two overlay clips the web app plays. Runs on the mini (ffmpeg).

seg.mp4   SAM 3 masks burned over the footage: hong sienna, chung slate,
          referee a faint bone wash; fighters get the family 1 px white edge.
pose.mp4  the darkroom cut: footage dimmed to a plate, cleaned SAM 3D Body
          2D projections drawn per fighter; a joint the cleaner dropped is
          simply not drawn.

Both render at the analysis cadence (mask frames at 15 Hz, pose at its own
rate), sized to the source, x264 + faststart for range-request streaming.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"

# BGR tokens from web/styles/main.css
C_RED = (50, 94, 194)      # sienna-2 #C25E32 - hong
C_BLUE = (200, 136, 74)    # steel #4A88C8 - chung (validated against hong)
C_REF = (203, 228, 239)    # bone #EFE4CB
C_EDGE = (255, 255, 255)
C_AMBER = (84, 180, 255)   # #FFB454
A_FILL = {"red": 0.40, "blue": 0.42, "ref": 0.16}
COL = {"red": C_RED, "blue": C_BLUE, "ref": C_REF}

MHR_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 19, 20, 69]
MHR_BONES = [(23, 0), (0, 1), (0, 2), (1, 3), (2, 4), (23, 5), (23, 6),
             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 18),
             (15, 19), (16, 20), (16, 21), (16, 22)]


def ff_writer(dest, w, h, fps):
    return subprocess.Popen(
        [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
         "-preset", "slow", "-movflags", "+faststart", str(dest)],
        stdin=subprocess.PIPE)


def mask_lookup():
    """frame -> {role: rle}; opens one chunk at a time, ordered"""
    files = sorted((common.WORK / "masks").glob("chunk_*.npz"))
    table = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith("f"):
                i, r = k[1:].split("_")
                table.setdefault(int(i), {})[r] = z[k]
    return table


def render_seg():
    import cv2
    src = common.video_path()
    meta = common.probe(src)
    W, H = meta["wh"]
    table = mask_lookup()
    frames = sorted(table)
    out = common.STORE / "media"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "seg.part.mp4"
    ff = ff_writer(part, W, H, common.FPS_PROC)
    kern = np.ones((3, 3), np.uint8)
    poster = None
    n = 0
    for i, fr in common.iter_frames(src):
        if i not in table:
            if frames and i > frames[-1]:
                break
            ff.stdin.write(fr.tobytes())
            continue
        im = fr.copy()
        for r in ("ref", "blue", "red"):
            rle = table[i].get(r)
            if rle is None:
                continue
            m = common.rle_decode(rle, (H, W))
            lay = im[m].astype(np.float32)
            a = A_FILL[r]
            im[m] = (lay * (1 - a) + np.array(COL[r], np.float32) * a).astype(np.uint8)
            if r != "ref":
                edge = (cv2.dilate(m.astype(np.uint8), kern) - m.astype(np.uint8)).astype(bool)
                le = im[edge].astype(np.float32)
                im[edge] = (le * 0.1 + np.array(C_EDGE, np.float32) * 0.9).astype(np.uint8)
        ff.stdin.write(im.tobytes())
        n += 1
        if poster is None and i > 2500 and "red" in table[i] and "blue" in table[i]:
            poster = im.copy()
        if n % 300 == 0:
            print(f"  seg {n}/{len(frames)}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("seg: ffmpeg failed")
    (out / "seg.mp4").unlink(missing_ok=True)
    part.rename(out / "seg.mp4")
    if poster is not None:
        cv2.imwrite(str(out / "seg_poster.jpg"), poster,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"seg.mp4: {n} frames", flush=True)


def render_pose():
    import cv2
    src = common.video_path()
    meta = common.probe(src)
    W, H = meta["wh"]
    skel = {}
    for r in ("red", "blue"):
        z = np.load(common.WORK / f"cleaned_{r}.npz")
        F = z["F"]
        skel[r] = {int(f): (z["K2"][q].astype(float), z["present"][q])
                   for q, f in enumerate(F)}
    all_f = sorted(set(skel["red"]) | set(skel["blue"]))
    step = int(np.median(np.diff(all_f))) if len(all_f) > 1 else common.STRIDE
    fps = common.FPS_SRC / max(step, 1)
    out = common.STORE / "media"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "pose.part.mp4"
    ff = ff_writer(part, W, H, fps)
    poster = None
    n = 0
    for i, fr in common.iter_frames(src, stride=step):
        if all_f and i > all_f[-1]:
            break
        im = (fr.astype(np.float32) * 0.30).astype(np.uint8)
        drawn = 0
        for r in ("red", "blue"):
            s = skel[r].get(i)
            if s is None:
                continue
            K2, pres = s
            pts = {}
            for jj, j in enumerate(MHR_IDX):
                if pres[jj] and np.isfinite(K2[j]).all():
                    pts[jj] = (int(round(K2[j][0])), int(round(K2[j][1])))
            for a, b in MHR_BONES:
                if a in pts and b in pts:
                    cv2.line(im, pts[a], pts[b], COL[r], 2, cv2.LINE_AA)
            for q in pts.values():
                cv2.circle(im, q, 2, C_AMBER, -1, cv2.LINE_AA)
            drawn += len(pts)
        ff.stdin.write(im.tobytes())
        n += 1
        if poster is None and i > 2500 and drawn > 30:
            poster = im.copy()
        if n % 300 == 0:
            print(f"  pose {n}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("pose: ffmpeg failed")
    (out / "pose.mp4").unlink(missing_ok=True)
    part.rename(out / "pose.mp4")
    if poster is not None:
        cv2.imwrite(str(out / "pose_poster.jpg"), poster,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"pose.mp4: {n} frames at {fps:g} fps", flush=True)


def render_master():
    """the plain footage clip the bout view scrubs (already CFR)"""
    out = common.STORE / "media"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "bout.mp4"
    if dest.exists():
        return
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(common.video_path()),
                    "-an", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                    "-movflags", "+faststart", str(dest)], check=True)
    print("bout.mp4 written", flush=True)


def render_clips():
    """the three fastest kicks, cut from the mask render as micro-clips"""
    hlf = common.STORE / "highlights.json"
    seg = common.STORE / "media" / "seg.mp4"
    if not hlf.exists() or not seg.exists():
        print("clips skipped (need highlights + seg.mp4)", flush=True)
        return
    ev = [e for e in json.load(open(hlf))["events"] if e["type"] == "kick"]
    ev.sort(key=lambda e: -e["v_peak_mm_s"])
    picked = []
    for e in ev:
        if all(abs(e["t"] - p["t"]) > 4.0 for p in picked):
            picked.append(e)
        if len(picked) == 3:
            break
    for q, e in enumerate(picked):
        t0 = max(0.0, e["t"] - 1.3)
        dest = common.STORE / "media" / f"kick_{q}.mp4"
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t0:.2f}",
                        "-i", str(seg), "-t", "2.8", "-an",
                        "-c:v", "libx264", "-crf", "21", "-preset", "slow",
                        "-movflags", "+faststart", str(dest)], check=True)
    json.dump([{"file": f"media/kick_{q}.mp4", **{k: e[k] for k in
                ("t", "fighter", "limb", "v_peak_mm_s", "knee_deg", "risk")}}
               for q, e in enumerate(picked)],
              open(common.STORE / "reel.json", "w"), indent=1)
    print(f"{len(picked)} kick clips written", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("seg", "all"):
        render_seg()
    if which in ("pose", "all"):
        render_pose()
    if which in ("master", "all"):
        render_master()
    if which in ("clips", "all"):
        render_clips()

"""Render the three synchronized clips the web app plays (ffmpeg).

seg.mp4   SAM 3.1 masks burned over the footage: hong sienna, chung slate;
          both fighters get the family 1 px white edge.
pose.mp4  the darkroom cut: footage dimmed to a plate, cleaned SAM 3D Body
          2D projections drawn per fighter; a joint the cleaner dropped is
          simply not drawn.
mesh.mp4  the actual reduced MHR surface meshes predicted by SAM 3D Body,
          shaded and projected from the inferred camera onto the darkroom.

All render at their recorded analysis cadence, sized to the source, x264 +
faststart for range-request streaming.
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


def keep_main(m):
    """drop mask islands under 15% of the largest component - kills the
    stray saturated blob a box prompt sometimes grabs, keeps a limb split
    in two by an occluding body"""
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = areas.max() * 0.15
    keep = np.zeros(n, bool)
    keep[1:] = areas >= floor
    return keep[lab]


def mask_lookup():
    """frame -> {role: rle}, plus per-role px stats; one pass over chunks"""
    table = {}
    px = {}
    sampled = []
    for f in common.mask_chunks():
        z = np.load(f)
        sampled.extend(int(v) for v in z["chunk_F"])
        for k in z.files:
            parts = k[1:].split("_") if k[:1] in ("f", "s") else []
            if (len(parts) != 2 or not parts[0].isdigit() or
                    parts[1] not in ("red", "blue", "ref")):
                continue
            i, r = parts
            if k.startswith("f"):
                table.setdefault(int(i), {})[r] = z[k]
            else:
                px.setdefault(r, {})[int(i)] = float(z[k][0])
    return table, px, sorted(set(sampled))


def render_seg():
    import cv2
    src = common.video_path()
    meta = common.probe(src)
    W, H = meta["wh"]
    table, px, sampled = mask_lookup()
    # a referee mask far below the referee's median area is a bridged box
    # that slid onto a broadcast graphic - withhold it, don't paint it
    ref_px = sorted(px.get("ref", {}).values())
    ref_floor = 0.25 * ref_px[len(ref_px) // 2] if ref_px else 0
    frames = sampled
    if not frames:
        raise SystemExit("seg: no SAM 3.1 video-mask frames")
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else common.STRIDE
    out = common.STORE / "media"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "seg.part.mp4"
    ff = ff_writer(part, W, H, common.FPS_SRC / step)
    kern = np.ones((3, 3), np.uint8)
    poster = None
    n = 0
    for i, fr in common.iter_frames(src, stride=step):
        im = fr.copy()
        for r in ("ref", "blue", "red"):
            rle = table.get(i, {}).get(r)
            if rle is None:
                continue
            if r == "ref" and px.get("ref", {}).get(i, 0) < ref_floor:
                continue
            m = keep_main(common.rle_decode(rle, (H, W)))
            lay = im[m].astype(np.float32)
            a = A_FILL[r]
            im[m] = (lay * (1 - a) + np.array(COL[r], np.float32) * a).astype(np.uint8)
            if r != "ref":
                edge = (cv2.dilate(m.astype(np.uint8), kern) - m.astype(np.uint8)).astype(bool)
                le = im[edge].astype(np.float32)
                im[edge] = (le * 0.1 + np.array(C_EDGE, np.float32) * 0.9).astype(np.uint8)
        ff.stdin.write(im.tobytes())
        n += 1
        at = table.get(i, {})
        if poster is None and i > 2500 and "red" in at and "blue" in at:
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


def mesh_lookup():
    """master frame -> role -> (reduced vertices, camera t, focal length)."""
    top = np.load(common.WORK / "pose3d" / "mesh_topology.npz")
    if int(top["schema"]) < 2:
        raise SystemExit("mesh: legacy topology; rerun 04_pose3d.py")
    faces = top["faces"].astype(np.int32)
    table = {}
    for path in common.pose_chunks():
        z = np.load(path)
        if "schema" not in z or int(z["schema"]) < 2 or "red_V" not in z:
            continue
        for role in ("red", "blue"):
            for q, f in enumerate(z[f"{role}_F"]):
                if z[f"{role}_ok"][q]:
                    table.setdefault(int(f), {})[role] = (
                        z[f"{role}_V"][q].astype(np.float32),
                        z[f"{role}_CT"][q].astype(np.float32),
                        float(z[f"{role}_FL"][q]))
    return table, faces


def draw_mhr_mesh(im, record, faces, base):
    """Small painter's-algorithm renderer for the real SAM 3D MHR surface."""
    import cv2
    vertices, cam_t, focal = record
    P = vertices + cam_t[None]
    z = P[:, 2]
    good_z = z > .05
    uv = np.full((len(P), 2), -10000.0, np.float32)
    uv[good_z, 0] = focal * P[good_z, 0] / z[good_z] + im.shape[1] / 2
    uv[good_z, 1] = focal * P[good_z, 1] / z[good_z] + im.shape[0] / 2

    tri3 = P[faces]
    tri2 = uv[faces]
    valid = good_z[faces].all(axis=1) & np.isfinite(tri2).all(axis=(1, 2))
    # Cull wildly off-canvas triangles from failed fits, but allow limbs at
    # the edge of the broadcast frame.
    valid &= ((tri2[..., 0].max(axis=1) >= -4) &
              (tri2[..., 0].min(axis=1) < im.shape[1] + 4) &
              (tri2[..., 1].max(axis=1) >= -4) &
              (tri2[..., 1].min(axis=1) < im.shape[0] + 4))
    idx = np.where(valid)[0]
    if not len(idx):
        return
    normal = np.cross(tri3[idx, 1] - tri3[idx, 0],
                      tri3[idx, 2] - tri3[idx, 0])
    nn = np.linalg.norm(normal, axis=1)
    idx = idx[nn > 1e-9]
    normal = normal[nn > 1e-9]
    nn = nn[nn > 1e-9]
    if not len(idx):
        return
    light = np.array([-.35, -.45, -1.0], np.float32)
    light /= np.linalg.norm(light)
    shade = .30 + .70 * np.abs((normal / nn[:, None]) @ light)
    order = np.argsort(tri3[idx].mean(axis=1)[:, 2])[::-1]
    base = np.asarray(base, np.float32)
    for n in order:
        pts = np.round(tri2[idx[n]]).astype(np.int32)
        colour = tuple(int(v) for v in np.clip(base * shade[n], 0, 255))
        cv2.fillConvexPoly(im, pts, colour, cv2.LINE_AA)


def render_mesh():
    """Render the true SAM 3D Body meshes in camera-aligned 3D."""
    import cv2
    table, faces = mesh_lookup()
    pose_index = json.load(open(common.WORK / "pose3d" / "pose_index.json"))
    frames = [int(f) for f in pose_index.get("F", [])]
    if not frames or not table:
        raise SystemExit("mesh: no SAM 3D mesh frames")
    step = int(pose_index.get("stride", common.STRIDE))
    fps = float(pose_index.get("fps", common.FPS_SRC / max(step, 1)))
    W, H = common.probe(common.video_path())["wh"]
    out = common.STORE / "media"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "mesh.part.mp4"
    ff = ff_writer(part, W, H, fps)
    poster = None
    n = 0
    last = frames[-1]
    for i in range(0, last + 1, step):
        # Subtle vertical gradient keeps the shaded surface readable without
        # pretending there is measured scene geometry behind it.
        y = np.linspace(1.0, .45, H, dtype=np.float32)[:, None, None]
        im = np.broadcast_to(y * np.array([13, 10, 7], np.float32),
                             (H, W, 3)).copy().astype(np.uint8)
        recs = table.get(i, {})
        # Farther inferred body first; individual triangles remain depth-sorted.
        roles = sorted(recs, key=lambda r: float(recs[r][1][2]), reverse=True)
        for role in roles:
            draw_mhr_mesh(im, recs[role], faces, COL[role])
        cv2.putText(im, "SAM 3D BODY / MHR SURFACE", (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, C_REF, 1, cv2.LINE_AA)
        cv2.putText(im, "MODEL-INFERRED MONOCULAR DEPTH", (14, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, C_AMBER, 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
        n += 1
        if poster is None and i > 2500 and len(recs) == 2:
            poster = im.copy()
        if n % 200 == 0:
            print(f"  mesh {n}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("mesh: ffmpeg failed")
    (out / "mesh.mp4").unlink(missing_ok=True)
    part.rename(out / "mesh.mp4")
    if poster is not None:
        cv2.imwrite(str(out / "mesh_poster.jpg"), poster,
                    [cv2.IMWRITE_JPEG_QUALITY, 84])
    print(f"mesh.mp4: {n} real MHR mesh frames at {fps:g} fps", flush=True)


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
    if which in ("mesh", "all"):
        render_mesh()
    if which in ("master", "all"):
        render_master()
    if which in ("clips", "all"):
        render_clips()

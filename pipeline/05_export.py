"""Turn Studio inference into the store the web app renders. Runs on the mini.

Reads work/track/fighters.json, work/pose3d/chunk_*.npz, work/masks/chunk_*.npz
and writes store/: skel3d per fighter (family schema + world placement),
joint-angle and angular-velocity strips, movement-vector series, kick
highlights, and load-screening indicators.

Ground truth discipline, stated everywhere it applies: monocular depth is
model-inferred; the shared fight space is camera-relative with both fighters
re-referenced to one RANSAC ground plane; 30 Hz (or 15 Hz) sampling floors
peak speeds and angular velocities - they are lower bounds, not maxima; and
the load-screening indicators are mechanical threshold events, not medical
findings. A joint the cleaner dropped is absent, not guessed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import skelclean  # noqa: E402

FIGHTERS = ("red", "blue")

# MHR-70 indices
J = {"l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_hip": 9, "r_hip": 10,
     "l_knee": 11, "r_knee": 12, "l_ank": 13, "r_ank": 14, "l_toe": 15,
     "l_heel": 17, "r_toe": 18, "r_heel": 20, "l_wri": 62, "r_wri": 41,
     "neck": 69}

# ~24-joint display subset (family schema, names per sam-3d-body mhr70)
MHR_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle",
             "left_big_toe", "left_small_toe", "left_heel",
             "right_big_toe", "right_small_toe", "right_heel", "neck"]
MHR_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14,
           15, 16, 17, 18, 19, 20, 69]
MHR_BONES = [(23, 0), (0, 1), (0, 2), (1, 3), (2, 4), (23, 5), (23, 6),
             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 18),
             (15, 19), (16, 20), (16, 21), (16, 22)]

V_KICK = 4000        # mm/s ankle-speed floor for a kick highlight
V_HAND = 4500        # mm/s wrist-speed floor
V_BURST = 2200       # mm/s CoM-speed floor for a movement burst
OMEGA_HI = 700.0     # deg/s knee angular velocity flag
EXT_HI = 172.0       # deg terminal knee extension flag
TRUNK_HI = 45.0      # deg trunk lean flag during a kick


def load_pose():
    files = sorted((common.WORK / "pose3d").glob("chunk_*.npz"))
    if not files:
        raise SystemExit("no pose3d chunks; pull from the Studio first")
    out = {}
    for r in FIGHTERS:
        F, K3, K2, CT, ok = [], [], [], [], []
        for f in files:
            z = np.load(f)
            F.append(z[f"{r}_F"]); K3.append(z[f"{r}_K3"].astype(np.float64))
            K2.append(z[f"{r}_K2"].astype(np.float64))
            CT.append(z[f"{r}_CT"].astype(np.float64)); ok.append(z[f"{r}_ok"])
        F = np.concatenate(F)
        order = np.argsort(F)
        out[r] = {"F": F[order], "K3": np.concatenate(K3)[order],
                  "K2": np.concatenate(K2)[order],
                  "CT": np.concatenate(CT)[order],
                  "ok": np.concatenate(ok)[order]}
    return out


def fit_ground(foot_pts):
    """RANSAC plane through heel/toe camera-space points (family fit)"""
    P = np.array(foot_pts)
    if len(P) < 60:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        idx = rng.choice(len(P), 3, replace=False)
        p0, p1, p2 = P[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = np.abs((P - p0) @ n)
        inl = (d < 0.06).sum()
        if best is None or inl > best[0]:
            best = (inl, n, p0)
    if best is None or best[0] < 0.35 * len(P):
        return None
    inl, n, p0 = best
    if n[1] > 0:
        n = -n
    tilt = np.degrees(np.arccos(np.clip(-n[1], -1, 1)))
    if tilt > 35:
        return None
    return {"n": n, "p0": p0, "inlier_frac": round(float(inl) / len(P), 2),
            "tilt_vs_camera_deg": round(float(tilt), 1)}


def world_basis(gn):
    """rotation taking camera space to ground space: y = ground up"""
    up = -gn if gn[1] > 0 else gn
    up = up / np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0])
    fwd = fwd - up * (fwd @ up)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    return np.stack([right, up, fwd])  # rows: world axes in camera coords


def ang(K, a, b, c):
    """angle at b, degrees, per frame; NaN where any joint absent"""
    v1 = K[:, a] - K[:, b]
    v2 = K[:, c] - K[:, b]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = (v1 * v2).sum(axis=1) / (n1 * n2 + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def smooth_series(x, w=5):
    """savgol inside continuously finite runs only; never bridges a hole"""
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    idx = np.where(ok)[0]
    if len(idx) < w:
        return x
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
        if len(run) >= w:
            x[run] = savgol_filter(x[run], w | 1, 2)
    return x


def d_dt(x, fps, w=5):
    """gradient of the smoothed series; NaN where the series is NaN"""
    xs = smooth_series(x, w)
    out = np.full_like(xs, np.nan)
    ok = np.isfinite(xs)
    idx = np.where(ok)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
        if len(run) >= 3:
            out[run] = np.gradient(xs[run], 1.0 / fps)
    return out


def jlist(x, nd=1):
    """np array -> JSON list with null for NaN, rounded"""
    out = []
    for v in np.asarray(x, float):
        out.append(None if not np.isfinite(v) else
                   (int(round(v)) if nd == 0 else round(float(v), nd)))
    return out


def main():
    common.STORE.mkdir(exist_ok=True)
    (common.STORE / "skel3d").mkdir(exist_ok=True)
    d, boxes = common.load_fighters()
    pose = load_pose()

    # ---- clean + shared ground plane (both fighters stand on one floor)
    cleaned, foot = {}, []
    for r in FIGHTERS:
        z = pose[r]
        F = z["F"]
        n = len(F)
        step = int(np.median(np.diff(F))) if n > 1 else common.STRIDE
        fps3 = common.FPS_SRC / max(step, 1)
        # smooth_win 5 (167 ms at 30 Hz): the default 7 flattens kick peaks
        K3o, oko, rep = skelclean.clean3d(z["K3"], z["ok"], fps3, smooth_win=5)
        cleaned[r] = {"F": F, "K3": K3o, "ok": oko, "rep": rep, "fps": fps3,
                      "CT": z["CT"], "K2": z["K2"]}
        for i in range(n):
            if z["ok"][i]:
                for j in (J["l_heel"], J["r_heel"], J["l_toe"], J["r_toe"]):
                    v = z["K3"][i, j] + z["CT"][i]
                    if np.isfinite(v).all():
                        foot.append(v)
        print(f"{r}: {n} frames at {fps3:g} Hz, "
              f"{rep['pct_joints_dropped']}% joint obs dropped", flush=True)
    gp = fit_ground(foot)
    R = world_basis(gp["n"]) if gp else np.eye(3)
    p0 = gp["p0"] if gp else np.zeros(3)
    print(f"ground: {'fit' if gp else 'none'}"
          + (f" inliers {gp['inlier_frac']}" if gp else ""), flush=True)

    # ---- per-fighter export
    bout_fighters = {}
    for r in FIGHTERS:
        c = cleaned[r]
        F, K3, oko, fps3 = c["F"], c["K3"], c["ok"], c["fps"]
        n = len(F)
        T = F / common.FPS_SRC

        # world positions: camera -> ground frame, mm
        root = np.full((n, 3), np.nan)
        Kw = np.full((n, 70, 3), np.nan)
        for i in range(n):
            if not oko[i]:
                continue
            hips = K3[i, [J["l_hip"], J["r_hip"]]]
            if not np.isfinite(hips).all():
                continue
            rt = hips.mean(axis=0)
            root[i] = R @ (rt + c["CT"][i] - p0)
            fin = np.isfinite(K3[i]).all(axis=1)
            Kw[i, fin] = (R @ (K3[i, fin] + c["CT"][i] - p0).T).T

        # ---- skel3d (family schema + world root for the shared fight view)
        frames, root_w = [], []
        broken = 0
        for i in range(n):
            hips = K3[i, [J["l_hip"], J["r_hip"]]]
            if not oko[i] or not np.isfinite(hips).all():
                frames.append(None)
                root_w.append(None)
                continue
            rt = hips.mean(axis=0)
            row = []
            for j in MHR_IDX:
                v = K3[i, j]
                if np.isfinite(v).all():
                    # rotate into the ground frame so the viewer's y is up
                    mm = np.round((R @ (v - rt)) * 1000).astype(int)
                    if np.abs(mm).max() >= 3000:
                        broken += 1
                        row.append(None)
                        continue
                    row.append([int(mm[0]), int(mm[1]), int(mm[2])])
                else:
                    row.append(None)
            frames.append(row)
            root_w.append([int(round(v)) for v in root[i] * 1000]
                          if np.isfinite(root[i]).all() else None)
        rep = dict(c["rep"])
        if broken:
            rep["joints_dropped_range"] = int(broken)
        obj = {"fighter": r, "fps": round(fps3, 3), "t0": round(float(T[0]), 3),
               "n": n, "F": [int(v) for v in F],
               "names": MHR_NAMES, "bones": [list(b) for b in MHR_BONES],
               "frames": frames, "root_w": root_w,
               "up": "y (ground normal); world frame = camera re-referenced "
                     "to the RANSAC floor, camera-relative, model-inferred depth",
               "ground_inlier_frac": gp["inlier_frac"] if gp else None,
               "clean": rep,
               "source": "SAM 3D Body (MHR rig) monocular 3D per frame, "
                         "box-prompted from colour-verified fighter tracks"}
        json.dump(common.jnum(obj),
                  open(common.STORE / "skel3d" / f"{r}.json", "w"),
                  separators=(",", ":"))

        # ---- angles + angular velocities
        series = {
            "knee_l": ang(K3, J["l_hip"], J["l_knee"], J["l_ank"]),
            "knee_r": ang(K3, J["r_hip"], J["r_knee"], J["r_ank"]),
            "hip_l": ang(K3, J["l_sho"], J["l_hip"], J["l_knee"]),
            "hip_r": ang(K3, J["r_sho"], J["r_hip"], J["r_knee"]),
            "elbow_l": ang(K3, J["l_sho"], J["l_elb"], J["l_wri"]),
            "elbow_r": ang(K3, J["r_sho"], J["r_elb"], J["r_wri"]),
        }
        neck_hip = Kw[:, J["neck"]] - (Kw[:, J["l_hip"]] + Kw[:, J["r_hip"]]) / 2
        nh = np.linalg.norm(neck_hip, axis=1)
        series["trunk"] = np.degrees(np.arccos(
            np.clip(neck_hip[:, 1] / (nh + 1e-9), -1, 1)))
        omega = {k: d_dt(v, fps3) for k, v in series.items()}
        json.dump(common.jnum({
            "fighter": r, "fps": round(fps3, 3), "t": jlist(T, 2),
            "deg": {k: jlist(v, 1) for k, v in series.items()},
            "omega_deg_s": {k: jlist(v, 0) for k, v in omega.items()},
            "note": f"sampled at {fps3:g} Hz - peak angular velocity is a "
                    "lower bound; trunk = neck-to-hip-centre lean from the "
                    "ground normal"}),
            open(common.STORE / f"angles_{r}.json", "w"), separators=(",", ":"))

        # ---- movement vectors: world velocities of CoM + striking endpoints
        ends = {"com": root, "l_ank": Kw[:, J["l_ank"]], "r_ank": Kw[:, J["r_ank"]],
                "l_wri": Kw[:, J["l_wri"]], "r_wri": Kw[:, J["r_wri"]],
                "l_toe": Kw[:, J["l_toe"]], "r_toe": Kw[:, J["r_toe"]]}
        vec = {}
        spd = {}
        for k, P in ends.items():
            V = np.stack([d_dt(P[:, a] * 1000, fps3) for a in range(3)], axis=1)
            vec[k] = [None if not np.isfinite(v).all()
                      else [int(round(v[0])), int(round(v[1])), int(round(v[2]))]
                      for v in V]
            spd[k] = np.linalg.norm(V, axis=1)
        json.dump(common.jnum({
            "fighter": r, "fps": round(fps3, 3), "t": jlist(T, 2),
            "v_mm_s": vec,
            "speed_mm_s": {k: jlist(v, 0) for k, v in spd.items()},
            "note": "world-frame velocities of the hip-centre (com) and the "
                    "striking endpoints; camera-relative, model-inferred depth; "
                    "sampling floors the peaks"}),
            open(common.STORE / f"vectors_{r}.json", "w"), separators=(",", ":"))

        bout_fighters[r] = {"clean": rep, "n_frames": n, "fps": round(fps3, 3),
                            "series": series, "omega": omega, "spd": spd,
                            "T": T, "root": root}

        # cleaned 2D projections for the overlay renderer (06)
        present = np.zeros((n, len(MHR_IDX)), bool)
        for i in range(n):
            if frames[i] is not None:
                for jj, v in enumerate(frames[i]):
                    present[i, jj] = v is not None
        np.savez_compressed(common.WORK / f"cleaned_{r}.npz",
                            F=F, K2=c["K2"].astype(np.float16), present=present)

    # ---- highlights: kicks, hand strikes, bursts
    events = []
    for r in FIGHTERS:
        b = bout_fighters[r]
        T, fps3 = b["T"], b["fps"]
        min_gap = max(3, int(round(0.5 * fps3)))
        for limb, key, floor, kind in (("left leg", "l_ank", V_KICK, "kick"),
                                       ("right leg", "r_ank", V_KICK, "kick"),
                                       ("left hand", "l_wri", V_HAND, "hand"),
                                       ("right hand", "r_wri", V_HAND, "hand")):
            s = np.nan_to_num(b["spd"][key])
            pk, _ = find_peaks(s, height=floor, distance=min_gap,
                               prominence=floor * 0.35)
            side = key[0]
            for i in pk:
                knee = b["series"][f"knee_{side}"][i]
                w0, w1 = max(0, i - 3), min(len(T), i + 4)
                om = np.nan_to_num(b["omega"][f"knee_{side}"][w0:w1])
                trunk = np.nan_to_num(b["series"]["trunk"][w0:w1])
                risk = []
                if kind == "kick":
                    if np.isfinite(knee) and knee >= EXT_HI:
                        risk.append("terminal_knee_extension")
                    if np.abs(om).max() >= OMEGA_HI:
                        risk.append("high_knee_omega")
                    if trunk.max() >= TRUNK_HI:
                        risk.append("trunk_compensation")
                events.append({
                    "t": round(float(T[i]), 2), "fighter": r, "type": kind,
                    "limb": limb, "v_peak_mm_s": int(round(s[i])),
                    "knee_deg": None if not np.isfinite(knee) else round(float(knee), 1),
                    "omega_knee_peak": int(round(np.abs(om).max())) if len(om) else None,
                    "trunk_max_deg": round(float(trunk.max()), 1) if len(trunk) else None,
                    "risk": risk})
        s = np.nan_to_num(b["spd"]["com"])
        pk, _ = find_peaks(s, height=V_BURST, distance=min_gap,
                           prominence=V_BURST * 0.4)
        for i in pk:
            events.append({"t": round(float(T[i]), 2), "fighter": r,
                           "type": "burst", "limb": "com",
                           "v_peak_mm_s": int(round(s[i])),
                           "knee_deg": None, "omega_knee_peak": None,
                           "trunk_max_deg": None, "risk": []})
    events.sort(key=lambda e: (e["t"], e["fighter"]))
    json.dump(common.jnum({
        "events": events,
        "thresholds": {"kick_mm_s": V_KICK, "hand_mm_s": V_HAND,
                       "burst_mm_s": V_BURST, "omega_deg_s": OMEGA_HI,
                       "terminal_ext_deg": EXT_HI, "trunk_deg": TRUNK_HI},
        "note": "peaks over model-inferred world velocities; a risk tag is a "
                "mechanical threshold event for load screening, not a medical "
                "finding"}),
        open(common.STORE / "highlights.json", "w"), indent=1)

    # ---- load screening summary
    risk = {}
    for r in FIGHTERS:
        b = bout_fighters[r]
        ev = [e for e in events if e["fighter"] == r and e["type"] == "kick"]
        kn = np.concatenate([np.nan_to_num(b["series"]["knee_l"]),
                             np.nan_to_num(b["series"]["knee_r"])])
        risk[r] = {
            "kicks": len(ev),
            "kicks_flagged": len([e for e in ev if e["risk"]]),
            "terminal_extension_events": len([e for e in ev
                                              if "terminal_knee_extension" in e["risk"]]),
            "high_omega_events": len([e for e in ev if "high_knee_omega" in e["risk"]]),
            "trunk_events": len([e for e in ev if "trunk_compensation" in e["risk"]]),
            "knee_ext_p95_deg": round(float(np.percentile(kn[kn > 0], 95)), 1)
            if (kn > 0).any() else None,
            "max_kick_speed_mm_s": max([e["v_peak_mm_s"] for e in ev], default=None),
        }
    json.dump(common.jnum({"fighters": risk, "method": [
        "terminal_knee_extension: knee angle >= "
        f"{EXT_HI} deg at a kick-speed peak - whipping full extension loads the "
        "posterior chain and the joint capsule",
        f"high_knee_omega: |knee angular velocity| >= {OMEGA_HI} deg/s around "
        "the peak - a lower bound at this sampling rate",
        f"trunk_compensation: trunk lean >= {TRUNK_HI} deg from the ground "
        "normal during the kick window",
        "all indicators are mechanical threshold events over model-inferred "
        "monocular 3D; they screen for load patterns worth reviewing on film, "
        "and diagnose nothing"]}),
        open(common.STORE / "risk.json", "w"), indent=1)

    # ---- timeline (mask px + centroids + fighter distance)
    mask_dir = common.WORK / "masks"
    stats = {}
    for f in sorted(mask_dir.glob("chunk_*.npz")):
        z = np.load(f)
        for k in z.files:
            if k.startswith("s"):
                stats[k] = z[k]
    frames_m = sorted({int(k[1:].split("_")[0]) for k in stats})
    tl = {"F": frames_m, "t": [round(i / common.FPS_SRC, 2) for i in frames_m],
          "px": {r: [] for r in common.ROLES},
          "cx": {r: [] for r in common.ROLES},
          "cy": {r: [] for r in common.ROLES}}
    for i in frames_m:
        for r in common.ROLES:
            s = stats.get(f"s{i}_{r}")
            if s is None or s[0] <= 0:
                tl["px"][r].append(None)
                tl["cx"][r].append(None)
                tl["cy"][r].append(None)
            else:
                tl["px"][r].append(int(s[0]))
                tl["cx"][r].append(round(float(s[1]), 1))
                tl["cy"][r].append(round(float(s[2]), 1))
    # camera-space distance between fighters where both reconstructed
    dist = []
    fa, fb = bout_fighters["red"], bout_fighters["blue"]
    ib = {int(f): q for q, f in enumerate(cleaned["blue"]["F"])}
    for q, f in enumerate(cleaned["red"]["F"]):
        p = ib.get(int(f))
        if p is None:
            continue
        ra, rb = fa["root"][q], fb["root"][p]
        if np.isfinite(ra).all() and np.isfinite(rb).all():
            dist.append([round(float(f) / common.FPS_SRC, 2),
                         round(float(np.linalg.norm(ra - rb)), 2)])
    tl["dist_m"] = dist
    json.dump(common.jnum(tl), open(common.STORE / "timeline.json", "w"),
              separators=(",", ":"))

    # ---- bout meta
    json.dump(common.jnum({
        "title": "Chuncheon 2024 - taekwondo bout",
        "source": {"file": "cb07f89a-ec64-457a-bdd5-b296362f2fb6.MP4",
                   "master": "fight_cfr30.mp4 (CFR 30, source is VFR ~31 fps)",
                   "wh": d["wh"], "duration_s": 328.4},
        "fighters": {
            "red": {"label": "L.S. GALLO", "flag": "MEX", "corner": "hong (red)"},
            "blue": {"label": "Y. XU", "flag": "CHN", "corner": "chung (blue)"}},
        "identity_note": "names read off the broadcast scoreboard; corners "
                         "verified by hogu colour on every tracked frame",
        "models": {
            "segmentation": "SAM 3 (Sam3Tracker, bf16), per-frame box prompts",
            "pose3d": "SAM 3D Body (MHR rig), monocular",
            "detector": "torchvision RetinaNet ResNet50-FPN v2"},
        "compute": "Apple M3 Ultra (Mac Studio), MPS",
        "tracking": d["report"],
        "clean": {r: bout_fighters[r]["clean"] for r in FIGHTERS},
        "ground": {"inlier_frac": gp["inlier_frac"],
                   "tilt_vs_camera_deg": gp["tilt_vs_camera_deg"]} if gp else None,
    }), open(common.STORE / "bout.json", "w"), indent=1)
    print(f"store written: {len(events)} highlight events", flush=True)


if __name__ == "__main__":
    main()

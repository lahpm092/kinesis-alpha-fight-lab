"""Shared skeleton cleaning: gate what the model cannot support, invent nothing.

The failure this module removes is the extra joint that appears out of
nowhere: a keypoint detected on the wrong body when players overlap, or a
low-confidence hallucination. The fix is gating plus temporal consistency,
never smoothing-over - a bad observation is removed and stays removed unless
its gap is short enough (<= max_gap frames) for a linear bridge to be honest.

clean2d (halpe26 image-plane keypoints), rules applied per joint across time:
1. conf gate - score below `conf` (or a non-finite coordinate) is absent.
2. bone gate - a bone longer than `bone_ratio` x its window median length
   (median over conf-passing frames only) drops the child joint - the
   endpoint farther from the root (19) - that frame: a stretched bone means
   the joint landed on someone else's body.
3. jump gate - a joint moving more than `jump_frac` x the window median torso
   length (dist 18-19) per frame of gap between consecutive present frames
   loses the later observation: a joint cannot teleport.
4. gap fill - interior gaps of <= `max_gap` frames are linearly interpolated;
   longer gaps stay NaN, and so do absences at the edges.

clean3d (MHR-70 camera-space metres), the same idea scaled by the body
(median shoulder-hip distance): a frame gate first (too many named joints
jumping at once marks the whole reconstruction bad), then the per-joint jump
gate, gap fill, and an optional light Savitzky-Golay pass applied only inside
runs that are continuously present - smoothing never crosses or fills a hole.

Both functions return a JSON-serialisable report of what each rule dropped,
so captions can state the cleaning instead of hiding it. Pure numpy + scipy,
deterministic, no torch.
"""
import numpy as np
from scipy.signal import savgol_filter

# halpe26 drawing bones, parent first; root = 19 (hip root)
BONES26 = [(19, 18), (18, 17), (17, 0), (18, 5), (18, 6), (5, 7), (7, 9),
           (6, 8), (8, 10), (19, 11), (19, 12), (11, 13), (13, 15), (12, 14),
           (14, 16), (15, 24), (16, 25), (15, 20), (16, 21)]
ROOT26 = 19
TORSO26 = (18, 19)

# MHR-70 named joints (build_biomech3d.py J) used for the 3D frame gate
J70 = {"l_sho": 5, "r_sho": 6, "l_hip": 9, "r_hip": 10, "l_knee": 11,
       "r_knee": 12, "l_ank": 13, "r_ank": 14, "l_toe": 15, "l_heel": 17,
       "r_toe": 18, "r_heel": 20, "neck": 69}


def _depth_from_root(bones, root):
    """graph distance of every joint from the root, over the bone tree"""
    depth = {root: 0}
    changed = True
    while changed:
        changed = False
        for a, b in bones:
            if a in depth and b not in depth:
                depth[b] = depth[a] + 1
                changed = True
            elif b in depth and a not in depth:
                depth[a] = depth[b] + 1
                changed = True
    return depth


def _jump_gate(K, present, j, thresh_per_gap):
    """sequential teleport gate on one joint; returns count dropped.

    Compares each present frame against the last kept one; the allowance
    grows with the frame gap, so a joint parked elsewhere is re-accepted
    only once enough time has passed for the move to be physically plain.
    """
    idx = np.where(present[:, j])[0]
    if len(idx) < 2:
        return 0
    dropped = 0
    last = idx[0]
    for i in idx[1:]:
        gap = i - last
        if np.linalg.norm(K[i, j] - K[last, j]) > thresh_per_gap * gap:
            present[i, j] = False
            dropped += 1
        else:
            last = i
    return dropped


def _gap_fill(out, present, max_gap):
    """linear interp of interior gaps <= max_gap frames; returns count filled"""
    filled = 0
    for j in range(present.shape[1]):
        idx = np.where(present[:, j])[0]
        for a, b in zip(idx[:-1], idx[1:]):
            g = b - a - 1
            if 0 < g <= max_gap:
                t = (np.arange(a + 1, b) - a) / (b - a)
                out[a + 1:b, j] = out[a, j] + t[:, None] * (out[b, j] - out[a, j])
                present[a + 1:b, j] = True
                filled += g
    return filled


def clean2d(kpts, scores, fps, *, conf=0.35, bone_ratio=1.75, jump_frac=0.35,
            max_gap=3):
    """kpts (n,26,2) float, scores (n,26). Returns (kpts_out, present, report).
    kpts_out (n,26,2) float with NaN where absent; present (n,26) bool.

    Gates are expressed per frame of gap, so `fps` is part of the shared
    signature but does not enter the 2D thresholds. Report percentages:
    pct_joints_dropped over all n x 26 observations; pct_gaps_filled over the
    dropped observations.
    """
    K = np.asarray(kpts, float).copy()
    S = np.asarray(scores, float)
    n, nj = K.shape[0], K.shape[1]

    # 1. conf gate (a non-finite coordinate is absent too)
    present = (S >= conf) & np.isfinite(K).all(axis=2)
    dropped_conf = int(n * nj - present.sum())

    # 2. bone gate: median over conf-passing frames only
    depth = _depth_from_root(BONES26, ROOT26)
    dropped_bone = 0
    for a, b in BONES26:
        if a >= nj or b >= nj:
            continue
        both = present[:, a] & present[:, b]
        if both.sum() < 3:
            continue
        L = np.linalg.norm(K[:, a] - K[:, b], axis=1)
        med = float(np.median(L[both]))
        if med <= 0:
            continue
        child = b if depth.get(b, 0) >= depth.get(a, 0) else a
        bad = both & (L > bone_ratio * med)
        dropped_bone += int(bad.sum())
        present[bad, child] = False

    # 3. jump gate, scaled by the window median torso length
    torso_ok = present[:, TORSO26[0]] & present[:, TORSO26[1]]
    torso = (float(np.median(np.linalg.norm(K[torso_ok, TORSO26[0]]
                                            - K[torso_ok, TORSO26[1]], axis=1)))
             if torso_ok.sum() >= 3 else 0.0)
    dropped_jump = 0
    if torso > 0:
        for j in range(nj):
            dropped_jump += _jump_gate(K, present, j, jump_frac * torso)

    # 4. gap fill
    out = K.copy()
    out[~present] = np.nan
    filled = _gap_fill(out, present, max_gap)

    dropped_total = dropped_conf + dropped_bone + dropped_jump
    n_obs = n * nj
    report = {
        "n_frames": int(n), "n_joints": int(nj), "n_obs": int(n_obs),
        "dropped_conf": dropped_conf, "dropped_bone": dropped_bone,
        "dropped_jump": dropped_jump, "dropped_total": dropped_total,
        "gaps_filled": int(filled),
        "pct_joints_dropped": round(100.0 * dropped_total / max(n_obs, 1), 2),
        "pct_gaps_filled": round(100.0 * filled / max(dropped_total, 1), 2),
        "torso_len": round(torso, 2),
    }
    return out, present, report


def clean3d(K3, ok, fps, *, jump_frac=0.30, max_gap=3, smooth_win=7):
    """K3 (n,70,3), ok (n,). Returns (K3_out with NaN frames where dropped,
    ok_out (n,) bool, report). Body scale = median dist(l_sho 5, r_hip 10).

    Frame gate: between consecutive ok frames, if more than 20% of the named
    joints (J70) move farther than jump_frac x body scale x frame gap, the
    later reconstruction is bad as a whole. Then the per-joint jump gate,
    gap fill (<= max_gap frames), and - when smooth_win >= 5 - a light
    Savitzky-Golay (order 2, window rounded up to odd) on each coordinate,
    only inside continuously present runs. ok_out marks frames with at least
    one present joint. Report percentages: pct_joints_dropped over the ok-in
    joint-observations; pct_gaps_filled over the absences before filling.
    """
    K = np.asarray(K3, float).copy()
    okm = np.asarray(ok, bool).copy()
    n, nj = K.shape[0], K.shape[1]
    key = [v for v in sorted(set(J70.values())) if v < nj]
    n_ok_in = int(okm.sum())

    scale = (float(np.median(np.linalg.norm(K[okm, 5] - K[okm, 10], axis=1)))
             if n_ok_in and nj > 10 else 0.0)

    # 2. frame gate
    frames_dropped = 0
    ok2 = okm.copy()
    if scale > 0 and key:
        idx = np.where(okm)[0]
        if len(idx) >= 2:
            last = idx[0]
            for i in idx[1:]:
                gap = i - last
                d = np.linalg.norm(K[i, key] - K[last, key], axis=1)
                if float((d > jump_frac * scale * gap).mean()) > 0.20:
                    ok2[i] = False
                    frames_dropped += 1
                else:
                    last = i

    # 3. per-joint jump gate on the surviving frames
    present = np.zeros((n, nj), bool)
    present[ok2] = np.isfinite(K[ok2]).all(axis=2) if n else False
    joints_dropped = 0
    if scale > 0:
        for j in range(nj):
            joints_dropped += _jump_gate(K, present, j, jump_frac * scale)

    # 4. gap fill
    out = K.copy()
    out[~present] = np.nan
    absent_before_fill = int(n * nj - present.sum())
    filled = _gap_fill(out, present, max_gap)

    # 5. light savgol, only inside continuously present runs
    smoothed_runs = 0
    if smooth_win and smooth_win >= 5:
        w = int(smooth_win) | 1
        for j in range(nj):
            idx = np.where(present[:, j])[0]
            if len(idx) < w:
                continue
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for run in np.split(idx, splits):
                if len(run) >= w:
                    out[run, j] = savgol_filter(out[run, j], w, 2, axis=0)
                    smoothed_runs += 1

    ok_out = present.any(axis=1)
    dropped_total = frames_dropped * nj + joints_dropped
    report = {
        "n_frames": int(n), "n_joints": int(nj), "n_ok_in": n_ok_in,
        "frames_dropped": int(frames_dropped),
        "joints_dropped_jump": int(joints_dropped),
        "gaps_filled": int(filled),
        "smoothed_runs": int(smoothed_runs),
        "n_ok_out": int(ok_out.sum()),
        "body_scale_m": round(scale, 3),
        "pct_joints_dropped": round(100.0 * dropped_total
                                    / max(n_ok_in * nj, 1), 2),
        "pct_gaps_filled": round(100.0 * filled
                                 / max(absent_before_fill, 1), 2),
    }
    return out, ok_out, report

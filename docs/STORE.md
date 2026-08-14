# The store contract

The web app renders artifacts; it never computes a metric. Everything under
`store/` is written by the pipeline, versioned with the repo, and carries its
provenance. A null is a stated absence, never a zero.

Timeline law: every artifact indexes master frames of `work/fight_cfr30.mp4`
(CFR 30 fps cut of the VFR source). Time in seconds is `F / 30.0`. The
analysis cadence is the per-artifact `fps` field.

## bout.json
Source, fighters (`red` = hong = L.S. GALLO MEX, `blue` = chung = Y. XU CHN,
names read off the broadcast scoreboard), models, tracking report, cleaning
reports, ground-plane fit quality.

## skel3d/{red,blue}.json
Family skel3d schema, fight-lab dialect:
- `fps`, `n`, `F[n]` master frame indices, `t0`
- `names[24]`, `bones[24][2]` — MHR-70 display subset
- `frames[n]`: `null` (frame dropped) or 24 × (`null` | `[x,y,z]` mm ints,
  **root-centred, ground frame** — y is the floor normal, not the camera)
- `root_w[n]`: `null` | `[x,y,z]` mm — root in the shared ground frame
  (camera-relative; both fighters share one RANSAC floor)
- `clean`: what the cleaner dropped and why it is allowed to
- `ground_inlier_frac`, `source`

## angles_{red,blue}.json
`t[n]`, `deg.{knee_l,knee_r,hip_l,hip_r,elbow_l,elbow_r,trunk}[n]`,
`omega_deg_s.*[n]`. Nulls where the joints were absent. Peaks are lower
bounds at the sampling rate.

## vectors_{red,blue}.json
`t[n]`, `v_mm_s.{com,l_ank,r_ank,l_wri,r_wri,l_toe,r_toe}[n]` (3-vectors,
ground frame), `speed_mm_s.*[n]`. Same nulls, same lower-bound caveat.

## highlights.json
`events[]`: `{t, fighter, type: kick|hand|burst, limb, v_peak_mm_s,
knee_deg, omega_knee_peak, trunk_max_deg, risk[]}` and the `thresholds`
that made them. A risk tag is a mechanical threshold event, not a finding.

## risk.json
Per-fighter screening counts + the `method` strings, verbatim, for display.

## timeline.json
Mask px counts and centroids per role per processed frame, and `dist_m`
(model-inferred camera-space pair distance).

## media/
`bout.mp4` (plain master), `seg.mp4` (SAM 3 masks burned), `pose.mp4`
(cleaned skeleton projections over the dimmed plate), posters.

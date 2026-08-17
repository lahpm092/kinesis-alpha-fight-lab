/* Method: what ran, on what, and where the honest edges are. */
import { el, put, chip, panel, kv } from "../bits.js";
import { data } from "../store.js";

export async function render(mount) {
  const bout = await data("bout.json");
  let hl = null, risk = null;
  try { hl = await data("highlights.json"); } catch {}
  try { risk = await data("risk.json"); } catch {}

  put(mount,
    el("div", "kicker", "method · provenance before polish"),
    el("div", "h-title", "One camera, and what was done to it"),
    el("div", "h-sub",
      "A single 832×480 broadcast-style clip, variable frame rate, re-cut to a ",
      "constant 30 fps master. Everything below is computed from that file and ",
      "stated with its limits. Nothing here is a measurement of the athletes' ",
      "bodies - it is a model's read of a small video."));

  put(mount, el("div", "section-head", el("span", "t", "The pipeline"),
    el("span", "m", bout.compute)));
  const stages = [
    ["01 · master", "VFR source (~31 fps true rate) → CFR 30 fps x264 master; one shared frame timeline for every stage", "mini"],
    ["02 · identity", "RetinaNet person boxes, greedy-IoU tracklets, then hogu colour over the torso region names each tracklet: hong red, chung blue, referee white. A clinch that merges the pair classifies as one box - the other fighter gets a gap, not a guess", "studio"],
    ["03 · segmentation", bout.models.segmentation + "; SAM 3.1 masks are refreshed on each analysis frame, reduced to one connected instance, assigned by mask-level hong/chung hogu evidence plus temporal spatial proposals, and rejected when they look like the referee, crowd, or mat", "Apple silicon"],
    ["04 · reconstruction", bout.models.pose3d + " per fighter per frame; camera-space joints, camera translation, and reduced MHR surface", "studio"],
    ["05 · export", "skeleton cleaning (conf/jump gates, ≤3-frame gap fill, run-local smoothing), shared RANSAC floor from both fighters' heels and toes, world re-referencing, angles, velocities, screening events", "mini"],
    ["06 · render", "synchronized mask, skeleton, and actual MHR surface clips burned to x264 with the family palette", "mini"],
  ];
  const lp = el("div", "grid grid-2");
  for (const [t, d, host] of stages) {
    lp.append(panel(t, chip(host, host === "studio" ? "accent" : "dim"), el("div", "strip-lab", d)));
  }
  put(mount, lp);

  put(mount, el("div", "section-head", el("span", "t", "Identity & tracking")));
  const tk = bout.tracking;
  const idp = panel("coverage of the processed timeline", null);
  for (const [k, v] of Object.entries(tk.coverage)) {
    put(idp, kv(k, Math.round(v * 1000) / 10, "%", `${tk.frames_with_box[k]} frames with a box`));
  }
  put(idp, el("div", "rule"),
    el("div", "strip-lab", bout.identity_note),
    el("div", "strip-lab", `${tk.n_tracklets} tracklets; ${tk.unclassified} stayed unclassified (crowd, officials, partial figures) and hold no role`));
  put(mount, idp);

  put(mount, el("div", "section-head", el("span", "t", "Cleaning, stated")));
  const cp = el("div", "grid grid-2");
  for (const role of ["red", "blue"]) {
    const c = bout.clean[role];
    const p = panel(bout.fighters[role].label, chip(role === "red" ? "hong" : "chung", role === "red" ? "hong" : "chung"));
    put(p,
      kv("3d samples", c.n_frames),
      kv("whole frames dropped", c.frames_dropped, "", "reconstruction jumped as a whole"),
      kv("joint observations dropped", c.joints_dropped_jump, "", "teleporting joints"),
      kv("gaps bridged", c.gaps_filled, "", "≤ 3 frames, both sides present"),
      kv("joint obs dropped", c.pct_joints_dropped, "%"),
      kv("body scale", c.body_scale_m, "m", "median shoulder–hip"));
    cp.append(p);
  }
  put(mount, cp);

  if (hl && risk) {
    put(mount, el("div", "section-head", el("span", "t", "Thresholds")));
    const tp = panel("screening thresholds", chip("modelled", "accent"));
    const th = hl.thresholds;
    put(tp,
      kv("kick highlight", th.kick_mm_s / 1000, "m/s", "ankle-speed peak floor"),
      kv("hand highlight", th.hand_mm_s / 1000, "m/s", "wrist-speed peak floor"),
      kv("burst", th.burst_mm_s / 1000, "m/s", "hip-centre speed floor"),
      kv("terminal extension", th.terminal_ext_deg, "deg", "knee at kick peak"),
      kv("high angular velocity", th.omega_deg_s, "deg/s", "|ω knee| near peak"),
      kv("trunk compensation", th.trunk_deg, "deg", "lean from floor normal in kick window"));
    for (const m of risk.method) put(tp, el("div", "strip-lab", m));
    put(mount, tp);
  }

  put(mount, el("div", "section-head", el("span", "t", "The honest edges")));
  const limits = [
    "depth is model-inferred: SAM 3D Body reads a monocular frame; the shared fight space is camera-relative and both fighters are re-referenced to one RANSAC floor (inliers " + (bout.ground ? bout.ground.inlier_frac : "—") + ")",
    "sampling floors the peaks: angular velocity and foot speed are computed at the analysis cadence; a faster true peak between samples is invisible",
    "clinches merge boxes: when the detector sees one figure, one fighter holds the box and the other holds a gap; nothing is interpolated across it",
    "identity is clothing: hong/chung come from hogu colour on every tracked frame and the scoreboard names the corners; nobody was identified by face",
    "the source is 832×480: joint precision is bounded by what four hundred rows of pixels can say about a moving body",
  ];
  const lpn = panel("limits that stay in the record", null);
  for (const l of limits) put(lpn, el("div", "strip-lab", "— " + l));
  put(mount, lpn);
  return () => {};
}

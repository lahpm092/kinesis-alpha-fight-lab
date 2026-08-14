/* Load & technique: joint-angle strips, angular velocity, foot speed, and
   the load-screening ledger. Left is a solid line, right is dashed - the
   side is carried by the stroke, never by a second hue. Every chart seeks
   the bout on click. */
import { el, put, chip, panel, stat, kv, tc, fmt } from "../bits.js";
import { data } from "../store.js";
import { T, FIGHTER } from "../theme.js";

function lineChart({ t, series, unit = "", h = 130, cursorT = null, onSeek = null }) {
  const wrap = el("div");
  const c = el("canvas", "spark");
  const W = 640, H = h;
  c.width = W * 2; c.height = H * 2;
  c.style.height = H + "px";
  const g = c.getContext("2d");
  g.scale(2, 2);
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.values) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!(hi > lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.06;
  lo -= pad; hi += pad;
  const X = (i) => (t[i] / t[t.length - 1]) * (W - 2) + 1;
  const Y = (v) => (H - 16) - ((v - lo) / (hi - lo)) * (H - 24) + 2;
  const draw = (hoverI) => {
    g.clearRect(0, 0, W, H);
    g.strokeStyle = T.hair;
    g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, W - 1, H - 15.5);
    for (const s of series) {
      g.strokeStyle = s.color;
      g.lineWidth = 1.1;
      g.setLineDash(s.dash ? [4, 3] : []);
      g.beginPath();
      let started = false;
      for (let i = 0; i < s.values.length; i++) {
        const v = s.values[i];
        if (v == null) { started = false; continue; }
        const x = X(i), y = Y(v);
        started ? g.lineTo(x, y) : g.moveTo(x, y);
        started = true;
      }
      g.stroke();
      g.setLineDash([]);
    }
    g.fillStyle = "rgba(179,163,130,0.9)";
    g.font = "8px ui-monospace, Menlo, monospace";
    g.fillText(`${Math.round(lo)}–${Math.round(hi)} ${unit}`, 3, H - 5);
    if (cursorT != null && t.length) {
      const x = (cursorT / t[t.length - 1]) * (W - 2) + 1;
      g.strokeStyle = "rgba(239,228,203,0.35)";
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H - 16); g.stroke();
    }
    if (hoverI != null) {
      const x = X(hoverI);
      g.strokeStyle = "rgba(255,180,84,0.55)";
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H - 16); g.stroke();
      const parts = [tc(t[hoverI])];
      for (const s of series) {
        const v = s.values[hoverI];
        parts.push(`${s.label} ${v == null ? "—" : Math.round(v)}`);
      }
      const txt = parts.join("  ·  ");
      g.font = "9px ui-monospace, Menlo, monospace";
      const tw = g.measureText(txt).width + 10;
      const tx = Math.min(Math.max(x - tw / 2, 2), W - tw - 2);
      g.fillStyle = "rgba(20,16,10,0.92)";
      g.fillRect(tx, 2, tw, 14);
      g.strokeStyle = T.hair;
      g.strokeRect(tx + 0.5, 2.5, tw - 1, 13);
      g.fillStyle = "#EFE4CB";
      g.fillText(txt, tx + 5, 12);
    }
  };
  draw(null);
  const idxAt = (ev) => {
    const r = c.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width;
    const tt = fx * t[t.length - 1];
    let best = 0, bd = Infinity;
    for (let i = 0; i < t.length; i++) {
      const d = Math.abs(t[i] - tt);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };
  c.addEventListener("mousemove", (ev) => draw(idxAt(ev)));
  c.addEventListener("mouseleave", () => draw(null));
  if (onSeek) {
    c.style.cursor = "pointer";
    c.addEventListener("click", (ev) => onSeek(t[idxAt(ev)]));
  }
  const legend = el("div", "filters");
  for (const s of series) {
    const sw = el("span", "");
    sw.style.cssText = `display:inline-block;width:14px;height:0;border-top:2px ${s.dash ? "dashed" : "solid"} ${s.color};vertical-align:middle;margin-right:6px;`;
    legend.append(el("span", "strip-lab", sw, s.label));
  }
  put(wrap, legend, c);
  return wrap;
}

const seek = (t) => { location.hash = `#/bout?t=${t.toFixed(1)}`; };

/* knee phase portrait: angle against angular velocity - a tight, repeatable
   loop is a repeatable technique; excursions to the far right at high |ω|
   are the whipping full extensions the screening ledger counts */
function phasePortrait(ang) {
  const c = el("canvas", "spark");
  const W = 300, H = 220;
  c.width = W * 2; c.height = H * 2;
  c.style.height = H + "px";
  c.style.maxWidth = W + "px";
  const g = c.getContext("2d");
  g.scale(2, 2);
  g.strokeStyle = T.hair;
  g.strokeRect(0.5, 0.5, W - 1, H - 15.5);
  const X = (a) => 1 + (a / 185) * (W - 2);
  const YMAX = 1600;
  const Y = (w) => (H - 16) / 2 - (Math.max(-YMAX, Math.min(YMAX, w)) / YMAX) * ((H - 20) / 2);
  g.strokeStyle = "rgba(58,47,31,0.9)";
  g.beginPath(); g.moveTo(1, Y(0)); g.lineTo(W - 1, Y(0)); g.stroke();
  // terminal-extension guide
  g.setLineDash([2, 3]);
  g.strokeStyle = "rgba(197,107,74,0.5)";
  g.beginPath(); g.moveTo(X(172), 1); g.lineTo(X(172), H - 16); g.stroke();
  g.setLineDash([]);
  for (const [key, dash] of [["knee_l", false], ["knee_r", true]]) {
    const A = ang.deg[key], O = ang.omega_deg_s[key];
    g.strokeStyle = "rgba(232,155,62,0.34)";
    g.setLineDash(dash ? [3, 3] : []);
    g.beginPath();
    let started = false;
    for (let i = 0; i < A.length; i++) {
      if (A[i] == null || O[i] == null) { started = false; continue; }
      const x = X(A[i]), y = Y(O[i]);
      started ? g.lineTo(x, y) : g.moveTo(x, y);
      started = true;
    }
    g.stroke();
    g.setLineDash([]);
  }
  g.fillStyle = "rgba(179,163,130,0.9)";
  g.font = "8px ui-monospace, Menlo, monospace";
  g.fillText("0–185 deg →", 3, H - 5);
  g.fillText("±1600 deg/s ↑", 3, 10);
  g.fillText("172°", X(172) - 10, H - 20);
  return c;
}

export async function render(mount) {
  const bout = await data("bout.json");
  const F = bout.fighters;
  let hl = { events: [] }, risk = null;
  try { hl = await data("highlights.json"); } catch {}
  try { risk = await data("risk.json"); } catch {}

  put(mount,
    el("div", "kicker", "load & technique · mechanical threshold events, not medical findings"),
    el("div", "h-title", "What the joints were asked to do"),
    el("div", "h-sub",
      "Angles and angular velocities from the cleaned monocular reconstruction, ",
      "sampled at the analysis cadence - peaks are ", el("em", "", "lower bounds"), ". ",
      "The screening ledger counts mechanical threshold events worth reviewing ",
      "on film; it diagnoses nothing."));

  for (const role of ["red", "blue"]) {
    let ang = null, vec = null;
    try {
      [ang, vec] = await Promise.all([data(`angles_${role}.json`), data(`vectors_${role}.json`)]);
    } catch (e) {
      put(mount, el("div", "section-head", el("span", "t", F[role].label)),
        panel("strips", null, el("div", "strip-lab", "not built — " + e.message)));
      continue;
    }
    const fc = FIGHTER[role];
    put(mount, el("div", "section-head",
      el("span", "t", F[role].label),
      el("span", "m", `${F[role].flag} · ${F[role].corner} · ${ang.fps} hz`)));

    const mk = (title, series, unit) => panel(title, null,
      lineChart({ t: ang.t, series, unit, cursorT: null, onSeek: seek }));
    const grid = el("div", "grid grid-2",
      mk("knee flexion–extension", [
        { label: "left", values: ang.deg.knee_l, color: T.amber2 },
        { label: "right", values: ang.deg.knee_r, color: T.amber2, dash: true }], "deg"),
      mk("hip angle (trunk–thigh)", [
        { label: "left", values: ang.deg.hip_l, color: T.amber2 },
        { label: "right", values: ang.deg.hip_r, color: T.amber2, dash: true }], "deg"),
      mk("foot speed", [
        { label: "left", values: vec.speed_mm_s.l_ank.map((v) => v == null ? null : v / 1000), color: fc.color },
        { label: "right", values: vec.speed_mm_s.r_ank.map((v) => v == null ? null : v / 1000), color: fc.color, dash: true },
        { label: "com", values: vec.speed_mm_s.com.map((v) => v == null ? null : v / 1000), color: T.bone2 }], "m/s"),
      mk("knee angular velocity", [
        { label: "left", values: ang.omega_deg_s.knee_l, color: T.amber2 },
        { label: "right", values: ang.omega_deg_s.knee_r, color: T.amber2, dash: true }], "deg/s"),
      panel("knee phase portrait", chip("technique consistency", "dim"),
        phasePortrait(ang),
        el("div", "strip-lab",
          "angle → against angular velocity ↑ · a tight loop is a repeatable ",
          "technique · past the dashed 172° line at speed is the whipping ",
          "extension the ledger counts")));
    put(mount, grid);

    if (risk && risk.fighters[role]) {
      const r = risk.fighters[role];
      const flagged = hl.events.filter((e) => e.fighter === role && e.risk && e.risk.length);
      const p = panel("screening ledger — " + F[role].label,
        chip(`${r.kicks_flagged}/${r.kicks} kicks flagged`, r.kicks_flagged ? "fail" : "sage"));
      const stats = el("div", "stats",
        stat("kicks over threshold", String(r.kicks)),
        stat("terminal extension", String(r.terminal_extension_events)),
        stat("high ω knee", String(r.high_omega_events)),
        stat("trunk compensation", String(r.trunk_events)),
        stat("knee ext p95", r.knee_ext_p95_deg, "deg"),
        stat("fastest kick", r.max_kick_speed_mm_s == null ? null : fmt(r.max_kick_speed_mm_s / 1000, 1), "m/s"));
      put(p, stats);
      for (const e of flagged.slice(0, 8)) {
        const row = kv(`${tc(e.t)} · ${e.limb}`,
          (e.v_peak_mm_s / 1000).toFixed(1), "m/s",
          e.risk.map((x) => x.replaceAll("_", " ")).join(" · "));
        row.style.cursor = "pointer";
        row.addEventListener("click", () => seek(Math.max(0, e.t - 1.2)));
        put(p, row);
      }
      put(mount, p);
    }
  }

  // provenance footer for this page
  if (risk) {
    const p = panel("how these flags are made", chip("modelled", "accent"));
    for (const m of risk.method) put(p, el("div", "strip-lab", m));
    put(mount, el("div", "rule"), p);
  }
  return () => {};
}

/* The bout: three comparative reads of the same seconds - segmentation,
   cleaned pose, real SAM 3D Body mesh - over one transport, and beneath them
   the fight view: the two of them alone on black, skeleton or body, with
   velocity arrows and live joint angles. */
import { el, put, chip, panel, stat, tc, fmt, scrim } from "../bits.js";
import { data, media, checkSkel } from "../store.js";
import { Skel, Vec } from "../skel.js";
import { Transport } from "../transport.js";
import { createFightScene } from "../fightview.js";
import { FIGHTER } from "../theme.js";

export async function render(mount, ctx) {
  const bout = await data("bout.json");
  let skels = null, vecs = null, hl = { events: [] };
  let skelWhy = null;
  try {
    const [sr, sb, vr, vb] = await Promise.all([
      data("skel3d/red.json"), data("skel3d/blue.json"),
      data("vectors_red.json"), data("vectors_blue.json")]);
    const bad = [...checkSkel(sr), ...checkSkel(sb)];
    if (bad.length) throw new Error(bad.join("; "));
    skels = { red: new Skel(sr), blue: new Skel(sb) };
    vecs = { red: new Vec(vr), blue: new Vec(vb) };
  } catch (e) { skelWhy = e.message; }
  try { hl = await data("highlights.json"); } catch {}

  const dur = bout.source.duration_s;
  const tr = new Transport(dur);
  const F = bout.fighters;

  // ---------- header
  const kicks = { red: 0, blue: 0 };
  let vmax = 0;
  for (const e of hl.events) {
    if (e.type === "kick") { kicks[e.fighter]++; vmax = Math.max(vmax, e.v_peak_mm_s); }
  }
  put(mount,
    el("div", "kicker", "chuncheon 2024 · world taekwondo · one camera, three reads"),
    el("div", "h-title", `${F.blue.label} against ${F.red.label}`),
    el("div", "h-sub",
      "The same five and a half minutes read three ways - temporally tracked SAM 3.1 masks, cleaned ",
      "SAM 3D Body projections, and the predicted MHR surface itself. Below them, the ",
      el("em", "", "fight view"), ": both athletes alone on black, with velocity ",
      "arrows and live joint angles. Depth is model-inferred; a joint the ",
      "cleaner dropped is absent, not guessed."));
  const stats = el("div", "stats");
  put(stats,
    stat("bout on film", tc(dur)),
    stat(`${F.red.label} kicks over ${fmt(hl.thresholds ? hl.thresholds.kick_mm_s / 1000 : 4, 0)} m/s`, String(kicks.red)),
    stat(`${F.blue.label} kicks`, String(kicks.blue)),
    stat("fastest foot", vmax ? fmt(vmax / 1000, 1) : null, "m/s"),
    stat("3d samples", skels ? String(skels.red.n + skels.blue.n) : null));
  put(mount, el("div", "section-head", el("span", "t", "The bout"), el("span", "m", "hong · " + F.red.label + " — chung · " + F.blue.label)), stats);

  // ---------- three comparative panes
  const mkVideo = (src, poster) => {
    const v = el("video");
    v.src = media(src);
    if (poster) v.poster = media(poster);
    v.muted = true; v.playsInline = true; v.preload = "auto";
    return v;
  };
  const segV = mkVideo("seg.mp4", "seg_poster.jpg");
  const poseV = mkVideo("pose.mp4", "pose_poster.jpg");
  const meshV = mkVideo("mesh.mp4", "mesh_poster.jpg");
  const cap = (a, b) => el("div", "cap", el("span", "", a), el("span", "", b));
  const three = el("div", "threeviews",
    el("div", "viewpane", segV, cap("i · segmentation", "sam 3.1 · temporal fighter IDs")),
    el("div", "viewpane", poseV, cap("ii · pose, cleaned", "sam 3d body · frame-aligned")),
    el("div", "viewpane", meshV, cap("iii · reconstruction", "sam 3d body · actual mhr mesh")));
  put(mount, three);

  // transport with kick/risk ticks
  const ticks = hl.events.filter((e) => e.type === "kick").map((e) => ({
    t: e.t, kind: e.risk && e.risk.length ? "fail" : "",
    title: `${tc(e.t)} · ${e.fighter === "red" ? F.red.label : F.blue.label} · ${e.limb} · ${(e.v_peak_mm_s / 1000).toFixed(1)} m/s`,
  }));
  put(mount, tr.ui(ticks));

  // ---------- the fight view
  put(mount, el("div", "section-head",
    el("span", "t", "The fight view"),
    el("span", "m", "camera-relative space · pair-centred · y is the floor normal")));

  let sceneMain = null;
  const fv = el("div", "fightview");
  if (skels) {
    sceneMain = createFightScene(fv, { skels, vecs, highlights: hl.events, fighters: F, compact: false });

    const hud = el("div", "fv-hud");
    const modeBtn = (role) => {
      const f = FIGHTER[role];
      const b = el("button", `fbtn is-${f.cls} is-on`, `${F[role].label} · skeleton`);
      b.addEventListener("click", () => {
        const m = sceneMain.getMode(role) === "skel" ? "model" : "skel";
        sceneMain.setMode(role, m);
        sceneMain.setTime(tr.t);
        b.textContent = `${F[role].label} · ${m === "skel" ? "skeleton" : "volume proxy"}`;
      });
      return b;
    };
    const flagBtn = (name, label2, on) => {
      const b = el("button", "fbtn" + (on ? " is-on" : ""), label2);
      sceneMain.flags[name] = on;
      b.addEventListener("click", () => {
        sceneMain.flags[name] = !sceneMain.flags[name];
        b.classList.toggle("is-on", sceneMain.flags[name]);
        sceneMain.setTime(tr.t);
      });
      return b;
    };
    put(hud, modeBtn("red"), modeBtn("blue"), el("span", "sp"),
      flagBtn("vectors", "vectors", true),
      flagBtn("angles", "angles", false),
      flagBtn("trails", "trails", true));
    const readout = el("div", "fv-readout");
    put(fv, hud, readout);
    put(mount, fv);

    const upd = () => {
      const r = sceneMain.readout();
      readout.replaceChildren(
        el("div", "", "pair distance ", el("b", "", r.dist == null ? "—" : r.dist.toFixed(2) + " m"), " · model-inferred"),
        el("div", "", `${F.red.label} `, el("b", "", r.red == null ? "—" : (r.red / 1000).toFixed(1) + " m/s"),
          ` · ${F.blue.label} `, el("b", "", r.blue == null ? "—" : (r.blue / 1000).toFixed(1) + " m/s")));
    };
    tr.on((t) => { sceneMain.setTime(t); upd(); });
    sceneMain.setTime(0); upd();
    window.__fv = sceneMain;  // test hook
  } else {
    put(mount, panel("the fight view", null, scrim("3d store not built — " + skelWhy)));
  }

  // ---------- three fastest kicks
  let reel3 = null;
  try { reel3 = await data("reel.json"); } catch {}
  if (reel3 && reel3.length) {
    put(mount, el("div", "section-head",
      el("span", "t", "The three fastest kicks"),
      el("span", "m", "cut from the mask render · click to open in the bout")));
    const g = el("div", "threeviews");
    for (const r of reel3) {
      const v = el("video");
      v.src = media(r.file.replace("media/", ""));
      v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true;
      v.style.cursor = "pointer";
      v.addEventListener("click", () => { tr.seek(Math.max(0, r.t - 1.2)); tr.playing = true; tr.emit(); window.scrollTo({ top: 0, behavior: "smooth" }); });
      g.append(el("div", "viewpane", v, cap(
        `${F[r.fighter].label} · ${r.limb}`,
        `${(r.v_peak_mm_s / 1000).toFixed(1)} m/s` + (r.risk && r.risk.length ? " · flagged" : ""))));
    }
    put(mount, g);
  }

  // ---------- the exchange reel
  const reel = el("div");
  const evs = hl.events.filter((e) => e.type !== "burst");
  if (evs.length) {
    put(mount, el("div", "section-head",
      el("span", "t", "The exchange reel"),
      el("span", "m", `${evs.length} strikes over threshold · click to jump`)));
    const cols = "70px 1fr 1fr 90px 90px 110px minmax(120px,1.4fr)";
    const head = el("div", "tr tr--h", el("span", "", "clock"), el("span", "", "fighter"),
      el("span", "", "limb"), el("span", "", "peak"), el("span", "", "knee"),
      el("span", "", "ω knee"), el("span", "", "screen"));
    head.style.gridTemplateColumns = cols;
    put(reel, head);
    for (const e of evs) {
      const f = e.fighter;
      const row = el("div", "tr is-link",
        el("b", "", tc(e.t)),
        el("span", "", chip(F[f].label, FIGHTER[f].cls)),
        el("span", "", e.limb),
        el("b", "", (e.v_peak_mm_s / 1000).toFixed(1), el("span", "u", "m/s")),
        el("b", "", e.knee_deg == null ? "—" : Math.round(e.knee_deg) + "°"),
        el("b", "", e.omega_knee_peak == null ? "—" : String(e.omega_knee_peak),
          e.omega_knee_peak == null ? null : el("span", "u", "deg/s")),
        el("span", "", ...(e.risk.length ? e.risk.map((r) => chip(r.replaceAll("_", " "), "fail")) : [chip("clean", "dim")])));
      row.style.gridTemplateColumns = cols;
      row.addEventListener("click", () => { tr.seek(Math.max(0, e.t - 1.2)); tr.playing = true; tr.emit(); });
      put(reel, row);
    }
    put(mount, reel);
  }

  // ---------- sync loop
  let raf = 0, last = performance.now(), disposed = false;
  const vids = [segV, poseV, meshV];
  const syncVideos = () => {
    for (const v of vids) {
      if (!v.duration) continue;
      const target = Math.min(tr.t, v.duration - 0.05);
      if (tr.playing) {
        if (v.paused) v.play().catch(() => {});
        v.playbackRate = tr.speed;
        if (Math.abs(v.currentTime - target) > 0.15) v.currentTime = target;
      } else {
        if (!v.paused) v.pause();
        if (Math.abs(v.currentTime - target) > 0.05) v.currentTime = target;
      }
    }
  };
  const loop = (now) => {
    if (disposed) return;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    tr.tick(dt);
    if (!tr.playing) syncVideos();
    raf = requestAnimationFrame(loop);
  };
  tr.on(syncVideos);
  raf = requestAnimationFrame(loop);

  const t0 = parseFloat(ctx.params.get("t"));
  if (Number.isFinite(t0)) tr.seek(t0);

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    for (const v of vids) { v.pause(); v.removeAttribute("src"); v.load(); }
    if (sceneMain) sceneMain.dispose();
  };
}

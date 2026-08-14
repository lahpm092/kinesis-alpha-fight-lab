/* The shared 3D scene: both fighters on one floor, drawn from the store and
   nothing else. Two body modes per fighter - the raw skeleton (bones as
   lines, joints as points) and a low-poly capsule body - plus optional
   velocity arrows, ankle trails, and live joint-angle arcs. A joint the
   cleaner dropped simply is not drawn; the pair is centred, because the
   space is camera-relative and only the pair's geometry is trustworthy. */
import * as THREE from "three";
import { OrbitControls } from "../vendor/OrbitControls.js";
import { T, FIGHTER } from "./theme.js";

const ROLES = ["red", "blue"];
const ENDS = { com: null, l_ank: "left_ankle", r_ank: "right_ankle",
               l_wri: "left_wrist", r_wri: "right_wrist" };
const ARCS = [
  ["left_hip", "left_knee", "left_ankle", "knee"],
  ["right_hip", "right_knee", "right_ankle", "knee"],
  ["left_shoulder", "left_hip", "left_knee", "hip"],
  ["right_shoulder", "right_hip", "right_knee", "hip"],
  ["left_shoulder", "left_elbow", "left_wrist", "elbow"],
  ["right_shoulder", "right_elbow", "right_wrist", "elbow"],
];
// capsule radii by bone (subset index pairs from the store's bone list)
const GIRTH = { torso: 0.062, thigh: 0.052, shin: 0.042, upper_arm: 0.036,
                forearm: 0.030, foot: 0.022, head: 0.0, face: 0.0 };

function boneKind(names, a, b) {
  const n1 = names[a], n2 = names[b];
  const both = n1 + "|" + n2;
  if (/hip.*knee|knee.*hip/.test(both)) return "thigh";
  if (/knee.*ankle|ankle.*knee/.test(both)) return "shin";
  if (/shoulder.*elbow|elbow.*shoulder/.test(both)) return "upper_arm";
  if (/elbow.*wrist|wrist.*elbow/.test(both)) return "forearm";
  if (/ankle|toe|heel/.test(n1) && /ankle|toe|heel/.test(n2)) return "foot";
  if (/eye|ear|nose/.test(n1) || /eye|ear|nose/.test(n2)) return "face";
  if (/neck.*nose|nose.*neck/.test(both)) return "face";
  return "torso";
}

function label(txt, color, px = 22) {
  const c = document.createElement("canvas");
  c.width = 128; c.height = 48;
  const g = c.getContext("2d");
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sp = new THREE.Sprite(mat);
  sp.scale.set(0.42, 0.158, 1);
  const draw = (s) => {
    g.clearRect(0, 0, 128, 48);
    g.font = `${px}px ui-monospace, Menlo, monospace`;
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillStyle = color;
    g.fillText(s, 64, 26);
    tex.needsUpdate = true;
  };
  draw(txt);
  return { sp, draw, last: txt };
}

class FighterRig {
  constructor(scene, skel, meta) {
    this.skel = skel;
    this.meta = meta;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.nj = skel.names.length;
    const col = new THREE.Color(meta.color);

    // skeleton: bones as lines, joints as points
    this.boneGeo = new THREE.BufferGeometry();
    this.bonePos = new Float32Array(skel.bones.length * 6);
    this.boneGeo.setAttribute("position", new THREE.BufferAttribute(this.bonePos, 3));
    this.lines = new THREE.LineSegments(this.boneGeo,
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.95 }));
    this.lines.frustumCulled = false;
    this.ptGeo = new THREE.BufferGeometry();
    this.ptPos = new Float32Array(this.nj * 3);
    this.ptGeo.setAttribute("position", new THREE.BufferAttribute(this.ptPos, 3));
    this.points = new THREE.Points(this.ptGeo,
      new THREE.PointsMaterial({ color: T.amber, size: 0.038, transparent: true, opacity: 0.9 }));
    this.points.frustumCulled = false;
    this.skelGroup = new THREE.Group();
    this.skelGroup.add(this.lines, this.points);
    this.group.add(this.skelGroup);

    // capsule body
    this.bodyGroup = new THREE.Group();
    this.caps = [];
    const mat = new THREE.MeshStandardMaterial({
      color: col, emissive: col, emissiveIntensity: 0.22,
      roughness: 0.55, metalness: 0.05, flatShading: true });
    for (let bi = 0; bi < skel.bones.length; bi++) {
      const [a, b] = skel.bones[bi];
      const kind = boneKind(skel.names, a, b);
      const r = GIRTH[kind];
      if (!r) { this.caps.push(null); continue; }
      const geo = new THREE.CapsuleGeometry(r, 1, 3, 8);
      const m = new THREE.Mesh(geo, mat);
      m.visible = false;
      this.caps.push(m);
      this.bodyGroup.add(m);
    }
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.085, 10, 8), mat);
    head.visible = false;
    this.head = head;
    this.bodyGroup.add(head);
    this.bodyGroup.visible = false;
    this.group.add(this.bodyGroup);

    // name tag
    this.tag = label(meta.label, T.bone2 + "", 20);
    this.tag.sp.scale.set(0.6, 0.22, 1);
    this.group.add(this.tag.sp);

    this.mode = "skel";
    this.world = new Float32Array(this.nj * 3).fill(NaN);
    this.rootV = new THREE.Vector3();
    this.present = false;
    this._up = new THREE.Vector3(0, 1, 0);
    this._d = new THREE.Vector3();
    this._q = new THREE.Quaternion();
  }

  setMode(m) {
    this.mode = m;
    this.skelGroup.visible = m === "skel" && this.present;
    this.bodyGroup.visible = m === "model" && this.present;
  }

  jointV(name, out) {
    const j = this.skel.byName[name];
    if (j == null) return null;
    const x = this.world[j * 3];
    if (!Number.isFinite(x)) return null;
    return out.set(x, this.world[j * 3 + 1], this.world[j * 3 + 2]);
  }

  update(t, center) {
    const f = this.skel.at(t);
    this.present = f.ok && !!f.root;
    this.skelGroup.visible = this.mode === "skel" && this.present;
    this.bodyGroup.visible = this.mode === "model" && this.present;
    this.tag.sp.visible = this.present;
    if (!this.present) return;
    const r = f.root;
    this.rootV.set(r[0] / 1000 - center.x, r[1] / 1000, r[2] / 1000 - center.z);
    // world joints in metres, centred
    let top = -1e9, topx = 0, topz = 0;
    for (let j = 0; j < this.nj; j++) {
      const mx = f.joints[j * 3];
      if (Number.isFinite(mx)) {
        const wx = this.rootV.x + mx / 1000;
        const wy = this.rootV.y + f.joints[j * 3 + 1] / 1000;
        const wz = this.rootV.z + f.joints[j * 3 + 2] / 1000;
        this.world[j * 3] = wx; this.world[j * 3 + 1] = wy; this.world[j * 3 + 2] = wz;
        this.ptPos[j * 3] = wx; this.ptPos[j * 3 + 1] = wy; this.ptPos[j * 3 + 2] = wz;
        if (wy > top) { top = wy; topx = wx; topz = wz; }
      } else {
        this.world[j * 3] = NaN;
        this.ptPos[j * 3] = 0; this.ptPos[j * 3 + 1] = -10; this.ptPos[j * 3 + 2] = 0;
      }
    }
    this.ptGeo.attributes.position.needsUpdate = true;
    this.tag.sp.position.set(topx, top + 0.22, topz);

    const bones = this.skel.bones;
    for (let bi = 0; bi < bones.length; bi++) {
      const [a, b] = bones[bi];
      const ax = this.world[a * 3], bx = this.world[b * 3];
      const okb = Number.isFinite(ax) && Number.isFinite(bx);
      const o = bi * 6;
      if (okb) {
        this.bonePos[o] = ax; this.bonePos[o + 1] = this.world[a * 3 + 1]; this.bonePos[o + 2] = this.world[a * 3 + 2];
        this.bonePos[o + 3] = bx; this.bonePos[o + 4] = this.world[b * 3 + 1]; this.bonePos[o + 5] = this.world[b * 3 + 2];
      } else {
        for (let k = 0; k < 6; k++) this.bonePos[o + k] = 0;
        this.bonePos[o + 1] = this.bonePos[o + 4] = -10;
      }
      const cap = this.caps[bi];
      if (cap) {
        cap.visible = okb && this.mode === "model";
        if (okb) {
          const ay = this.world[a * 3 + 1], az = this.world[a * 3 + 2];
          const by = this.world[b * 3 + 1], bz = this.world[b * 3 + 2];
          this._d.set(bx - ax, by - ay, bz - az);
          const len = this._d.length();
          cap.position.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2);
          this._q.setFromUnitVectors(this._up, this._d.normalize());
          cap.quaternion.copy(this._q);
          cap.scale.set(1, Math.max(len * 0.8, 0.02), 1);
        }
      }
    }
    this.boneGeo.attributes.position.needsUpdate = true;
    // head sphere between the ears / at the nose
    const nose = this.jointV("nose", this._d);
    if (this.head) {
      this.head.visible = !!nose && this.mode === "model";
      if (nose) this.head.position.copy(nose);
    }
  }
}

export function createFightScene(mount, { skels, vecs, highlights = [], fighters, compact = false }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  mount.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const bgc = compact ? T.coal : T.black;
  scene.background = new THREE.Color(bgc);
  scene.fog = new THREE.Fog(bgc, 9, 22);

  const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.05, 80);
  camera.position.set(compact ? 2.6 : 0.4, compact ? 2.6 : 2.1, compact ? 5.6 : 5.0);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 1.6;
  controls.maxDistance = 14;
  controls.target.set(0, 0.95, 0);

  scene.add(new THREE.HemisphereLight("#4a3f2e", "#0a0703", 0.85));
  const key = new THREE.DirectionalLight("#e8c98f", 0.9);
  key.position.set(3, 6, 2);
  scene.add(key);
  const rim = new THREE.DirectionalLight("#7d8fa8", 0.3);
  rim.position.set(-4, 3, -3);
  scene.add(rim);

  // the mat: octagon court lines on y=0, nothing else in the darkness
  const ground = new THREE.Group();
  const octo = (rad, color, opacity) => {
    const pts = [];
    for (let i = 0; i <= 8; i++) {
      const a = (i / 8) * Math.PI * 2 + Math.PI / 8;
      pts.push(new THREE.Vector3(Math.cos(a) * rad, 0, Math.sin(a) * rad));
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  };
  ground.add(octo(4.0, T.bone2, 0.30));
  ground.add(octo(2.8, T.amber2, 0.14));
  for (let r = 1; r <= 3; r++) ground.add(octo(r, T.hair, 0.10));
  const cross = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-0.12, 0, 0), new THREE.Vector3(0.12, 0, 0),
    new THREE.Vector3(0, 0, -0.12), new THREE.Vector3(0, 0, 0.12)]);
  ground.add(new THREE.LineSegments(cross, new THREE.LineBasicMaterial({ color: T.amber2, transparent: true, opacity: 0.4 })));
  scene.add(ground);

  const rigs = {};
  for (const role of ROLES) {
    rigs[role] = new FighterRig(scene, skels[role],
      { color: FIGHTER[role].color, label: fighters[role].label });
  }

  // velocity arrows + speed labels
  const flags = { vectors: !compact, angles: false, trails: !compact };
  const arrows = {};
  const _dir = new THREE.Vector3(), _pos = new THREE.Vector3();
  for (const role of ROLES) {
    arrows[role] = {};
    for (const key2 of Object.keys(ENDS)) {
      const ar = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.5,
        new THREE.Color(T.amber), 0.09, 0.05);
      ar.line.material.transparent = ar.cone.material.transparent = true;
      ar.visible = false;
      scene.add(ar);
      const lb = label("", T.amber, 22);
      lb.sp.visible = false;
      scene.add(lb.sp);
      arrows[role][key2] = { ar, lb };
    }
  }

  // ankle trails
  const TRAIL_N = 26;
  const trails = {};
  for (const role of ROLES) {
    trails[role] = {};
    for (const nm of ["left_ankle", "right_ankle"]) {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(TRAIL_N * 3);
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: FIGHTER[role].color, transparent: true, opacity: 0.38 }));
      line.frustumCulled = false;
      line.visible = false;
      scene.add(line);
      trails[role][nm] = { line, pos, buf: [], lastT: -1 };
    }
  }

  // angle arcs
  const arcs = [];
  for (const role of ROLES) {
    for (const [na, nb, nc, kind] of ARCS) {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(14 * 3);
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: T.amber2, transparent: true, opacity: 0.85 }));
      line.frustumCulled = false;
      line.visible = false;
      scene.add(line);
      const lb = label("", T.bone, 20);
      lb.sp.visible = false;
      lb.sp.scale.set(0.30, 0.113, 1);
      scene.add(lb.sp);
      arcs.push({ role, na, nb, nc, kind, line, pos, lb });
    }
  }

  const center = new THREE.Vector3(0, 0, 0);
  const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3();
  const _u = new THREE.Vector3(), _w = new THREE.Vector3();
  let lastT = 0;

  function setTime(t) {
    lastT = t;
    // centre on the pair (smoothed), x/z only
    let cx = 0, cz = 0, np = 0;
    for (const role of ROLES) {
      const f = skels[role].at(t);
      if (f.root) { cx += f.root[0] / 1000; cz += f.root[2] / 1000; np++; }
    }
    if (np) {
      _pos.set(cx / np, 0, cz / np);
      center.lerp(_pos, 0.10);
    }
    for (const role of ROLES) rigs[role].update(t, center);

    // active highlight windows, keyed per fighter+limb
    const hot = {};
    for (const e of highlights) {
      if (Math.abs(e.t - t) < 0.45 && e.type !== "burst") {
        const k = e.limb.includes("left") ? (e.type === "kick" ? "l_ank" : "l_wri")
                                          : (e.type === "kick" ? "r_ank" : "r_wri");
        hot[e.fighter + ":" + k] = e;
      }
    }

    for (const role of ROLES) {
      const rig = rigs[role];
      for (const [key2, jname] of Object.entries(ENDS)) {
        const { ar, lb } = arrows[role][key2];
        let show = flags.vectors && rig.present;
        let p = null;
        if (show) {
          p = jname ? rig.jointV(jname, _pos) : (rig.present ? _pos.copy(rig.rootV) : null);
          if (!p) show = false;
        }
        const v = show ? vecs[role].vAt(t, key2) : null;
        const sp = show ? vecs[role].sAt(t, key2) : null;
        const floor = key2 === "com" ? 900 : 2000;
        const e = hot[role + ":" + key2];
        if (show && v && sp != null && (sp > floor || e)) {
          _dir.set(v[0], v[1], v[2]).normalize();
          const len = Math.min(Math.max((sp / 1000) * 0.16, 0.14), 1.5) * (e ? 1.35 : 1);
          ar.position.copy(p);
          ar.setDirection(_dir);
          ar.setLength(len, 0.07 + (e ? 0.03 : 0), 0.04);
          ar.setColor(new THREE.Color(e ? T.bone : T.amber));
          ar.line.material.opacity = ar.cone.material.opacity = e ? 1 : 0.75;
          ar.visible = true;
          if (e) {
            const txt = (e.v_peak_mm_s / 1000).toFixed(1) + " m/s";
            if (lb.last !== txt) { lb.draw(txt); lb.last = txt; }
            lb.sp.position.copy(p).addScaledVector(_dir, len + 0.16);
            lb.sp.visible = true;
          } else lb.sp.visible = false;
        } else { ar.visible = false; lb.sp.visible = false; }
      }

      for (const nm of ["left_ankle", "right_ankle"]) {
        const tr = trails[role][nm];
        if (!flags.trails || !rig.present) { tr.line.visible = false; continue; }
        const p = rig.jointV(nm, _pos);
        if (tr.buf.length && t < tr.buf[tr.buf.length - 1][0] - 0.05) {
          tr.buf.length = 0;  // scrubbed backwards; the old future is stale
        }
        if (p && Math.abs(t - tr.lastT) > 1 / 61) {
          tr.buf.push([t, p.x, p.y, p.z]);
          tr.lastT = t;
        }
        while (tr.buf.length && (t - tr.buf[0][0] > 0.6 || tr.buf.length > TRAIL_N)) tr.buf.shift();
        if (tr.buf.length < 3) { tr.line.visible = false; continue; }
        for (let i = 0; i < TRAIL_N; i++) {
          const s = tr.buf[Math.min(i, tr.buf.length - 1)];
          tr.pos[i * 3] = s[1]; tr.pos[i * 3 + 1] = s[2]; tr.pos[i * 3 + 2] = s[3];
        }
        tr.line.geometry.attributes.position.needsUpdate = true;
        tr.line.visible = true;
      }
    }

    for (const arc of arcs) {
      const rig = rigs[arc.role];
      let ok = flags.angles && rig.present;
      if (ok) {
        const a = rig.jointV(arc.na, _a), b = rig.jointV(arc.nb, _b), c = rig.jointV(arc.nc, _c);
        ok = !!(a && b && c);
        if (ok) {
          _u.subVectors(_a, _b).normalize();
          _w.subVectors(_c, _b);
          const proj = _w.dot(_u);
          const ang = Math.acos(Math.min(Math.max(
            _u.dot(_c.clone().sub(_b).normalize()), -1), 1));
          _w.addScaledVector(_u, -proj).normalize();
          const R2 = 0.14;
          for (let i = 0; i < 14; i++) {
            const th = (i / 13) * ang;
            const px = _b.x + R2 * (Math.cos(th) * _u.x + Math.sin(th) * _w.x);
            const py = _b.y + R2 * (Math.cos(th) * _u.y + Math.sin(th) * _w.y);
            const pz = _b.z + R2 * (Math.cos(th) * _u.z + Math.sin(th) * _w.z);
            arc.pos[i * 3] = px; arc.pos[i * 3 + 1] = py; arc.pos[i * 3 + 2] = pz;
          }
          arc.line.geometry.attributes.position.needsUpdate = true;
          const deg = Math.round(ang * 180 / Math.PI);
          const txt = deg + "°";
          if (arc.lb.last !== txt) { arc.lb.draw(txt); arc.lb.last = txt; }
          const mth = ang / 2;
          arc.lb.sp.position.set(
            _b.x + 0.24 * (Math.cos(mth) * _u.x + Math.sin(mth) * _w.x),
            _b.y + 0.24 * (Math.cos(mth) * _u.y + Math.sin(mth) * _w.y),
            _b.z + 0.24 * (Math.cos(mth) * _u.z + Math.sin(mth) * _w.z));
        }
      }
      arc.line.visible = ok;
      arc.lb.sp.visible = ok;
    }
  }

  // readout: model-inferred pair distance + com speeds
  function readout() {
    const fr = skels.red.at(lastT), fb = skels.blue.at(lastT);
    let dist = null;
    if (fr.root && fb.root) {
      const dx = (fr.root[0] - fb.root[0]) / 1000;
      const dy = (fr.root[1] - fb.root[1]) / 1000;
      const dz = (fr.root[2] - fb.root[2]) / 1000;
      dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return { dist,
      red: vecs.red.sAt(lastT, "com"), blue: vecs.blue.sAt(lastT, "com") };
  }

  let disposed = false;
  let raf = 0;
  const loop = () => {
    if (disposed) return;
    if (renderer.domElement.isConnected) {
      controls.update();
      renderer.render(scene, camera);
    }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  const ro = new ResizeObserver(() => {
    const w = mount.clientWidth, h = mount.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  ro.observe(mount);

  return {
    setTime,
    readout,
    flags,
    setMode: (role, m) => rigs[role].setMode(m),
    getMode: (role) => rigs[role].mode,
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          if (o.material.map) o.material.map.dispose();
          o.material.dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

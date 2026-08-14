/* Skeleton data access. Frames are root-centred mm ints at the analysis
   cadence; root_w places the root in the shared ground frame. Lookup by bout
   time linearly interpolates between two neighbouring samples, but only
   where BOTH samples carry the joint - a cleaned-away joint stays absent,
   interpolation never invents one across a hole. */

export class Skel {
  constructor(json) {
    this.j = json;
    this.fps = json.fps;
    this.n = json.n;
    this.names = json.names;
    this.bones = json.bones;
    this.T = json.F.map((f) => f / 30.0);
    this.byName = Object.fromEntries(json.names.map((nm, i) => [nm, i]));
  }

  /** index of the last sample with T <= t (clamped) */
  idx(t) {
    const T = this.T;
    let lo = 0, hi = T.length - 1;
    if (t <= T[0]) return 0;
    if (t >= T[hi]) return hi;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (T[m] <= t) lo = m; else hi = m;
    }
    return lo;
  }

  /** { joints: Float32Array(nj*3) mm | NaN, root: [x,y,z] mm | null, ok } */
  at(t) {
    const q = this.idx(t);
    const T = this.T, fr = this.j.frames, rw = this.j.root_w;
    const q1 = Math.min(q + 1, this.n - 1);
    const span = T[q1] - T[q];
    const a = span > 0 ? Math.min(Math.max((t - T[q]) / span, 0), 1) : 0;
    // interpolate only across one native step; a wider gap is a hole
    const bridge = q1 !== q && span <= 1.6 / this.fps;
    const A = fr[q], B = bridge ? fr[q1] : null;
    const nj = this.names.length;
    const out = new Float32Array(nj * 3).fill(NaN);
    let any = false;
    const near = a < 0.5 ? A : (bridge ? B : A);
    for (let j = 0; j < nj; j++) {
      const va = A ? A[j] : null;
      const vb = B ? B[j] : null;
      let v = null;
      if (va && vb) v = [va[0] + (vb[0] - va[0]) * a, va[1] + (vb[1] - va[1]) * a, va[2] + (vb[2] - va[2]) * a];
      else if (near && near[j]) v = near[j];
      if (v) { out[j * 3] = v[0]; out[j * 3 + 1] = v[1]; out[j * 3 + 2] = v[2]; any = true; }
    }
    let root = null;
    const ra = A ? rw[q] : null, rb = B ? rw[q1] : null;
    if (ra && rb) root = [ra[0] + (rb[0] - ra[0]) * a, ra[1] + (rb[1] - ra[1]) * a, ra[2] + (rb[2] - ra[2]) * a];
    else root = (a < 0.5 ? ra : (rb || ra)) || rb || null;
    return { joints: out, root, ok: any };
  }
}

/* velocity series access (vectors_<fighter>.json) */
export class Vec {
  constructor(json) {
    this.j = json;
    this.T = json.t;
  }
  idx(t) {
    const T = this.T;
    let lo = 0, hi = T.length - 1;
    if (t <= T[0]) return 0;
    if (t >= T[hi]) return hi;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (T[m] <= t) lo = m; else hi = m;
    }
    return lo;
  }
  /** [vx,vy,vz] mm/s or null */
  vAt(t, key) {
    const q = this.idx(t);
    return this.j.v_mm_s[key][q] || null;
  }
  sAt(t, key) {
    const q = this.idx(t);
    const v = this.j.speed_mm_s[key][q];
    return v == null ? null : v;
  }
}

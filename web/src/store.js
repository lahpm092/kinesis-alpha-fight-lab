/* Store reader. The web app never computes a metric: it renders artifacts
   and their provenance. Every fetch is cached; a missing artifact renders
   its reason, never a blank. */

const cache = new Map();

export async function data(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch("./store/" + path).then(async (r) => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  });
  cache.set(path, p);
  try { return await p; }
  catch (e) { cache.delete(path); throw e; }
}

export function media(path) { return "./store/media/" + path; }

/** validate-before-render: returns a list of human-readable problems */
export function checkSkel(s) {
  const bad = [];
  if (!s || typeof s !== "object") return ["not an object"];
  if (!Array.isArray(s.frames) || !s.frames.length) bad.push("no frames");
  if (!Array.isArray(s.bones)) bad.push("no bones");
  if (!Array.isArray(s.names)) bad.push("no joint names");
  if (!(s.fps > 0)) bad.push("no fps");
  if (!Array.isArray(s.F) || s.F.length !== s.frames.length) bad.push("F/frames mismatch");
  if (!Array.isArray(s.root_w) || s.root_w.length !== s.frames.length) bad.push("root_w/frames mismatch");
  return bad;
}

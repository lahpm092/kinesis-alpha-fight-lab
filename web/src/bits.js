/* Shared atoms. The rendering law: numbers are serif with tabular figures,
   labels are mono uppercase tracked, null renders as an em dash with its
   reason, never as NaN and never as zero. */

export function el(tag, cls, ...kids) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  for (const k of kids) {
    if (k == null) continue;
    n.append(k.nodeType ? k : document.createTextNode(String(k)));
  }
  return n;
}

export function frag(...kids) {
  const f = document.createDocumentFragment();
  for (const k of kids) if (k != null) f.append(k);
  return f;
}

export function put(parent, ...kids) {
  for (const k of kids) if (k != null) parent.append(k);
  return parent;
}

export function swap(parent, ...kids) {
  parent.replaceChildren();
  return put(parent, ...kids);
}

export function fmt(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 10000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return n.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: digits });
}

/** bout clock: 87.4 -> 1′27″4 */
export function tc(t) {
  if (t == null || Number.isNaN(t)) return "—";
  const s = Math.max(0, t);
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  const d = Math.floor((s % 1) * 10);
  return `${m}′${String(ss).padStart(2, "0")}″${d}`;
}

export function chip(text, kind = "") {
  return el("span", "chip" + (kind ? " chip--" + kind : ""), text);
}

export function panel(title, right, ...kids) {
  const h = el("div", "panel-h", el("span", "", title));
  if (right) h.append(right.nodeType ? right : el("span", "", right));
  return el("div", "panel", h, ...kids);
}

export function stat(k, v, u) {
  const val = el("span", "v", v == null ? "—" : v);
  if (u && v != null) val.append(el("span", "u", u));
  return el("div", "stat", val, el("span", "k", k));
}

export function kv(label, value, unit, why) {
  const row = el("div", "kv", el("span", "", label));
  const b = el("b");
  if (value == null) b.append("—");
  else { b.append(String(value)); if (unit) b.append(el("span", "u", unit)); }
  const right = el("span", "", b);
  if (why) right.append(el("span", "why", " " + why));
  row.append(right);
  return row;
}

export function scrim(text) {
  return el("div", "scrim", el("div", "t", text),
    el("div", "bar", el("i")));
}

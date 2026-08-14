/* Hash router with the family view contract: a view module exports
   async render(mount, ctx) -> dispose(); a stale async load can never claim
   the mount (seq guard); dispose must cancel rAF and free GPU resources. */
import { el, scrim } from "./bits.js";
import { data } from "./store.js";

const view = document.getElementById("view");
const nav = document.getElementById("nav");
const slug = document.getElementById("slug");
const colophon = document.getElementById("colophon");

const routes = {
  bout: () => import("./views/bout.js"),
  analysis: () => import("./views/analysis.js"),
  method: () => import("./views/method.js"),
};

let seq = 0;
let disposeCurrent = null;

async function route() {
  const hash = location.hash.replace(/^#\/?/, "") || "bout";
  const [name, query] = hash.split("?");
  const mod = routes[name] ? name : "bout";
  const my = ++seq;
  for (const a of nav.querySelectorAll("a")) {
    a.classList.toggle("is-on", a.dataset.r === mod);
  }
  if (disposeCurrent) { try { disposeCurrent(); } catch {} disposeCurrent = null; }
  view.replaceChildren(scrim("reading the store"));
  try {
    const m = await routes[mod]();
    const params = new URLSearchParams(query || "");
    if (my !== seq) return;
    const mount = el("div");
    const d = await m.render(mount, { params });
    if (my !== seq) { if (d) try { d(); } catch {} return; }
    view.replaceChildren(mount);
    disposeCurrent = typeof d === "function" ? d : null;
  } catch (e) {
    if (my !== seq) return;
    view.replaceChildren(scrim("this artifact is missing — " + e.message));
  }
}

window.addEventListener("hashchange", route);

data("bout.json").then((b) => {
  slug.textContent = `${b.fighters.red.label} · ${b.fighters.blue.label} — chuncheon 2024`;
  colophon.append(
    el("span", "", "KINESIS · Fight Lab"),
    el("span", "", b.models.segmentation),
    el("span", "", b.models.pose3d),
    el("span", "", "depth is model-inferred; absences are stated, not filled"),
  );
}).catch(() => { slug.textContent = "store not built"; });

route();

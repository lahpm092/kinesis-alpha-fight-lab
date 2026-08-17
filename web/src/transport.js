/* One clock for every pane. Frame advance is dt-based; the transport owns
   t, play state and speed - views subscribe and draw, they never own time. */
import { el, tc } from "./bits.js";

export class Transport {
  constructor(duration) {
    this.dur = duration;
    this.t = 0;
    this.playing = false;
    this.speed = 1;
    this.subs = new Set();
  }
  on(fn) { this.subs.add(fn); return () => this.subs.delete(fn); }
  emit() { for (const fn of this.subs) fn(this.t, this); }
  seek(t) { this.t = Math.min(Math.max(t, 0), this.dur); this.emit(); }
  tick(dt) {
    if (!this.playing) return;
    this.t += dt * this.speed;
    if (this.t >= this.dur) { this.t = this.dur; this.playing = false; }
    this.emit();
  }
  toggle() { this.playing = !this.playing; this.emit(); }

  /** timerow UI: play, speeds, rail with event ticks, clock */
  ui(events = []) {
    const play = el("button", "fbtn", "play");
    play.addEventListener("click", () => this.toggle());
    const speeds = [0.25, 0.5, 1].map((s) => {
      const b = el("button", "fbtn" + (s === 1 ? " is-on" : ""), `×${s}`);
      b.addEventListener("click", () => {
        this.speed = s;
        for (const x of sbtns) x.classList.toggle("is-on", x === b);
      });
      return b;
    });
    const sbtns = speeds;
    const range = el("input");
    range.type = "range"; range.min = 0; range.max = this.dur; range.step = 0.05;
    range.value = String(this.t);
    range.addEventListener("input", () => this.seek(parseFloat(range.value)));
    const rail = el("div", "scenerail", range);
    for (const e of events) {
      const tick = el("i", "tick" + (e.kind === "fail" ? " tick--fail" : e.kind === "dim" ? " tick--dim" : ""));
      tick.style.left = ((e.t / this.dur) * 100).toFixed(2) + "%";
      tick.title = e.title || "";
      if (e.kind !== "dim") tick.addEventListener("click", () => this.seek(e.t));
      rail.append(tick);
    }
    const clock = el("span", "flabel", tc(0));
    this.on((t) => {
      play.textContent = this.playing ? "pause" : "play";
      clock.textContent = tc(t) + " / " + tc(this.dur);
      if (document.activeElement !== range) range.value = String(t);
    });
    return el("div", "timerow", play, ...speeds, rail, clock);
  }
}

"""Store selftest: the artifacts the web app trusts, checked before shipping.

Checks are counts and ranges, not vibes; a failure prints what broke and the
script exits nonzero. Run after 05/06: python3 tests/selftest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "store"

checks = 0
fails = []


def ok(cond, what):
    global checks
    checks += 1
    if not cond:
        fails.append(what)


def main():
    bout = json.load(open(STORE / "bout.json"))
    ok(bout["fighters"]["red"]["label"] == "L.S. GALLO", "red label")
    ok(bout["fighters"]["blue"]["label"] == "Y. XU", "blue label")
    dur = bout["source"]["duration_s"]
    ok(300 < dur < 360, f"duration {dur}")

    for role in ("red", "blue"):
        s = json.load(open(STORE / "skel3d" / f"{role}.json"))
        n = s["n"]
        ok(n == len(s["frames"]) == len(s["root_w"]) == len(s["F"]),
           f"{role}: n/frames/root_w/F mismatch")
        ok(len(s["names"]) == 24 and len(s["bones"]) == 24, f"{role}: joint subset")
        ok(s["fps"] > 10, f"{role}: fps {s['fps']}")
        present = [f for f in s["frames"] if f is not None]
        ok(len(present) / n > 0.5, f"{role}: only {len(present)}/{n} frames present")
        mx = 0
        for f in present[:: max(1, len(present) // 200)]:
            for j in f:
                if j is not None:
                    mx = max(mx, abs(j[0]), abs(j[1]), abs(j[2]))
        ok(mx < 3000, f"{role}: joint {mx} mm from root")
        roots = [r[1] for r in s["root_w"] if r is not None]
        med = sorted(roots)[len(roots) // 2]
        ok(300 < med < 1500, f"{role}: median root height {med} mm")

        a = json.load(open(STORE / f"angles_{role}.json"))
        ok(len(a["t"]) == n, f"{role}: angles length")
        for k, v in a["deg"].items():
            ok(len(v) == n, f"{role}: deg.{k} length")
            vals = [x for x in v if x is not None]
            ok(all(0 <= x <= 185 for x in vals), f"{role}: deg.{k} range")
            ok(len(vals) > 0.4 * n, f"{role}: deg.{k} coverage {len(vals)}/{n}")

        vv = json.load(open(STORE / f"vectors_{role}.json"))
        ok(len(vv["t"]) == n, f"{role}: vectors length")
        sp = [x for x in vv["speed_mm_s"]["l_ank"] if x is not None]
        ok(sp and max(sp) < 30000, f"{role}: ankle speed max {max(sp) if sp else 0}")

    hl = json.load(open(STORE / "highlights.json"))
    ts = [e["t"] for e in hl["events"]]
    ok(ts == sorted(ts), "highlights sorted")
    ok(all(0 <= t <= dur for t in ts), "highlight times in bout")
    kicks = [e for e in hl["events"] if e["type"] == "kick"]
    ok(len(kicks) >= 4, f"only {len(kicks)} kicks found")
    for e in hl["events"]:
        ok(e["fighter"] in ("red", "blue"), "event fighter")

    risk = json.load(open(STORE / "risk.json"))
    ok(set(risk["fighters"]) == {"red", "blue"}, "risk fighters")
    ok(len(risk["method"]) >= 3, "risk method strings")

    tl = json.load(open(STORE / "timeline.json"))
    nf = len(tl["F"])
    for r in ("red", "blue", "ref"):
        ok(len(tl["px"][r]) == nf, f"timeline px {r}")

    for m, floor in (("bout.mp4", 3e6), ("seg.mp4", 3e6), ("pose.mp4", 1e6),
                     ("seg_poster.jpg", 5e3), ("pose_poster.jpg", 5e3)):
        f = STORE / "media" / m
        ok(f.exists() and f.stat().st_size > floor, f"media/{m}")

    for w in ("index.html", "styles/main.css", "src/app.js", "src/fightview.js",
              "src/views/bout.js", "src/views/analysis.js", "src/views/method.js",
              "vendor/three.module.js", "vendor/three.core.js", "vendor/OrbitControls.js"):
        ok((ROOT / "web" / w).exists(), f"web/{w}")

    print(f"{checks} checks, {len(fails)} failed")
    for f in fails:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""
Find where the TBC was toggled, by watching per-row displacement over time.

The user turned the time-base corrector on and off while capturing, so the same tape
and the same deck contain a controlled experiment. `player.tbc_enabled` and
`player.tbc_strength` claim to model what a TBC does, and this is the material that can
check the claim.

Time-base correction acts field by field, so the measurement is per field and the
quantity is the spread of per-row displacement -- insensitive to what the scene is
doing, unlike a row-mean scan, which is what made a first attempt flag the scrolling
lyrics as instability.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_timebase import displacement_stats, extract_field, load_gray  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="track per-row displacement over time")
    ap.add_argument("clip")
    ap.add_argument("--from", dest="start", type=float, default=900.0)
    ap.add_argument("--to", dest="stop", type=float, default=1500.0)
    ap.add_argument("--step", type=float, default=15.0)
    ap.add_argument("--out", default="tools/scan/tbc")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("%9s %6s %9s %11s %9s" % ("time", "rows", "std(px)", "line_step", "range"))
    rows = []
    t = args.start
    while t < args.stop:
        p = os.path.join(args.out, "f%07.1f.png" % t)
        if extract_field(args.clip, t, p, 720):
            st = displacement_stats(load_gray(p))
            rows.append((t, st))
            print("%8.0fs %6d %9.3f %11.3f %9.2f"
                  % (t, st["rows"], st["std"], st["line_step"], st["range"]))
        t += args.step

    vals = [(t, s["std"]) for t, s in rows if np.isfinite(s["std"]) and s["rows"] > 40]
    if len(vals) < 4:
        print()
        print("too few usable fields to say anything")
        return 0
    arr = np.array([v for _t, v in vals])
    print()
    print("usable fields: %d, displacement std ranges %.3f .. %.3f px"
          % (len(arr), arr.min(), arr.max()))
    # the stretches where the spread is clearly below the median are the candidates
    # for "TBC engaged"
    med = float(np.median(arr))
    low = [(t, v) for t, v in vals if v < 0.55 * med]
    high = [(t, v) for t, v in vals if v > 1.45 * med]
    print("  clearly low  (<0.55x median): %s"
          % (", ".join("%.0fs(%.2f)" % (t, v) for t, v in low[:10]) or "none"))
    print("  clearly high (>1.45x median): %s"
          % (", ".join("%.0fs(%.2f)" % (t, v) for t, v in high[:10]) or "none"))
    if low and high:
        print("  ratio high/low: %.2fx"
              % (float(np.mean([v for _t, v in high]))
                 / max(1e-9, float(np.mean([v for _t, v in low])))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Calibrate a mechanism against a real reference frame.

Developing the model produced a habit that does not scale: pick a parameter, guess
a relationship, look at the output, adjust.  With real reference material available
the loop should instead be *measure the target, sweep the parameter, report the
error* -- so the setting is derived rather than eyeballed, and the residual is
visible.

    python tools/calibrate.py --target "path/to/frame.png" \
        --layer tape --key binder_loss \
        --metrics sat,mean,dark,hf

Metrics are reported for the target and for each swept value, plus the absolute
error against the target, so the best candidate is obvious and any mismatch in
*shape* (a metric that moves the wrong way) shows up immediately rather than being
papered over.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values                      # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS                 # noqa: E402
from atsim.io.imagecodec import load_image, save_image          # noqa: E402
from atsim.render.renderer import Renderer                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def metrics(img: np.ndarray) -> dict:
    """The handful of numbers that characterise a degraded frame."""
    from scipy.ndimage import laplace
    g = img @ LUMA
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    row = g.mean(axis=1)
    ripple = float((row - np.convolve(row, np.ones(3) / 3, mode="same"))[4:-4].std())
    return {
        "mean": float(img.mean()),
        "sat": float(sat.mean()),
        "hf": float(np.abs(laplace(g)).mean()),
        "dark": float((g < 0.12).mean()),
        "ripple": ripple,
        "contrast": float(g.std()),
    }


class StillScene:
    """A source that renders one still frame at the target's own size."""

    def __init__(self, img: np.ndarray, frames: int = 24, fps: float = 25.0):
        self._img = img.astype(np.float32)
        self.height, self.width = self._img.shape[:2]
        self.frame_count, self.fps = int(frames), float(fps)

    def frame(self, index: int) -> np.ndarray:
        return self._img.copy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="calibrate a parameter against a reference")
    ap.add_argument("--target", required=True, help="reference frame (any image)")
    ap.add_argument("--layer", default="tape")
    ap.add_argument("--key", default="binder_loss")
    ap.add_argument("--values", default="0.1,0.2,0.35,0.5,0.7,0.9,1.0")
    ap.add_argument("--metrics", default="mean,sat,dark,hf")
    ap.add_argument("--source", default="",
                    help="source clip/still to run through the chain; default = the target itself")
    ap.add_argument("--base-layer", default="format",
                    help="which layer to tweak for the bandwidth match")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--sheet", default=os.path.join(OUT_DIR, "calibrate.png"))
    args = ap.parse_args(argv)

    target = load_image(args.target)
    print(f"target: {args.target}  {target.shape[1]}x{target.shape[0]}")
    tm = metrics(target)
    print("  " + "  ".join(f"{k}={tm[k]:.4f}" for k in args.metrics.split(",")))

    # The *source* is the undamaged picture; the target is that picture after the
    # mechanism.  Running the target through the chain would compound the damage.
    src_is_target = not args.source
    src_img = load_image(args.source) if args.source else None
    if src_img is None:
        print()
        print("  note: no --source given, so the target itself is used as the input.")
        print("        The swept engine can then only ever ADD damage, so values")
        print("        below the best match are the meaningful ones.")
        src_img = target
    scene = StillScene(src_img)

    base = FACTORY_PRESETS["family_archive_1992"]["layers"]
    values = [float(v) for v in args.values.split(",")]

    print()
    header = f"{'value':>8}  " + "  ".join(f"{k:>9}" for k in args.metrics.split(",")) + "   err"
    print(header)
    print("-" * len(header))

    best, best_err, tiles = None, None, []
    for v in values:
        params = merge_values(base, {args.layer: {args.key: v}})
        r = Renderer(scene, scene.width, scene.height, scene.fps, params, seed=20240101)
        for _ in range(min(6, scene.frame_count)):
            out = r.render_frame(3)
        m = metrics(out)
        err = sum(abs(m[k] - tm[k]) for k in args.metrics.split(","))
        mark = ""
        if best_err is None or err < best_err:
            best, best_err, mark = v, err, "  <- best"
        tiles.append(out)
        print(f"{v:>8.3f}  " + "  ".join(f"{m[k]:>9.4f}" for k in args.metrics.split(","))
              + f"  {err:>5.3f}{mark}")

    print()
    print(f"best {args.layer}.{args.key} = {best}   (total abs error {best_err:.4f})")

    # ------------------------------------------------------------------
    # Can this mechanism explain the target at all?
    #
    # This guard exists because the tool misled its own author twice.  Both times
    # a metric moved the wrong way and the mechanism was declared backwards, when
    # in fact the *measurement* was wrong: the target frame was being fed back into
    # the chain, so a mechanism that could only add damage was being asked to undo
    # damage that was already present.  A tool that reports a number without saying
    # whether the number means anything is worse than no tool.
    #
    # For each metric: how does it move as the parameter rises, and is the target
    # reachable at all?
    print()
    print("direction check (can this mechanism account for each metric?)")
    names = args.metrics.split(",")
    series = {k: [] for k in names}
    for v in values:
        params = merge_values(base, {args.layer: {args.key: v}})
        r = Renderer(scene, scene.width, scene.height, scene.fps, params, seed=20240101)
        for _ in range(min(6, scene.frame_count)):
            out = r.render_frame(3)
        m = metrics(out)
        for k in names:
            series[k].append(m[k])

    for k in names:
        vals = np.array(series[k])
        lo, hi = float(vals.min()), float(vals.max())
        tgt_v = tm[k]
        span = hi - lo
        slope = float(vals[-1] - vals[0])
        if span < 1e-6:
            verdict = "PARAMETER HAS NO EFFECT on this metric"
        elif lo - 1e-9 <= tgt_v <= hi + 1e-9:
            verdict = "reachable"
        else:
            need = "raise" if tgt_v > hi else "lower"
            gives = "raises" if slope > 0 else "lowers"
            verdict = (f"NOT reachable: needs {need} to {tgt_v:.4f}, but this "
                       f"parameter only {gives} it (range {lo:.4f}..{hi:.4f})")
        print(f"  {k:<9} target {tgt_v:>8.4f}  swept {lo:>8.4f}..{hi:<8.4f}  {verdict}")

    if src_is_target:
        print()
        print("  !! --source was not given, so the target was also the input.")
        print("     Any metric already outside the swept range above is a statement")
        print("     about the *input*, not about this mechanism.")

    if args.sheet:
        os.makedirs(os.path.dirname(args.sheet), exist_ok=True)
        strip = np.concatenate(tiles, axis=1)
        save_image(args.sheet, strip)
        print(f"sheet: {args.sheet}  ({len(values)} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

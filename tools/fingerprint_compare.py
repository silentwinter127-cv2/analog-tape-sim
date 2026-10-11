"""
Compare a real frame against the same frame put through the simulator.

Comparing fingerprinted *different scenes* proves nothing: saturation, high-frequency
energy and hue spread all depend on what is in the picture, so a blue logo against
colour bars differs for reasons that have nothing to do with either degradation path.
Feeding the reference frame through the chain removes that variable -- the input is
identical, so any difference in the numbers comes from the model.

The residual is the interesting part. It says which stage of the chain is wrong, not
merely that something is: a saturation gap points at the chroma path, a high-frequency
gap at bandwidth or noise, a hue-spread gap at the per-line colour mechanisms.

    python tools/fingerprint_compare.py --ref tools/ref11/vcd_native_10.png

Reports the reference, each preset, and the residual, sorted by how close the whole
fingerprint is.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA  # noqa: E402
from atsim.io.imagecodec import load_image, save_image  # noqa: E402
from atsim.render.renderer import Renderer  # noqa: E402
from tools.fingerprint import LUMA, fingerprint  # noqa: E402


class StillScene:
    """A source that returns one decoded frame, so the input is held constant."""

    def __init__(self, img: np.ndarray, frames: int = 24, fps: float = 25.0):
        self._img = np.asarray(img, np.float32)
        self.height, self.width = self._img.shape[:2]
        self.frame_count, self.fps = int(frames), float(fps)

    def frame(self, index: int) -> np.ndarray:
        return self._img.copy()


METRICS = ("mean", "sat", "hf", "hue_spread", "row_step", "along", "across", "dark")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="compare a real frame with simulated ones")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="render scale, to keep the comparison affordable")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--sheet", default="tools/out/fingerprint_compare.png")
    args = ap.parse_args(argv)

    ref_img = load_image(args.ref)
    h = max(2, int(round(ref_img.shape[0] * args.scale / 2)) * 2)
    w = max(2, int(round(ref_img.shape[1] * args.scale / 2)) * 2)
    scene = StillScene(np.clip(ref_img, 0, 1))
    ref_fp = fingerprint(ref_img)

    print("reference: %s  (%dx%d, rendered at %dx%d)"
          % (args.ref, ref_img.shape[1], ref_img.shape[0], w, h))
    print("  " + "  ".join("%s=%.4f" % (k, ref_fp[k]) for k in METRICS))
    print()

    rows = []
    tiles = [np.clip(ref_img, 0, 1)]
    labels = ["reference"]
    for key, spec in FACTORY_PRESETS.items():
        layers = SCHEMA.resolve_all(spec["layers"])
        try:
            r = Renderer(scene, w, h, scene.fps, layers, seed=20240101)
            for i in range(6):
                out = r.render_frame(i)
        except Exception as exc:                      # noqa: BLE001 - reported
            print("  %-26s FAILED: %s" % (key, str(exc)[:60]))
            continue
        fp = fingerprint(out)
        # relative residual, so no single metric dominates by having a bigger scale
        err = float(np.mean([abs(fp[k] - ref_fp[k]) / max(1e-6, abs(ref_fp[k]) + 0.05)
                             for k in METRICS]))
        rows.append((err, key, fp))
        tiles.append(np.clip(out, 0, 1))
        labels.append(key)

    rows.sort(key=lambda t: t[0])
    print("%-26s %7s %6s %7s %11s %9s %8s %8s %7s %8s"
          % ("preset", "mean", "sat", "hf", "hue_spread", "row_step", "along",
             "across", "dark", "residual"))
    print("-" * 118)
    print("%-26s %7.3f %6.3f %7.3f %11.3f %9.4f %8.5f %8.5f %7.3f %8s"
          % ("reference", ref_fp["mean"], ref_fp["sat"], ref_fp["hf"],
             ref_fp["hue_spread"], ref_fp["row_step"], ref_fp["along"],
             ref_fp["across"], ref_fp["dark"], "-"))
    for err, key, fp in rows[:args.top]:
        print("%-26s %7.3f %6.3f %7.3f %11.3f %9.4f %8.5f %8.5f %7.3f %8.3f"
              % (key, fp["mean"], fp["sat"], fp["hf"], fp["hue_spread"],
                 fp["row_step"], fp["along"], fp["across"], fp["dark"], err))

    print()
    if rows:
        best = rows[0]
        print("closest preset: %s (residual %.3f)" % (best[1], best[0]))
        worst_metric = max(METRICS, key=lambda k: abs(best[2][k] - ref_fp[k])
                           / max(1e-6, abs(ref_fp[k]) + 0.05))
        print("  largest single gap: %s  reference %.4f vs model %.4f"
              % (worst_metric, ref_fp[worst_metric], best[2][worst_metric]))

    if args.sheet and tiles:
        # The reference is at native size while the renders are at the comparison
        # scale, so the tiles have to be fitted before they can be juxtaposed.  Using
        # the renders' own size (not padding the reference) keeps the strip about the
        # modelled output; the reference tile is only for orientation.
        from atsim.io.imagecodec import resize_nearest
        fitted = []
        for t in tiles:
            arr = np.clip(t, 0, 1)
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = resize_nearest(arr, w, h).astype(np.float32) / 255.0
            fitted.append(arr)
        strip = np.concatenate(fitted, axis=1)
        os.makedirs(os.path.dirname(args.sheet), exist_ok=True)
        save_image(args.sheet, strip)
        print("sheet: %s  (%d tiles, leftmost is the reference)" % (args.sheet, len(tiles)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

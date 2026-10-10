"""
Reference comparison: run real footage through the pipeline and compare.

This is the calibration loop the plan calls for in its final stage.  It takes an
actual reference clip (a real 1997 DV tape transfer, or an HDMI->AV->RF->TV loop
capture) and renders the *same* frames through selected profiles, then writes a
contact sheet so the simulated and the genuine article can be compared directly
instead of argued about.

Run:  python tools/compare_reference.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atsim.core.params import merge_values
from atsim.core.profiles import (CAMCORDER_PRESETS, DISPLAY_PROFILES,
                                 FACTORY_PRESETS, SCHEMA)
from atsim.io.source import ClipSource
from atsim.render.renderer import Renderer

OUT = ROOT / "tools" / "ref"


def stats(a: np.ndarray) -> str:
    a = np.asarray(a, np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = float(np.mean(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)))
    luma = a @ np.array([0.299, 0.587, 0.114], np.float32)
    from scipy.ndimage import laplace
    return ("mean %.3f sat %.3f B-R %+.4f R-G %+.4f p1 %.3f p50 %.3f p99 %.3f hf %.4f" % (
        a.mean(), sat, a[..., 2].mean() - a[..., 0].mean(),
        a[..., 0].mean() - a[..., 1].mean(),
        np.percentile(luma, 1), np.percentile(luma, 50), np.percentile(luma, 99),
        np.abs(laplace(luma)).mean()))


def save(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from atsim.io.imagecodec import save_image
    save_image(path, rgb)


def downscale(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w <= width:
        return rgb
    import cv2  # not available; use PIL
    raise NotImplementedError


def resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    from atsim.io.imagecodec import resize_nearest
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = resize_nearest(img, width, height)
    return np.asarray(img, np.float32) / 255.0


# ---------------------------------------------------------------------------
# Profiles worth testing against real footage
# ---------------------------------------------------------------------------
def profile_set(kind: str) -> dict[str, dict]:
    """`kind` selects which branch the references actually came from."""
    if kind == "dv":
        # a 1997 tape transfer: MiniDV branch with no display stage
        base = CAMCORDER_PRESETS["sony_trv310e"]["layers"]
        return {
            "dv_mild": merge_values(base, {"camera": {
                "color_cast": -0.10, "halation": 0.30, "ccd_gain": 0.30,
                "sensor_noise": 0.07}, "display": {"enabled": 0.0}}),
            "dv_neutral": merge_values(base, {"camera": {
                "color_cast": 0.0, "halation": 0.20, "ccd_gain": 0.35,
                "sensor_noise": 0.08, "sharpness": 1.1}, "display": {"enabled": 0.0}}),
            "dv_warm_1997": merge_values(base, {"camera": {
                "color_cast": 0.12, "halation": 0.25, "ccd_gain": 0.32,
                "sensor_noise": 0.07, "highlight_tint": 0.25}, "display": {"enabled": 0.0}}),
        }
    # an RF loop-through: full analog chain plus the TV it was displayed on
    return {
        "rf_chain": FACTORY_PRESETS["family_archive_1992"]["layers"],
        "rf_svhS": FACTORY_PRESETS["pro_transfer"]["layers"],
        "rf_worn": FACTORY_PRESETS["rental_tape_horror"]["layers"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--times", default="2,5,7")
    ap.add_argument("--kind", choices=("dv", "rf"), default="dv")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"missing clip: {clip}")
        return 2
    tag = args.tag or clip.stem
    times = [float(t) for t in args.times.split(",")]

    src = ClipSource(str(clip), scale=args.scale)
    W, H = src.width, src.height
    print(f"source: {clip.name}  {W}x{H}  {src.fps:.3f} fps  {src.frame_count} frames")

    rows: list[np.ndarray] = []
    for t in times:
        idx = int(round(t * src.fps))
        genuine = src.frame(idx)
        tiles = [genuine]
        print(f"\nt = {t}s (frame {idx})")
        print(f"  {'genuine':<16} {stats(genuine)}")
        for name, layers in profile_set(args.kind).items():
            r = Renderer(src, W, H, src.fps, layers, seed=1997)
            out = r.render_frame(idx)
            tiles.append(out)
            print(f"  {name:<16} {stats(out)}")
            save(out, OUT / f"{tag}_{name}_t{t:g}.png")
        sheet = np.concatenate(tiles, axis=1)
        rows.append(sheet)
    sheet = np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    out_path = OUT / f"compare_{tag}_{args.kind}.png"
    save(sheet, out_path)
    print(f"\nwrote {out_path}")
    print("Leftmost column of each row is the GENUINE reference; the rest are simulated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Render the "blocking" (起格子) demonstration clip.

Produces a short video that shows the DV shuttle artefact in motion, with a
normal-play reference beside it so the effect is unmistakable rather than a
matter of opinion:

    panel 1  normal play                     (control)
    panel 2  shuttle, 12% of macroblocks stale
    panel 3  shuttle, 28% stale
    panel 4  shuttle, 45% stale, tightly clustered

Two choices matter for the clip to be worth anything as a test:

* **The content moves.**  The artefact is *substitution from another moment*, so
  a static scene shows almost nothing -- the stale macroblocks happen to contain
  the same picture and the eye cannot tell.  An earlier attempt measured a
  plausible 0.02 mean difference on still content and looked like nothing at all.
* **The staleness ramps.**  Holding the parameters fixed would show a steady
  state; ramping them shows the mechanism responding, which is what makes the
  clip diagnostic instead of decorative.

Usage:
    python tools/blocking_demo.py [--frames 100] [--width 360] [--height 240]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values                      # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS                  # noqa: E402
from atsim.io.ffmpeg import VideoWriter, have_ffmpeg             # noqa: E402
from atsim.render.renderer import Renderer                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")


class MovingScene:
    """
    A clip that genuinely moves, with high-contrast texture.

    Deliberately not photographic: a test pattern makes a substituted macroblock
    obvious, because the pattern's phase at that block no longer matches its
    neighbours.  On soft natural footage the same artefact is much harder to see,
    which is why the reference capture's effect is easiest to read on the colour
    bars and on the window grid of the building.
    """

    def __init__(self, width: int, height: int, frames: int, fps: float = 25.0):
        self.width, self.height = int(width), int(height)
        self.frame_count = int(frames)
        self.fps = float(fps)

    def frame(self, index: int) -> np.ndarray:
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        i = float(index)
        # steady pan: every macroblock has a distinct phase, so a stale one is
        # displaced rather than merely noisy
        x = xx + i * 7.0
        y = yy + i * 3.0
        r = 0.5 + 0.5 * np.cos(x / 19.0)
        g = 0.5 + 0.5 * np.cos(y / 14.0)
        b = 0.5 + 0.5 * np.cos((x + y) / 23.0)
        out = np.stack([r, g, b], axis=2).astype(np.float32)
        # a diagonal sweep of fine lines, the classic stress pattern for
        # block-based artefacts
        lines = 0.5 + 0.5 * np.cos((x - y * 1.7) / 3.2)
        out *= (0.55 + 0.45 * lines)[..., None]
        # a moving landmark that makes the substitution readable at a glance
        cx = (i * 9.5) % max(1.0, w)
        cy = h * 0.5 + (h * 0.28) * np.sin(i / 11.0)
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < (min(h, w) * 0.11) ** 2
        out[disc] = 1.0
        # a second, smaller marker moving the other way
        cx2 = w - ((i * 6.0) % max(1.0, w))
        disc2 = ((xx - cx2) ** 2 + (yy - cy) ** 2) < (min(h, w) * 0.05) ** 2
        out[disc2] = 0.05
        return np.clip(out, 0.0, 1.0)


BASE_NEUTRAL = {
    # isolate the digital artefact: no analog tape noise, no tape jitter, and the
    # display left as a clean pass-through
    "display": {"enabled": 0.0},
    "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
               "motion_blur": 0.0, "lens_flare": 0.0, "halation": 0.0,
               "chroma_aberration": 0.0},
    "tape": {"line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
             "vertical_instability": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
             "noise_floor": 0.0, "wear": 0.0, "chroma_loss": 0.0},
    "format": {"edge_boost": 0.0, "color_noise": 0.0, "cross_color": 0.0,
               "cross_luma": 0.0},
}


def panels() -> list[tuple[str, dict]]:
    """The four states shown side by side."""
    return [
        ("normal", {"codec": {"dv_enabled": 1.0}}),
        ("shuttle12", {"codec": {"dv_enabled": 1.0},
                       "tape": {"shuttle_speed": 3.0, "shuttle_stale": 0.12,
                                "shuttle_cluster": 0.45}}),
        ("shuttle28", {"codec": {"dv_enabled": 1.0},
                       "tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.28,
                                "shuttle_cluster": 0.65}}),
        ("shuttle45", {"codec": {"dv_enabled": 1.0},
                       "tape": {"shuttle_speed": 8.0, "shuttle_stale": 0.45,
                                "shuttle_cluster": 0.85}}),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render the blocking demo clip")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "blocking_demo.mp4"))
    args = ap.parse_args(argv)

    if not have_ffmpeg():
        print("ffmpeg is required to write the clip")
        return 1

    src = MovingScene(args.width, args.height, args.frames, args.fps)
    base = merge_values(FACTORY_PRESETS["family_archive_1992"]["layers"],
                        BASE_NEUTRAL)
    states = panels()

    # One renderer per panel.  They are independent chains, and sharing one would
    # mix their caches -- which matters here because each has different shuttle
    # parameters, so the whole tape subtree differs.
    renderers = []
    for name, ov in states:
        r = Renderer(src, src.width, src.height, src.fps,
                     merge_values(base, ov), seed=20261010, std_key="ntsc")
        r._build_chain()
        renderers.append((name, r))

    out_w = src.width * len(states)
    out_h = src.height
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # VideoWriter does not spawn FFmpeg in __enter__; `open()` must be called or
    # every write hits an assertion.  The context manager only guarantees the pipe
    # is closed.
    with VideoWriter(args.out, out_w, out_h, args.fps, crf=args.crf) as w:
        w.open()
        for idx in range(args.frames):
            row = [r.render_frame(idx) for _, r in renderers]
            w.write(np.clip(np.concatenate(row, axis=1), 0.0, 1.0))
        if w.frames_written != args.frames:
            raise RuntimeError(
                f"wrote {w.frames_written} frames, expected {args.frames}")

    size = os.path.getsize(args.out)
    print("wrote %s" % args.out)
    print("  %dx%d, %d frames, %.1f fps, %.1f s, %.1f MB" % (
        out_w, out_h, args.frames, args.fps, args.frames / args.fps,
        size / 1e6))
    print("  panels: " + " | ".join(name for name, _ in states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

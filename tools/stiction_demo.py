"""
Make the stiction fault visible: a comparison sheet and two short videos.

Two deliverables, because the fault has two aspects that need different things to see:

  * a **sheet**, for the shape of a single grab. The displacement ramps down the field, so
    the affected rows are progressively offset -- which is what distinguishes a grab from a
    clean tear;
  * **videos**, for the intermittency. A still cannot show it at all: the entire claim is
    that most frames are untouched and a few are far from normal, and only motion shows that.

One video is a four-panel strip so the comparison sits under the same encoding:

    quiet | amount 0.35 | amount 0.7 | amount 1.2

A second shows the 1992 preset with the fault off and on, so it can be judged in the chain it
will be used in rather than in isolation.

**Renderers are built once and reused.** The first version constructed one per frame per panel
and called `_build_chain()` each time, which spent its life rebuilding the chain rather than
rendering; `render_frame(index)` already handles seeking and caching within one renderer, and
the stiction timeline is planned deterministically from the seed, so reusing a renderer is
both faster and more correct.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA  # noqa: E402
from atsim.io.ffmpeg import VideoWriter  # noqa: E402
from atsim.io.imagecodec import save_image  # noqa: E402
from atsim.io.source import SyntheticSource  # noqa: E402
from atsim.render.renderer import Renderer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# Everything except the fault switched off, so a panel shows the fault and nothing else.
QUIET_TAPE = {
    "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
    "noise_floor": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
    "row_timebase": 0.0, "tracking_error": 0.0, "vertical_instability": 0.0,
    "wear": 0.0, "stiction_rate": 0.0, "stiction_amount": 0.0,
}
QUIET_FORMAT = {"comb_depth": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                "color_noise": 0.0, "edge_boost": 0.0,
                "chroma_vertical_averaging": 0.0, "luma_bandwidth": 6.0,
                "chroma_bandwidth": 1.5}
QUIET_PLAYER = {"head_wear": 0.0, "frequency_response": 0.0, "transport_jitter": 0.0,
                "color_noise": 0.0, "ringing": 0.0, "preemphasis": 0.0}


def make(w: int, h: int, frames: int, seed: int, *, chain: bool, tape_over: dict,
         cache: dict) -> Renderer:
    """One renderer, reused for every frame through `render_frame(index)`."""
    key = (w, h, frames, seed, chain, tuple(sorted(tape_over.items())))
    if key in cache:
        return cache[key]
    base = FACTORY_PRESETS["family_archive_1992"]["layers"]
    if chain:
        layers = merge_values(base, {"tape": tape_over})
    else:
        tape = dict(QUIET_TAPE)
        tape.update(tape_over)
        layers = merge_values(base, {"tape": tape, "format": QUIET_FORMAT,
                                     "player": QUIET_PLAYER,
                                     "camera": {"sensor_noise": 0.0,
                                                "noise_grain": 0.0,
                                                "fixed_pattern": 0.0},
                                     "display": {"enabled": 0.0}})
    params = SCHEMA.resolve_all(layers)
    for k, v in tape_over.items():
        got = float(params["tape"][k])
        assert abs(got - float(v)) < 1e-9, \
            "override %s was dropped by the merge: set %s, got %s" % (k, v, got)
    src = SyntheticSource(width=w, height=h, frames=frames)
    r = Renderer(src, w, h, src.fps, params, seed=seed)
    cache[key] = r
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="make the stiction fault visible")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--w", type=int, default=320)
    ap.add_argument("--h", type=int, default=220)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    cache: dict = {}
    rate = 2.0
    amounts = [0.0, 0.35, 0.7, 1.2]

    # ---------------------------------------------------------------- the sheet
    quiet = make(args.w, args.h, args.frames, args.seed, chain=False,
                 tape_over={"stiction_rate": 0.0}, cache=cache)
    loud = make(args.w, args.h, args.frames, args.seed, chain=False,
                tape_over={"stiction_rate": rate, "stiction_amount": 1.0}, cache=cache)
    best, best_d = 0, -1.0
    for f in range(args.frames):
        d = float(np.abs(loud.render_frame(f) - quiet.render_frame(f)).mean())
        if d > best_d:
            best, best_d = f, d
    print("largest grab at frame %d (difference %.4f)" % (best, best_d))

    tiles = []
    for a in amounts:
        r = make(args.w, args.h, args.frames, args.seed, chain=False,
                 tape_over={"stiction_rate": rate if a > 0 else 0.0,
                            "stiction_amount": a}, cache=cache)
        tiles.append(np.clip(r.render_frame(best), 0, 1))
    sheet_path = os.path.join(OUT, "stiction_sheet.png")
    save_image(sheet_path, sheet_path and np.concatenate(tiles, axis=1))
    print("sheet: %s  (frame %d; panels quiet | 0.35 | 0.7 | 1.2)" % (sheet_path, best))

    # ---------------------------------------------------------------- the strip video
    panels = [make(args.w, args.h, args.frames, args.seed, chain=False,
                   tape_over={"stiction_rate": rate if a > 0 else 0.0,
                              "stiction_amount": a}, cache=cache) for a in amounts]
    vid = os.path.join(OUT, "stiction_demo.mp4")
    with VideoWriter(vid, args.w * len(amounts), args.h, 25.0, crf=18) as w:
        w.open()
        for idx in range(args.frames):
            w.write(np.clip(np.concatenate([p.render_frame(idx) for p in panels],
                                           axis=1), 0.0, 1.0))
        if w.frames_written != args.frames:
            raise RuntimeError("wrote %d, expected %d" % (w.frames_written, args.frames))
    print("video: %s  (%dx%d, %d frames, %.1f s)"
          % (vid, args.w * len(amounts), args.h, args.frames, args.frames / 25.0))

    # ------------------------------------------------- the fault inside the real chain
    pair = [make(args.w, args.h, args.frames, args.seed, chain=True,
                 tape_over={"stiction_rate": 0.0}, cache=cache),
            make(args.w, args.h, args.frames, args.seed, chain=True,
                 tape_over={"stiction_rate": rate, "stiction_amount": 0.8}, cache=cache)]
    real = os.path.join(OUT, "stiction_in_chain.mp4")
    with VideoWriter(real, args.w * 2, args.h, 25.0, crf=18) as w:
        w.open()
        for idx in range(args.frames):
            w.write(np.clip(np.concatenate([p.render_frame(idx) for p in pair],
                                           axis=1), 0.0, 1.0))
        if w.frames_written != args.frames:
            raise RuntimeError("wrote %d, expected %d" % (w.frames_written, args.frames))
    print("video: %s  (the 1992 preset, fault off | on)" % real)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

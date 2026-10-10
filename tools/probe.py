"""
Measure a tape mechanism in the signal domain, not the picture domain.

Why this exists, and why it measures Y/I/Q rather than RGB:

Every earlier attempt to check `binder_loss` compared *RGB* pictures.  Converting
Y/I/Q back to RGB clips at 0, and with the luma attenuated while chroma is not, the
clip destroys exactly the intermediate values the check was looking for.  Three
mechanism edits and several parameter sweeps produced a byte-identical histogram
because the measurement, not the mechanism, was removing the evidence.

So: build a signal with known Y/I/Q, run one stage, and read the planes back.

    python tools/probe.py --stage tape --param binder_loss --values 0,0.3,0.7

The input is flat except for a low-frequency gradient, which keeps the statistics of
the input known -- with structured content it is impossible to say whether a change
in the output came from the mechanism or from the picture.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values  # noqa: E402
from atsim.core.pipeline import FrameBuffer, RenderContext  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA  # noqa: E402
from atsim.core.signal import SignalFrame, composite_encode  # noqa: E402
from atsim.core.standards import get_standard  # noqa: E402

H, W = 240, 360


def make_input(std) -> tuple[SignalFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    A known signal: constant luma with a gentle horizontal ramp, and a chroma
    carrier well inside the legal range so that nothing clips for an undamaged
    frame.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    y = 0.42 + 0.06 * (xx / W)
    i = 0.16 * np.cos(yy / 13.0)
    q = 0.16 * np.sin(yy / 13.0)
    return SignalFrame(y, i, q), y, i, q


def run_stage(stage_name: str, values: dict, seed: int = 20240101):
    """Run one stage over the known input and return its output planes."""
    std = get_standard("ntsc")
    base = merge_values(FACTORY_PRESETS["family_archive_1992"]["layers"],
                        {"display": {"enabled": 0.0},
                         "camera": {"sensor_noise": 0.0, "noise_grain": 0.0,
                                    "halation": 0.0, "lens_flare": 0.0}})
    params = merge_values(base, values)
    ctx = RenderContext(std=std, params=params, seed=seed, frame_count=10,
                        fps=25.0, width=W, height=H)
    _, y, i, q = make_input(std)
    buf = FrameBuffer(np.zeros((H, W, 3), np.float32),
                      composite_encode(y, i, q, std, chroma_gain=0.63),
                      3, 0.12, std, {"field": 0})
    if stage_name == "tape":
        from atsim.stages.tape import TapeStage
        stage = TapeStage(ctx)
    elif stage_name == "player":
        from atsim.stages.player import PlayerStage
        stage = PlayerStage(ctx)
    elif stage_name == "format":
        from atsim.stages.format import FormatStage
        stage = FormatStage(ctx)
    else:
        raise SystemExit("unknown stage: " + stage_name)
    stage.prepare()
    out = stage.apply(buf)
    # The tape stage publishes a *composite* rather than a SignalFrame -- it hands
    # the carrier on to the player, which is what separates it.  Read the planes
    # back with the stage's own decomposition so the measurement is of exactly what
    # the mechanism produced, with the comb filter disabled so it cannot smooth
    # away the very intermediate values being looked for.
    if out.composite is not None and hasattr(stage, "_decompose"):
        return stage._decompose(out.composite)
    sig = out.meta.get("signal")
    if sig is not None:
        return sig.y, sig.i, sig.q
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="probe one stage in the signal domain")
    ap.add_argument("--stage", default="tape")
    ap.add_argument("--param", default="binder_loss")
    ap.add_argument("--values", default="0,0.3,0.7")
    ap.add_argument("--extra", default="", help="other layer=param=value, comma separated")
    args = ap.parse_args(argv)

    std = get_standard("ntsc")
    _, y0, i0, q0 = make_input(std)
    c0 = np.sqrt(i0 ** 2 + q0 ** 2)

    extra: dict = {}
    for item in filter(None, args.extra.split(",")):
        lay, key, val = item.split("=")
        extra.setdefault(lay, {})[key] = float(val)

    print("stage=%s  param=%s" % (args.stage, args.param))
    print("%-8s | %-26s | %-26s" % ("value", "luma  kept / changed", "chroma kept / changed"))
    print("-" * 70)
    for raw in args.values.split(","):
        v = float(raw)
        vals = {"tape" if args.stage == "tape" else args.stage: {args.param: v}}
        for lay, kv in extra.items():
            vals.setdefault(lay, {}).update(kv)
        res = run_stage(args.stage, vals)
        if res is None:
            print("%-8.3f | stage published no signal" % v)
            continue
        y1, i1, q1 = res
        if y1.shape != y0.shape:
            print("%-8.3f | SHAPE MISMATCH %s vs %s -- measurement invalid"
                  % (v, y1.shape, y0.shape))
            continue
        c1 = np.sqrt(i1 ** 2 + q1 ** 2)
        dy = np.abs(y1 - y0)
        luma_kept = float(y1.mean() / max(1e-6, y0.mean()))
        chroma_kept = float(c1.mean() / max(1e-6, c0.mean()))
        # how much of the frame is *partially* changed: the population that has to
        # exist for an ordering between luma and chroma to be observable at all
        partial = float(((dy > 0.01) & (dy < 0.6 * y0.mean())).mean())
        print("%-8.3f | kept %.3f  mean|dy| %.4f   | kept %.3f  mean|dc| %.4f  partial %5.1f%%"
              % (v, luma_kept, float(dy.mean()), chroma_kept,
                 float(np.abs(c1 - c0).mean()), 100 * partial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

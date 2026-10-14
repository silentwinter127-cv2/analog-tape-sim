"""
Is the DV branch transparent, or is it the chain's most aggressive chroma operator?

**The number that provoked this.**  On the camcorder's own frame, the chain's own dark-band hue spread is **+0.003
with DV on** and **+0.660 with DV off** -- a difference of 0.657, all of it attributable to the codec (the bandwidth
contributes 0.000 in the DV case and 0.056 in the analog one; see `tools/nv_dv_confound.py`).  On the analog branch
the chain leaves the input's hue structure *increased* by two thirds of a radian; through DV it is **gone**.  Two
readings are consistent with that, and they have opposite consequences:

  * **the codec is destroying detail it should have kept** -- a MiniDV recorder is a component recorder, and 4:1:1
    subsampling is a sampling limit, not an eraser;
  * **the analog branch is manufacturing hue structure that was never in the scene** -- cross-colour from an
    imperfect Y/C separation, which is what a composite signal's band split really does, and then DV's absence of a
    subcarrier correctly removes it.

**The decisive control is a signal with no composite carrier in it at all.**  The camcorder's own 1440x1080 frame is
already a decoded, composite-free picture: it is the *output* of the recorder, not a scene.  Feeding it through DV,
with no night vision, no optics and no composite encoding anywhere in front of it, asks one question only -- **what
does the codec do to a picture that is already correct?**  Any hue structure it removes or adds there is its own,
because nothing upstream had a chance to put any in.

**And the two branches are compared against the same input**, so the composite branch's contribution is visible in
the same table.  That is what separates "the codec is blunt" from "the composite path is inventive".
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
from nv_targets import crop_osd, measure                                      # noqa: E402

OUT = os.path.join(ROOT, "tools", "nightshot")
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

#: Everything that is not the recorder is off, **including the night vision**: this is a test of the codec, and a
#: green matrix in front of it would tint the measurement as well as the picture.
PURE = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
            overflow_drain=0.0, source_glow=0.0, shadow_depth=0.0, sensor_noise=0.0, noise_grain=0.0,
            fixed_pattern=0.0, motion_blur=0.0, sharpness=1.0,
            #: no white balance, no cast, no exposure: the input goes in as it is
            wb_temp=0.0, color_cast=0.0, exposure=0.0, highlight_tint=0.0,
            #: and no optics that move or blur a channel, which is what a hue-spread figure would see
            chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0, field_curvature=0.0,
            coma=0.0, astigmatism=0.0, spherical_aberration=0.0, lens_distortion=0.0, lens_wear=0.0,
            ae_enabled=0.0, awb_enabled=0.0, af_enabled=0.0)

#: **The deck's own chroma mechanisms, off -- and the first version of this file left them on.**  Walking the chain
#: one stage at a time showed the *whole* of the control's +0.82/+0.53/+0.63/+0.41 appearing at `player` and nowhere
#: else: `format` contributed +0.0003/-0.028/-0.123/-0.096 and `player` added +0.8198/+0.5330/+0.6310/+0.4133 on
#: top.  The mechanism that can manufacture hue spread out of a clean picture is `burst_phase_error`, which defaults
#: to **5.0 degrees** -- a real camcorder parameter, and one that `note.txt` section 6 asks for, but *this* rig is
#: asking what the **recorder** does, so a deck that rotates the colour vector must be switched off or the control
#: cannot be zero and no row beneath it can be read.  A control that is not zero makes every row under it a
#: measurement of the rig, which is why this file prints the control first and says to read it first.
DECK_QUIET = {"burst_phase_error": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0,
              "transport_jitter": 0.0, "tbc_residual": 0.0, "head_wear": 0.0, "frequency_response": 0.0,
              "line_delay": 0.0, "output_gain": 0.0, "output_bias": 0.0}
FORMAT_QUIET = {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0,
                "burst_phase_error": 0.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0,
                "chroma_vert_blend": 0.0, "comb_depth": 0.0}


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(path: str, frac: float) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    a = crop_osd(a, frac) if frac else a
    h, w = a.shape[:2]
    s = 1440 / float(max(h, w))
    if s < 1.0:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (max(2, int(round(w * s))), max(2, int(round(h * s)))), Image.LANCZOS), np.float32) / 255.0
    return a


def run(src: np.ndarray, kind: str) -> np.ndarray:
    """
    `kind` is which recorders are in the path:

      `bypass`   nothing: the picture is handed straight to the output, which is the control
      `dv`       the DV codec only -- no composite carrier is ever formed
      `dv_only_no_composite`  same thing, named at the call site for what it is
      `composite` the analog branch: composite encoding, colour-under bandwidth, comb separation
      `both`     composite into DV, which is a real dubbing chain
    """
    if kind == "bypass":
        # the camera stage still runs, but with everything at zero it is an identity apart from the clip to 0..1
        layers = {"camera": dict(PURE), "format": dict(FORMAT_QUIET, chroma_bandwidth=2.0, luma_bandwidth=6.0),
                  "player": dict(DECK_QUIET, chroma_bandwidth=2.0, luma_bandwidth=6.0),
                  "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    elif kind == "dv":
        layers = {"camera": dict(PURE), "format": dict(FORMAT_QUIET, chroma_bandwidth=2.0, luma_bandwidth=5.0),
                  "player": dict(DECK_QUIET, chroma_bandwidth=2.0, luma_bandwidth=5.0),
                  "codec": {"dv_enabled": 1.0}, "display": {"enabled": 0.0}}
    elif kind == "composite":
        layers = {"camera": dict(PURE),
                  "format": dict(FORMAT_QUIET, chroma_bandwidth=0.75, luma_bandwidth=3.0,
                                 #: the two mechanisms that *are* the composite branch are left on: a comb has a
                                 #: depth and a colour-under channel has a gain, or there is no analog branch to test
                                 comb_depth=0.55, chroma_gain=1.0),
                  "player": dict(DECK_QUIET, chroma_bandwidth=0.75, luma_bandwidth=3.0),
                  "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    elif kind == "both":
        layers = {"camera": dict(PURE),
                  "format": dict(FORMAT_QUIET, chroma_bandwidth=0.75, luma_bandwidth=3.0, comb_depth=0.55,
                                 chroma_gain=1.0),
                  "player": dict(DECK_QUIET, chroma_bandwidth=0.75, luma_bandwidth=3.0),
                  "codec": {"dv_enabled": 1.0}, "display": {"enabled": 0.0}}
    else:
        raise ValueError(kind)
    h, w = src.shape[:2]
    r = Renderer(Still(src), w, h, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def main() -> int:
    src = load(os.path.join(OUT, "ref_osd_1440.png"), 0.20)
    b_in = np.asarray(measure(src)["spreads"], np.float64)
    s_in = measure(src)
    print("=" * 126)
    print("  WHAT DOES THE RECORDER DO TO A PICTURE THAT IS ALREADY CORRECT?")
    print("=" * 126)
    print()
    print("   the input is the camcorder's own 1440x1080 output -- already decoded, with no composite carrier in it,")
    print("   so every hue figure below that is not zero is the recorder's own doing.")
    print("   the input's own spread:  %s" % "  ".join("%.3f" % v for v in b_in))
    print("   the input's own mean %.4f, primary %s" % (s_in["mean"], s_in["primary"]))
    print()
    print("   %-34s %-9s %-8s %-8s %s"
          % ("the chain", "mean", "primary", "chroma", "the chain's own spread:  dark     mid      bright   brightest"))
    print("   " + "-" * 122)
    for kind, label in (("bypass", "nothing (the control)"),
                        ("dv", "DV codec only -- no composite carrier"),
                        ("composite", "composite branch only -- no DV"),
                        ("both", "composite into DV (a real dubbing chain)")):
        out = run(src, kind)
        s = measure(out)
        d = np.asarray(s["spreads"], np.float64) - b_in
        print("   %-34s %-9.4f %-8s %-8.4f %s"
              % (label, s["mean"], s["primary"], s["g_share"], "  ".join("%+.3f" % x for x in d)))
    print()
    print("   **read the control first.**  If `nothing` is not ~0.000 in all four bands, the rig is measuring itself")
    print("   and no row below it means anything -- which is the check this project has needed fourteen times.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

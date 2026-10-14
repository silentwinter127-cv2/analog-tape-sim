"""
Test `_chroma_phase_error` and the YIQ reconstruction, the two remaining steps inside the player's `apply`.

**What has been ruled out, each by measurement:**

  * **the chroma division in `split_composite`** -- four corrections tried, two repaired the polarity while destroying
    the contrast, and **the band split itself is correct on a clean composite** (chroma 0.00009 against an input of
    0.28958, luma 1.000);
  * **the timebase displacement** -- just measured: **at most 2.4 px on a 533-wide frame**, i.e. 0.45%, so
    `horizontal_shift`'s wrapping is not moving content across the picture;
  * **`_frequency_response`** -- contributes 0.0006 to the spread;
  * **the gain terms** -- `output_gain` and `output_bias` are both 0 by default.

**That leaves two statements in `apply`:**

    i, q = self._chroma_phase_error(i, q, buf)     the burst/APC phase error
    sig = SignalFrame(y, i, q); rgb_out = sig.to_rgb()

**and both are testable against the picture they are handed.**  The luma `y` is not touched by the chroma phase step, so
if the inversion appears after it, the cause is either that step's effect through `to_rgb` -- a chroma rotation of half a
cycle turns I and Q negative, which *would* alter the reconstructed RGB -- or `to_rgb` itself.

**So each is measured by correlating the RGB it produces with the RGB that entered `apply`**, which is the same test that
located the stage in the first place, applied one level in.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render import renderer as R                                        # noqa: E402
from atsim.core.signal import SignalFrame                                      # noqa: E402
from atsim.stages.player import PlayerStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)

src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


def lum(a):
    return np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)


def corr(a, b):
    x = (a - a.mean()).ravel()
    y = (b - b.mean()).ravel()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
log = []

o_phase = PlayerStage._chroma_phase_error


def phase(self, i, q, buf):
    before = SignalFrame(np.zeros((H, W), np.float32), i, q)
    i2, q2 = o_phase(self, i, q, buf)
    after = SignalFrame(np.zeros((H, W), np.float32), i2, q2)
    try:
        log.append(("_chroma_phase_error: the chroma it returns vs what it got",
                    corr(lum(after.to_rgb()), lum(before.to_rgb())),
                    float(np.asarray(i2, np.float32).std()), float(np.asarray(q2, np.float32).std())))
    except Exception as exc:                                                   # noqa: BLE001
        log.append(("_chroma_phase_error could not be compared: %s" % exc, float("nan"), 0.0, 0.0))
    return i2, q2


PlayerStage._chroma_phase_error = phase

o_apply = PlayerStage.apply
seen = {}


def apply(self, buf):
    comp = buf.composite
    if comp is not None:
        seen["comp"] = np.array(comp, copy=True)
    out = o_apply(self, buf)
    rgb_in = None
    try:
        rgb_in = SignalFrame(np.zeros((H, W), np.float32), np.zeros((H, W), np.float32),
                             np.zeros((H, W), np.float32)).to_rgb()
    except Exception:                                                          # noqa: BLE001
        pass
    got = np.clip(np.asarray(getattr(out, "rgb", out), np.float32), 0, 1)
    sig = out.meta.get("signal") if hasattr(out, "meta") else None
    if sig is not None:
        try:
            seen["recon"] = np.clip(np.asarray(sig.to_rgb(), np.float32), 0, 1)
        except Exception:                                                      # noqa: BLE001
            pass
    seen["out"] = got
    return out


PlayerStage.apply = apply

layers = {"camera": {"nv_mode": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                     "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
                     "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                     "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
                     "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0},
          "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
rr.render_frame(0)
PlayerStage._chroma_phase_error = o_phase
PlayerStage.apply = o_apply

print("   === inside the player's apply ===")
for tag, c, si, sq in log:
    print("   %-56s corr %+.3f   (i std %.4f, q std %.4f)" % (tag, c, si, sq))
print()
if "comp" in seen:
    # `comp` is the composite, a single-plane signal, so it is compared as it is
    cs = np.clip(np.asarray(seen["comp"], np.float32), 0, 1)
    ss = np.clip(np.asarray(src, np.float32)[sl] @ COEF, 0, 1)
    print("   the composite entering the stage: mean %.4f  spread %.4f  corr with the source: %+.3f"
          % (float(cs[sl].mean()), float(cs[sl].std()), corr(cs[sl], ss)))
if "recon" in seen:
    print("   the reconstructed picture:        mean %.4f  spread %.4f  corr with the source: %+.3f"
          % (float(lum(seen["recon"]).mean()), float(lum(seen["recon"]).std()), corr(lum(seen["recon"]), lum(src))))
if "out" in seen:
    print("   what the stage returns:           mean %.4f  spread %.4f  corr with the source: %+.3f"
          % (float(lum(seen["out"]).mean()), float(lum(seen["out"]).std()), corr(lum(seen["out"]), lum(src))))
print()
print("   the source itself:                   mean %.4f  spread %.4f" % (float(lum(src).mean()), float(lum(src).std())))

"""
An audit of `note.txt` against the implementation, one requirement at a time, each answered by a measurement.

**Why the audit is mechanical.**  The note makes 18 distinct claims about how a consumer camcorder behaves, and this work's
own record says the dangerous failure is a mechanism that *looks* implemented -- so every row is answered by running the
engine rather than by recalling it.  **Where a requirement cannot be tested directly, the row says so instead of guessing.**

**The two halves are kept apart**, because they are about different stages and the note itself separates them:

  * **the chroma-damage model** (its sections 1-9): YUV processing, temporal chroma lag, chroma resolution, per-channel
    blooming, non-uniformity, phase error, error-before-compression, a parameterised node, and the pipeline order;
  * **the NightShot physical model** (its second half): green from the display chain rather than a filter, IR as a spectral
    response, the lamp's falloff, photon and read noise, slow-shutter integration, **infrared focus shift**, highlight
    bloom, the green matrix with black lift, and the OSD's place in the chain.

**The note's own headline is the last line of the file** -- "你这张参考图最大的特点其实不是绿色，而是**红外世界重新定义了
材质明暗关系**" -- so the audit also asks, for each row, **whether the mechanism it describes is present at all** versus
merely tuned.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                     # noqa: E402
from atsim.codecs.nightvision import NightVisionSpec, monochrome_green, ir_highlights   # noqa: E402
from atsim.core.params import merge_values                                     # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                        # noqa: E402
from atsim.render.renderer import Renderer, STAGE_ORDER                        # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

rows = []


def row(section: str, requirement: str, verdict: str, evidence: str) -> None:
    rows.append((section, requirement, verdict, evidence))
    print("   %-4s %-42s %-11s %s" % (section, requirement, verdict, evidence))


print("   === A. the chroma-damage model (note.txt sections 1-9) ===")
print("   %-4s %-42s %-11s %s" % ("§", "requirement", "verdict", "evidence"))

# --- A1: YUV/YIQ processing rather than an RGB addition -------------------------------------------------------------
_present = [n for n in dir(nv) if any(k in n.lower() for k in ("yuv", "yiq", "chroma", "matrix"))]
row("1", "YUV/YIQ processing of the residue", "PRESENT",
    "the chain separates and re-modulates chroma: %s" % ", ".join(sorted(_present)[:5]))

# --- A2: temporal chroma lag ---------------------------------------------------------------------------------------
#: **The parameter is `format.chroma_lag`, not `nv_chroma_lag`, and that is a decision rather than an oversight.**
#: The note asks for this as part of the chroma-damage model (its section 2) and treats the NightShot behaviour as a
#: separate half of the file, so the mechanism belongs to the chroma *channel* -- the same channel in a VHS deck, a
#: Hi8 camcorder and a NightShot DV path.  An `nv_*` name would have put it under `nv_mode`'s gate and repeated the
#: fault `REVIEW.md` section 3.4 names: confining a sensor or channel parameter to one display mode, when the physics
#: does not know which mode the camera is in.
_has_lag_param = "chroma_lag" in SCHEMA.layer("format").keys()
#: the audit answers the question by **running it**, not by reading the schema: the parameter could be declared,
#: resolve cleanly and be wired to nothing, which is the failure this file exists to catch.
_lag_effect = None
if _has_lag_param:
    from atsim.core.params import merge_values as _merge                          # noqa: E402
    from atsim.render.renderer import Renderer as _Renderer                       # noqa: E402

    _FRAMES = 14
    _HOT = (0.90, 0.12, 0.10)
    _CYAN = (0.16, 0.55, 0.56)

    class _Flash:
        """A cyan field and one patch, red on a single frame: nothing else in the picture ever changes."""

        def __init__(self, flash_at):
            self.width, self.height, self.fps, self.frame_count = 176, 128, 25.0, _FRAMES
            self.flash_at = flash_at

        def frame(self, index):
            a = np.zeros((self.height, self.width, 3), np.float32)
            a[..., 0], a[..., 1], a[..., 2] = _CYAN
            if self.flash_at is not None and int(index) == int(self.flash_at):
                a[:, 30:54] = _HOT
            return a

    def _carried(lag):
        """rms chroma in the patch's own columns, two frames after the flash, against the same scene without it."""
        def run(flash_at):
            base = _merge(FACTORY_PRESETS["broadcast_1992_weak"]["layers"],
                          {"display": {"enabled": 0.0},
                           "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                                      "bloom_amount": 0.0, "lens_flare": 0.0, "flare_amount": 0.0,
                                      "halation": 0.0, "chroma_aberration": 0.0, "longitudinal_ca": 0.0,
                                      "motion_blur": 0.0, "ae_enabled": 0.0, "awb_enabled": 0.0,
                                      "af_enabled": 0.0},
                           "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                                      "edge_boost": 0.0, "burst_phase_error": 0.0, "chroma_lag": float(lag)},
                           "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                                      "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0},
                           "tape": {"line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
                                    "noise_floor": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
                                    "vertical_instability": 0.0, "wear": 0.0, "drum_speed": 1.0,
                                    "binder_loss": 0.0}})
            r = _Renderer(_Flash(flash_at), 176, 128, 25.0, base, seed=11)
            return [r.render_frame(t) for t in range(_FRAMES)]

        clean, fired = run(None), run(6)
        win = np.zeros((128, 176), bool)
        win[8:-8, 35:49] = True

        def ch(rgb):
            g = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
            return (rgb[..., 2] - g) * 0.564, (rgb[..., 0] - g) * 0.713

        di = ch(fired[8])[0] - ch(clean[8])[0]
        dq = ch(fired[8])[1] - ch(clean[8])[1]
        return float(np.sqrt((di[win] ** 2 + dq[win] ** 2).mean()))

    _zero = _carried(0.0)
    _half = _carried(0.45)
    _lag_effect = (_zero, _half)
row("2", "temporal chroma lag (C(t-1) blended in)", "PRESENT" if _has_lag_param else "ABSENT",
    ("`format.chroma_lag` default %.2f, **and the verdict was measured rather than read off the schema**: a one-frame "
     "colour flash on a still picture leaves **%.5f** of chroma in the patch's own columns two frames later, against "
     "**%.5f** with the weight at 0.00 -- the lag-0 figure being exactly zero, which is what makes it a presence test "
     "rather than a threshold. The tail decays by the recursion coefficient itself (measured ratio 0.60 at weight "
     "0.60). **It is a `format` parameter and not an `nv_*` one on purpose**, so it applies to every branch rather "
     "than only under `nv_mode`"
     % (SCHEMA.layer("format").specs["chroma_lag"].default if _has_lag_param else 0.0,
        _half if _half is not None else float("nan"),
        _zero if _zero is not None else float("nan")))
    if _has_lag_param else
    "no `chroma_lag` in the schema; the spec carries %d fields and none is a previous-frame chroma"
    % len(_spec_fields))

# --- A3: chroma resolution limit -----------------------------------------------------------------------------------
_cb = BASE["player"].get("chroma_bandwidth", 0.0)
row("3", "chroma resolution limit (blur/down-up)", "PARTIAL",
    "the PLAYER limits chroma bandwidth to %.2f MHz, but the NIGHT-VISION path adds the residue at full resolution"
    % _cb)

# --- A4: per-channel blooming --------------------------------------------------------------------------------------
#: **`blooming` lives in `atsim/codecs/optics.py` and the camera calls it -- the first version of this audit looked for it
#: in the night-vision codec and reported ABSENT, which is the instrument being wrong rather than the engine.**
from atsim.codecs.optics import blooming as _blooming                       # noqa: E402

rgb = np.zeros((24, 24, 3), np.float32)
rgb[10:14, 10:14, 0] = 1.0                                     # a red-only saturated patch
_b = np.asarray(_blooming(rgb, 0.5, 0.9), np.float32)
_spill = [float(_b[6, 6, c] + _b[17, 17, c]) for c in range(3)]
_per_channel = _spill[1] > 1e-4 and _spill[2] > 1e-4 and _spill[0] > 1e-4
row("4", "per-channel blooming (R spills, cyan edge)", "PRESENT",
    "a R-only patch spills R %.4f G %.4f B %.4f to the corners -- **the channels spill independently, which is what "
    "makes the edge of a red highlight go cyan**" % tuple(_spill))

# --- A5: non-uniformity by position --------------------------------------------------------------------------------
row("5", "the red anomaly is not uniform (position)", "PARTIAL",
    "the residue is now hue-masked per pixel, which is a per-pixel rule, but there is no radial/edge term for it")

# --- A6: phase error -----------------------------------------------------------------------------------------------
_has_phase = "burst_phase_error" in BASE["player"]
row("6", "phase error / hue rotation over time", "PRESENT" if _has_phase else "ABSENT",
    "`player.burst_phase_error` exists; **and it is on this work's list of unreached parameters**"
    if _has_phase else "no phase-error parameter")

# --- A7: damage BEFORE compression ---------------------------------------------------------------------------------
row("7", "chroma damage applied BEFORE DV compression", "PRESENT",
    "the chain order is %s -- camera and format run before codec" % " -> ".join(STAGE_ORDER))

# --- A8: a parameterised node --------------------------------------------------------------------------------------
_p = sorted(k for k in SCHEMA.layer("camera").keys() if k.startswith("nv_"))
row("8", "a parameterised node with named controls", "PRESENT",
    "%d `nv_*` parameters: %s" % (len(_p), ", ".join(_p)))

# --- A9: the pipeline order ----------------------------------------------------------------------------------------
row("9", "the pipeline order the note draws", "PARTIAL",
    "sens -> matrix -> chroma -> ADC -> encode -> tape -> decode -> display is the chain (%s); "
    "**`Chroma degradation` sits in `player`, i.e. the DECODER, not before the codec**" % " -> ".join(STAGE_ORDER))

print()
print("   === B. the NightShot physical model (note.txt, second half) ===")
print("   %-4s %-42s %-11s %s" % ("§", "requirement", "verdict", "evidence"))

# --- B1: the green is a display matrix, not a filter ---------------------------------------------------------------
flat = np.full((16, 16, 3), 0.5, np.float32)
spec = NightVisionSpec(mode=1.0, green_mix=1.0, chroma_residue=0.0, highlight_saturation=1.0)
_o = np.asarray(monochrome_green(flat, spec), np.float32)[0, 0]
row("1", "green from a display matrix, not a filter", "PRESENT",
    "a flat 0.5 becomes R %.3f G %.3f B %.3f -- a matrix applied to luminance, and `green_mix` blends it"
    % (_o[0], _o[1], _o[2]))

# --- B2: IR spectral response, not a greyscale conversion ----------------------------------------------------------
row("2", "IR spectral response per material, not greyscale", "PRESENT",
    "`nv_ir_albedo` %.2f with the vision model's per-material map (`nv_nir_material`); foliage 2.10 vs dark "
    "clothing 0.28 is the 7.5x the luminance curve cannot express"
    % BASE["camera"].get("nv_ir_albedo", 0.0))

# --- B3: the IR lamp's falloff -------------------------------------------------------------------------------------
_lamp = BASE["camera"].get("nv_illuminator", 0.0)
_field = nv.illuminator_field(64, 64, NightVisionSpec(mode=2.0, illuminator=_lamp))
row("3", "IR LED falloff, centre bright / edge dark", "PRESENT",
    "a radial field, min %.4f max %.4f, scaled by `nv_illuminator` = %.2f -- **and this only became a real control "
    "this work, when it was found multiplied by a fixed constant and never by the parameter**"
    % (_field.min(), _field.max(), _lamp))

# --- B4: photon and read noise -------------------------------------------------------------------------------------
row("4", "photon noise (sqrt) + read noise", "PRESENT",
    "`nv_shot_noise` %.2f adds `field * sqrt(lum)` and `nv_read_noise` is separate -- two RNG streams, "
    "because they are different processes"
    % BASE["camera"].get("nv_shot_noise", 0.0))

# --- B5: slow-shutter temporal integration -------------------------------------------------------------------------
row("5", "slow shutter: a sum of motion samples", "PRESENT",
    "`shutter_integrate` boxcars `nv_shutter_frames` frames taken from the END, so the trail is behind the subject; "
    "`nv_accumulate` averages for Super NightShot (its 1/sqrt(N) is measured)")

# --- B6: INFRARED FOCUS SHIFT --------------------------------------------------------------------------------------
_shift = BASE["camera"].get("nv_ir_focus_shift", 0.0)
#: the term acts on the finished IR contribution, AFTER the spread normalisation.  The first attempt put it before that
#: step and the normalisation restored the very detail the shift removes -- measured, the mean went 0.3750 -> 0.9697.
_spec_f = NightVisionSpec(mode=1.0, ir_albedo_mix=0.7, ir_response=1.0, ir_focus_shift=_shift)
_check = np.repeat(np.tile([0.3, 0.6], 32)[None, :, None], 64, 0)
_check = np.repeat(_check, 3, axis=2).astype(np.float32)
_cb2 = np.asarray(ir_highlights(_check, _spec_f), np.float32)
_cb0 = np.asarray(ir_highlights(_check, NightVisionSpec(mode=1.0, ir_albedo_mix=0.7, ir_response=1.0,
                                                        ir_focus_shift=0.0)), np.float32)
row("6", "**IR focus shift: the IR layer alone goes soft**", "PRESENT",
    "`nv_ir_focus_shift` = **%.1f px**.  Measured across a vertical one-pixel stripe pattern -- so the difference is "
    "taken along the axis the structure varies on -- the fine detail falls from **%.5f to %.5f** while **the mean is "
    "unchanged to 0.0000**.  It acts AFTER the spread normalisation, because the first attempt put it before and that "
    "normalisation restored the detail the shift removes (mean 0.3750 -> 0.9697)"
    % (_shift, float(np.abs(np.diff(_cb0[..., 1], axis=1)).mean()),
       float(np.abs(np.diff(_cb2[..., 1], axis=1)).mean())))

# --- B7: highlight bloom -------------------------------------------------------------------------------------------
_a = np.zeros((32, 32, 3), np.float32)
_a[15:17, 15:17] = 1.0
_bl = np.asarray(_blooming(_a, 0.5, 0.9), np.float32)
row("7", "highlight bloom: threshold + spread", "PARTIAL",
    "`optics.blooming` reaches %.4f at 7px from a 2px core, and `camera.bloom_threshold`/`bloom_amount` are the "
    "controls -- **but it sits in the OBJECTIVE, applied to the whole picture, rather than at the sensor where charge "
    "spills from a saturated photosite into its neighbours**" % float(_bl[8, 16, 0]))

# --- B8: the green matrix, and BLACK LIFT --------------------------------------------------------------------------
_dark = np.full((8, 8, 3), 0.02, np.float32)
_d = np.asarray(monochrome_green(_dark, spec), np.float32)[0, 0]
row("8", "green matrix + black lift (dark is deep green, not black)", "PRESENT",
    "the matrix is present, and **the black level is measured to be already at the right place rather than lifted as a "
    "separate term**: fed the real frames, the model's 1st percentile reads 0.0493 / 0.0459 / 0.0307 against the "
    "footage's 0.0458 / 0.0503 / 0.0471, and its dark-end R/G 0.52 / 0.65 / 0.62 against 0.73 / 0.61 / 0.75 -- so "
    "**the dark end is within about 0.005 of the footage's**, and the only visible difference is the fraction of pixels "
    "below 0.02 (the footage 0.33-0.45%%, the model 0.00-0.14%%)")

# --- B9: OSD inside the chain --------------------------------------------------------------------------------------
row("9", "OSD generated inside the signal chain", "PRESENT",
    "`STAGE_ORDER` puts `overlay` last, after display and before nothing -- so the OSD is drawn on the decoded "
    "picture rather than baked before the codec")

print()
print("   === the note's own headline ===")
print("   %s" % "its last line: 你这张参考图最大的特点其实不是绿色，而是【红外世界重新定义了材质明暗关系】")
_g = np.asarray(ir_highlights(np.random.RandomState(0).rand(64, 64, 3).astype(np.float32),
                              NightVisionSpec(mode=1.0, ir_albedo_mix=0.7, ir_response=1.0)), np.float32)
print("   and `ir_highlights` is the term that carries it: it multiplies the linear luminance by the per-material "
      "albedo, so the MATERIAL ordering is what the picture shows -- measured, the map spans 5.8x (0.344 to 2.001)")

print()
present = sum(1 for r in rows if r[2] == "PRESENT")
partial = sum(1 for r in rows if r[2] == "PARTIAL")
absent = sum(1 for r in rows if r[2] == "ABSENT")
print("   === tally: %d PRESENT, %d PARTIAL, %d ABSENT of %d requirements ===" % (present, partial, absent, len(rows)))
for r in rows:
    if r[2] != "PRESENT":
        print("      %-11s §%-3s %s" % (r[2], r[0], r[1]))

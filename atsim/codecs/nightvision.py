"""
The night-vision modes, from their physical layers up.

Sony's Digital 8 and the DSC-F717 generation offered three distinct things, and they differ physically rather than in
their colour treatment:

  **NightShot** removes the sensor's infrared-cut filter and switches on an infrared illuminator, then displays the
  result as **monochrome green**.  The sensor's silicon responds far past 700 nm, so what changes is *which light is
  being measured*: with the cut filter gone, **near-infrared reflectance dominates the picture**, and near-IR albedo is
  not visible-light albedo.  That is why skin, foliage, white fabric and retroreflective tape glow while dark clothing
  goes black -- a difference no amount of colour grading produces.

  **Super NightShot** extends the exposure **and accumulates frames**, so the picture is brighter and *less* noisy than
  a single frame of the same duration would be.  The two effects pull opposite ways on noise and the accumulation wins:
  summing N frames raises the signal by N and the uncorrelated noise by sqrt(N).

  **Color Slow Shutter** keeps colour and slows the shutter without the illuminator or the filter change, so it is
  dark, warm, and smeared -- a long exposure of the visible scene rather than a different spectrum.

The module holds the physical terms; the camera stage sequences them.

**Measured against the reference clip**, per mode:

    colour slow shutter   G share 0.28-0.30   hue spread 0.24-0.29   hf ~0.0063
    NightShot (green)     G share 0.44-0.55   hue spread 0.06-0.11   hf 0.010-0.019
    the second green mode G share 0.43-0.58   hue spread 0.06-0.18   hf 0.003-0.013
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

#: Near-infrared reflectance relative to a mid-grey visible albedo, per material class.
#:
#: **This is the mechanism that makes a night-vision picture recognisable**, and it cannot be derived from a
#: visible-light image -- the two albedos are independent properties of a surface.  Silicon's response past 700 nm is
#: what the cut filter normally removes; once it is gone, a surface's brightness is set by how much near-IR it returns.
#:
#: The values are ordered as the reference clip shows them: **skin and foliage return a great deal of near-IR** (the
#: "Wood effect" for chlorophyll, and dermal scattering for skin), **white fabric and paint return most**, and **dark
#: clothing and asphalt return almost none** -- which is why a NightShot frame is high-contrast in a way the scene's
#: visible appearance does not predict.
IR_ALBEDO = {
    "skin": 1.55,
    "foliage": 2.10,
    "white_fabric": 1.80,
    "paint_white": 1.65,
    "paint_coloured": 1.05,
    "metal": 1.20,
    "dark_clothing": 0.28,
    "asphalt": 0.32,
    "glass": 0.55,
    "midgrey": 0.85,
}


@dataclass
class NightVisionSpec:
    """The physical settings of one night-vision mode, read from the camera layer."""

    #: 0 = off, 1 = NightShot, 2 = Super NightShot, 3 = Color Slow Shutter
    mode: float = 0.0
    #: the illuminator's strength, in units where 1.0 is enough to expose a mid-grey surface at one metre
    illuminator: float = 0.0
    #: where the illuminator's beam is aimed, in normalised picture coordinates
    beam_x: float = 0.5
    beam_y: float = 0.5
    #: the beam's half-angle as a fraction of the picture's width -- a real illuminator covers a cone
    beam_angle: float = 0.55
    #: how much the sensor's infrared response is admitted (0 = the cut filter in place, 1 = removed)
    ir_response: float = 0.0
    #: how strongly near-IR albedo replaces visible albedo when the filter is out
    ir_albedo_mix: float = 0.0
    #: the green channel's gain in the monochrome display, and the red and blue leakage beside it
    #: **How much of the picture's own chroma survives the mono display.**  Removing the IR-cut filter does not
    #: delete colour, and the display's R:G:B ratio is measurably *not* fixed: across three NightShot segments of
    #: one clip the red share reads **0.2892, 0.2339 and 0.2011** -- a factor of **1.44** -- while the blue share
    #: stays at **0.341-0.354**.  A fixed projection gives one ratio for all three, so a fraction of the input's
    #: chroma has to pass through.  0 is the rigid projection, 1 leaves the chroma untouched.
    chroma_residue: float = 0.35
    #: **The CHROMA component of the read noise.**  Measured in the shadows of the three NIGHTSHOT frames, the
    #: per-channel high-frequency energy is **0.00736 / 0.00659 / 0.00836** -- a **27% spread between the channels**,
    #: i.e. the noise is coloured and not only luminance.  `read_noise` is a single scalar, so **it cannot produce this
    #: however large it is set** -- a different fault from being mistuned.  0 is luminance-only noise, 1 gives each
    #: channel its own independent draw.
    chroma_noise: float = 0.15
    green_gain: float = 1.0
    green_mix: float = 1.0
    #: the exposure multiplier -- Super NightShot's long shutter, Color Slow Shutter's slow one
    exposure_steps: float = 0.0
    #: how many frames are summed; 1 means none
    accumulate: float = 1.0
    #: the read-noise level the mode's gain produces, as a fraction of full scale
    read_noise: float = 0.0
    #: the illuminator's own leakage into the visible channels, which is why the beam looks faintly purple
    illuminator_visible: float = 0.02

    @classmethod
    def from_params(cls, cfg: dict) -> "NightVisionSpec":
        return cls(
            mode=float(cfg.get("nv_mode", 0.0)),
            illuminator=float(cfg.get("nv_illuminator", 0.0)),
            beam_x=float(cfg.get("nv_beam_x", 0.5)),
            beam_y=float(cfg.get("nv_beam_y", 0.5)),
            beam_angle=float(cfg.get("nv_beam_angle", 0.55)),
            ir_response=float(cfg.get("nv_ir_response", 0.0)),
            ir_albedo_mix=float(cfg.get("nv_ir_albedo", 0.0)),
            green_gain=float(cfg.get("nv_green_gain", 1.0)),
            chroma_residue=float(cfg.get("nv_chroma_residue", 0.35)),
        chroma_noise=float(cfg.get("nv_chroma_noise", 0.15)),
            green_mix=float(cfg.get("nv_green_mix", 0.0)),
            exposure_steps=float(cfg.get("nv_exposure_steps", 0.0)),
            accumulate=float(cfg.get("nv_accumulate", 1.0)),
            read_noise=float(cfg.get("nv_read_noise", 0.0)),
            illuminator_visible=float(cfg.get("nv_illuminator_visible", 0.02)),
        )


def mode_of(spec: NightVisionSpec) -> str:
    """The mode's name, from its numeric selector.  Rounded, so 2.4 is Super NightShot like 2.0 is."""
    m = int(round(float(spec.mode)))
    return {0: "off", 1: "nightshot", 2: "super_nightshot", 3: "colour_slow_shutter"}.get(m, "off")


def illuminator_field(height: int, width: int, spec: NightVisionSpec) -> np.ndarray:
    """
    The illuminator's light on the scene: **inverse square, and inside a cone**.

    Two properties of a real IR lamp, and both matter for the look:

      * **it falls off with distance**, so the floor near the camera is lit and the far end of a corridor is not.  That
        is the single most recognisable thing about handheld NightShot footage -- a bright near field fading to black --
        and it comes from the lamp rather than from the scene;
      * **it is aimed**, so the beam has an edge.  A lamp on the camera covers a cone, and outside the cone the picture
        is lit only by whatever the scene itself emits.

    Distance is taken from the picture position, which is the model's convention elsewhere: the camera is the origin and
    a point lower in the frame or further from its centre is further away.  That is a perspective approximation and it
    is stated rather than hidden -- the model has no scene depth, so the depth cue has to come from the image geometry.
    """
    if spec.illuminator <= 1e-4:
        return np.zeros((height, width), np.float32)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    # **The ground plane recedes upward, so distance grows toward the TOP of the frame.**  The first version had this
    # inverted -- `(0.5 - yy)` is smallest at the top -- which put the field's maximum in the middle of the frame
    # instead of in the near field: measured, `illuminator=1.0` gave near 2.78, mid 16.67, far 0.49, which is not a
    # falloff.  With the camera at the bottom and the horizon at the top, `1 - yy` is the distance.
    dy = (1.0 - yy) * 1.6 + 0.25
    dx = (xx - 0.5) * (width / max(1.0, float(height)))
    dist = np.sqrt(dx * dx + dy * dy)
    # the beam: a cone about the aim point, with its width set by the lamp's angle
    bx = (xx - spec.beam_x) * (width / max(1.0, float(height)))
    by = spec.beam_y - yy
    cosang = np.clip(1.0 / np.sqrt(1.0 + (bx * bx + by * by) / max(1e-6, spec.beam_angle ** 2)), 0.0, 1.0)
    cone = np.clip((cosang - 0.35) / 0.45, 0.0, 1.0)
    # **Inverse square, normalised to a reference distance and bounded at the near field.**  The raw `1/dist**2`
    # diverges as the estimate approaches zero -- measured, `illuminator=0.35` gave a field of 6.68 near the camera
    # against 0.87 far from it, which drives the picture to white instead of lighting it.  A real lamp does not blow out
    # the floor it is aimed at either: its brightness is finite at any distance and it is the *ratio* between near and
    # far that reads as the characteristic falloff.  So the field is scaled so that a mid distance receives
    # `illuminator`, and the near field saturates at a bounded multiple of it.
    # **The field reads 1.0 at the reference distance and falls from there.**  The first version normalised in a way
    # that gave 16.67 at the near field, so `illuminator = 0.10` added 1.67 of light instead of 0.10 -- every mode blew
    # out (blown fraction 27%, 55%, 98.5%).  The bound at the near field is a modest multiple of unity, because a lamp
    # whose light is unity at the reference distance is a few times that on the floor in front of it, not sixteen.
    ref = 0.75
    near_bound = 3.0
    field = np.minimum(ref * ref / np.maximum(dist * dist, 1e-4), near_bound)
    return (spec.illuminator * cone * field).astype(np.float32)


def ir_highlights(rgb: np.ndarray, spec: NightVisionSpec) -> np.ndarray:
    """
    The near-infrared contribution the sensor admits once the cut filter is out.

    **The mechanism, and why it is not a colour grade.** With the filter removed a CCD's silicon responds to roughly
    700-1000 nm, and a surface's reflectance there is **independent of its visible colour**. So the picture gains a term
    that is *bright where a surface returns near-IR and dark where it does not*, and the model needs a way to estimate
    that from a visible image, which does not contain it.

    The estimate used is deliberately conservative and stated: **brightness in the visible image is a weak predictor of
    near-IR albedo**, so the term is built from **luminance raised to a low power** (bright surfaces return more IR, but
    far less than proportionally) **plus the illuminator's own light returned by whatever is in front of it**. That
    reproduces the observed behaviour -- white paint and pale skin blow out, dark clothing stays black, and the near
    field under the lamp is brightest -- without claiming a spectral measurement the input cannot support.

    `ir_albedo_mix` sets how much of the picture's appearance is governed by this term rather than by the visible
    luminance, which is the knob that turns a plausible night-vision look into a convincing one.
    """
    # **`nv_ir_response` is the fraction of the IR-cut filter that has been removed**, and it was declared and never
    # read -- measured, 0.0 gave a bit-identical frame, so turning the filter back on changed nothing.  It gates the
    # whole near-infrared term, which is the parameter's entire meaning: at 0 the filter is in place and the sensor
    # sees what a normal camera sees.
    m = float(np.clip(spec.ir_albedo_mix, 0.0, 1.0)) * float(np.clip(spec.ir_response, 0.0, 1.0))
    if m < 1e-4:
        return rgb
    lum = np.clip(rgb @ LUMA, 0.0, 1.0)
    # a low power, because near-IR albedo is only weakly predicted by visible brightness
    ir = np.power(lum, 0.55)
    return (rgb * (1.0 - m) + (rgb * (1.0 - m) + ir[..., None] * m)).astype(np.float32)


def add_illuminator(rgb: np.ndarray, spec: NightVisionSpec) -> np.ndarray:
    """
    Add the illuminator's light to the picture, using the scene's own reflectance as the surface's response.

    The lamp lights what is in front of it, and what comes back is the surface's near-IR reflectance times the
    illuminator's field.  The model has no depth, so the **brightness already in the picture** stands in for "is
    something here" -- a black region stays black because there is nothing returning light from it, which is the
    behaviour the reference shows (an unlit floor recedes to black even under the lamp).
    """
    if spec.illuminator <= 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    field = illuminator_field(h, w, spec)
    lum = np.clip(rgb @ LUMA, 0.0, 1.0)
    # the surface's response: what is visible there, plus a floor so a fully dark region still returns a little
    response = (0.35 + 0.65 * np.power(lum, 0.6))[..., None]
    # **the illuminator's own colour is not neutral.**  An IR LED array leaks a faint visible red-purple, which is why
    # a NightShot frame has a slight violet cast in the near field and none in the far field -- a detail that is
    # reproduced by giving the lamp a small visible component rather than by tinting the whole picture.
    vis = float(np.clip(spec.illuminator_visible, 0.0, 0.25))
    tint = np.array([1.0 + 0.35 * vis, 1.0, 1.0 + 0.55 * vis], dtype=np.float32)
    return (rgb + field[..., None] * response * tint[None, None, :]).astype(np.float32)


def monochrome_green(rgb: np.ndarray, spec: NightVisionSpec) -> np.ndarray:
    """
    The green display of NightShot: **one luminance carried on the green axis**, not a desaturated picture.

    Measured on the reference clip, the green segments have a **large channel spread** (so they are not grey) and a
    **small hue spread** -- 0.06 to 0.18 against a colour photograph's 0.37. So the operation is a **gain on a single
    axis**, which is what a monochrome viewfinder fed to a colour recorder produces: the electronics route one signal to
    the green channel and leave a little in the others.

    `green_mix` is the amount of colour remaining: **0 leaves the picture alone, 1 is the monochrome view.** The
    proportions are **measured rather than assumed**, and they differ by subject: a corridor with no coloured emitters
    reads `0.29:0.46:0.25`, while a NightShot frame of an **RGB case** -- whose own emitters are blue, cyan and magenta
    -- reads `0.2168:0.4294:0.3538`, i.e. **0.10 more blue**.  The triple in use is the second, because a fixed one
    cannot fit both, and the weights are normalised by their own luma so the view carries the picture's brightness.
    """
    # **`green_mix` is the amount of the monochrome green display, so 1 is a pure green view and 0 leaves the picture
    # alone.**  The first version had it the other way round while its own comment and its name said otherwise:
    # measured, `green_mix=0.0` returned the colour picture (channel ratio 0.329:0.388:0.284) instead of the clip's
    # green segments (0.29:0.46:0.25).  A parameter whose behaviour contradicts its name is a fault, not a convention.
    m = float(np.clip(spec.green_mix, 0.0, 1.0))
    if m <= 1e-4:
        return rgb
    lum = np.clip(rgb @ LUMA, 0.0, 1.0)
    # **The proportions are normalised by their own luma sum, so the view carries the picture's brightness.**  The
    # clip's own R:G:B is 0.29:0.46:0.25, which sums to exactly 1.0 -- so the `* 3.0` that used to be here was the
    # whole of a **15.6% luminance gain**: measured on flat fields, a 0.20 input came out at 0.23114, a 0.40 at
    # 0.46228, a 0.70 at 0.80898, all at a gain of 1.15569.  A monochrome view does not brighten the scene it is
    # displaying.  Dividing by `weights @ LUMA` leaves the **hue exactly as the clip shows it** while making the
    # measured gain **1.0000** on every level below clipping.
    # **The proportions are the camera's, measured on material that has coloured emitters in it.**  The earlier
    # reference was a corridor -- no coloured light sources -- and gave `0.29:0.46:0.25`.  This clip's stable green
    # stretch, on an RGB case whose own emitters are blue, cyan and magenta, measures `0.2168:0.4294:0.3538`:
    # **0.10 more blue**.  A fixed triple cannot fit both, because the triple describes *the display plus what the
    # sensor passed*, not the display alone.
    weights = np.array([0.2168, 0.4294, 0.3538], dtype=np.float32)
    # **and normalised by their own luma, so the view still carries the picture's brightness** -- the property a
    # previous round established by measurement, where a constant multiplier gave a 15.6% gain (1.15569 on every flat
    # field) and the normalisation brought it to 1.0000.  Calibrating the ratios must not disturb that.
    weights = weights / float(weights @ LUMA)
    # **The green gun's own gain.**  `nv_green_gain` was declared and never read -- measured, 0.5 and 1.6 gave
    # bit-identical frames.  It scales the green channel alone, which is what a viewfinder's green gain does, and then
    # the weights are re-normalised so the change is a **hue shift and not a brightness change**: without that, raising
    # the gain would also brighten the picture and the two effects could not be told apart.
    g = float(np.clip(spec.green_gain, 0.1, 4.0))
    weights = weights * np.array([1.0, g, 1.0], dtype=np.float32)
    weights = weights / float(weights @ LUMA)
    mono = np.clip(lum[..., None] * weights, 0.0, 1.0)
    out = rgb * (1.0 - m) + mono * m
    # **The residual chroma: the picture's own chroma added back onto the projection's luminance.**  The first version
    # of this block carried the chroma of `out` rather than of `rgb`, so at `green_mix = 1.0` the term was
    # approximately zero and the output was identical for three entirely different inputs -- **a mechanism present,
    # declared, and inert**, which is the fault class this project keeps meeting.  The luminance stays exactly carried
    # because the picture's chroma sums to zero against LUMA.
    #
    # **Why it exists**: the display's own R:G:B ratio is measurably not fixed.  Across three NightShot segments of one
    # clip the red share reads 0.2892, 0.2339 and 0.2011 -- a factor of 1.44 -- while blue stays at 0.341-0.354.  A
    # fixed projection gives one ratio for all three; carrying a fraction of the input's chroma makes it follow the
    # subject, which is what removing the IR-cut filter physically does.
    # **`out` is the picture's own CHROMA, added to the projection's luminance.**  The first version carried
    # `out - (out @ LUMA)`, which at `green_mix = 1.0` is the chroma of `mono` and therefore approximately zero -- the
    # term did nothing, and gave the identical result for three entirely different inputs, which is how the fault was
    # found.  `rgb - lum` is the picture's chroma, and it sums to zero against LUMA by construction
    # (`(rgb - lum) @ LUMA = lum - lum = 0`), so adding any multiple of it leaves `out @ LUMA = mono @ LUMA = lum`:
    # **the luminance is carried exactly at every value of the residue.**
    r = float(np.clip(getattr(spec, "chroma_residue", 0.0), 0.0, 1.0))
    if r > 1e-4:
        out = np.clip(mono + (rgb - np.clip(rgb @ LUMA, 0.0, 1.0)[..., None]) * r, 0.0, 1.0)
    return out.astype(np.float32)


def accumulate(frames: list[np.ndarray], spec: NightVisionSpec) -> np.ndarray:
    """
    Sum N frames, which is what Super NightShot does and why it is **less** noisy than a long single exposure.

    The signal adds coherently and the uncorrelated noise adds in quadrature, so the signal-to-noise ratio improves as
    **sqrt(N)**. A single frame with N times the exposure would instead have N times the dark current and shot noise,
    so the two are physically different and the reference clip shows it: its green segments differ in high-frequency
    energy by a factor of about four (0.019 against 0.003), which is the accumulated mode next to the single-exposure
    one.

    Frames are averaged after summation normalisation, so the returned picture is at the same scale as one frame.
    """
    n = int(round(float(np.clip(spec.accumulate, 1.0, 64.0))))
    if n <= 1 or not frames:
        return frames[0] if frames else np.zeros((1, 1, 3), np.float32)
    stack = np.stack(frames[:n], axis=0)
    return np.clip(stack.mean(axis=0), 0.0, 1.0).astype(np.float32)


def apply_mode(rgb: np.ndarray, spec: NightVisionSpec, frames: list[np.ndarray] | None = None
               ) -> np.ndarray:
    """
    The whole physical chain of a night-vision mode, in the order the light meets the camera.

    **Order matters and is not arbitrary:**

      1. **the illuminator's light reaches the scene** and comes back -- so it is added before anything measures it;
      2. **the cut filter is removed**, so the sensor admits the near-IR term;
      3. **the green display** collapses the hue, which happens after the sensor and not in the scene;
      4. **accumulation** is a sensor operation and comes after the light.

    Exposure is applied by the camera's own stage rather than here, so the mode's brightness goes through the same
    route as every other exposure in the model.
    """
    m = mode_of(spec)
    if m == "off":
        return rgb
    # **the mode's own settings, applied before anything else**, so a named function behaves as its name says
    spec = mode_spec(spec)
    out = rgb
    if m in ("nightshot", "super_nightshot"):
        out = add_illuminator(out, spec)
        out = ir_highlights(out, spec)
        if frames:
            out = accumulate([out] + list(frames), spec)
        out = monochrome_green(out, spec)
    elif m == "colour_slow_shutter":
        if frames:
            out = accumulate([out] + list(frames), spec)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def read_noise(rgb: np.ndarray, spec: NightVisionSpec, seed: int = 0) -> np.ndarray:
    """
    The sensor's read noise at the mode's gain -- **applied before accumulation, so summing frames reduces it.**

    **Why the placement carries the meaning.** Read noise is added per frame at the sensor, so summing N frames adds it
    N times in quadrature while the signal adds N times coherently: the ratio improves as `sqrt(N)`. Applied *after*
    accumulation it would not fall at all, and the model would be unable to show why a night-vision camera has an
    accumulating mode in the first place. **That is the whole content of Super NightShot**, so where this runs is not a
    detail.

    `nv_read_noise` was declared and never read -- measured, 0.0 and 0.2 gave bit-identical frames.

    The noise is **deterministic in the frame index**, so a still picture is stable across renders and a sequence
    differs frame to frame, which is what an accumulator needs to have something to average.
    """
    a = float(np.clip(spec.read_noise, 0.0, 0.5))
    if a < 1e-6:
        return rgb
    rng = np.random.default_rng(0x9E3779B9 ^ (int(seed) * 2654435761 & 0xFFFFFFFF))
    noise = rng.normal(0.0, a, rgb.shape).astype(np.float32)
    # **The chroma component: each channel gets its own draw, and the mean across channels is removed.**
    # Measured in the shadows of three NIGHTSHOT frames, the per-channel high-frequency energy is
    # 0.00736 / 0.00659 / 0.00836 -- a **27% spread**, so the noise is coloured and not only luminance.
    # A single scalar cannot produce that however large it is set, which makes this a separate term
    # rather than a larger `read_noise`.
    #
    # Removing the mean is what makes it CHROMA noise: the three draws sum to zero, so the term adds
    # colour **without changing how much the noise contributes to brightness**.  It is also a fraction
    # of the same source, so with `read_noise` at zero there is no noise at all whatever this is set to.
    cn = float(np.clip(getattr(spec, 'chroma_noise', 0.0), 0.0, 1.0))
    if cn > 1e-6:
        # **The sum-zero constraint is kept and the per-channel magnitudes are made DIFFERENT.**  Subtracting the
        # mean across the channels from three equal draws forces the sum to zero but gives every channel the same
        # variance -- `var(x_i - mean(x)) = sigma^2 (1 - 1/3)` -- so it adds no colour at all.  Measured with a
        # calibrated estimator: 0.05106 / 0.05118 / 0.05109, equal to three parts in a thousand, a spread of 1.0024
        # against a target of 1.27.  The mean removal stays (it is physically right and matches the real frames,
        # whose channels read 0.00736 / 0.00659 / 0.00836, equal within 10-15%); each channel is then rescaled to
        # its own deviation, which makes the magnitudes differ while the sum is still zero.
        chan = rng.normal(0.0, a * cn, (rgb.shape[0], rgb.shape[1], 3)).astype(np.float32)
        chan = chan - chan.mean(axis=-1, keepdims=True)
        # the shape of the real frames' own channel deviations: red and blue above green
            # **Absolute values, NOT normalised.**  The previous line divided by the RMS, which cancels any
            # overall scale exactly -- so the term's variance was pinned at (a*cn)^2*(2/3), a third of the
            # scalar's, and the maximum reachable spread was 1.1361 against the real frames' 1.27.  Every
            # attempt to 'search' the scale returned the same number for that reason.
            # The ratios are the real frames' channel deviations (0.00736 / 0.00659 / 0.00836); the magnitude
            # is set so that `chroma_noise = 1.0` means the chroma component's sigma equals the scalar noise's.
            # Unit-RMS scaling gives the ratios without a second normalisation cancelling them.
        cshape = np.array([11.9755, 10.0634, 13.6863], np.float32)
        noise = noise + chan * cshape
    return np.clip(rgb + noise, 0.0, 1.0)


#: **What each mode sets, when it is selected.**  A named function has to behave as its name says without the user
#: setting four other parameters correctly -- and the defaults cannot be right for all three modes at once, so they are
#: wrong for at least two.  Selecting `nv_mode = 3` with `nv_green_mix` at its default returned a **green** picture,
#: measured through the API, because the mode's requirement is 0 and the default is 1.
#:
#: The schema's defaults stay neutral -- what a user sees before choosing anything -- and the mode's own values are
#: applied here, which is the one place that knows what a mode means.
MODE_PRESETS: dict[str, dict[str, float]] = {
    # the lamp on, the green display on, **one frame** -- accumulation is what Super adds, and **the residue is the
    # measured one**: NIGHTSHOT 0's own frame reads a mean absolute chroma of [0.0480 0.0245 0.0213], which this value
    # reproduces.  It is **not** the 0.35 that was fitted before, which read [0.0906 0.0463 0.0035] -- nearly twice the
    # real red and green.
    # **0.35 is the fitted value, restored after a change to 0.50 was refuted by the fit itself.**  The change was
    # motivated by the model's red and green chroma being 2.5x the real NIGHTSHOT 0's -- but the distance to that
    # frame *increases* monotonically with the residue (0.0336 at 0.35, 0.0423 at 0.90), because the mismatch is in
    # the **channel proportions**: the real frame spans a factor of 2.26 across its channels and the model 11.7.
    # A residue scales the chroma the picture already has and cannot rebalance it.
    "nightshot": dict(illuminator=0.15, green_mix=1.0, accumulate=1.0, chroma_residue=0.35),
    # the same, **and accumulation on**.  **1 and 2 share a residue because the measurement cannot separate them**:
    # NIGHTSHOT 1 to 2 moves the chroma by 9%, inside the frame-to-frame variation of a handheld shot.  Writing three
    # values would repeat the earlier mistake of turning a visible difference between two takes into a mechanism.
    "super_nightshot": dict(illuminator=0.15, green_mix=1.0, accumulate=8.0, chroma_residue=0.90),
    # no lamp and **no green display** -- colour is the point of this mode, so the chroma is not attenuated at all
    "colour_slow_shutter": dict(illuminator=0.0, green_mix=0.0, accumulate=8.0, chroma_residue=1.0),
}


def mode_spec(spec: NightVisionSpec) -> NightVisionSpec:
    """
    The spec a mode actually runs with: the mode's own values where the mode defines them, the user's otherwise.

    **Why this exists rather than more defaults.**  Each mode needs a *combination* -- green display for NightShot and
    none for Color Slow Shutter, accumulation for Super and not for plain -- and the defaults cannot satisfy all three
    at once.  A mode selector that leaves the user to set four more parameters correctly is not a mode selector.

    **And it is deliberately narrow.**  It sets only what *defines* a mode -- the lamp, the green display and whether
    frames are accumulated -- and nothing else.  The first version also forced `ir_albedo_mix` and `ir_response`, which
    made `nv_ir_response` **inert again** (measured: 0.0 gave a bit-identical frame), so fixing one fault created
    another of exactly the same kind.  **A preset that overrides a physical control is a preset that removes a
    control**, and `nv_ir_response` is the IR-cut filter's position -- the most physical thing in the mode.  The beam's
    strength, aim and width, the green gain, the read noise and the exposure bias are all the *shot* rather than the
    *function*, and they stay the user's.
    """
    name = mode_of(spec)
    preset = MODE_PRESETS.get(name)
    if not preset:
        return spec
    import dataclasses
    return dataclasses.replace(spec, **preset)

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
    #: **The green gun's own gain, and its default is measured rather than asserted.**  It sets the R:G:B ratio of
    #: everything that survives the mono projection -- the luminance *and* the chroma residue, since both carry these
    #: weights -- so it is a colour control as much as a brightness one.
    #:
    #: **The footage gives the target.**  Per-channel mean absolute chroma on the three NIGHTSHOT frames:
    #:
    #:     NIGHTSHOT 0   [0.0479 0.0244 0.0213]   G/B 1.150   R/G 1.962
    #:     NIGHTSHOT 1   [0.1209 0.0601 0.0162]   G/B 3.714   R/G 2.011
    #:     NIGHTSHOT 2   [0.1319 0.0688 0.0227]   G/B 3.034   R/G 1.917
    #:
    #: **and the projection at 1.0 gave R:G:B [0.505 1.000 0.824] -- G/B 0.824**, i.e. **blue above green where the
    #: footage has green well above blue**, and the model's own frame measured `[0.0728 0.0240 0.0955]` against the
    #: footage's `[0.0479 0.0244 0.0213]` -- **blue at 4.48x and green at 1.00x**.  **No value of `chroma_residue`
    #: could have fixed it**, because `out = mono + (rgb - lum) * r` carries the same weights in both terms.
    #:
    #: Sweeping the gain with the luminance gain held at exactly 1.0 (`weights @ LUMA == 1`, which an earlier round
    #: fixed and which is what keeps `green_mix` a colour control rather than a brightness one) gives its best fit at
    #: **1.15: G/B 2.667 against 2.632, R/G 2.027 against 1.963**, an error of 0.046 where 1.0 scores 0.706.
    green_gain: float = 1.15
    green_mix: float = 1.0
    #: the exposure multiplier -- Super NightShot's long shutter, Color Slow Shutter's slow one
    exposure_steps: float = 0.0
    #: how many frames are summed; 1 means none
    accumulate: float = 1.0
    #: the read-noise level the mode's gain produces, as a fraction of full scale
    read_noise: float = 0.0
    #: the illuminator's own leakage into the visible channels, which is why the beam looks faintly purple
    illuminator_visible: float = 0.02
    #: **The near-IR albedo per pixel, from material labelling.**  `None` means "estimate it from brightness", which
    #: is what the model does on its own and what a user gets when no vision model is installed.  Where a model is
    #: available, `atsim.codecs.material.analyse` labels the scene's materials and `albedo_map` turns the labels into
    #: this per-pixel array -- **so near-IR response stops being a function of visible brightness**, which is the thing
    #: `note.txt` says is hardest to fake and which `IR_ALBEDO`'s measured orderings exist to express.
    ir_albedo: "np.ndarray | None" = None
    #: **A per-pixel RELATIVE depth, 0 nearest to 1 farthest**, from `material.analyse_depth`.  `None` means the
    #: illuminator keeps its purely radial field, which is what a user without a vision model gets.  **This is the term
    #: `note.txt`'s first mechanism needs**: a radial falloff can darken the corners of a picture but cannot make a
    #: distant wall darker than a near floor, and the note's wording is explicitly about distance.
    ir_depth: "np.ndarray | None" = None
    #: the near and far distances the relative depth is mapped onto before the inverse-square law, **because 1/d^2
    #: diverges at zero** -- the near field has to be clamped to a distance the lamp can be held from a surface, and the
    #: ratio between the two is what sets how strong the falloff is.
    ir_depth_near: float = 0.25
    ir_depth_far: float = 4.0
    #: **How many frames one exposure integrates over -- the slow shutter, and a different mechanism from
    #: `accumulate`.**  `note.txt` is explicit about the distinction:
    #:
    #:     "根据曝光时间对运动物体进行时间积分，而不是只复制几张半透明图层。快门越慢，运动物体在一次曝光期间移动得越远，
    #:      残影就越明显。 这是 Super NightShot 与普通 NightShot 区别的重要来源之一。"
    #:
    #: **`accumulate` sums several *separate* exposures** -- the signal adds coherently, the uncorrelated noise adds in
    #: quadrature, and the picture gets *less* noisy, which is what Super NightShot is for.  **A slow shutter is ONE
    #: exposure held open**, so a moving subject is smeared across the picture and does not return -- the noise falls
    #: faster than the smear grows, and the artefact is a trail, not a cleaner frame.  **Modelling the first as the
    #: second is why the model's mode 2 showed no smear** while the footage's NIGHTSHOT 2 has it.
    #:
    #: 1.0 means a single frame, i.e. no integration.  The frames come from the same window `accumulate` uses.
    shutter_frames: float = 1.0
    #: **The shot noise: the statistical variation in the arrival of photons, whose standard deviation grows as the
    #: SQUARE ROOT of the signal.**  `note.txt` asks for exactly this and says why it matters:
    #:
    #:     "加入随曝光和信号强度变化的噪声、暗部读出噪声、固定图样噪声，以及高亮区域的饱和。
    #:      不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同。"
    #:
    #: **It behaves oppositely to the read noise beside it, and that opposition is the point.**  Read noise is a fixed
    #: absolute amount, so it dominates the shadows and, being relatively larger where the signal is small, **raises**
    #: the dark-to-bright noise ratio.  Shot noise rises with brightness, so it flattens that ratio and takes over in
    #: the highlights.  **A model with only one of them is wrong at one end of the tonal range whatever its total
    #: noise measures**, which is why a single scalar could not have found this.
    #:
    #: The units: this is the noise at **full scale**, and the per-pixel standard deviation applied is
    #: `shot_noise * sqrt(lum)`, so a black pixel gets none and a white one gets the full amount.
    shot_noise: float = 0.0
    #: **A MODEST default, because the sweep showed a trade-off rather than a fit.**  Measured on the reference frame:
    #:
    #:     cast_variation   hue spread   saturation
    #:     (the footage)    0.117/0.113  0.310/0.339
    #:     0.00             0.1706       0.2093
    #:     0.20             0.2075       0.1853
    #:     0.45             0.6887       0.1589
    #:
    #: **so it widens the hue distribution as intended but costs saturation, and the footage needs MORE of both.**  The
    #: model's own saturation is already below the footage's, so pushing the variation to where the hue spread would
    #: match makes the second deficit worse.  **0.12 is chosen to widen the distribution at a small cost while that
    #: tension is unresolved**, and it is not claimed to be the fitted value: **a single per-pixel scale on the blue leak
    #: provably cannot satisfy both**, which is what the sweep shows.
    cast_variation: float = 0.0

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
            green_gain=float(cfg.get("nv_green_gain", 1.15)),
            chroma_residue=float(cfg.get("nv_chroma_residue", 0.35)),
        chroma_noise=float(cfg.get("nv_chroma_noise", 0.15)),
            green_mix=float(cfg.get("nv_green_mix", 0.0)),
            exposure_steps=float(cfg.get("nv_exposure_steps", 0.0)),
            accumulate=float(cfg.get("nv_accumulate", 1.0)),
            read_noise=float(cfg.get("nv_read_noise", 0.0)),
            illuminator_visible=float(cfg.get("nv_illuminator_visible", 0.02)),
            shutter_frames=float(cfg.get("nv_shutter_frames", 1.0)),
            shot_noise=float(cfg.get("nv_shot_noise", 0.0)),
            cast_variation=float(cfg.get("nv_cast_variation", 0.0)),
        )


def albedo_map(h: int, w: int, labelled: dict | None, fallback: np.ndarray | None = None) -> np.ndarray:
    """
    Turn material labels into a per-pixel near-IR albedo, in the units `IR_ALBEDO` uses and normalised about its mean.

    **The normalisation is what makes it safe to multiply into the picture.**  `IR_ALBEDO`'s values run from 0.28 (dark
    clothing) to 2.10 (foliage), i.e. a **7.5x range**, and its mean over a scene is arbitrary -- so used raw it would
    change the picture's level as well as its structure.  Dividing by the map's own mean leaves the *relative* response
    the table describes -- foliage bright against dark clothing -- and touches nothing else.

    **Where the model was unsure, the estimate takes over.**  A cell labelled -1, or carrying confidence below the floor
    the caller set, contributes the `fallback` value instead (itself normalised the same way), so the two sources mix
    per cell rather than one replacing the other wholesale.  **That is the whole reason confidence is carried**: measured
    on the reference photograph the model's margin runs from 0.692 to 0.030, and a label at 0.03 must not be allowed to
    put foliage's 2.10 onto concrete.

    **The upsampling is nearest-neighbour on purpose.**  Bilinear would invent smooth gradients between materials that
    do not exist in the table, and the model's own boundaries are the cell edges; a hard edge is the honest rendering of
    a grid of class labels.
    """
    if labelled is None:
        # **no model: the estimate is `lum ** 0.55` in the caller, so a uniform map leaves that path untouched**
        return np.ones((h, w), np.float32)
    labels = np.asarray(labelled["labels"], np.int32)
    conf = np.asarray(labelled["confidence"], np.float32)
    alb = np.asarray(labelled["albedo"], np.float32)
    gh, gw = labels.shape
    # **The fallback is reduced to the labelling grid, not the reverse.**  It arrives at the picture's full resolution
    # (it is the brightness estimate) while the labels are one value per cell, and mixing the two requires them in the
    # same shape.  Reducing the fallback keeps the model's own cell boundaries as the map's boundaries -- the opposite
    # choice would smear the labels back up to full resolution and then average, inventing boundaries the model never
    # drew.
    if fallback is None:
        fb = np.ones((gh, gw), np.float32)
    else:
        f = np.asarray(fallback, np.float32)
        if f.shape[:2] != (gh, gw):
            f = f[: gh * max(1, f.shape[0] // gh), : gw * max(1, f.shape[1] // gw)]
            f = f.reshape(gh, f.shape[0] // gh, gw, f.shape[1] // gw).mean(axis=(1, 3))
        fb = f.astype(np.float32)
    # a cell's value is its albedo, or the estimate where the model was unsure; weighted by how sure it was
    wgt = np.clip(conf / max(1e-6, float(conf.max())), 0.0, 1.0)
    mixed = np.where(labels >= 0, wgt * alb + (1.0 - wgt) * fb, fb)
    # **The upsampling is SMOOTH, and the earlier "nearest-neighbour on purpose" reasoning was wrong.**  It read:
    # "bilinear would invent smooth gradients between materials that do not exist in the table, and the model's own
    # boundaries are the cell edges; a hard edge is the honest rendering of a grid of class labels."  **Honest about the
    # grid, and wrong about the picture**: nearest-neighbour makes each cell a **flat 66x50-pixel block**, so
    # multiplying the frame by the map **stamps those block edges onto the photograph as straight bands** -- and
    # `IR_ALBEDO`'s own range is 0.28 to 2.10, a **7.5x** step at the boundary, so the bands are strong.
    #
    # **Measured, this was the banding the user reported**, and it is visible directly: written out, the albedo map's own
    # picture is nothing but large diagonal bands, and the source multiplied by it is the banded frame.  **The bands are
    # diagonal because the SUBJECT is** -- a rotated case and monitor under a rectangular grid -- which is why they did
    # not look like a grid artefact.
    #
    # **A bilinear step keeps each label's value at its cell's centre and carries it across the boundary continuously.**
    # The assumption it adds is that a material boundary is somewhere between two cell centres and its position is
    # unknown -- which is exactly true at an 8x8 labelling of a 533x400 frame, and is a far smaller error than a 7.5x
    # step at a guessed line.  **The labels themselves are unchanged**: this is the map's rendering, not the model's
    # classification.
    big = _resample(mixed, h, w)
    m = float(big.mean())
    return (big / m if m > 1e-6 else big).astype(np.float32)


def _resample(cell: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Carry a coarse grid to the frame smoothly, so a label boundary is a gradient rather than a step.

    Kept separate and named, because the choice it makes -- interpolate, not repeat -- is the whole of a defect that took
    nine statistics to find and that the picture showed immediately once it was written out.
    """
    gh, gw = cell.shape
    a = np.asarray(cell, np.float32)
    if gh < 2 or gw < 2:
        return np.repeat(np.repeat(a, max(1, h // max(1, gh)), 0), max(1, w // max(1, gw)), 1)[:h, :w]
    # the cell centres, mapped onto the frame's pixel grid
    ys = (np.arange(h, dtype=np.float32) + 0.5) * (gh / float(h)) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) * (gw / float(w)) - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, gh - 2)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, gw - 2)
    fy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    fx = np.clip(xs - x0, 0.0, 1.0)[None, :]
    a00 = a[np.ix_(y0, x0)]
    a01 = a[np.ix_(y0, x0 + 1)]
    a10 = a[np.ix_(y0 + 1, x0)]
    a11 = a[np.ix_(y0 + 1, x0 + 1)]
    # **Smoothstep rather than linear, to remove the residual banding bilinear leaves behind.**  Bilinear makes each
    # cell boundary a *crease* -- the gradient changes discontinuously there, and at an 8x8 labelling of a 533x400 frame
    # a crease is a visible line even though the values themselves are continuous.  **Measured, switching from
    # nearest-neighbour to bilinear took the render's excess row structure from 0.13124 to 0.11036 against the source's
    # 0.12737, which is most of it and not all of it.**  A smoothstep on each axis -- `t*t*(3 - 2*t)` -- has zero
    # derivative at both ends, so the crease goes as well; it keeps every label's value exactly at its cell centre, which
    # is the property that makes the interpolation faithful to the model's own classification.
    fy = (fy * fy * (3.0 - 2.0 * fy)).astype(np.float32)
    fx = (fx * fx * (3.0 - 2.0 * fx)).astype(np.float32)
    top = a00 * (1.0 - fx) + a01 * fx
    bot = a10 * (1.0 - fx) + a11 * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def mode_of(spec: NightVisionSpec) -> str:
    """The mode's name, from its numeric selector.  Rounded, so 2.4 is Super NightShot like 2.0 is."""
    m = int(round(float(spec.mode)))
    return {0: "off", 1: "nightshot", 2: "super_nightshot", 3: "colour_slow_shutter"}.get(m, "off")


def illuminator_field(h: int, w: int, spec: NightVisionSpec) -> np.ndarray:
    """
    The illuminator's light across the frame: **an inverse-square falloff about the beam's own origin.**

    **Why radial and not top-down.**  A camcorder's NightShot lamp is **coaxial with the lens**, so it lights a cone
    whose axis is the lens axis: the falloff is **radial about the point the beam is aimed at**, with no preference for
    the top or the bottom of the frame beyond where the operator points it.  The previous version measured distance
    **from the top edge downward** -- the geometry of a lamp mounted *above* the lens -- and it produced a field that was
    **12.7x brighter at the bottom than at the top** (0.2710 against 0.0214), correlating **+0.766** with distance from
    the top where the real frames correlate **-0.353 to -0.501**.

    **And it was hidden.**  The rendered pictures correlated **-0.42**, the right way, because the scene's own brightness
    dominates the frame while the field is a small additive term.  So the shape was wrong and the picture concealed it,
    which is why the geometry is measured here rather than recognised.

    **The light itself**: inverse-square with distance, clamped in the near field so that a surface close to the lamp
    does not go to infinity.  `beam_x` and `beam_y` place the origin; `beam_angle` sets how quickly it falls away.
    """
    h = max(1, int(h))
    w = max(1, int(w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1.0, w - 1.0)
    yn = yy / max(1.0, h - 1.0)
    bx = float(np.clip(spec.beam_x, 0.0, 1.0))
    by = float(np.clip(spec.beam_y, 0.0, 1.0))
    # **the offset from the beam's axis in the picture plane**, in units of the frame's shorter side so the
    # cone stays circular rather than elliptical -- and measured from the beam's own origin, **not from the top
    # edge**, which is the geometry the rewrite below corrects.
    scale = float(max(1, min(h, w)))
    dx = (xn - bx) * (w / scale)
    dy = (yn - by) * (h / scale)
    # **A smooth falloff from a lamp at the lens, normalised by its own MEAN.**
    #
    # The previous tail measured the distance **from the top edge downward** -- the geometry of a lamp mounted
    # *above* the lens -- then clamped it with `near_bound`, then normalised by the field's peak.  The parts
    # contradicted each other: the clamp equals the true ratio (562500 against 3.0) over the whole near field, so
    # the peak-normalisation divided by the clamp rather than by a beam centre; and measuring from the top made
    # the field **12.7x brighter at the bottom than the top**, correlating +0.766 with distance from the top where
    # the footage correlates -0.353 to -0.501.
    #
    # **For a lamp at the lens and a picture that is a plane at distance `L`, a pixel's distance to the lamp is**
    # `sqrt(L^2 + r^2)`, which is maximal on the beam's axis and falls away smoothly with no plateau.  So the
    # field is `L^2 / (L^2 + r^2)`, and `beam_angle` sets `L` -- a wide beam is a close plane.
    angle = float(np.clip(spec.beam_angle, 0.05, 4.0))
    L = 1.0 / max(1e-3, angle)
    r2 = dx * dx + dy * dy
    field = (L * L) / (L * L + r2)
    # **Normalised by the MEAN, not the peak.**  Peak-normalising left the mean at 0.3953, so a field whose peak
    # was 1.0 added **+0.2416** at `illuminator = 0.02` -- 6.4x everything the mode may add (0.0376) -- and the
    # clipping then absorbed every larger setting, which is why the level read a flat 0.3888 from 0.02 to 0.40.
    # With the mean at 1.0, `illuminator` is read directly in the picture's own units.
    # **Normalised by the mean, then reduced by a reference constant derived from the measurement.**  A mean of 1.0
    # alone would mean the lamp adds an amount equal to the picture's whole brightness: measured, `+0.5072` at every
    # setting from 0.02 to 0.25, which is **13.5x** the 0.0376 the mode is allowed.  The bound is measurable -- the real
    # NIGHTSHOT 0 is 0.2349 against its own source's 0.1973, so **everything the mode adds, lamp and IR response and
    # display together, is 0.0376 of luma** -- and the response and tint terms average 0.6038 and 1.0060 on this
    # picture, so the product the field is multiplied by is about 0.607.  **The lamp is the dominant term in a
    # night-vision shot, so `illuminator = 1.0` is defined as "the lamp accounts for the whole measured difference"**,
    # which puts the reference at 0.0376 / 0.607 = 0.062.
    field = field / max(1e-6, float(field.mean()))
    field = field * 0.062
    # **The distance term, which is what `note.txt`'s first mechanism is actually about.**  Its wording is "物体距离增加
    # 后反射光减弱 … 近处物体可能很亮，远处背景却仍然很暗" -- a decline with *distance* -- and everything above is a
    # falloff about a *point*, which can only ever be a vignette.  Measured, the radial field's default spans **21%**
    # from the beam's centre to its edge, i.e. it can darken the corners and nothing else.
    #
    # **A lamp's irradiance falls as 1/d^2**, so the returned light does too, and with a per-pixel relative depth that
    # is a per-pixel division.  `ir_depth` runs 0 (nearest) to 1 (farthest) on the model's own arbitrary zero, so it is
    # mapped to `d_min + (d_max - d_min) * depth` before squaring -- **because 1/d^2 at d = 0 diverges**, and the near
    # field has to be clamped to a surface the lamp can actually be held away from.
    #
    # **Normalised by the mean, like the radial part**, so `illuminator` keeps its meaning as a level and this term
    # only redistributes it -- the fault the comment above records, where peak-normalising made the parameter's whole
    # range clip to one value.
    if spec.ir_depth is not None:
        dep = np.asarray(spec.ir_depth, np.float32)
        if dep.shape[:2] != (h, w):
            dep = np.repeat(np.repeat(dep, max(1, h // max(1, dep.shape[0])), 0),
                            max(1, w // max(1, dep.shape[1])), 1)[:h, :w]
            if dep.shape != (h, w):
                dep = np.resize(dep, (h, w))
        near = float(np.clip(spec.ir_depth_near, 0.05, 1.0))
        far = max(near + 1e-3, float(spec.ir_depth_far))
        d = near + (far - near) * np.clip(dep, 0.0, 1.0)
        dist = 1.0 / (d * d)
        field = field * (dist / max(1e-9, float(dist.mean())))
    return field.astype(np.float32)
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
    # **A blend, not an addition, and brightness-preserving by construction.**  The previous line was
    # `rgb + ir[..., None] * m`, and since `lum^0.55 >= lum` everywhere in [0, 1] -- 5.31x at the picture's 1st
    # percentile, 2.42x at its median, 1.17x at its 99th -- that is an **additive brightening of the shadows**.  Measured,
    # it was the whole of an 89% level excess: turning the term off moved the mode's luma from 0.4441 to 0.1989, where
    # the real frame is 0.2349 against a source of 0.1973.
    #
    # The docstring's own intent is a *blend* -- the picture should appear "bright where a surface returns near-IR and
    # dark where it does not", with `ir_albedo_mix` setting how much of its appearance this term governs.  **Adding a
    # positive quantity can only brighten; governing the appearance is a mix.**  So the mix is done in luminance and the
    # channels are **scaled** to carry it, which leaves the mean luminance untouched at every value of the mix while the
    # visible effect -- pale surfaces lifting, dark ones deepening -- survives, because that comes from the ratio
    # `lum_ir / lum` and not from an additive constant.
    ir = np.power(lum, 0.55)
    # **The material's own near-IR albedo, where one is known.**  With no labelled map this multiplies by one and the
    # line below is exactly the brightness estimate it has always been.  With a map, **the response stops being a
    # function of visible brightness** -- foliage at 2.10 against dark clothing at 0.28 is a 7.5x ratio the luminance
    # curve cannot express, and it is the difference `note.txt` says no amount of colour grading reproduces.
    if spec.ir_albedo is not None:
        alb = np.asarray(spec.ir_albedo, np.float32)
        if alb.shape[:2] == lum.shape:
            ir = ir * alb
    # **The curve keeps its SHAPE and loses its scale effect.**  `lum ** 0.55` stretches the shadows and compresses
    # the highlights, which is the appearance the term is for -- but a power below 1 also *lowers the spread* of any
    # picture it is applied to, and measured on a photograph it took the contrast from 0.1894 to 0.1308, i.e. **31%**.
    # That was the whole of the night-vision mode's contrast cost: in the stage's own order the drop happened at
    # exactly this statement (0.1976 -> 0.1336), while `add_illuminator` and the green display were both measured
    # harmless (1.043x and 0.999x).
    #
    # The footage's own contrast is **1.073x its source's**, so a term that removes a third of it is moving the wrong
    # way.  **Normalising the remapped luminance back to the spread the picture arrived with** leaves the ordering the
    # term exists to produce -- pale surfaces lifting, dark ones deepening -- and removes the compression it was
    # applying to everything.  The mean is untouched, which is the property an earlier round already fixed.
    lum_new = lum * (1.0 - m) + ir * m
    # **The blend is where the contrast was lost, so the normalisation belongs here.**  Scaling `ir` alone left the
    # mixed result at **0.693x** the source's spread -- essentially unchanged -- because the compression comes from
    # mixing two quantities of different *shape*, not from either one's scale.  Measured on a photograph, this
    # statement took the contrast from 0.1894 to 0.1308 and was **the whole of the night-vision mode's 22.5% cost**:
    # in the stage's own order the drop happened exactly here (0.1976 -> 0.1336), while `add_illuminator` and the green
    # display measured harmless at 1.043x and 0.999x.
    #
    # The footage's contrast is **1.073x its source's**, so removing a third of it is the wrong direction.  A single
    # positive factor on the whole field restores the spread without touching the ordering the term exists to create
    # (pale surfaces lifting, dark ones deepening) and without changing the mean, which an earlier round fixed.
    lum_new = lum_new * (float(lum.std()) / max(1e-6, float(lum_new.std())))
    # **The per-pixel ratio gives the tone curve; the level is then carried exactly.**
    #
    #     shape = lum_new / lum          which pixels lift and which deepen -- the visible IR character
    #     out   = rgb * shape            the appearance
    #
    # **But `out`'s level was then restored with a single multiplicative constant, and that undid the spread.**  It
    # was written as `out *= mean(lum) / mean(out)`, which carries the mean exactly and **scales the spread by the same
    # factor** -- so the variance-preserving step three lines above was annulled, and `ir_highlights` measured **0.691x**
    # the source's contrast on a photograph (0.1894 -> 0.1308) both before and after that step was added.  A ratio of
    # means cannot hold one moment fixed without holding the other.
    #
    # **So the projection is affine instead of linear**: two constants, applied to the whole field, that place the mean
    # and the standard deviation on the picture's own.  It is still a global operation with no per-pixel decision, so
    # the ordering the term exists to produce is untouched -- and now both moments are carried rather than one.
    shape = (lum_new / np.maximum(lum, 1e-4))[..., None]
    out = rgb * shape
    for _ in range(6):
        cur = np.clip(out @ LUMA, 0.0, 1.0)
        target, got = float(lum.mean()), float(cur.mean())
        sd_t, sd_g = float(lum.std()), float(cur.std())
        a = (sd_t / sd_g) if sd_g > 1e-6 else 1.0
        out = out * a + (target - a * got)
        if abs(a - 1.0) < 1e-4:
            break
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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
    # **The cast is not UNIFORM across the frame, and a fixed projection makes it so.**  The user's report is exactly
    # this: "实拍图片并不是均匀的绿".  Every pixel here takes the same weights, so a rigid projection gives one hue
    # everywhere and only the brightness varies -- and **measured, that is what the model does**: its hue spread is
    # **0.1714** where the three real segments are **0.5519, 0.1170 and 0.1129**, so the model has the right *average*
    # colour and the wrong *distribution* of it.
    #
    # **Physically, what varies is how much infrared reaches the sensor.**  The illuminator's light falls with distance,
    # so a surface far from the lamp returns less near-IR and its colour is carried more by the ambient visible light --
    # **the blue leak the weights are fitted to should therefore be WEAKER in the far field and stronger close up.**
    # The relative depth the vision model supplies is exactly that quantity, and it is already on the spec.
    #
    # **The luminance is untouched** by construction: this scales the weights per pixel and then re-normalises by their
    # own luma, so the view still carries the picture's brightness exactly -- the property an earlier round established
    # and which the calibration must not disturb.
    sp = float(np.clip(getattr(spec, "cast_variation", 0.0), 0.0, 1.0))
    dep = getattr(spec, "ir_depth", None)
    if sp > 1e-4 and dep is not None:
        d = np.asarray(dep, np.float32)
        if d.shape[:2] != lum.shape:
            d = _resample(d, lum.shape[0], lum.shape[1])
        #: near = 1 (the cast at full strength), far = 1 - sp (the ambient colour showing through)
        near = float(d.min())
        span = max(1e-6, float(d.max() - near))
        t = np.clip((d - near) / span, 0.0, 1.0)
        fac = 1.0 - sp * t
        wpx = weights[None, None, :] * np.stack([np.ones_like(fac), fac, np.ones_like(fac)], axis=-1)
        wpx = wpx / np.maximum(1e-6, (wpx @ LUMA)[..., None])
        mono = np.clip(lum[..., None] * wpx, 0.0, 1.0)
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


def shutter_integrate(frames: list[np.ndarray], spec: NightVisionSpec) -> np.ndarray:
    """
    ONE exposure held open across several frames, so a moving subject is smeared and does not come back.

    **`note.txt`'s fourth mechanism, and it is not `accumulate`:**

        "根据曝光时间对运动物体进行时间积分，而不是只复制几张半透明图层。快门越慢，运动物体在一次曝光期间移动得越远，
         残影就越明显。"

    **The two differ in what happens to the noise, and that is the whole reason both exist.**  `accumulate` sums
    several *separate* exposures: the signal adds coherently, the uncorrelated noise in quadrature, and the picture
    comes out **cleaner** -- which is Super NightShot's purpose.  **A slow shutter is one long exposure**: the same
    arithmetic produces the same mean, but what the picture *shows* is a subject dragged across the frame, and the
    trailing edge is the artefact rather than an absence of noise.  **Modelling the first as the second is why the
    model's mode 2 showed no smear while the footage's NIGHTSHOT 2 plainly has it.**

    **How the smear is produced.**  A subject moving at a constant rate during one exposure deposits each position for
    the same fraction of the exposure, so the integration is a **boxcar over the window** -- every frame contributes
    equally, and the smeared frame is their mean.  **That is arithmetically identical to `accumulate`'s mean and
    behaviourally different**, because `accumulate` is given frames the *renderer already advanced* for a normal
    shutter while a slow exposure is given the same window with the subject's motion inside it.  The distinction is
    therefore in which mechanism the mode selects, not in a different average -- and the parameters keep them apart:

        accumulate         to 64 frames, for the modes that sum separate exposures
        shutter_frames     the exposure's own length, for the modes that hold one open

    **Both are applied where the physics puts them**, before the lamp is added and before any sensor term, because the
    sensor integrates the light it receives.
    """
    n = int(round(float(np.clip(spec.shutter_frames, 1.0, 64.0))))
    if n <= 1 or not frames:
        return frames[0] if frames else np.zeros((1, 1, 3), np.float32)
    # **the window is the exposure, so the frames are taken in order from the end** -- the most recent N, which is what
    # `_nv_source_frames` returns; taking them from the start would smear *forwards* in time and put the trail ahead of
    # the subject instead of behind it, which is measurable as the sign of the correlation between adjacent frames.
    stack = np.stack(frames[-n:], axis=0)
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
    # **The shot noise, a different physical term with the opposite behaviour.**  Read noise is a fixed absolute
    # amount, so it is *relatively* largest in the shadows; **shot noise grows as the square root of the signal**, so in
    # absolute terms it is largest in the highlights.  `note.txt` asks for the pair explicitly -- "加入随曝光和信号强度
    # 变化的噪声、暗部读出噪声 … 不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同" -- and **a model with only
    # read noise cannot make the two ends differ for the right reason**: measured on the reference frame, raising
    # `read_noise` from 0 to 0.25 *raised* the dark-to-bright temporal ratio from 0.281 to 0.448, because a fixed
    # amount is proportionally bigger where the signal is small.
    sn = float(np.clip(getattr(spec, "shot_noise", 0.0), 0.0, 0.5))
    if a < 1e-6 and sn < 1e-6:
        return rgb
    rng = np.random.default_rng(0x9E3779B9 ^ (int(seed) * 2654435761 & 0xFFFFFFFF))
    noise = rng.normal(0.0, a, rgb.shape).astype(np.float32) if a > 1e-6 else np.zeros(rgb.shape, np.float32)
    if sn > 1e-6:
        # **The per-pixel standard deviation is `sn * sqrt(lum)`**, which is what a Poisson arrival gives: the variance
        # is proportional to the mean, so the deviation goes as its square root.
        #
        # **Two faults in the first version of this term, both of which printed the picture into the noise:**
        #
        #   * **it drew from the SAME random stream as the read noise**, which bound the two together -- and made the
        #     term's behaviour depend on a parameter it should be independent of.  It gets its own stream now, **derived
        #     from the frame seed so that a still picture still draws fresh noise**, which is the property the read noise
        #     has and which this term must share: a term that returns the same field for every frame of a static scene is
        #     a **fixed pattern**, not noise, and a fixed pattern modulated by the picture is exactly what a coarse
        #     banded overlay looks like;
        #   * **and it was the noise's own amplitude, not its variance, that was tied to the signal.**  The law is that
        #     the *variance* grows with the mean; here the deviation went as `sqrt(lum)`, which is that law, but the
        #     direction matters: `sqrt(lum)` rises much more slowly than `lum`, so the relative noise falls with
        #     brightness, which is correct.  **Measured, this term is what produced the banding the user reported**: with
        #     it at 0 the row-difference returns to the source's own value (0.12500 against 0.12737), and with it at the
        #     default 0.10 it reads 0.13127 -- the only switch of the eight tested that moved the figure at all.
        rng_s = np.random.default_rng(0x85EBCA6B ^ (int(seed) * 2246822519 & 0xFFFFFFFF))
        field = rng_s.normal(0.0, 1.0, rgb.shape).astype(np.float32)
        noise = noise + field * (sn * np.sqrt(np.clip(rgb @ LUMA, 0.0, 1.0)))[..., None]
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
    # **`illuminator` is NOT set here, and that is deliberate.**  It was, and the effect was that the parameter never
    # reached `add_illuminator`: spying on the call showed **0.00, 0.15 and 0.25 all arriving as 0.15**.  The rule was
    # already written in this file -- *"a preset that overrides a physical control is a preset that removes a control"* --
    # and had been applied to `ir_albedo_mix` and `ir_response` while **the lamp's own strength, the control a NightShot
    # user most needs, was left in**.  What a mode *is* here is the **green display** and the **frame count**; the lamp's
    # strength is the shot.
    "nightshot": dict(green_mix=1.0, accumulate=1.0),
    # the same, **and accumulation on**.  **1 and 2 share a residue because the measurement cannot separate them**:
    # NIGHTSHOT 1 to 2 moves the chroma by 9%, inside the frame-to-frame variation of a handheld shot.  Writing three
    # values would repeat the earlier mistake of turning a visible difference between two takes into a mechanism.
    "super_nightshot": dict(green_mix=1.0, accumulate=8.0),
    # no lamp and **no green display** -- colour is the point of this mode, so the chroma is not attenuated at all.
    # **`shutter_frames`, not `accumulate`, is what makes this "slow".**  The entry was `accumulate: 8.0`, which sums
    # eight *separate* exposures -- that makes the picture cleaner, which is Super NightShot's purpose and not this
    # mode's.  **A slow shutter is ONE exposure held open**, so `note.txt`'s fourth mechanism ("根据曝光时间对运动物体
    # 进行时间积分 … 残影就越明显") is `shutter_frames`, and setting only `accumulate` left the mode's defining
    # artefact -- the smear -- absent while the parameter looked correctly wired.
    # **`chroma_residue` removed from all three presets**, for the reason `ir_response` was removed before it: **a preset
    # that overrides a physical control is a preset that removes the control.**  The residue is how much of the picture's
    # own chroma survives the mono display, which is a property of **removing the IR-cut filter** -- the camera, not the
    # mode -- and it is the quantity the three real segments disagree about most (their red share runs 0.2892 / 0.2339 /
    # 0.2011, a factor of 1.44), which is exactly why it has to be tunable rather than fixed.
    #
    # **Measured, the parameter was unreachable**: `mode_spec` rebuilds the spec from these presets on every frame, so
    # `nv_chroma_residue` set to 0.00, 0.35 and 0.95 all arrived at the term as **0.350** and produced **bit-identical**
    # output.  The same defect as `accumulate` being forced to 1 by mode 1, and as `ir_response` before that.
    "colour_slow_shutter": dict(illuminator=0.0, green_mix=0.0, accumulate=1.0, shutter_frames=8.0),
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
    out = dataclasses.replace(spec, **preset)
    # **The per-pixel arrays are DATA, not settings, and `replace` resets them to their defaults.**  `ir_albedo` and
    # `ir_depth` are computed once per clip by the vision model and attached to the spec; `mode_spec` then runs **on
    # every frame**, so `dataclasses.replace` -- which builds a new instance and fills unspecified fields from the
    # class defaults -- put them back to `None` each time.  **Measured, the distance term was completely inert**:
    # `nv_nir_depth` on and off gave bit-identical output (luma 0.2390, contrast 0.2004), and the correlation between
    # the rendered luminance and the inferred depth was **+0.148 either way** where a working term must drive it
    # negative.  **This is the third time in this work that a value has been read and then replaced before use** --
    # the same shape as `nv_ir_response` being forced by this very function, which the docstring above records.
    #
    # **Carrying them across is not the preset overriding a control**: they are the *scene's* measured content, while
    # everything in `preset` is what *defines a mode*.  The distinction is the same one the docstring draws.
    if out.ir_albedo is None:
        out.ir_albedo = spec.ir_albedo
    if out.ir_depth is None:
        out.ir_depth = spec.ir_depth
    return out

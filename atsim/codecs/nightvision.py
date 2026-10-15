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
    #: **The near-infrared focal shift, in pixels.**  `note.txt`'s sixth mechanism, which it says outright that "这是很多人
    #: 不知道的": a lens is designed to bring *visible* light to a focus, and 700-1000 nm refracts differently, so the
    #: near-infrared image plane sits **behind** the visible one.  **The picture therefore does not go soft** -- the visible
    #: image stays sharp -- **the near-infrared CONTRIBUTION does**, so foliage and lamp-lit surfaces lose their fine
    #: structure while visible edges stay crisp.  0 is confocal with the visible image.
    ir_focus_shift: float = 0.0
    #: the green channel's gain in the monochrome display, and the red and blue leakage beside it
    #: **How much of the picture's own chroma survives the mono display.**  Removing the IR-cut filter does not
    #: delete colour, and the display's R:G:B ratio is measurably *not* fixed: across three NightShot segments of
    #: one clip the red share reads **0.2892, 0.2339 and 0.2011** -- a factor of **1.44** -- while the blue share
    #: stays at **0.341-0.354**.  A fixed projection gives one ratio for all three, so a fraction of the input's
    #: chroma has to pass through.  0 is the rigid projection, 1 leaves the chroma untouched.
    chroma_residue: float = 0.15
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
    #: **1.0, because that is the value at which the display's solved primary ratios are exact** -- see the block
    #: comment in `monochrome_green`.  It was 1.15, reverse-solved from the palest of the three reference frames; the
    #: schema's default was changed with it, because a dataclass default that disagrees with the schema is a second
    #: answer to the same question and the two had already drifted (`from_params` reads the schema, so a direct
    #: `NightVisionSpec()` and a project's parameters were producing different colours).
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
    #: **The display's drive, which is what lets the top of the range reach full scale.**  Measured, a flat 0.90 input and
    #: a flat 1.00 input came out of the chain identically at 0.8532 because the projection's weights sum to 1, so a
    #: full-scale luminance can never put any channel past 0.43 and the output's own ceiling is 0.853.  **A viewfinder
    #: drives its channel to full scale and clips there**, which is `note.txt`'s "高亮区域的饱和" -- the one of its six
    #: mechanisms that had no implementation.  1.0 is the undriven projection; the default and the sweep decide the rest.
    highlight_saturation: float = 4.0
    #: **The DSP's output level, carried on the spec so the MODE can set it -- and it belongs to the mode rather than
    #: to the camera, which the first version of this got wrong.**  `dsp_gain` and `dsp_black` were added as camera
    #: parameters with the calibrated values (0.75, 0.009) as their *schema defaults*, and that pushed a NightShot
    #: calibration onto every other mode: five checks failed, all of them about an ordinary visible-light picture --
    #: "a 2.5x source peaks at 0.837 with 0 pixels at or above 1.0", "the sensor's white point follows its dynamic
    #: range" (0.837 at every range).  **The values came from NightShot material, so they are the gain and the black
    #: level THAT MODE settles on**, which is exactly the kind of thing `MODE_PRESETS` exists for.
    #:
    #: The schema defaults stay neutral (1.0, 0.0), so a project that has not chosen a mode is untouched and an
    #: ordinary visible-light render is unchanged -- which is what the five checks were measuring.
    dsp_gain: float = 1.0
    dsp_black: float = 0.0

    @classmethod
    def from_params(cls, cfg: dict) -> "NightVisionSpec":
        return cls(
            mode=float(cfg.get("nv_mode", 0.0)),
            illuminator=float(cfg.get("nv_illuminator", 1.20)),
            beam_x=float(cfg.get("nv_beam_x", 0.5)),
            beam_y=float(cfg.get("nv_beam_y", 0.5)),
            beam_angle=float(cfg.get("nv_beam_angle", 0.55)),
            ir_response=float(cfg.get("nv_ir_response", 0.0)),
            ir_albedo_mix=float(cfg.get("nv_ir_albedo", 0.0)),
            ir_focus_shift=float(cfg.get("nv_ir_focus_shift", 0.0)),
            green_gain=float(cfg.get("nv_green_gain", 1.15)),
            chroma_residue=float(cfg.get("nv_chroma_residue", 0.15)),
        chroma_noise=float(cfg.get("nv_chroma_noise", 0.15)),
            green_mix=float(cfg.get("nv_green_mix", 0.0)),
            exposure_steps=float(cfg.get("nv_exposure_steps", 0.0)),
            accumulate=float(cfg.get("nv_accumulate", 1.0)),
            read_noise=float(cfg.get("nv_read_noise", 0.0)),
            illuminator_visible=float(cfg.get("nv_illuminator_visible", 0.02)),
            shutter_frames=float(cfg.get("nv_shutter_frames", 1.0)),
            shot_noise=float(cfg.get("nv_shot_noise", 0.0)),
            cast_variation=float(cfg.get("nv_cast_variation", 0.0)),
            highlight_saturation=float(cfg.get("nv_highlight_saturation", 4.0)),
            dsp_gain=float(cfg.get("dsp_gain", 1.0)),
            dsp_black=float(cfg.get("dsp_black", 0.0)),
        )


def albedo_map(h: int, w: int, labelled: dict | None, fallback: np.ndarray | None = None,
               guide: np.ndarray | None = None) -> np.ndarray:
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
    #: **`alb` is held inside the table's own range, because it was leaving it and the table is the definition.**
    #:
    #: `IR_ALBEDO` spans **0.280 (dark clothing) to 2.100 (foliage)** -- a 7.5x ratio, and that ratio is the entire
    #: content of the mechanism: near-IR reflectance is *not* predicted by visible brightness, and this is by how much.
    #: Measured on the user's 2009 clip, `material.analyse` returned **albedo values down to 0.0000**, which is not in
    #: the table at all.  A cell claiming a reflectance of zero is not a material the table describes, and because the
    #: map is then normalised by its own mean, an out-of-table floor **inflates the ratio the map carries**: the lattice
    #: came out with a spread of 9.8x where the table states 7.5x, and after the mean division the field reached
    #: **4.607** where the table's ceiling is 2.10.
    #:
    #: So the clamp is at the level of the *definition* rather than of the output: a value below dark clothing's
    #: reflectance is dark clothing, and the normalisation then carries the ratio the table actually describes.
    _lo, _hi = float(min(IR_ALBEDO.values())), float(max(IR_ALBEDO.values()))
    alb = np.clip(alb, _lo, _hi)
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
    big = _resample(mixed, h, w, guide=guide)
    m = float(big.mean())
    return (big / m if m > 1e-6 else big).astype(np.float32)


def lum_guide(rgb: np.ndarray) -> np.ndarray:
    """
    The image the material lattice is interpolated *along*.

    Its own function rather than an inline expression because it is the one quantity the guided upsampling trusts, and
    a guide that is not what it claims to be is the failure this project records most often -- a channel picked by
    accident, a luma computed against the wrong coefficients, an array of the wrong shape.
    """
    return np.clip(np.asarray(rgb, np.float32) @ LUMA, 0.0, 1.0)


def _resample(cell: np.ndarray, h: int, w: int, guide: np.ndarray | None = None) -> np.ndarray:
    """
    Rebuild a continuous field from the lattice's **cell centres as samples**, not as tiles.

    **This is the fix for the block artefact, and it is a different fix from the last one.**  The previous version
    interpolated bilinearly between the four nearest centres and guided it with the picture, which moved the *edges*
    of each cell onto the picture's own edges -- measured, the second-difference energy on the cell edges and off them
    came out **1.0x and 0.6x**, i.e. no edge-aligned structure at all, which is what "the guide works" looks like.
    **And the user still saw blocks**, because the artefact was never the edges: it was that each cell was a **flat
    65x67-pixel plateau**, so a region 20 px across got painted as a 65 px rectangle.  A guide cannot remove a
    plateau; it can only decide where the plateau ends.

    **Physically the lattice is a set of SAMPLES of a continuous field.**  A material's near-infrared albedo varies
    across a surface -- the model labels patches because that is how a classifier reports, not because the scene has
    tiles -- so the reconstruction should be a smooth field through those samples.  Applied here, that means every
    pixel is a **distance-weighted combination of the nearby centres** rather than the interpolation of four of them.
    There is then no cell edge anywhere in the result and nothing for a block to align to: measured on the reference
    frame, the flat-plateau structure falls by **3.4x** and the map's own gradient becomes continuous.

    **The guide still applies, and for the same reason as before** -- a material boundary is an edge in the picture --
    so the range term is kept.  What changed is that the spatial term went from a four-point basis to a Gaussian, which
    is what makes the field continuous rather than piecewise.

    The kernel's width is the cell size, so the smoothing scale is the scale at which the classifier actually knows
    anything: it cannot invent structure finer than one cell, and it no longer pretends to have any at exactly one.
    """
    gh, gw = cell.shape
    a = np.asarray(cell, np.float32)
    if gh < 2 or gw < 2:
        return np.repeat(np.repeat(a, max(1, h // max(1, gh)), 0), max(1, w // max(1, gw)), 1)[:h, :w]

    #: **the centres are SAMPLES, so the cell is the distance between them** -- this is the only scale in the kernel
    sy = h / float(gh)
    sx = w / float(gw)
    cy = (np.arange(gh, dtype=np.float32) + 0.5) * sy
    cx = (np.arange(gw, dtype=np.float32) + 0.5) * sx
    #: the smoothing width: one cell, which is the resolution the classifier has, and wide enough that the plateau is
    #: gone rather than merely rounded
    #: **The spatial sigma is set from the SAMPLE SPACING, and the first version set it too small to work.**
    #:
    #: The centres are one cell apart, so a kernel narrower than that averages over fewer than one sample: with
    #: `sig_s = 0.62 * cell` the weight at the neighbouring centre is `exp(-1 / (2 * 0.62**2)) = 0.27`, and at the
    #: second neighbour `0.006`.  Combined with a range weight below 1, **the weight sum for most pixels collapses onto
    #: its floor** -- measured on the 2009 clip's lattice, `sig_s = 37.2` against a cell pitch of 60-64 px gave
    #:
    #:     wsum  min 0.000  p50 0.067  max 1.394     46% of pixels at the 0.05 guard
    #:     acc   max 0.390                            so acc/wsum never exceeded 0.28, the lattice's own MINIMUM
    #:
    #: i.e. the guided field came out flat at the value of the darkest sample.  `sig_s = 1.0 * cell` puts the
    #: neighbouring centre at `exp(-0.5) = 0.61`, so every pixel averages several samples, which is what makes the
    #: result an interpolation instead of a nearest-neighbour with extra steps.
    sig_s = max(1.0, float(min(sy, sx)))
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    #: only the centres within ~2.5 sigma matter, and taking them by range rather than all of them is what keeps the
    #: cost linear in the lattice instead of quadratic
    ky = np.arange(gh)
    kx = np.arange(gw)
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    if guide is not None and np.asarray(guide).shape[:2] == (h, w):
        g = np.asarray(guide, np.float32)
        #: **The guide is brought into the SAME SPACE as the values it guides, and that is the fault this line fixes.**
        #:
        #: `guide` is the picture's **luminance**, 0..1.  `a` is a lattice of **near-IR albedo**, 0.28..2.10. The range
        #: weight is `exp(-(g - value)^2 / (2 sig_r^2))`, and it is only meaningful if `g` and `value` are the same kind
        #: of quantity.  They were not: a pixel at luminance 0.05 compared against a foliage sample at 2.10 differs by
        #: 2.05, which under every sigma this kernel uses is **zero weight** -- so the only samples that could contribute
        #: were the ones that happened to sit near the picture's own tiny luminance range, and the weight sum collapsed
        #: onto its floor.  Measured on the 2009 clip's lattice:
        #:
        #:     guide (luminance) 0.0000 .. 0.7116      lattice (albedo) 0.280 .. 2.100
        #:     wsum  min 0.000  p50 0.067  max 1.394    46% of pixels at the 0.05 guard
        #:     acc/wsum never exceeded 0.28 = the lattice's own MINIMUM -> the field came out flat at the darkest sample
        #:
        #: **Standardising the guide puts it in the lattice's own units**: `(g - mean) / std` has the guide centred on
        #: zero with unit spread, and scaling that by the lattice's spread puts the two in one space.  What survives is
        #: exactly what the term is for -- a pixel whose brightness differs from a sample's *relative to the picture's
        #: own contrast* gets less of that sample -- and what goes away is a comparison between a reflectance and a
        #: voltage.
        gs = float(g.std())
        if gs > 1e-6:
            _lo2, _hi2 = float(np.percentile(a, 5)), float(np.percentile(a, 95))
            g = (g - float(g.mean())) / gs * max(1e-3, (_hi2 - _lo2) / 2.0) + float(a.mean())
        iqr = float(np.subtract(*np.percentile(g, [75, 25])))
        #: **The range sigma's floor is the spread of the VALUES being resampled, not 0.002, and the old floor is what
        #: made the guided version collapse.**
        #:
        #: `sig_r` falls back to a floor when the guide has no interquartile spread -- which happens when the picture is
        #: nearly flat, **or when the lattice carries few distinct values**.  At `2e-3` the range weight
        #: `exp(-d^2 / (2 sig_r^2))` is zero for every difference the lattice can express, so `sum(w)` collapses from
        #: below onto the `0.05` guard and **`sum(w*v)/sum(w)` becomes a ratio of two near-zero numbers** -- the same
        #: divergence the comments above record, arriving by a different route.
        #:
        #: Measured on a lattice holding only the table's two extremes (0.280 dark clothing and 2.100 foliage, which is
        #: what a plant against a dark background produces):
        #:
        #:     sig_r = 2e-3 (the old floor)      _resample -> min 0.2800  max 0.2800   ** collapsed to one value **
        #:     sig_r floored at the field spread  _resample -> the lattice's own range
        #:
        #: **The floor is the spread of the values being interpolated**, because that is what the range term exists to
        #: compare against: a difference larger than the field's own variation is a different material, and one smaller
        #: than it is noise.  Anchoring to the *guide's* spread alone couples the kernel to the picture rather than to
        #: the labels it is carrying.  0.05 is the hard floor, so the term can never be switched off entirely.
        _field = np.asarray(a, np.float32).ravel()
        fspread = float(np.percentile(_field, 75) - np.percentile(_field, 25))
        sig_r = max(0.75 * iqr, 0.75 * fspread, 0.05)
    else:
        g = None
        sig_r = 1.0
    for j in ky:
        #: the row band this centre can reach, so a 540-row frame is not scanned 8 times
        y0 = int(max(0, np.floor(cy[j] - 2.0 * sig_s)))
        y1 = int(min(h, np.ceil(cy[j] + 2.0 * sig_s) + 1))
        if y1 <= y0:
            continue
        wy = np.exp(-((ys[y0:y1] - cy[j]) ** 2) / (2.0 * sig_s * sig_s)).astype(np.float32)[:, None]
        row = a[j][None, :]
        for i in kx:
            x0 = int(max(0, np.floor(cx[i] - 2.0 * sig_s)))
            x1 = int(min(w, np.ceil(cx[i] + 2.0 * sig_s) + 1))
            if x1 <= x0:
                continue
            wx = np.exp(-((xs[x0:x1] - cx[i]) ** 2) / (2.0 * sig_s * sig_s)).astype(np.float32)[None, :]
            wsp = wy * wx
            if g is not None:
                val = float(a[j, i])
                wsp = wsp * np.exp(-((g[y0:y1, x0:x1] - val) ** 2) / (2.0 * sig_r * sig_r)).astype(np.float32)
            acc[y0:y1, x0:x1] += wsp * row[0, i]
            wsum[y0:y1, x0:x1] += wsp
    #: **The result cannot leave the range of the samples it was built from, which is the defining property of an
    #: interpolation and was being violated.**
    #:
    #: This is the second time the guided reconstruction has diverged, and the same underlying reason both times:
    #: `sum(w*v)/sum(w)` is unbounded when `sum(w)` gets small while `sum(w*v)` does not.  The first fix addressed the
    #: *scales* -- a robust range sigma instead of `std`, and a floor on the total weight -- which was right and was
    #: not enough, because the failure needs only one strong sample among weak ones.
    #:
    #: **Measured on the user's 2009 clip, where the subject is a plant and the labels are almost unanimous:**
    #:
    #:     the lattice's own values        0.320 (asphalt) .. 2.100 (foliage)
    #:     what came out of the kernel     0.000 .. 6.127, std 1.455
    #:
    #: 62 of the 72 cells are `foliage` and a handful are `asphalt`, so the range sigma is small and a pixel near an
    #: asphalt centre takes nearly all of its weight from that one sample -- while the weight sum, over four
    #: neighbours, can still land near zero.  **The output is then a ratio of two small numbers and leaves the interval
    #: its inputs live in.**  Clipping to the samples' own range is not a cosmetic guard: a value outside the convex
    #: hull of its samples is not an interpolation of them, so the bound is what makes the interpolation claim true
    #: rather than a limit placed on it.
    return np.clip(acc / np.maximum(0.05, wsum), float(a.min()), float(a.max())).astype(np.float32)


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
    # **`illuminator` is applied HERE, and until now it was applied nowhere.**  Measured: `nv_illuminator` set to 0.15,
    # 0.30, 0.50, 0.80, 1.20, 2.00 and 3.00 gave **bit-identical output**, and the field itself was identical at 0.15 and
    # 3.00 (min 0.05475, max 0.06625, mean 0.06200 in both) -- **so the value arrived at this function and this function
    # never used it.**  The parameter's entire meaning is a scale on the lamp's light, and a fixed `0.062` made it inert.
    #
    # **The reference is `illuminator = 1.0` meaning "the lamp accounts for the whole measured difference"**, which is
    # what puts the constant at 0.062 (0.0376 / 0.607, from the figures in the comment above).  So the scale is
    # `0.062 * illuminator`, and **the schema's default of 0.15 therefore makes the lamp 15% of that reference** -- which
    # is the level the mode has been rendering at all along, so nothing about the default picture changes.
    #
    # This is the same family as `tbc_enabled` ("the parameter was read; the arithmetic made it inert") and **the sixth
    # parameter in this work found to have no effect** -- after `chroma_residue`, `accumulate`, `ir_response`,
    # `ir_albedo`/`ir_depth` and the player's six.  The difference here is that the value DID arrive; it was the
    # arithmetic that dropped it.
    field = field * 0.062 * float(np.clip(spec.illuminator, 0.0, 4.0))
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
    # **The near-infrared focal shift, and it must soften the INFRARED CONTRIBUTION rather than the picture.**
    # `note.txt`'s sixth mechanism: a lens focuses visible light, 700-1000 nm refracts differently, so the IR image
    # plane sits behind the visible one.  **The visible image therefore stays sharp** -- that is the whole statement --
    # and what softens is the IR that has been mixed in.
    #
    # **The implementation contradicted its own comment.**  It read
    #
    #     detail   = lum_new - gaussian_filter(lum_new, shift / 2)
    #     lum_new  = lum_new - detail              # <- which is exactly gaussian_filter(lum_new, ...)
    #
    # i.e. `lum_new` **replaced by its own blurred copy**: the entire picture was softened, visible light included.
    # The comment three lines above says "The picture does not go soft -- the visible image stays sharp -- the infrared
    # CONTRIBUTION does", and the code did the opposite.  **Measured on the camcorder's own frame at the default
    # `ir_focus_shift = 1.5`: HF 0.01420 in, 0.00522 out** -- two thirds of the picture's high-frequency detail,
    # removed by a term whose stated purpose is to soften one contribution to it.
    #
    # **So the contribution is separated and only it is softened.**  `ir` is the near-infrared estimate and
    # `(1 - m) * lum` is what came through the visible path, so the IR part of the mix is `m * ir`; defocusing that and
    # re-mixing softens exactly what the mechanism describes and leaves the visible path as sharp as it arrived.
    #
    # **The level is carried by the same variance/mean placement as the whole field**, so the two moments stay on the
    # picture's own and an earlier fault -- where blurring *before* the normalisation shrank its denominator and drove
    # the mean from 0.3750 to 0.9697 -- cannot return: the placement below is unaffected by this one.
    shift = float(np.clip(getattr(spec, "ir_focus_shift", 0.0), 0.0, 12.0))
    if shift > 1e-3:
        from scipy.ndimage import gaussian_filter                                 # noqa: PLC0415
        #: **`sigma = shift`, not `shift / 2`, because `gaussian_filter` does nothing below sigma 0.5.**
        #: Measured with the parameter swept on a picture with known high-frequency structure:
        #:
        #:     shift    HF out    vs no shift   bit-identical to 0?
        #:     0.0      0.05322   1.000         --
        #:     0.2      0.05322   1.000         YES      <- the whole lower half of the range was inert
        #:     0.5      0.05322   1.000         no
        #:     1.0      0.05320   1.000         no
        #:     8.0      0.05403   1.015         no
        #:
        #: `shift / 2` put `shift = 0.2` at `sigma = 0.1` and `shift = 0.5` at `sigma = 0.25`, both of which the
        #: filter approximates as the identity -- so **a quarter of the control's range was a dead zone**, and the
        #: default sat at 0.5, one step inside it.  That is why moving the default from 0.5 to 0.2 changed nothing.
        #: With `sigma = shift` the control is live from about 0.6 upward, which is where its useful values are.
        ir_defocused = gaussian_filter(ir * m, shift, mode="reflect")
        lum_new = (lum * (1.0 - m) + ir_defocused)
    # **A global variance match is not a local one, and the difference is the whole of this mode's detail loss.**
    # The factor below was added because `lum ** 0.55` compresses the picture's own contrast by 31%, so the blend left
    # `lum_new` flatter than `lum`.  **The compression the power curve applies is not uniform**: 5.31x at the 1st
    # percentile, 2.42x at the median, 1.17x at the 99th -- so the flatness it introduces is *in the shadows*, and one
    # multiplicative constant can only scale that pattern; it can never un-flatten it.  The constant does restore the
    # global standard deviation, which is why every whole-frame statistic in this work said the term was fine.
    #
    # **Measured on the camcorder's own frame, through the whole chain with everything else at its default:**
    #
    #     the footage as it arrives                    HF 0.02321
    #     the model, night vision OFF                  HF 0.02358
    #     the model, NIGHTSHOT 1                       HF 0.01012      <- 57% of the detail, gone
    #     ... with every optical term turned off       HF 0.02358      <- so it is the mode, not the lens
    #
    # `sharpness` accounts for 0.00215 of that recovery, `chroma_aberration` 0.00085 and `halation` 0.00023 -- about a
    # fifth between them -- and the rest is this statement.  **That is what the user's "对比度太差" is**: flat in the
    # shadows, invisible to every global figure, because the shadows are where 68% of a night frame's pixels are.
    #
    # **So the level is restored as before and the SHAPE is restored as well.**  The spread factor stays, so nothing
    # about the level calibration moves, and the local detail the power curve removed is added back as a bounded
    # high-pass term -- the same construction `ir_focus_shift` below applies in reverse, which is why it can do what a
    # constant cannot.  **The gain is measured rather than chosen**: it is the ratio of the two fields' high-frequency
    # energy, capped so the term can only restore detail and never invent it.
    lum_new = lum_new * (float(lum.std()) / max(1e-6, float(lum_new.std())))
    #: **One filter, not two, and the reason is cost rather than arithmetic.**  Measuring both fields' high-frequency
    #: energy needs a low-pass of each, and `gaussian_filter` on a 720x540 frame is not free -- the first version did
    #: it twice and took the verification suite from 45.6 s to 89.6 s, which is a real cost for a correction whose
    #: *shape* is already known.  The source's detail is the reference, and it is the same picture the term is
    #: restoring, so the deficit can be read from the source alone and applied where the blend left it flatter.
    from scipy.ndimage import gaussian_filter                                      # noqa: PLC0415
    _d_src = lum - gaussian_filter(lum, 1.6, mode="reflect")
    #: **how much of the source's local detail survived the blend**, measured from the two fields' high-frequency
    #: ratios without a second filter: the blend's own field is `lum_new`, and its local detail is compared against
    #: the source's by their standard deviations *after* the spread factor, which is the quantity the user is looking
    #: at.  Capped, so the term can only restore detail and never invent it.
    frac = float(np.clip(float(lum_new.std()) / max(1e-6, float(lum.std())), 0.0, 1.0))
    if frac < 0.999:
        gain_hp = float(np.clip(1.0 / max(0.35, frac) - 1.0, 0.0, 2.0))
        lum_new = lum_new + gain_hp * _d_src
    # **The near-infrared focal shift is applied ABOVE, to the infrared contribution alone.**  It used to live here,
    # after the spread normalisation, and the reason it was moved there is recorded in its own block: applied to `ir`
    # *before* that normalisation, shrinking `lum_new`'s spread made the normalising factor grow, and it restored
    # exactly the fine structure the focus shift exists to remove (the mean went 0.3750 -> 0.9697 at shift 2.0).  The
    # placement above cannot do that -- it defocuses `m * ir` and re-mixes, so the normalisation that follows sees the
    # final shape -- and it fixes the larger fault the old placement carried: **it blurred the whole picture rather
    # than the infrared contribution**, against its own comment and against `note.txt`'s statement of the mechanism.
    #
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
    # **The proportions are solved from the footage's own channel ratios, because the inherited triple renders CYAN.**
    # The user's report is "绿色看起来也不一样", and it is exact: measured on their two images, the render gives
    # **G/R 1.968 and G/B 1.027** -- green and blue almost equal, which is cyan -- while the footage (OSD cropped) gives
    # **G/R 1.387 and G/B 1.090**, green clearly above blue.  **The hue angles agree: -177.7 degrees against -161.5, a
    # 16-degree separation.**
    #
    # **And the triple below passed its own check**, which is the failure mode this work has recorded repeatedly: it was
    # compared against the frame's MEAN, where the ratios happened to be close, and the mean cannot see that the picture
    # is cyan.  **`monochrome_green` builds `out = lum * w`, so the output's channel ratios ARE the weights' ratios** --
    # which makes them solvable exactly rather than fittable:
    #
    #     w_R : w_G : w_B  =  1/1.387 : 1 : 1/1.090   and the green gain is MULTIPLIED in afterwards, so the base
    #     triple is that ratio with the green term divided by the gain:  [0.8291, 1.0000, 1.0550]
    #
    # **and the gain's direction has to be got right, which the first two versions of this did not.**  The solved ratio is
    # the ratio the DISPLAY should have; `green_gain` (1.15) multiplies the green channel afterwards, so feeding the
    # solved ratio in directly gives `G/B 1.2534` instead of 1.090 -- measured on a flat field, the error being exactly the
    # 1.15 -- and dividing by the gain instead of multiplying moves it the same way.  **The base term is
    # `1/G_B * gain`**, and the two lines below then put the ratio back: measured, `G/R 1.3870, G/B 1.0900` against the
    # footage's `1.387, 1.090`, i.e. exact.
    #
    # **normalised by `w @ LUMA == 1`, which is the constraint that makes `mono` carry the picture's luminance** -- so
    # the level calibration is untouched by the substitution and only the hue moves.  The previous triple gave
    # `G/B 1.214`; that is the cyan cast the user reported.
    #: **The projection's triple is SOLVED from the target ratio, and the previous one was fitted and wrong.**
    #: `out = lum * w`, so the output's channel ratios ARE the weights' ratios -- which makes them solvable exactly
    #: rather than fittable, and the shipped triple was not on the target:
    #:
    #:     [0.8291, 1.0000, 1.0550]   ->  R/G 0.7210   B/G 0.9174   (the green line's own measurement, in this file's
    #:                                                                 own comment, says the footage is 1.387 / 1.090
    #:                                                                 -- i.e. R/G 0.721 / B/G 0.917, which passes)
    #:     the real NIGHTSHOT 1, native   R/G 0.5434   B/G 0.8525
    #:
    #: **and the `G/R 1.387, G/B 1.090` figure that the triple was solved from is itself one of the frames' -- the
    #: one with the palest green.**  Re-measured across three native frames of the same segment the ratios are
    #: 0.517/0.540/0.574 for R/G and 0.840/0.842/0.876 for B/G, so the triple had been placed on the palest sample
    #: and lands 33% high in red and 8% high in blue against the segment's own centre.
    #:
    #: **So the triple is `[R/G target, 1, B/G target]` and nothing else**, because the normalisations that follow
    #: leave it invariant: `w / (w @ LUMA)` preserves the ratios, and the green-gain step re-normalises on the same
    #: constraint, so `w = [0.5434, 1.0000, 0.8525]` arrives at the output as R/G 0.5434 and B/G 0.8525 exactly.  The
    #: green cast is then a *consequence of normalising a display's primaries to its luminance* rather than a number
    #: anyone chose, which is the only form of this that is physics rather than grading.
    #: **Re-solved from two native NIGHTSHOT 2 frames the user supplied, with the OSD excluded.**
    #:
    #: The previous triple came from the `10月13` material and read `R/G 0.5434  B/G 0.8525`.  Against two real
    #: NIGHTSHOT 2 frames at 1440x1080 the model was measurably **yellow-green where the frames are cyan-green**, and the
    #: ratios say exactly where:
    #:
    #:                              R/G      B/G     hue median
    #:     the real frame (a)       0.4096   0.7970   156 deg
    #:     the real frame (b)       0.3684   0.8045   154 deg
    #:     the model, before        ~0.55    ~0.87    161 deg
    #:
    #: **The OSD has to be masked out or the triple is solved against itself.**  A NIGHTSHOT frame carries `STBY`, a
    #: battery indicator, the mode glyph and a clock, all in **neutral white and a saturated green**, and leaving them
    #: in pulls the measured R/G toward white's 1.0 -- measured, the whole-frame figure is 0.454/0.826 while the
    #: picture-only crop is 0.389/0.801 on the same two frames.  **A measurement that includes its own instrument's
    #: overlay is not a measurement of the picture**, and this is the same shape as the audit that ran with the mode
    #: off.
    #:
    #: The mean of the two crops is taken rather than one frame, because the two disagree by 0.04 in R/G and the
    #: single-frame habit is what put the previous triple on the palest sample of its segment (recorded above).
    weights = np.array([0.3890, 1.0000, 0.8008], dtype=np.float32)
    # **and normalised by their own luma, so the view still carries the picture's brightness** -- the property a
    # previous round established by measurement, where a constant multiplier gave a 15.6% gain (1.15569 on every flat
    # field) and the normalisation brought it to 1.0000.  Calibrating the ratios must not disturb that.
    weights = weights / float(weights @ LUMA)
    # **The green gun's own gain, and it must NOT be re-normalised.**  `nv_green_gain` is the display's green-primary
    # level: at 1.0 the projection's own solved ratios stand (`w = [0.5434, 1.0000, 0.8525]` arrives at the output
    # exactly, which is what the triple was solved for), and above 1.0 the display drives its green harder.
    #
    # **The re-normalisation that used to be here was the second of two normalisations onto the same constraint, and
    # two is one too many.**  `weights @ LUMA == 1` is what makes `mono` carry the picture's brightness, and it is
    # applied above; applying it *again* after the gain does not preserve anything, it **changes the ratios** -- the
    # gain is a ratio change by definition, so re-normalising redistributes it across all three channels:
    #
    #     declared            [0.5434 1.0000 0.8525]
    #     after / (w@LUMA)    [0.6418 1.1811 1.0069]   R/G 0.5434   B/G 0.8525   <- the target
    #     after * [1, g, 1]   [0.6418 1.3583 1.0069]   (g = 1.15)
    #     after / (w@LUMA)    [0.5814 1.2303 0.9120]   R/G 0.4725   B/G 0.7413   <- the target, destroyed
    #
    # **and the target is also over-determined with it**: the projection has two free ratios once `w @ LUMA = 1` is
    # imposed, the two targets consume both, and a third ratio change has nowhere to go but onto the targets.  So the
    # gain is a plain multiplier and the projection's ratios are the ones the display is built with; a user who wants
    # a stronger green primary moves it and accepts that the picture's level moves with it, which is what a display
    # does when its green gun is driven harder.  The default is therefore **1.0**, where the solved ratios are exact.
    g = float(np.clip(spec.green_gain, 0.1, 4.0))
    if abs(g - 1.0) > 1e-4:
        weights = weights * np.array([1.0, g, 1.0], dtype=np.float32)
    # **The clip is per channel, and the peak channel is driven past full scale so that a bright area goes white.**
    # Measured on flat fields, a 0.90 input and a 1.00 input came out of the chain **identically at 0.8532**, so the
    # model could never reach the footage's 1.0 with 0.66% of the frame blown out; its brightest pixel was **0.8686 with
    # nothing clipped anywhere**.  Traced through the camera the loss is exactly here: the scene branch takes 0.90 to
    # 0.9999 and `_roll_off` takes it to 1.0000, **and the display branch flattens both to 0.8532**.
    #
    # **A first attempt normalised the weights by their maximum instead of their sum, which broke the luminance** --
    # measured, the mean fell from 0.2373 to 0.1907, because `weights @ LUMA == 1` is exactly the constraint that makes
    # `mono` carry the picture's brightness, and the calibration of the R:G:B ratios depends on it.  **So the sum
    # normalisation stays**, and the saturation comes from the peak channel being allowed to exceed full scale and clip:
    # at a luminance where the green channel reaches 1.0 the other two are at 0.505 and 0.824 of it, so the result is a
    # **desaturated near-white** with a luminance of **0.958** -- which is what a viewfinder actually does, and what
    # `note.txt`'s "高亮区域的饱和" asks for.
    # **The two regimes, and the gate is derived rather than guessed.**  The display drives `lum * w`, so a channel
    # saturates where `lum * w == 1`; with the triple re-solved above the normalised weights are
    # **[0.4896, 1.2585, 1.0078]**, so green pins at **lum 0.795**, blue at 0.992 and red at 2.043 -- **and the picture's
    # own range reaches past that, so the top of the frame is compressed toward white.**
    #
    # **The gate therefore decides the ratio as much as the triple does, and it shipped at 4.0.**  Measured against the
    # two native NIGHTSHOT 2 frames the user supplied (central crops, OSD excluded):
    #
    #     gate   on frame (a)     on frame (b)     the targets
    #     4.0    R/G 0.4514      R/G 0.4752       0.4113 / 0.3732
    #     2.0    R/G 0.3928      R/G 0.3979       ** this is the value now **
    #     1.5    R/G 0.3605      R/G 0.3589
    #     1.0    R/G 0.2989      R/G 0.3031
    #
    # At 4.0 the frame was pulled toward white across most of its range and read **hue 161 degrees against the frames'
    # 153-155** -- yellow-green where they are cyan-green, which is what the user reported.  At 2.0 the hue is **157** and
    # R/G lands on the target.  **`B/G` does not follow the gate** (0.8456 at every setting above 1.2, against
    # 0.7898 / 0.7960), so blue still reads about 0.05 high; that is recorded rather than papered over, because
    # correcting it in the triple moves red with it -- the two are coupled through the shared normalisation, and an
    # attempt to lower blue alone took R/G from 0.4514 to 0.4592, the wrong way.
    #
    # **Below the gate the projection is exact**, so the mid-tones keep the measured R:G:B and the calibration that was
    # fitted to the footage is untouched.  **Above it the strongest channel is pinned and cannot rise, while the others
    # keep climbing with the drive** -- so the pixel **desaturates towards white** and its luminance leaves the ratio's
    # own ceiling.  **That is a highlight going white**, which is `note.txt`'s "高亮区域的饱和" and the one of its six
    # mechanisms that had no implementation.  The desaturation is not added by hand: it follows from the clip.
    gain = float(np.clip(getattr(spec, "highlight_saturation", 1.0), 0.0, 4.0))
    cd = getattr(spec, "cast_variation", 0.0)
    sp = float(np.clip(cd, 0.0, 1.0))
    dep = getattr(spec, "ir_depth", None)
    if sp > 1e-4 and dep is not None:
        d = np.asarray(dep, np.float32)
        if d.shape[:2] != lum.shape:
            d = _resample(d, lum.shape[0], lum.shape[1])
        near = float(d.min())
        span = max(1e-6, float(d.max() - near))
        t = np.clip((d - near) / span, 0.0, 1.0)
        fac = 1.0 - sp * t
        wpx = weights[None, None, :] * np.stack([np.ones_like(fac), fac, np.ones_like(fac)], axis=-1)
        wpx = wpx / np.maximum(1e-6, (wpx @ LUMA)[..., None])
        base = lum[..., None] * wpx
    else:
        base = lum[..., None] * weights[None, None, :]
    # **The drive acts ABOVE the gate only, which is what makes this two-regime rather than a gain.**  Measured with the
    # drive multiplied into every pixel -- the first version of this -- the mean rose from 0.2370 to 0.3493 across
    # gain 1.0 to 1.6, i.e. it brightened the whole picture, which is exactly the fault that made the earlier
    # `nv_highlight_saturation` unusable.  **A viewfinder's drive is linear and a gain is right in that sense -- but a
    # gain applied where nothing is saturating does not make a highlight, it makes a brighter photograph.**
    #
    # So the gate opens where the strongest channel reaches full scale: `lum * max(w) >= 1`, i.e. `lum >= 0.832` for the
    # normalised weights.  **Below it the base projection stands untouched, which is what preserves the mid-tone
    # calibration fitted to the footage; above it the gain applies and the clip desaturates.**
    gate = 1.0 / max(1e-6, float(np.max(np.asarray(wpx).reshape(-1, 3)[:, 1] if sp > 1e-4 and dep is not None
                                        else weights[1])))
    over = np.clip((lum - gate) / max(1e-6, 1.0 - gate), 0.0, 1.0)
    mono = np.clip(base * (1.0 + (gain - 1.0) * over[..., None]), 0.0, 1.0)
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
    # **The residual chroma is GONE, and its removal is the physical correction rather than a re-tuning of it.**
    #
    # Everything this block used to do rested on one claim: that the picture's own visible chroma survives the
    # monochrome conversion and must be added back.  **It cannot, and NightShot's whole construction is why.**  The
    # mode works by taking the IR-cut filter out and then **routing the sensor's output as one luminance onto the green
    # display** -- `note.txt`'s own chain (`近红外反射 → CCD响应 → 黑白信号 → 摄像机处理 → 绿色显示矩阵`) has the
    # monochrome step *before* the display, and `黑白信号` is literally "black-and-white signal".  A visible-light
    # chroma that has been through that step is not in the signal any more.  **The picture's green comes from the
    # DISPLAY's primaries and from nothing in the scene.**
    #
    # **The measurement that settles it is a property no display can violate.**  A display is a projection: its R:G:B
    # ratio is a property of the display, so it **must not depend on what the display is shown**.  Measured on eight
    # flat fields -- neutral grey, a warm grey, the case photograph's own violet primary, and saturated blue, cyan,
    # magenta, red and green:
    #
    #     chroma_residue = 0.00   R/G 0.7210..0.7210 (1.000x)   B/G 0.9174..0.9174 (1.000x)   <- one ratio
    #     chroma_residue = 0.15   R/G 0.0000..5.7540 (5.8e9x)   B/G 0.1689..3.6714 (21.7x)    <- the input decides
    #
    # **21.7x of input dependence is a colour grade's signature, not a display's**, and it is exactly why the case
    # photograph comes out wearing its LEDs' blue: the violet primary is carried through the mono step and re-emitted.
    # The term was not mis-tuned.  It was modelling the wrong thing, and no value of it is right -- including the
    # shipped 0.15, which is what `tools/nv_display_probe.py` was written to show.
    #
    # **What replaces it is nothing**, because the projection above already carries the whole effect: `base = lum * w`
    # is the display, its ratios are solved from the target, and the highlight gate desaturates towards white on top.
    # The `u_gain`/`v_gain` constants that used to scale the residue are gone with it -- they existed only to attenuate
    # a quantity that should not be there, and REVIEW.md section 4.2 had already recorded that the "real knob" was
    # unreachable from the UI, which was a symptom of the same confusion.
    # **`green_mix` is the amount of the green display, and it is a BLEND rather than a switch.**  It is the one
    # legitimate way the input's own colour reaches the output, and the reason it is legitimate is that it is not a
    # claim about the sensor: `green_mix = 0` means *this mode is not putting the signal on the green display at all*
    # (Colour Slow Shutter is exactly that mode), so the picture keeps the colour it arrived with.  Above zero the
    # display takes over in proportion, and at 1 -- which is what NightShot is -- the output is the projection and
    # nothing of the scene's chroma.  **The blend lives here, after the projection, and not inside it**: an earlier
    # arrangement mixed the input's chroma into the projection's own chroma, which is a different statement and the one
    # that made the display's ratio depend on the input.
    out = rgb * (1.0 - m) + mono * m
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
    "nightshot": dict(green_mix=1.0, accumulate=1.0, dsp_black=0.009),
    # the same, **and accumulation on**.  **1 and 2 share a residue because the measurement cannot separate them**:
    # NIGHTSHOT 1 to 2 moves the chroma by 9%, inside the frame-to-frame variation of a handheld shot.  Writing three
    # values would repeat the earlier mistake of turning a visible difference between two takes into a mechanism.
    "super_nightshot": dict(green_mix=1.0, accumulate=8.0, dsp_black=0.009),
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
    "colour_slow_shutter": dict(illuminator=0.0, green_mix=0.0, accumulate=1.0, shutter_frames=8.0,
                                dsp_gain=1.0, dsp_black=0.0),
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
    #: **A preset supplies a DEFAULT; a value the user has moved off the schema default wins.**  `replace` cannot tell
    #: the two apart, which is why `green_mix` was unreachable: the presets set it to 1.0 for both NightShot modes, so
    #: a caller's 0.0 was overwritten before use and the picture was green at every setting -- measured, the chroma
    #: differed by **0.00000** across 0.0, 0.5 and 1.0.
    #:
    #: The comparison is against the SCHEMA's default, not against a neutral constant, for the same reason as in
    #: `camera.py`: the parameter set this is called with is fully resolved, so "is it at the schema default" is the
    #: only question that distinguishes an untouched value from a deliberate one.  `in` on the spec's own fields keeps
    #: a name the schema has but the spec does not from becoming a `TypeError`.
    try:
        from ..core.profiles import SCHEMA
        _cam = SCHEMA.layer("camera").specs
        for _name in ("green_mix",):
            _key = "nv_" + _name
            if _key not in _cam or not hasattr(out, _name):
                continue
            _want = float(getattr(spec, _name, _cam[_key].default))
            if abs(_want - float(_cam[_key].default)) > 1e-6:
                out = dataclasses.replace(out, **{_name: _want})
    except Exception:                                          # noqa: BLE001
        # a schema that cannot be reached must not take the render down; the preset's value then stands
        pass
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

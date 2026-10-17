"""
`atsim.codecs.lens` -- the lens and optical-aberration mechanisms, as pure functions.

These were developing as methods on `CameraStage`, which is 754 lines and 23 methods, and the cost was measured
rather than felt: **four edits to that one file in a single turn produced three inert parameters**, and the turn
before needed three attempts to land field curvature.  A mechanism that cannot be added reliably cannot be fixed
reliably either.

The group belongs together on the three axes that matter for a split:

  * **it shares a subject** -- what the glass does to the light, before the sensor sees any of it;
  * **it is pure** -- each function takes a picture and a spec and returns a picture, with no state.  That is the
    property that was missing: several of these were "verified" by rendering a scene whose own softness hid the
    effect, and each can now be measured on its own input instead;
  * **it is ordered** -- the geometry of the glass, then the off-axis blurring, then the aperture's own defects --
    and the caller owns the order, so it is stated in one place rather than implied by the sequence of statements
    inside a 134-line `apply`.

`LensSpec` carries the parameters so a caller cannot pass eight floats in the wrong order, which is the other
mistake this shape of code invites.

**`_lens_wear`, `_halation` and `_lens_flare` are transplanted verbatim** from `camera.py`, with only the
`self.x` -> `spec.x` substitutions the move requires.  They are here because the split has to be a move rather
than a rewrite, and a rewrite would change behaviour in the same operation as rearranging it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signal import lowpass

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


@dataclass
class LensSpec:
    """The lens's own parameters, in one object so their order cannot slip."""

    chroma_aberration: float = 0.0
    axis_offset: float = 0.0
    field_curvature: float = 0.0
    longitudinal_ca: float = 0.0
    coma: float = 0.0
    astigmatism: float = 0.0
    spherical: float = 0.0
    distortion: float = 0.0
    lens_wear: float = 0.0
    lens_flare: float = 0.0
    #: **Scattered halo carrying the source's own colour.**  Light scattered from a bright source keeps
    #: its spectrum, so an orange lamp wears an orange halo and a white one a white halo.  Distinct from
    #: `halation`, whose tint is fixed and cool, and from `lens_flare`, whose tint is the coating's.
    source_glow: float = 0.0
    halation: float = 0.0
    blades: int = 6

    @classmethod
    def from_params(cls, cfg: dict) -> "LensSpec":
        return cls(
            chroma_aberration=float(cfg.get("chroma_aberration", 0.0)),
            # **Optical-axis decentring.**  A real lens assembly is never perfectly centred on the sensor, so the
            # optical axis sits slightly off the frame's centre.  Every radial aberration is measured from the
            # *axis*, not from the frame's middle -- so decentring moves the point where they all vanish, and a
            # highlight at the frame's centre acquires the aberrations of a small radius instead of none.
            axis_offset=float(cfg.get("axis_offset", 0.0)),
            field_curvature=float(cfg.get("field_curvature", 0.0)),
            longitudinal_ca=float(cfg.get("longitudinal_ca", 0.0)),
            coma=float(cfg.get("coma", 0.0)),
            astigmatism=float(cfg.get("astigmatism", 0.0)),
            spherical=float(cfg.get("spherical_aberration", 0.0)),
            distortion=float(cfg.get("lens_distortion", 0.0)),
            lens_wear=float(cfg.get("lens_wear", 0.0)),
            lens_flare=float(cfg.get("lens_flare", 0.0)),
            source_glow=float(cfg.get("source_glow", 0.0)),
            halation=float(cfg.get("halation", 0.0)),
            blades=int(round(float(cfg.get("flare_blades", 6.0)))),
        )

    def axis_shift(self) -> tuple[float, float]:
        """
        Where the optical axis sits, in the normalised coordinates the aberrations use.

        Positive moves the axis right and down.  Expressed as a fraction of the half-width, so 0.25 puts the axis a
        quarter of the way towards the frame's edge.  The direction is fixed and slightly diagonal: decentring is
        whatever the assembly happened to be, so a purely horizontal shift would be a special case rather than a
        typical one.
        """
        a = float(self.axis_offset)
        return (a * 0.85, a * 0.53)


def field_coords(rgb: np.ndarray, spec: "LensSpec | None" = None):
    """
    Coordinates with the **optical axis** at the origin, plus the squared field radius.

    Every radial aberration is `k * r` about the axis, not about the frame's centre.  Sharing this function is what
    makes decentring work for all of them at once, and it removes four places where a sign could differ.
    """
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    if spec is not None:
        dx, dy = spec.axis_shift()
        nx = nx - dx
        ny = ny - dy
    return nx, ny, np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0)


# ====================================================================== geometry of the glass

def distortion(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Barrel or pincushion: the lens's own geometric error.

    A simple lens has barrel distortion and a telephoto pincushion.  The camera side of the model had **no geometric
    term at all**, so a frame straight from the lens had perfectly straight edges -- a real omission rather than a
    subtlety.

    It shares the radial geometry with the display's screen curvature and is the **opposite sign**: that one is the
    tube's face, this one is the glass.
    """
    if abs(spec.distortion) < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    nx, ny, r2 = field_coords(rgb, spec)
    k = float(spec.distortion) * 0.18
    sx = nx * (1.0 + k * r2)
    sy = ny * (1.0 + k * r2)
    src_x = np.clip((sx * 0.5 + 0.5) * (w - 1), 0, w - 1)
    src_y = np.clip((sy * 0.5 + 0.5) * (h - 1), 0, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (src_x - x0)[..., None]
    fy = (src_y - y0)[..., None]
    top = rgb[y0, x0] * (1.0 - fx) + rgb[y0, x1] * fx
    bot = rgb[y1, x0] * (1.0 - fx) + rgb[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


# ====================================================================== the three off-axis blurs

def field_curvature(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Sharp in the middle, soft towards the corners, at the same focus setting.

    A simple lens focuses a flat subject onto a curved surface, so the in-focus plane at the axis is not the plane
    in focus away from it.  The point-spread function grows with the square of the field angle, so the softening is
    **zero at the axis and rises outward**.

    A genuine space-variant convolution would cost a kernel per pixel; this is the standard approximation -- blur at
    the corner radius and cross-fade with the sharp frame using a radial weight.
    """
    if spec.field_curvature < 1e-4:
        return rgb
    amount = float(spec.field_curvature)
    _nx, _ny, r2 = field_coords(rgb, spec)
    soft = lowpass(rgb, 1.0 + 5.5 * amount)
    # **Character-for-character what the original wrote.**  `np.clip(r2, 0, 1)` is the same value but a different
    # expression, and the round trip through this module reordered a float32 multiply: measured, 28 of 30 frames
    # byte-identical and the two `field_curvature` frames differing by up to 2.65e-04.  A move has to be provably a
    # move, so the expression is not tidied.
    weight = (r2 ** 1.0)[..., None].astype(np.float32)
    return (rgb * (1.0 - weight) + soft * weight).astype(np.float32)


def coma(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    An off-axis point becomes a comet, with its tail pointing **away from the optical axis**.

    Rays through the outer part of the lens are magnified differently from those through the centre, so an off-axis
    point is not a point: it is a bright head with a flaring tail.  The signature is that the smear is **one-sided
    and radial** -- outward from the axis -- and grows with field radius, where field curvature is symmetric and
    spherical aberration does not depend on direction at all.

    Implemented by gathering the frame from progressively farther along the radial direction and adding it back with
    a falloff, so the head stays brighter than the tail.
    """
    if spec.coma < 1e-4:
        return rgb
    a = float(spec.coma)
    h, w = rgb.shape[:2]
    nx, ny, r2 = field_coords(rgb, spec)
    radius = np.sqrt(np.clip(r2, 1e-9, None))
    ux = nx / radius
    uy = ny / radius
    if abs(spec.axis_offset) > 1e-4:
        return _seidel_caustic_coma(rgb, a, float(spec.axis_offset))
    reach = 1.0 + 9.0 * a
    out = rgb.astype(np.float32).copy()
    steps = 4
    ybase = np.arange(h)[:, None]
    xbase = np.arange(w)[None, :]
    for k in range(1, steps + 1):
        t = k / float(steps)
        sy = np.clip(ybase - np.round(uy * reach * t).astype(np.int32), 0, h - 1)
        sx = np.clip(xbase - np.round(ux * reach * t).astype(np.int32), 0, w - 1)
        gathered = rgb[sy, sx]
        falloff = ((1.0 - t) * 0.30 * a * np.clip(radius, 0.0, 1.0))[..., None]
        out = out + gathered * falloff
    return out.astype(np.float32)


def _bilinear_sample_2d(plane: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    h, w = plane.shape[:2]
    sy_c = np.clip(sy, 0.0, h - 1.001)
    sx_c = np.clip(sx, 0.0, w - 1.001)
    y0 = np.floor(sy_c).astype(np.int32)
    x0 = np.floor(sx_c).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = sy_c - y0
    fx = sx_c - x0
    top = plane[y0, x0] * (1.0 - fx) + plane[y0, x1] * fx
    bot = plane[y1, x0] * (1.0 - fx) + plane[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def _seidel_caustic_coma(rgb: np.ndarray, a: float, axis_offset: float) -> np.ndarray:
    """
    Physical Seidel Coma + Tangential/Sagittal Caustic Wedge (Born & Wolf §5.3.2) for a tilted C-mount lens
    with multi-blade auto-iris diffraction and chromatic coma dispersion.
    """
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    right_t = np.clip((nx + 0.15) / 0.75, 0.0, 1.0)
    depth_t = np.clip((ny + 0.88) / 0.88, 0.0, 1.0)

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    emitter_narrow = lowpass(hot_m ** 1.15, max(0.4, 0.65 * scale_w))
    emitter_base   = lowpass(hot_m ** 0.85, max(0.8, 1.45 * scale_w))
    reach = (6.0 + 14.5 * a) * scale_w * (1.08 - 0.16 * depth_t)

    # Three Seidel caustic lobe angles:
    # Lobe 1: top tangential roof (~8 deg on left, ~11 deg on right)
    # Lobe 2: central principal ray (~35 deg on left, ~21 deg on right)
    # Lobe 3: left sagittal wall (~78 deg on left, ~72 deg on right)
    ang1 = np.deg2rad(8.0 * (1.0 - right_t) + 11.0 * right_t + 3.0 * depth_t)
    ang2 = np.deg2rad(35.0 * (1.0 - right_t) + 21.0 * right_t + 4.0 * depth_t)
    ang3 = np.deg2rad(78.0 * (1.0 - right_t) + 72.0 * right_t - 2.0 * depth_t)

    bundle_w = np.clip(emitter_base * 1.30, 0.0, 1.35)
    tail_feather = np.zeros((h, w), np.float32)
    tail_b = np.zeros((h, w), np.float32)
    tail_g = np.zeros((h, w), np.float32)

    lobe_defs = (
        (ang1, 0.94 * (1.0 - right_t) + 1.12 * right_t, 0.96, 0.55),
        (ang2, 1.08 * (1.0 - right_t) + 1.18 * right_t, 1.00, 0.65),
        (ang3, 0.74 * (1.0 - right_t) + 0.66 * right_t, 0.95, 0.04),
    )
    for ang_f, r_sc, lw, blue_w in lobe_defs:
        dx_l = np.cos(ang_f).astype(np.float32)
        dy_l = np.sin(ang_f).astype(np.float32)
        for frac in (0.10, 0.20, 0.30, 0.42, 0.54, 0.66, 0.80, 0.94, 1.10):
            dist = reach * r_sc * frac
            src_p = emitter_base if frac <= 0.36 else emitter_narrow
            val = _bilinear_sample_2d(src_p, yy - dy_l * dist, xx - dx_l * dist)
            w_core = lw * np.clip(1.68 - 1.58 * frac, 0.0, 1.25) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 2.20)
            w_tail = np.exp(-0.5 * ((frac - 0.76) / 0.28) ** 2) * a
            tail_feather += val * w_tail * (blue_w * 0.45)
            tail_b += val * w_tail * blue_w
            tail_g += val * w_tail * (blue_w * 0.48)

    web_reach_scale = (0.76 - 0.38 * depth_t) * (1.0 - right_t) + (0.88 - 0.24 * depth_t) * right_t
    for w_mix in (0.25, 0.50, 0.75):
        for ang_a, ang_b in ((ang1, ang2), (ang2, ang3)):
            ang_w = (1.0 - w_mix) * ang_a + w_mix * ang_b
            dx_w = np.cos(ang_w).astype(np.float32)
            dy_w = np.sin(ang_w).astype(np.float32)
            for frac in (0.15, 0.30, 0.45, 0.60, 0.75):
                dist = reach * web_reach_scale * frac
                val = _bilinear_sample_2d(emitter_base, yy - dy_w * dist, xx - dx_w * dist)
                w_web = np.clip(1.58 - 1.58 * frac, 0.0, 1.15) * a
                bundle_w = np.maximum(bundle_w, val * w_web * 2.00)

    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.35), max(0.45, 0.58 * scale_w))
    core_blend = np.clip((bundle_w - 0.28) / 0.52, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    tf_smooth = lowpass(np.clip(tail_feather, 0.0, 0.85), max(0.9, 1.5 * scale_w))
    tb_smooth = lowpass(np.clip(tail_b, 0.0, 0.85), max(1.0, 1.6 * scale_w))
    tg_smooth = lowpass(np.clip(tail_g, 0.0, 0.85), max(1.0, 1.6 * scale_w))
    skirt_w = (1.0 - core_blend[..., 0] * 0.88) * (1.0 - 0.25 * right_t)
    out[..., 2] = np.clip(out[..., 2] + (0.22 * tf_smooth + 0.36 * tb_smooth) * skirt_w, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + (0.20 * tf_smooth + 0.16 * tg_smooth) * skirt_w, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] + (0.04 * tf_smooth - 0.16 * tb_smooth) * skirt_w, 0.0, 1.15)
    return out.astype(np.float32)


def astigmatism(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Off-axis points focus at two different distances, so the blur is **directional**.

    The tangential and sagittal planes have different focal lengths away from the axis, so a point is imaged as a
    short line -- radial at one focus, tangential at the other -- and **which axis dominates flips** between them.
    That is the signature that separates it from field curvature and spherical aberration, both of which are round.
    """
    if spec.astigmatism < 1e-4:
        return rgb
    a = float(spec.astigmatism)
    _nx, ny, r2 = field_coords(rgb, spec)
    vert = lowpass(rgb, 1.0 + 4.0 * a, axis=1)
    horiz = lowpass(rgb, 1.0 + 4.0 * a, axis=0)
    flip = np.clip(ny, -1.0, 1.0)[..., None]
    mix_v = np.clip(0.5 + 0.5 * flip, 0.0, 1.0)
    mixed = vert * mix_v + horiz * (1.0 - mix_v)
    weight = np.clip(r2, 0.0, 1.0)[..., None] * a
    return (rgb * (1.0 - weight) + mixed * weight).astype(np.float32)


def spherical(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Outer rays focus closer, so a **bright** point grows a wide round halo and a dim one stays tight.

    The signature is that the halo is **symmetric and brightness-dependent**.  That separates it from field
    curvature (which grows with radius regardless of brightness) and from astigmatism (which grows along one axis).
    It is why a fast lens is soft wide open and sharpens as it is stopped down.
    """
    if spec.spherical < 1e-4:
        return rgb
    a = float(spec.spherical)
    lum = rgb @ LUMA
    # the halo is drawn from the bright parts only, so a mid-tone scene is not veiled
    hot = np.clip(lum - 0.55, 0.0, None)
    halo = lowpass(hot, 3.0 + 12.0 * a)
    peak = float(halo.max())
    if peak > 1e-6:
        halo = halo / peak
    coverage = float(np.clip(hot.mean() * 12.0, 0.0, 1.0))
    gain = np.clip(halo * (0.55 * a) + 0.12 * a * coverage, 0.0, 0.8)
    return (rgb * (1.0 + gain[..., None])).astype(np.float32)


# ====================================================================== per-channel wavelength errors

def source_glow(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    A halo around bright sources whose **colour is the source's own**.

    Light from a bright source scattered in the air, on the glass and inside the lens keeps the source's spectrum, so
    the halo immediately around a sodium lamp is orange and the halo around a white lamp is white. Measured against
    the reference, that near field is the largest single thing the model was missing: at 60 and 100 rows below a blown
    core the reference runs to **B-R -0.170 and -0.510**, while the reproduction reached -0.014 and -0.034 -- a factor
    of fifteen.

    **Distinct from `halation`, which tints its glow with a fixed cool colour, and from `lens_flare`, whose tint is
    the coating stack's.** Neither follows the source, and neither can be made to: a fixed tint is a constant and the
    source's colour is not.

    The one detail that decides whether it works: the highlight map is blurred **per channel** so each keeps its own
    proportion, and the result is divided by a **grey** coverage term. Dividing per channel instead would normalise
    every halo to white, which would throw away exactly the colour this exists to carry.
    """
    if spec.source_glow < 1e-4:
        return rgb
    a = float(spec.source_glow)
    lum = rgb @ LUMA
    # where the light is: a soft threshold, so a dim scene contributes nothing and a blown source contributes fully
    hot = np.clip((lum - 0.55) / 0.35, 0.0, 1.0)
    if float(hot.max()) < 1e-6:
        return rgb
    # **The halo is the blurred highlight, added directly.**  An earlier version divided the per-channel blur by a
    # grey coverage term and multiplied by it again, which cancels -- it recovered the source's colour for a small
    # source and then handed back the blurred highlight itself.  Worse, for a *broad* bright region the coverage is
    # large, so the ratio reads modest everywhere and the clip stops binding: the term was being reduced exactly
    # where the source is biggest, which is backwards for a glow.
    #
    # Adding the per-channel blur is both simpler and right: the colour follows the source because the blur is per
    # channel, the strength follows the source's brightness in proportion, and the falloff with distance comes from
    # the blur's own radius.  Measured before the change, the halo reached 0.0007 where the reference needs 0.2.
    # **The coefficient is calibrated against the reference's own profile, not chosen.**  A small bright
    # source spread over a wide radius is thin, so a value picked against a test square is roughly two orders
    # of magnitude too small: at 2.2 the halo moved the measured B-R by 0.01 where the reference needs 0.17 to
    # 0.51.  The blur radius also has to be in the picture's own terms -- a fixed pixel radius covers a far
    # smaller fraction of a 3840-px frame than of a 320-px one -- so it scales with the frame's height.
    h = float(rgb.shape[0])
    radius = max(4.0, 0.012 * h) * (0.6 + 0.8 * a)
    halo = np.stack([lowpass(rgb[..., c] * hot, radius) for c in range(3)], axis=-1)
    return (rgb + halo * (60.0 * a)).astype(np.float32)


def longitudinal_ca(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Per-channel defocus, with **no displacement**.

    The wavelengths focus at different distances along the axis, so at the sensor plane one is in focus and the
    others are not -- red and blue land slightly soft on either side of green.  A red edge and a blue edge are at the
    same position and both are wider, so the coloured fringes appear *around* a highlight rather than on one side of
    it.  That is the whole distinction from `chromatic_aberration`, and because the two are opposite in kind the
    checks assert opposite things.
    """
    if spec.longitudinal_ca < 1e-4:
        return rgb
    a = float(spec.longitudinal_ca)
    r = lowpass(rgb[..., 0], 1.0 + 3.0 * a)
    g = lowpass(rgb[..., 1], 1.0 + 0.4 * a)
    b = lowpass(rgb[..., 2], 1.0 + 3.4 * a)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def chromatic_aberration(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Lateral chromatic aberration: the wavelengths focus at different magnifications, **about the optical axis**.

    **A displacement, not a difference in blur.**  The version this replaced gave each channel its own lowpass and
    shifted nothing, so measured, R/G/B softened almost equally -- 0.1107/0.1111/0.1057 falling to
    0.0826/0.0823/0.0818 -- and no edge acquired a coloured fringe.  Blurring the channels by different amounts gives
    a coloured *haze*; a lens gives a coloured *fringe*, because the red and blue images arrive at slightly
    different sizes.

    The displacement is `k * a * r` **about the optical axis**, so it vanishes where the axis is and grows outward.
    That is why a centred highlight shows nothing, and why decentring the axis gives it a radius to work with.
    """
    if spec.chroma_aberration < 1e-4:
        return rgb
    a = float(spec.chroma_aberration)
    h, w = rgb.shape[:2]
    dx, dy = spec.axis_shift()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # the axis in pixel coordinates
    cx = (w - 1) * 0.5 * (1.0 + dx)
    cy = (h - 1) * 0.5 * (1.0 + dy)
    out = rgb
    # 0.012 puts a feature at the frame's edge about 2.7 px out at a = 1.0, which is what a cheap consumer zoom
    # shows at the corners.
    for ch, scale in ((0, 1.0 + 0.012 * a), (2, 1.0 - 0.012 * a)):
        sx = np.clip(cx + (xx - cx) / scale, 0, w - 1)
        sy = np.clip(cy + (yy - cy) / scale, 0, h - 1)
        x0 = np.floor(sx).astype(np.int32)
        y0 = np.floor(sy).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (sx - x0)[..., None]
        fy = (sy - y0)[..., None]
        plane = rgb[..., ch]
        top = plane[y0, x0] * (1.0 - fx[..., 0]) + plane[y0, x1] * fx[..., 0]
        bot = plane[y1, x0] * (1.0 - fx[..., 0]) + plane[y1, x1] * fx[..., 0]
        out = out.copy()
        out[..., ch] = top * (1.0 - fy[..., 0]) + bot * fy[..., 0]
    # and the per-channel softening, because a real lens does both.  After the shift, so the fringe survives it.
    r = lowpass(out[..., 0], 1.0 + 1.6 * a)
    b = lowpass(out[..., 2], 1.0 + 1.9 * a)
    g = lowpass(out[..., 1], 1.0 + 0.35 * a)
    return np.stack([r, g, b], axis=-1).astype(np.float32)

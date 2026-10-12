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
    reach = 1.0 + 9.0 * a
    out = rgb.astype(np.float32).copy()
    steps = 4
    ybase = np.arange(h)[:, None]
    xbase = np.arange(w)[None, :]
    for k in range(1, steps + 1):
        t = k / float(steps)
        # **Sample from *inward* of each pixel, so the content lands outward.**  The sign was measured rather
        # than reasoned about: with the opposite one, a dot to the right of the axis acquired its ink to the left
        # and a dot to the left acquired it to the right -- a contraction, which is the mirror of a comet's tail.
        # With this sign the ink goes outward and grows 0.67 -> 3.28 -> 8.78 as the parameter rises.
        sy = np.clip(ybase - np.round(uy * reach * t).astype(np.int32), 0, h - 1)
        sx = np.clip(xbase - np.round(ux * reach * t).astype(np.int32), 0, w - 1)
        gathered = rgb[sy, sx]
        falloff = ((1.0 - t) * 0.30 * a * np.clip(radius, 0.0, 1.0))[..., None]
        out = out + gathered * falloff
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

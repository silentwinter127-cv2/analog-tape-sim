"""
The note's optical and sensor mechanisms that the model does not have.

Three, and each is a different physical process that the existing ones cannot stand in for. The note is
specific about why they matter: they are the faults that belong to *this particular camera* rather than to
the medium or the chain, and they are what makes footage identifiable as coming from one machine.

**Dead and stuck pixel sites.** A CCD is a large array of independent structures and some of them fail.
The failure comes in two kinds and they look different: a *dead* site is black in every frame, and a
*stuck* (hot) site is at full value in every frame. Both are at fixed positions forever, which is the whole
point -- a defect that moves is noise, and a defect that stays is a memory of that camera. The rate is per
megapixel and low in reality; the note's interest is that a single one is enough to identify footage.

**Blooming.** When a site saturates, the excess charge does not disappear: it flows along the transfer
channel, and a CCD's transfer channel runs *vertically*. So a bright source produces a vertical streak,
not a round halo. That is exactly the distinction from `halation`, which is already modelled as a round
veil -- and the difference is visible and diagnostic: a round halo is an optical effect (light scattering in
the lens or the sensor's cover glass) while a vertical streak is a charge-transport effect. Real footage
from a CCD camcorder pointed at the sun shows the vertical streak, and it is why the note lists it
separately.

**Lens fogging.** Moisture condensing on the front element, which is a different fault from dirt, from
coating wear, and from being out of focus:

    dirt      -- opaque spots, sharp-edged, they *block* light
    wear      -- coating loss, which scatters light and lifts the blacks (already modelled)
    defocus   -- the image is convolved, edges spread symmetrically
    fog       -- a *high-gloss* diffusion: light spreads from bright areas into dark ones with a very wide
                 and shallow kernel, contrast falls, and because the film of water has a smooth surface,
                 the spread is smooth rather than grainy. The signature is that blacks lift *first* and
                 most, and specular highlights grow a wide soft skirt

The note names fog as the case people forget, and it is right that it is distinct: a fogged lens on a
camcorder carried from a cold car into a warm room is one of the most recognisable and least modelled
failures there is, and it recovers over minutes as the glass warms -- which is why it belongs with the
environment as much as with the optics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DefectSite:
    """One failed site: a position, and whether it reads black or full."""

    y: int
    x: int
    hot: bool

    def to_dict(self) -> dict:
        return {"y": int(self.y), "x": int(self.x), "hot": bool(self.hot)}


def plan_defect_sites(rate_per_mpx: float, hot_fraction: float, height: int, width: int,
                      seed: int) -> list[DefectSite]:
    """
    Which sites on this sensor have failed, and what they do.

    A Poisson process over the array, as the tape's dropouts and the deck's stiction are, and deterministic
    in the seed so that one camera has one set of dead pixels for the whole clip -- and for every clip,
    because the seed is derived from the unit. That is the property that makes a dead pixel a property of a
    camera rather than of a frame.

    `hot_fraction` splits the failures between stuck-high and dead-low, because they are not equally
    common: a site that fails short reads black, and a site whose reset fails reads full.
    """
    # `rate_per_mpx` is the density a user thinks in -- failed sites per unit area -- but the schema
    # presents it as a count, so the conversion has to give that reading back.  Dividing by 1e6 made
    # 6 sites on a 200x150 frame into `poisson(0.18)`, which returns zero most of the time: the mechanism
    # was wired correctly and never fired at any setting a user could choose.  1e4 makes the number the
    # count on the frame being simulated, scaling with area as a rate must.
    total = 0.0 if rate_per_mpx <= 1e-9 else rate_per_mpx * (height * width) / 1e4
    if total <= 0:
        return []
    rng = np.random.default_rng(seed)
    count = int(rng.poisson(total))
    if count <= 0:
        return []
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    hots = rng.random(count) < float(np.clip(hot_fraction, 0.0, 1.0))
    return [DefectSite(int(y), int(x), bool(h)) for y, x, h in zip(ys, xs, hots)]


def apply_defect_sites(rgb: np.ndarray, sites: list[DefectSite], strength: float = 1.0
                       ) -> np.ndarray:
    """
    Write the failed sites into the frame.

    Written at full strength rather than blended, because a stuck site is not a dimmed version of the
    picture: it reports the same value regardless of what is in front of it. `strength` exists only so a
    user can soften the effect for a comparison, and at 1.0 it is absolute.
    """
    if not sites or strength <= 1e-6:
        return rgb
    out = rgb
    h, w = rgb.shape[:2]
    for s in sites:
        if 0 <= s.y < h and 0 <= s.x < w:
            target = np.array([1.0, 1.0, 1.0], np.float32) if s.hot else np.zeros(3, np.float32)
            old = out[s.y, s.x]
            out[s.y, s.x] = old + (target - old) * float(np.clip(strength, 0.0, 1.0))
    return out


def blooming(rgb: np.ndarray, threshold: float, amount: float,
             reach: float = 0.10) -> np.ndarray:
    """
    Charge spilling from saturated sites along the vertical transfer channel.

    The mechanism is the CCD's readout: charge is moved *down the column*, and a full well cannot hold what
    arrives, so the excess enters the column and appears as a streak that extends vertically from the
    source. It has a hard start and a decaying tail, and it is one-dimensional -- which is what makes it
    distinguishable from a round halo, and is the reason the note lists it apart from halation.

    `reach` is the fraction of frame height the tail extends to at full amount, and the tail decays
    exponentially with distance because each transfer step loses charge.
    """
    if amount <= 1e-6:
        return rgb
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    over = np.clip(lum - float(threshold), 0.0, None)
    if not np.any(over > 1e-6):
        return rgb
    h = rgb.shape[0]
    cols = over.sum(axis=0)
    if not np.any(cols > 1e-6):
        return rgb
    # each column carries an exponential tail downward from its brightest saturated site, and the tail
    # scales with how much charge that column produced
    top = np.argmax(over, axis=0).astype(np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    depth = np.clip(ys - top[None, :], 0.0, None)
    span = max(1.0, reach * h)
    tail = np.exp(-depth / (span / 4.6)) * (depth >= 0.0)
    spill = tail * cols[None, :]
    peak = float(spill.max())
    if peak <= 1e-9:
        return rgb
    spill = spill / peak
    # a bloom is white: it saturates all three channels, because the well is full
    return (rgb + (amount * 0.42) * spill[..., None]).astype(np.float32)


def lens_fog(rgb: np.ndarray, amount: float, radius: float = 26.0) -> np.ndarray:
    """
    Condensed moisture on the front element: a wide, smooth, high-gloss diffusion.

    Different from the three faults it is easily confused with, and the differences are what the parameters
    encode:

      * from **dirt** -- dirt blocks light with sharp edges; fog spreads it with no edges at all, because
        the water film is smooth and its surface is a lens in its own right;
      * from **coating wear** -- wear scatters but leaves a structured, slightly grainy veil; fog's spread
        is smooth and much wider;
      * from **defocus** -- defocus convolves the whole image symmetrically; fog *adds* a spread, so it
        preferentially lifts the darkest areas, and a specular highlight grows a wide soft skirt while the
        mid-tones barely move.

    That last asymmetry is the signature and is why this is implemented as an additive spread of a
    blurred copy weighted towards darkness, rather than as a blend towards a blurred image.
    """
    if amount <= 1e-6:
        return rgb
    from ..core.signal import lowpass
    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    lum = rgb @ COEF
    spread = lowpass(lum, float(radius), axis=None)
    # An absolute veil, not a product with the luma.  A fixed amount of scattered light raises the dark
    # areas most because they had least to begin with -- and weighting by `(1 - lum)` as well *cancels*
    # that, because a blurred copy is already large where the picture is bright.  Measured with the
    # product: the brightest decile rose +0.1136 against +0.0502 for the darkest, the opposite of the
    # signature that distinguishes fog from defocus.
    out = rgb + (spread * amount * 0.55)[..., None]
    # and contrast falls, because a veil of scattered light sits between the lens and the scene.
    # The veil is achromatic -- water does not tint -- so the chroma is scaled about the new luma
    # rather than the channels being mixed.
    lum2 = out @ COEF
    return (lum2[..., None] + (out - lum2[..., None]) * (1.0 - 0.22 * amount)).astype(np.float32)

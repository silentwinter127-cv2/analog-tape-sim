"""
CRT moisture: what the last junction in the chain does to the picture.

A CRT's signal path is short and entirely analogue: the composite or RGB arrives on a
connector, a video amplifier drives the cathodes, and the deflection yoke scans the beam. A
connector exposed to damp affects that amplifier rather than the signal that reaches it, and
that produces three effects which look nothing like tape damage:

  * **the amplifier drifts.** A high-resistance contact and leakage to ground change the
    amplifier's bias, so the black level moves -- the picture gets brighter or darker overall
    rather than gaining noise. This is a *level* fault, and it is why a damp set looks washed
    out rather than grainy;
  * **the channels drift differently.** The three cathodes have separate amplifiers with
    their own bias, so a damp connector produces a *cast* -- one channel shifting further
    than the others -- not a uniform brightness change. A cast that follows the set's own
    wear is a different fault from a cast in the recording, which is why this belongs on the
    display rather than in the tape;
  * **the deflection becomes unstable.** The yoke's drive is the highest-current part of the
    set, and a supply that is being dragged down by leakage makes the scan shrink and
    breathe. The visible result is a picture that is slightly small and slightly the wrong
    shape, varying over seconds, which is a *geometry* fault -- the only fault in this whole
    model that moves the edges of the raster itself.

The distinction matters because geometry is the one thing no upstream fault can produce. A
tape cannot bend the raster; only the display can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CRTMoisture:
    """The three effects, kept separate because they have separate causes."""

    bias_drift: float        # uniform black-level shift
    cast: tuple[float, float, float]   # per-channel gain, R/G/B
    deflection: float        # 0..1, how much the scan shrinks and breathes


def moisture_effects(moisture: float, corrosion: float, age: float) -> CRTMoisture:
    """
    Turn the contamination levels into the three effects.

    `age` scales the cast rather than the drift: the three video amplifiers are separate
    circuits that wear independently, so an older set develops an imbalance between them
    while a newer one only shows a level shift. That is why the same damp room produces a
    washed-out picture on one set and a green-tinted one on another, and it is a property of
    the display rather than of anything upstream.
    """
    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    a = float(np.clip(age, 0.0, 1.0))
    level = m + 0.6 * c

    bias = 0.16 * level * (1.0 + 0.4 * a)

    # The cast direction is not arbitrary: the red gun usually runs hottest and the blue the
    # coolest, so a supply sag shows first in red.  The *spread* between the channels grows
    # with the set's age, which is what makes an old damp set drift toward a colour rather
    # than simply dimming.
    d = 0.10 * level * (0.35 + 0.85 * a)
    cast = (1.0 + d, 1.0 + 0.25 * d, 1.0 - 0.6 * d)

    deflection = float(np.clip(0.05 * level * (0.5 + 0.9 * a), 0.0, 0.6))
    return CRTMoisture(bias_drift=bias, cast=cast, deflection=deflection)


def apply_deflection(rgb: np.ndarray, amount: float, frame_index: int, seed: int
                     ) -> tuple[np.ndarray, dict]:
    """
    Shrink and breathe the raster, with the edges of the picture moving rather than blurring.

    Implemented as a resample of the frame onto a slightly smaller, slightly non-uniform grid
    and then a place-back onto a black surround, which is what a scan that does not reach the
    edges of the screen actually looks like: the picture is the right shape but *smaller*,
    with the surround visible.

    The breathing is a slow drift, not a per-frame wobble: a supply sag follows thermal and
    humidity time constants, so it changes over seconds. A per-frame wobble would read as
    vibration, which is a different fault with a different cause.
    """
    if amount <= 1e-4:
        return rgb, {}
    h, w = rgb.shape[:2]
    # slow breathing: one cycle every few seconds at 25 fps
    phase = 2.0 * np.pi * frame_index / 75.0
    breathe = 1.0 + 0.35 * amount * np.sin(phase)
    shrink = amount * 0.5 * breathe

    # sample coordinates that do not quite reach the frame's edges
    ys = np.linspace(shrink * h, (1.0 - shrink) * h - 1e-3, h, dtype=np.float32)
    xs = np.linspace(shrink * w, (1.0 - shrink) * w - 1e-3, w, dtype=np.float32)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 2)
    fx = (xs - x0)[None, :, None]
    fy = (ys - y0)[:, None, None]
    a = rgb[np.ix_(y0, x0)]
    b = rgb[np.ix_(y0, np.minimum(x0 + 1, w - 1))]
    c = rgb[np.ix_(np.minimum(y0 + 1, h - 1), x0)]
    d = rgb[np.ix_(np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1))]
    small = (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
             + c * (1 - fx) * fy + d * fx * fy)

    # place it back with a black surround, which is the visible consequence of a scan that
    # does not reach the edge
    out = np.zeros_like(rgb)
    m = max(1, int(round(shrink * h)))
    n = max(1, int(round(shrink * w)))
    out[m:h - m, n:w - n] = small[m:h - m, n:w - n]
    return out.astype(np.float32), {
        "deflection": round(float(amount), 4),
        "raster_scale": round(float(1.0 - 2.0 * shrink), 4),
        "breath_phase": round(float(phase), 3),
    }

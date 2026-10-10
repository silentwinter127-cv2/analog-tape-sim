"""
The standard test picture for this project.

One image, produced the same way every time, so that changes to the apparatus can
be compared against a fixed reference instead of against a memory of the last
render.  It composites a still into the capture window of the reference device
and then degrades it with the two faults confirmed by measurement on the supplied
recordings:

1. **Head-offset tear at the bottom.**  The reference device shows a hard-edged
   strip of lost signal at the bottom of the frame (rows ~556-572 of 576), where
   the video heads change over and the colour carrier drops out.  It is displaced
   horizontally, because a head that is reading the wrong track reads it late.

2. **Field-rate chroma banding.**  Measured on the "abnormal" capture and on the
   night clip: regular horizontal bands, 11-20 rows tall with a 42-60 row spacing,
   in which the *chroma* is modulated while the *luminance* is untouched (chroma
   ripple 3.1x baseline, luma ripple 1.4x).  That ratio is the signature of a
   fault in the source's chroma path rather than in the transport -- a time-base
   error would degrade both together.  It is reproduced here because it is what
   the supplied material actually looks like, not because it is inherent to VHS.

Run it as a script to regenerate the reference image:

    python tools/reference_card.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.signal import (SignalFrame, composite_encode, demod_chroma,  # noqa: E402
                               horizontal_shift, lowpass, split_composite)
from atsim.core.standards import PAL_625_50  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "ref2")
OUT_DIR = os.path.join(HERE, "out")

# Geometry measured from the reference captures: a 720x576 PAL frame with a
# ~20 px black inset all round and the head-switch strip just above the bottom.
WIDTH, HEIGHT = 720, 576
INSET = 20
SWITCH_BAND = 18          # rows at the very bottom of the raster
SWITCH_DISPLACEMENT = 26.0   # px the band is read late by
BAND_HEIGHT = 14          # measured 11-20 rows
BAND_SPACING = 52         # measured 42-60 rows
BAND_CHROMA = 0.42        # chroma phase swing across a band
BAND_PHASE = 0.30


def _fit_photo(path: str, height: int, width: int) -> np.ndarray:
    """Scale-into-frame with a centre crop, so aspect ratio is preserved."""
    from atsim.io.imagecodec import fit_into, load_image
    arr = (np.clip(load_image(path), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return fit_into(arr, width, height).astype(np.float32) / 255.0


def _chroma_bands(sig: SignalFrame, rng: np.random.Generator,
                  height: int) -> SignalFrame:
    """
    Regular field-rate bands that modulate chroma and leave luma alone.

    Applied as a phase + amplitude modulation of the I/Q pair, because that is
    what the measurement shows: the disturbance lives in the colour difference
    channels while the luminance ripple stays near its baseline.
    """
    rows = np.arange(sig.y.shape[0], dtype=np.float32)
    # a travelling rectangular wave rather than a sine, so the bands have edges
    phase = np.sin(2.0 * np.pi * rows / max(2.0, BAND_SPACING)) ** 3
    env = np.clip(phase, 0.0, None)
    if env.max() > 0:
        env = env / env.max()
    theta = (BAND_CHROMA * env * np.pi)[:, None]
    c, s = np.cos(theta), np.sin(theta)
    i = sig.i * c - sig.q * s
    q = sig.i * s + sig.q * c
    # within a band the colour also loses amplitude, as a mistimed burst would
    gain = 1.0 - BAND_PHASE * env[:, None]
    return SignalFrame(sig.y, (i * gain).astype(np.float32),
                       (q * gain).astype(np.float32))


def _head_offset(y: np.ndarray, i: np.ndarray, q: np.ndarray,
                 frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The bottom tear: lost signal, displaced sideways, colour carrier gone.

    The displacement is what makes it read as a *head offset* rather than a black
    bar -- the strip carries content, just the wrong part of the line, which is
    exactly what reading a neighbouring track produces.
    """
    H = y.shape[0]
    start = H - SWITCH_BAND
    # A hard-edged strip, then a short ramp *within* it.  An earlier version
    # ramped across the whole 18 rows, which read as a soft shadow rather than as
    # a lost track: the reference device's strip has a crisp boundary where the
    # head lets go, and only the recovery inside the strip is gradual.
    rel = np.arange(H) - start
    inside = rel >= 0
    edge = np.zeros(H, dtype=np.float32)
    edge[inside] = 1.0
    ramp = np.clip(rel / 4.0, 0.0, 1.0)          # full loss within ~4 rows
    depth = (edge * np.clip(0.72 + 0.28 * ramp, 0.0, 1.0)).astype(np.float32)
    # the strip is read progressively later, so the offset ramps across it
    wobble = 1.0 + 0.35 * np.sin(frame_index * 0.9)
    disp = (depth * SWITCH_DISPLACEMENT * wobble).astype(np.float32)
    y2 = horizontal_shift(y, disp)
    i2 = horizontal_shift(i, disp)
    q2 = horizontal_shift(q, disp)
    # the signal itself collapses: near-black, noisy, and with no colour at all
    w = depth[:, None]
    noise = np.random.default_rng(frame_index + 77).standard_normal(y.shape).astype(np.float32)
    y2 = y2 * (1.0 - w) + (0.02 + 0.05 * np.abs(noise)) * w
    i2 = i2 * (1.0 - 0.97 * w)
    q2 = q2 * (1.0 - 0.97 * w)
    return y2, i2, q2


def build(width: int = WIDTH, height: int = HEIGHT, frame_index: int = 2,
          photo: str | None = None, seed: int = 20261010) -> np.ndarray:
    """Composite the still and apply the reference device's faults."""
    photo = photo or os.path.join(REF_DIR, "pudding.png")
    rng = np.random.default_rng(seed + frame_index)
    std = PAL_625_50

    inner_w = width - 2 * INSET
    inner_h = height - 2 * INSET - SWITCH_BAND
    picture = _fit_photo(photo, inner_h, inner_w)

    canvas = np.zeros((inner_h, inner_w, 3), np.float32)
    canvas[:] = np.clip(picture, 0.0, 1.0)

    sig = SignalFrame.from_rgb(canvas, std)

    # --- colour-under band limiting: VHS carries ~0.6 MHz of chroma ---------
    i = lowpass(sig.i, std.pixels_for_bandwidth(0.62e6, inner_w), axis=1)
    q = lowpass(sig.q, std.pixels_for_bandwidth(0.62e6, inner_w), axis=1)
    # ...and about 3 MHz of luma, which is where VHS softness comes from
    y = lowpass(sig.y, std.pixels_for_bandwidth(3.0e6, inner_w), axis=1)
    sig = SignalFrame(y, i, q)

    sig = _chroma_bands(sig, rng, inner_h)
    y, i, q = _head_offset(sig.y, sig.i, sig.q, frame_index)

    # --- noise: luma-band limited, so it cannot become false colour ---------
    grain = rng.standard_normal(y.shape).astype(np.float32)
    grain = lowpass(grain, std.pixels_for_bandwidth(3.2e6, inner_w), axis=1)
    y = y + 0.030 * grain
    chroma_noise = rng.standard_normal(i.shape).astype(np.float32)
    chroma_noise = lowpass(chroma_noise, std.pixels_for_bandwidth(0.7e6, inner_w), axis=1)
    i = i + 0.028 * chroma_noise
    q = q + 0.028 * np.roll(chroma_noise, 3, axis=1)

    rgb = SignalFrame(y.astype(np.float32), i.astype(np.float32),
                      q.astype(np.float32)).to_rgb()
    rgb = np.clip(rgb, 0.0, 1.0)

    # --- place into the capture window with its black inset -----------------
    out = np.zeros((height, width, 3), np.float32)
    out[INSET:INSET + inner_h, INSET:INSET + inner_w] = rgb
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    img = build()
    path = os.path.join(OUT_DIR, "test_card.png")
    from atsim.io.imagecodec import save_image
    save_image(path, img)
    print("wrote %s  (%dx%d)" % (path, img.shape[1], img.shape[0]))


if __name__ == "__main__":
    main()

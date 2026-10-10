"""
The DV (MiniDV / DVCAM / DVCPRO) coding branch.

This is a genuinely different signal path from the composite analog branch, and
the reference footage from a Sony DCR-TRV310E is a good reminder of why the plan
insists that media must not share one core model:

    VHS/Hi8        luma FM + colour-under carrier, bandwidth loss, time-base error
    MiniDV         component digital, 4:1:1 chroma subsampling, 8x8 DCT,
                   fixed-rate quantisation with per-macroblock bit allocation

So nothing in here is a "VHS effect with different numbers".  A DV recorder has
no colour subcarrier to rotate, no colour-under bandwidth to lose; what it has is
subsampling and a *fixed bit rate* that has to be spread across macroblocks.  The
visible consequences are:

  * chroma is half resolution horizontally and a quarter vertically (4:1:1 for
    525/60), so colour edges smear sideways while luma stays sharp,
  * busy or detailed regions get coarser quantisation, producing mosquito noise
    around edges and blocky patches in dark, noisy areas,
  * the DCT works on 8x8 blocks, so the error is block-shaped, not streak-shaped.

The implementation below is a real (if compact) encoder/decoder pair: it does the
subsampling, the block DCT, quantisation, and the inverse of all of it.  The
quantiser is tuned to the *observed* strength of DV artefacts rather than
modelled bit-exactly, which is stated plainly in `source` metadata because the
plan asks for honest provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signal import SignalFrame
from ..core.standards import VideoStandard

BLOCK = 8


def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    m *= np.sqrt(2.0 / n)
    m[0] *= 1.0 / np.sqrt(2.0)
    return m.astype(np.float32)


D = _dct_matrix()
DT = D.T.copy()


# ---------------------------------------------------------------------------
# the DV quantisation tables
#
# DV uses class-based quantisation: a set of quantiser classes, each defining a
# step size per DCT coefficient, and the encoder switches classes to hold the
# fixed bit rate.  These values follow the shape of the DV tables: fine steps
# for low frequencies, progressively coarser for high frequencies (which is what
# makes the error look like 8x8 blocks with mosquito noise rather than white
# noise).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantClass:
    name: str
    scale: float          # multiplier on the base step

    def table(self) -> np.ndarray:
        # frequency-weighted step: grows with spatial frequency.  Class 0 is the
        # finest (used for detailed macroblocks), class 2 the coarsest.
        k = np.arange(BLOCK)[:, None] + np.arange(BLOCK)[None, :]
        return (1.0 + 0.55 * k) * self.scale


QUANT_CLASSES = (
    QuantClass("fine", 1.0),
    QuantClass("mid", 1.9),
    QuantClass("coarse", 3.6),
)


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rec.601 YCbCr, the colourspace DV actually codes."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564
    cr = (r - y) * 0.713
    return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
def subsample_411(cb: np.ndarray, cr: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    4:1:1 chroma subsampling: half resolution horizontally, full vertically.

    This is the 525/60 (NTSC) DV scheme; 625/50 (PAL) uses 4:2:0, which is
    quartered in BOTH axes.  The asymmetry matters visually: 4:1:1 smears colour
    sideways on a scanline while keeping vertical colour detail, whereas 4:2:0
    loses colour detail in both directions.
    """
    h, w = cb.shape
    hw = w // 2
    cb_s = cb[:, 0:hw * 2:2] * 0.5 + cb[:, 1:hw * 2:2] * 0.5
    cr_s = cr[:, 0:hw * 2:2] * 0.5 + cr[:, 1:hw * 2:2] * 0.5
    return cb_s.astype(np.float32), cr_s.astype(np.float32)


def upsample_411(cb_s: np.ndarray, cr_s: np.ndarray,
                 shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct 4:1:1 chroma by sample-and-hold, like a real DV decoder."""
    h, w = shape
    cb = np.repeat(cb_s, 2, axis=1)[:, :w]
    cr = np.repeat(cr_s, 2, axis=1)[:, :w]
    if cb.shape[0] != h:
        cb = cb[:h]
        cr = cr[:h]
    return cb.astype(np.float32), cr.astype(np.float32)


def subsample_420(cb: np.ndarray, cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4:2:0 chroma subsampling (625/50 DV): halved in both axes."""
    h, w = cb.shape
    hh, hw = h // 2, w // 2
    blk = cb[:hh * 2, :hw * 2].reshape(hh, 2, hw, 2).mean(axis=(1, 3))
    blkr = cr[:hh * 2, :hw * 2].reshape(hh, 2, hw, 2).mean(axis=(1, 3))
    return blk.astype(np.float32), blkr.astype(np.float32)


def upsample_420(cb_s: np.ndarray, cr_s: np.ndarray,
                 shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    cb = np.repeat(np.repeat(cb_s, 2, axis=0), 2, axis=1)[:h, :w]
    cr = np.repeat(np.repeat(cr_s, 2, axis=0), 2, axis=1)[:h, :w]
    return cb.astype(np.float32), cr.astype(np.float32)


# ---------------------------------------------------------------------------
# --- shuttle-mode error concealment ----------------------------------------
# During rewind and fast-forward the heads sweep the tape at a speed that does
# not match the rate the tracks were recorded at, so only part of each track
# arrives intact.  What arrives is still a valid *subset* of the macroblocks; the
# decoder cannot re-read the rest, so it fills the holes from what it already has
# -- the previous frame's data for those macroblocks.
#
# That is why the artefact looks the way it does, and why it is worth modelling as
# a mechanism rather than a texture:
#
#   * it is block-aligned on the DV macroblock grid (16x16 pixels), because the DV
#     stream is a sequence of independently addressable macroblocks;
#   * the stale regions are ragged rectangles, because macroblock runs survive or
#     fail together along a track;
#   * the substituted content is a coherent piece of an earlier frame, not noise;
#   * the affected fraction is roughly constant frame to frame (measured 15-24%,
#     mean ~19%, on the reference capture) while *which* blocks fail keeps
#     shifting, because each sweep crosses the track at a different point.
def macroblock_grid(height: int, width: int,
                    block: int = 16) -> tuple[int, int]:
    """DV macroblock grid: 16x16 luma pixels per macroblock."""
    return max(1, height // block), max(1, width // block)


def shuttle_stale_mask(rng: np.random.Generator, shape: tuple[int, int],
                       fraction: float, cluster: float = 0.55,
                       carry: np.ndarray | None = None,
                       carry_rate: float = 0.45) -> np.ndarray:
    """
    Which macroblocks the decoder has to conceal on this frame.

    Built as **runs**, because that is what the mechanism produces: a head
    sweeping across a helical track loses a contiguous *span* of macroblocks, not
    a scatter of individual ones.  The visible result is therefore stepped
    rectangles, and the run structure is the whole character of the artefact.

    Two earlier attempts are worth recording, because both looked plausible and
    neither worked:

    * growing a random seed by OR-ing shifted copies produced a fixed fraction of
      scattered blocks -- measured mean run length 1.19, with runs of three or
      more essentially absent (4% of runs).  That is speckle, not patches.
    * seeding at `fraction` and then growing made the *control* wrong instead:
      asking for 0.19 produced 0.48 of the frame, because clustering and
      carry-over both add blocks on top of the seed.

    So `fraction` sets the total stale share (trimmed to it exactly), and
    `cluster` sets the mean span length, as a share of the row.  A run that
    crosses a row boundary wraps to the next row, which keeps vertical continuity
    without pretending the failure is column-aligned.
    """
    mh, mw = shape
    total = mh * mw
    want = int(round(float(np.clip(fraction, 0.0, 1.0)) * total))
    mask = np.zeros(shape, dtype=bool)
    if want <= 0:
        return mask

    # mean span length in blocks: cluster 0 -> isolated blocks, cluster 1 -> long
    # horizontal spans
    span = float(np.clip(1.0 + cluster * (mw * 0.55 - 1.0), 1.0, mw))
    flat = mask.ravel()
    placed = 0
    while placed < want:
        length = int(max(1, min(want - placed, rng.geometric(1.0 / span))))
        # one span is repeated down several rows: a head loses the same part of
        # the track across consecutive lines, so the patches have height too
        rows = int(max(1, rng.integers(1, max(2, int(2 + cluster * 6)))))
        start = int(rng.integers(0, max(1, mw)))
        cols = (np.arange(start, start + length) % mw)
        for r in range(rows):
            row = int(rng.integers(0, max(1, mh))) if r else int(rng.integers(0, max(1, mh)))
            idx = row * mw + cols
            new = ~flat[idx]
            flat[idx] = True
            placed += int(new.sum())
        if placed >= want:
            break

    if carry is not None and getattr(carry, "shape", None) == shape:
        mask = flat.reshape(shape)
        mask = mask | (carry & (rng.random(shape) < carry_rate))
        flat = mask.ravel()

    have = int(flat.sum())
    if have > want:
        on = np.flatnonzero(flat)
        flat[rng.choice(on, size=have - want, replace=False)] = False
    elif have < want:
        off = np.flatnonzero(~flat)
        flat[rng.choice(off, size=min(want - have, off.size), replace=False)] = True
    return flat.reshape(shape)


def apply_stale_macroblocks(frame: np.ndarray, stale: np.ndarray,
                            donor: np.ndarray) -> np.ndarray:
    """Substitute whole macroblocks from `donor` where `stale` is set."""
    H, W = frame.shape[:2]
    out = frame.copy()
    mh, mw = stale.shape
    bh = max(1, H // mh)
    bw = max(1, W // mw)
    ys, xs = np.nonzero(stale)
    for by, bx in zip(ys, xs):
        y0, y1 = by * bh, min(H, (by + 1) * bh)
        x0, x1 = bx * bw, min(W, (bx + 1) * bw)
        out[y0:y1, x0:x1] = donor[y0:y1, x0:x1]
    return out


# ---------------------------------------------------------------------------
# line-based (analogue 8 mm / Hi8 / VHS) shuttle corruption
# ---------------------------------------------------------------------------
def line_run_mask(rng: np.random.Generator, lines: int, fraction: float,
                  span_lines: float = 6.0,
                  carry: np.ndarray | None = None,
                  carry_rate: float = 0.45) -> np.ndarray:
    """
    Which scanlines the transport failed to read, for a *line-based* format.

    This is the analogue-family counterpart of `shuttle_stale_mask`, and the
    distinction is not cosmetic.  Hi8 and VHS record a continuous analogue
    waveform along each helical track: there is no macroblock, because there is no
    block transform and no addressable block structure.  What the mechanism
    produces when the heads sweep faster than the recording rate is therefore:

    * **whole scanlines** that arrive too weak or too mistimed to use, so a failed
      region has full width and *row* granularity;
    * those lines arrive in **contiguous runs**, because a head loses lock partway
      through a track and regains it later;
    * the run boundaries fall wherever lock was lost and regained, which is what
      makes the patch edges ragged vertically rather than aligned to a grid.

    That is exactly why no 16-pixel periodicity can be measured on the Hi8
    reference: the structure is a stack of variable-height line bands.  Applying a
    DV block model to this material asserts a grid the format does not have.

    `span_lines` is the mean run length in scanlines.
    """
    n = int(lines)
    want = int(round(float(np.clip(fraction, 0.0, 1.0)) * n))
    mask = np.zeros(n, dtype=bool)
    if want <= 0:
        return mask
    if carry is not None and getattr(carry, "shape", None) == (n,):
        mask |= carry & (rng.random(n) < carry_rate)
    span = float(max(1.0, span_lines))
    placed = int(mask.sum())
    guard = 0
    limit = want * 4 + 64
    while placed < want and guard < limit:
        guard += 1
        length = int(max(1, rng.geometric(1.0 / span)))
        start = int(rng.integers(0, n))
        idx = np.arange(start, start + length) % n
        placed += int((~mask[idx]).sum())
        mask[idx] = True
    have = int(mask.sum())
    if have > want:
        on = np.flatnonzero(mask)
        mask[rng.choice(on, size=have - want, replace=False)] = False
    return mask


def line_run_displacement(rng: np.random.Generator, mask: np.ndarray,
                          amount: float) -> np.ndarray:
    """
    Per-line horizontal displacement for the failed lines.

    A mistimed line is read late, and the delay drifts across the run, so the
    displacement grows line by line.  That progression is what produces the
    stepped left and right edges on the reference; a constant offset would cut a
    clean rectangle instead.
    """
    n = mask.size
    disp = np.zeros(n, dtype=np.float32)
    if amount <= 0.0 or not mask.any():
        return disp
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    for start in starts:
        length = 0
        while length < n and mask[(start + length) % n]:
            length += 1
        if length <= 0:
            continue
        ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
        disp[(start + np.arange(length)) % n] = ramp * amount * (0.6 + 0.8 * rng.random())
    return disp


def apply_stale_lines(y: np.ndarray, mask: np.ndarray, donor: np.ndarray,
                      disp: np.ndarray | None = None) -> np.ndarray:
    """Substitute whole scanlines from `donor` where `mask` is set."""
    out = y.copy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    if disp is not None and np.any(np.abs(disp[idx]) > 1e-3):
        from ..core.signal import horizontal_shift
        shifted = horizontal_shift(y, disp)
        out[idx] = shifted[idx]
    out[idx] = donor[idx]
    return out


def _blocks(plane: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Cut a plane into non-overlapping 8x8 blocks (edge-padded)."""
    h, w = plane.shape
    bh = (h + BLOCK - 1) // BLOCK
    bw = (w + BLOCK - 1) // BLOCK
    ph, pw = bh * BLOCK, bw * BLOCK
    if (ph, pw) != (h, w):
        plane = np.pad(plane, ((0, ph - h), (0, pw - w)), mode="edge")
    blocks = plane.reshape(bh, BLOCK, bw, BLOCK).transpose(0, 2, 1, 3)
    return blocks, bh, bw


def _unblocks(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    bh, bw = blocks.shape[0], blocks.shape[1]
    out = blocks.transpose(0, 2, 1, 3).reshape(bh * BLOCK, bw * BLOCK)
    return out[: shape[0], : shape[1]]


def dct2(blocks: np.ndarray) -> np.ndarray:
    """2-D DCT-II on the last two axes, orthogonal (unit gain)."""
    return np.einsum("ij,...jk,lk->...il", D, blocks, D, optimize=True)


def idct2(coeffs: np.ndarray) -> np.ndarray:
    return np.einsum("ji,...jk,kl->...il", D, coeffs, D, optimize=True)


def quantise(blocks: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Quantise DCT coefficients.

    DV quantises DC separately and more finely than AC, which is also what keeps
    flat areas from banding while letting detailed areas degrade first.
    """
    q = np.round(blocks / step[None, :, :])
    return q.astype(np.float32)


def dequantise(q: np.ndarray, step: np.ndarray) -> np.ndarray:
    return (q * step[None, :, :]).astype(np.float32)


# ---------------------------------------------------------------------------
@dataclass
class DVParams:
    """Everything about the DV branch that a profile may set."""
    chroma_mode: str = "411"        # "411" (525/60) | "420" (625/50)
    quality: float = 1.0            # 1.0 = standard 25 Mbit/s DV25
    base_step: float = 0.0125       # luma DC-preserving base quantiser step
    chroma_step_scale: float = 1.45  # chroma is quantised more coarsely
    adaptive: bool = True           # per-macroblock bit allocation
    block_artifacts: float = 1.0    # 0..1.5 extra blockiness (dropouts, damage)


class DVCodec:
    """
    A DV25-style encode/decode pair.

    The encoder subsamples chroma, transforms 8x8 blocks, and quantises them with
    a class chosen per macroblock so that detailed areas keep more of the bit
    rate.  The decoder does the inverse and reconstructs chroma by sample-and-hold.
    Running a frame through encode+decode is therefore not a no-op: what comes out
    is what a DV recorder would have stored.
    """

    def __init__(self, std: VideoStandard, params: DVParams | None = None):
        self.std = std
        self.p = params or DVParams()
        # `quality` scales the whole quantiser: 1.0 is standard 25 Mbit/s DV25,
        # below that the bit budget is effectively tighter (worse artefacts),
        # above it the frame is coded more finely than a real recorder would.
        self._base_step = max(1e-4, self.p.base_step / max(0.05, self.p.quality))
        # the DC coefficient gets its own, much finer step (DV codes DC to 8 bits)
        self._dc_step = max(1e-4, self._base_step * 0.22)
        self._dc_chroma_step = max(1e-4, self._base_step * 0.30)

    # ------------------------------------------------------------------
    def _plane_step_and_class(self, ac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-block quantiser step for the AC coefficients.

        `ac` is (bh, bw, 8, 8).  A macroblock is busier when its high-frequency
        energy is higher, and DV answers busy regions with a COARSER step (it has
        a fixed bit budget, so more coefficients means fewer bits each).  That is
        the mechanism behind mosquito noise: it is worst exactly where detail is.
        """
        if not self.p.adaptive:
            step = QUANT_CLASSES[1].table() * self._base_step
            flat = np.full((ac.shape[0], ac.shape[1]), step)
            return flat, step

        energy = np.sqrt((ac[..., 2:, 2:] ** 2).mean(axis=(2, 3)))
        lo, hi = np.percentile(energy, 35), np.percentile(energy, 92)
        span = max(1e-6, hi - lo)
        t = np.clip((energy - lo) / span, 0.0, 1.0)          # 0 flat .. 1 busy
        # busy -> coarser class
        idx = np.clip(np.round(t * (len(QUANT_CLASSES) - 1)), 0,
                      len(QUANT_CLASSES) - 1).astype(np.int32)
        steps = np.stack([qc.table() * self._base_step for qc in QUANT_CLASSES])
        return steps[idx], steps[idx[0, 0]]

    def _code_channel(self, plane: np.ndarray, chroma: bool) -> np.ndarray:
        scale = self.p.chroma_step_scale if chroma else 1.0
        blocks, bh, bw = _blocks(plane)
        coeffs = dct2(blocks)
        ac = coeffs.copy()
        ac[..., 0, 0] = 0.0
        step_map, _ = self._plane_step_and_class(ac * scale)
        # DC uses its own step; AC uses the class step
        dc_step = self._dc_chroma_step if chroma else self._dc_step
        s11 = step_map[..., 1:, 1:]
        s01 = step_map[..., 0, 1:]
        s10 = step_map[..., 1:, 0]
        q = np.empty_like(coeffs)
        q[..., 0, 0] = np.round(coeffs[..., 0, 0] / dc_step)
        q[..., 1:, 1:] = np.round(coeffs[..., 1:, 1:] / s11)
        q[..., 0, 1:] = np.round(coeffs[..., 0, 1:] / s01)
        q[..., 1:, 0] = np.round(coeffs[..., 1:, 0] / s10)

        out = np.empty_like(q)
        out[..., 0, 0] = q[..., 0, 0] * dc_step
        out[..., 1:, 1:] = q[..., 1:, 1:] * s11
        out[..., 0, 1:] = q[..., 0, 1:] * s01
        out[..., 1:, 0] = q[..., 1:, 0] * s10
        return _unblocks(idct2(out), plane.shape).astype(np.float32)

    # ------------------------------------------------------------------
    def encode_decode(self, sig: SignalFrame) -> SignalFrame:
        """
        Pass a frame through the DV recorder and read it back.

        Input and output are baseband YCbCr derived from the frame's YIQ; the
        composite carrier is meaningless on this branch, which is the whole point
        of having a separate path.
        """
        rgb = sig.to_rgb()
        y, cb, cr = rgb_to_ycbcr(rgb)

        if self.p.chroma_mode == "420":
            cb_s, cr_s = subsample_420(cb, cr)
        else:
            cb_s, cr_s = subsample_411(cb, cr)

        y_c = self._code_channel(y, chroma=False)

        if self.p.chroma_mode == "420":
            cb_u, cr_u = upsample_420(cb_s, cr_s, y.shape)
        else:
            cb_u, cr_u = upsample_411(cb_s, cr_s, y.shape)
        # chroma is coded at reduced resolution, so it is coded from the
        # subsampled image (that is what the recorder has to work with)
        cb_c = self._code_channel(cb_u, chroma=True)
        cr_c = self._code_channel(cr_u, chroma=True)

        rgb_out = ycbcr_to_rgb(y_c, cb_c, cr_c)
        out = SignalFrame.from_rgb(np.clip(rgb_out, 0.0, 1.0), self.std)
        return out

    # ------------------------------------------------------------------
    def augment_blockiness(self, sig: SignalFrame, amount: float) -> SignalFrame:
        """
        Extra 8x8 block structure, for damaged tape / dirty heads.

        On a DV tape a dropout is not a white RF streak: the error-correction
        code fails and the decoder conceals the missing macroblocks, so the
        artefact is block-shaped and repeats the neighbouring block.
        """
        if amount <= 1e-3:
            return sig
        rgb = sig.to_rgb()
        h, w = rgb.shape[:2]
        bh, bw = (h + BLOCK - 1) // BLOCK, (w + BLOCK - 1) // BLOCK
        rng = np.random.default_rng(1234)
        pick = rng.random((bh, bw)) < min(0.6, 0.10 * amount)
        if not pick.any():
            return sig
        out = rgb.copy()
        for by in range(bh):
            for bx in range(bw):
                if not pick[by, bx]:
                    continue
                y0, y1 = by * BLOCK, min(h, (by + 1) * BLOCK)
                x0, x1 = bx * BLOCK, min(w, (bx + 1) * BLOCK)
                sy0 = max(0, y0 - BLOCK)
                out[y0:y1, x0:x1] = rgb[sy0:sy0 + (y1 - y0), x0:x1]
        return SignalFrame.from_rgb(np.clip(out, 0.0, 1.0), self.std)

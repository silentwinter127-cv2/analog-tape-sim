"""
Stage 02 -- signal encoding / the recording format.

This is where a picture *becomes a signal*, and it is the stage that most
cleanly separates tape formats that look similar in still frames:

  composite analog (VHS / Video8 / Hi8)
      1. luma is bandwidth limited (VHS ~3 MHz, S-VHS ~5 MHz)
      2. chroma is heterodyned *below* luma and limited to <1 MHz
      3. the two are ADDED into one composite signal
      4. vertical chroma resolution collapses (chroma is ~40 lines tall)

  digital component (DV / MiniDV)
      keeps Y and chroma separate but subsamples chroma and quantises it
      (see atsim.codecs.dv)

The composite *sum* in step 3 is deliberately kept as a real signal plane, so
the artefacts created by trying to separate it again appear later in the player
instead of being painted on here.
"""
from __future__ import annotations

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..codecs.analog_iir import FilterChoice, lowpass_hz
from ..core.rng import OUProcess, ValueNoise3D, rng_for
from ..core.signal import (SignalFrame, composite_encode, highpass, lowpass,
                           subcarrier_phase)
from ..core.standards import VideoStandard


class FormatStage(Stage):
    name = "format"
    layers = ("format",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        cfg = self.cfg_of("format")
        self.luma_bw = float(cfg.get("luma_bandwidth", 3.0)) * 1e6
        self.chroma_bw = float(cfg.get("chroma_bandwidth", 0.75)) * 1e6
        self.chroma_vavg = float(cfg.get("chroma_vertical_averaging", 0.85))
        # ACC: overall chroma channel gain.  This is the physical source of
        # "saturation" -- the receiver normalises chroma against the burst
        # amplitude, so a bad reference drives the colour too strong or too weak.
        self.chroma_gain = float(cfg.get("chroma_gain", 1.0))
        self.edge_boost = float(cfg.get("edge_boost", 0.25))
        self.color_noise = float(cfg.get("color_noise", 0.25))
        self.std: VideoStandard = ctx.std
        # line-to-line chroma phase error (fast) -- "velocity error"
        self._line_phase = OUProcess(
            rng_for(ctx.seed, "format", "line_phase"), sigma=0.35, corr_len=1.6)
        # field-to-field phase error (slow) -- burst / APC drift
        self._field_phase = OUProcess(
            rng_for(ctx.seed, "format", "field_phase"), sigma=0.6, corr_len=12.0)
        # Chroma noise is not white per frame: it is a coarse spatial texture
        # (head-to-tape contact) modulated by a slower temporal envelope, so it
        # crawls and flickers instead of turning into "TV static" every frame.
        sc = max(1e-6, ctx.scale)
        self._noise_field = ValueNoise3D(
            seed=ctx.seed * 3 + 1, height=ctx.height, width=ctx.width,
            spatial_scale=2.6 / sc, time_scale=max(1.5, ctx.fps / 3.0))
        self._noise_env = ValueNoise3D(
            seed=ctx.seed * 3 + 2, height=ctx.height, width=ctx.width,
            spatial_scale=40.0 / sc, time_scale=max(2.0, ctx.fps / 6.0))
        self._noise_shape = (ctx.height, ctx.width)

    # ------------------------------------------------------------------
    def _resize_noise(self, shape: tuple[int, int]) -> None:
        """
        Rebuild the noise fields for the shape actually being rendered.

        The fields are built in `prepare()` from the context's dimensions, but the
        frame that arrives in `apply()` can be a different size -- preview scaling
        rounds the height to an even number, so a 480-row source previewed at scale
        0.37 arrives as 178 rows while the noise was built for 480.  Subtracting the
        two raised

            ValueError: operands could not be broadcast together with shapes (178,236) (480,1)

        which names two shapes and says nothing about the cause.  Resampling per shape
        keeps the noise aligned with the picture, and a stage that is handed a
        different frame size mid-run is a normal thing in this application.
        """
        h, w = int(shape[0]), int(shape[1])
        if getattr(self, "_noise_shape", None) == (h, w):
            return
        sc = max(1e-6, self.ctx.scale)
        seed = self.ctx.seed
        self._noise_field = ValueNoise3D(
            seed=seed * 3 + 1, height=h, width=w,
            spatial_scale=2.6 / sc, time_scale=max(1.5, self.ctx.fps / 3.0))
        self._noise_env = ValueNoise3D(
            seed=seed * 3 + 2, height=h, width=w,
            spatial_scale=40.0 / sc, time_scale=max(2.0, self.ctx.fps / 6.0))
        self._noise_shape = (h, w)

    def _noise_for(self, shape: tuple[int, int], index: int) -> np.ndarray:
        self._resize_noise(shape)
        return self._noise_field.sample(index).astype(np.float32)

    def _noise_env_for(self, shape: tuple[int, int], index: int) -> np.ndarray:
        self._resize_noise(shape)
        return self._noise_env.sample(index).astype(np.float32)

    # ------------------------------------------------------------------
    def _choice(self) -> FilterChoice:
        """
        The filter family this format uses.

        Kept on the stage rather than threaded through every call: the family is
        a property of the *apparatus* -- a consumer VCR's output amplifier is a
        constant-k cascade, a lab instrument is not -- so one choice covers luma
        and chroma together, which is also physically right because both pass
        through that same output stage.
        """
        cfg = self.cfg_of("format")
        kind = int(round(float(cfg.get("filter_kind", 1.0))))
        sections = int(round(float(cfg.get("filter_sections", 3.0))))
        return FilterChoice(
            {0: "gaussian", 1: "constant_k", 2: "butterworth"}.get(kind, "constant_k"),
            sections=max(1, sections))

    def _limit_luma(self, y: np.ndarray) -> np.ndarray:
        return lowpass_hz(y, self.luma_bw, self.std.pixel_rate_hz, self._choice())

    def _boost_edges(self, y: np.ndarray) -> np.ndarray:
        if self.edge_boost < 1e-4:
            return y
        px_per_cycle = 2.0 * np.pi / self.std.subcarrier_rad_per_sample
        hp = highpass(y, max(1.0, px_per_cycle / 3.2), axis=1)
        return y + (0.45 * self.edge_boost) * hp

    def _limit_chroma(self, i: np.ndarray, q: np.ndarray):
        choice = self._choice()
        rate = self.std.pixel_rate_hz
        i = lowpass_hz(i, self.chroma_bw, rate, choice)
        q = lowpass_hz(q, self.chroma_bw, rate, choice)
        if self.chroma_vavg > 1e-4:
            # colour-under recording: the chroma signal is only ~40 lines tall,
            # so the receiver averages neighbouring lines (a 1H/2H delay line)
            k = float(np.clip(self.chroma_vavg, 0.0, 1.0))
            i = (1.0 - k) * i + k * lowpass(i, 1.35, axis=0)
            q = (1.0 - k) * q + k * lowpass(q, 1.35, axis=0)
        # ACC: the gain the receiver applies to bring chroma back to nominal
        # level, using the burst amplitude as its reference.  When that reference
        # is itself wrong -- a failing chroma card, a damaged burst, a
        # multi-generation copy -- the correction goes the wrong way and the
        # colour ends up systematically too strong or too weak.  That is the
        # mechanism behind a heavily over-saturated look, and it is why this is a
        # gain on the chroma channel rather than a saturation control on the
        # output.
        #
        # Gain above unity also moves the blanking level, because the chroma
        # amplitude adds around the luma pedestal: scaling chroma alone pushes
        # peaks into clipping and measurably *lowers* the picture's mean level
        # (measured 0.692 -> 0.638 going from gain 1.0 to 3.0), which is the
        # opposite of the reference, where over-saturation comes with a raised
        # level.  So luma is lifted alongside, keeping the ratio the clip shows.
        if abs(self.chroma_gain - 1.0) > 1e-4:
            i = i * self.chroma_gain
            q = q * self.chroma_gain
        return i, q

    # ------------------------------------------------------------------
    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        rgb = buf.rgb
        sig = SignalFrame.from_rgb(rgb, self.std)
        y, i, q = sig.y, sig.i, sig.q

        y = self._limit_luma(y)
        y = self._boost_edges(y)
        i, q = self._limit_chroma(i, q)
        # A luma lift was tried here to mimic the raised pedestal that accompanies
        # over-saturation in the reference.  It is the wrong correction: lifting
        # luma *reduces* measured saturation (gain 3.0 fell from 0.47 to 0.40)
        # while pushing the mean level to 0.80 against a reference of 0.66, so it
        # made the picture both paler and brighter.  The clipping ceiling on
        # chroma is a real limitation of adding gain in I/Q; leaving it honest is
        # better than trading saturation for brightness.

        # chroma channel noise: the colour-under path has a much worse SNR than
        # luma, and this is what makes VHS colour sparkle rather than merely blur
        out_phase = 0.0
        if self.color_noise > 1e-4:
            field = int(buf.meta.get("field", 0))
            # phase error per line: a fast component (velocity error) plus a
            # slow component resolved for this frame/field
            ph = self._line_phase.advance(i.shape[0]) * 0.9
            fp = self._field_phase.prefix(buf.index + field + 1)[-1] * 0.5
            ph = ph + fp
            cos_p = np.cos(ph)[:, None].astype(np.float32)
            sin_p = np.sin(ph)[:, None].astype(np.float32)
            amp = self.color_noise * 0.018
            n = self._noise_for(i.shape, buf.index)
            env = 0.55 + 0.75 * np.abs(self._noise_env_for(i.shape, buf.index))
            # injected in the I/Q plane with a rotating phase, which is why VHS
            # colour noise reads as coloured crawling speckle, not grey grain
            i = i + amp * (n * env) * cos_p
            q = q + amp * (n * env) * sin_p
            out_phase = float(ph.mean())

        # The subcarrier is encoded at 0.63 of full scale (SMPTE 170M).  This
        # matters beyond bookkeeping: cross-colour, dot crawl and chroma noise
        # are all proportional to the carrier amplitude, so getting this right
        # is what keeps those artefacts at a believable level.
        comp = composite_encode(y, i, q, self.std, chroma_gain=0.63)

        frame = SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))
        out = FrameBuffer(
            rgb=rgb, composite=comp, index=buf.index, time=buf.time, std=self.std,
            meta=dict(buf.meta),
        )
        out.meta["signal"] = frame
        out.meta["format"] = "composite"
        out.meta["chroma_phase"] = out_phase
        return out

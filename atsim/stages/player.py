"""
Stage 04 -- reading, separating, correcting, outputting.

This stage is deliberately separate from the tape stage, because *the same
tape behaves differently in different machines*.  Here the composite signal is
read back:

  1. transport jitter of the player is added (or the tape's error is damped),
  2. time-base correction removes fast phase error but cannot remove slow drift,
     and adds a residual of its own,
  3. the composite signal is separated into luma + chroma -- imperfectly --
     which is where cross-colour and dot crawl are actually born,
  4. chroma is demodulated, its phase is disturbed by burst error, and the
     output amplifiers set level, black level and frequency response.

Steps 3 and 4 are the reason this project can say "the artefact comes from the
separation, not from an effect": the numbers going in were added by the format
stage, and here we try to take them apart again with limited success.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import OUProcess, ValueNoise3D, rng_for
from ..codecs.analog_iir import notch, preemphasis
from ..core.signal import (SignalFrame, bandpass, demod_chroma, highpass,
                           lowpass, remod_chroma, split_composite,
                           subcarrier_phase)
from ..core.standards import VideoStandard



# The fixed messages a deck's character generator can put on screen.  A real one
# has a small vocabulary, which is why the parameter selects an index rather than
# accepting text.
OSD_MESSAGES: dict[int, tuple[str, ...]] = {
    0: ("NO SIGNAL",),
    1: ("SEARCH", "CHAPTER 17"),
    2: ("PLAY",),
}

class PlayerStage(Stage):
    name = "player"
    layers = ("player", "format")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        p = self.cfg_of("player")
        f = self.cfg_of("format")
        self.transport_jitter = float(p.get("transport_jitter", 0.35))
        self.tbc_enabled = float(p.get("tbc_enabled", 1.0)) > 0.5
        self.tbc_strength = float(p.get("tbc_strength", 0.65))
        self.tbc_residual = float(p.get("tbc_residual", 0.10))
        self.line_delay = float(p.get("line_delay", 0.05))
        self.freq_response = float(p.get("frequency_response", 0.10))
        self.head_wear = float(p.get("head_wear", 0.20))
        self.output_gain = float(p.get("output_gain", 0.0))
        self.output_bias = float(p.get("output_bias", 0.0))
        self.preemphasis = float(p.get("preemphasis", 0.0))
        self.ringing = float(p.get("ringing", 0.0))
        self.ringing_freq = float(p.get("ringing_freq", 0.5))
        # The deck's own output when it cannot use the tape at all.  This is not a
        # degradation of the picture -- the machine stops displaying the tape.
        self.blue_screen = float(p.get("blue_screen", 0.0))
        # The deck's on-screen display has a fixed vocabulary, not free text.
        self.blue_osd = int(round(float(p.get("blue_osd", 0.0))))

        # from the format layer: what the receiver does with the signal
        self.comb_depth = float(f.get("comb_depth", 0.55))
        self.cross_color = float(f.get("cross_color", 0.45))
        self.cross_luma = float(f.get("cross_luma", 0.30))
        self.burst_error_deg = float(f.get("burst_phase_error", 5.0))
        self.chroma_bw = float(f.get("chroma_bandwidth", 0.75)) * 1e6
        self.luma_bw_hz = float(f.get("luma_bandwidth", 3.0)) * 1e6
        self.color_noise = float(f.get("color_noise", 0.25))
        self.std: VideoStandard = ctx.std

        self._residual = OUProcess(rng_for(ctx.seed, "player", "residual"),
                                   sigma=1.0, corr_len=1.2)
        self._burst = OUProcess(rng_for(ctx.seed, "player", "burst"),
                                sigma=1.0, corr_len=9.0)
        self._transport = OUProcess(rng_for(ctx.seed, "player", "transport"),
                                    sigma=1.0, corr_len=5.0)
        self._head_grain: ValueNoise3D | None = None
        # The residues of the band split are not re-drawn per frame either: the
        # separation is the same imperfect circuit every time, so its error
        # pattern is anchored to the tape area and only drifts slowly.
        sc = max(1e-6, ctx.scale)
        self._sep_noise = ValueNoise3D(
            seed=ctx.seed * 19 + 13, height=ctx.height, width=ctx.width,
            spatial_scale=8.0 / sc, time_scale=max(2.0, ctx.fps * 1.2))

    def prepare(self) -> None:
        self._head_grain = ValueNoise3D(
            seed=self.ctx.seed * 13 + 5, height=self.ctx.height, width=self.ctx.width,
            spatial_scale=2.0 / max(1e-6, self.ctx.scale),
            time_scale=max(2.0, self.ctx.fps * 0.8))

    # ------------------------------------------------------------------
    def _timebase_offset(self, buf: FrameBuffer, height: int) -> np.ndarray:
        """
        The sampling-phase error the corrector has to deal with, in pixels.

        Returned rather than applied, because *how* it is applied decides whether
        colour survives: see :meth:`_separate`.
        """
        phase = buf.meta.get("line_phase")
        n = int(height)
        # the player's own transport adds error independent of the cassette's
        own = self._transport.advance(n) * 0.5
        total = (np.asarray(phase, dtype=np.float64) if phase is not None
                 else np.zeros(n, dtype=np.float64)) + self.transport_jitter * own * 1.6

        if self.tbc_enabled:
            # What a corrector removes, in order of how much it matters:
            #
            #  * the *shared offset* of the field.  It dominates the uncorrected
            #    error -- measured on a real capture, the per-row spread falls about
            #    8.8x when the corrector is engaged -- and a corrector removes it
            #    because its whole job is to relock each line to the house reference;
            #  * the *fast* per-line component: lines are pulled towards each other,
            #    estimated by a short local average;
            #  * it leaves a slow drift it cannot undo (tape stretch, capstan wander)
            #    and its own residual jitter, both of which are parameters.
            #
            # The previous version blended towards `mean(total)` and then subtracted
            # `mean(total)` again at the end, so the correction cancelled itself
            # exactly: switching it on changed the per-row spread by 0.79x, i.e. not
            # at all.  The parameter was read; the arithmetic made it inert. Same shape
            # of defect as the deck's blue screen being rebuilt from the tape signal it
            # had just rejected -- a feature present, reachable, and annulled by a
            # later step.
            gain = float(np.clip(self.tbc_strength, 0.0, 1.0))
            w = max(3, int(round(height / 10.0)))
            k = np.ones(w, np.float64) / w
            smoothed = np.convolve(total, k, mode="same")
            fast = total - smoothed                      # the part lines get pulled out
            shared = float(np.mean(total))
            # remove the shared offset, and most of the fast component
            total = smoothed - gain * shared
            total = total + (1.0 - gain) * fast
            total = total + self._residual.advance(n) * self.tbc_residual * 2.2

        # Only the *relative* displacement matters visually, so a shared offset is
        # taken out to keep the picture inside the raster.  This is a constant for the
        # frame: it does not undo the correction above, which acts on the differences
        # between rows.
        disp = total - float(np.median(total))

        return disp + self.line_delay

    def _separate(self, comp: np.ndarray, buf: FrameBuffer
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Composite -> (luma, I, Q), with the receiver's real imperfections.

        The time-base error is handled the way a corrector handles it.  Luma is
        displaced geometrically (it is just a picture), but the colour carrier is
        NOT shifted -- instead the demodulating reference is displaced by the
        same amount, which is what phase-locking to the recorded colour burst
        amounts to.  Shifting the carrier instead rotates its phase and destroys
        hue (measured: chroma correlation fell from 0.93 to 0.02).

        The residue of the band split is what creates cross-colour and dot crawl;
        `cross_color` / `cross_luma` scale how much of it the (imperfect) comb
        filter leaves behind.  The residue is modulated by a slowly drifting
        spatial field rather than redrawn each frame, so the fringing crawls the
        way it does on a real tape instead of flickering like noise.
        """
        from ..core.signal import horizontal_shift, remod_chroma
        gain = 0.63
        disp = self._timebase_offset(buf, comp.shape[0])

        # The time-base error is a geometric displacement, so it is applied in
        # baseband, exactly as the tape stage does.  Displacing the *carrier*
        # instead rotates chroma phase (1 px = 0.42 of a cycle), which measured
        # as chroma correlation 0.50 and a Q that was pure noise; de-modulating
        # against a compensating reference only got to 0.45.  In baseband the same
        # displacement costs nothing: 0.99.
        window = float(np.clip(self.chroma_bw * 1.15, 0.35e6, 0.9e6))
        y_band, chroma_band = split_composite(
            comp, self.std, comb_depth=0.0, chroma_bandwidth_hz=window)
        i, q = demod_chroma(chroma_band / gain, self.std,
                            i_bw_hz=1.5e6, q_bw_hz=1.5e6)
        y_band = horizontal_shift(y_band, disp)
        i = horizontal_shift(i, disp)
        q = horizontal_shift(q, disp)
        comp_shifted = (y_band + remod_chroma(i, q, self.std, chroma_gain=gain)
                        ).astype(np.float32)

        # Re-split with the receiver's own comb and bandwidth so the downstream
        # crosstalk model operates on the signal the receiver actually sees.
        y_band, chroma_band = split_composite(
            comp_shifted, self.std, comb_depth=self.comb_depth,
            chroma_bandwidth_hz=self.chroma_bw)
        chroma_band = chroma_band / gain

        field = self._sep_noise.sample(buf.index).astype(np.float32) if (
            self.cross_color > 1e-4 or self.cross_luma > 1e-4) else None

        # some luma high frequencies are mistaken for chroma
        # (-> rainbow fringes on fine detail) ...
        if self.cross_color > 1e-4:
            w = comp.shape[1]
            detail = highpass(y_band, self.std.pixels_for_bandwidth(
                self.std.subcarrier_hz * 0.5, w), axis=1)
            mod = 0.6 + 0.7 * np.abs(field) if field is not None else 1.0
            chroma_band = chroma_band + self.cross_color * 0.55 * detail * mod

        # ... and some chroma is mistaken for luma (-> moving dot pattern)
        if self.cross_luma > 1e-4:
            resid = chroma_band - remod_chroma(i, q, self.std, chroma_gain=1.0)
            mod = 0.6 + 0.7 * np.abs(field) if field is not None else 1.0
            y_band = y_band + self.cross_luma * 0.5 * resid * mod

        return y_band.astype(np.float32), i.astype(np.float32), q.astype(np.float32)

    def _chroma_phase_error(self, i: np.ndarray, q: np.ndarray,
                            buf: FrameBuffer) -> tuple[np.ndarray, np.ndarray]:
        """
        Burst/APC phase error: a slow rotation of the whole colour vector.

        The player locks its local subcarrier to the recorded colour burst.  If
        that lock is off (or the APC loop drifts), *all* hues rotate together.
        The rotation is slow -- a fraction of a degree per line -- so it shows
        up as an unstable colour cast over a shot, not as line-by-line noise.
        """
        if abs(self.burst_error_deg) < 1e-3 and abs(float(buf.meta.get("chroma_phase", 0.0))) < 1e-3:
            return i, q
        n = max(1, buf.index + 2)
        drift = self._burst.prefix(n)[-1]
        err = np.radians(self.burst_error_deg) * (0.8 * drift + 0.4)
        # the encoder's velocity error rotates the whole colour vector as well:
        # that is why a mistracking VHS picture drifts in hue, not only in space
        err = err + float(buf.meta.get("chroma_phase", 0.0))
        c, s = np.cos(err), np.sin(err)
        return (i * c - q * s).astype(np.float32), (i * s + q * c).astype(np.float32)

    def _frequency_response(self, y: np.ndarray) -> np.ndarray:
        """
        Player output amplifier + worn heads: a high-frequency shelf.

        This is also where two record/replay artefacts belong, because both are
        *this* circuit's behaviour rather than the medium's:

        * pre-emphasis.  Recorders boost high frequencies before recording to
          protect them from noise, and the player is supposed to undo that.
          Imperfect de-emphasis leaves an overshoot on hard edges -- the white
          outline that reads as "over-sharpened video".
        * ringing.  A band-limited channel passes an edge with overshoot and
          undershoot; modelled as an IIR notch so the response actually rings
          instead of merely being smeared.
        """
        soft = float(np.clip(self.freq_response + 0.55 * self.head_wear, -1.0, 1.5))
        out = y
        if abs(soft) >= 1e-3:
            if soft > 0:
                # one-pole low-pass, corner moving down as `soft` grows
                fc = float(np.clip(1.0 - 0.8 * soft, 0.06, 1.0))
                a = np.exp(-2.0 * np.pi * fc * 0.25)
                filt = lfilter([1.0 - a], [1.0, -a], out, axis=1).astype(np.float32)
                out = (1.0 - soft) * out + soft * filt
            else:
                amount = -soft
                blurred = lowpass(out, 0.9)
                out = (out + 0.55 * amount * (out - blurred)).astype(np.float32)

        if self.preemphasis > 1e-4:
            out = preemphasis(out, self.luma_bw_hz, self.std.pixel_rate_hz,
                              self.preemphasis * 0.6)
        if self.ringing > 1e-4:
            # `ringing_freq` was being ignored here: the notch was called with a
            # hardcoded 0.5, so moving the control changed nothing at all.  The
            # amount is also scaled down, because an unscaled notch at 1.0
            # perturbed 98% of the pixels -- that is a filter, not an edge
            # artefact.
            out = notch(out, self.ringing_freq, 2.0, self.ringing * 0.18)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        comp = buf.composite
        if comp is None:
            return buf
        if self._head_grain is None:
            self.prepare()

        y, i, q = self._separate(comp, buf)
        i, q = self._chroma_phase_error(i, q, buf)
        y = self._frequency_response(y)

        # head wear also degrades chroma SNR -- via the same crawling texture as
        # the format stage rather than fresh white noise, so the colour speckle
        # stays anchored to the tape surface
        if self.head_wear > 1e-3 and self._head_grain is not None:
            n = self._head_grain.sample(buf.index).astype(np.float32)
            amt = 0.030 * self.head_wear * (0.4 + self.color_noise)
            i = i + amt * n
            q = q + amt * np.roll(n, 3, axis=1)

        y = y * (1.0 + 0.25 * self.output_gain) + 0.08 * self.output_bias

        if self.blue_screen > 1e-3:
            # The deck gives up: the tape's picture is discarded entirely and its
            # own graphic takes its place.  A partly-transparent substitution would
            # be wrong -- the point of the behaviour is that the tape is *not* on
            # screen at all.
            from ..io.imagecodec import deck_blue_screen
            h, w = y.shape[:2]
            lines = OSD_MESSAGES.get(self.blue_osd, ("NO SIGNAL",))
            rgb = deck_blue_screen(h, w, lines)
            # Drop the upstream signal, and drop it decisively.  The display stage
            # prefers `meta["signal"]` over the RGB buffer, so leaving the earlier
            # stage's signal in place meant the deck's own picture was rebuilt from
            # the tape signal it had just rejected -- the substitution appeared to
            # do nothing at all while every attribute involved read correctly.
            meta = {k: v for k, v in buf.meta.items()
                    if k not in ("signal", "composite", "format")}
            meta["format"] = "deck_osd"
            meta["deck_gave_up"] = True
            return FrameBuffer(rgb, None, buf.index, buf.time, self.std, meta)

        sig = SignalFrame(y, i, q)
        # Publish the reconstructed picture, not the pass-through RGB that
        # travelled down the chain from the camera.  With the display stage
        # disabled the frame handed to the user *is* this image, and keeping the
        # original buffer there silently discarded every luma/chroma change the
        # format and medium stages had made.
        rgb_out = sig.to_rgb()
        out = FrameBuffer(rgb_out, comp, buf.index, buf.time, buf.std, dict(buf.meta))
        out.meta["signal"] = sig
        out.meta["stage_done"] = out.meta.get("stage_done", []) + ["player"]
        return out

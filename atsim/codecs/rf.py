"""
The RF link: modulation onto a carrier and demodulation back to baseband.

This is a genuinely separate physical stage from composite baseband, and the
reference capture of an HDMI->AV->RF->TV loop is what makes the distinction
concrete.  On a baseband composite cable the picture is degraded by *bandwidth*
and *crosstalk*; once the signal has been put on a VHF/UHF carrier and taken off
again, four additional mechanisms appear that have no baseband equivalent:

1. **Vestigial sideband (VSB).**  The vision carrier is amplitude modulated, and
   the lower sideband is mostly filtered away to fit the 6 MHz channel.  What
   survives is the carrier plus the upper sideband plus a *vestige* of the lower
   one.  Because the sidebands are no longer symmetric, high-frequency detail is
   lost asymmetrically and the demodulated picture is soft in a way a low-pass
   filter does not reproduce.
2. **Carrier-to-noise ratio.**  The tuner's front end adds noise proportional to
   its noise figure and inversely to the received carrier power.  Unlike tape
   noise this is *broadband and before detection*, so it lands on luma and chroma
   together and its visible form depends on signal level.
3. **Sync-lock loss.**  When C/N falls far enough the horizontal AFC cannot hold.
   The line clock slips, and lines are displaced sideways -- sometimes by a lot --
   with the colour underneath them scrambled because the subcarrier reference is
   no longer valid.  This is the serpentine tearing and colour corruption in the
   reference frames, and it is the single most recognisable RF artefact.
4. **Adjacent-channel interference.**  A strong neighbouring channel beats
   against the vision carrier and appears as a moving diagonal or herringbone
   pattern, because the beat frequency is not harmonically related to the line or
   field rate.

So the RF stage takes a composite signal and returns a composite signal, but the
signal that comes back is not the signal that went in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.rng import OUProcess, derive_seed, rng_for
from ..core.signal import horizontal_shift, lowpass, subcarrier_phase
from ..core.standards import VideoStandard


@dataclass
class RFParams:
    """Everything about the RF link that a profile may set."""
    # --- channel geometry -------------------------------------------
    channel_mhz: float = 6.0            # channel width (6 MHz for M/NTSC)
    carrier_offset_mhz: float = 1.25    # vision carrier above channel bottom
    vestige_mhz: float = 0.75           # width of the retained lower sideband
    # --- receiver ------------------------------------------------------
    noise_figure_db: float = 8.0        # tuner noise figure
    signal_dbm: float = -60.0           # received carrier power
    agc_range: float = 1.0              # AGC correction strength 0..1
    # --- impairments ---------------------------------------------------
    sync_loss_threshold_db: float = 30.0   # C/N at which the AFC starts to slip
    sync_loss_amount: float = 1.0          # scales how badly it slips
    adjacent_channel: float = 0.0          # 0..1 interference level
    broadcast_noise: float = 0.0           # 0..1 off-air noise floor
    adjacent_offset_mhz: float = 1.5       # beat offset of the interferer


class RFLink:
    """
    Stateful modulator/demodulator pair.

    Stateful because adjacent-channel interference and sync slip are *processes*
    over time, not per-frame effects: the beat pattern has to drift continuously
    and the AFC has to accumulate error, or the result looks like noise rather
    than like a mistuned television.
    """

    def __init__(self, std: VideoStandard, params: RFParams | None = None,
                 seed: int = 20240101):
        self.std = std
        self.p = params or RFParams()
        self.seed = int(seed)
        # carrier-to-noise ratio in dB, derived rather than set directly
        self.cnr_db = self._cnr_db()
        rng = rng_for(seed, "rf", "processes")
        # beat phase drifts slowly: the interferer and the vision carrier are
        # never exactly on a rational frequency relationship
        self._beat = OUProcess(rng, sigma=1.0, corr_len=40.0)
        # AFC error accumulates and then snaps back when it re-locks
        self._afc = 0.0
        self._afc_rng = rng_for(seed, "rf", "afc")
        self._locked = True

    # ------------------------------------------------------------------
    def _cnr_db(self) -> float:
        """
        Carrier-to-noise ratio at the detector.

        C/N = received power - noise floor, where the noise floor is
        kTB plus the receiver's noise figure, widened by the IF bandwidth.  The
        absolute numbers matter less than the *relationship*: lowering the
        received level or raising the noise figure degrades the picture, which
        is what a user expects a "signal strength" control to do.
        """
        kTB_dbm = -174.0 + 10.0 * np.log10(6.0e6)      # ~ -106 dBm in 6 MHz
        noise_dbm = kTB_dbm + float(self.p.noise_figure_db)
        return float(self.p.signal_dbm - noise_dbm)

    def set_reception(self, signal_dbm: float | None = None,
                      noise_figure_db: float | None = None) -> None:
        if signal_dbm is not None:
            self.p.signal_dbm = float(signal_dbm)
        if noise_figure_db is not None:
            self.p.noise_figure_db = float(noise_figure_db)
        self.cnr_db = self._cnr_db()

    # ------------------------------------------------------------------
    def _vestigial_filter(self, comp: np.ndarray) -> np.ndarray:
        """
        Channel shaping: the video information the 6 MHz channel can carry.

        A note on honesty.  Vestigial sideband shaping is a property of the RF
        *spectrum around the carrier*, and the mapping from that to a response on
        the baseband composite is not a simple one: the vestige is a narrow band
        adjacent to the carrier frequency, which corresponds to baseband
        frequencies a few hundred kHz above DC, while the bulk of the picture
        rides on the single remaining sideband at full amplitude.

        Modelling that mapping naively -- reading the vestige window as if it were
        a baseband bandpass -- attenuated 0.5-2 MHz to 5% while passing
        everything above it, which is the opposite of what a channel does.  So
        this implements the part that is unambiguous and visible instead: the
        signal occupies the baseband the channel's upper limit allows, with a
        smooth roll-off at the edge.  That is a *simplification*, and it is
        recorded as one rather than dressed up as a VSB model.
        """
        n = comp.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0) * self.std.pixel_rate_hz
        upper_limit = (self.p.channel_mhz - self.p.carrier_offset_mhz) * 1e6
        # smooth (raised-cosine) edge over the last 0.7 MHz of the channel
        edge = 0.7e6
        response = np.ones_like(freqs)
        roll = (freqs - (upper_limit - edge)) / edge
        response = np.where(freqs <= upper_limit - edge, 1.0,
                            np.where(freqs >= upper_limit, 0.0,
                                     0.5 * (1.0 + np.cos(np.pi * np.clip(roll, 0.0, 1.0)))))
        spec = np.fft.rfft(comp, axis=1)
        out = np.fft.irfft(spec * response[None, :], n=n, axis=1)
        return out.astype(np.float32)

    def _if_noise(self, comp: np.ndarray, frame_index: int,
                  rng: np.random.Generator) -> np.ndarray:
        """
        Add receiver noise *before* detection, scaled by C/N.

        Because it is added to the modulated signal, its effect is level
        dependent: the same noise power looks far worse in dark areas (where the
        carrier is weak) than in highlights, which is why RF noise reads as
        "sparkle in the shadows" rather than as an even grain.
        """
        cnr = self.cnr_db
        # C/N of 45 dB is a clean picture; below ~30 dB it is visibly noisy
        severity = float(np.clip((45.0 - cnr) / 30.0, 0.0, 2.0))
        bn = float(np.clip(self.p.broadcast_noise, 0.0, 1.0))
        # The early return has to account for *both* sources.  It tested only the
        # C/N-derived severity, so a strong signal (severity 0) returned before the
        # off-air noise was ever reached -- the control read 0.9 on the object and
        # changed nothing about the output.  A guard on one of two independent
        # inputs is the bug, not the noise code below it.
        if severity <= 1e-4 and bn <= 1e-4:
            return comp
        amp = 0.012 * severity
        noise = rng.standard_normal(comp.shape).astype(np.float32)
        # IF noise is band-limited to the channel, not white to Nyquist
        sigma = self.std.pixels_for_bandwidth(3.2e6, comp.shape[1])
        noise = lowpass(noise, max(0.4, sigma), axis=1)
        out = comp + amp * noise

        # Off-air noise floor.  Deliberately separate from the C/N-driven term
        # above: that one is proportional to receiver noise and produces the
        # familiar "sparkle in the shadows", while this models a signal that is
        # simply buried.  The reference for it (a 1992 weak-signal off-air
        # recording) shows high-frequency energy at 0.024 with saturation only
        # 0.178, i.e. the picture is destroyed by noise right across the frame
        # rather than shifted in colour -- which is why it cannot be expressed by
        # the chroma-side controls that handle tape damage.
        if bn > 1e-4:
            # The noise must be band-limited to roughly the *chroma* bandwidth, not
            # to the whole channel.  Broadband noise on the composite put most of
            # its energy where the chroma demodulator could read it, so this control
            # *raised* saturation (0.34 -> 0.54) instead of burying it -- the
            # opposite of the reference, where the picture is destroyed across the
            # whole frame and saturation falls to 0.178.
            chroma_bw = float(np.clip(float(self.p.channel_mhz) * 1e6 * 0.5,
                                      0.5e6, 1.6e6))
            fine = rng.standard_normal(comp.shape).astype(np.float32)
            fine = lowpass(fine, max(0.35, self.std.pixels_for_bandwidth(
                chroma_bw, comp.shape[1])), axis=1)
            # A coarser component supplies the low-frequency part of the noise;
            # fine grain alone reads as sparkle rather than as a buried picture.
            coarse = rng.standard_normal((comp.shape[0] // 4 + 1,
                                          comp.shape[1] // 4 + 1)).astype(np.float32)
            coarse = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)[
                :comp.shape[0], :comp.shape[1]]
            out = out + (0.05 + 0.90 * bn) * fine + 0.22 * bn * coarse
        return out.astype(np.float32)

    def _sync_lock_artefacts(self, comp: np.ndarray, frame_index: int,
                             rng: np.random.Generator
                             ) -> tuple[np.ndarray, dict]:
        """
        Horizontal AFC slip: the tuner loses the line clock.

        This is the RF signature.  The AFC accumulates phase error from noise; if
        the C/N is below the threshold the error grows until the loop slips,
        displacing a run of lines sideways.  Crucially the *colour* is corrupted
        as well, because the subcarrier reference is derived from the same
        timing -- so the slipped lines carry wrong hue, not merely wrong
        position, which is what makes the artefact look like torn rainbows.

        Implemented as a per-line displacement plus a per-line colour phase
        error, both driven by the same slip process (they share a cause).
        """
        thr = float(self.p.sync_loss_threshold_db)
        # Degradation must appear only when the carrier is *weaker* than the
        # threshold.  Two earlier formulations got the sign inverted and made the
        # picture tear worse at high C/N than at low -- the direction of the
        # effect is the one thing this model has to get right.
        deficit = float(thr - self.cnr_db)
        if deficit <= 0.0 or self.p.sync_loss_amount <= 0.0:
            self._afc *= 0.5        # well above threshold: the loop always holds
            self._last_phase_err = None
            return comp, {"slipped_lines": 0, "disp": None, "severity": 0.0,
                          "slip_event": False, "afc": round(self._afc, 3)}
        severity = float(np.clip(deficit / 14.0, 0.0, 2.0)) * self.p.sync_loss_amount

        H = comp.shape[0]
        # Slip is a per-frame event whose probability rises as the carrier gets
        # weaker, plus a persistent offset while the loop is unlocked.  A purely
        # accumulating model was tried first and was wrong in both directions: it
        # never slipped at realistic C/N, and once it did slip the error stayed
        # latched.  The loop re-locks on vertical sync every field, so the error
        # must also relax -- but not so fast that a marginal picture never tears.
        slip_p = float(np.clip(1.15 * severity, 0.0, 0.85))
        slipped_now = bool(self._afc_rng.random() < slip_p)
        if slipped_now:
            self._afc = float(np.clip(
                self._afc + self._afc_rng.standard_normal() * 0.5, -1.5, 1.5))
        else:
            self._afc *= 0.88
        self._locked = not slipped_now and abs(self._afc) < 0.35

        # displacement grows with the current error and decays towards the top of
        # the frame (the servo re-locks on vertical sync, so the damage is worst
        # just above it)
        rows = np.arange(H, dtype=np.float32)
        ramp = np.clip((rows / max(1.0, H - 1.0)), 0.0, 1.0) ** 1.4
        jitter = self._beat.advance(H).astype(np.float32)
        scale = (1.5 + 18.0 * severity)
        disp = (jitter * ramp * scale * (0.4 + 3.0 * abs(self._afc))).astype(np.float32)
        if slipped_now:
            # the slip itself displaces a run of lines hard
            run = int(self._afc_rng.integers(H // 8, max(H // 8 + 1, H // 2)))
            start = int(self._afc_rng.integers(0, max(1, H - run)))
            disp[start:start + run] += float(
                self._afc_rng.choice([-1.0, 1.0]) * (6.0 + 22.0 * severity))
        self._slip_disp = disp
        slipped = int(np.count_nonzero(np.abs(disp) > 0.5 * scale))
        # The displacement is applied to the composite, but the *timing* error it
        # represents is shared: the same slip that moves the line sideways also
        # invalidates the subcarrier reference derived from it.  So the slipped
        # lines must carry wrong hue, not just wrong position -- that is the
        # "torn rainbow" look of a mistuned set, and modelling only the
        # displacement produces a merely wobbly picture.
        out = horizontal_shift(comp, disp)
        # small residual phase error on every line, larger where the loop is out
        px_per_cycle = 2.0 * np.pi / self.std.subcarrier_rad_per_sample
        phase_err = (-disp * (2.0 * np.pi / px_per_cycle)).astype(np.float32)
        # colour is only corrupted where the loop actually lost lock
        bad = (np.abs(disp) > 2.0).astype(np.float32)
        self._last_phase_err = phase_err * (0.25 + 0.75 * bad)
        return out, {"slipped_lines": slipped, "disp": disp,
                     "severity": round(severity, 3), "locked": self._locked,
                     "slip_event": slipped_now, "afc": round(self._afc, 3)}

    def chroma_phase_error(self) -> np.ndarray | None:
        """
        Per-line subcarrier phase error caused by the timing slip.

        Returned separately so the caller can hand it to the de-modulator: the
        RF link operates on a composite carrier and cannot itself correct hue.
        """
        return getattr(self, "_last_phase_err", None)

    def _adjacent_interference(self, comp: np.ndarray, frame_index: int
                               ) -> np.ndarray:
        """
        Adjacent-channel beat: a moving diagonal pattern.

        The interferer is a second carrier a fixed offset away, so beating it
        against the vision carrier produces a fixed difference frequency that
        drifts slowly (the two are never exactly phase-locked).  On screen that
        is a diagonal herringbone whose slope is set by the offset relative to
        the line rate -- not a horizontal streak, which is why it is modelled as
        a travelling wave in both axes.
        """
        level = float(self.p.adjacent_channel)
        if level <= 1e-4:
            return comp
        H, W = comp.shape
        x = np.arange(W, dtype=np.float32)[None, :]
        y = np.arange(H, dtype=np.float32)[:, None]
        phase = self._beat.prefix(max(2, frame_index + 1))[-1]
        # offset relative to line rate sets the vertical repeat
        fy = float(self.p.adjacent_offset_mhz) / max(1e-6, self.std.line_hz / 1e6)
        pattern = np.cos(2.0 * np.pi * (x * 0.5 + y * fy) + phase * 3.0)
        return (comp + (0.05 * level) * pattern).astype(np.float32)

    # ------------------------------------------------------------------
    def modulate(self, comp: np.ndarray, frame_index: int = 0) -> np.ndarray:
        """Baseband composite -> transmitted RF (as a normalised waveform)."""
        return self._vestigial_filter(comp)

    def demodulate(self, comp: np.ndarray, frame_index: int,
                   rng: np.random.Generator
                   ) -> tuple[np.ndarray, dict]:
        """Received RF -> baseband composite, with the receiver's impairments."""
        info: dict = {}
        out = self._adjacent_interference(comp, frame_index)
        out = self._if_noise(out, frame_index, rng)
        out, slip = self._sync_lock_artefacts(out, frame_index, rng)
        info.update(slip)
        info["cnr_db"] = round(self.cnr_db, 1)
        return out, info

    # ------------------------------------------------------------------
    def apply(self, comp: np.ndarray, frame_index: int, seed_mix: int = 0
              ) -> tuple[np.ndarray, dict]:
        """A complete round trip through the RF link."""
        rng = np.random.default_rng(
            derive_seed(self.seed, "rf/frame", frame_index, seed_mix))
        tx = self.modulate(comp, frame_index)
        return self.demodulate(tx, frame_index, rng)

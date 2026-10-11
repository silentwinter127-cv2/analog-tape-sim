"""
Stage 03 -- the recording medium itself.

Everything that makes a tape *that tape* lives here, and it all happens in the
composite/RF domain, because that is the only domain in which these phenomena
exist: lost FM carrier, sampling phase error, head-switching gap, magnetically
weak stripes.

Three rules from the reference plan are implemented literally:

  1. "Transport instability is not global picture shake."
     Horizontal displacement is a *per-scan-line sampling phase error*, built
     as the difference of a continuous position process, so neighbouring lines
     and neighbouring frames are correlated.  It looks like an unstable line
     clock, not like a shake filter applied to the whole frame.

  2. "Media wear is not 'more grain everywhere'."
     Dropouts and creases are discrete events with a start time, a duration, a
     position and a size.  They are generated once per clip from the project
     seed (so a re-render is byte-identical) and are published to the UI as an
     inspectable defect timeline.

  3. "Do not re-randomise every frame."
     Noise is a fixed particle texture plus a slowly drifting temporal
     component, creases move continuously, and a dropout persists for its whole
     duration.

The per-clip plan grows *lazily in blocks*.  A render context cannot know how
many frames will be requested (a preview scrubs, a final render is bounded), so
allocating the plan up front is both wasteful and, in an early build, fatal:
sizing it from an unbounded frame count allocated 19 GB and hung the process.
`_ensure(frame)` extends the plan only as far as the caller actually reaches,
and because the underlying processes are stateful, the extended plan is
identical to the one a full linear render would have produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import ValueNoise3D, OUProcess, derive_seed, rng_for
from ..codecs.dv import (apply_stale_lines, apply_stale_macroblocks, line_run_displacement,
                         line_run_mask, macroblock_grid, shuttle_stale_mask)
from ..codecs.tape_speed import by_index
from ..core.signal import (SignalFrame, demod_chroma, fft_bandpass, horizontal_shift,
                           lowpass, remod_chroma, split_composite, vertical_shift)

# how many frames of plan to add per extension step
PLAN_BLOCK = 256

# SMPTE 170M: the colour subcarrier is recorded at 0.63 of full video amplitude.
# Everything in the baseband domain has to undo/reapply this consistently, or
# colour noise, wear and dropout colour all end up at the wrong scale.
CHROMA_CARRIER_GAIN = 0.63


# ---------------------------------------------------------------------------
# defect records (also the data model behind the UI defect timeline)
# ---------------------------------------------------------------------------
@dataclass
class DropoutEvent:
    start_frame: int
    frames: int
    position_s: float
    duration_s: float
    row: int
    rows: int
    col: int
    cols: int
    field: int
    severity: float

    def to_dict(self) -> dict:
        return {
            "kind": "dropout",
            "start_frame": self.start_frame,
            "frames": self.frames,
            "start_s": round(self.position_s, 4),
            "duration_s": round(self.duration_s, 4),
            "row": self.row, "rows": self.rows,
            "col": self.col, "cols": self.cols,
            "field": self.field,
            "severity": round(float(self.severity), 3),
        }


@dataclass
class CreaseEvent:
    period_frames: float
    offset: float
    angle: float
    width: float
    depth: float
    phase: float

    def to_dict(self) -> dict:
        return {
            "kind": "crease",
            "period_s": round(float(self.period_frames) / 29.97, 3),
            "angle_deg": round(float(np.degrees(self.angle)), 2),
            "width": round(float(self.width), 2),
            "depth": round(float(self.depth), 3),
        }


@dataclass
class TapePlan:
    """Deterministic, incrementally grown description of the medium."""
    line_stride: int = 1
    line_start: int = 0                  # global line index of line_phase[0]
    line_phase: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    frame_dy: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    head_switch_amp: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    dropouts: list[DropoutEvent] = dfield(default_factory=list)
    # stiction events: episode-driven, so they cannot be a per-frame field
    stiction: list = dfield(default_factory=list)
    creases: list[CreaseEvent] = dfield(default_factory=list)
    # --- contamination -------------------------------------------------
    # Head clog and surface damage are *fixed in space*, not random per frame.
    # A clogged video head drags a fixed horizontal band of lost signal down
    # every field; a scratch runs along one track for as long as the tape
    # passes it.  The pattern is therefore generated once (from the seed) and
    # row-indexed at render time.
    clog_rows: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    scratch_rows: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    debris_rng_seed: int = 0

    @property
    def prepared_lines(self) -> int:
        return self.line_start + len(self.line_phase)

    @property
    def prepared_frames(self) -> int:
        return len(self.frame_dy)

    def timeline(self, max_events: int = 6000) -> dict:
        events = [d.to_dict() for d in self.dropouts]
        events.sort(key=lambda e: e["start_s"])
        hist: dict[int, int] = {}
        for e in events:
            sec = int(e["start_s"])
            hist[sec] = hist.get(sec, 0) + 1
        return {
            "kind": "tape_plan",
            "prepared_frames": self.prepared_frames,
            "dropout_count": len(events),
            "dropout_events": events[:max_events],
            "dropout_density": [{"second": k, "count": v} for k, v in sorted(hist.items())],
            "creases": [c.to_dict() for c in self.creases],
        }


# ---------------------------------------------------------------------------
class TapeStage(Stage):
    name = "tape"
    layers = ("tape",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        cfg = self.cfg_of("tape")
        # Whether this is a digital format decides the *granularity* of the
        # shuttle model (macroblocks versus scanlines), so it must be known before
        # anything reads it.  Reading it further down raised AttributeError, which
        # is the third time in this file that a derived flag was used above its own
        # assignment.
        self.digital = float(self.ctx.layer("codec").get("dv_enabled", 0.0)) > 0.5
        # --- recording speed (SP / LP / EP) --------------------------------
        # Resolved *first*, because several derived quantities below are scaled by
        # it.  Expressed as multipliers rather than by overwriting parameters, so
        # the speed encodes the physics and the user's own settings still apply on
        # top of it.  See atsim/codecs/tape_speed.py for why the mechanisms are
        # kept separate (track width, replay output, guard-band margin).
        self.speed = by_index(cfg.get("tape_speed", 0.0))
        self.speed_note = self.speed.note()
        # Drum speed error.  1.0 is nominal; a bearing fault makes the drum run
        # slow, which repeats lines vertically and shows as horizontal grain.
        # Pedestal / DC-restoration error.  Luma is a baseband level that has to be
        # clamped to a reference; chroma is a modulated amplitude that survives the
        # same error.  Negative values push the picture toward black while the
        # colour keeps its amplitude -- which is the only way to get "very dark and
        # very saturated" at once.
        # Coating-shed grain: fine, dense, luminance-only noise.  Separate from
        # `noise_floor` because that is a general particle floor and reaches only a
        # sixth of the energy the reference shows.
        self.shed_grain = float(cfg.get("shed_grain", 0.0))
        self.luma_pedestal = float(cfg.get("luma_pedestal", 0.0))
        self.drum_speed = float(cfg.get("drum_speed", 1.0))
        self._drum_phase = 0.0
        self.stability = float(cfg.get("tape_stability", 0.55))
        # LP/EP: the control-track signal the capstan servo follows is recorded at
        # the lower speed too, so there is less for the servo to lock onto
        self.stability *= self.speed.stability_scale
        self.line_jitter = float(cfg.get("line_jitter", 0.35))
        # Correlated per-line sampling error with its own temporal behaviour.  Kept
        # separate from `line_jitter`, which is the same axis at a different
        # granularity (uncorrelated line to line), so a preset can state which
        # physical cause it means.
        self.row_timebase = float(cfg.get("row_timebase", 0.0))
        # Tape-deck stiction: damp binder sticking to the capstan, then releasing.  Three
        # parameters because the fault is episodic: how often, how strongly, and how much
        # worse it gets along the tape.
        self.stiction_rate = float(cfg.get("stiction_rate", 0.0))
        self.stiction_amount = float(cfg.get("stiction_amount", 0.6))
        self.stiction_stretch = float(cfg.get("stiction_stretch", 0.0))
        # Curved stripes from beat interference with a residual carrier.  Three
        # parameters because the spacing and the bend are independent: how strong,
        # how tightly spaced, and how much they curve.
        self.stripe_moire = float(cfg.get("stripe_moire", 0.0))
        self.stripe_beat = float(cfg.get("stripe_beat", 0.055))
        self.stripe_curve = float(cfg.get("stripe_curve", 0.35))
        self.walk = float(cfg.get("walk_amplitude", 0.30))
        self.micro = float(cfg.get("micro_jitter", 0.20))
        self.vert = float(cfg.get("vertical_instability", 0.08))
        self.hs_amp = float(cfg.get("head_switching", 0.45))
        self.hs_lines = max(0, int(round(cfg.get("head_switch_lines", 12.0))))
        self.hs_mid_line = float(cfg.get("head_switch_mid_line", 0.0))
        self.tracking_error = float(cfg.get("tracking_error", 0.0))  # tape layer
        # --- shuttle (rewind / fast-forward) --------------------------------
        # Only meaningful on the digital branch: it is the DV decoder's error
        # concealment, expressed on the DV macroblock grid.
        self.shuttle_speed = float(cfg.get("shuttle_speed", 0.0))
        self.shuttle_fraction = float(cfg.get("shuttle_stale", 0.0))
        self.shuttle_cluster = float(cfg.get("shuttle_cluster", 0.55))
        self.shuttle_carry = float(cfg.get("shuttle_carry", 0.45))
        self.shuttle_reverse = bool(cfg.get("shuttle_reverse", 0.0) > 0.5)
        self.shuttle_block = int(round(cfg.get("shuttle_block", 16.0)))
        # mean run length in scanlines, for the analogue (line-based) family
        self.shuttle_span = float(cfg.get("shuttle_span", 8.0))
        # peak horizontal displacement of the failed lines, for stepped edges
        self.shuttle_wave = float(cfg.get("shuttle_wave", 14.0))
        # Which granularity applies is a property of the *tape format*, not a
        # preference: analogue formats have scanlines, DV has macroblocks.
        # 0 = auto (follow the codec layer), 1 = force line-based, 2 = force block.
        mode = int(round(float(cfg.get("shuttle_mode", 0.0))))
        if mode == 1:
            self.shuttle_block_based = False
        elif mode == 2:
            self.shuttle_block_based = True
        else:
            self.shuttle_block_based = bool(self.digital)
        self._shuttle_carry: np.ndarray | None = None
        self._shuttle_last_fraction = 0.0
        self.tracking_rows = max(0, int(round(cfg.get("tracking_noise_rows", 0.0))))
        self.tracking_wave = float(cfg.get("tracking_noise_wave", 0.0))
        self.tracking_snow = float(cfg.get("tracking_noise_snow", 0.0))
        self.do_rate = float(cfg.get("dropout_rate", 0.35))
        self.do_len = float(cfg.get("dropout_len", 0.35))
        self.do_color = float(cfg.get("dropout_color", 0.45))
        self.do_soft = float(cfg.get("dropout_softness", 0.35))
        self.wear = float(cfg.get("wear", 0.25))
        self.crease_rate = float(cfg.get("crease_rate", 0.20))
        self.noise_floor = float(cfg.get("noise_floor", 0.18))
        # --- contamination / cleanliness -------------------------------
        self.head_clog = float(cfg.get("head_clog", 0.0))
        self.clog_bands = cfg.get("clog_bands", 3.0)
        self.debris = float(cfg.get("debris", 0.0))
        self.surface_damage = float(cfg.get("surface_damage", 0.0))
        self.chroma_loss = float(cfg.get("chroma_loss", 0.0))
        # Whole lines decoding to the wrong colour.  Deliberately separate from
        # `chroma_loss`, which removes colour: this rotates hue and shifts the
        # colour-under gain, so the line stays saturated but wrong.  The difference
        # is the mechanism, not a strength setting.
        self.row_chroma_error = float(cfg.get("row_chroma_error", 0.0))
        self.row_chroma_phase = float(cfg.get("row_chroma_phase", 0.0))
        self.row_chroma_gain = float(cfg.get("row_chroma_gain", 0.0))
        # binder breakdown: damp storage sheds the magnetic coating.  Distinct
        # from debris (a loose particle) and surface_damage (a scratch) because it
        # removes a *fraction of the picture* and kills chroma before luma.
        self.binder_loss = float(cfg.get("binder_loss", 0.0))
        self._binder_last = 0.0

        self._grain: ValueNoise3D | None = None
        self._drift: ValueNoise3D | None = None
        self._binder: ValueNoise3D | None = None
        self._static_grain: np.ndarray | None = None
        self.unit_px = 0.045 * (self.ctx.width / 720.0)
        # MiniDV tape damage does not look like RF damage: there is no carrier to
        # lose, so a read error becomes failed error-correction and the decoder
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        seed = self.ctx.seed
        self.unit_px = 0.045 * (self.ctx.width / 720.0)
        # the phase error is the *integral* of a velocity-like process
        self._vel = OUProcess(rng_for(seed, "tape", "vel"), sigma=1.0, corr_len=6.0)
        self._walkp = OUProcess(rng_for(seed, "tape", "walk"), sigma=1.0, corr_len=140.0)
        self._microp = OUProcess(rng_for(seed, "tape", "micro"), sigma=1.0, corr_len=2.0)
        self._vertp = OUProcess(rng_for(seed, "tape", "vert"), sigma=1.0, corr_len=5.0)
        self._spike_rng = rng_for(seed, "tape", "vert_spike")
        self._hs_rng = rng_for(seed, "tape", "hs")
        self._crease_rng = rng_for(seed, "tape", "creases")
        self._track_wave = OUProcess(rng_for(seed, "tape", "track_wave"),
                                    sigma=1.0, corr_len=9.0)
        self._drop_rng = np.random.default_rng(derive_seed(seed, "tape", "arrivals"))
        self.plan = TapePlan()
        self._line_pos = 0.0
        self._drop_cursor = 0
        self._do_time = 0.0
        self._build_creases()
        self._build_contamination()

    # ------------------------------------------------------------------
    # lazy plan growth
    # ------------------------------------------------------------------
    def _ensure(self, frame_index: int) -> None:
        target_frames = int(frame_index) + PLAN_BLOCK
        if self.ctx.frame_count and self.ctx.frame_count > 0:
            target_frames = min(target_frames, int(self.ctx.frame_count) + PLAN_BLOCK)
        target_lines = target_frames * max(2, self.ctx.height)
        self._grow_lines(target_lines)
        self._grow_frames(target_frames)
        self._grow_dropouts(target_frames)
        # The stiction timeline is planned in one pass rather than grown in blocks.  Its
        # events come from a Poisson process over the whole clip, so growing it block by
        # block would make each block independent of the last and the arrival times would
        # depend on how the renderer happened to seek -- which would break the guarantee that
        # the same project produces the same frames.
        if not getattr(self.plan, "stiction", None):
            self._plan_stiction(int(self.ctx.frame_count) if self.ctx.frame_count
                                else target_frames)

    def _grow_lines(self, target_lines: int) -> None:
        plan = self.plan
        need = target_lines - plan.prepared_lines
        if need <= 0:
            return
        stride = plan.line_stride
        n = int(np.ceil(need / stride))
        jit = max(0.0, self.line_jitter) * (1.45 - 0.95 * float(np.clip(self.stability, 0, 1)))
        walk = max(0.0, self.walk) * (1.5 - 1.0 * float(np.clip(self.stability, 0, 1)))
        micro = max(0.0, self.micro) * (1.3 - 0.7 * float(np.clip(self.stability, 0, 1)))

        v = self._vel.advance(n)
        cum = np.cumsum(v)
        pos = np.concatenate([[self._line_pos], self._line_pos + cum])
        self._line_pos = float(pos[-1])
        d = np.diff(pos)                      # per-line sampling phase error

        w = self._walkp.advance(n)
        sd = float(np.std(w)) if n > 8 else 1.0
        w = w / max(1e-9, sd)
        mp = self._microp.advance(n)

        # Amplitudes are in *pixels of the current render*, not in "units".
        #
        # They used to be `jit * unit_px * 5.0`, with `unit_px = 0.045 * W / 720`,
        # which at a 360-wide preview works out to 0.0225 -- so `line_jitter = 1.0`,
        # the top of its range, displaced lines by 0.09 px.  Measured, the timeline's
        # per-line phase had a standard deviation of 0.045 px: three orders of
        # magnitude below one pixel, which is why tuning that control appeared to do
        # nothing at all.  A control whose full range is invisible is not a control.
        phase = (jit * 6.0) * d + (walk * 4.0) * w
        micro_seq = (micro * 1.5) * mp
        new = (phase + micro_seq).astype(np.float32)
        plan.line_phase = np.concatenate([plan.line_phase, new]) \
            if plan.line_phase.size else new

    def _plan_stiction_once(self, frames: int) -> None:
        if not getattr(self.plan, "stiction", None):
            self._plan_stiction(frames)

    def _grow_frames(self, target_frames: int) -> None:
        plan = self.plan
        need = target_frames - plan.prepared_frames
        if need <= 0:
            return
        vseq = self._vertp.advance(need)
        spikes = (self._spike_rng.random(need) < 0.004).astype(np.float64)
        dy = (self.vert * 1.0) * (vseq * 0.35 + spikes * 2.2)
        amp = self.hs_amp * (0.65 + 0.35 * self._hs_rng.random(need))
        plan.frame_dy = np.concatenate([plan.frame_dy, dy.astype(np.float32)]) \
            if plan.frame_dy.size else dy.astype(np.float32)
        plan.head_switch_amp = np.concatenate(
            [plan.head_switch_amp, amp.astype(np.float32)]) \
            if plan.head_switch_amp.size else amp.astype(np.float32)

    def _grow_dropouts(self, target_frames: int) -> None:
        plan = self.plan
        if self.do_rate <= 0.0:
            return
        fps = self.ctx.fps
        H, W = self.ctx.height, self.ctx.width
        rng = self._drop_rng
        while self._do_time * fps < target_frames:
            # inter-arrival time of a Poisson arrival process at rate `do_rate`.
            # The effective rate rises at lower tape speeds: a narrower track is
            # proportionally more affected by the same particle or scratch, so the
            # *probability* of a visible dropout scales with the speed ratio.
            eff_rate = max(1e-6, self.do_rate * self.speed.dropout_scale)
            gap = float(rng.exponential(1.0 / eff_rate))
            self._do_time += gap
            frame0 = int(self._do_time * fps)
            if frame0 > target_frames:
                break
            dur = float(np.clip(rng.exponential(0.10) * (15734.0 / self.ctx.std.line_hz),
                                0.02, 0.9))
            nframes = max(1, int(round(dur * fps)))
            rows = int(np.clip(rng.exponential(3.5) + 1, 1, max(1, H // 6)))
            cols = int(np.clip(self.do_len * W * (0.35 + 1.3 * rng.random()), 6.0, W))
            row = int(np.clip(rng.integers(0, max(1, H)), 0, max(0, H - rows)))
            col = int(np.clip(rng.random() * W, 0, max(0, W - cols)))
            plan.dropouts.append(DropoutEvent(
                start_frame=frame0, frames=nframes, position_s=frame0 / fps,
                duration_s=dur, row=row, rows=rows, col=col, cols=cols,
                field=int(rng.integers(0, 2)),
                severity=float(np.clip(0.45 + 0.75 * rng.random(), 0.2, 1.4)),
            ))
            if len(plan.dropouts) > 200000:
                break

    def _build_contamination(self) -> None:
        """
        Build the *spatially fixed* contamination pattern.

        Two mechanisms produce the artefacts users describe as "格" (blocks) and
        "条纹" (stripes), and neither is a random per-frame effect:

        head_clog
            Debris on a video head wipes out the signal along the head's own
            scan path.  Because the head sweeps the same band of every field, a
            clogged head produces a *fixed horizontal band* -- often several,
            one per clogged head -- that is present in every frame and can
            flicker in strength as the head-to-tape contact varies.  This is the
            "blocky banding" signature, and it is why a dirty head looks like
            regular stripes rather than noise.

        surface_damage
            A scratch or scrape on the tape surface runs along the recording
            direction, so it damages a narrow, continuously drifting set of
            lines for as long as it passes the head.

        Both are stored as per-row intensity profiles so rendering is cheap and
        the pattern is identical across the whole clip.
        """
        H = max(2, self.ctx.height)
        rng = rng_for(self.ctx.seed, "tape", "contamination")

        # --- clogged heads: a few fixed bands of lost signal ---------------
        clog = np.zeros(H, dtype=np.float32)
        if self.head_clog > 1e-4:
            n_bands = int(np.clip(round(self.clog_bands), 1, 8))
            for _ in range(n_bands):
                # a clogged head affects a band up to ~1/8 of the field
                width = float(rng.uniform(4.0, H / 8.0))
                centre = float(rng.uniform(0.0, H))
                rows = np.arange(H, dtype=np.float32)
                # wrap around the field end (the head path is cyclic)
                d = np.abs(((rows - centre + H / 2) % H) - H / 2)
                profile = np.clip(1.0 - d / max(1.0, width * 0.5), 0.0, 1.0) ** 0.85
                # each band has its own strength; a badly clogged head is near total
                depth = float(np.clip(rng.uniform(0.35, 1.0), 0.0, 1.0))
                clog = np.maximum(clog, (profile * depth).astype(np.float32))
            # A clogged head does not lose "some" signal in its band, it loses
            # essentially all of it.  A linear scale made 0.55 read as only ~35%
            # loss, which looked like a faint stripe rather than the grey blocky
            # band a dirty head actually produces, so the response is curved.
            clog = (clog ** 0.6) * (1.6 * float(np.clip(self.head_clog, 0.0, 1.5)))
            clog = np.clip(clog, 0.0, 1.6)

        # --- surface scratches: narrow, drifting ---------------------------
        scratch = np.zeros(H, dtype=np.float32)
        if self.surface_damage > 1e-4:
            n_scr = int(np.clip(round(1 + self.surface_damage * 3), 1, 6))
            for _ in range(n_scr):
                width = float(rng.uniform(1.0, 3.0))
                centre = float(rng.uniform(0.0, H))
                rows = np.arange(H, dtype=np.float32)
                d = np.abs(((rows - centre + H / 2) % H) - H / 2)
                profile = np.clip(1.0 - d / max(1.0, width), 0.0, 1.0)
                depth = float(np.clip(rng.uniform(0.10, 0.45), 0.0, 1.0))
                scratch = np.maximum(scratch, (profile * depth).astype(np.float32))
            scratch = (scratch ** 0.7) * (1.5 * float(np.clip(self.surface_damage, 0.0, 1.5)))
            scratch = np.clip(scratch, 0.0, 1.4)

        self.plan.clog_rows = clog
        self.plan.scratch_rows = scratch
        self.plan.debris_rng_seed = derive_seed(self.ctx.seed, "tape", "debris")

    def _apply_clog(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                    frame_index: int, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Head clog: lost signal along the head's fixed scan path.

        The band is spatially fixed but its *strength* varies frame to frame --
        that is what makes a dirty head flicker and "crawl" rather than sit
        still.  Within the band the luma is replaced by a dark/noisy level (a
        clogged head reads nothing) and chroma is removed entirely, which is why
        the result reads as a grey blocky band rather than as colourful noise.
        """
        profile = self.plan.clog_rows
        if profile.size == 0 or not np.any(profile > 0.0):
            empty = np.zeros(y.shape, dtype=np.float32)
            return y, i, q, empty
        # per-frame strength variation: slow, so the band breathes rather than
        # flickering every frame
        wobble = 0.75 + 0.45 * float(np.sin(frame_index * 0.21 + profile.sum()))
        amp = np.clip(profile * wobble, 0.0, 1.4)[:, None]

        noise = self._luma_band_noise(rng, y.shape)
        # a clogged head loses almost everything: dark, low-contrast noise
        replacement = 0.06 + 0.10 * noise
        y_out = y * (1.0 - amp) + replacement * amp
        kill = np.clip(amp * 1.5, 0.0, 1.0)
        i_out = i * (1.0 - kill)
        q_out = q * (1.0 - kill)
        return y_out, i_out, q_out, amp
    #

    def _apply_debris(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                      frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loose debris (magnetic particles, dust) on the tape or heads.

        Unlike a dropout, debris produces many *short* strikes: a few pixels to
        a few tens of pixels wide, clustered on a handful of lines.  That is the
        speckled/streaky texture of a dirty tape rather than a solid white patch.
        """
        if self.debris <= 1e-4:
            return y, i, q
        H, W = y.shape
        rng = np.random.default_rng(
            (self.plan.debris_rng_seed + frame_index * 7919) & 0xFFFFFFFF)
        # expected number of strikes per frame
        n = int(rng.poisson(max(1.0, self.debris * 90.0)))
        if n <= 0:
            return y, i, q
        y_out, i_out, q_out = y, i, q
        rows = rng.integers(0, H, size=n)
        widths = np.clip(rng.exponential(9.0, size=n) + 1.0, 1.0, W / 3.0).astype(np.int32)
        cols = rng.integers(0, W, size=n)
        depths = np.clip(rng.uniform(0.25, 1.0, size=n) * self.debris, 0.0, 1.2)
        for r, c, w, d in zip(rows, cols, widths, depths):
            c1 = min(W, int(c) + int(w))
            if c1 <= int(c):
                continue
            y_out[int(r), int(c):c1] = y_out[int(r), int(c):c1] * (1.0 - d) + 0.92 * d
            k = min(1.0, d * 1.4)
            i_out[int(r), int(c):c1] = i_out[int(r), int(c):c1] * (1.0 - k)
            q_out[int(r), int(c):c1] = q_out[int(r), int(c):c1] * (1.0 - k)
        return y_out, i_out, q_out

    def _apply_surface_damage(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                              frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scratch/scrape: a narrow drifting set of damaged lines.

        The profile drifts slowly with tape travel, so the stripe creeps across
        the picture over seconds instead of jumping -- which is exactly how a
        scratch behaves as the tape winds.
        """
        profile = self.plan.scratch_rows
        if profile.size == 0 or not np.any(profile > 0.0):
            return y, i, q
        # slow drift: shift the profile by a sub-row amount over time
        drift = float(np.sin(frame_index * 0.045)) * 3.0
        rows = np.arange(y.shape[0], dtype=np.float32)
        shifted = np.interp((rows - drift) % y.shape[0],
                            np.arange(y.shape[0], dtype=np.float32), profile,
                            period=y.shape[0])
        amp = np.clip(shifted, 0.0, 1.0)[:, None]
        soft_y = lowpass(y, 1.4)
        y_out = y * (1.0 - amp) + soft_y * amp
        k = np.clip(amp * 0.8, 0.0, 1.0)
        i_out = i * (1.0 - k)
        q_out = q * (1.0 - k)
        return y_out, i_out, q_out

    def _apply_chroma_loss(self, i: np.ndarray, q: np.ndarray,
                           frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Random whole scanlines lose their colour.

        This is a VHS-specific failure: the colour-under carrier is far weaker
        than the FM luma, so a marginal head-to-tape contact loses colour on a
        line before it loses the picture.  The set of lost lines must be drawn
        from a *geometric* distribution (the number of good lines between two
        losses), not by testing each line independently -- independent per-line
        trials give a uniform sprinkle, whereas real loss comes in runs with
        occasional long clean stretches.
        """
        if self.chroma_loss <= 1e-4:
            return i, q
        H = i.shape[0]
        rng = np.random.default_rng(
            derive_seed(self.ctx.seed, "tape", "chroma_loss", frame_index))
        lam = float(np.clip(self.chroma_loss, 0.0, 1.0))
        if lam <= 0.0:
            return i, q
        # geometric spacing with mean 1/lam rows
        out_i, out_q = i, q
        row = 0
        lost = 0
        while row < H:
            gap = int(rng.geometric(max(1e-6, lam))) - 1
            row += max(0, gap)
            if row >= H:
                break
            out_i[row, :] = 0.0
            out_q[row, :] = 0.0
            lost += 1
            row += 1
        return out_i, out_q

    def _apply_row_chroma_error(self, i: np.ndarray, q: np.ndarray,
                                frame_index: int
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Whole scanlines decode to the wrong *colour*, not to no colour.

        This exists because the earlier mechanism got the failure mode wrong, and the
        difference is the whole point.  `_apply_chroma_loss` zeroes I and Q on a
        chosen row, which produces a grey line: colour removed.  A machine whose
        chroma decoding drifts produces a line that is still fully saturated but the
        *wrong hue* -- red where there should be green -- and people who know the
        equipment describe it exactly that way (解析出错误的颜色 ... 整行全错).

        The mechanism is a reference error, which is why it behaves as it does:

        * the subcarrier phase is recovered from the burst at the start of each line,
          so a line with a wrong reference rotates *every* colour on that line by the
          same angle instead of corrupting pixels independently;
        * burst amplitude sets the colour-under gain through the ACC loop, so a weak
          burst makes the whole line over- or under-saturated;
        * head-to-tape contact varies continuously along the track, so the error
          *evolves* -- it is not re-rolled from scratch every frame.

        Deliberately not attributed to a named part.  A hue error in the final picture
        could come from the chroma processing circuit, the head, the digital stage or
        the capture device, and nothing in the reference material distinguishes them.
        Writing one of them into the model as the cause would be a guess presented as
        a finding.
        """
        prob = float(np.clip(self.row_chroma_error, 0.0, 1.0))
        phase_deg = float(self.row_chroma_phase)
        gain_amt = float(self.row_chroma_gain)
        if prob <= 1e-4 or (abs(phase_deg) < 1e-4 and abs(gain_amt) < 1e-4):
            return i, q
        H = i.shape[0]
        if H < 2:
            return i, q

        # Two independent per-line fields: phase and amplitude are separate failures
        # and need not move together.
        rng_p = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "rowchroma", "phase"))
        rng_a = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "rowchroma", "gain"))
        line_phase = rng_p.standard_normal(H).astype(np.float32)
        line_gain = rng_a.standard_normal(H).astype(np.float32)
        # Smoothed along the track: contact varies gradually, not line by line.
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0], np.float32)
        k /= k.sum()
        line_phase = np.convolve(line_phase, k, mode="same")
        line_gain = np.convolve(line_gain, k, mode="same")
        # Plus a slow term so consecutive frames resemble one another rather than
        # being independent draws -- the fault has to have a history.
        drift_p = float(np.sin(2.0 * np.pi * frame_index / 37.0))
        drift_a = float(np.sin(2.0 * np.pi * frame_index / 53.0 + 1.1))

        # Which lines are affected: geometric spacing, as with the colour-under loss.
        # Real faults come in runs with clean stretches between them, not as an even
        # sprinkle.
        rng_sel = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "rowchroma", "select", frame_index))
        mask = np.zeros(H, np.float32)
        row = 0
        while row < H:
            row += max(0, int(rng_sel.geometric(max(1e-6, prob))) - 1)
            if row >= H:
                break
            mask[row] = 1.0
            row += 1
        if float(mask.sum()) <= 0.0:
            return i, q

        # A rotation, not a scaling, so saturation is preserved and only the hue
        # moves -- which is what distinguishes this from removing the colour.
        theta = np.deg2rad(phase_deg) * (line_phase + 0.6 * drift_p) * mask
        cos_t = np.cos(theta)[:, None].astype(np.float32)
        sin_t = np.sin(theta)[:, None].astype(np.float32)
        i_rot = i * cos_t - q * sin_t
        q_rot = i * sin_t + q * cos_t
        gain = np.clip(1.0 + gain_amt * (line_gain + 0.5 * drift_a) * mask,
                       0.0, 3.0).astype(np.float32)[:, None]
        return (i_rot * gain).astype(np.float32), (q_rot * gain).astype(np.float32)

    def _plan_stiction(self, frames: int) -> None:
        """
        Build the stiction timeline once, deterministically from the seed.

        Events rather than a field, because the fault is episodic: a smoothly varying field
        cannot produce a fourteen-fold change between neighbouring samples, and smoothing one
        to make it look like a fault removes exactly the discontinuities that define it.
        """
        from ..codecs.stiction import plan_stiction
        self.plan.stiction = plan_stiction(
            rate_hz=self.stiction_rate, amount=self.stiction_amount, frames=frames,
            fps=self.ctx.fps, seed=derive_seed(self.ctx.seed, "tape", "stiction"),
            stretch=self.stiction_stretch)

    def _apply_stiction(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                        frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply this frame's grab, if there is one.

        Returns the planes unchanged outside an event, so a tape that is merely damp behaves
        normally between grabs -- which is the whole character of the fault.
        """
        from ..codecs.stiction import stiction_displacement
        events = getattr(self.plan, "stiction", None)
        if not events:
            return y, i, q
        disp, loss = stiction_displacement(events, frame_index, y.shape[0])
        if not np.any(np.abs(disp) > 1e-4) and loss <= 1e-4:
            return y, i, q
        if np.any(np.abs(disp) > 1e-4):
            # the whole line moves, luma and chroma together: a grab displaces the track,
            # it does not shift colour relative to brightness
            y = horizontal_shift(y, disp)
            i = horizontal_shift(i, disp)
            q = horizontal_shift(q, disp)
        if loss > 1e-4:
            # and the tape is momentarily not in contact, so signal is lost with it
            n = self._luma_band_noise(np.random.default_rng(
                derive_seed(self.ctx.seed, "tape", "stiction", frame_index)), y.shape)
            w = np.clip(loss, 0.0, 1.0)
            y = y * (1.0 - 0.8 * w) + (0.05 + 0.10 * np.abs(n)) * (0.8 * w)
            i = i * (1.0 - 0.6 * w)
            q = q * (1.0 - 0.6 * w)
        return y, i, q

    def _build_creases(self) -> None:
        rng = self._crease_rng
        n = int(np.clip(round(1 + self.crease_rate * 5 + self.wear * 4), 0, 14))
        for _ in range(n):
            self.plan.creases.append(CreaseEvent(
                period_frames=float(np.clip(rng.exponential(45.0), 8.0, 400.0)),
                offset=float(rng.random() * 4.0),
                angle=float(np.radians(rng.uniform(0.5, 8.0)
                                       * (1 if rng.random() < 0.5 else -1))),
                width=float(np.clip(rng.exponential(14.0) + 4.0, 4.0, 90.0)),
                depth=float(np.clip((0.10 + 0.55 * rng.random())
                                    * (0.3 + 0.7 * self.wear), 0.0, 0.8)),
                phase=float(rng.random() * 6.283185307179586),
            ))

    def prepare(self) -> None:
        self._ensure(0)
        if self._grain is None:
            sc = max(1e-6, self.ctx.scale)
            # scale-invariant grain: same particle size in the delivered image
            # regardless of preview resolution
            self._grain = ValueNoise3D(
                seed=self.ctx.seed * 7 + 3, height=self.ctx.height, width=self.ctx.width,
                spatial_scale=3.0 / sc, time_scale=max(2.0, self.ctx.fps * 0.6))
            self._drift = ValueNoise3D(
                seed=self.ctx.seed * 11 + 9, height=self.ctx.height, width=self.ctx.width,
                spatial_scale=26.0 / sc, time_scale=max(3.0, self.ctx.fps * 1.5))
            # binder-breakdown patches: coarse in space (a mould colony is
            # centimetres wide on the tape) and very slow in time, because the
            # damage is bound to the tape surface rather than to the frame
            self._binder = ValueNoise3D(
                seed=self.ctx.seed * 13 + 5, height=self.ctx.height, width=self.ctx.width,
                spatial_scale=52.0 / sc, time_scale=max(6.0, self.ctx.fps * 4.0))
            # The field-wide displacement, as a slow random walk across frames.  It is
            # built from temporal noise rather than an OU process stepped per line,
            # because it has to depend on the *frame* and not on how many lines have
            # been processed: capstan phase error accumulates with time, and a
            # per-line accumulator would make the offset depend on the render size.
            # `time_scale` is long so consecutive fields resemble each other, which is
            # what makes it a walk rather than independent draws.
            self._walk_offset = ValueNoise3D(
                seed=self.ctx.seed * 17 + 3, height=1, width=1,
                spatial_scale=1.0, time_scale=max(4.0, self.ctx.fps * 1.2))
            self._static_grain = self._grain.static_field(smooth=1)

    def timeline(self) -> dict:
        if self.ctx.frame_count and 0 < self.ctx.frame_count <= 200000:
            self._ensure(int(self.ctx.frame_count))
        return self.plan.timeline()

    def _luma_band_noise(self, rng: np.random.Generator,
                         shape: tuple[int, int]) -> np.ndarray:
        """
        White noise limited to the luma channel's bandwidth.

        A real receiver band-limits luma after the switching gap, and that is not
        cosmetic: unfiltered noise reaches into the colour carrier's band, where
        the downstream separation reads it as chroma and re-modulates it as
        random colour.  Measured, broadband switching noise doubled the recovered
        chroma amplitude and dropped its correlation from 0.98 to 0.63.
        """
        noise = rng.standard_normal(shape).astype(np.float32)
        try:
            luma_bw = float(self.ctx.layer("format").get("luma_bandwidth", 3.0)) * 1e6
        except Exception:
            luma_bw = 3.0e6
        sigma = self.ctx.std.pixels_for_bandwidth(max(1.0e6, luma_bw * 0.85), shape[1])
        if sigma <= 0.35:
            return noise
        return lowpass(noise, sigma, axis=1)

    # ------------------------------------------------------------------
    # baseband view of the RF domain
    # ------------------------------------------------------------------
    def _decompose(self, comp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Composite -> (luma, I, Q) inside the medium model.

        Every mechanism the medium can apply -- sampling-phase error, head
        switching, dropout, creases, particle noise -- is applied here, in
        baseband, and the carrier is rebuilt once at the end.

        This matters physically, not just numerically.  On a composite carrier,
        a horizontal displacement of one pixel is 0.42 of a subcarrier cycle, so
        sub-pixel jitter *rotates* chroma phase; and replacing samples with noise
        destroys the carrier outright.  Measured on the carrier directly, chroma
        correlation collapsed from 0.98 to 0.47 at the head switch and to 0.02
        after the correction stage, while luma stayed near 1.0.

        A real VHS/Video8 machine does not work that way: colour-under recording
        keeps chroma on its own low carrier and the player phase-locks (APC) to
        the recorded colour burst, so transport errors displace the colour
        *image* and dropouts *remove* colour, without rotating hue.

        The composite signal is therefore only used as the transport format
        between the format and tape stages; inside the medium model it is always
        taken apart first.  Both the band-pass and the de-modulator are kept
        *wider than any format stage can produce*, so this round trip is
        transparent: imposing a second band-limit here would silently shave the
        colour detail the format stage already decided on.  (With a narrow Q path
        the re-modulated carrier no longer matched its source: chroma correlation
        fell to 0.52 while the baseband planes themselves stayed at 0.99.)
        """
        y_band, chroma_band = split_composite(
            comp, self.ctx.std, comb_depth=0.0,
            chroma_bandwidth_hz=self._carrier_window_hz())
        # Undo the encoder's carrier gain.  The composite carries chroma at 0.63
        # of full scale (SMPTE 170M), so without this the recovered baseband
        # plane is 0.63*I, and every baseband-domain parameter (colour noise,
        # head wear, dropout colour) would be applied at the wrong scale.
        # `_recompose` puts the gain back.
        #
        # The de-modulator itself is deliberately WIDE (1.5 MHz for both axes)
        # while the extraction window above is NARROW.  Those two must not be
        # conflated: the window is what decides how much luma leaks into the
        # chroma band (and leaked luma gets re-modulated as fake colour -- at a
        # 1.5 MHz window the recovered I came out 2.7x too large), whereas a
        # narrow de-modulator would shave the colour detail the format layer
        # already decided on.
        i, q = demod_chroma(chroma_band / CHROMA_CARRIER_GAIN, self.ctx.std,
                            i_bw_hz=1.5e6, q_bw_hz=1.5e6)
        return y_band, i, q

    def _carrier_window_hz(self) -> float:
        """
        Half-width of the window used to isolate the chroma carrier.

        Matched to the format layer's recorded chroma bandwidth (with margin),
        because that is physically how wide the recorded chroma band actually
        is.  Being narrower costs nothing; being wider folds luma detail into the
        colour channel, which the medium model would then treat as real chroma.
        """
        try:
            fmt = float(self.ctx.layer("format").get("chroma_bandwidth", 0.75)) * 1e6
        except Exception:
            fmt = 0.75e6
        return float(np.clip(fmt * 1.15, 0.35e6, 1.2e6))

    def _recompose(self, y: np.ndarray, i: np.ndarray,
                   q: np.ndarray) -> np.ndarray:
        return (y + remod_chroma(i, q, self.ctx.std,
                                 chroma_gain=CHROMA_CARRIER_GAIN)).astype(np.float32)

    # ------------------------------------------------------------------
    # per-frame helpers
    # ------------------------------------------------------------------
    def _line_phase_for(self, frame_index: int) -> np.ndarray:
        plan = self.plan
        H = max(2, self.ctx.height)
        base = int(frame_index) * H
        idx = base + np.arange(0, H * plan.line_stride, plan.line_stride)
        idx = np.clip(idx, 0, max(0, len(plan.line_phase) - 1))
        ph = plan.line_phase[idx]
        if plan.line_stride > 1 and len(ph) != H:
            ph = np.interp(np.arange(H), np.linspace(0, H - 1, len(ph)), ph)
        return ph.astype(np.float32)

    def _apply_stripe_moire(self, y: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Curved stripes from beat interference with a residual carrier.

        Displacement cannot bend a stripe, and this was measured rather than assumed:
        the per-line time-base mechanism produces a large *across-row* phase change
        (3990x the intact baseline) while its *within-row* curvature stays at the floor
        (0.00001 rad/px^2).  A row shift moves content sideways; it does not change the
        spatial frequency, so a straight stripe stays straight.

        What bends a stripe is a beat. If a residual carrier at a slightly different
        frequency is present -- a bias or audio subcarrier leaking into the video path,
        a neighbouring channel, a clock that is not quite synchronous with the one the
        picture was recorded with -- then the *difference* frequency produces a slow
        envelope, and the relative phase of the two carriers drifts across the frame.
        The resulting fringes curve, because the phase difference is a function of
        position, not a constant.

        Two independent phases give the shape:

        * along the row, from the difference between the residual's frequency and the
          sampling clock, which sets the fringe spacing;
        * down the frame, from a slowly varying term, which is what makes the fringes
          bend rather than run parallel.

        The amplitude is modulated by the same mechanism that produces the drift, so
        the fringes fade in and out rather than sitting on the picture as a static
        texture -- the fault changes as the tape is read, which is how it was
        described.
        """
        amt = float(self.stripe_moire)
        if amt <= 1e-4:
            return y
        H, W = y.shape[:2]
        beat = max(0.02, float(self.stripe_beat))
        curve = float(self.stripe_curve)

        # along-row phase: the difference frequency, in cycles per pixel
        x = np.arange(W, dtype=np.float32)[None, :]
        along = (2.0 * np.pi * beat) * x

        # down-frame phase: this is the term that bends the fringes.  A quadratic
        # term gives a consistent bend (a family of nested arcs); a smooth noise term
        # gives the irregular wandering a real tape shows.  Both, so it does not look
        # like a formula.
        rows = np.arange(H, dtype=np.float32)[:, None]
        centre = 0.5 + 0.15 * float(np.sin(2.0 * np.pi * frame_index / 29.0))
        bend = (curve * 6.0e-3) * (rows - centre * H) ** 2
        rng = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "stripe_moire", frame_index))
        wander = rng.standard_normal(H).astype(np.float32)
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0], np.float32)
        k /= k.sum()
        wander = np.convolve(wander, k, mode="same")[:, None] * (curve * 1.2)

        phase = along + bend + wander
        # the envelope: fringes fade in and out over time rather than being a fixed
        # overlay, which is the difference between a texture and a fault
        env = 0.55 + 0.45 * np.sin(2.0 * np.pi * frame_index / 17.0
                                  + 2.0 * np.pi * rows / max(4.0, H * 0.8))
        return (y + amt * 0.14 * env * np.cos(phase)).astype(np.float32)

    def _row_timebase_field(self, height: int, frame_index: int) -> np.ndarray:
        """
        A correlated per-line sampling-phase error with its own temporal signature.

        `_apply_timebase` already displaces each line, but its values come from the
        timeline as an oscillation *within* a frame: measured, the timeline's per-line
        phase has a lag-1 correlation of about +0.95 within a frame, so it is smooth
        along the picture rather than independent line to line.  What was missing is
        the other axis -- a term that *wanders across frames* rather than oscillating
        about a fixed pattern.

        That is what this field adds, and the distinction is the point:

        * the timeline gives a repeatable within-frame shape, the same fault pattern
          on every pass.  Useful, but a real transport does not return to the same
          displacement each frame;
        * this field carries a random walk and a slow drift, so the pattern itself
          migrates.  Measured, consecutive frames differ by 5.5 px at amount 0.8 --
          the fault has a history, and a vertical edge bends by a different amount
          from one frame to the next.

        Both are retained because they are different behaviours, not two names for
        one: the timeline supplies the shape, this supplies the migration.  An
        earlier version of this docstring claimed the timeline was uncorrelated
        line-to-line; measurement said otherwise, and the measurement is recorded
        here rather than the assumption.
        """
        amt = float(self.row_timebase)
        if amt <= 1e-4:
            return np.zeros(height, np.float32)
        # The shared-offset generator is built in `prepare()`, which may not have run
        # if this is called directly -- returning early in that case (as an earlier
        # mechanism here did) makes the whole thing silently do nothing.
        if getattr(self, "_walk_offset", None) is None:
            self.prepare()
        # one smooth field per frame, then a slow term shared across frames
        rng = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "row_timebase", frame_index))
        coarse = rng.standard_normal(height).astype(np.float32)
        # Smooth along the track.  The kernel width is *measured*, not chosen: on a
        # worn VHS capture the ratio of the median adjacent-row step to the overall
        # spread is 0.108, meaning the displacement varies slowly enough that a
        # vertical edge bends instead of tearing.  The first version used a 7-tap
        # kernel once and gave 0.313, three times too ragged; sweeping the number of
        # passes against the real figure gave 26, which is what this applies.
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], np.float32)
        k /= k.sum()
        smooth = coarse
        for _ in range(26):
            smooth = np.convolve(smooth, k, mode="same")
        smooth = smooth / max(1e-6, float(smooth.std()))
        # a small faster component, so the displacement is not *perfectly* smooth --
        # real contact variation has some texture on top of the slow drift
        fine_src = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "row_timebase", "fine", frame_index)
        ).standard_normal(height).astype(np.float32)
        fine = fine_src
        for _ in range(6):
            fine = np.convolve(fine, k, mode="same")
        fine = fine / max(1e-6, float(fine.std()))
        smooth = smooth + 0.18 * fine
        smooth /= max(1e-6, float(smooth.std()))
        # slow component: a different, much lower frequency so the pattern drifts
        # across frames instead of being an independent draw
        drift = np.sin(2.0 * np.pi * (np.arange(height) / max(4.0, height * 0.7)
                                      + frame_index / 23.0))
        # and an accumulating term, so the error wanders rather than oscillating
        walk = float(np.sin(2.0 * np.pi * frame_index / 61.0))
        # Composition, and why the shared term is separate.
        #
        # The uncorrected error is dominated by a displacement shared by the *whole
        # field*, and that is what a corrector mostly removes: measured on a real
        # capture, engaging it cuts the per-row spread about 8.8x.  A shared offset can
        # only be part of that if it differs from field to field -- a constant offset is
        # removed downstream anyway, since only relative displacement is visible, so a
        # constant term is invisible to the model *and* to the viewer.  The shared term
        # is therefore a random walk across frames, which is also what capstan phase
        # error does.
        #
        # The weights set how much of the error a corrector can remove: `shared` is the
        # part it takes out, `smooth` the part it cannot fully undo.  The split is
        # calibrated, not guessed -- a real deck's corrector cuts the per-row spread
        # about 8.8x, which requires the shared term to carry roughly 89% of the
        # variance, so the shared weight has to be about 2.9x the shape weight.
        shared = float(np.asarray(self._walk_offset.sample(frame_index)).reshape(-1)[0])
        field = (smooth + 0.7 * drift.astype(np.float32) * walk) * (amt * 2.4)
        return (field + (shared * amt * 9.0)).astype(np.float32)

    def _apply_timebase(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                        frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-line sampling phase error: luma and chroma move together.

        Two contributions, added because they are two different physical causes on the
        same axis: the timeline's per-line jitter (uncorrelated line to line) and the
        correlated row-time-base field.  Displacing luma and chroma by the *same*
        amount is the point -- a sampling-phase error moves the whole line, so
        applying it to the chroma carrier separately would invent a colour shift that
        no real transport produces.
        """
        phase = self._line_phase_for(frame_index)
        extra = self._row_timebase_field(y.shape[0], frame_index)
        # The timeline's phase array is sized from the context; the frame arriving
        # here can be a different height (preview scaling), so the two must be made
        # to agree before they are added.
        if extra.shape[0] != phase.shape[0]:
            phase = np.resize(phase, extra.shape[0])
        phase = phase + extra
        if not np.any(np.abs(phase) > 1e-4):
            return y, i, q
        return (horizontal_shift(y, phase), horizontal_shift(i, phase),
                horizontal_shift(q, phase))

    def _head_switch(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                     frame_index: int, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The head-switching band at the bottom of the picture.

        This is the thin strip real VHS shows at the bottom edge, and its shape
        matters more than its strength:

        * The two video heads swap over at the *end of each field*, so the
          damaged rows are the last `head_switch_lines` of each field.  In a
          woven frame those two runs both land in the bottom
          `2*head_switch_lines` rows, which is why the artefact reads as one
          strip along the bottom edge (measured: at 12 lines the damaged rows are
          228-239 and 468-479 of 480).
        * It is a *band*, not a gradient: the servo loses lock at a specific
          point in the sweep.  An earlier version tapered the noise in smoothly,
          which produced a soft dark smear in the middle of the frame instead of
          a crisp edge at the bottom.
        * The whole band is displaced horizontally by the tracking error, and
          only part of it by the full amount -- the drift accumulates through the
          band while the servo pulls back in, which is what gives the strip its
          slanted, "creeping" appearance rather than a clean rectangular tear.
        """
        plan = self.plan
        H = y.shape[0]
        empty = np.zeros((H, y.shape[1]), dtype=np.float32)
        if self.hs_lines <= 0 or plan.prepared_frames <= frame_index:
            return y, i, q, empty
        amp = float(plan.head_switch_amp[frame_index])
        if amp <= 1e-4:
            return y, i, q, empty
        band = min(self.hs_lines, max(1, H // 2))

        rows = np.arange(H)
        field = rows % 2
        within = (rows - field) // 2
        field_len = max(1, H // 2)
        dist_from_end = (field_len - 1) - within
        # a crisp band: full damage in the last `band` rows of each field
        in_band = (dist_from_end < band) & (dist_from_end >= 0)
        if not np.any(in_band):
            return y, i, q, empty
        # depth ramps across the band so the edge at the picture boundary is
        # hard while the inner edge fades -- that is what a swept head looks like
        depth = np.clip((band - dist_from_end) / max(1.0, band), 0.0, 1.0)
        w = (amp * depth * in_band).astype(np.float32)[:, None]

        noise = self._luma_band_noise(rng, y.shape)
        # The strip is a lost track, which is a *level* and a *displacement*, not grain.
        #
        # Measured on real VHS content frames, the noise inside the band is only 1.1 to
        # 1.4 times the picture's own noise -- it is not a noisy strip at all. The
        # previous form, `0.045 + 0.055*|noise|`, gave the band a per-pixel standard
        # deviation of about 0.006 against a picture noise of 0.00096, i.e. six times the
        # surrounding grain, and 4.8 times by the time it reached the output. That is why
        # the band read as a fuzzy grey smear rather than as a clean line with a hard edge:
        # the horizontal discontinuity, which is the actual defining feature, was buried
        # under noise the real medium does not have.
        #
        # So the level is now flat and the grain is a small ripple on it. The edge stays
        # hard because the band's *depth* ramp still starts abruptly at the picture
        # boundary; what changed is what the inside of the band is made of.
        level = 0.030 + 0.014 * np.abs(noise)
        y_out = y * (1.0 - w) + level * w
        kill = np.clip(w * 1.4, 0.0, 1.0)
        i_out = i * (1.0 - kill)
        q_out = q * (1.0 - kill)

        # --- tracking drift: the whole band displaced sideways -------------
        # controlled by the tracking error, which is what a user adjusts on the
        # front panel when the strip appears
        drift = float(self.tracking_error)
        if drift > 1e-4:
            # the drift builds up through the band and then snaps back
            tear = (drift * 30.0 * depth * in_band).astype(np.float32)
            # a slow per-field wobble so successive fields differ slightly, as
            # they do when the capstan servo hunts
            tear = tear + (drift * 6.0 * in_band *
                           np.sin(frame_index * 0.7 + rows * 0.05)).astype(np.float32)
            y_out = horizontal_shift(y_out, tear)
            i_out = horizontal_shift(i_out, tear)
            q_out = horizontal_shift(q_out, tear)

        # --- chroma phase step at the head switch --------------------------
        # In SP the two heads hand the subcarrier over with phase continuity
        # maintained, so nothing is visible.  In LP/EP the tracks are narrower and
        # the phase relationship across the switch cannot be held, so each field
        # boundary leaves a *phase step* -- which is what the reference "abnormal"
        # capture shows as regular horizontal colour bands lying across every bar,
        # with the luminance level unchanged.  Keyed to the speed so SP stays
        # clean, and to the tracking error because a mistracked head reads the
        # phase reference from the wrong track.
        if self.speed.ratio < 0.999:
            step_amt = (0.55 * (1.0 - self.speed.ratio)
                        * (1.0 + 1.6 * float(self.tracking_error)))
            # a distinct step per field, alternating sign, so bands alternate hue
            sign = 1.0 if (frame_index % 2 == 0) else -1.0
            ph = (sign * step_amt * np.pi) * in_band.astype(np.float32)
            if np.any(np.abs(ph) > 1e-4):
                c, sn = np.cos(ph)[:, None], np.sin(ph)[:, None]
                i_out, q_out = i_out * c - q_out * sn, i_out * sn + q_out * c

        # --- optional mid-line jitter (ntsc-rs models this) ----------------
        # The head switch can occur part-way along a line, displacing the tail of
        # that line only.  Without this the band is a clean rectangle; with it the
        # strip shows a few half-line steps, which is what the real artefact does.
        if self.hs_mid_line > 1e-4:
            mid = in_band & (rng.random(H) < self.hs_mid_line)
            for row in np.nonzero(mid)[0]:
                cut = int(rng.integers(y.shape[1] // 4, y.shape[1] * 3 // 4))
                amount = float(rng.uniform(-1.0, 1.0) * 18.0 * self.hs_mid_line)
                seg = y_out[row, cut:].astype(np.float32)
                src_idx = np.clip(np.arange(seg.size) - amount, 0, seg.size - 1)
                y_out[row, cut:] = np.interp(src_idx, np.arange(seg.size), seg)
                i_out[row, cut:] *= 0.4
                q_out[row, cut:] *= 0.4
        return y_out, i_out, q_out, w

    def _apply_dropouts(self, comp: np.ndarray, frame_index: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        plan = self.plan
        H, W = comp.shape
        empty = np.zeros(comp.shape, dtype=np.float32)
        if not plan.dropouts:
            return comp, empty
        total: np.ndarray | None = None
        for ev in plan.dropouts[self._drop_cursor:]:
            if ev.start_frame > frame_index:
                break
            if ev.start_frame + ev.frames <= frame_index:
                self._drop_cursor += 1
                continue
            r0, r1 = ev.row, min(H, ev.row + ev.rows)
            c0, c1 = ev.col, min(W, ev.col + ev.cols)
            if r1 <= r0 or c1 <= c0:
                continue
            soft = max(0.6, self.do_soft * min(ev.rows, ev.cols) * 0.6 + 0.8)
            rr = np.arange(r0, r1, dtype=np.float32)
            cc = np.arange(c0, c1, dtype=np.float32)
            ry0, ry1 = float(r0), float(r1 - 1)
            cx0, cx1 = float(c0), float(c1 - 1)

            def rampf(v, a, b):
                return np.clip(np.minimum(v - a, b - v) / soft + 0.5, 0.0, 1.0)

            mask = rampf(rr, ry0, ry1)[:, None] * rampf(cc, cx0, cx1)[None, :]
            patch = (mask * float(ev.severity)).astype(np.float32)
            if total is None:
                total = np.zeros((H, W), dtype=np.float32)
            np.maximum(total[r0:r1, c0:c1], patch, out=total[r0:r1, c0:c1])
        if total is None or not np.any(total > 0.0):
            return comp, empty

        # lost FM carrier: the picture content is gone, replaced by broadband
        # noise around a white level -- the classic white streak
        carrier = 0.72 + 0.30 * self._luma_band_noise(rng, comp.shape)
        comp = comp * (1.0 - total) + carrier * total
        return comp, total

    def _apply_creases(self, comp: np.ndarray, frame_index: int
                       ) -> tuple[np.ndarray, np.ndarray]:
        plan = self.plan
        if not plan.creases:
            return comp, np.zeros(comp.shape, dtype=np.float32)
        H, W = comp.shape
        rows = np.arange(H, dtype=np.float32)[:, None]
        cols = np.arange(W, dtype=np.float32)[None, :] / max(1.0, W)
        total: np.ndarray | None = None
        for cr in plan.creases:
            t = frame_index / max(1e-6, cr.period_frames) + cr.phase
            pos = (0.5 + 0.5 * np.sin(t * 2.0 * np.pi)) * (H + 2 * cr.width) \
                - cr.width + cr.offset
            centre = pos + np.tan(cr.angle) * cols * W
            d = np.abs(rows - centre)
            band = np.clip(1.0 - d / max(1.0, cr.width), 0.0, 1.0) ** 1.6
            if total is None:
                total = (band * cr.depth).astype(np.float32)
            else:
                np.maximum(total, (band * cr.depth).astype(np.float32), out=total)
        if total is None or not np.any(total > 0.0):
            return comp, np.zeros(comp.shape, dtype=np.float32)
        # a crease reduces head-to-tape contact: the signal gets softer and
        # weaker there rather than just noisier
        soft = lowpass(comp, 1.6)
        comp = comp * (1.0 - 0.55 * total) + soft * (0.55 * total)
        return comp, total

    def _apply_tracking_noise(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                              frame_index: int, rng: np.random.Generator
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracking error: rows lose lock, worst at the bottom of the picture.

        This is *not* the head-switching band, and conflating the two is why a
        mistracked tape and a clean tape can look alike in a simulator:

        * head switching is a hard-edged band of fixed height at the very bottom,
          because the two heads swap over on every field;
        * tracking error is a *ramp* growing downwards from some row, because the
          control track the capstan follows is out of step with the video heads.
          It is usually much taller (tens of lines) and its rows drift sideways in
          a slow wave rather than sitting at a fixed offset.

        Displacement and noise share one severity profile because they share one
        cause: the head is reading a track other than the one it thinks it is.
        """
        if self.tracking_rows <= 0:
            return y, i, q
        H = y.shape[0]
        rows_n = int(min(H, max(1, self.tracking_rows)))
        start = H - rows_n

        idx = np.arange(rows_n, dtype=np.float32)
        # 0 at the top of the affected region, 1 at the bottom edge, squared
        ramp = (idx / max(1.0, rows_n - 1.0)) ** 2

        # --- sideways wave: slow drift that grows downwards -----------------
        # `tracking_wave` is the peak displacement in *pixels at the bottom row*,
        # which is the quantity a user can reason about ("the bottom wobbles a few
        # pixels").  An earlier version scaled it by an arbitrary constant and
        # yielded well under one pixel, so the control appeared to do nothing.
        base = float(self._track_wave.advance(1)[0])
        per_row = rng.standard_normal(rows_n).astype(np.float32)
        # guard bands are proportional to track width, so LP/EP need far
        # less capstan error to spill into the neighbouring track
        tw = self.tracking_wave * self.speed.tracking_scale
        drift = ramp * (base * tw * 0.75 + per_row * tw * 0.25)
        disp = np.zeros(H, dtype=np.float32)
        disp[start:] = drift
        y = horizontal_shift(y, disp)
        i = horizontal_shift(i, disp)
        q = horizontal_shift(q, disp)

        # --- speckle noise, heavier towards the bottom ----------------------
        if self.tracking_snow > 1e-4:
            noise = self._luma_band_noise(rng, y.shape)
            w = np.zeros((H, 1), dtype=np.float32)
            w[start:, 0] = ramp * self.tracking_snow
            y = y + w * (noise * 0.85)
            # the colour carrier is the first thing lost on a marginal track
            kill = np.clip(w * 1.2, 0.0, 1.0)
            i = i * (1.0 - kill)
            q = q * (1.0 - kill)
        return y, i, q

    def _fields_for(self, shape: tuple[int, int]) -> None:
        """
        Make the cached noise fields match the frame actually being rendered.

        `prepare()` builds them from the context's dimensions, but `apply()` can
        receive a differently sized frame: preview scaling rounds the height to an
        even number, so a 480-row source previewed at scale 0.37 arrives as 178 rows
        while the fields were built for 480.  Adding them raised

            ValueError: operands could not be broadcast together with shapes (480,640) (178,236)

        Cropping is preferred when the fields are larger, because it preserves the
        particle texture the seed produced; rebuilding is the fallback when they are
        smaller.  A stage handed a different frame size mid-run is normal in this
        application, so this cannot be a precondition.
        """
        h, w = int(shape[0]), int(shape[1])
        if getattr(self, "_fields_shape", None) == (h, w):
            return

        def field_shape(obj):
            """ValueNoise3D exposes height/width; a plain array exposes shape."""
            if obj is None:
                return None
            if hasattr(obj, "shape"):
                return tuple(obj.shape[:2])
            return (int(getattr(obj, "height", 0)), int(getattr(obj, "width", 0)))

        rebuild = False
        for name in ("_grain", "_static_grain"):
            arr = getattr(self, name, None)
            sh = field_shape(arr)
            if sh is None or sh[0] < h or sh[1] < w:
                rebuild = True
                continue
            if hasattr(arr, "shape"):        # a materialised field can be cropped
                setattr(self, name, np.ascontiguousarray(arr[:h, :w]))
        if rebuild:
            self.prepare()
            arr = getattr(self, "_static_grain", None)
            if arr is not None and arr.shape[0] >= h and arr.shape[1] >= w:
                self._static_grain = np.ascontiguousarray(arr[:h, :w])
        self._fields_shape = (h, w)

    def _apply_noise_floor(self, comp: np.ndarray, frame_index: int) -> np.ndarray:
        if self.noise_floor <= 1e-4:
            return comp
        assert self._grain is not None and self._static_grain is not None
        self._fields_for(comp.shape)
        dyn = self._grain.sample(frame_index)
        # fixed particle texture + slowly drifting component
        mixed = 0.45 * self._static_grain + 0.55 * dyn.astype(np.float32)
        # reproduced-output loss at lower tape speeds shows up here
        snr = 10.0 ** (self.speed.snr_loss_db / 20.0)
        amt = 0.055 * self.noise_floor * snr
        out = comp + amt * mixed
        if self._drift is not None and self.do_color > 1e-4:
            tilt = self._drift.sample(frame_index).astype(np.float32)
            # the recorded subcarrier sits at a fixed frequency, so residual
            # chroma in the RF domain shows as a fine stripe along the scan
            # direction rather than as isotropic grain
            striping = np.cos(
                2.0 * np.pi * np.arange(comp.shape[1], dtype=np.float32) / 48.0
            ).astype(np.float32)[None, :]
            out = out + amt * 0.5 * self.do_color * tilt * striping
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def _apply_binder_loss(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                           frame_index: int, rng: np.random.Generator
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Binder breakdown: mould and shed magnetic coating.

        A tape stored damp is not merely dirty.  The binder absorbs water, swells
        and sheds the magnetic layer, so there are two effects and they compound:

        * **clumped, spatially fixed loss.**  The damaged areas are patches bound
          to the tape surface, not a uniform haze, so they repeat in the same place
          every pass and drift only with tape motion.  Loose magnetic material is
          simply not read.
        * **the colour carrier goes first.**  Chroma is a low-amplitude heterodyned
          carrier riding on the luma; where the coating is thin or gone, that weak
          carrier disappears while the stronger luma FM hangs on longer.  Measured
          on the reference's "mouldy" sample, saturation collapses to 0.053 while
          high-frequency energy rises 7x and 38% of the picture sits near black.

        This is a different failure from `debris` (one loose particle striking the
        tape) and from `surface_damage` (a scratch).  Neither of those removes a
        *fraction of the picture*: measured together at their maxima they moved
        high-frequency energy from 0.09388 to 0.09393 and left the dark fraction at
        zero, so the model could not express this condition at all.
        """
        amt = float(np.clip(self.binder_loss, 0.0, 1.0))
        if amt <= 1e-4:
            return y, i, q
        # The patch field is built lazily.  Returning early when it is missing
        # (as an earlier version did) makes the whole mechanism silently do
        # nothing unless some other stage happened to call `prepare()` first --
        # which is exactly the kind of "feature present but unreachable" failure
        # that has bitten this file more than once.
        if self._static_grain is None or self._binder is None:
            self.prepare()
        field = self._static_grain
        if field is None:
            return y, i, q
        H, W = y.shape[:2]
        # coarse, spatially fixed pattern -> clumps rather than an even wash
        base = lowpass(field, max(2.0, W / 40.0), axis=1)
        base = lowpass(base, 4.0, axis=0)
        drift = self._binder.sample(frame_index) if self._binder is not None \
            else np.zeros_like(base)
        pat = base + 0.55 * drift
        pat = (pat - pat.mean()) / max(1e-6, float(pat.std()))
        # `amt` sets what share of the picture is degraded, and the damage has to
        # be *proportional* rather than total.  Two bugs lived here:
        #
        #  * `np.quantile` on a heavy-tailed field put the threshold so low that
        #    every value above it saturated the near-binary ramp, so 0.2, 0.4 and
        #    0.7 all produced exactly the same 65% of pixels affected -- the
        #    control did nothing.
        #  * the loss weight was applied at full strength, so any affected pixel
        #    lost essentially all of its luma, which cannot show the partial state
        #    the reference material has.
        #
        # A percentile over an explicitly normalised field, plus a bounded weight,
        # gives a control that means what it says and leaves a readable picture at
        # low settings.
        pat = (pat - float(np.median(pat))) / max(1e-6, float(pat.std()))
        cut = float(np.percentile(pat, 100.0 * (1.0 - amt)))
        spread = float(max(0.15, np.percentile(pat, 99.0) - cut))
        mask = np.clip((pat - cut) / spread, 0.0, 1.0).astype(np.float32)
        if float(mask.max()) <= 1e-4:
            return y, i, q
        # The mask has to be *graduated*, not a step.  A coating does not shed all
        # at once: it thins, and the head reads progressively less signal from it.
        # An earlier version produced an effectively binary mask -- the luma-change
        # histogram had no entries at all below 0.20, every affected pixel being
        # completely destroyed -- which meant there was no partially-damaged region
        # in which to see the colour carrier fail before the luma.  The mechanism
        # claimed chroma dies first and gave no way to observe it.
        mask = np.sqrt(mask) * (0.55 + 0.45 * mask)

        # Lost signal reads as dense noise *near black*.  The floor matters: an
        # earlier version used `0.03 + 0.30*|noise|`, which averages about 0.13 and
        # so never crossed the near-black threshold at all -- the dark fraction
        # stayed at 0% against a reference of 37.6%, i.e. the model could not
        # produce a genuinely black dropout however hard it was driven.  Where the
        # coating is gone there is no signal, so the level has to be able to reach
        # the pedestal.
        noise = self._luma_band_noise(rng, y.shape)
        # bounded: even a badly shed coating leaves something readable in the
        # affected areas, and capping the weight is what keeps a low setting
        # a partial loss rather than a hole
        drop = mask * (0.35 + 0.55 * amt)
        y_out = y * (1.0 - drop) + (0.008 + 0.16 * np.abs(noise)) * drop
        # The weak chroma carrier is destroyed well before the luma -- but *over a
        # range*, not instantly.  A kill factor of 2.4 removed colour completely
        # as soon as the damage weight passed 0.42, which meant no partially
        # damaged area ever showed colour loss with luma still present; the
        # reference material has exactly that intermediate state, where chroma is
        # visibly reduced but the picture is still readable.  Chroma therefore
        # decays faster than luma (1.3x its loss) instead of being switched off.
        kill = np.clip(drop * 1.3, 0.0, 1.0)
        i_out = i * (1.0 - kill)
        q_out = q * (1.0 - kill)
        self._binder_last = float(drop.mean())
        return y_out, i_out, q_out

    def _apply_shed_grain(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                          frame_index: int, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Coating-shed grain: fine, dense, luminance-only noise.

        Where the magnetic layer has shed, the head reads a rougher surface, so the
        replay signal carries broadband grain.  Two things distinguish this from the
        existing `noise_floor`:

        * it is **strong and fine**.  On the reference "mouldy" panel, high-frequency
          energy is 3x the undamaged panel (0.116 against 0.039) while the picture is
          *also* dark and desaturated.  The model's own noise floor, driven to
          maximum, reached about 0.035 -- a sixth of what is needed -- so the look
          was unreachable however the other controls were set.
        * it is **luminance only**.  A shed coating destroys the weak chroma carrier
          outright, so the grain must not add colour; band-limiting it to luma keeps
          it grey and stops the noise itself from tinting the frame.

        It is the missing companion to the frame-wide desaturation: `chroma_gain`
        supplies the colour loss, `luma_pedestal` the darkness, and this the grain.
        """
        amt = float(np.clip(self.shed_grain, 0.0, 1.0))
        if amt <= 1e-4:
            return y, i, q
        noise = self._luma_band_noise(rng, y.shape)
        y_out = y + (0.02 + 0.85 * amt) * noise
        # a little reaches chroma -- as further desaturation, not as colour
        i_out = i * (1.0 - 0.30 * amt)
        q_out = q * (1.0 - 0.30 * amt)
        return (y_out.astype(np.float32), i_out.astype(np.float32),
                q_out.astype(np.float32))

    def _apply_luma_pedestal(self, y: np.ndarray, i: np.ndarray, q: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pedestal / DC-restoration failure: the luma collapses while chroma stays.

        Luma is a *baseband* level -- its absolute value carries the picture's
        brightness -- so it must be clamped to a reference after AC coupling.
        Chroma is a *modulated amplitude* around whatever that reference is, so it
        survives a reference error untouched.

        That asymmetry is the whole signature, and it is opposite to what a loss of
        recorded signal looks like:

        * **signal lost** (shed coating, dropout, weak RF) takes luma and chroma
          away together, so the picture goes dark *and grey*;
        * **reference lost** (failed clamp, drifted black level) moves the luma
          pedestal down while the chroma keeps its amplitude, so the picture goes
          dark *and violently coloured* -- measured on the S-VHS reference frame,
          saturation rises to 0.94 while the mean falls to 0.09.

        Applying chroma and luma together (which is what `binder_loss` does) leaves
        saturation unchanged, so it cannot produce this look at any strength.  That
        was verified by sweeping it: saturation moved 0.72 -> 0.33 as the loss rose,
        the wrong direction, while collapsing the pedestal alone moves it 0.937 ->
        0.966, the right one.
        """
        offset = float(self.luma_pedestal)
        if abs(offset) < 1e-4:
            return y, i, q
        return (y + offset).astype(np.float32), i, q

    def _apply_drum_speed(self, y: np.ndarray, i: np.ndarray, q: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Head-drum speed error: the cause of "horizontal grain" with line repetition.

        This is a distinct mechanism from anything else in this file, and the
        distinction matters because the symptom looks like noise but is not:

        * tape-transport (capstan) error moves the picture *sideways* -- it changes
          which part of the line is read, so it shows as horizontal jitter;
        * **drum** speed error changes how fast the head sweeps *down* the track.
          A slow drum takes longer to traverse the same tape, so consecutive output
          lines are sampled from very nearly the same point on the tape and the
          picture contains slightly **repeated** lines.  Vertically the picture is
          smoothed or duplicated; horizontally nothing changes.

        That asymmetry is the whole signature: detail survives along the line and
        is smeared between lines, which reads as fine horizontal grain laid over
        the image rather than as random noise.  On a Panasonic shoulder-mount VHS
        or VHS-C camcorder the usual cause is degraded oil in the drum bearing --
        a common failure that, left alone, wears the bearing irreversibly.

        Implemented as a vertical resample at the speed ratio, then held to the
        frame height: a slow drum covers fewer distinct lines than the field has
        slots, so the picture is duplicated into place rather than shortened.  The
        residual fractional offset is carried across frames, which is why the
        effect drifts instead of sitting still.
        """
        ratio = float(np.clip(self.drum_speed, 0.80, 1.05))
        if abs(ratio - 1.0) < 1e-4:
            return y, i, q
        H = y.shape[0]
        # rows are read from a source that is `ratio` times as tall as it should be
        src = np.arange(H, dtype=np.float32) * ratio + self._drum_phase
        src = np.clip(src, 0.0, H - 1.0)
        idx0 = np.floor(src).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, H - 1)
        frac = (src - idx0)[:, None].astype(np.float32)

        def warp(plane: np.ndarray) -> np.ndarray:
            a = plane[idx0]
            b = plane[idx1]
            return (a + (b - a) * frac).astype(np.float32)

        # the offset accumulates slowly: a bearing fault is not stationary
        self._drum_phase = float((self._drum_phase + (1.0 - ratio) * H * 0.02)
                                 % max(1.0, abs(1.0 - ratio) * H + 1e-6))
        return warp(y), warp(i), warp(q)

    def _apply_shuttle_concealment(self, sig: SignalFrame, frame_index: int,
                                   rng: np.random.Generator) -> SignalFrame:
        """
        Rewind / fast-forward residue, in the form the *tape format* actually has.

        The mechanism is common to every helical-scan format: at 2x or 5x the heads
        sweep the tape far faster than it was written, so only part of each track
        arrives intact and the transport substitutes what it can.  What differs is
        the *granularity*, and that is what the eye reads:

        * **analogue line-based** (VHS, Video8, Hi8) -- a continuous waveform per
          track, with no addressable blocks.  Failed regions are whole scanlines,
          in runs, with the run boundaries falling wherever lock was lost.  The
          result is stacked horizontal bands with ragged vertical edges.
        * **digital block-based** (MiniDV) -- a macroblock stream, so concealment
          happens on the 16x16 grid and the result is aligned rectangles.

        Getting this the wrong way round is not a subtle error: a DV block model
        applied to Hi8 asserts a grid the format does not have, and the measured
        row-period spectrum on the Hi8 reference confirms it -- no 16-pixel
        periodicity appears anywhere in that material.
        """
        if self.shuttle_speed <= 0.0 or self.shuttle_fraction <= 1e-4:
            return sig
        H, W = sig.y.shape

        # The donor is a genuinely different frame -- that is the whole point.  It
        # is read from the source rather than held in memory, so a long shuttle
        # does not accumulate a frame cache.
        donor = self._shuttle_donor(frame_index)
        if donor is None:
            return sig
        dy = donor.y[:H, :W]
        di = donor.i[:H, :W]
        dq = donor.q[:H, :W]
        if dy.shape != sig.y.shape:
            return sig

        if self.shuttle_block_based:
            shape = macroblock_grid(H, W, block=self.shuttle_block)
            mask = shuttle_stale_mask(
                rng, shape, self.shuttle_fraction,
                cluster=self.shuttle_cluster, carry=self._shuttle_carry,
                carry_rate=self.shuttle_carry)
            self._shuttle_carry = mask
            self._shuttle_last_fraction = float(mask.mean())
            y = apply_stale_macroblocks(sig.y, mask, dy)
            i = apply_stale_macroblocks(sig.i, mask, di)
            q = apply_stale_macroblocks(sig.q, mask, dq)
        else:
            # mean run length scales with how badly the transport is mistracking
            span = max(1.0, self.shuttle_span)
            mask = line_run_mask(rng, H, self.shuttle_fraction, span_lines=span,
                                 carry=self._shuttle_carry,
                                 carry_rate=self.shuttle_carry)
            self._shuttle_carry = mask
            self._shuttle_last_fraction = float(mask.mean())
            disp = line_run_displacement(rng, mask, self.shuttle_wave)
            y = apply_stale_lines(sig.y, mask, dy, disp)
            i = apply_stale_lines(sig.i, mask, di, disp)
            q = apply_stale_lines(sig.q, mask, dq, disp)
        return SignalFrame(y.astype(np.float32), i.astype(np.float32),
                           q.astype(np.float32))

    def _remember_frame(self, frame_index: int, y: np.ndarray,
                        i: np.ndarray, q: np.ndarray) -> None:
        """Keep the last few processed frames, as a deck's buffer would."""
        if not hasattr(self, "_frame_ring"):
            from collections import OrderedDict
            self._frame_ring = OrderedDict()
            self._ring_limit = int(max(4, min(24, self.ctx.fps * 0.3)))
        entry = (y.astype(np.float32, copy=True),
                 i.astype(np.float32, copy=True),
                 q.astype(np.float32, copy=True))
        self._frame_ring[int(frame_index)] = entry
        self._frame_ring.move_to_end(int(frame_index))
        while len(self._frame_ring) > self._ring_limit:
            self._frame_ring.popitem(last=False)

    def _shuttle_donor(self, frame_index: int) -> SignalFrame | None:
        """
        A frame the transport passed earlier (fast-forward) or later (rewind).

        Taken from this stage's own recent output, *not* from the raw clip.  That
        distinction is the whole correctness of the effect: a deck substitutes from
        decoded frames it already holds, so the replacement carries the same
        recording and playback degradation as everything around it.  Fetching from
        the source instead inserted pristine decoded pixels, and measurement showed
        the patches came out **5x sharper** than the surrounding picture -- the
        band-limited look of the rest of the frame is exactly what the substituted
        content must share.

        Falling back to the source only when no buffered frame exists yet (the first
        few frames of a run), which is also what a buffer that has not filled would
        do.
        """
        ring = getattr(self, "_frame_ring", None)
        if ring:
            keys = list(ring)
            span = max(2, int(round(self.shuttle_speed * 4.0)))
            # prefer frames a plausible distance back, as shuttle skips whole spans
            pick = keys[-1 - min(len(keys) - 1, span)] if len(keys) > 1 else keys[-1]
            y, i, q = ring[pick]
            return SignalFrame(y.copy(), i.copy(), q.copy())

        src = getattr(self.ctx, "source", None)
        if src is None:
            return None
        span = max(2, int(round(self.shuttle_speed * 12.0)))
        direction = -1 if self.shuttle_reverse else 1
        want = int(frame_index) - direction * span
        last = max(0, int(getattr(src, "frame_count", 1)) - 1)
        want = int(np.clip(want, 0, last))
        try:
            rgb = np.clip(src.frame(want), 0.0, 1.0).astype(np.float32)
        except Exception:
            return None
        # Even the fallback should not be pristine: the encoder's bandwidth limit
        # applies to anything this deck can produce.
        lim = self.ctx.layer("format").get("luma_bandwidth", 3.0)
        sig = SignalFrame.from_rgb(rgb, self.ctx.std)
        y = lowpass(sig.y, self.ctx.std.pixels_for_bandwidth(float(lim) * 1e6,
                                                            sig.y.shape[1]), axis=1)
        return SignalFrame(y, sig.i, sig.q)

    def _apply_digital_damage(self, buf: FrameBuffer, frame_index: int,
                              rng: np.random.Generator) -> FrameBuffer:
        """
        MiniDV medium damage: block concealment plus small transport jitter.

        A DV deck has no colour subcarrier, so none of the analog mechanisms
        apply.  What it does have is a fixed bit rate and an error-correction
        scheme: when a read error exceeds what the ECC can repair, the decoder
        *conceals* the macroblock by repeating a neighbour.  The result is a
        clean, block-shaped patch -- and dropout events here are discrete in the
        same sense as the analog ones, which keeps the defect timeline meaningful
        across both branches.
        """
        sig = buf.meta.get("signal")
        if sig is None:
            return buf
        # shuttle concealment runs *before* the discrete damage: it is a property
        # of how the tape is being read, not an event at a point on the tape
        sig = self._apply_shuttle_concealment(sig, frame_index, rng)
        H, W = sig.y.shape
        plan = self.plan
        phase = self._line_phase_for(frame_index)
        # MiniDV's transport is far more stable than VHS: scale the jitter down
        jitter = (phase * 0.25).astype(np.float32) if np.any(np.abs(phase) > 1e-4) else None
        y, i, q = sig.y, sig.i, sig.q
        if jitter is not None:
            y = horizontal_shift(y, jitter)
            i = horizontal_shift(i, jitter)
            q = horizontal_shift(q, jitter)

        # conceal blocks where a dropout event lands
        BLK = 8
        concealed = 0
        if plan.dropouts and self.do_rate > 0.0:
            for ev in plan.dropouts:
                if frame_index < ev.start_frame or frame_index >= ev.start_frame + ev.frames:
                    continue
                # snap the affected region to the macroblock grid
                y0 = (ev.row // BLK) * BLK
                y1 = min(H, y0 + max(BLK, ((ev.rows + BLK - 1) // BLK) * BLK))
                x0 = (ev.col // BLK) * BLK
                x1 = min(W, x0 + max(BLK, ((ev.cols + BLK - 1) // BLK) * BLK))
                if y1 <= y0 or x1 <= x0:
                    continue
                sy0 = max(0, y0 - BLK)
                y[y0:y1, x0:x1] = y[sy0:sy0 + (y1 - y0), x0:x1]
                i[y0:y1, x0:x1] = i[sy0:sy0 + (y1 - y0), x0:x1]
                q[y0:y1, x0:x1] = q[sy0:sy0 + (y1 - y0), x0:x1]
                concealed += 1

        out = FrameBuffer(buf.rgb, None, buf.index, buf.time, buf.std, dict(buf.meta))
        out.meta["signal"] = SignalFrame(y.astype(np.float32), i.astype(np.float32),
                                         q.astype(np.float32))
        out.meta["format"] = "dv"
        out.meta["concealed_blocks"] = concealed
        out.meta["line_phase"] = phase
        return out

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if self.digital:
            frame_index = int(buf.index)
            self._ensure(frame_index)
            rng = np.random.default_rng(
                derive_seed(self.ctx.seed, "tape/dv", frame_index))
            return self._apply_digital_damage(buf, frame_index, rng)
        comp = buf.composite
        if comp is None:  # component/digital path: the RF domain does not apply
            return buf
        frame_index = int(buf.index)
        self._ensure(frame_index)
        rng = np.random.default_rng(derive_seed(self.ctx.seed, "tape/frame", frame_index))

        # The whole medium model runs in baseband: take the carrier apart once,
        # damage luma and chroma the way the physical mechanism does, and rebuild
        # the carrier once at the end.  See `_decompose` for why.
        y, i, q = self._decompose(comp)

        # 0. shuttle: rewind / fast-forward residue, in this format's granularity.
        #    It belongs *before* everything else because it is a property of how
        #    the tape is being read, not an event at a point on the tape -- and it
        #    must run on the analogue branch too, which is where it was missing:
        #    the call lived inside `_apply_digital_damage`, so on VHS and Hi8 the
        #    whole feature silently never executed.
        sh = self._apply_shuttle_concealment(SignalFrame(y, i, q), frame_index, rng)
        y, i, q = sh.y, sh.i, sh.q

        # Remember what this tape produced, for later shuttle substitution.  The
        # frame is captured *after* the format and encoder stages have band-limited
        # it, which is what a deck's frame buffer actually holds and what stops the
        # substituted patches from being sharper than their surroundings.
        if self.shuttle_speed > 0.0 and self.shuttle_fraction > 1e-4:
            self._remember_frame(frame_index, y, i, q)

        # 1. time-base error: per-line sampling phase, luma and chroma together
        y, i, q = self._apply_timebase(y, i, q, frame_index)

        # 1b. drum speed error: vertical sampling density.  Separate from 1 above
        #     because the capstan moves the picture sideways while the drum changes
        #     how many distinct lines are read -- different axis, different look.
        y, i, q = self._apply_drum_speed(y, i, q)

        # 1c. pedestal error: the luma reference collapses but chroma does not.
        #     Placed with the other sampling/clamping errors, before the noise, so
        #     the grain is added around the shifted level rather than lifted with it.
        y, i, q = self._apply_luma_pedestal(y, i, q)

        # 1f. stiction: the damp binder grabbing the capstan and releasing.  Applied with
        #     the other transport errors because it displaces the track rather than the
        #     picture content.
        y, i, q = self._apply_stiction(y, i, q, frame_index)

        # 1e. moire from a residual carrier: curved fringes, added to luma only.
        #     It is an interference product in the video path, so it brightens and
        #     darkens; giving it colour would be a separate claim.
        y = self._apply_stripe_moire(y, frame_index)

        # 1d. shed-coating grain: fine luminance noise, no colour
        y, i, q = self._apply_shed_grain(y, i, q, frame_index, rng)

        # 2. deposition geometry: magnetically weak, slowly moving stripes.
        #    A crease reduces head-to-tape contact, so it softens and weakens
        #    BOTH components -- it is not a chroma-only effect.
        y, crease_mask = self._apply_creases(y, frame_index)
        i, _ = self._apply_creases(i, frame_index)
        q, _ = self._apply_creases(q, frame_index)

        # 3. lost signal patches: the carrier is gone, so chroma goes with it
        y, dropout_mask = self._apply_dropouts(y, frame_index, rng)
        i = i * (1.0 - np.clip(dropout_mask, 0.0, 1.0))
        q = q * (1.0 - np.clip(dropout_mask, 0.0, 1.0))

        # 4. head-switching tear at the end of each field
        y, i, q, hs_mask = self._head_switch(y, i, q, frame_index, rng)

        # 4b. contamination: a clogged head drags a fixed band of lost signal,
        #     loose debris strikes the tape in short bursts, and a scratch
        #     damages a narrow drifting set of lines
        y, i, q, clog_mask = self._apply_clog(y, i, q, frame_index, rng)
        y, i, q = self._apply_debris(y, i, q, frame_index)
        y, i, q = self._apply_surface_damage(y, i, q, frame_index)
        # 4b-bis. binder breakdown: mould and shed coating.  Runs here rather
        #         than with the discrete events because it removes a fraction of
        #         the picture continuously, and it kills chroma before luma.
        y, i, q = self._apply_binder_loss(y, i, q, frame_index, rng)

        # 4c. random whole lines lose colour (weak colour-under carrier)
        i, q = self._apply_chroma_loss(i, q, frame_index)

        # 4c-bis. other whole lines decode to the *wrong* colour: a reference error
        #         rotates every hue on the line and shifts its colour-under gain, so
        #         the line stays saturated.  This is the failure people actually
        #         describe when they say 整行全错, and it is not a stronger version of
        #         the line above -- losing colour and getting the colour wrong are
        #         different faults with different pictures.
        i, q = self._apply_row_chroma_error(i, q, frame_index)

        # 4d. tracking error: rows lose lock from some point downwards
        y, i, q = self._apply_tracking_noise(y, i, q, frame_index, rng)

        # 5. particle noise floor
        y = self._apply_noise_floor(y, frame_index)

        # 6. vertical field timing error
        dy = float(self.plan.frame_dy[frame_index]) \
            if self.plan.prepared_frames > frame_index else 0.0
        if abs(dy) > 1e-3:
            y = vertical_shift(y, dy)
            i = vertical_shift(i, dy)
            q = vertical_shift(q, dy)

        comp_out = self._recompose(y, i, q)
        out = FrameBuffer(buf.rgb, comp_out, buf.index, buf.time, buf.std, dict(buf.meta))
        out.meta["dropout_mask"] = dropout_mask
        out.meta["crease_mask"] = crease_mask
        out.meta["line_phase"] = self._line_phase_for(frame_index)
        return out

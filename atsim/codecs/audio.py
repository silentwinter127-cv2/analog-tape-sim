"""
`atsim.codecs.audio` -- Physical analog & digital-tape audio simulation with frame-accurate A/V fault synchronization.

Why audio must share the video chain's physical state (`TapePlan`, `ConnectorStage`, `RFLink`, `DisplayStage`)
rather than running as an independent post-filter:

  1. **Transport Coupling (Capstan & Drum)**:
     - On VHS, Betamax, and 8mm/Hi8, the same capstan pinch-roller that pulls the tape past the video drum
       pulls it past the linear audio head (and sets the track pitch for AFM / Hi-Fi heads).
     - Therefore, capstan wow & flutter (`line_phase`, `frame_dy`, `walk_amplitude`, `transport_jitter`) and
       binder stiction (`TapePlan.stiction` grab-and-release episodes) modulate audio pitch/speed at the exact
       same millisecond that the video lines tear or jump.
     - During shuttle (fast-forward / rewind, `shuttle_speed > 0`), the tape speed changes and the heads cross
       azimuth tracks: analog audio warps in pitch and chops across guard bands in lockstep with the video
       shuttle bars, while MiniDV PCM produces choppy buffer-hold / click bursts synchronized with macroblock
       concealment.

  2. **Format-Specific Audio Recording Paths (`audio.mode`)**:
     - **0: Auto** -- Resolves automatically from the project's video format (`codec` & `tape` layers):
       - `codec.kind == 2` -> Video CD (MPEG-1 Layer II subband quantization, birdies, pre-echo)
       - `codec.dv_enabled > 0.5` -> MiniDV / Digital8 PCM (12-bit 32 kHz nonlinear / 16-bit 48 kHz)
       - Otherwise -> VHS Hi-Fi AFM Stereo with automatic fallback to VHS Linear Track on severe carrier loss
     - **1: VHS Linear Track** -- Longitudinal track at the upper edge of the 1/2-inch tape moving at
       33.35 mm/s (SP), 16.67 mm/s (LP), or 11.12 mm/s (EP). Very narrow head gap and slow tape speed
       limit high frequencies (~8 kHz in SP, ~4.5 kHz in LP, ~2.8 kHz in EP), head azimuth misalignment
       (`azimuth_loss`) rolls off highs and introduces L/R phase offset, magnetic hysteresis (`tanh` +
       even/odd harmonics) saturates peaks, and oxide particle hiss (`hiss_floor`) sits high.
     - **2: VHS Hi-Fi AFM Stereo** -- Depth-multiplexed FM carriers (1.3 MHz L, 1.7 MHz R) recorded by
       rotating heads on the video drum beneath the video layer. Wide 20 Hz - 20 kHz response and low hiss,
       but suffers from **59.94/50 Hz head-switching transient crackle** at the end of every video field,
       companding noise-reduction breathing, and **automatic fallback to the muffled linear track** whenever
       a deep tape dropout, head clog, crease, or tracking error causes the FM limiter to lose carrier lock.
     - **3: 8mm / Hi8 AFM Audio** -- 1.5 MHz FM audio carrier recorded alongside the video track on the
       rotating drum; tracking errors and dropouts cause brief FM discriminator static bursts.
     - **4: MiniDV / Digital8 PCM** -- 12-bit 32 kHz (nonlinear companded) or 16-bit 48 kHz uncompressed
       PCM recorded in dedicated helical track sectors. Immune to analog hiss and wow, but when tape dropouts
       or shuttle exceed ECC capacity, the decoder mutes, holds the last sample buffer, or emits digital
       framing clicks at the exact frames where video macroblocks conceal.
     - **5: Video CD (MPEG-1 Layer II)** -- 16-bit 44.1 kHz audio compressed via 32-subband psychoacoustic
       quantization: high-frequency "watery" spectral warble (birdies) and transient pre-echo smear.

  3. **Strict Frame- & Row-Accurate Fault Synchronization**:
     - Every video frame `f` spans `[f / fps, (f + 1) / fps]`, and scanline `r` (out of `H`) maps to the
       sub-frame instant `(f + r / H) / fps`.
     - **Dropouts (`TapePlan.dropouts`)**: A dropout at `(start_frame, frames, row, rows, severity)` occurs at
       the exact sub-frame timestamp `(frame + row / H) / fps`. On Hi-Fi/AFM it triggers an FM carrier-loss
       pop/crackle burst and forces fallback to the muffled linear track; on Linear it causes an oxide
       contact-loss amplitude dip and high-frequency dulling; on MiniDV it causes a PCM sample-hold/click.
     - **Head Clog (`TapePlan.clog_rows`) & Creases (`TapePlan.creases`)**: Produce periodic field-rate
       muffling / FM static bursts and low-frequency thumps (`< 120 Hz` head-contact shock) when the crease
       sweeps past the heads.
     - **Base Cracking & Breakage (`base_cracking`, `base_breakage`)**: Sub-surface magnetic-layer cracks
       produce sharp impulse clicks at the exact scanline positions of the video white/black spikes; severe
       base breakage produces control-track (CTL) servo unlock warble and white-noise snow bursts.
     - **Head-Switching Buzz (`head_switching`, `tracking_error`)**: 59.94 Hz (NTSC) / 50 Hz (PAL) pulse
       train and harmonics modulated per frame by `TapePlan.head_switch_amp[f]`.
     - **Connector & RF Intercarrier Faults (`connector` & `rf` layers)**:
       - **AV Ground Loop (`av_ground_loop`, `av_moisture`, `av_corrosion`)**: 50/60 Hz (`mains_hz`) hum +
         odd/even rectifier harmonics synchronized in phase with the rolling video hum bar.
       - **RF Intercarrier Vision-into-Sound Buzz (`rf_moisture`, `rf_corrosion`, `rf_present`, `rf`)**:
         In an RF TV link, audio rides on a 4.5 MHz (NTSC) / 5.5 MHz (PAL) FM sound carrier above the video
         carrier. When damp/corroded RF coaxial reflections or sync-slipped line-clamp surges (`slipped_state`,
         `f_mod != 2`) over-modulate the video carrier, video sync and bright streaking bars bleed into the
         sound IF as harsh 59.94/50 Hz + 15.734/15.625 kHz intercarrier buzz and FM static bursts.
       - **HDMI Bit Errors (`hdmi_present`, `hdmi_moisture`, `hdmi_corrosion`)**: High BER produces sharp
         PCM packet-drop clicks/gaps, and when `ber > 0.55` (`link == "no signal"`), audio drops out completely
         in lockstep with the black video frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt


@dataclass
class AudioSpec:
    """Resolved parameters of the `audio` layer."""
    enabled: bool = True
    mode: int = 0                       # 0=Auto, 1=Linear, 2=Hi-Fi, 3=8mm AFM, 4=MiniDV PCM, 5=VCD MP2
    synth_type: int = 0                 # 0=Retro Synth Chord+Beat, 1=1kHz+400Hz Reference Tone, 2=Voice+Percussion
    hifi_unlock_thresh: float = 0.32    # Dropout/tracking severity that forces Hi-Fi -> Linear fallback
    linear_bandwidth: float = 7500.0    # Nominal SP linear track bandwidth (Hz)
    azimuth_loss: float = 0.22          # Head azimuth misalignment HF rolloff & L/R phase skew (0..1)
    wow_flutter: float = 0.35           # Capstan speed instability / wow & flutter depth (0..1.5)
    stiction_pitch: float = 0.65        # Pitch drop/overshoot depth during capstan stiction (0..1.5)
    dropout_sensitivity: float = 0.55   # How strongly video dropouts/creases/clogs damage audio (0..1.5)
    head_switch_buzz: float = 0.30      # 50/59.94 Hz head-switching pulse & tracking buzz (0..1)
    ground_hum: float = 0.20            # 50/60 Hz mains ground-loop hum coupled to AV connector (0..1)
    rf_buzz: float = 0.30               # Intercarrier vision-into-sound buzz coupled to RF link (0..1)
    saturation: float = 0.30            # Magnetic tape hysteresis soft-clipping & harmonics (0..1.2)
    hiss_floor: float = 0.22            # Oxide particle / FM discriminator noise floor (0..1)
    speaker_sim: float = 0.35           # CRT TV small cabinet speaker bandpass resonance (0..1)
    flyback_whine: float = 0.15         # 15.734 kHz / 15.625 kHz CRT horizontal flyback whine (0..1)
    master_gain: float = 1.0            # Output master gain (0..2)

    @classmethod
    def from_params(cls, params: dict) -> "AudioSpec":
        p = params or {}
        cfg = p.get("audio") if isinstance(p.get("audio"), dict) else p
        return cls(
            enabled=float(cfg.get("enabled", 1.0)) > 0.5,
            mode=int(round(float(cfg.get("mode", 0.0)))),
            synth_type=int(round(float(cfg.get("synth_type", 0.0)))),
            hifi_unlock_thresh=float(np.clip(cfg.get("hifi_unlock_thresh", 0.32), 0.05, 1.0)),
            linear_bandwidth=float(np.clip(cfg.get("linear_bandwidth", 7500.0), 800.0, 16000.0)),
            azimuth_loss=float(np.clip(cfg.get("azimuth_loss", 0.22), 0.0, 1.0)),
            wow_flutter=float(np.clip(cfg.get("wow_flutter", 0.35), 0.0, 1.5)),
            stiction_pitch=float(np.clip(cfg.get("stiction_pitch", 0.65), 0.0, 1.5)),
            dropout_sensitivity=float(np.clip(cfg.get("dropout_sensitivity", cfg.get("afm_switch_pop", cfg.get("dropout_crack", 0.55))), 0.0, 1.5)),
            head_switch_buzz=float(np.clip(cfg.get("head_switch_buzz", 0.30), 0.0, 1.0)),
            ground_hum=float(np.clip(cfg.get("ground_hum", 0.20), 0.0, 1.0)),
            rf_buzz=float(np.clip(cfg.get("rf_buzz", 0.30), 0.0, 1.0)),
            saturation=float(np.clip(cfg.get("saturation", 0.30), 0.0, 1.2)),
            hiss_floor=float(np.clip(cfg.get("hiss_floor", cfg.get("hiss", 0.22)), 0.0, 1.0)),
            speaker_sim=float(np.clip(cfg.get("speaker_sim", 0.35), 0.0, 1.0)),
            flyback_whine=float(np.clip(cfg.get("flyback_whine", 0.15), 0.0, 1.0)),
            master_gain=float(np.clip(cfg.get("master_gain", 1.0), 0.0, 2.0)),
        )


@dataclass
class AudioFaultFrame:
    """Per-video-frame summary of synchronized A/V fault intensities (for timeline inspection & verification)."""
    frame_index: int
    time_s: float
    pitch_deviation_pct: float = 0.0
    dropout_level: float = 0.0
    hifi_fallback: float = 0.0
    stiction_grab: float = 0.0
    head_switch_buzz: float = 0.0
    rf_sync_buzz: float = 0.0
    ground_hum: float = 0.0
    hdmi_drop: float = 0.0
    crack_spike: float = 0.0
    shuttle_chop: float = 0.0

    @property
    def afm_unlock(self) -> bool:
        return self.hifi_fallback > 0.15

    @property
    def pitch_shift_cents(self) -> float:
        return float(1200.0 * np.log2(max(0.1, 1.0 + self.pitch_deviation_pct / 100.0)))

    @property
    def pitch_cents(self) -> float:
        return self.pitch_shift_cents

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "frame_index": self.frame_index,
            "t": round(self.time_s, 4),
            "time_s": round(self.time_s, 4),
            "pitch_pct": round(self.pitch_deviation_pct, 3),
            "pitch_cents": round(self.pitch_shift_cents, 2),
            "pitch_shift_cents": round(self.pitch_shift_cents, 2),
            "dropout": round(self.dropout_level, 3),
            "dropout_level": round(self.dropout_level, 3),
            "hifi_fallback": round(self.hifi_fallback, 3),
            "afm_unlock": bool(self.afm_unlock),
            "stiction": round(self.stiction_grab, 3),
            "hs_buzz": round(self.head_switch_buzz, 3),
            "head_switch_buzz": round(self.head_switch_buzz, 3),
            "rf_buzz": round(self.rf_sync_buzz, 3),
            "hum": round(self.ground_hum, 3),
            "hum_buzz": round(max(self.ground_hum, self.rf_sync_buzz), 3),
            "hdmi_drop": round(self.hdmi_drop, 3),
            "crack": round(self.crack_spike, 3),
            "shuttle": round(self.shuttle_chop, 3),
        }


class AudioRun:
    """Result of simulating an audio window."""
    def __init__(
        self,
        samples: np.ndarray | None = None,
        clean_samples: np.ndarray | None = None,
        sample_rate: int = 44100,
        start_s: float = 0.0,
        duration_s: float = 0.0,
        resolved_mode: str = "vhs_hifi",
        frame_faults: list[AudioFaultFrame] | None = None,
        summary: dict[str, Any] | None = None,
        *,
        pcm: np.ndarray | None = None,
        effective_mode: str | None = None,
        fault_frames: list[AudioFaultFrame] | None = None,
        frames: list[AudioFaultFrame] | None = None,
    ):
        arr = pcm if pcm is not None else samples
        self.samples = np.asarray(arr, dtype=np.float32) if arr is not None else np.zeros((0, 2), dtype=np.float32)
        self.clean_samples = np.asarray(clean_samples, dtype=np.float32) if clean_samples is not None else self.samples.copy()
        self.sample_rate = int(sample_rate)
        self.start_s = float(start_s)
        self.duration_s = float(duration_s)
        self.resolved_mode = str(effective_mode if effective_mode is not None else resolved_mode)
        ff = fault_frames if fault_frames is not None else (frames if frames is not None else frame_faults)
        self.frame_faults: list[AudioFaultFrame] = list(ff) if ff is not None else []
        self.summary: dict[str, Any] = dict(summary) if summary is not None else {}

    @property
    def pcm(self) -> np.ndarray:
        return self.samples

    @property
    def effective_mode(self) -> str:
        return self.resolved_mode

    @property
    def frames(self) -> list[AudioFaultFrame]:
        return self.frame_faults

    @property
    def fault_frames(self) -> list[AudioFaultFrame]:
        return self.frame_faults


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------
def _lowpass_audio(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    fc = float(np.clip(cutoff_hz, 80.0, nyq * 0.96))
    if fc >= nyq * 0.95:
        return sig
    sos = butter(order, fc / nyq, btype="low", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _highpass_audio(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    fc = float(np.clip(cutoff_hz, 15.0, nyq * 0.80))
    sos = butter(order, fc / nyq, btype="high", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _bandpass_audio(sig: np.ndarray, lo_hz: float, hi_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    lo = float(np.clip(lo_hz, 20.0, nyq * 0.85))
    hi = float(np.clip(hi_hz, lo + 100.0, nyq * 0.96))
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Built-in reference audio synthesizer (when clip has no audio or is synthetic)
# ---------------------------------------------------------------------------
def synthesize_test_audio(duration_s: float, sample_rate: int = 44100,
                          start_s: float = 0.0, synth_type: int = 0,
                          seed: int = 20240101, kind: int | None = None,
                          fps: float = 29.97, **kwargs) -> np.ndarray:
    """
    Generate a clean stereo test audio signal `(N, 2)` in `float32` so any project (including
    built-in test patterns, still images, or silent clips) has rich, pitch-sensitive and
    transient-rich audio to make wow & flutter, dropouts, Hi-Fi unlock, and hum unmistakable.

    synth_type:
      0 = Warm Retro Synth Pad + Arpeggio + Crisp Transients (ideal for hearing pitch wow, azimuth & dropouts)
      1 = Broadcast Reference Calibration Tone (1 kHz L / 440 Hz R + 8 kHz pilot)
      2 = Speech-like Formant Cadence +Metronome Ticks (ideal for lip-sync / transient fault checks)
    """
    if kind is not None:
        synth_type = int(kind)
    n = max(1, int(round(duration_s * sample_rate)))
    t = start_s + np.arange(n, dtype=np.float64) / float(sample_rate)

    if synth_type == 1:
        # Broadcast calibration tone: 1000 Hz + 440 Hz + 6000 Hz high-frequency probe (to hear azimuth/LP cutoff)
        left = (0.42 * np.sin(2.0 * np.pi * 1000.0 * t)
                + 0.14 * np.sin(2.0 * np.pi * 6000.0 * t)
                + 0.06 * np.sin(2.0 * np.pi * 10000.0 * t))
        right = (0.42 * np.sin(2.0 * np.pi * 440.0 * t)
                 + 0.14 * np.sin(2.0 * np.pi * 6000.0 * t)
                 + 0.06 * np.sin(2.0 * np.pi * 10000.0 * t))
        return np.stack([left, right], axis=-1).astype(np.float32)

    if synth_type == 2:
        # Formant syllables + high-frequency sibilance / harmonic overtones + 2 Hz frame-sync clicks
        syl_env = np.clip(np.sin(2.0 * np.pi * 1.75 * t) * 1.4, 0.0, 1.0) ** 1.5
        f0 = 135.0 + 22.0 * np.sin(2.0 * np.pi * 0.45 * t)
        phase0 = 2.0 * np.pi * np.cumsum(f0) / float(sample_rate)
        glottal = (0.45 * np.sin(phase0)
                   + 0.32 * np.sin(2.0 * phase0)
                   + 0.22 * np.sin(5.0 * phase0)
                   + 0.16 * np.sin(9.0 * phase0)
                   + 0.12 * np.sin(18.0 * phase0)
                   + 0.16 * np.sin(2.0 * np.pi * 9800.0 * t)
                   + 0.14 * np.sin(2.0 * np.pi * 12800.0 * t))
        voice = syl_env * glottal * 0.42
        # Crisp 2 Hz metronome tick for A/V sync verification
        tick_phase = np.mod(t, 0.5)
        tick = np.exp(-tick_phase / 0.008) * (
            0.34 * np.sin(2.0 * np.pi * 2400.0 * t)
            + 0.22 * np.sin(2.0 * np.pi * 11200.0 * t)
        )
        left = voice + tick
        right = np.roll(voice, int(0.0004 * sample_rate)) + tick * 0.9
        return np.clip(np.stack([left, right], axis=-1), -0.95, 0.95).astype(np.float32)

    # Default (0): Warm Retro Synth Chord (Fmaj7 -> Em7 progression) + Arpeggio + Hi-Hat transients
    prog_step = (np.floor(t / 2.0).astype(np.int32)) % 2
    # Chord frequencies: Fmaj7 (174.61, 220.0, 261.63, 329.63) / Em7 (164.81, 196.0, 246.94, 293.66)
    f1 = np.where(prog_step == 0, 174.61, 164.81)
    f2 = np.where(prog_step == 0, 220.00, 196.00)
    f3 = np.where(prog_step == 0, 261.63, 246.94)
    f4 = np.where(prog_step == 0, 329.63, 293.66)

    def _rich_voice(freq: np.ndarray, detune: float = 1.0) -> np.ndarray:
        ph = 2.0 * np.pi * np.cumsum(freq * detune) / float(sample_rate)
        return (0.55 * np.sin(ph)
                + 0.25 * np.sin(2.0 * ph + 0.3)
                + 0.14 * np.sin(3.0 * ph + 0.7)
                + 0.08 * np.sin(4.0 * ph))

    pad_l = 0.18 * (_rich_voice(f1, 0.999) + _rich_voice(f3, 1.001) + 0.7 * _rich_voice(f4, 0.998))
    pad_r = 0.18 * (_rich_voice(f2, 1.001) + _rich_voice(f4, 0.999) + 0.7 * _rich_voice(f3, 1.002))

    # 4 Hz arpeggio notes with crisp attack and high harmonics (up to 9 kHz so bandwidth/azimuth changes shine)
    arp_notes = np.array([523.25, 659.25, 783.99, 987.77, 1318.51, 987.77, 783.99, 659.25], dtype=np.float64)
    note_idx = (np.floor(t * 4.0).astype(np.int32)) % len(arp_notes)
    f_arp = arp_notes[note_idx]
    note_t = np.mod(t, 0.25)
    arp_env = (1.0 - np.exp(-note_t / 0.004)) * np.exp(-note_t / 0.11)
    ph_arp = 2.0 * np.pi * np.cumsum(f_arp) / float(sample_rate)
    arp_sig = arp_env * (0.45 * np.sin(ph_arp)
                         + 0.22 * np.sin(2.0 * ph_arp)
                         + 0.12 * np.sin(4.0 * ph_arp)
                         + 0.08 * np.sin(7.0 * ph_arp))
    pan = 0.5 + 0.35 * np.sin(2.0 * np.pi * 0.5 * t)

    # Subtle 2 Hz brushed hi-hat transient (for hearing high-frequency loss above 5 kHz)
    rng = np.random.default_rng(int(seed) ^ 0x5A17C3)
    hat_t = np.mod(t, 0.5)
    hat_env = np.exp(-hat_t / 0.018) * 0.10
    white = rng.standard_normal(n)
    # Simple first-difference highpass for crispness
    hat_hp = np.diff(white, prepend=white[0]) * hat_env

    left = pad_l + arp_sig * pan + hat_hp
    right = pad_r + arp_sig * (1.0 - pan) + hat_hp * 0.92
    return np.clip(np.stack([left, right], axis=-1), -0.92, 0.92).astype(np.float32)


# ---------------------------------------------------------------------------
# Main AudioSimulator
# ---------------------------------------------------------------------------
class AudioSimulator:
    """
    Simulates analog & digital-tape audio degradation in strict lockstep with the video chain.
    """

    MODE_NAMES = {
        0: "auto",
        1: "vhs_linear",
        2: "vhs_hifi",
        3: "hi8_afm",
        4: "minidv_pcm",
        5: "vcd_mp2",
    }

    def __init__(self, params_or_renderer: Any, fps: float = 29.97, seed: int = 20240101,
                 std_name: str = "ntsc", sample_rate: int = 44100):
        self._renderer = None
        self._tape_stage = None
        if hasattr(params_or_renderer, "params") and hasattr(params_or_renderer, "fps"):
            r = params_or_renderer
            self._renderer = r
            if getattr(r, "chain", None) is None and hasattr(r, "_build_chain"):
                r._build_chain()
            self._tape_stage = (getattr(r, "stage", None) or {}).get("tape")
            raw_params = r.params or {}
            fps = float(getattr(r, "fps", fps))
            seed = int(getattr(r, "seed", seed))
            std_obj = getattr(r, "std", None)
            if std_obj is not None and hasattr(std_obj, "name"):
                std_name = str(std_obj.name).lower()
        else:
            raw_params = params_or_renderer or {}
        if isinstance(raw_params, dict) and "audio" not in raw_params and any(k in raw_params for k in ("mode", "wow_flutter", "hiss_floor", "hiss")):
            self.params = {"audio": dict(raw_params)}
        else:
            self.params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (raw_params or {}).items()}
        self.spec = AudioSpec.from_params(self.params)
        self.fps = float(max(1.0, fps))
        self.seed = int(seed)
        self.std_name = str(std_name).lower()
        self.sample_rate = int(sample_rate)

    def resolve_mode_name(self) -> str:
        m = int(self.spec.mode)
        if m in (1, 2, 3, 4, 5):
            return self.MODE_NAMES[m]
        # Auto-resolve from codec & tape layers
        codec_cfg = self.params.get("codec") or {}
        if int(round(float(codec_cfg.get("kind", 0.0)))) == 2:
            return "vcd_mp2"
        if float(codec_cfg.get("dv_enabled", 0.0)) > 0.5:
            return "minidv_pcm"
        return "vhs_hifi"

    def _frame_params_at(self, frame_index: int, keyframes=None, environment=None,
                         ranges: dict | None = None, layer_of=None) -> dict:
        """Resolve per-frame parameters including environment & keyframe tracks."""
        if keyframes is None and environment is None:
            return self.params
        p = {lay: dict(vals) for lay, vals in self.params.items()}
        t = frame_index / self.fps
        if environment is not None and ranges is not None and layer_of is not None:
            from ..core.environment import apply_environment
            p, _ = apply_environment(p, environment, ranges, layer_of)
        if keyframes is not None and ranges is not None:
            p, _ = keyframes.apply(p, t, ranges)
        return p

    def render_window(self, clean_stereo: np.ndarray, start_s: float = 0.0, duration_s: float | None = None,
                      tape_stage=None, tape_plan=None, keyframes=None, environment=None,
                      ranges: dict | None = None, layer_of=None) -> AudioRun:
        """
        Process a stereo audio buffer `(N, 2)` spanning `[start_s, start_s + duration_s]`
        with exact frame-by-frame and scanline-by-scanline fault synchronization to `tape_stage`.
        """
        sr = self.sample_rate
        clean = np.asarray(clean_stereo, dtype=np.float32)
        if clean.ndim == 1:
            clean = np.stack([clean, clean], axis=-1)
        if duration_s is None:
            duration_s = max(1.0 / self.fps, len(clean) / float(sr))
        n_samples = max(1, int(round(duration_s * sr)))
        if len(clean) < n_samples:
            pad = np.zeros((n_samples - len(clean), 2), dtype=np.float32)
            clean = np.concatenate([clean, pad], axis=0)
        elif len(clean) > n_samples:
            clean = clean[:n_samples]
        clean = clean.copy()

        mode_name = self.resolve_mode_name()
        if not self.spec.enabled:
            return AudioRun(
                samples=np.clip(clean * self.spec.master_gain, -1.0, 1.0),
                clean_samples=clean,
                sample_rate=sr,
                start_s=start_s,
                duration_s=duration_s,
                resolved_mode="bypass",
                frame_faults=[],
                summary={"mode": "bypass", "fault_frames": 0},
            )

        if tape_stage is None:
            tape_stage = self._tape_stage
        start_frame = max(0, int(np.floor(start_s * self.fps + 1e-6)))
        end_frame = max(start_frame, int(np.ceil((start_s + duration_s) * self.fps - 1e-6)) - 1)
        if tape_plan is not None:
            plan = tape_plan
            H_vid = max(2, int(getattr(getattr(tape_stage, "ctx", None), "height", 240) or 240))
        elif tape_stage is not None:
            tape_stage._ensure(end_frame + 2)
            plan = tape_stage.plan
            H_vid = max(2, int(tape_stage.ctx.height))
        else:
            plan = None
            H_vid = 240

        # Time array and continuous frame/row coordinates for every audio sample
        t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)
        frame_float = t_arr * self.fps
        frame_idx_arr = np.maximum(0, np.floor(frame_float).astype(np.int32))
        row_float = (frame_float - frame_idx_arr) * H_vid
        row_idx_arr = np.clip(np.floor(row_float).astype(np.int32), 0, H_vid - 1)

        # Per-sample envelope arrays (built from frame & scanline video events)
        pitch_shift_rate = np.zeros(n_samples, dtype=np.float64)   # fractional speed delta (dv/v)
        dropout_env = np.zeros(n_samples, dtype=np.float32)        # 0..1.5 oxide dropout / clog / crease
        crease_thump = np.zeros(n_samples, dtype=np.float32)       # low-frequency head-bump impulse
        hs_pulse_env = np.zeros(n_samples, dtype=np.float32)       # head-switch band envelope
        rf_buzz_env = np.zeros(n_samples, dtype=np.float32)        # RF intercarrier vision-into-sound buzz
        rf_static_env = np.zeros(n_samples, dtype=np.float32)      # RF sync-slip FM static burst
        hum_env = np.zeros(n_samples, dtype=np.float32)            # AV ground-loop hum amplitude
        hdmi_mute_env = np.zeros(n_samples, dtype=np.float32)      # 1.0 when HDMI link drops
        hdmi_ber_env = np.zeros(n_samples, dtype=np.float32)       # HDMI bit-error rate
        crack_clicks = np.zeros(n_samples, dtype=np.float32)       # sharp impulse spikes from base_cracking
        ctl_unlock_env = np.zeros(n_samples, dtype=np.float32)     # base_breakage CTL servo unlock
        shuttle_env = np.zeros(n_samples, dtype=np.float32)        # shuttle track-crossing chop mask
        speed_ratio_arr = np.ones(n_samples, dtype=np.float32)     # 1.0=SP, 0.5=LP, 0.333=EP

        frame_faults: list[AudioFaultFrame] = []

        # Pre-index dropouts by frame for fast lookup
        dropouts_by_frame: dict[int, list] = {}
        if plan is not None and plan.dropouts:
            for ev in plan.dropouts:
                if ev.start_frame + ev.frames < start_frame or ev.start_frame > end_frame:
                    continue
                for f in range(max(start_frame, ev.start_frame),
                               min(end_frame + 1, ev.start_frame + ev.frames)):
                    dropouts_by_frame.setdefault(f, []).append(ev)

        # Pre-index stiction episodes by frame
        stiction_by_frame: dict[int, list] = {}
        if plan is not None and getattr(plan, "stiction", None):
            for sev in plan.stiction:
                f0 = int(sev.start_frame)
                hold_f = int(getattr(sev, "hold_frames", getattr(sev, "frames", 4)))
                rel_f_cnt = int(getattr(sev, "release_frames", max(2, hold_f // 2)))
                f1 = int(f0 + hold_f + rel_f_cnt)
                if f1 < start_frame or f0 > end_frame:
                    continue
                for f in range(max(start_frame, f0), min(end_frame + 1, f1)):
                    stiction_by_frame.setdefault(f, []).append(sev)

        # Populate per-frame and per-scanline synchronized fault envelopes
        from ..codecs.tape_speed import by_index
        from ..codecs.connectors import CONNECTORS, impedance, reflection_coefficient

        for f in range(start_frame, end_frame + 1):
            # Sample range belonging to frame `f`
            s0 = int(round((f / self.fps - start_s) * sr))
            s1 = int(round(((f + 1) / self.fps - start_s) * sr))
            s0_c = max(0, min(n_samples, s0))
            s1_c = max(0, min(n_samples, s1))
            if s1_c <= s0_c:
                continue
            sl = slice(s0_c, s1_c)
            n_f = s1_c - s0_c
            u_f = np.linspace(0.0, 1.0, n_f, endpoint=False, dtype=np.float32)
            rows_f = np.clip((u_f * H_vid).astype(np.int32), 0, H_vid - 1)

            f_params = self._frame_params_at(f, keyframes, environment, ranges, layer_of)
            tape_cfg = f_params.get("tape") or {}
            conn_cfg = f_params.get("connector") or {}
            rf_cfg = f_params.get("rf") or {}
            player_cfg = f_params.get("player") or {}
            aud_spec = AudioSpec.from_params(f_params)

            ff = AudioFaultFrame(frame_index=f, time_s=f / self.fps)

            # 1. Tape Speed (SP / LP / EP + sp_lp_mix transition)
            sp_obj = by_index(tape_cfg.get("tape_speed", 0.0))
            sp_lp_mix = float(np.clip(tape_cfg.get("sp_lp_mix", 0.0), 0.0, 1.0))
            eff_ratio = float(sp_obj.ratio) * (1.0 - 0.5 * sp_lp_mix)
            speed_ratio_arr[sl] = eff_ratio

            # 2. Capstan Wow & Flutter coupled directly to TapePlan line_phase & frame_dy
            wf_amt = aud_spec.wow_flutter * (1.4 - 0.85 * float(np.clip(tape_cfg.get("tape_stability", 0.55), 0.0, 1.0)))
            wf_amt /= max(0.33, eff_ratio ** 0.5)
            p_dev = 0.0
            if plan is not None and plan.prepared_lines > f * H_vid:
                l0 = f * H_vid
                l1 = min(len(plan.line_phase), (f + 1) * H_vid)
                if l1 > l0:
                    lp_seg = plan.line_phase[l0:l1]
                    # Resample scanline phase velocity onto the audio samples of this frame
                    lp_interp = np.interp(u_f * (len(lp_seg) - 1), np.arange(len(lp_seg)), lp_seg)
                    pitch_shift_rate[sl] += wf_amt * 0.0022 * lp_interp
                    p_dev += float(np.mean(np.abs(lp_interp))) * wf_amt * 0.22
            if plan is not None and f < len(plan.frame_dy):
                f_dy = float(plan.frame_dy[f])
                pitch_shift_rate[sl] += wf_amt * 0.0035 * f_dy * np.sin(2.0 * np.pi * u_f)
                p_dev += abs(f_dy) * wf_amt * 0.35

            # Drum speed bearing error (`drum_speed` != 1.0)
            drum_sp = float(tape_cfg.get("drum_speed", 1.0))
            if abs(drum_sp - 1.0) > 1e-4:
                pitch_shift_rate[sl] += (drum_sp - 1.0) * 0.35
                p_dev += abs(drum_sp - 1.0) * 35.0

            # 3. Capstan Stiction Grab & Release (`TapePlan.stiction`)
            if f in stiction_by_frame and aud_spec.stiction_pitch > 1e-4:
                for sev in stiction_by_frame[f]:
                    hold_f = float(max(1, getattr(sev, "hold_frames", getattr(sev, "frames", 4))))
                    rel_f_cnt = float(max(1, getattr(sev, "release_frames", max(2.0, hold_f * 0.5))))
                    rel_f = (f + u_f - sev.start_frame) / hold_f
                    # During grab (0 <= rel_f <= 1): tape slows down sharply (negative pitch shift)
                    # Just after release (1 < rel_f <= 1 + rel_f_cnt/hold_f): capstan spring-back overshoots pitch upward!
                    grab_mask = (rel_f >= 0.0) & (rel_f <= 1.0)
                    rel_mask = (rel_f > 1.0) & (rel_f <= 1.0 + rel_f_cnt / hold_f)
                    grab_env = np.where(grab_mask, np.maximum(0.0, np.sin(np.pi * np.clip(rel_f, 0.0, 1.0))) ** 1.2, 0.0)
                    overshoot_env = np.where(
                        rel_mask,
                        0.60 * np.exp(-(rel_f - 1.0) / max(0.15, 0.35 * rel_f_cnt / hold_f))
                        * np.sin(2.0 * np.pi * (rel_f - 1.0) / max(0.25, 0.75 * rel_f_cnt / hold_f)),
                        0.0,
                    )
                    st_depth = float(getattr(sev, "peak_px", getattr(sev, "magnitude", 24.0))) / 32.0 * aud_spec.stiction_pitch
                    pitch_shift_rate[sl] += (-0.068 * grab_env + 0.036 * overshoot_env) * st_depth
                    dropout_env[sl] = np.maximum(dropout_env[sl], (0.45 * grab_env * st_depth).astype(np.float32))
                    ff.stiction_grab = max(ff.stiction_grab, float(np.max(grab_env + np.abs(overshoot_env)) * st_depth))
                    p_dev += float(np.max(np.abs(-6.8 * grab_env + 3.6 * overshoot_env)) * st_depth)

            ff.pitch_deviation_pct = float(p_dev)

            # 4. Dropouts (`TapePlan.dropouts`), Head Clog (`clog_rows`), Creases (`creases`), Binder Loss
            do_sens = aud_spec.dropout_sensitivity
            if do_sens > 1e-4:
                # 4a. Discrete DropoutEvents mapped to exact scanline timestamp within the frame
                if f in dropouts_by_frame:
                    for ev in dropouts_by_frame[f]:
                        r0_frac = max(0.0, (ev.row - 2.0) / float(H_vid))
                        r1_frac = min(1.0, (ev.row + ev.rows + 4.0) / float(H_vid))
                        # Even a 2-scanline video dropout excites the audio head / FM discriminator for ~2-6 ms
                        min_span = max(0.06, (ev.rows / float(H_vid)) * 2.5)
                        mid_frac = 0.5 * (r0_frac + r1_frac)
                        half_w = max(0.5 * (r1_frac - r0_frac), 0.5 * min_span)
                        dist = np.abs(u_f - mid_frac) / max(1e-4, half_w)
                        win = np.clip(1.0 - dist * dist, 0.0, 1.0)
                        sev_val = float(ev.severity) * do_sens
                        dropout_env[sl] = np.maximum(dropout_env[sl], (win * sev_val).astype(np.float32))

                # 4b. Spatially fixed Head Clog (`plan.clog_rows`)
                if plan is not None and plan.clog_rows.size == H_vid and float(tape_cfg.get("head_clog", 0.0)) > 1e-4:
                    wobble = 0.75 + 0.45 * float(np.sin(f * 0.21 + float(plan.clog_rows.sum())))
                    clog_r = np.clip(plan.clog_rows[rows_f] * wobble * do_sens, 0.0, 1.4)
                    dropout_env[sl] = np.maximum(dropout_env[sl], clog_r.astype(np.float32))

                # 4c. Tape Creases (`plan.creases`) -> low-frequency mechanical head thump + contact loss
                if plan is not None and plan.creases and float(tape_cfg.get("crease_rate", 0.0)) > 1e-4:
                    for cr in plan.creases:
                        t_cr = f / max(1e-6, cr.period_frames) + cr.phase
                        pos_row = (0.5 + 0.5 * np.sin(t_cr * 2.0 * np.pi)) * (H_vid + 2.0 * cr.width) - cr.width + cr.offset
                        if -cr.width <= pos_row <= H_vid + cr.width:
                            center_u = float(np.clip(pos_row / float(H_vid), 0.02, 0.98))
                            w_u = max(0.04, (cr.width / float(H_vid)) * 1.5)
                            cr_env = np.exp(-0.5 * ((u_f - center_u) / w_u) ** 2) * float(cr.depth) * do_sens
                            dropout_env[sl] = np.maximum(dropout_env[sl], (cr_env * 0.85).astype(np.float32))
                            # Mechanical head-to-tape bump wave (damped 75 Hz thump)
                            thump_wave = cr_env * np.sin(2.0 * np.pi * (u_f - center_u) / max(0.03, w_u * 1.2))
                            crease_thump[sl] += (thump_wave * 0.45).astype(np.float32)

                # 4d. Binder Loss / Debris / Demagnetization
                binder = float(tape_cfg.get("binder_loss", 0.0))
                debris = float(tape_cfg.get("debris", 0.0))
                demag = float(tape_cfg.get("demagnetization", 0.0))
                if binder > 1e-4 or debris > 1e-4 or demag > 1e-4:
                    rng_bd = np.random.default_rng((self.seed * 131 + f * 997) & 0xFFFFFFFF)
                    patch = (0.35 * binder + 0.25 * demag) * (0.6 + 0.4 * np.sin(2.0 * np.pi * u_f + f * 0.37))
                    if debris > 1e-4:
                        n_strikes = int(rng_bd.poisson(max(1.0, debris * 6.0)))
                        for _ in range(n_strikes):
                            cu = float(rng_bd.uniform(0.0, 1.0))
                            patch = np.maximum(patch, debris * 0.75 * np.exp(-0.5 * ((u_f - cu) / 0.015) ** 2))
                    dropout_env[sl] = np.maximum(dropout_env[sl], (patch * do_sens).astype(np.float32))

            ff.dropout_level = float(np.max(dropout_env[sl])) if n_f > 0 else 0.0

            # 5. Base Cracking (`base_cracking`) & Base Breakage (`base_breakage`)
            cracking = float(tape_cfg.get("base_cracking", 0.0))
            if cracking > 1e-4:
                rng_crk = np.random.default_rng((self.seed * 7919 + f * 2654435761) & 0xFFFFFFFF)
                n_cracks = int(rng_crk.poisson(cracking * 3.5))
                for _ in range(n_cracks):
                    pos_s = int(rng_crk.integers(0, max(1, n_f - 12)))
                    pol = 1.0 if rng_crk.random() < 0.65 else -0.85
                    amp_c = float(pol * (0.35 + 0.55 * cracking))
                    # Sharp 0.4 ms magnetic-oxide crack impulse followed by brief exponential tail
                    tail_len = min(24, n_f - pos_s)
                    crack_clicks[s0_c + pos_s : s0_c + pos_s + tail_len] += (
                        amp_c * np.exp(-np.arange(tail_len, dtype=np.float32) / 4.5)
                        * np.cos(np.arange(tail_len, dtype=np.float32) * 0.9)
                    )
                    ff.crack_spike = max(ff.crack_spike, abs(amp_c))

            breakage = float(tape_cfg.get("base_breakage", 0.0))
            if breakage > 1e-4:
                # Severe physical tape snap / CTL loss episode
                brk_w = breakage * (0.65 + 0.35 * float(np.sin(f * 0.43)))
                ctl_unlock_env[sl] = np.maximum(ctl_unlock_env[sl], brk_w)
                dropout_env[sl] = np.maximum(dropout_env[sl], (brk_w * 0.95).astype(np.float32))
                pitch_shift_rate[sl] += -0.12 * brk_w * np.cos(2.0 * np.pi * u_f * 2.0)

            # 6. Shuttle (`shuttle_speed`, `shuttle_stale`)
            sh_speed = float(tape_cfg.get("shuttle_speed", 0.0))
            sh_stale = float(tape_cfg.get("shuttle_stale", 0.0))
            if sh_speed > 1e-4 and sh_stale > 1e-4:
                # During shuttle, heads cross multiple helical tracks per field, producing rhythmic
                # track-crossing bursts at `max(2, round(sh_speed))` bands per frame
                n_bands = max(2.0, min(10.0, sh_speed * 1.5))
                chop_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * n_bands * u_f + f * 1.1)
                sh_mask = np.clip((chop_wave - (1.0 - sh_stale)) / max(0.1, sh_stale), 0.0, 1.0)
                shuttle_env[sl] = sh_mask.astype(np.float32)
                # Varispeed pitch offset during analog shuttle
                sh_dir = -1.0 if float(tape_cfg.get("shuttle_reverse", 0.0)) > 0.5 else 1.0
                pitch_shift_rate[sl] += sh_dir * min(0.45, 0.08 * sh_speed) * (0.7 + 0.3 * chop_wave)
                ff.shuttle_chop = float(np.mean(sh_mask))

            # 7. Head-Switching Buzz & Tracking Error (`head_switching`, `tracking_error`, `tracking_noise_snow`)
            hs_amp_frame = float(plan.head_switch_amp[f]) if (plan is not None and f < len(plan.head_switch_amp)) else float(tape_cfg.get("head_switching", 0.45))
            hs_lines = float(tape_cfg.get("head_switch_lines", 12.0))
            tr_err = float(tape_cfg.get("tracking_error", 0.0)) + 0.6 * float(tape_cfg.get("tracking_noise_snow", 0.0))
            buzz_gain = aud_spec.head_switch_buzz * (0.45 * hs_amp_frame + 0.85 * tr_err + 0.30 * sp_lp_mix)
            if buzz_gain > 1e-4:
                # Video fields switch twice per frame (at u ~ 0.48..0.50 and u ~ 0.96..1.00)
                band_frac = max(0.025, min(0.18, hs_lines / max(1.0, float(H_vid) * 0.5)))
                in_sw1 = np.clip(1.0 - np.abs(u_f - (0.50 - 0.5 * band_frac)) / band_frac, 0.0, 1.0)
                in_sw2 = np.clip(1.0 - np.abs(u_f - (1.00 - 0.5 * band_frac)) / band_frac, 0.0, 1.0)
                sw_pulse = np.maximum(in_sw1, in_sw2)
                # Continuous tracking error also bleeds a sawtooth field-rate buzz across the whole field
                hs_pulse_env[sl] = (buzz_gain * (0.75 * sw_pulse + 0.25 * tr_err)).astype(np.float32)
                ff.head_switch_buzz = float(buzz_gain)

            # 8. AV Connector Ground-Loop Hum (`av_present`, `av_ground_loop`, `av_moisture`, `av_corrosion`)
            av_on = float(conn_cfg.get("av_present", 0.0)) > 0.5
            av_gnd = float(conn_cfg.get("av_ground_loop", 0.0))
            av_m = float(conn_cfg.get("av_moisture", 0.0))
            av_c = float(conn_cfg.get("av_corrosion", 0.0))
            eff_hum = aud_spec.ground_hum * (
                (av_gnd + 0.35 * av_m + 0.45 * av_c) if av_on else (0.65 * av_gnd + 0.25 * av_m + 0.30 * av_c)
            )
            if eff_hum > 1e-4:
                hum_env[sl] = float(np.clip(eff_hum, 0.0, 1.2))
                ff.ground_hum = float(hum_env[s0_c])

            # 9. RF Coaxial Link & Damp Interface Sync-Slip Buzz (`rf_present`, `rf_moisture`, `rf_corrosion`, `rf`)
            rf_on = float(conn_cfg.get("rf_present", 0.0)) > 0.5
            rf_m = float(conn_cfg.get("rf_moisture", 0.0))
            rf_c = float(conn_cfg.get("rf_corrosion", 0.0))
            rf_dbm = float(rf_cfg.get("signal_dbm", -60.0))
            rf_weak = float(np.clip((-60.0 - rf_dbm) / 25.0, 0.0, 1.0))
            eff_rf_m = float(np.clip(rf_m + 0.45 * rf_c, 0.0, 1.0)) if (rf_on or rf_m > 1e-4 or rf_c > 1e-4) else 0.0
            if (eff_rf_m > 1e-4 or rf_weak > 1e-4) and aud_spec.rf_buzz > 1e-4:
                # Match `degrade_rf_signal` bistable sync slip (`f_mod = frame_index % 4`, slipped when `f_mod != 2`)
                f_mod = int(f) % 4
                slipped = (f_mod != 2) and (eff_rf_m >= 0.30)
                multi_band = (f_mod in (0, 3)) and slipped
                base_rf_buzz = aud_spec.rf_buzz * (0.55 * eff_rf_m + 0.45 * rf_weak)
                if slipped:
                    # On slipped frames with horizontal line-clamp surge bars (`sy_frac = 0.515, 0.692`),
                    # vision-into-sound intercarrier buzz spikes sharply at those scanline bands!
                    surge_env = np.ones_like(u_f) * 1.25
                    if not multi_band:
                        b1 = np.exp(-0.5 * ((u_f - 0.515) / 0.045) ** 2)
                        b2 = np.exp(-0.5 * ((u_f - 0.692) / 0.045) ** 2)
                        surge_env += 1.65 * (b1 + b2)
                    else:
                        # Multi-band chroma unlock frames (`f_mod in (0, 3)`)
                        b_top = np.exp(-0.5 * ((u_f - 0.09) / 0.05) ** 2)
                        b_bot = np.exp(-0.5 * ((u_f - 0.83) / 0.07) ** 2)
                        surge_env += 1.35 * (b_top + b_bot)
                    rf_buzz_env[sl] = (base_rf_buzz * surge_env).astype(np.float32)
                    rf_static_env[sl] = (base_rf_buzz * 0.65 * (0.4 + 0.6 * (surge_env - 1.0))).astype(np.float32)
                else:
                    # Frame `f_mod == 2`: V-sync top wrap-around & lower static bands (`0.66`, `0.80`)
                    sb = np.exp(-0.5 * ((u_f - 0.66) / 0.05) ** 2) + np.exp(-0.5 * ((u_f - 0.80) / 0.06) ** 2)
                    rf_buzz_env[sl] = (base_rf_buzz * (0.55 + 0.55 * sb)).astype(np.float32)
                    rf_static_env[sl] = (base_rf_buzz * 0.45 * sb).astype(np.float32)
                ff.rf_sync_buzz = float(np.max(rf_buzz_env[sl]))

            # 10. HDMI Digital Link Errors & Handshake Drop (`hdmi_present`, `hdmi_moisture`, `hdmi_corrosion`)
            hdmi_on = float(conn_cfg.get("hdmi_present", 0.0)) > 0.5
            hdmi_m = float(conn_cfg.get("hdmi_moisture", 0.0))
            hdmi_c = float(conn_cfg.get("hdmi_corrosion", 0.0))
            if hdmi_on and (hdmi_m > 1e-4 or hdmi_c > 1e-4):
                z_h = impedance(CONNECTORS["hdmi"], hdmi_m, hdmi_c)
                gamma_h = abs(reflection_coefficient(CONNECTORS["hdmi"].nominal_z, z_h))
                ber = float(np.clip(gamma_h * 3.4 + 0.42 * hdmi_c, 0.0, 1.0))
                if ber > 0.55:
                    # Complete link handshake drop -> NO SIGNAL (silent audio + brief unlock click at edge)
                    hdmi_mute_env[sl] = 1.0
                    ff.hdmi_drop = 1.0
                elif ber > 1e-4:
                    hdmi_ber_env[sl] = ber
                    ff.hdmi_drop = round(ber, 3)

            # Record Hi-Fi fallback fraction for this frame
            if mode_name == "vhs_hifi":
                unlk = np.clip((dropout_env[sl] + 0.6 * tr_err - aud_spec.hifi_unlock_thresh) / 0.22, 0.0, 1.0)
                ff.hifi_fallback = float(0.65 * np.max(unlk) + 0.35 * np.mean(unlk))

            frame_faults.append(ff)

        # ------------------------------------------------------------------
        # Step 1: Time-Domain Varispeed Resampling (Wow & Flutter + Stiction + Shuttle)
        # ------------------------------------------------------------------
        # Smooth the fractional speed deviation slightly (~1.5 ms) so frame boundaries are C1 continuous
        pitch_shift_rate = gaussian_filter1d(pitch_shift_rate, sigma=max(1.0, 0.0015 * sr))
        # Integrate instantaneous speed ratio `1 + pitch_shift_rate(t)` to get warped sample indices
        inst_speed = np.clip(1.0 + pitch_shift_rate, 0.25, 2.50)
        warped_pos = np.cumsum(inst_speed) - inst_speed[0]
        idx_float = np.clip(warped_pos, 0.0, float(n_samples - 1.001))
        idx0 = np.floor(idx_float).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, n_samples - 1)
        frac = (idx_float - idx0)[:, None].astype(np.float32)
        sig = clean[idx0] * (1.0 - frac) + clean[idx1] * frac

        # ------------------------------------------------------------------
        # Step 2: Head Azimuth Misalignment (L/R micro-delay skew + HF comb/rolloff)
        # ------------------------------------------------------------------
        az = float(self.spec.azimuth_loss)
        if az > 1e-4 and mode_name in ("vhs_linear", "vhs_hifi", "hi8_afm"):
            # Azimuth tilt delays the right channel relative to left by up to 0.35 ms
            delay_samples = int(round(az * 0.00035 * sr))
            if delay_samples > 0 and delay_samples < n_samples:
                sig[delay_samples:, 1] = sig[:-delay_samples, 1]

        # ------------------------------------------------------------------
        # Step 3: Format-Specific Signal Path & Fault Coupling
        # ------------------------------------------------------------------
        rng = np.random.default_rng((self.seed * 2654435761 + int(round(start_s * 1000))) & 0xFFFFFFFF)
        mean_speed_ratio = float(np.clip(np.mean(speed_ratio_arr), 0.25, 1.0))

        # Build the VHS Linear Track signal (used both in Mode 1 and as the fallback track in Mode 2)
        def _build_linear_track(inp: np.ndarray) -> np.ndarray:
            # Linear track cutoff scales directly with tape speed (SP=1.0, LP=0.5, EP=0.333) and azimuth loss
            fc_lin = min(6200.0, self.spec.linear_bandwidth) * (mean_speed_ratio ** 0.78) * (1.0 - 0.52 * az)
            lin = _highpass_audio(inp, 95.0, sr, order=2)
            lin = _lowpass_audio(lin, fc_lin, sr, order=6)
            # Right channel loses extra high frequencies under azimuth misalignment
            if az > 0.05:
                lin[:, 1] = _lowpass_audio(lin[:, 1:2], fc_lin * (1.0 - 0.35 * az), sr, order=2)[:, 0]
            # Magnetic oxide dropouts dull and attenuate the longitudinal track
            do_atten = np.clip(1.0 - 0.85 * dropout_env, 0.04, 1.0)[:, None]
            lin_muffled = _lowpass_audio(lin, max(450.0, fc_lin * 0.24), sr, order=2)
            drop_mix = np.clip(dropout_env, 0.0, 1.0)[:, None]
            lin = (lin * (1.0 - 0.72 * drop_mix) + lin_muffled * (0.72 * drop_mix)) * do_atten
            # Linear tape hiss (rises at slower tape speeds, roll-off above fc_lin)
            hiss_raw = rng.standard_normal((n_samples, 2)).astype(np.float32)
            hiss_shaped = _bandpass_audio(hiss_raw, 350.0, min(6200.0, fc_lin * 0.95), sr, order=3)
            hiss_amp = 0.024 * self.spec.hiss_floor / (mean_speed_ratio ** 0.45)
            return lin + hiss_shaped * hiss_amp

        if mode_name == "vhs_linear":
            sig = _build_linear_track(sig)
            if np.any(shuttle_env > 1e-4):
                # During analog shuttle, linear head reads high-pitched warble with guard-band dips
                sig = sig * (1.0 - 0.45 * shuttle_env[:, None])

        elif mode_name == "vhs_hifi":
            # Hi-Fi AFM track: wide 20 Hz - 18 kHz frequency response, compander breathing,
            # and automatic switching to the muffled Linear Track when FM carrier unlocks!
            hifi = _highpass_audio(sig, 25.0, sr, order=2)
            hifi = _lowpass_audio(hifi, 18000.0, sr, order=4)
            # Subtle compander noise-reduction breathing (hiss modulated by signal envelope)
            sig_env = gaussian_filter1d(np.mean(np.abs(hifi), axis=1), sigma=max(10, int(0.012 * sr)))
            fm_hiss = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 800.0, 15000.0, sr, order=2)
            hifi = hifi + fm_hiss * (0.0045 * self.spec.hiss_floor * (0.35 + 1.8 * sig_env))[:, None]

            # Evaluate FM carrier unlock envelope from dropouts + tracking error + shuttle
            unlock_drive = dropout_env + 0.85 * shuttle_env + 0.85 * ctl_unlock_env
            unlock_raw = np.clip((unlock_drive - self.spec.hifi_unlock_thresh) / 0.18, 0.0, 1.0)
            # Hold relay time constant (~8 ms attack/release) + derivative crackle burst at switching edges!
            unlock_env = gaussian_filter1d(unlock_raw, sigma=max(1.0, 0.006 * sr))
            switch_edge = np.abs(np.diff(unlock_env, prepend=unlock_env[0])) * float(sr) * 0.008
            switch_crackle = (rng.standard_normal((n_samples, 2)).astype(np.float32)
                              * np.clip(switch_edge, 0.0, 1.2)[:, None] * 0.28 * self.spec.dropout_sensitivity)

            if np.any(unlock_env > 1e-3):
                lin_fallback = _build_linear_track(sig)
                u_col = np.clip(unlock_env, 0.0, 1.0)[:, None]
                sig = hifi * (1.0 - u_col) + lin_fallback * u_col + switch_crackle
            else:
                sig = hifi

        elif mode_name == "hi8_afm":
            # 8mm / Hi8 AFM audio: 1.5 MHz single FM carrier on drum; dropouts cause direct FM static bursts
            afm = _highpass_audio(sig, 35.0, sr, order=2)
            afm = _lowpass_audio(afm, 15000.0 * (1.0 - 0.25 * az), sr, order=4)
            burst_env = np.clip(dropout_env * 0.85 + shuttle_env * 0.75 + ctl_unlock_env, 0.0, 1.0)[:, None]
            fm_static = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 500.0, 11000.0, sr, order=2)
            hiss_base = fm_static * (0.009 * self.spec.hiss_floor)
            sig = afm * (1.0 - 0.82 * burst_env) + fm_static * (burst_env * 0.32 * self.spec.dropout_sensitivity) + hiss_base

        elif mode_name == "minidv_pcm":
            # MiniDV 12-bit 32 kHz nonlinear PCM quantization + sample-hold / framing clicks on dropout/shuttle
            # 1. Downsample-and-hold at 32 kHz
            step_32k = max(1, int(round(sr / 32000.0)))
            if step_32k > 1:
                held_idx = (np.arange(n_samples) // step_32k) * step_32k
                sig = sig[held_idx]
            # 2. 12-bit nonlinear companded quantization (fewer steps at higher levels)
            levels = 1024.0
            companded = np.sign(sig) * np.log1p(15.0 * np.abs(sig)) / np.log1p(15.0)
            quant = np.round(companded * levels) / levels
            sig = np.sign(quant) * (np.expm1(np.abs(quant) * np.log1p(15.0)) / 15.0).astype(np.float32)

            # 3. Digital ECC failure during severe dropouts or shuttle -> 12 ms sample-block hold or mute + click
            ecc_fail = (dropout_env > 0.42) | (shuttle_env > 0.35) | (ctl_unlock_env > 0.35)
            if np.any(ecc_fail):
                blk_len = max(32, int(0.010 * sr))
                for b0 in range(0, n_samples, blk_len):
                    b1 = min(n_samples, b0 + blk_len)
                    if np.mean(ecc_fail[b0:b1]) > 0.30:
                        if (b0 // blk_len) % 2 == 0 and b0 > 0:
                            # Repeat last valid sample or short grain buffer (classic DV shuttle stutter)
                            grain_len = min(128, b0)
                            reps = int(np.ceil((b1 - b0) / float(grain_len)))
                            sig[b0:b1] = np.tile(sig[b0 - grain_len : b0], (reps, 1))[: b1 - b0] * 0.75
                        else:
                            # Brief PCM mute packet
                            sig[b0:b1] *= 0.05

        elif mode_name == "vcd_mp2":
            # Video CD MPEG-1 Layer II (1152-sample frame subband quantization, birdies & pre-echo)
            codec_cfg = self.params.get("codec") or {}
            bitrate = float(codec_cfg.get("bitrate", 1150.0))
            q_coarseness = float(np.clip(1200.0 / max(250.0, bitrate), 0.5, 4.0))
            sig = _lowpass_audio(sig, min(16500.0, 18000.0 / (q_coarseness ** 0.35)), sr, order=6)
            # STFT-like subband thresholding on 512-sample windows to produce authentic MP2 "watery" birdies & pre-echo
            win_len = 512
            hop = 256
            win = np.hanning(win_len).astype(np.float32)[:, None]
            out_mp2 = np.zeros_like(sig)
            w_norm = np.zeros((n_samples, 1), dtype=np.float32)
            for p0 in range(0, n_samples - win_len, hop):
                seg = sig[p0 : p0 + win_len] * win
                spec_f = np.fft.rfft(seg, axis=0)
                mag = np.abs(spec_f)
                # Psychoacoustic masking threshold per frame: drop quiet spectral bins
                thresh = np.max(mag, axis=0, keepdims=True) * (0.018 * q_coarseness)
                keep = np.clip((mag - thresh) / np.maximum(thresh * 0.5, 1e-6), 0.0, 1.0)
                # Quantize surviving spectral magnitudes
                q_step = max(1e-4, 0.012 * q_coarseness)
                spec_q = np.round(spec_f * keep / q_step) * q_step
                rec = np.fft.irfft(spec_q, n=win_len, axis=0).astype(np.float32)
                out_mp2[p0 : p0 + win_len] += rec * win
                w_norm[p0 : p0 + win_len] += win * win
            valid = w_norm[:, 0] > 1e-4
            sig[valid] = out_mp2[valid] / w_norm[valid]

        # ------------------------------------------------------------------
        # Step 4: Add Synchronized Physical Fault Signals
        # (Crease Thump, Base Cracks, CTL Breakage, Head-Switch Buzz, Mains Hum, RF Buzz, HDMI Drops)
        # ------------------------------------------------------------------
        # 4a. Low-frequency mechanical crease thump
        if np.any(np.abs(crease_thump) > 1e-5):
            thump_lp = _lowpass_audio(crease_thump[:, None], 110.0, sr, order=2)
            sig = sig + thump_lp

        # 4b. Base cracking sharp impulse clicks
        if np.any(np.abs(crack_clicks) > 1e-5):
            sig[:, 0] += crack_clicks
            sig[:, 1] += crack_clicks * 0.92

        # 4c. Base breakage CTL unlock static burst
        if np.any(ctl_unlock_env > 1e-4):
            brk_noise = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 200.0, 8000.0, sr, order=2)
            c_env = np.clip(ctl_unlock_env, 0.0, 1.0)[:, None]
            sig = sig * (1.0 - 0.75 * c_env) + brk_noise * (0.28 * c_env)

        # 4d. Head-Switching 59.94 Hz (NTSC) / 50 Hz (PAL) Pulse Train + Harmonics
        if np.any(hs_pulse_env > 1e-4):
            field_hz = 50.0 if "pal" in self.std_name else 59.94
            # Sawtooth + narrow switching spike at field rate
            ph_f = 2.0 * np.pi * field_hz * t_arr
            buzz_wave = (0.45 * np.sin(ph_f)
                         + 0.35 * np.sin(2.0 * ph_f + 0.4)
                         + 0.25 * np.sin(3.0 * ph_f + 0.9)
                         + 0.18 * np.sin(5.0 * ph_f)
                         + 0.12 * np.sin(9.0 * ph_f))
            sig = sig + (buzz_wave[:, None] * hs_pulse_env[:, None] * 0.14).astype(np.float32)

        # 4e. AV Connector Ground-Loop Mains Hum (`mains_hz`, 50 or 60 Hz + rectifier harmonics)
        if np.any(hum_env > 1e-4):
            conn_cfg = self.params.get("connector") or {}
            mains_hz = float(conn_cfg.get("mains_hz", 50.0))
            ph_m = 2.0 * np.pi * mains_hz * t_arr
            # Fundamental + 2nd harmonic (rectified corrosion) + 3rd harmonic (transformer core saturation)
            hum_wave = (0.65 * np.sin(ph_m)
                        + 0.25 * np.sin(2.0 * ph_m + 0.3)
                        + 0.20 * np.sin(3.0 * ph_m)
                        + 0.08 * np.sin(5.0 * ph_m))
            sig = sig + (hum_wave[:, None] * hum_env[:, None] * 0.16).astype(np.float32)

        # 4f. RF Coaxial Link Intercarrier Vision-into-Sound Buzz & Sync-Slip Static
        if np.any(rf_buzz_env > 1e-4) or np.any(rf_static_env > 1e-4):
            field_hz = 50.0 if "pal" in self.std_name else 59.94
            # Vision-into-sound intercarrier buzz has a harsh rasp at field_hz + harmonics up to 1.5 kHz
            ph_rf = 2.0 * np.pi * field_hz * t_arr
            rasp = (0.40 * np.sin(ph_rf)
                    + 0.35 * np.sin(2.0 * ph_rf)
                    + 0.30 * np.cos(4.0 * ph_rf)
                    + 0.25 * np.sin(8.0 * ph_rf)
                    + 0.18 * np.cos(16.0 * ph_rf))
            rf_noise = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 600.0, 9000.0, sr, order=2)
            sig = (sig * (1.0 - np.clip(0.35 * rf_static_env, 0.0, 0.75)[:, None])
                   + (rasp[:, None] * rf_buzz_env[:, None] * 0.13).astype(np.float32)
                   + rf_noise * (rf_static_env[:, None] * 0.22))

        # 4g. HDMI Bit-Error Clicks & Link-Drop Mute
        if np.any(hdmi_ber_env > 1e-4):
            # Serial bit errors corrupt PCM words in short runs, producing digital splats/clicks and brief dropouts
            blk = max(64, int(0.008 * sr))
            for b0 in range(0, n_samples, blk):
                b1 = min(n_samples, b0 + blk)
                ber_local = float(np.mean(hdmi_ber_env[b0:b1]))
                if ber_local > 1e-4 and rng.random() < ber_local * 0.85:
                    if rng.random() < 0.55:
                        sig[b0:b1] = 0.0
                    else:
                        # Corrupted MSB burst
                        sig[b0 : min(b1, b0 + 24)] = rng.uniform(-0.65, 0.65, size=(min(24, b1 - b0), 2)).astype(np.float32)
        if np.any(hdmi_mute_env > 0.5):
            sig = sig * (1.0 - hdmi_mute_env[:, None])

        # ------------------------------------------------------------------
        # Step 5: Magnetic Tape Hysteresis Soft-Clipping & Harmonic Saturation
        # ------------------------------------------------------------------
        sat = float(self.spec.saturation)
        if sat > 1e-4 and mode_name in ("vhs_linear", "vhs_hifi", "hi8_afm"):
            drive = 1.0 + 2.4 * sat
            # Asymmetric bias nonlinearity adds warm 2nd harmonic before odd-order `tanh` tape ceiling
            asym = 0.08 * sat
            driven = sig * drive + asym * (sig * sig)
            sig = (np.tanh(driven) - np.tanh(asym)) / np.tanh(drive)

        # ------------------------------------------------------------------
        # Step 6: CRT TV Speaker Cabinet Resonance & Horizontal Flyback Whine
        # ------------------------------------------------------------------
        display_cfg = self.params.get("display") or {}
        disp_on = float(display_cfg.get("enabled", 1.0)) > 0.5
        spk_mix = float(self.spec.speaker_sim) if disp_on else float(self.spec.speaker_sim) * 0.35
        if spk_mix > 1e-4:
            # Small 3-inch plastic TV cabinet speaker: rolls off bass below 190 Hz, resonant bump around 1.1 kHz,
            # and paper-cone rolloff above 5.8 kHz
            spk = _highpass_audio(sig, 185.0, sr, order=2)
            spk = _lowpass_audio(spk, 5800.0, sr, order=3)
            mid_bump = _bandpass_audio(spk, 750.0, 1800.0, sr, order=2)
            spk = spk + 0.42 * mid_bump
            sig = sig * (1.0 - spk_mix) + spk * spk_mix

        whine = float(self.spec.flyback_whine) if disp_on else 0.0
        if whine > 1e-4:
            # 15,734.26 Hz (NTSC) or 15,625.0 Hz (PAL) horizontal deflection flyback transformer whine
            f_line = 15625.0 if "pal" in self.std_name else 15734.26
            if f_line < 0.48 * sr:
                fly_wave = np.sin(2.0 * np.pi * f_line * t_arr)[:, None].astype(np.float32)
                sig = sig + fly_wave * (0.018 * whine)

        # Final output gain & soft limiter
        sig = sig * float(self.spec.master_gain)
        if np.any(hdmi_mute_env > 0.5):
            sig = sig * (1.0 - hdmi_mute_env[:, None])
        sig = np.clip(sig, -0.99, 0.99).astype(np.float32)

        # Compute summary metrics for UI and verification
        fault_frame_count = sum(
            1 for f in frame_faults
            if (f.dropout_level > 0.05 or f.stiction_grab > 0.05 or f.head_switch_buzz > 0.05
                or f.rf_sync_buzz > 0.05 or f.ground_hum > 0.05 or f.hdmi_drop > 0.05
                or f.crack_spike > 0.05 or f.shuttle_chop > 0.05)
        )
        summary = {
            "mode": mode_name,
            "sample_rate": sr,
            "start_s": round(start_s, 3),
            "duration_s": round(duration_s, 3),
            "speed_ratio": round(mean_speed_ratio, 3),
            "max_pitch_dev_pct": round(max((f.pitch_deviation_pct for f in frame_faults), default=0.0), 3),
            "max_dropout": round(max((f.dropout_level for f in frame_faults), default=0.0), 3),
            "hifi_fallback_frames": sum(1 for f in frame_faults if f.hifi_fallback > 0.1),
            "fault_frames": fault_frame_count,
            "rms_db": round(float(20.0 * np.log10(max(1e-6, float(np.sqrt(np.mean(sig ** 2)))))), 2),
        }
        return AudioRun(
            samples=sig,
            clean_samples=clean,
            sample_rate=sr,
            start_s=start_s,
            duration_s=duration_s,
            resolved_mode=mode_name,
            frame_faults=frame_faults,
            summary=summary,
        )

    def simulate(self, clean_stereo: np.ndarray, start_s: float = 0.0,
                 duration_s: float | None = None, **kwargs) -> AudioRun:
        return self.render_window(clean_stereo, start_s=start_s, duration_s=duration_s, **kwargs)

    def process(self, clean_stereo: np.ndarray, start_s: float = 0.0,
                duration_s: float | None = None, tape_plan=None, tape_stage=None, **kwargs) -> AudioRun:
        return self.render_window(
            clean_stereo, start_s=start_s, duration_s=duration_s,
            tape_stage=tape_stage, tape_plan=tape_plan, **kwargs
        )

    def extract_fault_envelope(self, start_frame: int, n_frames: int, tape_plan=None) -> list[AudioFaultFrame]:
        dur_s = max(1, int(n_frames)) / self.fps
        dummy = np.zeros((max(16, int(round(dur_s * 8000))), 2), dtype=np.float32)
        orig_sr = self.sample_rate
        try:
            self.sample_rate = 8000
            run = self.render_window(dummy, start_s=start_frame / self.fps, duration_s=dur_s, tape_plan=tape_plan)
            return run.frame_faults
        finally:
            self.sample_rate = orig_sr

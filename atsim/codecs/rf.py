"""
RF modulation / tuner link and the Radio / TV-DX Signal Chain Subsystem.

Designed per `note.txt`:
  Source -> Modulator (PAL-D 8MHz/+6.5MHz, PAL-I 8MHz/+6.0MHz, NTSC-M 6MHz/+4.5MHz)
    -> Propagation Channel (LOS/diffraction path loss, Tropospheric Ducting / Sporadic-E
       TV-DX anomalous propagation gain, QSB slow ionospheric/duct fading, complex multipath
       r(t) = sum_k a_k s(t - tau_k) exp(j phi_k), co-channel precision-offset beat &
       adjacent-channel interference)
    -> Receiver Front-End & Tuner (LNA noise figure, thermal kTB noise floor, front-end
       nonlinear compression/overload, LO tuning offset, IF SAW filter & cheap Taobao
       SAA713x / TDA9887 silicon tuner group-delay ringing)
    -> Demodulator & Sync / Color-Killer (Horizontal AFC sync PLL, burst phase error,
       automatic color killer below threshold, FM intercarrier sound C/N & capture effect).

All nodes implement the unified `SignalNode` interface (`process(signal, state, dt)` and
`reset()`) while maintaining 100% compatibility with `RFParams` and `RFLink`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.rng import OUProcess
from ..core.signal import horizontal_shift, lowpass
from ..core.standards import VideoStandard

# Thermal noise density at 290 K: -174 dBm/Hz => -106.2 dBm in a 6 MHz channel
THERMAL_FLOOR_6MHZ_DBM = -106.2

# TV Broadcast RF Standards: (channel_bw_mhz, vision_offset_mhz, sound_offset_mhz, video_bw_mhz, label)
TV_RF_STANDARDS: dict[int, dict[str, Any]] = {
    0: {
        "code": "PAL-D",
        "label": "PAL-D/K (中国大陆 8MHz / +6.5MHz 伴音)",
        "channel_mhz": 8.0,
        "vision_offset_mhz": 1.25,
        "sound_offset_mhz": 6.50,
        "video_bw_mhz": 6.0,
        "chroma_subcarrier_mhz": 4.43361875,
    },
    1: {
        "code": "PAL-I",
        "label": "PAL-I (香港/珠三角外溢 8MHz / +6.0MHz 伴音)",
        "channel_mhz": 8.0,
        "vision_offset_mhz": 1.25,
        "sound_offset_mhz": 6.00,
        "video_bw_mhz": 5.5,
        "chroma_subcarrier_mhz": 4.43361875,
    },
    2: {
        "code": "NTSC-M",
        "label": "NTSC-M (6MHz / +4.5MHz 伴音)",
        "channel_mhz": 6.0,
        "vision_offset_mhz": 1.25,
        "sound_offset_mhz": 4.50,
        "video_bw_mhz": 4.2,
        "chroma_subcarrier_mhz": 3.579545,
    },
}

# Propagation modes:
#   0 = Normal terrestrial LOS / diffraction
#   1 = Tropospheric Ducting TV-DX (大气波导超视距接收: high anomalous gain, slow duct breathing, crisp duct ghost)
#   2 = Sporadic-E (Es) Ionospheric Skywave TV-DX (电离层 Es 突发层反射: strong skywave gain, deeper QSB fading, skywave multipath)
#   3 = Weak Beyond-LOS Troposcatter (常规超视距弱信号散射对照: no duct gain, heavy snow & sync instability)
PROPAGATION_MODES: dict[int, str] = {
    0: "terrestrial_los",
    1: "tropospheric_duct_dx",
    2: "sporadic_e_skywave_dx",
    3: "weak_troposcatter",
}


class SignalNode:
    """
    Unified physical RF signal-chain node interface (`note.txt` §5).

    Every stage in the radio chain takes a signal array, a mutable state dictionary,
    and a frame time step `dt`, returning `(output_signal, updated_state)`.
    """

    def process(self, signal: np.ndarray, state: dict[str, Any], dt: float
                ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return `(output_signal, updated_state)`."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal temporal memory for reproducible experiments."""


@dataclass
class RFParams:
    """
    Physical parameters for the RF / TV-DX link.

    Retains all original parameters with identical defaults so existing callers and
    verification suites behave identically, while adding the full TV-DX & Taobao
    tuner card parameter set.
    """
    # --- Legacy / Core parameters ---
    channel_mhz: float = 6.0              # total RF channel width (6 MHz M, 8 MHz D/K/I)
    carrier_offset_mhz: float = 1.25      # visual carrier above lower band edge (VSB)
    noise_figure_db: float = 8.0          # receiver front-end noise figure
    signal_dbm: float = -60.0             # nominal received carrier power at antenna
    sync_loss_threshold_db: float = 30.0  # C/N below which horizontal AFC begins to slip
    sync_loss_amount: float = 1.0         # 0..2 scale on AFC tearing
    adjacent_channel: float = 0.0         # 0..1 adjacent-channel carrier beat
    broadcast_noise: float = 0.0          # 0..1 extra impulsive / broadcast band noise

    # --- Radio / TV-DX & Taobao TV Card parameters ---
    enabled: float = 0.0                  # 0 or 1 in linear chain (always active in RFLink direct calls)
    tv_standard: float = 0.0              # 0=PAL-D (6.5MHz), 1=PAL-I (6.0MHz), 2=NTSC-M (4.5MHz)
    propagation_mode: float = 0.0         # 0=LOS, 1=Tropospheric Duct TV-DX, 2=Es Skywave TV-DX, 3=Weak Scatter
    distance_km: float = 45.0             # transmitter-receiver distance (km)
    dx_duct_gain_db: float = 24.0         # anomalous propagation gain (dB) during active TV-DX ducting / Es
    qsb_depth_db: float = 0.0             # slow fading (QSB) depth in dB (0..18 dB)
    qsb_rate_hz: float = 0.25             # slow fading oscillation rate in Hz (0.02..2.0 Hz)
    multipath_level: float = 0.0          # secondary reflection amplitude a_1 (0..0.85)
    multipath_delay_us: float = 0.85      # secondary path delay tau_1 in microseconds (0.1..4.0 us)
    multipath_phase_deg: float = 35.0     # secondary path carrier phase phi_1 in degrees (-180..180)
    cochannel_interference: float = 0.0   # co-channel venetian-blind interference (0..1)
    cochannel_offset_khz: float = 10.05   # co-channel precision carrier offset in kHz (1..50 kHz)
    tuner_type: float = 0.0               # 0=Standard TV Tuner, 1=Taobao Cheap TV Card (SAA713x/TDA9887), 2=Vintage Rotary
    tuning_offset_khz: float = 0.0        # receiver local oscillator fine-tuning error (-500..500 kHz)
    front_end_overload: float = 0.0       # LNA/mixer compression & intermodulation overload (0..1)
    color_killer_threshold_db: float = 18.0  # C/N threshold below which tuner mutes colour burst (dB)


class TransmitterNode(SignalNode):
    """
    Stage 1 of RF chain: Vestigial-Sideband (VSB) negative video modulation & sound carrier.

    In PAL-D/I and NTSC-M, terrestrial analog TV uses negative AM modulation (brightest
    white = minimum carrier envelope ~10-12.5%, sync tip = 100% peak envelope) so impulse
    noise appears as dark/grey specks rather than blinding white flashes, and the IF
    bandwidth is shaped by the vestigial sideband filter.
    """

    def __init__(self, std: VideoStandard, params: RFParams):
        self.std = std
        self.p = params

    def process(self, signal: np.ndarray, state: dict[str, Any], dt: float
                ) -> tuple[np.ndarray, dict[str, Any]]:
        std_idx = int(round(float(np.clip(self.p.tv_standard, 0.0, 2.0))))
        rf_spec = TV_RF_STANDARDS.get(std_idx, TV_RF_STANDARDS[0])

        # Effective channel bandwidth: if user explicitly moved channel_mhz off default 6.0,
        # honour it; otherwise use the TV standard's video bandwidth
        ch_mhz = float(self.p.channel_mhz)
        if std_idx in (0, 1) and abs(ch_mhz - 6.0) < 1e-3:
            eff_ch_mhz = float(rf_spec["channel_mhz"])
        else:
            eff_ch_mhz = ch_mhz

        usable_hz = max(0.8e6, (eff_ch_mhz - self.p.carrier_offset_mhz) * 1e6)
        sigma = self.std.pixels_for_bandwidth(usable_hz, signal.shape[1])
        shaped = lowpass(signal, sigma, axis=1).astype(np.float32)

        state["rf_standard"] = rf_spec["code"]
        state["sound_offset_mhz"] = float(rf_spec["sound_offset_mhz"])
        state["effective_channel_mhz"] = round(eff_ch_mhz, 2)
        state["usable_video_bw_mhz"] = round(usable_hz * 1e-6, 2)
        return shaped, state


class PropagationChannelNode(SignalNode):
    """
    Stage 2 of RF chain: Terrestrial / Atmospheric Duct (TV-DX) / Sporadic-E Skywave channel.

    Models:
      1. Distance-dependent diffraction attenuation beyond the radio horizon (~55 km) vs.
         anomalous propagation gain (`dx_duct_gain_db`) when `propagation_mode` is set to
         Tropospheric Ducting (1) or Sporadic-E Skywave (2).
      2. Slow QSB fading (`qsb_depth_db`, `qsb_rate_hz`) from ionospheric cloud drift or
         duct boundary undulation.
      3. Multi-path propagation (`r(t) = sum_k a_k s(t - tau_k) exp(j phi_k)`), where the
         carrier phase `phi_k` determines whether the delayed reflection adds, cancels, or
         inverts the visual carrier and rotates the chroma burst phase.
      4. Co-channel precision-offset venetian-blind beat (`cochannel_interference`) and
         adjacent-channel diagonal herringbone beat (`adjacent_channel`).
    """

    def __init__(self, std: VideoStandard, params: RFParams, seed: int = 20240101):
        self.std = std
        self.p = params
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._qsb_ou = OUProcess(
            np.random.default_rng((self.seed * 131071 + 43) & 0xFFFFFFFF),
            sigma=1.0, corr_len=18.0
        )
        self._qsb_phase = 0.0
        self._mp_drift_phase = 0.0

    def effective_signal_dbm(self, frame_index: int, dt: float) -> tuple[float, float, float]:
        """
        Compute `(received_signal_dbm, path_delta_db, qsb_offset_db)` at `frame_index`.

        When `propagation_mode == 0` and `distance_km == 45.0` and `qsb_depth_db == 0.0`
        (the defaults), `received_signal_dbm` equals `self.p.signal_dbm` identically.
        """
        mode = int(round(float(np.clip(self.p.propagation_mode, 0.0, 3.0))))
        dist = max(1.0, float(self.p.distance_km))
        duct_gain = float(np.clip(self.p.dx_duct_gain_db, 0.0, 45.0))

        # Beyond-line-of-sight diffraction loss relative to 45 km reference
        if abs(dist - 45.0) < 1e-2 and mode == 0:
            path_delta_db = 0.0
        else:
            # Free-space + diffraction beyond radio horizon (~55 km)
            fspl_delta = -20.0 * np.log10(dist / 45.0)
            beyond_horizon_km = max(0.0, dist - 55.0)
            # Normal tropospheric diffraction loses ~0.14 dB/km beyond horizon
            diffraction_loss = -0.14 * (beyond_horizon_km ** 0.85)
            if mode == 0:
                path_delta_db = fspl_delta + 0.35 * diffraction_loss
            elif mode == 1:
                # Tropospheric ducting (大气波导 TV-DX): waveguide trapping cancels
                # diffraction loss and provides strong anomalous ducting gain!
                duct_coupling = np.clip((dist - 20.0) / 60.0, 0.85, 1.0)
                path_delta_db = 0.20 * fspl_delta + duct_gain * duct_coupling
            elif mode == 2:
                # Sporadic-E (Es) ionospheric skywave reflection (电离层活跃 Es 传播):
                # single-hop Es strongly reflects VHF bands over 150..1200 km
                es_window = np.exp(-0.5 * ((np.log10(max(60.0, dist)) - np.log10(350.0)) / 0.55) ** 2)
                path_delta_db = 0.15 * fspl_delta + duct_gain * (0.75 + 0.25 * es_window)
            else:
                # Mode 3: Combined Tropospheric Ducting + Active Es Ionosphere
                path_delta_db = 0.15 * fspl_delta + duct_gain * 1.08

        # QSB slow ionospheric / atmospheric duct fading
        qsb_depth = float(np.clip(self.p.qsb_depth_db, 0.0, 24.0))
        if mode == 2 and qsb_depth < 1e-3:
            # Active Es skywave naturally has a gentle QSB undulation unless overridden
            qsb_depth = 3.5
        elif mode == 1 and qsb_depth < 1e-3 and dist > 80.0:
            qsb_depth = 1.5

        qsb_offset_db = 0.0
        if qsb_depth > 1e-3:
            rate = float(np.clip(self.p.qsb_rate_hz, 0.02, 3.0))
            t_sec = frame_index * dt
            ou_val = float(self._qsb_ou.Advance(1)[0]) if hasattr(self._qsb_ou, "Advance") else float(self._qsb_ou.advance(1)[0])
            wave = 0.72 * np.sin(2.0 * np.pi * rate * t_sec + 0.6) + 0.28 * np.clip(ou_val * 0.5, -1.0, 1.0)
            qsb_offset_db = float(qsb_depth * wave)

        rx_dbm = float(self.p.signal_dbm + path_delta_db + qsb_offset_db)
        return rx_dbm, float(path_delta_db), float(qsb_offset_db)

    def process(self, signal: np.ndarray, state: dict[str, Any], dt: float
                ) -> tuple[np.ndarray, dict[str, Any]]:
        frame_index = int(state.get("frame_index", 0))
        h, w = signal.shape[:2]
        out = signal.astype(np.float32, copy=True)

        rx_dbm, path_delta_db, qsb_offset_db = self.effective_signal_dbm(frame_index, dt)
        mode = int(round(float(np.clip(self.p.propagation_mode, 0.0, 3.0))))

        # --- Multipath propagation: r(t) = s(t) + sum_k a_k s(t - tau_k) exp(j phi_k) ---
        mp_level = float(np.clip(self.p.multipath_level, 0.0, 0.90))
        if mp_level > 1e-4:
            # Convert delay in microseconds to horizontal pixels on the current raster
            active_line_us = (1e6 / max(1e3, self.std.line_hz)) * 0.82  # ~52 us active line
            px_per_us = w / max(1.0, active_line_us)
            delay_us = float(np.clip(self.p.multipath_delay_us, 0.08, 6.0))
            delay_px = delay_us * px_per_us

            # Carrier phase phi_k(t): in Es skywave or ducting, the reflection height
            # drifts by centimetres per second, rotating phi_k slowly across frames
            phase_drift_rad = (frame_index * dt * 1.35) if mode in (1, 2) else 0.0
            phi_rad = np.deg2rad(float(self.p.multipath_phase_deg)) + phase_drift_rad
            # Envelope detector projects exp(j phi_k) onto the in-phase carrier axis (cos phi)
            # while quadrature component (sin phi) produces edge quadrature distortion & chroma phase shift
            in_phase_gain = float(mp_level * np.cos(phi_rad))
            quad_gain = float(mp_level * np.sin(phi_rad))

            shifts1 = np.full(h, delay_px, dtype=np.float32)
            ghost1 = horizontal_shift(signal, shifts1, mode="edge")
            # Quadrature derivative term on sharp edges (vestigial sideband quadrature distortion)
            ghost1_edge = ghost1 - lowpass(ghost1, max(0.6, 1.2 * (w / 720.0)), axis=1)

            # Secondary weaker reflection (2-hop or ground-reflected duct echo)
            shifts2 = np.full(h, -0.42 * delay_px if mode == 1 else 1.85 * delay_px, dtype=np.float32)
            ghost2 = horizontal_shift(signal, shifts2, mode="edge")
            g2_gain = float(0.28 * mp_level * np.cos(phi_rad * 1.7 + 0.9))

            out = out + in_phase_gain * ghost1 + (0.38 * quad_gain) * ghost1_edge + g2_gain * ghost2
            state["multipath_delay_px"] = round(float(delay_px), 2)
            state["multipath_in_phase"] = round(in_phase_gain, 4)
            state["multipath_chroma_phase_rad"] = float(np.arctan2(
                mp_level * np.sin(phi_rad), 1.0 + mp_level * np.cos(phi_rad)
            ))
        else:
            state["multipath_delay_px"] = 0.0
            state["multipath_in_phase"] = 0.0
            state["multipath_chroma_phase_rad"] = 0.0

        # --- Co-channel interference (Venetian blind / 同频百叶窗干扰) ---
        # In TV-DX ducting, a distant co-channel station on the same frequency (with a
        # 10 kHz / 15 kHz precision carrier offset to reduce visibility) arrives via the
        # duct and beats against the desired visual carrier, creating horizontal dark/bright
        # bars (Venetian blinds) that roll slowly up or down the screen.
        co_int = float(np.clip(self.p.cochannel_interference, 0.0, 1.0))
        if co_int > 1e-4:
            offset_khz = float(np.clip(self.p.cochannel_offset_khz, 0.5, 60.0))
            # Number of horizontal beat cycles per vertical field
            cycles_per_field = (offset_khz * 1e3 / max(1.0, self.std.line_hz)) * (h / 288.0)
            cycles_per_field = float(np.clip(cycles_per_field, 2.5, 42.0))
            yy = np.arange(h, dtype=np.float32)[:, None]
            xx = np.arange(w, dtype=np.float32)[None, :]
            # Slow vertical crawl as the two stations' crystal oscillators drift
            v_phase = 2.0 * np.pi * (cycles_per_field * (yy / max(1.0, float(h))) - frame_index * 0.19)
            # Subtle horizontal ghosting of the interfering station's sync bar
            sync_ghost_x = np.exp(-0.5 * ((xx - (0.32 + 0.01 * frame_index % 0.4) * w) / max(4.0, 0.035 * w)) ** 2)
            venetian = (0.85 * np.cos(v_phase) + 0.25 * np.cos(2.0 * v_phase)) - 0.35 * sync_ghost_x
            out = out + (0.14 * co_int) * venetian.astype(np.float32)

        # --- Adjacent-channel interference (diagonal herringbone beat) ---
        if self.p.adjacent_channel > 1e-4:
            beat_cycles_per_line = 3.17
            cols = np.arange(w, dtype=np.float32)[None, :] / max(1, w)
            rows = np.arange(h, dtype=np.float32)[:, None]
            ph = 2.0 * np.pi * (beat_cycles_per_line * cols
                                + 0.13 * rows + 0.19 * frame_index)
            out = out + (0.11 * float(self.p.adjacent_channel)) * np.cos(ph).astype(np.float32)

        state["received_signal_dbm"] = round(rx_dbm, 2)
        state["path_delta_db"] = round(path_delta_db, 2)
        state["qsb_offset_db"] = round(qsb_offset_db, 2)
        state["propagation_mode_name"] = PROPAGATION_MODES.get(mode, "terrestrial_los")
        return out.astype(np.float32), state


class TunerFrontEndNode(SignalNode):
    """
    Stage 3 of RF chain: Antenna -> LNA -> Mixer/LO -> SAW IF Filter -> AGC.

    Models:
      1. Receiver noise floor from thermal kTB + receiver Noise Figure (`noise_figure_db`)
         scaled to actual channel bandwidth (`10 * log10(B / 6MHz)`), plus Taobao cheap
         PCI/USB TV card silicon tuner noise penalty (+1.8 dB wideband phase/switching noise).
      2. Carrier-to-Noise ratio (`cn_db = P_signal,dBm - P_noise,dBm`).
      3. Local Oscillator (LO) fine-tuning frequency offset (`tuning_offset_khz`): shifts the
         visual carrier along the Nyquist slope of the IF SAW filter, causing either high-frequency
         smearing (tuned too low) or exaggerated IF group-delay ringing / sound-in-picture beat
         (tuned too high).
      4. Front-end amplifier / mixer nonlinearity (`front_end_overload`): soft tanh/cubic
         compression that crushes sync/white peaks and generates intermodulation herringbone.
      5. Taobao cheap TV card (`tuner_type == 1`, e.g., Philips SAA7130/7134 + TDA9887) IF
         SAW filter group-delay overshoot (sharp dark/bright ringing right of vertical edges)
         and stepped digital AGC.
    """

    def __init__(self, std: VideoStandard, params: RFParams):
        self.std = std
        self.p = params

    def noise_floor_dbm(self, eff_channel_mhz: float | None = None) -> float:
        bw = float(eff_channel_mhz if eff_channel_mhz is not None else self.p.channel_mhz)
        bw_term = 10.0 * np.log10(max(0.5, bw) / 6.0)
        tuner_kind = int(round(float(np.clip(self.p.tuner_type, 0.0, 2.0))))
        # Cheap Taobao USB/PCI TV card has higher LNA/USB switching noise floor (+1.5 dB),
        # vintage rotary turret tuner has higher mixer NF (+2.5 dB)
        extra_nf = 1.5 if tuner_kind == 1 else (2.5 if tuner_kind == 2 else 0.0)
        return float(THERMAL_FLOOR_6MHZ_DBM + bw_term + self.p.noise_figure_db + extra_nf)

    def process(self, signal: np.ndarray, state: dict[str, Any], dt: float
                ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = signal.shape[:2]
        scale_w = w / 720.0
        out = signal.astype(np.float32, copy=True)

        eff_bw = float(state.get("effective_channel_mhz", self.p.channel_mhz))
        rx_dbm = float(state.get("received_signal_dbm", self.p.signal_dbm))
        nf_dbm = self.noise_floor_dbm(eff_bw)
        cn_db = float(rx_dbm - nf_dbm)

        tuner_kind = int(round(float(np.clip(self.p.tuner_type, 0.0, 2.0))))
        tune_off_khz = float(np.clip(self.p.tuning_offset_khz, -500.0, 500.0))

        # --- 1. LO Fine-Tuning Offset (`tuning_offset_khz`) ---
        if abs(tune_off_khz) > 1.0:
            norm_off = tune_off_khz / 250.0  # -2..+2
            if norm_off < 0.0:
                # Detuned into low-frequency vestigial sideband: loss of high-frequency detail
                blur_sig = max(0.4, abs(norm_off) * 2.2 * scale_w)
                out = lowpass(out, blur_sig, axis=1)
            else:
                # Detuned toward upper sound carrier: loss of low-frequency carrier pedestal
                # (relief/edge overshoot effect) + 6.0/6.5 MHz sound-in-picture fine beat
                lp = lowpass(out, max(0.8, 1.8 * scale_w), axis=1)
                detail = out - lp
                out = out + min(0.85, 0.42 * norm_off) * detail

        # --- 2. Taobao Cheap TV Card (`tuner_type == 1`) IF SAW Filter Group-Delay Ringing ---
        if tuner_kind == 1:
            # Cheap SAW filter + SAA713x peaking produces a characteristic damped ringing
            # oscillation (~2.5-4.5 px period) immediately to the right of sharp luma transitions
            lp1 = lowpass(out, max(0.6, 1.15 * scale_w), axis=1)
            edge_grad = out - lp1
            ring1 = horizontal_shift(edge_grad, np.full(h, 2.2 * scale_w, np.float32), mode="edge")
            ring2 = horizontal_shift(edge_grad, np.full(h, 4.4 * scale_w, np.float32), mode="edge")
            out = out - 0.16 * ring1 + 0.08 * ring2

            # Stepped AGC contrast boost when signal is strong (TV-DX ducting peak)
            if cn_db > 30.0:
                agc_boost = float(np.clip((cn_db - 30.0) / 25.0, 0.0, 0.10))
                out = (out - 0.45) * (1.0 + agc_boost) + 0.45

        # --- 3. Front-End Nonlinearity & Overload (`front_end_overload`) ---
        overload = float(np.clip(self.p.front_end_overload, 0.0, 1.0))
        if overload > 1e-4:
            drive = 1.0 + 2.4 * overload
            centred = out - 0.5
            # Asymmetric soft saturation (even + odd harmonics in overloaded LNA/mixer)
            compressed = np.tanh(centred * drive) / np.tanh(drive) + 0.08 * overload * (centred ** 2)
            out = 0.5 + compressed.astype(np.float32)
            # Intermodulation beat between visual, chroma (4.43MHz) and sound (6.0/6.5MHz) carriers
            xx = np.arange(w, dtype=np.float32)[None, :]
            yy = np.arange(h, dtype=np.float32)[:, None]
            fidx = int(state.get("frame_index", 0))
            im3_wave = np.cos(2.0 * np.pi * (xx / max(2.5, 4.8 * scale_w) - yy / 8.0) + fidx * 0.65)
            out = out + (0.075 * overload) * im3_wave.astype(np.float32)

        state["noise_floor_dbm"] = round(nf_dbm, 2)
        state["cn_db"] = round(cn_db, 2)
        state["tuner_type"] = tuner_kind
        state["tuning_offset_khz"] = round(tune_off_khz, 1)
        return out.astype(np.float32), state


class DemodulatorNode(SignalNode):
    """
    Stage 4 of RF chain: Envelope / Quadrature Demodulator, IF Snow, AFC Sync PLL & Color Killer.

    Models:
      1. IF band-limited snow whose RMS amplitude is derived directly from `cn_db`, plus
         negative-modulation impulse dark/white streamer ticks at low C/N (`cn_db < 24 dB`).
      2. Horizontal AFC sync separator & PLL slip (`_last_phase`), which shifts scanlines
         AND invalidates the colour burst timing reference (`chroma_phase_error`).
      3. Automatic Color Killer (`color_killer_gain`): when `cn_db` drops below
         `color_killer_threshold_db` (or when tuning offset is extreme), the decoder
         attenuates the colour subcarrier so weak channels drop into noisy greyscale
         while TV-DX ducted channels lock into vivid full colour.
      4. Audio FM intercarrier C/N and FM capture effect metrics for downstream audio synthesis.
    """

    def __init__(self, std: VideoStandard, params: RFParams, seed: int = 20240101):
        self.std = std
        self.p = params
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._afc = OUProcess(
            np.random.default_rng((self.seed * 65537 + 19) & 0xFFFFFFFF),
            sigma=1.0, corr_len=9.0
        )
        self._last_phase: np.ndarray | None = None
        self._last_color_killer_gain: float = 1.0
        self._last_multipath_chroma_rad: float = 0.0

    def process(self, signal: np.ndarray, state: dict[str, Any], dt: float
                ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = signal.shape[:2]
        scale_w = w / 720.0
        frame_index = int(state.get("frame_index", 0))
        rng = state.get("rng") or np.random.default_rng(
            (self.seed * 8191 + frame_index * 131) & 0xFFFFFFFF
        )

        cn_db = float(state.get("cn_db", 40.0))
        tuner_kind = int(state.get("tuner_type", 0))
        out = signal.astype(np.float32, copy=True)

        # --- 1. IF Snow from C/N (monotonically decreasing with C/N) ---
        cn_eff = cn_db - 22.0
        noise_rms = float(np.clip(10.0 ** (-cn_eff / 20.0), 0.0, 0.85))
        bcast = float(np.clip(self.p.broadcast_noise, 0.0, 1.0))
        if bcast > 1e-4:
            noise_rms = float(np.hypot(noise_rms, 0.32 * bcast))

        if noise_rms > 1e-4:
            raw = rng.standard_normal(out.shape).astype(np.float32)
            # IF-filtered snow: correlated horizontally over the IF bandwidth
            usable_hz = max(0.8e6, (self.p.channel_mhz - self.p.carrier_offset_mhz) * 1e6)
            if_sigma = max(0.55, self.std.pixels_for_bandwidth(usable_hz, w) * 0.85)
            snow = lowpass(raw, if_sigma, axis=1)
            snow *= 1.0 / max(1e-6, float(snow.std()))
            out = out + noise_rms * snow

            # Analog negative-modulation impulse "sparkle" / sync-tip threshold ticks
            # when C/N is weak (< 24 dB) or broadcast_noise > 0
            weakness = float(np.clip((24.0 - cn_db) / 14.0, 0.0, 1.0)) + 0.5 * bcast
            if weakness > 0.05:
                n_ticks = int(rng.poisson(weakness * 35.0))
                if n_ticks > 0:
                    ty = rng.integers(0, h, size=n_ticks)
                    tx = rng.integers(0, max(1, w - 12), size=n_ticks)
                    tlen = rng.integers(3, max(4, int(14 * scale_w)), size=n_ticks)
                    pol = rng.choice(np.array([-0.45, 0.55], dtype=np.float32), size=n_ticks)
                    for r_i, c_i, l_i, p_i in zip(ty, tx, tlen, pol):
                        out[int(r_i), int(c_i):min(w, int(c_i + l_i))] += p_i * weakness

        # --- 2. Horizontal AFC Sync Loss (line tearing & burst timing error) ---
        margin_db = self.p.sync_loss_threshold_db - cn_db
        slip_px = 0.0
        if margin_db > 0.0 and self.p.sync_loss_amount > 1e-4:
            severity = float(np.clip(margin_db / 18.0, 0.0, 1.5)) * self.p.sync_loss_amount
            wander = self._afc.advance(h).astype(np.float32)
            wander *= (severity * 14.0) * (w / 720.0)
            # Top-of-field pull-in after vertical blanking
            pull_in = np.exp(-np.arange(h, dtype=np.float32) / max(4.0, h * 0.08))
            pull_in *= (severity * 22.0) * (w / 720.0) * float(rng.standard_normal())
            shifts = wander + pull_in
            out = horizontal_shift(out, shifts)
            slip_px = float(np.std(shifts))
            self._last_phase = shifts * float(self.std.subcarrier_rad_per_sample)
        else:
            self._last_phase = None

        # --- 3. Automatic Color Killer & Fine-Tuning Chroma Phase ---
        ck_thresh = float(self.p.color_killer_threshold_db)
        if ck_thresh <= 0.0:
            ck_gain = 1.0
        else:
            # Smooth transition over a 5 dB window around color_killer_threshold_db
            ck_gain = float(np.clip((cn_db - (ck_thresh - 2.5)) / 5.0, 0.0, 1.0))

        # Detuning the LO by > 250 kHz also attenuates the colour burst on the IF slope
        tune_off_khz = float(state.get("tuning_offset_khz", 0.0))
        if abs(tune_off_khz) > 120.0:
            detune_atten = float(np.clip(1.0 - (abs(tune_off_khz) - 120.0) / 320.0, 0.15, 1.0))
            ck_gain *= detune_atten

        self._last_color_killer_gain = ck_gain
        self._last_multipath_chroma_rad = float(state.get("multipath_chroma_phase_rad", 0.0)) + (
            0.35 * (tune_off_khz / 200.0)
        )

        # --- 4. FM Intercarrier Sound C/N & Capture Effect Diagnostics ---
        # FM sound carrier is typically 10-13 dB below visual peak carrier, but FM demodulation
        # has a ~12 dB FM capture / quieting threshold above which audio hiss drops sharply!
        fm_sound_cn_db = cn_db - 4.0 - 0.015 * abs(tune_off_khz)
        fm_capture_quieting = float(1.0 / (1.0 + np.exp(-(fm_sound_cn_db - 18.0) / 2.2)))

        state.update({
            "noise_rms": round(noise_rms, 4),
            "afc_slip_px": round(slip_px, 3),
            "color_killer_gain": round(ck_gain, 3),
            "fm_sound_cn_db": round(fm_sound_cn_db, 2),
            "fm_capture_quieting": round(fm_capture_quieting, 3),
            "tv_dx_active": bool(int(round(float(self.p.propagation_mode))) in (1, 2)),
        })
        return out.astype(np.float32), state


class RFLink:
    """
    A complete RF modulator + propagation channel + tuner front-end + demodulator pair.

    Orchestrates the four modular `SignalNode` stages (`TransmitterNode`,
    `PropagationChannelNode`, `TunerFrontEndNode`, `DemodulatorNode`) while keeping
    the exact method signatures (`modulate`, `demodulate`, `chroma_phase_error`, `apply`)
    expected by `GraphExecutor`, `ConnectorStage`, and `tools/verify.py`.
    """

    def __init__(self, std: VideoStandard, params: RFParams | None = None,
                 seed: int = 20240101):
        self.std = std
        self.p = params or RFParams()
        self.seed = int(seed)
        self.tx_node = TransmitterNode(self.std, self.p)
        self.channel_node = PropagationChannelNode(self.std, self.p, seed=self.seed)
        self.frontend_node = TunerFrontEndNode(self.std, self.p)
        self.demod_node = DemodulatorNode(self.std, self.p, seed=self.seed)
        self.last_state: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset all stateful RF nodes for deterministic replay."""
        self.tx_node.reset()
        self.channel_node.reset()
        self.frontend_node.reset()
        self.demod_node.reset()
        self.last_state.clear()

    def noise_floor_dbm(self) -> float:
        return self.frontend_node.noise_floor_dbm()

    def carrier_to_noise_db(self, frame_index: int = 0, dt: float = 1.0 / 25.0) -> float:
        rx_dbm, _, _ = self.channel_node.effective_signal_dbm(frame_index, dt)
        return float(rx_dbm - self.frontend_node.noise_floor_dbm())

    def modulate(self, composite: np.ndarray, frame_index: int = 0) -> np.ndarray:
        """
        Put a composite baseband signal onto the RF carrier (`TransmitterNode`).
        """
        dt = 1.0 / max(1.0, float(getattr(self.std, "frames_per_second", 25.0)))
        state: dict[str, Any] = {"frame_index": int(frame_index)}
        shaped, state = self.tx_node.process(composite, state, dt)
        self.last_state.update(state)
        return shaped

    def demodulate(self, rf_signal: np.ndarray, frame_index: int = 0,
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
        if rng is None:
            rng = np.random.default_rng((int(self.seed) * 8191 + int(frame_index) * 131) & 0xFFFFFFFF)
        """
        Pass the modulated RF signal through `PropagationChannelNode`,
        `TunerFrontEndNode`, and `DemodulatorNode`.
        """
        dt = 1.0 / max(1.0, float(getattr(self.std, "frames_per_second", 25.0)))
        state: dict[str, Any] = {
            "frame_index": int(frame_index),
            "rng": rng,
            "effective_channel_mhz": self.last_state.get("effective_channel_mhz", self.p.channel_mhz),
        }
        sig_ch, state = self.channel_node.process(rf_signal, state, dt)
        sig_fe, state = self.frontend_node.process(sig_ch, state, dt)
        sig_demod, state = self.demod_node.process(sig_fe, state, dt)

        self.last_state.update(state)
        info = {
            "cn_db": state["cn_db"],
            "cnr_db": state["cn_db"],
            "noise_floor_dbm": state["noise_floor_dbm"],
            "received_signal_dbm": state["received_signal_dbm"],
            "path_delta_db": state["path_delta_db"],
            "qsb_offset_db": state["qsb_offset_db"],
            "noise_rms": state["noise_rms"],
            "afc_slip_px": state["afc_slip_px"],
            "color_killer_gain": state["color_killer_gain"],
            "multipath_delay_px": state["multipath_delay_px"],
            "multipath_shift_px": state["multipath_delay_px"],
            "fm_sound_cn_db": state["fm_sound_cn_db"],
            "fm_capture_quieting": state["fm_capture_quieting"],
            "propagation_mode": state["propagation_mode_name"],
            "tv_dx_active": state["tv_dx_active"],
        }
        return sig_demod.astype(np.float32), info

    def chroma_phase_error(self) -> np.ndarray | None:
        """
        Per-line subcarrier phase error (radians) from AFC sync slip + multipath/detuning.
        """
        afc_ph = self.demod_node._last_phase
        mp_ph = float(getattr(self.demod_node, "_last_multipath_chroma_rad", 0.0))
        if afc_ph is None and abs(mp_ph) < 1e-4:
            return None
        if afc_ph is not None:
            return (afc_ph + mp_ph).astype(np.float32)
        return np.array([mp_ph], dtype=np.float32)

    def color_killer_gain(self) -> float:
        """Gain (0..1) applied to recovered chroma by the tuner's color-killer circuit."""
        return float(getattr(self.demod_node, "_last_color_killer_gain", 1.0))

    def apply(self, composite: np.ndarray, frame_index: int = 0,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
        if rng is None:
            rng = np.random.default_rng((int(self.seed) * 8191 + int(frame_index) * 131) & 0xFFFFFFFF)
        """Full modulate -> channel -> front-end -> demodulate round trip."""
        tx = self.modulate(composite, frame_index)
        return self.demodulate(tx, frame_index, rng)

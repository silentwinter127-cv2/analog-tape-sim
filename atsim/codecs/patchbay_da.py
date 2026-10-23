"""
Broadcast Patchbay, Distribution Amplifier (DA), and 75-Ohm Coaxial Termination Physics (`atsim.codecs.patchbay_da`).

Models the physical electrical network of a studio / home-deck coaxial interconnection and routing bay:

1. **75-Ohm Source-Termination Voltage Divider & Transmission-Line Reflection (`da_termination_mode`, `da_load_count`)**:
   Every standard composite/component video output stage consists of a $2\,\text{V}_{\text{pp}}$ open-circuit voltage
   source in series with a precision $R_s = 75\,\Omega$ back-termination resistor, designed to drive a $Z_0 = 75\,\Omega$
   coaxial transmission line terminated in $R_L = 75\,\Omega$:
     * **Matched $75\,\Omega$ (`termination_mode = 0`, $R_L = 75\,\Omega$)**:
       Voltage divider $H_V = \frac{2 R_L}{R_s + R_L} = 1.000$ (`0.00 dB`, `100 IRE` white, `-40 IRE` sync tip)
       with reflection coefficient $\Gamma_L = \frac{R_L - Z_0}{R_L + Z_0} = 0.0$.
     * **Unterminated High-Z Loop-Through (`termination_mode = 1`, $R_L \approx 10\,\text{k}\Omega$)**:
       When a PVM monitor's `75Ω EXT TERM` switch is left OFF (High-Z loop-through open), the voltage divider becomes
       $H_V = \frac{2 \times 10000}{75 + 10000} \approx 1.9851$ (`+5.96 dB` overdrive), slamming video levels into the
       receiver ADC / video-amp white-clip ceiling (`> 100 IRE`) and sending a strong positive reflection
       ($\Gamma_L \approx +0.9851$) back along the coaxial line of length $L_{\text{m}}$ (`da_cable_len_m`).
     * **Passive BNC T-Adapter Double / Multi-Termination (`termination_mode = 2`, $N = \text{load\_count} \in [2..6]$)**:
       Splitting one composite output across $N$ $75\,\Omega$-terminated monitors/decks using passive BNC T-pieces
       without an active Distribution Amplifier places $N$ loads in parallel ($R_L = 75\,\Omega / N$):
       - $N=2$ ($37.5\,\Omega$): $H_V = 0.6667$ (`-3.52 dB`, sync tip crushed to `-26.7 IRE`, $\Gamma_L = -0.3333$);
       - $N=3$ ($25.0\,\Omega$): $H_V = 0.5000$ (`-6.02 dB`, sync tip crushed to `-20.0 IRE`, $\Gamma_L = -0.5000$);
       - $N=4$ ($18.75\,\Omega$): $H_V = 0.4000$ (`-7.96 dB`, sync tip crushed to `-16.0 IRE`, $\Gamma_L = -0.6000$).
     * **Active Distribution Amplifier (`termination_mode = 3`, `Active 1xN DA`)**:
       Terminates the incoming coax in a precision $75\,\Omega$ load ($\Gamma_L = 0$) and buffers each of the $N$ outputs
       with an isolated $75\,\Omega$ op-amp driver (`H_V = 1.0`, plus adjustable `da_gain_db` trim and `da_eq_boost`
       high-frequency cable equalizer peaking to compensate $\sqrt{f}$ coaxial skin-effect loss).

2. **Broadcast Patchbay Crosspoint Crosstalk, Normalled-Jack Contact Aging & Format Patch Mismatch**:
   * **Adjacent-Channel Capacitive Crosstalk (`patch_crosstalk`)**:
     Stray capacitance ($C_c \approx 2\text{--}8\,\text{pF}$) between adjacent patchbay busbars couples the derivative
     $C_c \frac{dV_{\text{adj}}}{dt}$ of an unsynchronized studio bus (`bus_b_yiq` or synthetic adjacent feed) into the
     active signal path.
   * **Normalled-Jack Contact Oxidation & Micro-Arcing (`patch_contact_aging`)**:
     Tarnished normalled leaf springs introduce series contact attenuation, intermittent horizontal contact scratch spikes,
     and burst phase jitter.
   * **Patchbay Format Patching Mismatch (`patch_format_mismatch`)**:
     - `0`: Normal Composite
     - `1`: S-Video Y-only pin patched into Composite BNC (no colour burst / chroma -> pure monochrome)
     - `2`: Component YPbPr patched into Composite BNC (G-Y/Pb matrix tint & tri-level sync timing offset)
     - `3`: Uncompressed Digital SDI bitstream patched into Analog Composite BNC (high-frequency LFSR digital snow)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode, horizontal_shift, lowpass
from ..core.standards import VideoStandard, get_standard


@dataclass
class PatchbayDASpec:
    enabled: bool = False
    termination_mode: int = 0       # 0=75Ω Matched, 1=High-Z Open (~10kΩ), 2=Passive T-Split (75Ω/N), 3=Active 1xN DA
    load_count: int = 1             # Number of parallel 75Ω loads (1..6)
    cable_len_m: float = 15.0       # Coaxial cable length in meters (1..150 m)
    da_gain_db: float = 0.0         # Active DA output gain trim (-6..+6 dB)
    da_eq_boost: float = 0.0        # Active DA high-frequency cable EQ boost (0..1.5)
    patch_crosstalk: float = 0.0    # Crosspoint capacitive crosstalk (0..1)
    patch_contact_aging: float = 0.0  # Normalled jack contact oxidation & scratch noise (0..1)
    patch_format_mismatch: int = 0  # 0=Normal, 1=S-Video Y->Comp, 2=YPbPr->Comp, 3=SDI->Comp

    @classmethod
    def from_params(cls, cfg: dict | None) -> "PatchbayDASpec":
        c = cfg or {}
        en = float(c.get("da_enabled", 0.0)) > 0.5
        t_mode = int(np.clip(round(float(c.get("da_termination_mode", c.get("termination_mode", 0.0)))), 0, 3))
        l_cnt = int(np.clip(round(float(c.get("da_load_count", c.get("split_load_count", 2.0 if t_mode == 2 else 1.0)))), 1, 6))
        c_len = float(np.clip(float(c.get("da_cable_len_m", c.get("cable_length_m", 15.0))), 0.5, 150.0))
        g_db = float(np.clip(float(c.get("da_gain_db", 0.0)), -6.0, 6.0))
        eq_b = float(np.clip(float(c.get("da_eq_boost", 0.0)), 0.0, 1.5))
        xt = float(np.clip(float(c.get("patch_crosstalk", 0.0)), 0.0, 1.0))
        ag = float(np.clip(float(c.get("patch_contact_aging", 0.0)), 0.0, 1.0))
        fm = int(np.clip(round(float(c.get("patch_format_mismatch", 0.0))), 0, 3))
        return cls(
            enabled=en,
            termination_mode=t_mode,
            load_count=l_cnt,
            cable_len_m=c_len,
            da_gain_db=g_db,
            da_eq_boost=eq_b,
            patch_crosstalk=xt,
            patch_contact_aging=ag,
            patch_format_mismatch=fm,
        )

    def is_active(self) -> bool:
        return bool(
            self.enabled
            or self.termination_mode != 0
            or self.patch_crosstalk > 1e-4
            or self.patch_contact_aging > 1e-4
            or self.patch_format_mismatch != 0
        )


def compute_termination_physics(spec: PatchbayDASpec) -> dict:
    """
    Compute exact 75-Ohm source/load voltage divider, reflection coefficient, round-trip cable delay,
    sync-tip amplitude in IRE, and induced sync-clip factor for downstream H-AFC / V-Hold.
    """
    z0 = 75.0
    r_source = 75.0
    mode = int(np.clip(spec.termination_mode, 0, 3))
    n_loads = int(np.clip(spec.load_count, 1, 6))
    length_m = float(np.clip(spec.cable_len_m, 0.5, 150.0))

    if mode == 1:
        # Unterminated High-Z loop-through open (~10 kOhm input impedance of monitor with 75Ω switch OFF)
        r_load = 10000.0
        voltage_gain = (2.0 * r_load) / (r_source + r_load)
        gamma_load = (r_load - z0) / (r_load + z0)
        mode_name = "high_z_open"
    elif mode == 2:
        # Passive BNC T-Adapter splitting across N parallel 75-ohm loads without an active DA
        eff_n = max(2, n_loads)
        r_load = z0 / float(eff_n)
        voltage_gain = (2.0 * r_load) / (r_source + r_load)
        gamma_load = (r_load - z0) / (r_load + z0)
        mode_name = f"passive_t_split_{eff_n}x"
    elif mode == 3:
        # Active 1xN Distribution Amplifier: isolated 75-ohm input termination + N independent 75-ohm drivers
        r_load = 75.0
        voltage_gain = float(10.0 ** (float(spec.da_gain_db) / 20.0))
        gamma_load = 0.0
        mode_name = f"active_da_1x{max(1, n_loads)}"
    else:
        # Standard 75-ohm matched termination
        r_load = 75.0
        voltage_gain = 1.0
        gamma_load = 0.0
        mode_name = "matched_75ohm"

    gain_db = float(20.0 * np.log10(max(1e-4, voltage_gain)))
    # Coaxial propagation velocity v ~ 0.66c (~ 198 m/us) -> round-trip delay tau = 2*L / v (~ 10.1 ns/m)
    round_trip_ns = 10.10 * length_m
    # Coaxial skin-effect loss at 4.0 MHz (RG-59/8281 coax: ~0.055 dB/m at 4 MHz)
    coax_loss_4mhz_db = 0.055 * length_m
    eq_boost_db = (float(spec.da_eq_boost) * 4.2) if mode == 3 else (float(spec.da_eq_boost) * 1.5)
    net_hf_db = eq_boost_db - coax_loss_4mhz_db

    # Nominal sync tip is -40.0 IRE; passive T-split crushes it proportionally toward 0 IRE
    sync_tip_ire = -40.0 * voltage_gain
    # SyncClockStage scales sync tip by (1 - 0.68 * sync_clip); compute equivalent induced_sync_clip
    induced_sync_clip = float(np.clip((1.0 - voltage_gain) / 0.68, 0.0, 0.95)) if voltage_gain < 0.95 else 0.0

    return {
        "mode": mode_name,
        "termination_mode": mode,
        "load_count": n_loads,
        "r_load_ohm": round(float(r_load), 2),
        "load_impedance_ohm": round(float(r_load), 2),
        "voltage_gain": round(float(voltage_gain), 4),
        "gain_db": round(gain_db, 2),
        "reflection_gamma": round(float(gamma_load), 4),
        "cable_len_m": round(length_m, 1),
        "round_trip_ns": round(round_trip_ns, 1),
        "coax_loss_4mhz_db": round(coax_loss_4mhz_db, 2),
        "eq_boost_db": round(eq_boost_db, 2),
        "net_hf_db": round(net_hf_db, 2),
        "sync_tip_ire": round(sync_tip_ire, 2),
        "induced_sync_clip": round(induced_sync_clip, 4),
    }


def apply_patchbay_da(
    y: np.ndarray,
    i: np.ndarray,
    q: np.ndarray,
    spec: PatchbayDASpec,
    frame_index: int = 0,
    fps: float = 29.97,
    std: VideoStandard | None = None,
    seed: int = 20240101,
    bus_b_yiq: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Apply 75-Ohm termination divider & reflection, coaxial skin-effect loss & Active DA Cable EQ,
    patchbay crosspoint capacitive crosstalk, normalled-jack contact oxidation, and format mismatch
    in the YIQ / composite domain.
    """
    if std is None:
        std = get_standard("ntsc")
    phys = compute_termination_physics(spec)
    y_out = np.asarray(y, dtype=np.float32).copy()
    i_out = np.asarray(i, dtype=np.float32).copy()
    q_out = np.asarray(q, dtype=np.float32).copy()

    if not spec.is_active():
        phys["clipped_white_pct"] = 0.0
        phys["crosstalk_rms_ire"] = 0.0
        phys["format_state"] = "bypass"
        return y_out, i_out, q_out, phys

    h, w = y_out.shape[:2]
    scale_w = max(0.25, w / 720.0)
    rng = np.random.default_rng((int(seed) * 1664525 + int(frame_index) * 1013904223 + 73) & 0xFFFFFFFF)

    # ------------------------------------------------------------------
    # 1. Patchbay Format Patching Mismatch (`patch_format_mismatch`)
    # ------------------------------------------------------------------
    fmt_mis = int(spec.patch_format_mismatch)
    if fmt_mis == 1:
        # S-Video Y-only pin patched into Composite BNC input:
        # No colour burst or 3.58/4.43 MHz carrier on the Y pin -> receiver colour killer mutes chroma to 0
        i_out = np.zeros_like(i_out)
        q_out = np.zeros_like(q_out)
        phys["format_state"] = "svideo_y_only_monochrome"
    elif fmt_mis == 2:
        # Component YPbPr patched into Composite BNC input:
        # Green/Magenta un-matrixed chrominance bias + tri-level sync horizontal timing offset
        comp_shift = np.full(h, 4.5 * scale_w, dtype=np.float32) + 0.8 * np.sin(np.arange(h, dtype=np.float32) * 0.15)
        y_out = horizontal_shift(y_out, comp_shift) * 0.92
        i_mis = -0.24 * y_out + 0.65 * i_out
        q_mis = 0.28 * y_out - 0.55 * q_out
        i_out, q_out = i_mis.astype(np.float32), q_mis.astype(np.float32)
        phys["format_state"] = "ypbpr_in_composite_mismatch"
    elif fmt_mis == 3:
        # Uncompressed Digital SDI (270 Mbps NRZI LFSR stream) patched into Analog Composite BNC input:
        # High-frequency digital hash snow + diagonal SDI TRS (Timing Reference Signal) blanking bars
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sdi_snow = rng.standard_normal((h, w)).astype(np.float32)
        sdi_bar = 0.5 + 0.5 * np.cos(2.0 * np.pi * (yy / max(6.0, h * 0.20) + xx / max(12.0, w * 0.45) - frame_index * 0.21))
        y_out = np.clip(0.42 + 0.32 * sdi_snow + 0.18 * sdi_bar + 0.12 * y_out, 0.0, 1.0).astype(np.float32)
        i_out = (rng.standard_normal((h, w)).astype(np.float32) * 0.18).astype(np.float32)
        q_out = (rng.standard_normal((h, w)).astype(np.float32) * 0.18).astype(np.float32)
        phys["format_state"] = "sdi_digital_in_analog_composite"
    else:
        phys["format_state"] = "normal_composite"

    # ------------------------------------------------------------------
    # 2. Coaxial Cable Skin-Effect Loss & Active DA High-Frequency Cable EQ
    # ------------------------------------------------------------------
    net_hf_db = float(phys["net_hf_db"])
    if net_hf_db < -0.15 and fmt_mis != 3:
        atten_db = -net_hf_db
        sigma_coax = max(0.35, min(6.0, 0.24 * atten_db * scale_w))
        y_low = lowpass(y_out, sigma_coax, axis=1)
        hf_keep = float(np.clip(10.0 ** (net_hf_db / 20.0), 0.22, 1.0))
        y_out = y_low + hf_keep * (y_out - y_low)
        chroma_atten = float(np.clip(10.0 ** (net_hf_db / 24.0), 0.20, 1.0))
        i_out = lowpass(i_out, max(0.4, 0.45 * sigma_coax), axis=1) * chroma_atten
        q_out = lowpass(q_out, max(0.4, 0.45 * sigma_coax), axis=1) * chroma_atten
    elif net_hf_db > 0.15 and fmt_mis != 3:
        sigma_eq = max(0.55, 1.25 * scale_w)
        y_detail = y_out - lowpass(y_out, sigma_eq, axis=1)
        boost_gain = float(np.clip(10.0 ** (net_hf_db / 20.0) - 1.0, 0.0, 2.0))
        y_out = y_out + boost_gain * y_detail
        if net_hf_db > 2.0:
            ring_shift = max(1, int(round(2.0 * scale_w)))
            ring = np.roll(y_detail, ring_shift, axis=1)
            ring[:, :ring_shift] = 0.0
            y_out = y_out - (0.20 * (net_hf_db - 1.5) / 6.0) * ring
        chroma_boost = float(np.clip(1.0 + 0.35 * boost_gain, 1.0, 1.55))
        i_out = i_out * chroma_boost
        q_out = q_out * chroma_boost

    # ------------------------------------------------------------------
    # 3. 75-Ohm Voltage Divider (Matched vs High-Z +6dB vs Passive T-Split) & Reflection Ghost
    # ------------------------------------------------------------------
    v_gain = float(phys["voltage_gain"])
    gamma = float(phys["reflection_gamma"])
    clipped_white_pct = 0.0

    if abs(v_gain - 1.0) > 1e-3:
        y_scaled = y_out * v_gain
        if v_gain > 1.05:
            # High-Z Unterminated Loop-Through (+5.96 dB, ~1.985x voltage):
            # Overdrives video into the receiver ADC white-clip ceiling (> 1.0)
            clipped_white_pct = float(np.mean(y_scaled > 1.0) * 100.0)
            y_out = np.where(y_scaled <= 0.90, y_scaled, 0.90 + 0.10 * np.tanh((y_scaled - 0.90) / 0.15)).astype(np.float32)
            white_sat_gate = np.clip(1.0 - np.clip((y_scaled - 0.92) / 0.40, 0.0, 0.85), 0.15, 1.0)
            i_out = (i_out * min(1.75, v_gain) * white_sat_gate).astype(np.float32)
            q_out = (q_out * min(1.75, v_gain) * white_sat_gate).astype(np.float32)
        else:
            # Passive BNC T-Split (37.5Ω / 25Ω / 18.75Ω): attenuates both luma and chroma
            y_out = y_scaled.astype(np.float32)
            i_out = (i_out * v_gain).astype(np.float32)
            q_out = (q_out * v_gain).astype(np.float32)

    if abs(gamma) > 0.02:
        delay_px = max(2, int(round((1.8 + 0.18 * float(spec.cable_len_m)) * scale_w)))
        if delay_px < w - 2:
            y_echo = np.roll(y_out, delay_px, axis=1)
            y_echo[:, :delay_px] = y_out[:, :delay_px]
            i_echo = np.roll(i_out, delay_px, axis=1)
            i_echo[:, :delay_px] = i_out[:, :delay_px]
            q_echo = np.roll(q_out, delay_px, axis=1)
            q_echo[:, :delay_px] = q_out[:, :delay_px]

            rt_atten = float(np.exp(-0.010 * float(spec.cable_len_m)))
            echo_amp = float(gamma * 0.26 * rt_atten)
            y_out = np.clip(y_out + echo_amp * (y_echo - float(y_out.mean()) * 0.35), 0.0, 1.08).astype(np.float32)
            theta_echo = float((2.0 * np.pi * std.subcarrier_hz * (phys["round_trip_ns"] * 1e-9)) % (2.0 * np.pi))
            c_e, s_e = float(np.cos(theta_echo)), float(np.sin(theta_echo))
            i_out = (i_out + echo_amp * (i_echo * c_e - q_echo * s_e)).astype(np.float32)
            q_out = (q_out + echo_amp * (i_echo * s_e + q_echo * c_e)).astype(np.float32)
            phys["reflection_delay_px"] = int(delay_px)

    # ------------------------------------------------------------------
    # 4. Patchbay Crosspoint Capacitive Crosstalk (`patch_crosstalk`)
    # ------------------------------------------------------------------
    xtalk = float(spec.patch_crosstalk)
    crosstalk_rms_ire = 0.0
    if xtalk > 1e-4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        drift_x = int(round((w * 0.34 + frame_index * 9.0 * scale_w) % max(1, w)))
        drift_y = int(round((h * 0.26 + frame_index * 3.0) % max(1, h)))
        if bus_b_yiq is not None and len(bus_b_yiq) == 3 and bus_b_yiq[0].shape == y_out.shape:
            adj_y, adj_i, adj_q = bus_b_yiq[0], bus_b_yiq[1], bus_b_yiq[2]
        else:
            adj_y = np.roll(np.roll(y_out[:, ::-1], drift_y, axis=0), drift_x, axis=1)
            adj_i = np.roll(np.roll(i_out[:, ::-1], drift_y, axis=0), drift_x, axis=1)
            adj_q = np.roll(np.roll(q_out[:, ::-1], drift_y, axis=0), drift_x, axis=1)

        # Capacitive coupling C_c * dV/dt emphasizes sharp horizontal edges + drifting HBI sync bar
        adj_dy = np.gradient(adj_y, axis=1).astype(np.float32) * (2.2 / max(0.5, scale_w))
        sync_col = (xx - drift_x) % max(1.0, float(w))
        adj_hbi = (
            np.exp(-0.5 * ((sync_col - 14.0 * scale_w) / max(1.2, 4.0 * scale_w)) ** 2) * (-0.38)
            + np.exp(-0.5 * ((sync_col - 21.0 * scale_w) / max(1.0, 2.2 * scale_w)) ** 2) * (+0.46)
        )
        beat_wave = np.cos(2.0 * np.pi * (xx / max(2.2, 4.8 * scale_w) + yy / 7.0) + frame_index * 0.85)
        xt_delta = xtalk * (0.32 * adj_dy + 0.14 * adj_y + 0.24 * adj_hbi + 0.08 * beat_wave).astype(np.float32)
        crosstalk_rms_ire = float(np.sqrt(np.mean(xt_delta ** 2)) * 100.0)
        y_out = (y_out + xt_delta).astype(np.float32)
        i_out = (i_out + xtalk * (0.22 * adj_i + 0.08 * beat_wave)).astype(np.float32)
        q_out = (q_out + xtalk * (0.22 * adj_q - 0.08 * beat_wave)).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Patchbay Normalled-Jack Contact Oxidation & Scratch (`patch_contact_aging`)
    # ------------------------------------------------------------------
    aging = float(spec.patch_contact_aging)
    if aging > 1e-4:
        atten_c = 1.0 - 0.10 * aging
        y_out = y_out * atten_c
        i_out = i_out * atten_c
        q_out = q_out * atten_c

        n_scratches = int(rng.poisson(aging * 5.0 + 0.5))
        for _ in range(n_scratches):
            r_idx = int(rng.integers(0, max(1, h)))
            span = int(np.clip(rng.integers(1, 3), 1, max(1, h - r_idx)))
            scratch_val = float(rng.uniform(-0.20, 0.26) * aging)
            y_out[r_idx:r_idx + span, :] = (
                y_out[r_idx:r_idx + span, :]
                + scratch_val
                + rng.standard_normal((span, w)).astype(np.float32) * (0.08 * aging)
            )
            phi_s = float(rng.uniform(-0.8, 0.8) * aging)
            cs, sn = float(np.cos(phi_s)), float(np.sin(phi_s))
            ir = i_out[r_idx:r_idx + span, :].copy()
            qr = q_out[r_idx:r_idx + span, :].copy()
            i_out[r_idx:r_idx + span, :] = ir * cs - qr * sn
            q_out[r_idx:r_idx + span, :] = ir * sn + qr * cs

        rows_norm = np.arange(h, dtype=np.float32)[:, None] / max(1.0, float(h))
        hum = 0.04 * aging * np.sin(2.0 * np.pi * (rows_norm + frame_index * 0.09))
        y_out = (y_out + hum).astype(np.float32)

    y_out = np.clip(y_out, 0.0, 1.05).astype(np.float32)
    phys["clipped_white_pct"] = round(clipped_white_pct, 2)
    phys["crosstalk_rms_ire"] = round(crosstalk_rms_ire, 2)
    return y_out, i_out, q_out, phys




def compute_patchbay_audio_transient(
    spec: PatchbayDASpec,
    t_samples: np.ndarray,
    seed: int = 20240101,
) -> dict:
    """
    Compute audio-band transients coupled from the broadcast patchbay & 75-Ohm distribution system:
      - `crackle`: intermittent normalled-jack contact oxidation micro-arcing & scratch noise (`patch_contact_aging`),
        plus digital hash burst if `patch_format_mismatch == 3` (SDI bitstream in analog jack);
      - `crosstalk_tone`: capacitive crosspoint bleed (`patch_crosstalk`) from adjacent studio timecode/reference tone
        (1 kHz reference tone + 15.734 kHz horizontal sync buzz derivative);
      - `hum_add`: shield-ground contact resistance hum contribution (`patch_contact_aging` + passive T-split ground loop).
    """
    t = np.asarray(t_samples, dtype=np.float64)
    n = t.size
    crackle = np.zeros(n, dtype=np.float32)
    xt_tone = np.zeros(n, dtype=np.float32)
    hum_add = 0.0

    if not spec.is_active() or n == 0:
        return {"crackle": crackle, "crosstalk_tone": xt_tone, "hum_add": hum_add}

    frame_bucket = int(round(float(t[0]) * 30.0)) if n > 0 else 0
    rng = np.random.default_rng((int(seed) * 2654435761 + frame_bucket * 40503 + 101) & 0xFFFFFFFF)

    aging = float(spec.patch_contact_aging)
    if aging > 1e-4:
        # Sparse contact scratch impulses + band-limited crackle burst
        impulses = (rng.random(n) < (0.018 * aging)).astype(np.float32)
        burst_env = np.sin(2.0 * np.pi * 7.0 * t + rng.uniform(0, 6.28)) > (0.55 - 0.35 * aging)
        crackle = (
            impulses * rng.standard_normal(n).astype(np.float32) * (0.14 * aging)
            + burst_env.astype(np.float32) * rng.standard_normal(n).astype(np.float32) * (0.045 * aging)
        ).astype(np.float32)
        hum_add += 0.22 * aging

    if int(spec.patch_format_mismatch) == 3:
        # SDI digital bitstream patched into analog port -> dense white digital hash
        crackle = (crackle + rng.standard_normal(n).astype(np.float32) * 0.22).astype(np.float32)

    xtalk = float(spec.patch_crosstalk)
    if xtalk > 1e-4:
        # Capacitive coupling of adjacent channel 1 kHz studio reference tone + 2.4 kHz LTC bi-phase timecode chatter
        ltc_mod = np.sign(np.sin(2.0 * np.pi * 1200.0 * t + 0.7)).astype(np.float32)
        tone_1k = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        xt_tone = (xtalk * (0.045 * tone_1k + 0.032 * ltc_mod)).astype(np.float32)
        hum_add += 0.12 * xtalk

    if int(spec.termination_mode) == 2 and int(spec.load_count) >= 2:
        # Multi-load passive BNC T-split creates shared shield ground loop between multiple monitors
        hum_add += 0.08 * min(4, int(spec.load_count))

    return {
        "crackle": crackle,
        "crosstalk_tone": xt_tone,
        "hum_add": float(hum_add),
    }


# ===========================================================================
# Dedicated Node-Graph Stages (`dist_amp` and `patch_router`)
# ===========================================================================

class DistAmpStage(Stage):
    """
    Standalone Node-Graph Stage for `dist_amp` (1xN Active Video Distribution Amplifier & 75-Ohm Load Box).
    """
    name = "dist_amp"
    layers = ("connector",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = dict(self.cfg_of("connector") or {})
        if "da_enabled" not in c or float(c.get("da_enabled", 0.0)) <= 0.5:
            c["da_enabled"] = 1.0
        if "da_termination_mode" not in c:
            c["da_termination_mode"] = 3.0
        self.spec = PatchbayDASpec.from_params(c)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        if sig is None:
            sig = SignalFrame.from_rgb(np.clip(buf.rgb, 0.0, 1.0), buf.std)
        y_da, i_da, q_da, da_info = apply_patchbay_da(
            sig.y, sig.i, sig.q, self.spec,
            frame_index=int(buf.index), fps=self.ctx.fps,
            std=buf.std, seed=int(self.ctx.seed) + 41,
        )
        new_sig = SignalFrame(y_da, i_da, q_da)
        rgb_out = np.clip(new_sig.to_rgb(), 0.0, 1.0).astype(np.float32)
        comp = composite_encode(new_sig.y, new_sig.i, new_sig.q, buf.std, chroma_gain=0.63)
        meta = dict(buf.meta)
        meta["signal"] = new_sig
        meta["patchbay_da"] = da_info
        return FrameBuffer(rgb_out, comp, buf.index, buf.time, buf.std, meta)


class PatchRouterStage(Stage):
    """
    Standalone Node-Graph Stage for `patch_router` (Broadcast Video Patchbay & Crosspoint Router).
    """
    name = "patch_router"
    layers = ("connector",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = dict(self.cfg_of("connector") or {})
        c["da_enabled"] = 1.0
        self.spec = PatchbayDASpec.from_params(c)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        if sig is None:
            sig = SignalFrame.from_rgb(np.clip(buf.rgb, 0.0, 1.0), buf.std)
        bus_b_yiq = buf.meta.get("bus_b_yiq")
        y_da, i_da, q_da, da_info = apply_patchbay_da(
            sig.y, sig.i, sig.q, self.spec,
            frame_index=int(buf.index), fps=self.ctx.fps,
            std=buf.std, seed=int(self.ctx.seed) + 59,
            bus_b_yiq=bus_b_yiq,
        )
        new_sig = SignalFrame(y_da, i_da, q_da)
        rgb_out = np.clip(new_sig.to_rgb(), 0.0, 1.0).astype(np.float32)
        comp = composite_encode(new_sig.y, new_sig.i, new_sig.q, buf.std, chroma_gain=0.63)
        meta = dict(buf.meta)
        meta["signal"] = new_sig
        meta["patchbay_da"] = da_info
        return FrameBuffer(rgb_out, comp, buf.index, buf.time, buf.std, meta)

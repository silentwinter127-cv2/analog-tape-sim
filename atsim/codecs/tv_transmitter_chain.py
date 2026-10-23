"""
Candidate N: Television Broadcast Transmitter Exciter/HPA Chain & End-to-End Off-Air Test Bench (`tv_transmitter_chain`).

Implements Section 4 of the Broadcast Infrastructure Specification (`README.txt`):
  1. **Calibratable Broadcast Transmitter Subsystem**:
     - **Frequency Reference & Exciter (`tx_chain_mode`)**:
       * Analog Vestigial-Sideband (VSB) Exciter (`mode = 1`): Group-delay pre-correction, negative
         modulation mapping, and white-clipper protection (maintains $\\ge 10\\%$ residual visual carrier
         on 100 IRE peak white so intercarrier FM sound never buzzes out).
       * Digital Terrestrial Television (DTT COFDM / 8-VSB) Exciter (`mode = 2`): Coded OFDM / 8-VSB
         constellation modulation with Modulation Error Ratio (`MER`, dB) governed by HPA linearity
         and local oscillator phase noise.
     - **High-Power Amplifier (`HPA` — Tetrode Tube / LDMOS Solid-State, `tx_pa_drive`, `tx_pa_am_pm_deg`)**:
       * **AM-AM Nonlinearity (Differential Gain `DG`)**: S-curve saturation compresses peak white / sync
         and modulates colour subcarrier amplitude as luminance rises (`diff_gain_pct`, %).
       * **AM-PM Phase Conversion (Differential Phase `DP`)**: Varactor-like output capacitance of high-power
         tubes/transistors shifts RF carrier & colour subcarrier phase as a function of instantaneous
         luminance (`diff_phase_deg`, degrees), causing vivid hue rotation on bright highlights!
       * **Vision-Chroma-Sound Intermodulation Beat (`tx_sound_intermod`)**: Nonlinear HPA mixing between
         the visual carrier, colour subcarrier ($3.58 / 4.43\\text{ MHz}$), and FM sound carrier
         ($4.5 / 6.0 / 6.5\\text{ MHz}$) produces the classic $920\\text{ kHz}$ (NTSC) / $1.57\\text{ MHz}$ (PAL-I)
         herringbone beat in bright saturated areas.
     - **Output Cavity Diplexer & Tower Feeder Line VSWR (`tx_filter_gd_ns`, `tx_feeder_vswr`)**:
       * High-power vestigial-sideband & sound/vision cavity diplexer introduces chrominance-luminance
         group delay (`tx_filter_gd_ns`, ns) and $2T$ ringing.
       * Tower coaxial feeder mismatch (`tx_feeder_vswr > 1.0`, reflection $\\Gamma = (\\text{VSWR}-1)/(\\text{VSWR}+1)$)
         reflects a delayed echo along the mast before radiation!
  2. **End-to-End Off-Air Broadcast Test Bench (`tx_chain_mode = 3`)**:
     - `编辑机 → 播出输出 → 发射链模型 → 传播信道 → 电视接收机 → CRT` vs `直通基带参考` in a 2×2 quad view,
       isolating Transmitter Nonlinearity (DG/DP/VSWR) from Propagation Channel Noise/Multipath.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode, horizontal_shift, lowpass, rgb_to_yiq, yiq_to_rgb
from .master_playout import _draw_text_5x7, _resample_nn
from .satellite_ird import SatelliteIRDSpec, simulate_digital_dvbs_qpsk

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

TX_CHAIN_MODES: dict[int, str] = {
    0: "off",
    1: "analog_vsb_transmitter",
    2: "digital_dtt_transmitter",
    3: "end_to_end_broadcast_lab",
    4: "corner_1_studio_baseband_ref",
    5: "corner_2_tx_hpa_probe",
    6: "corner_3_analog_offair_rx",
    7: "corner_4_digital_dtt_rx",
}


@dataclass
class TVTransmitterSpec:
    """Configuration for Candidate N (`rf` layer / `tv_exciter_pa` node)."""
    mode: int = 0                    # 0=Off, 1=Analog VSB, 2=Digital DTT, 3=2x2 Lab, 4=Corner1(TL), 5=Corner2(TR), 6=Corner3(BL), 7=Corner4(BR)
    pa_drive: float = 0.35           # High-Power Amplifier drive level (0..1.5) -> AM-AM Differential Gain
    pa_am_pm_deg: float = 8.0        # HPA AM-PM phase conversion at 100 IRE (0..35 deg) -> Differential Phase
    filter_gd_ns: float = 65.0       # Cavity diplexer & VSB filter Y/C group delay (0..350 ns)
    feeder_vswr: float = 1.15        # Antenna feeder Voltage Standing Wave Ratio (1.0..3.0)
    sound_intermod: float = 0.25     # Vision-Chroma-Sound HPA IM3 beat (0..1.0)
    quad_corner: int = 0             # 0=2x2 Quad Split, 1=Top-Left(Ref), 2=Top-Right(TX Probe), 3=Bottom-Left(Analog RX), 4=Bottom-Right(Digital DTT RX)
    show_border: int = 1             # 0=No Border (Seamless), 1=Thin Crosshair Border (Default), 2=Framed Studio Bezel
    show_text: int = 0               # 0=No Text (Clean Picture, Default), 1=Corner Labels Only, 2=Corner Labels + Bottom Telemetry Bar

    @classmethod
    def from_params(cls, cfg: dict) -> "TVTransmitterSpec":
        c = cfg or {}
        raw_mode = int(np.clip(round(float(c.get("tx_chain_mode", 0.0))), 0, 7))
        q_corner = int(np.clip(round(float(c.get("tx_quad_corner", 0.0))), 0, 4))
        if raw_mode in (4, 5, 6, 7) and q_corner == 0:
            q_corner = raw_mode - 3
        return cls(
            mode=raw_mode,
            pa_drive=float(np.clip(float(c.get("tx_pa_drive", 0.35)), 0.0, 1.5)),
            pa_am_pm_deg=float(np.clip(float(c.get("tx_pa_am_pm_deg", 8.0)), 0.0, 45.0)),
            filter_gd_ns=float(np.clip(float(c.get("tx_filter_gd_ns", 65.0)), 0.0, 400.0)),
            feeder_vswr=float(np.clip(float(c.get("tx_feeder_vswr", 1.15)), 1.0, 3.5)),
            sound_intermod=float(np.clip(float(c.get("tx_sound_intermod", 0.25)), 0.0, 1.0)),
            quad_corner=q_corner,
            show_border=int(np.clip(round(float(c.get("tx_show_border", 1.0))), 0, 2)),
            show_text=int(np.clip(round(float(c.get("tx_show_text", 0.0))), 0, 2)),
        )

    def is_active(self) -> bool:
        return self.mode > 0


def evaluate_transmitter_telemetry(spec: TVTransmitterSpec) -> dict[str, Any]:
    """
    Compute engineering telemetry for the broadcast transmitter chain:
      - Feeder reflection coefficient $\\Gamma = (\\text{VSWR} - 1) / (\\text{VSWR} + 1)$ and Return Loss (dB).
      - Differential Gain (`diff_gain_pct`, %) from HPA AM-AM saturation slope at peak white.
      - Differential Phase (`diff_phase_deg`, deg) from HPA AM-PM conversion.
      - Digital Transmitter MER (`mer_db`, dB) from HPA intermodulation, AM-PM phase error, and feeder echo.
    """
    vswr = max(1.0, float(spec.feeder_vswr))
    gamma = (vswr - 1.0) / (vswr + 1.0)
    return_loss_db = -20.0 * np.log10(max(1e-4, gamma))

    drive = float(spec.pa_drive)
    # Differential gain (% change in subcarrier amplitude from blanking to 100 IRE white)
    diff_gain_pct = float(np.clip((drive ** 1.55) * 28.5, 0.0, 48.0))
    # Differential phase (degrees of subcarrier phase rotation from blanking to 100 IRE white)
    diff_phase_deg = float(spec.pa_am_pm_deg) * (0.45 + 0.85 * drive)

    # Digital DTT Modulation Error Ratio (MER, dB): ideal exciter ~38 dB, degraded by HPA backoff, AM-PM & VSWR
    mer_penalty = 11.5 * (drive ** 1.6) + 0.22 * diff_phase_deg + 18.0 * gamma + 0.015 * float(spec.filter_gd_ns)
    mer_db = float(np.clip(38.0 - mer_penalty, 8.0, 38.0))

    return {
        "tx_mode": TX_CHAIN_MODES.get(spec.mode, "off"),
        "pa_drive": round(drive, 3),
        "diff_gain_pct": round(diff_gain_pct, 2),
        "diff_phase_deg": round(diff_phase_deg, 2),
        "filter_gd_ns": round(float(spec.filter_gd_ns), 1),
        "feeder_vswr": round(vswr, 2),
        "feeder_reflection_gamma": round(float(gamma), 4),
        "feeder_return_loss_db": round(float(return_loss_db), 1),
        "sound_intermod_level": round(float(spec.sound_intermod), 3),
        "digital_mer_db": round(mer_db, 2),
    }


def simulate_transmitter_exciter_hpa(
    rgb: np.ndarray,
    spec: TVTransmitterSpec,
    frame_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Pass baseband program video through the Broadcast Transmitter Exciter -> High-Power Amplifier (HPA)
    -> Cavity Diplexer -> Antenna Feeder Line (`tx_chain_mode = 1`).

    Notice how every mechanism acts on the *transmitted signal* (`Y, I, Q` envelope, phase, and group delay)
    rather than painting an arbitrary texture:
      1. **HPA AM-AM Nonlinearity (Differential Gain)**: Compresses luminance peaks above the HPA knee
         and attenuates chroma saturation in high-luminance regions by `1 - DG * Y^2`.
      2. **HPA AM-PM Phase Conversion (Differential Phase)**: Rotates $(I, Q)$ phase by
         $\\Delta\\phi(y,x) = \\text{DP}_{\\text{rad}} \\cdot Y(y,x)^{1.4}$, shifting highlight hues!
      3. **Cavity Diplexer & VSB Filter Group Delay (`filter_gd_ns`)**: Shifts chroma horizontally
         relative to luma (`~73.9 ns/px` at 720px SD) and adds $2T$ asymmetric K-factor ringing on luma edges.
      4. **Tower Feeder VSWR Reflection (`feeder_vswr`)**: Adds a feeder-line reflection echo at
         $\\Gamma = (\\text{VSWR}-1)/(\\text{VSWR}+1)$.
      5. **Vision-Chroma-Sound Intermodulation (`sound_intermod`)**: Adds 920 kHz / 1.57 MHz beat in
         regions where both luminance carrier and chroma subcarrier are strong.
    """
    h, w = rgb.shape[:2]
    scale_w = max(0.5, w / 720.0)
    telem = evaluate_transmitter_telemetry(spec)

    y, i, q = rgb_to_yiq(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    drive = float(spec.pa_drive)

    # 1. Exciter White Clipper & HPA AM-AM Nonlinearity (Differential Gain)
    if drive > 1e-4:
        # S-curve HPA envelope compression + white clipper at 0.96
        k_pa = 1.0 + 1.85 * drive
        y_comp = np.tanh(y * k_pa) / max(1e-6, float(np.tanh(k_pa)))
        y = (1.0 - 0.65 * min(1.0, drive)) * y + (0.65 * min(1.0, drive)) * y_comp
        # Differential Gain: chroma subcarrier sits on top of the luminance ramp on the HPA transfer curve,
        # so high luminance compresses chroma gain by diff_gain_pct
        dg_frac = float(telem["diff_gain_pct"]) / 100.0
        c_gain_map = np.clip(1.0 - dg_frac * (np.clip(y, 0.0, 1.0) ** 1.5), 0.35, 1.15).astype(np.float32)
        i = i * c_gain_map
        q = q * c_gain_map

    # 2. HPA AM-PM Conversion (Differential Phase: luminance-dependent subcarrier phase rotation!)
    dp_deg = float(telem["diff_phase_deg"])
    if abs(dp_deg) > 1e-3:
        phi_map = np.deg2rad(dp_deg) * (np.clip(y, 0.0, 1.0) ** 1.35)
        cs = np.cos(phi_map).astype(np.float32)
        sn = np.sin(phi_map).astype(np.float32)
        i, q = (i * cs - q * sn), (i * sn + q * cs)

    # 3. Cavity Diplexer & VSB Filter Group Delay (`filter_gd_ns`)
    gd_ns = float(spec.filter_gd_ns)
    if gd_ns > 1.0:
        # At 13.5 MHz Rec.601 sampling, 1 pixel = 74.07 ns
        delay_px = (gd_ns / 74.07) * scale_w
        shifts_gd = np.full(h, delay_px, dtype=np.float32)
        i = horizontal_shift(i, shifts_gd, mode="edge")
        q = horizontal_shift(q, shifts_gd, mode="edge")
        # VSB filter asymmetric group-delay 2T ringing on sharp luma edges
        lp_y = lowpass(y, max(0.6, 1.25 * scale_w), axis=1)
        edge_y = y - lp_y
        ring_shift = np.full(h, max(1.0, 2.2 * scale_w), dtype=np.float32)
        ring_y = horizontal_shift(edge_y, ring_shift, mode="edge")
        k_ring = float(np.clip(gd_ns / 350.0, 0.0, 0.85)) * 0.24
        y = y - k_ring * ring_y + (0.45 * k_ring) * horizontal_shift(edge_y, ring_shift * 2.0, mode="edge")

    # 4. Tower Coaxial Feeder VSWR Reflection (`feeder_vswr`)
    gamma = float(telem["feeder_reflection_gamma"])
    if gamma > 0.01:
        # 80m tower feeder round-trip delay ~0.8 us (~11 px at 720w)
        feeder_delay_px = max(2.0, 10.5 * scale_w)
        f_shifts = np.full(h, feeder_delay_px, dtype=np.float32)
        y = y + (0.65 * gamma) * horizontal_shift(y, f_shifts, mode="edge")
        i = i + (0.50 * gamma) * horizontal_shift(i, f_shifts, mode="edge")
        q = q + (0.50 * gamma) * horizontal_shift(q, f_shifts, mode="edge")

    # 5. Vision-Chroma-Sound 920 kHz / 1.57 MHz HPA Intermodulation Beat (`sound_intermod`)
    im_amt = float(spec.sound_intermod) * (0.45 + 0.75 * drive)
    if im_amt > 1e-4:
        chroma_mag = np.hypot(i, q)
        # Beat is strongest where both luma carrier drive and chroma subcarrier amplitude are high
        im_gate = np.clip(y * (chroma_mag * 2.8 + 0.25), 0.0, 1.0)
        xx = np.arange(w, dtype=np.float32)[None, :]
        yy = np.arange(h, dtype=np.float32)[:, None]
        # 920 kHz / 1.57 MHz corresponds to ~5.2 px horizontal period at 720px width
        period_px = max(2.4, 5.1 * scale_w)
        beat_wave = np.cos(2.0 * np.pi * (xx / period_px - yy / 6.0) + float(frame_index) * 0.85).astype(np.float32)
        y = y + (0.14 * im_amt) * im_gate * beat_wave

    out_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)
    return out_rgb, telem


def simulate_off_air_channel_and_receiver(
    tx_rgb: np.ndarray,
    spec: TVTransmitterSpec,
    channel_cn_db: float = 25.0,
    multipath_level: float = 0.24,
    frame_index: int = 0,
    seed: int = 20240101,
) -> np.ndarray:
    """
    Pass the transmitted analog VSB signal (`tx_rgb`) through the Terrestrial Propagation Channel
    (distance attenuation + multipath ghost + thermal noise) -> Consumer TV Receiver -> CRT observation.
    """
    h, w = tx_rgb.shape[:2]
    scale_w = max(0.5, w / 720.0)
    rng = np.random.default_rng((int(seed) * 104729 + int(frame_index) * 179 + 31) & 0xFFFFFFFF)

    y, i, q = rgb_to_yiq(tx_rgb)
    # Receiver IF vestigial-sideband bandwidth limiting
    y = lowpass(y, max(0.7, 1.35 * scale_w), axis=1)
    i = lowpass(i, max(1.4, 2.8 * scale_w), axis=1)
    q = lowpass(q, max(1.6, 3.2 * scale_w), axis=1)

    # Terrestrial multipath ghost (distinct from transmitter feeder VSWR!)
    if multipath_level > 1e-4:
        mp_shifts = np.full(h, max(3.0, 18.0 * scale_w), dtype=np.float32)
        y = y + (0.42 * multipath_level) * horizontal_shift(y, mp_shifts, mode="edge")

    # Channel thermal snow
    snow_rms = float(np.clip(10.0 ** (-(channel_cn_db - 18.0) / 20.0), 0.005, 0.25))
    snow = lowpass(rng.standard_normal((h, w)).astype(np.float32), max(0.6, 1.1 * scale_w), axis=1)
    y = y + snow_rms * snow

    # Subtle CRT scanline & phosphor observation
    rx_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)
    scan_mod = (0.92 + 0.08 * np.cos(np.arange(h, dtype=np.float32) * np.pi))[:, None, None]
    return np.clip(rx_rgb * scan_mod, 0.0, 1.0).astype(np.float32)


def simulate_digital_dtt_transmitter_and_rx(
    rgb: np.ndarray,
    spec: TVTransmitterSpec,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Simulate Digital Terrestrial Television (`DTT`, `tx_chain_mode = 2`):
      - Maps Transmitter HPA MER (`digital_mer_db`) into COFDM / 8-VSB receiver decoding margin.
      - When HPA overdrive / AM-PM / VSWR pushes `MER < 19.5 dB`, the DTT receiver hits the
        digital cliff effect (macroblocking / freeze), whereas above `MER >= 19.5 dB` the
        received picture remains clean despite transmitter nonlinearity!
    """
    telem = evaluate_transmitter_telemetry(spec)
    mer_db = float(telem["digital_mer_db"])
    # Map MER (threshold ~18.5 dB) to equivalent DVB-S C/N (threshold ~5.8 dB)
    eq_cn_db = 5.8 + (mer_db - 18.5) * 0.48
    sat_spec = SatelliteIRDSpec(mode=2, cn_db=max(0.0, eq_cn_db))
    from .satellite_ird import evaluate_satellite_rf_link
    dtt_link = evaluate_satellite_rf_link(sat_spec)
    out = simulate_digital_dvbs_qpsk(rgb, dtt_link, sat_spec, frame_index=frame_index, seed=seed)
    telem["dtt_cliff_state"] = dtt_link["cliff_state"]
    telem["dtt_rs_uncorrected_pkts"] = dtt_link["rs_uncorrected_pkts"]
    return out, telem


def render_end_to_end_broadcast_lab(
    baseband_rgb: np.ndarray,
    tx_probe_rgb: np.ndarray,
    analog_offair_rgb: np.ndarray,
    digital_offair_rgb: np.ndarray,
    telem: dict[str, Any],
    quad_corner: int = 0,
    show_border: int = 1,
    show_text: int = 0,
) -> np.ndarray:
    """
    Render the End-to-End Broadcast Transmission Lab (`tx_chain_mode = 3..7`, `README.txt` §4.4):
      - `quad_corner == 0`: 2×2 Quad View
      - `quad_corner == 1`: Top-Left (`1. STUDIO BASEBAND REF`) full screen
      - `quad_corner == 2`: Top-Right (`2. TX HPA+DIPLEXER PROBE`) full screen
      - `quad_corner == 3`: Bottom-Left (`3. ANALOG OFF-AIR RX+CRT`) full screen
      - `quad_corner == 4`: Bottom-Right (`4. DIGITAL DTT OFF-AIR RX`) full screen
      - `show_border`: `0` = seamless (no border), `1` = thin 2px crosshair divider, `2` = framed studio bezel
      - `show_text`: `0` = no text (clean picture, default), `1` = corner labels only, `2` = corner labels + bottom telemetry bar
    """
    h, w = baseband_rgb.shape[:2]
    scale = max(1, min(w // 240, h // 165))
    corner = int(np.clip(quad_corner, 0, 4))
    b_mode = int(np.clip(show_border, 0, 2))
    t_mode = int(np.clip(show_text, 0, 2))

    entries = [
        (baseband_rgb, "1.STUDIO BASEBAND REF (DIRECT)", (0.35, 0.92, 0.98)),
        (tx_probe_rgb, f"2.TX HPA PROBE (DG:{telem['diff_gain_pct']:.0f}% DP:{telem['diff_phase_deg']:.0f}deg)", (0.98, 0.82, 0.26)),
        (analog_offair_rgb, f"3.ANALOG OFF-AIR+CRT (VSWR:{telem['feeder_vswr']:.2f})", (0.98, 0.56, 0.28)),
        (digital_offair_rgb, f"4.DIGITAL DTT RX (MER:{telem['digital_mer_db']:.1f}dB)", (0.32, 0.96, 0.48)),
    ]

    foot_txt = (
        f"TX CHAIN: DG={telem['diff_gain_pct']:.1f}%  DP={telem['diff_phase_deg']:.1f}deg  "
        f"GD={telem['filter_gd_ns']:.0f}ns  VSWR={telem['feeder_vswr']:.2f}:1  MER={telem['digital_mer_db']:.1f}dB"
    )

    # Single-corner full-screen selection (Corners 1..4)
    if corner in (1, 2, 3, 4):
        sel_img, sel_label, sel_col = entries[corner - 1]
        canvas = _resample_nn(sel_img, h, w).copy()
        if b_mode == 2:
            bw = max(2, min(h, w) // 120)
            canvas[:bw, :] = (0.14, 0.16, 0.20)
            canvas[-bw:, :] = (0.14, 0.16, 0.20)
            canvas[:, :bw] = (0.14, 0.16, 0.20)
            canvas[:, -bw:] = (0.14, 0.16, 0.20)
        if t_mode >= 1:
            _draw_text_5x7(
                canvas, sel_label[:max(16, w // (6 * scale) - 1)],
                6, 6, scale=scale, color=sel_col, bg=(0.04, 0.06, 0.10),
            )
        if t_mode >= 2:
            footer_h = max(12, 10 * scale)
            by0 = h - footer_h
            canvas[by0:h, :] = (0.04, 0.05, 0.07)
            _draw_text_5x7(
                canvas, foot_txt[:max(20, w // (6 * scale) - 1)],
                6, by0 + 2, scale=scale, color=(0.94, 0.92, 0.84),
            )
        return np.clip(canvas, 0.0, 1.0).astype(np.float32)

    # 2x2 Quad View
    footer_h = max(12, 10 * scale) if t_mode >= 2 else 0
    avail_h = max(16, h - footer_h)

    if b_mode == 2:
        pad = max(2, min(h, w) // 95)
        canvas = np.full((h, w, 3), (0.05, 0.06, 0.08), dtype=np.float32)
        cell_h = max(8, (avail_h - 3 * pad) // 2)
        cell_w = max(8, (w - 3 * pad) // 2)
        coords = [
            (pad, pad, cell_h, cell_w),
            (pad, 2 * pad + cell_w, cell_h, cell_w),
            (2 * pad + cell_h, pad, cell_h, cell_w),
            (2 * pad + cell_h, 2 * pad + cell_w, cell_h, cell_w),
        ]
    else:
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        mid_y = avail_h // 2
        mid_x = w // 2
        coords = [
            (0, 0, mid_y, mid_x),
            (0, mid_x, mid_y, w - mid_x),
            (mid_y, 0, avail_h - mid_y, mid_x),
            (mid_y, mid_x, avail_h - mid_y, w - mid_x),
        ]

    for idx_q, (img_q, label_q, col_q) in enumerate(entries):
        y0, x0, ch, cw = coords[idx_q]
        tile = _resample_nn(img_q, ch, cw)
        canvas[y0:y0 + ch, x0:x0 + cw] = tile
        if t_mode >= 1:
            _draw_text_5x7(
                canvas, label_q[:max(16, cw // (6 * scale) - 1)],
                x0 + 4, y0 + 4, scale=scale, color=col_q, bg=(0.04, 0.06, 0.10),
            )

    if b_mode == 1:
        mid_y = avail_h // 2
        mid_x = w // 2
        canvas[max(0, mid_y - 1):min(avail_h, mid_y + 1), :] = (0.12, 0.14, 0.18)
        canvas[:avail_h, max(0, mid_x - 1):min(w, mid_x + 1)] = (0.12, 0.14, 0.18)

    if t_mode >= 2 and footer_h > 0:
        by0 = h - footer_h
        canvas[by0:h, :] = (0.04, 0.05, 0.07)
        _draw_text_5x7(
            canvas, foot_txt[:max(20, w // (6 * scale) - 1)],
            6, by0 + 2, scale=scale, color=(0.94, 0.92, 0.84),
        )
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def apply_tv_transmitter_chain(
    rgb: np.ndarray,
    spec: TVTransmitterSpec,
    frame_index: int = 0,
    seed: int = 20240101,
    channel_cn_db: float = 25.0,
    multipath_level: float = 0.22,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute the Television Broadcast Transmitter Exciter/HPA Chain (`Candidate N`).
    """
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    if not spec.is_active():
        return src, {"enabled": False}

    tx_probe_rgb, telem = simulate_transmitter_exciter_hpa(src, spec, frame_index=frame_index)
    tx_only_l1 = float(np.mean(np.abs(tx_probe_rgb - src)))
    telem["isolated_tx_l1"] = round(tx_only_l1, 4)

    if spec.mode == 1:
        out = tx_probe_rgb
    elif spec.mode == 2:
        out, dtt_t = simulate_digital_dtt_transmitter_and_rx(src, spec, frame_index=frame_index, seed=seed)
        telem.update(dtt_t)
    else:
        ana_rx = simulate_off_air_channel_and_receiver(
            tx_probe_rgb, spec, channel_cn_db=channel_cn_db,
            multipath_level=multipath_level, frame_index=frame_index, seed=seed,
        )
        dig_rx, dtt_t = simulate_digital_dtt_transmitter_and_rx(src, spec, frame_index=frame_index, seed=seed)
        telem.update(dtt_t)
        # Measure isolated contributions: Transmitter distortion alone vs Channel+Receiver addition
        offair_total_l1 = float(np.mean(np.abs(ana_rx - src)))
        channel_added_l1 = float(np.mean(np.abs(ana_rx - tx_probe_rgb)))
        telem["isolated_channel_l1"] = round(channel_added_l1, 4)
        telem["end_to_end_total_l1"] = round(offair_total_l1, 4)
        out = render_end_to_end_broadcast_lab(
            src, tx_probe_rgb, ana_rx, dig_rx, telem,
            quad_corner=spec.quad_corner,
            show_border=spec.show_border,
            show_text=spec.show_text,
        )

    report: dict[str, Any] = {
        "enabled": True,
        **telem,
    }
    report["report"] = dict(report)
    return out, report


def compute_transmitter_audio_effect(spec: TVTransmitterSpec, **kwargs: Any) -> dict[str, Any]:
    """
    Compute companion audio telemetry for the TV Broadcast Transmitter Exciter/HPA chain:
      - Vision-into-Sound Intercarrier Buzz (`intercarrier_buzz_level`) caused by HPA AM-PM & common amplification
        of peak-white video lines modulating the +4.5/+6.0 MHz FM sound carrier at 50/60 Hz field rate.
      - Sound-into-Vision 920 kHz / 1.57 MHz intermodulation beat (`sound_chroma_beat`).
    """
    if not spec.is_active():
        return {"active": False, "intercarrier_buzz_level": 0.0, "sound_chroma_beat": 0.0}
    telem = evaluate_transmitter_telemetry(spec)
    buzz = float(np.clip(0.18 * spec.pa_drive + 0.32 * spec.sound_intermod + 0.004 * spec.pa_am_pm_deg, 0.0, 1.0))
    return {
        "active": True,
        "tx_mode": telem["tx_mode"],
        "intercarrier_buzz_level": round(buzz, 4),
        "sound_chroma_beat": round(float(spec.sound_intermod), 4),
        "digital_mer_db": telem["digital_mer_db"],
    }


class TVExciterPAStage(Stage):
    """
    Node Graph stage (`tv_exciter_pa`): Broadcast TV Exciter + High-Power Amplifier + Cavity Diplexer.
    """
    name = "tv_exciter_pa"
    layers = ("rf",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        r = self.cfg_of("rf")
        self.spec = TVTransmitterSpec.from_params(r)
        if self.spec.mode <= 0:
            self.spec.mode = 1
        self.channel_cn_db = max(14.0, float(r.get("signal_dbm", -60.0)) + 90.0)
        self.multipath_level = float(r.get("multipath_level", 0.22))

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig = buf.meta.get("signal")
        base_rgb = buf.rgb if (buf.rgb is not None and sig is None) else (sig.to_rgb() if sig is not None else buf.rgb)
        out_rgb, rep = apply_tv_transmitter_chain(
            base_rgb, self.spec,
            frame_index=int(buf.index),
            seed=int(self.ctx.seed),
            channel_cn_db=self.channel_cn_db,
            multipath_level=self.multipath_level,
        )
        meta = dict(buf.meta)
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        meta["signal"] = sig_out
        meta["tv_transmitter"] = rep
        comp = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63)
        return FrameBuffer(out_rgb, comp, buf.index, buf.time, buf.std, meta)

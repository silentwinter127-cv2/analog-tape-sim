"""
Candidate M: Satellite IRD Receiver (`satellite_ird`) & Analog FM vs Digital DVB-S Cliff Effect System.

Implements Section 3 of the Broadcast Infrastructure Specification (`README.txt`):
  1. **Six-Layer Satellite Reception Chain**:
     - Layer 1 (`Antenna & Feedhorn`): Parabolic dish pointing loss ($G(\\theta) = G_0 - 12(\\theta/\\theta_{3\\text{dB}})^2$),
       Ku/C-band rain attenuation (`sat_rain_fade_db`, ITU-R P.618), and solar conjunction noise spike (`Sun Outage`).
     - Layer 2 (`LNB - Low Noise Block Downconverter`): C-band (5150 MHz LO) / Ku-band (9750/10600 MHz LO)
       conversion to L-band IF (950-2150 MHz), LNB noise figure / noise temperature, DRO local oscillator
       frequency drift (`sat_lnb_drift_khz`), and oscillator phase noise.
     - Layer 3 (`Tuner & IF Processing`): 27/36 MHz SAW transponder filter, AGC level, and carrier AFC.
     - Layer 4 (`Demodulation & Forward Error Correction`):
       * **Analog Satellite Branch (`sat_ird_mode = 1`, C-Band FM Video + 6.2/6.8 MHz Audio Subcarrier)**:
         FM discriminator + CCIR Rec. 405 de-emphasis. Below the FM threshold ($C/N \\lesssim 11.5\\text{ dB}$),
         noise peaks cause instantaneous $2\\pi$ phase clicks in the FM discriminator, which de-emphasis
         stretches into sharp **horizontal black-and-white comet-tail "sparkles" (FM 门限脉冲打火流星噪点)**
         and subcarrier audio impulse crackle.
       * **Digital DVB-S Branch (`sat_ird_mode = 2`, QPSK + Viterbi Rate 3/4 + Reed-Solomon RS(204,188))**:
         Exhibits the **Digital Cliff Effect (数字悬崖效应)**:
           - **Above Cliff ($C/N > 6.5\\text{ dB}$)**: Viterbi + RS(204,188) corrects all channel bit errors
             (`rs_uncorrected_pkts == 0`), delivering a 100% snow-free picture even while `pre_viterbi_ber`
             and `ts_stats` report front-end signal degradation!
           - **Cliff Edge ($C/N \\in [4.5, 6.5]\\text{ dB}$)**: RS(204,188) (max 8 byte errors per 188-byte
             packet) overflows; uncorrected TS packets spike, causing **MPEG-2 slice macroblock corruption,
             motion-vector error propagation, PTS/DTS audio-video sync slip**, and partial frame freeze.
           - **Below Cliff ($C/N < 4.5\\text{ dB}$)**: QPSK Costas loop loses lock -> **Service Lost / IRD
             Freeze or Black/OSD (`NO SATELLITE SIGNAL`)**.
     - Layer 5 (`Transport Stream Processing`): PID demux (`PAT/PMT`, `Video PID 0x0100`, `Audio PID 0x0101`,
       `PCR PID 0x0100`), continuity counter check, and PTS/DTS clock recovery.
     - Layer 6 (`Dual Output Interface`): Simultaneously provides **Raw Transport Stream Telemetry (`mpeg_ts`)**
       and **Decoded Baseband Video/Audio (`rgb` / `composite` / `yiq`)**.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode, horizontal_shift, lowpass
from .master_playout import _draw_text_5x7, _resample_nn

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

SAT_IRD_MODES: dict[int, str] = {
    0: "off",
    1: "analog_cband_fm",
    2: "digital_dvbs_qpsk",
    3: "analog_vs_digital_cliff_lab",
}


@dataclass
class SatelliteIRDSpec:
    """Configuration for Candidate M (`rf` layer / `sat_dish_lnb` & `sat_ird_receiver` nodes)."""
    mode: int = 0                    # 0=Off, 1=Analog C-Band FM, 2=Digital DVB-S QPSK, 3=Analog vs Digital Cliff Split Lab
    cn_db: float = 12.0              # Nominal clear-sky transponder C/N (dB)
    rain_fade_db: float = 0.0        # Rain fade / sun outage attenuation (0..18 dB)
    dish_mispoint_deg: float = 0.0   # Parabolic antenna pointing error (0..3.5 deg)
    lnb_drift_khz: float = 0.0       # LNB DRO local oscillator drift (-500..500 kHz)
    ird_osd: bool = False            # Overlay professional rack-mount IRD telemetry HUD (default False = no text)
    quad_corner: int = 0             # 0=Split Comparison, 1=Left/Top-Left (Analog FM), 2=Right/Top-Right (Digital DVB-S), 3=Bottom-Left, 4=Bottom-Right
    show_border: int = 1             # 0=No Border, 1=Thin Divider (Default), 2=Framed Bezel
    show_text: int = 0               # 0=No Text (Default), 1=Labels Only, 2=Labels + Telemetry Bar

    @classmethod
    def from_params(cls, cfg: dict) -> "SatelliteIRDSpec":
        c = cfg or {}
        osd_val = int(np.clip(round(float(c.get("sat_ird_osd", 0.0))), 0, 2))
        txt_val = int(np.clip(round(float(c.get("sat_show_text", osd_val))), 0, 2))
        return cls(
            mode=int(np.clip(round(float(c.get("sat_ird_mode", 0.0))), 0, 3)),
            cn_db=float(np.clip(float(c.get("sat_cn_db", 12.0)), 0.0, 26.0)),
            rain_fade_db=float(np.clip(float(c.get("sat_rain_fade_db", 0.0)), 0.0, 20.0)),
            dish_mispoint_deg=float(np.clip(float(c.get("sat_dish_mispoint_deg", 0.0)), 0.0, 4.0)),
            lnb_drift_khz=float(np.clip(float(c.get("sat_lnb_drift_khz", 0.0)), -500.0, 500.0)),
            ird_osd=bool(txt_val > 0),
            quad_corner=int(np.clip(round(float(c.get("sat_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(c.get("sat_show_border", 1.0))), 0, 2)),
            show_text=txt_val,
        )

    def is_active(self) -> bool:
        return self.mode > 0


def evaluate_satellite_rf_link(spec: SatelliteIRDSpec) -> dict[str, Any]:
    """
    Compute Layers 1–5 of the satellite RF & FEC link budget:
      - Parabolic dish pointing loss: $L_{\\text{point}} = 12 (\\theta / \\theta_{3\\text{dB}})^2$ dB
        with $\\theta_{3\\text{dB}} = 1.4^\\circ$.
      - Rain fade attenuation + sky noise temperature increase ($\\Delta T_{\\text{rain}}$).
      - LNB DRO frequency offset penalty & phase noise degradation.
      - Analog FM threshold margin vs Digital DVB-S QPSK ($E_b/N_0$, Pre-Viterbi BER, Post-Viterbi BER,
        and Reed-Solomon `RS(204,188)` Uncorrected TS Packet rate).
    """
    beamwidth_deg = 1.4
    pointing_loss_db = 12.0 * ((spec.dish_mispoint_deg / beamwidth_deg) ** 2)
    # Rain both attenuates the carrier AND raises sky noise temperature toward 275 K (+0.22 dB noise floor per dB rain fade)
    rain_loss_db = float(spec.rain_fade_db) * 1.22
    # LNB drift outside AFC pull-in window (>150 kHz) degrades effective Eb/N0
    lnb_penalty_db = max(0.0, (abs(spec.lnb_drift_khz) - 80.0) / 110.0) * 1.4

    eff_cn_db = float(spec.cn_db - pointing_loss_db - rain_loss_db - lnb_penalty_db)

    # --- Analog C-Band FM Telemetry (Threshold at C/N = 10.5 dB) ---
    fm_threshold_db = 10.5
    fm_margin_db = eff_cn_db - fm_threshold_db
    # Impulse sparkle rate rises exponentially as C/N drops below ~12.5 dB
    if eff_cn_db >= 14.5:
        fm_sparkle_density = 0.0
    else:
        fm_sparkle_density = float(np.clip(((14.5 - eff_cn_db) / 9.5) ** 1.65, 0.0, 1.0))

    # --- Digital DVB-S QPSK + Viterbi (Rate 3/4) + Reed-Solomon RS(204,188) Telemetry ---
    # For QPSK rate 3/4 in 27.5 Msym/s BW, Eb/N0 ≈ C/N - 1.25 dB.
    # Quasi-Error-Free (QEF) cliff threshold is at C/N ≈ 5.8 dB (Eb/N0 ≈ 4.55 dB).
    dvbs_cliff_cn_db = 5.8
    dvbs_margin_db = eff_cn_db - dvbs_cliff_cn_db
    eb_n0_db = eff_cn_db - 1.25

    # Pre-Viterbi raw channel BER: smooth complementary error function approx of QPSK in AWGN
    eb_lin = max(0.05, 10.0 ** (eb_n0_db / 10.0))
    pre_viterbi_ber = float(np.clip(0.5 * np.exp(-0.85 * eb_lin), 1e-8, 0.45))

    # Post-Viterbi BER (coding gain ~5.2 dB, steep waterfall slope)
    post_viterbi_ber = float(np.clip(80.0 * (pre_viterbi_ber ** 2.6), 1e-11, 0.40))

    # Reed-Solomon RS(204,188) corrects up to t=8 byte errors per 204-byte packet.
    # Below post_viterbi_ber ~ 2e-4 (C/N > 6.5 dB), RS corrects 100% of packets -> 0 uncorrected packets!
    # Between C/N 4.5 and 6.5 dB, RS overflows -> cliff edge macroblocking!
    # Below C/N 4.5 dB, QPSK demodulator loses carrier/symbol lock -> service lost!
    if eff_cn_db >= 6.5:
        rs_corrected_pkts = int(round(pre_viterbi_ber * 18000.0))
        rs_uncorrected_pkts = 0
        cliff_state = "LOCKED_QEF"
        cliff_corruption = 0.0
        service_lost = False
        pts_dts_offset_ms = 0.0
    elif eff_cn_db >= 4.4:
        rs_corrected_pkts = int(round(pre_viterbi_ber * 25000.0))
        overflow_t = float(np.clip((6.5 - eff_cn_db) / 2.1, 0.0, 1.0))
        rs_uncorrected_pkts = max(1, int(round(overflow_t ** 1.8 * 420.0)))
        cliff_state = "CLIFF_EDGE_MACROBLOCK"
        cliff_corruption = float(np.clip(0.18 + 0.78 * (overflow_t ** 1.3), 0.15, 0.95))
        service_lost = False
        pts_dts_offset_ms = round(float(overflow_t * 185.0), 1)
    else:
        rs_corrected_pkts = 0
        rs_uncorrected_pkts = 1880
        cliff_state = "CLIFF_SERVICE_LOST"
        cliff_corruption = 1.0
        service_lost = True
        pts_dts_offset_ms = 999.0

    tuner_locked = not service_lost and abs(spec.lnb_drift_khz) < 420.0

    return {
        "nominal_cn_db": round(float(spec.cn_db), 2),
        "pointing_loss_db": round(float(pointing_loss_db), 2),
        "rain_loss_db": round(float(rain_loss_db), 2),
        "lnb_penalty_db": round(float(lnb_penalty_db), 2),
        "cn_db": round(eff_cn_db, 2),
        "eb_n0_db": round(eb_n0_db, 2),
        "tuner_locked": bool(tuner_locked),
        # Analog C-band FM metrics
        "fm_margin_db": round(fm_margin_db, 2),
        "fm_sparkle_density": round(fm_sparkle_density, 4),
        # Digital DVB-S & MPEG-TS metrics
        "dvbs_margin_db": round(dvbs_margin_db, 2),
        "pre_viterbi_ber": float(f"{pre_viterbi_ber:.3e}"),
        "post_viterbi_ber": float(f"{post_viterbi_ber:.3e}"),
        "rs_corrected_pkts": int(rs_corrected_pkts),
        "rs_uncorrected_pkts": int(rs_uncorrected_pkts),
        "cliff_state": cliff_state,
        "cliff_corruption": round(cliff_corruption, 4),
        "cliff_service_lost": bool(service_lost),
        "pts_dts_offset_ms": pts_dts_offset_ms,
        "mpeg_ts": {
            "pmt_pid": "0x0020",
            "video_pid": "0x0100",
            "audio_pid": "0x0101",
            "pcr_pid": "0x0100",
            "continuity_errors": int(rs_uncorrected_pkts),
            "pcr_jitter_ns": round(18.0 + abs(spec.lnb_drift_khz) * 1.4 + (320.0 if rs_uncorrected_pkts > 0 else 0.0), 1),
            "pts_dts_slip_ms": pts_dts_offset_ms,
            "transport_error_indicator": bool(rs_uncorrected_pkts > 0),
        },
    }


def simulate_analog_cband_fm(
    rgb: np.ndarray,
    link_info: dict[str, Any],
    spec: SatelliteIRDSpec,
    frame_index: int = 0,
    seed: int = 20240101,
) -> np.ndarray:
    """
    Simulate Analog C-Band FM Satellite Reception (`sat_ird_mode = 1`):
      1. Triangular FM thermal noise spectrum ($N(f) \\propto f^2$) shaped by CCIR Rec. 405 de-emphasis.
      2. **FM Threshold Sparkles ("鱼骨黑白打火流星点")**:
         When $C/N$ approaches or falls below the FM discriminator threshold (~10.5 dB), noise spikes
         cause instantaneous $\\pm 2\\pi$ phase jumps in the limiter-discriminator. The RC de-emphasis
         integrator ($\tau = 1.5\\text{–}3\\,\\mu\\text{s}$) stretches each impulse into a sharp horizontal
         black or white "comet tail" streak (sharp head on the left, exponential tail to the right).
         Crucially, positive/negative click polarity depends on instantaneous signal deviation (white
         sparkles dominate in dark picture areas; black sparkles dominate in bright picture areas!).
      3. LNB DRO drift shifts the FM deviation DC baseline (clamping error + subcarrier buzz).
    """
    h, w = rgb.shape[:2]
    scale_w = max(0.5, w / 720.0)
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 163 + 53) & 0xFFFFFFFF)

    out = np.asarray(rgb, dtype=np.float32).copy()
    cn_db = float(link_info["cn_db"])
    sparkle_density = float(link_info["fm_sparkle_density"])

    # 1. FM triangular thermal noise + de-emphasis coloration
    cn_eff = max(0.0, cn_db - 6.0)
    fm_noise_rms = float(np.clip(0.28 * (10.0 ** (-cn_eff / 22.0)), 0.002, 0.28))
    raw_n = rng.standard_normal((h, w)).astype(np.float32)
    # FM noise is high-frequency weighted before de-emphasis, leaving a fine horizontal grain + chroma phase noise
    fine_n = raw_n - lowpass(raw_n, max(0.6, 1.4 * scale_w), axis=1)
    deemp_n = lowpass(fine_n, max(0.5, 1.1 * scale_w), axis=1)
    out = out + (fm_noise_rms * 0.85) * deemp_n[..., None]
    out[..., 0] += (fm_noise_rms * 0.45) * lowpass(rng.standard_normal((h, w)).astype(np.float32), max(1.0, 2.4 * scale_w), axis=1)
    out[..., 2] += (fm_noise_rms * 0.45) * lowpass(rng.standard_normal((h, w)).astype(np.float32), max(1.0, 2.4 * scale_w), axis=1)

    # 2. LNB DRO Frequency Drift DC Offset & Edge Over-Deviation Truncation
    if abs(spec.lnb_drift_khz) > 10.0:
        drift_norm = float(spec.lnb_drift_khz) / 500.0
        out = out + 0.08 * drift_norm
        # High-frequency chroma/luma tips clip against the edge of the IF SAW filter when LNB drifts
        if abs(drift_norm) > 0.25:
            lp = lowpass(out, max(0.8, 2.0 * scale_w), axis=1)
            out = lp + (out - lp) * (1.0 - 0.55 * abs(drift_norm))

    # 3. Signature C-Band FM Threshold Sparkles ("黑白流星鱼骨噪点")
    if sparkle_density > 1e-4:
        # Expected number of FM phase-click sparkles per frame scales from ~15 (just below threshold) to ~950 (deep fade)
        n_sparkles = int(rng.poisson(max(2.0, (sparkle_density ** 1.35) * 520.0 * (h / 240.0))))
        if n_sparkles > 0:
            sy = rng.integers(0, h, size=n_sparkles)
            sx = rng.integers(0, max(1, w - 4), size=n_sparkles)
            # De-emphasis RC decay tail length: 4..18 pixels
            tail_lens = rng.integers(max(3, int(4 * scale_w)), max(6, int(19 * scale_w)), size=n_sparkles)
            lum_at_click = (0.299 * out[sy, sx, 0] + 0.587 * out[sy, sx, 1] + 0.114 * out[sy, sx, 2])

            for r_i, c_i, t_len, l_val in zip(sy, sx, tail_lens, lum_at_click):
                r_int = int(r_i)
                c_int = int(c_i)
                c_end = min(w, c_int + int(t_len))
                span = c_end - c_int
                if span <= 0:
                    continue
                # Physical FM click law: when instantaneous carrier frequency is deviated toward white (high freq),
                # noise causes negative-going clicks (black sparkles); in dark regions (low freq), positive-going
                # clicks (blazing white sparkles)!
                go_black = (rng.random() < (0.22 + 0.62 * float(l_val)))
                target_val = 0.01 if go_black else 0.99
                # Sharp impulse head at c_int, exponential RC de-emphasis decay to the right
                decay = np.exp(-np.arange(span, dtype=np.float32) / max(1.5, 0.38 * span))[:, None]
                seg = out[r_int, c_int:c_end]
                out[r_int, c_int:c_end] = seg * (1.0 - decay) + target_val * decay

        # Severe FM threshold collapse also causes line-sync discriminator spikes
        if sparkle_density > 0.35:
            line_spikes = (rng.random(h) < (0.18 * sparkle_density)).astype(np.float32)
            shifts = line_spikes * rng.uniform(-8.0 * scale_w, 8.0 * scale_w, size=h).astype(np.float32)
            for ch in range(3):
                out[..., ch] = horizontal_shift(out[..., ch], shifts)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def simulate_digital_dvbs_qpsk(
    rgb: np.ndarray,
    link_info: dict[str, Any],
    spec: SatelliteIRDSpec,
    frame_index: int = 0,
    seed: int = 20240101,
) -> np.ndarray:
    """
    Simulate Digital DVB-S QPSK + Viterbi + Reed-Solomon Reception (`sat_ird_mode = 2`):
      - **Above the Cliff (`cliff_state == "LOCKED_QEF"`, $C/N \\ge 6.5\\text{ dB}$)**:
        Zero snow, zero sparkles! The picture is clean MPEG-2 4:2:0 digital video because
        Viterbi + RS(204,188) corrects every bit error (`rs_uncorrected_pkts == 0`).
      - **At the Cliff Edge (`cliff_state == "CLIFF_EDGE_MACROBLOCK"`, $C/N \\in [4.4, 6.5)\\text{ dB}$)**:
        Sudden onset of MPEG-2 16x16 macroblock slice loss, horizontal slice error propagation,
        8x8 DCT high-frequency spike blocks (green/magenta YCbCr corruption), and displaced stale blocks!
      - **Below the Cliff (`cliff_state == "CLIFF_SERVICE_LOST"`, $C/N < 4.4\\text{ dB}$)**:
        QPSK demodulator unlocks -> IRD freezes last corrupt I-frame or cuts to black with
        `"DVB-S IRD: NO CARRIER LOCK (E48-32)"` banner.
    """
    h, w = rgb.shape[:2]
    out = np.asarray(rgb, dtype=np.float32).copy()
    cliff_state = str(link_info["cliff_state"])
    corruption = float(link_info["cliff_corruption"])

    # Slight baseline MPEG-2 4:2:0 chroma subsampling (2x2 chroma grid) always present in DVB-S
    y = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    cb = (out[..., 2] - y) * 0.564
    cr = (out[..., 0] - y) * 0.713
    if h >= 4 and w >= 4:
        hh, ww = (h // 2) * 2, (w // 2) * 2
        cb_sub = cb[:hh:2, :ww:2]
        cr_sub = cr[:hh:2, :ww:2]
        cb[:hh, :ww] = np.repeat(np.repeat(cb_sub, 2, axis=0), 2, axis=1)
        cr[:hh, :ww] = np.repeat(np.repeat(cr_sub, 2, axis=0), 2, axis=1)
        out[..., 0] = np.clip(y + 1.402 * cr, 0.0, 1.0)
        out[..., 1] = np.clip(y - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0)
        out[..., 2] = np.clip(y + 1.772 * cb, 0.0, 1.0)

    if cliff_state == "LOCKED_QEF" or corruption <= 1e-4:
        return out

    rng = np.random.default_rng((int(seed) * 198491317 + int(frame_index) * 251 + 19) & 0xFFFFFFFF)
    mb = max(8, min(16, w // 24))
    gh = (h + mb - 1) // mb
    gw = (w + mb - 1) // mb

    if cliff_state == "CLIFF_SERVICE_LOST":
        # Severe freeze + sparse surviving macroblock islands over black/dark-blue IRD screen
        frozen = np.roll(out, shift=mb * 2, axis=1) * 0.18
        for by in range(gh):
            if rng.random() < 0.22:
                y0, y1 = by * mb, min(h, (by + 1) * mb)
                x0 = int(rng.integers(0, max(1, gw // 2))) * mb
                x1 = min(w, x0 + int(rng.integers(2, max(3, gw))) * mb)
                frozen[y0:y1, x0:x1] = np.roll(out[y0:y1, x0:x1], int(rng.integers(-3, 4)) * mb, axis=1)
        scale = max(1, min(w // 175, h // 115))
        _draw_text_5x7(
            frozen,
            f"DVB-S IRD E48: NO SIGNAL LOCK (C/N={link_info['cn_db']:.1f}dB)",
            int(w * 0.08), int(h * 0.44),
            scale=scale, color=(0.98, 0.28, 0.24), bg=(0.06, 0.06, 0.10),
        )
        return np.clip(frozen, 0.0, 1.0).astype(np.float32)

    # CLIFF_EDGE_MACROBLOCK: MPEG-2 slice desynchronization & 16x16 macroblock corruption
    n_slices = max(2, int(round(corruption * gh * 0.65)))
    slice_rows = rng.choice(np.arange(gh), size=min(gh, n_slices), replace=False)
    for by in slice_rows:
        y0, y1 = int(by * mb), min(h, int((by + 1) * mb))
        start_bx = int(rng.integers(0, max(1, gw - 2)))
        end_bx = min(gw, start_bx + int(rng.integers(2, max(3, gw - start_bx + 1))))
        for bx in range(start_bx, end_bx):
            x0, x1 = int(bx * mb), min(w, int((bx + 1) * mb))
            mode_r = float(rng.random())
            if mode_r < 0.48:
                # Motion-vector displacement from neighboring macroblock (freeze/repeat block)
                dy_mb = int(rng.integers(-2, 3)) * mb
                dx_mb = int(rng.integers(-3, 4)) * mb
                sy0 = int(np.clip(y0 + dy_mb, 0, max(0, h - (y1 - y0))))
                sx0 = int(np.clip(x0 + dx_mb, 0, max(0, w - (x1 - x0))))
                out[y0:y1, x0:x1] = out[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
            elif mode_r < 0.78:
                # DC-only flat macroblock quantization collapse
                dc_col = out[y0:y1, x0:x1].mean(axis=(0, 1), keepdims=True)
                out[y0:y1, x0:x1] = dc_col
            else:
                # Corrupted YCbCr VLC coefficient -> high-saturation green or magenta DCT basis block
                yy_b, xx_b = np.mgrid[0:(y1 - y0), 0:(x1 - x0)].astype(np.float32)
                dct_pat = 0.5 + 0.45 * np.cos(np.pi * xx_b * 2.0 / max(1, x1 - x0)) * np.cos(np.pi * yy_b * 2.0 / max(1, y1 - y0))
                if rng.random() < 0.5:
                    # YCbCr (0,0,0) green block artifact
                    out[y0:y1, x0:x1, 0] = 0.08 * dct_pat
                    out[y0:y1, x0:x1, 1] = 0.82 * dct_pat
                    out[y0:y1, x0:x1, 2] = 0.12 * dct_pat
                else:
                    # Magenta Cr spike block
                    out[y0:y1, x0:x1, 0] = 0.88 * dct_pat
                    out[y0:y1, x0:x1, 1] = 0.14 * dct_pat
                    out[y0:y1, x0:x1, 2] = 0.84 * dct_pat

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def render_satellite_cliff_comparison_lab(
    analog_rgb: np.ndarray,
    digital_rgb: np.ndarray,
    link_info: dict[str, Any],
    quad_corner: int = 0,
    show_border: int = 1,
    show_text: int = 0,
) -> np.ndarray:
    """
    Render the side-by-side Analog C-Band FM vs Digital DVB-S Cliff Effect Comparison Lab (`sat_ird_mode = 3`).
    """
    h, w = analog_rgb.shape[:2]
    scale = max(1, min(w // 230, h // 150))
    corner = int(np.clip(quad_corner, 0, 4))
    b_mode = int(np.clip(show_border, 0, 2))
    t_mode = int(np.clip(show_text, 0, 2))

    if corner in (1, 3):
        out = analog_rgb.copy()
    elif corner in (2, 4):
        out = digital_rgb.copy()
    else:
        out = np.zeros_like(analog_rgb)
        mid_x = w // 2
        out[:, :mid_x] = analog_rgb[:, :mid_x]
        out[:, mid_x:] = digital_rgb[:, mid_x:]
        if b_mode >= 1:
            out[:, max(0, mid_x - 1):min(w, mid_x + 1)] = (0.14, 0.16, 0.20) if b_mode == 1 else (0.94, 0.78, 0.22)
        if b_mode == 2:
            bw = max(2, min(h, w) // 120)
            out[:bw, :] = (0.14, 0.16, 0.20)
            out[-bw:, :] = (0.14, 0.16, 0.20)
            out[:, :bw] = (0.14, 0.16, 0.20)
            out[:, -bw:] = (0.14, 0.16, 0.20)

    if t_mode >= 1:
        if corner in (1, 3):
            _draw_text_5x7(
                out, f"ANALOG C-BAND FM (SPARKLES:{link_info['fm_sparkle_density']*100:.0f}%)",
                6, 6, scale=scale, color=(0.98, 0.86, 0.30), bg=(0.05, 0.07, 0.12),
            )
        elif corner in (2, 4):
            _draw_text_5x7(
                out, f"DVB-S QPSK ({link_info['cliff_state']})",
                6, 6, scale=scale,
                color=(0.28, 0.96, 0.48) if link_info["rs_uncorrected_pkts"] == 0 else (0.98, 0.32, 0.28),
                bg=(0.05, 0.07, 0.12),
            )
        else:
            mid_x = w // 2
            _draw_text_5x7(
                out, f"ANALOG C-BAND FM (SPARKLES:{link_info['fm_sparkle_density']*100:.0f}%)",
                6, 6, scale=scale, color=(0.98, 0.86, 0.30), bg=(0.05, 0.07, 0.12),
            )
            _draw_text_5x7(
                out, f"DVB-S QPSK ({link_info['cliff_state']})",
                mid_x + 6, 6, scale=scale,
                color=(0.28, 0.96, 0.48) if link_info["rs_uncorrected_pkts"] == 0 else (0.98, 0.32, 0.28),
                bg=(0.05, 0.07, 0.12),
            )

    if t_mode >= 2:
        bar_h = max(12, 10 * scale)
        out[h - bar_h:h, :] = (0.05, 0.06, 0.09)
        ts_str = (
            f"SAT C/N:{link_info['cn_db']:.1f}dB  Eb/N0:{link_info['eb_n0_db']:.1f}dB  "
            f"VITERBI_BER:{link_info['post_viterbi_ber']:.1e}  RS_ERR:{link_info['rs_uncorrected_pkts']}pkts"
        )
        _draw_text_5x7(
            out, ts_str[:max(20, w // (6 * scale) - 1)],
            6, h - bar_h + 2, scale=scale, color=(0.88, 0.94, 0.98),
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_satellite_ird(
    rgb: np.ndarray,
    spec: SatelliteIRDSpec,
    frame_index: int = 0,
    seed: int = 20240101,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute the 6-Layer Satellite IRD Receiver & Analog/Digital Cliff Effect model (`Candidate M`).
    """
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    if not spec.is_active():
        return src, {"enabled": False}

    link_info = evaluate_satellite_rf_link(spec)

    if spec.mode == 1:
        out = simulate_analog_cband_fm(src, link_info, spec, frame_index=frame_index, seed=seed)
    elif spec.mode == 2:
        out = simulate_digital_dvbs_qpsk(src, link_info, spec, frame_index=frame_index, seed=seed)
    else:
        ana = simulate_analog_cband_fm(src, link_info, spec, frame_index=frame_index, seed=seed)
        dig = simulate_digital_dvbs_qpsk(src, link_info, spec, frame_index=frame_index, seed=seed)
        out = render_satellite_cliff_comparison_lab(
            ana, dig, link_info,
            quad_corner=spec.quad_corner,
            show_border=spec.show_border,
            show_text=spec.show_text,
        )

    if spec.ird_osd and spec.mode in (1, 2):
        h, w = out.shape[:2]
        scale = max(1, min(w // 220, h // 145))
        hud = (
            f"IRD [{SAT_IRD_MODES.get(spec.mode, '').upper()}] C/N:{link_info['cn_db']:.1f}dB "
            f"RS_ERR:{link_info['rs_uncorrected_pkts']}"
        )
        _draw_text_5x7(out, hud, 6, h - 11 * scale, scale=scale, color=(0.38, 0.95, 0.62), bg=(0.04, 0.06, 0.09))

    report: dict[str, Any] = {
        "enabled": True,
        "ird_mode": SAT_IRD_MODES.get(spec.mode, "off"),
        **link_info,
    }
    report["report"] = dict(report)
    return out, report


def compute_satellite_audio_effect(spec: SatelliteIRDSpec, **kwargs: Any) -> dict[str, Any]:
    """
    Compute companion audio telemetry for Satellite IRD reception:
      - Analog C-band FM (6.2/6.8 MHz audio subcarrier): threshold impulse crackle & FM hiss.
      - Digital DVB-S MPEG-1 Layer II audio: clean above cliff, PTS/DTS sync slip & chirp/mute on cliff edge.
    """
    if not spec.is_active():
        return {"active": False, "pts_slip_ms": 0.0, "fm_crackle": 0.0, "dvbs_drop_mute": 0.0}
    info = evaluate_satellite_rf_link(spec)
    return {
        "active": True,
        "ird_mode": SAT_IRD_MODES.get(spec.mode, "off"),
        "pts_slip_ms": float(info["pts_dts_offset_ms"]) if spec.mode in (2, 3) else 0.0,
        "fm_crackle": float(info["fm_sparkle_density"]) if spec.mode in (1, 3) else 0.0,
        "dvbs_drop_mute": float(info["cliff_corruption"]) if spec.mode in (2, 3) else 0.0,
    }


class SatDishLNBStage(Stage):
    """Node Graph stage (`sat_dish_lnb`): Parabolic Dish + Feedhorn + LNB Downconverter."""
    name = "sat_dish_lnb"
    layers = ("rf",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        self.spec = SatelliteIRDSpec.from_params(self.cfg_of("rf"))
        if self.spec.mode <= 0:
            self.spec.mode = 2

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        link_info = evaluate_satellite_rf_link(self.spec)
        meta = dict(buf.meta)
        meta["sat_lnb_if"] = link_info
        sig = meta.get("signal") or SignalFrame.from_rgb(buf.rgb, buf.std)
        meta["signal"] = sig
        comp = buf.composite if buf.composite is not None else composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
        return FrameBuffer(buf.rgb, comp, buf.index, buf.time, buf.std, meta)


class SatIRDReceiverStage(Stage):
    """
    Node Graph stage (`sat_ird_receiver`): Professional Rack-Mount Satellite IRD Receiver.
    Outputs both Decoded Baseband (`out` port) and MPEG-TS / Telemetry (`ts` port).
    """
    name = "sat_ird_receiver"
    layers = ("rf",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        self.spec = SatelliteIRDSpec.from_params(self.cfg_of("rf"))
        if self.spec.mode <= 0:
            self.spec.mode = 2

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig = buf.meta.get("signal")
        base_rgb = buf.rgb if (buf.rgb is not None and sig is None) else (sig.to_rgb() if sig is not None else buf.rgb)
        out_rgb, rep = apply_satellite_ird(
            base_rgb, self.spec, frame_index=int(buf.index), seed=int(self.ctx.seed)
        )
        meta = dict(buf.meta)
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        meta["signal"] = sig_out
        meta["satellite_ird"] = rep
        meta["mpeg_ts"] = rep.get("mpeg_ts", {})
        if rep.get("cliff_service_lost", False) and self.spec.mode == 2:
            meta["format"] = "no_signal"
        comp = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63)
        return FrameBuffer(out_rgb, comp, buf.index, buf.time, buf.std, meta)

"""
Candidate J: Transfer, Dubbing & Transcoding Workstation (`transfer_workstation`)
and Shared Infrastructure A/C (`TransferEvent` Provenance Chain & 4-Way Replication Comparison Lab).

Models how video and audio signals evolve across multi-generation copying, analog-to-digital ingest,
and cross-codec digital transcoding, built on three core engineering principles from `README.txt`:

  1. **Three Distinct Workstation Topologies (`transfer_station_mode`)**:
     - **Mode 1 — Analog-to-Digital Ingest Station (`VHS/Video8/Betacam -> TBC -> Capture Card ADC`)**:
       Models playback time-base error, TBC lock mode (`free_run_no_tbc`, `line_tbc`, `full_frame_tbc_genlock`),
       BT.601 8-bit `16..235` quantization range clamping, active-line HBI cropping/ringing (`transfer_adc_crop`),
       and independent audio/video crystal clock slip (`transfer_av_drift_ms`).
       Crucially, **the capture interface never magically repairs upstream source defects** (dropouts,
       head-switching tears, and baked-in colour-under bleed survive intact into the captured file).
     - **Mode 2 — Multi-Generation Tape Dubbing Station (`Source VTR -> DA/ProcAmp/TBC -> Target VTR`)**:
       Models `G = 1..8` sequential tape-to-tape dubbing generations (`transfer_generations`).
       Separates **what a TBC can correct** from **what accumulates irreversibly**:
         * Without TBC (`transfer_tbc_mode = 0`): per-line horizontal sampling jitter compounds across
           generations (`jitter_px ~ G^0.82`), bending vertical edges into severe serpentine waves.
         * With TBC (`transfer_tbc_mode >= 1`): horizontal line jitter is re-clocked at each dubbing pass
           (`jitter_px` stays bounded), **YET** luma FM bandwidth roll-off (`BW_0 / G^0.48`), constant-k
           filter group-delay ringing, 629 kHz colour-under rightward shift & phase noise, FM limiter
           noise floor (`+3 dB` per doubling of `G`), and S-curve magnetic dynamic compression **still
           accumulate irreversibly**!
     - **Mode 3 — Digital Format Transcoding Station (`4:2:2 -> 4:1:1 DV25 -> 4:2:0 MPEG-1 VCD`)**:
       Models chroma subsampling geometry mismatch (`4:2:2` vs `4:1:1` vs `4:2:0`, where cascading
       `4:1:1 -> 4:2:0` leaves `1/4 x 1/2 = 1/8` chroma resolution), multi-generation 8x8 DCT
       re-quantization, and motion-compensated P-frame residual accumulation while keeping horizontal
       time-base jitter at `0.00 px`.
     - **Mode 4 — 4-Way Replication Chain Comparison Lab (`2x2 Quad Split + Telemetry & Provenance`)**:
       Simultaneously runs the current frame through:
         * `CH1: GEN-0 MASTER SOURCE` (Reference input)
         * `CH2: PATH A - ANALOG DUB xG (NO TBC)` (Severe cumulative jitter + bandwidth/chroma loss)
         * `CH3: PATH B - ANALOG DUB xG (WITH TBC)` (Stable geometry, irreversible bandwidth/chroma/noise loss)
         * `CH4: PATH C - DIGITAL TRANSCODE (4:1:1 -> 4:2:0)` (Zero jitter, DCT block grid & chroma decimation)
       and overlays per-branch waveform/signal metrics (`PSNR dB`, `BW MHz`, `JITTER px`, `CHROMA LOSS %`,
       `AV OFFSET ms`) and the `TransferEvent` provenance chain.

  2. **`TransferEvent` Provenance History (`Infrastructure A & C`)**:
     Every transfer, dubbing, or transcode pass emits a structured `TransferEvent` recording
     `source_id`, `destination_id`, `operation`, `input_format`, `output_format`, `processing_chain`,
     `timebase_mode`, `audio_sync_offset`, `generation_index`, and quantitative degradation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import (
    SignalFrame,
    composite_encode,
    horizontal_shift,
    lowpass,
    rgb_to_yiq,
    yiq_to_rgb,
)
from .dv import (
    DVCodec,
    DVParams,
    rgb_to_ycbcr,
    subsample_411,
    subsample_420,
    upsample_411,
    upsample_420,
    ycbcr_to_rgb,
)
from .mpeg1 import MPEG1Codec, MPEG1CodecParams
from .multiview_wall import _draw_bitmap_text

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

STATION_MODE_NAMES: dict[int, str] = {
    0: "off",
    1: "analog_ingest_station",
    2: "multi_gen_tape_dub_station",
    3: "digital_transcode_station",
    4: "replication_comparison_lab",
}

TBC_MODE_NAMES: dict[int, str] = {
    0: "free_run_no_tbc",
    1: "line_tbc",
    2: "full_frame_tbc_genlock",
}

CODEC_TARGET_NAMES: dict[int, str] = {
    0: "BT601_YUV422_8BIT",
    1: "DV25_411_25MBPS",
    2: "MPEG1_VCD_420_1150KBPS",
    3: "CASCADE_422_TO_DV411_TO_VCD420",
}


@dataclass
class TransferEvent:
    """
    Structured provenance record for a single ingest, dubbing, or transcode generation
    (`README.txt` Section II.2 & Section VI Infrastructure A/C).
    """
    source_id: str
    destination_id: str
    operation: str
    input_format: str
    output_format: str
    processing_chain: list[str]
    timebase_mode: str
    audio_sync_offset: float      # milliseconds (ms)
    generation_index: int         # 1..N
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "operation": self.operation,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "processing_chain": list(self.processing_chain),
            "timebase_mode": self.timebase_mode,
            "audio_sync_offset": round(float(self.audio_sync_offset), 2),
            "generation_index": int(self.generation_index),
            "metrics": dict(self.metrics),
        }


@dataclass
class TransferWorkstationSpec:
    """Configuration of the Transfer, Dubbing & Transcoding Workstation (`player` layer)."""
    station_mode: int = 0         # 0=Off, 1=Analog Ingest, 2=Multi-Gen Tape Dub, 3=Digital Transcode, 4=2x2 Comparison Lab
    generations: int = 3          # 1..8 generations
    tbc_mode: int = 1             # 0=No TBC (Free-Run), 1=Line TBC, 2=Full-Frame TBC + Genlock
    codec_target: int = 3         # 0=BT.601 4:2:2, 1=DV25 4:1:1, 2=MPEG-1 VCD 4:2:0, 3=4:2:2->DV411->VCD420 Cascade
    adc_crop: float = 0.45        # Capture card HBI blanking crop / anti-alias filter ringing & 16..235 legal range clamp
    av_drift_ms: float = 35.0     # Audio-video clock slip per pass (ms)
    quad_corner: int = 0          # 0=2x2 Quad Lab, 1=Top-Left(Gen-0), 2=Top-Right(No-TBC), 3=Bottom-Left(With-TBC), 4=Bottom-Right(Digital)
    show_border: int = 1          # 0=Seamless, 1=Thin Crosshair, 2=Framed Studio Bezel
    show_text: int = 0            # 0=No Text (Default), 1=Corner Titles, 2=Corner Titles + Telemetry Bar

    @classmethod
    def from_params(cls, cfg: dict | None) -> "TransferWorkstationSpec":
        c = cfg or {}
        return cls(
            station_mode=int(np.clip(round(float(c["transfer_station_mode"])) if "transfer_station_mode" in c else 0, 0, 4)),
            generations=int(np.clip(round(float(c["transfer_generations"])) if "transfer_generations" in c else 3, 1, 8)),
            tbc_mode=int(np.clip(round(float(c["transfer_tbc_mode"])) if "transfer_tbc_mode" in c else 1, 0, 2)),
            codec_target=int(np.clip(round(float(c["transfer_codec_target"])) if "transfer_codec_target" in c else 3, 0, 3)),
            adc_crop=float(np.clip(float(c["transfer_adc_crop"]) if "transfer_adc_crop" in c else 0.45, 0.0, 1.0)),
            av_drift_ms=float(np.clip(float(c["transfer_av_drift_ms"]) if "transfer_av_drift_ms" in c else 35.0, -250.0, 250.0)),
            quad_corner=int(np.clip(round(float(c.get("transfer_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(c.get("transfer_show_border", 1.0))), 0, 2)),
            show_text=int(np.clip(round(float(c.get("transfer_show_text", 0.0))), 0, 2)),
        )

    def is_active(self) -> bool:
        return int(self.station_mode) > 0


def _draw_badge(
    canvas: np.ndarray,
    text: str,
    x0: int,
    y0: int,
    scale: int = 1,
    color: tuple[float, float, float] = (0.94, 0.96, 0.94),
    bg: tuple[float, float, float] | None = (0.06, 0.08, 0.12),
) -> None:
    h, w = canvas.shape[:2]
    sc = max(1, int(scale))
    if bg is not None:
        bw = len(text) * 6 * sc + 4 * sc
        bh = 7 * sc + 4 * sc
        by0 = max(0, int(y0) - 2 * sc)
        by1 = min(h, by0 + bh)
        bx0 = max(0, int(x0) - 2 * sc)
        bx1 = min(w, bx0 + bw)
        if by1 > by0 and bx1 > bx0:
            canvas[by0:by1, bx0:bx1] = np.array(bg, dtype=np.float32)
    _draw_bitmap_text(canvas, text, int(x0), int(y0), color=color, scale=sc)


def _resample_nn(img: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (h_out, w_out):
        return img
    ys = np.clip((np.arange(h_out) * h) // max(1, h_out), 0, h - 1)
    xs = np.clip((np.arange(w_out) * w) // max(1, w_out), 0, w - 1)
    return img[ys][:, xs]


def _measure_branch_metrics(
    ref_rgb: np.ndarray,
    out_rgb: np.ndarray,
    jitter_px: float,
    eff_bw_mhz: float,
    dct_block_err: float = 0.0,
) -> dict[str, float]:
    """Compute quantitative comparison metrics between `out_rgb` and `ref_rgb`."""
    diff = np.abs(out_rgb.astype(np.float32) - ref_rgb.astype(np.float32))
    mse = float(np.mean(diff ** 2))
    psnr = 48.0 if mse < 1e-6 else float(np.clip(10.0 * np.log10(1.0 / mse), 8.0, 48.0))

    _, i_ref, q_ref = rgb_to_yiq(np.clip(ref_rgb, 0.0, 1.0))
    _, i_out, q_out = rgb_to_yiq(np.clip(out_rgb, 0.0, 1.0))
    c_ref_hi = float(np.mean(np.abs(np.diff(i_ref, axis=1))) + np.mean(np.abs(np.diff(q_ref, axis=0))))
    c_out_hi = float(np.mean(np.abs(np.diff(i_out, axis=1))) + np.mean(np.abs(np.diff(q_out, axis=0))))
    c_err = float(np.mean(np.hypot(i_out - i_ref, q_out - q_ref)))
    chroma_loss_pct = float(np.clip((1.0 - min(1.0, c_out_hi / max(1e-5, c_ref_hi))) * 65.0 + c_err * 180.0, 0.0, 99.0))

    return {
        "psnr_db": round(psnr, 2),
        "luma_bw_mhz": round(float(eff_bw_mhz), 2),
        "timebase_jitter_px": round(float(jitter_px), 3),
        "chroma_loss_pct": round(chroma_loss_pct, 1),
        "dct_block_err": round(float(dct_block_err), 4),
        "l1_diff": round(float(diff.mean()), 4),
    }


def _simulate_analog_ingest(
    rgb_in: np.ndarray,
    spec: TransferWorkstationSpec,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, list[TransferEvent], dict[str, Any]]:
    """
    Workstation A: Analog Tape Digitization / Ingest (`VHS/Video8/Betacam -> TBC -> Capture Card ADC`).
    """
    h, w = rgb_in.shape[:2]
    scale_w = w / 720.0
    rng = np.random.default_rng((int(seed) * 19391 + int(frame_index) * 409 + 11) & 0xFFFFFFFF)

    y, i, q = rgb_to_yiq(np.clip(rgb_in, 0.0, 1.0).astype(np.float32))
    rows = np.arange(h, dtype=np.float32)

    # 1. TBC Mode vs Capture Card PLL Line-Lock Response
    if spec.tbc_mode == 0:
        # Free-run capture without TBC: capture card's horizontal PLL hunts on tape jitter,
        # introducing upper-raster hook + per-line sampling phase error
        hook = np.exp(-rows / max(4.0, h * 0.08)) * (3.6 * scale_w) * float(np.sin(frame_index * 0.9 + 0.4))
        line_err = rng.standard_normal(h).astype(np.float32) * (1.35 * scale_w)
        line_err = np.convolve(line_err, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same") + hook
        y = horizontal_shift(y, line_err)
        i = horizontal_shift(i, line_err)
        q = horizontal_shift(q, line_err)
        # Subcarrier burst phase error from un-stabilized H-sync gate
        ph_jit = (line_err / max(1.0, scale_w)) * 0.14
        cs, sn = np.cos(ph_jit)[:, None].astype(np.float32), np.sin(ph_jit)[:, None].astype(np.float32)
        i, q = i * cs - q * sn, i * sn + q * cs
        jitter_rms = float(np.std(line_err))
        av_offset = float(spec.av_drift_ms) * 1.35
        chain = ["ANALOG_VTR_PB", "DIRECT_COMPOSITE_BNC", "ADC_PLL_UNLOCKED_8BIT"]
    elif spec.tbc_mode == 1:
        # Line TBC: stabilizes horizontal sync & burst phase per line, small residual jitter
        line_err = rng.standard_normal(h).astype(np.float32) * (0.24 * scale_w)
        y = horizontal_shift(y, line_err)
        i = horizontal_shift(i, line_err)
        q = horizontal_shift(q, line_err)
        jitter_rms = float(np.std(line_err))
        av_offset = float(spec.av_drift_ms) * 0.45
        chain = ["ANALOG_VTR_PB", "LINE_TBC_FIFO", "ADC_BT601_YUV422_8BIT"]
    else:
        # Full-Frame TBC + Genlock: frame-synchronizer locks H & V timing and audio sample clock
        line_err = rng.standard_normal(h).astype(np.float32) * (0.06 * scale_w)
        y = horizontal_shift(y, line_err)
        jitter_rms = float(np.std(line_err))
        av_offset = float(spec.av_drift_ms) * 0.10
        chain = ["ANALOG_VTR_PB", "FRAME_SYNC_TBC_GENLOCK", "PROC_AMP_CLAMP", "ADC_BT601_YUV422_10BIT"]

    # 2. Capture Card Anti-Aliasing Filter & 4:2:2 Chroma Sampling
    y = lowpass(y, max(0.35, 0.65 * scale_w), axis=1)
    i = lowpass(i, max(0.65, 1.35 * scale_w), axis=1)
    q = lowpass(q, max(0.65, 1.35 * scale_w), axis=1)
    out = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0)

    # 3. Capture Card Active Window HBI Cropping / Ringing & BT.601 Legal Range (16..235) Quantization
    crop = float(spec.adc_crop)
    if crop > 1e-3 and w >= 32:
        # BT.601 8-bit legal range clamping & quantization step
        levels = max(64.0, 219.0 - 35.0 * crop)
        out = np.round(np.clip(out, 0.02 * crop, 1.0 - 0.03 * crop) * levels) / levels
        # Left anti-aliasing filter step ringing & right HBI blanking porch exposed by capture window
        lw = max(3, int(round((4 + 8 * crop) * scale_w)))
        rw = max(3, int(round((3 + 7 * crop) * scale_w)))
        xs_l = np.linspace(0.0, 1.0, lw, dtype=np.float32)
        env_l = np.clip(0.06 + 0.94 * xs_l + 0.18 * crop * np.sin(xs_l * 2.5 * np.pi) * np.exp(-xs_l * 1.5), 0.0, 1.2)
        out[:, :lw, :] = np.clip(out[:, :lw, :] * ((1.0 - crop) + crop * env_l[None, :, None]), 0.0, 1.0)
        xs_r = np.linspace(1.0, 0.0, rw, dtype=np.float32)
        env_r = np.clip(0.05 + 0.95 * xs_r, 0.0, 1.0)
        out[:, -rw:, :] = np.clip(out[:, -rw:, :] * ((1.0 - crop) + crop * env_r[None, :, None]), 0.0, 1.0)

    metrics = _measure_branch_metrics(rgb_in, out, jitter_px=jitter_rms, eff_bw_mhz=4.6 - 0.4 * crop)
    ev = TransferEvent(
        source_id="tape_master_vhs",
        destination_id="ingest_bt601_capture.mov",
        operation="analog_to_digital_ingest",
        input_format="VHS_PAL_COMPOSITE",
        output_format="BT601_YUV422_8BIT",
        processing_chain=chain,
        timebase_mode=TBC_MODE_NAMES.get(spec.tbc_mode, "line_tbc"),
        audio_sync_offset=av_offset,
        generation_index=1,
        metrics=metrics,
    )
    return out.astype(np.float32), [ev], metrics


def _simulate_multi_gen_tape_dub(
    rgb_in: np.ndarray,
    generations: int,
    tbc_mode: int,
    av_drift_ms: float = 25.0,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, list[TransferEvent], dict[str, Any]]:
    """
    Workstation B: Multi-Generation Tape Dubbing (`Source VTR -> DA/TBC -> Target VTR`, Gen 1..8).

    Explicitly separates:
      - **Correctable time-base error** (suppressed when `tbc_mode >= 1`, cumulative when `tbc_mode == 0`)
      - **Irreversible generation losses** (luma FM bandwidth narrowing, constant-k group delay ringing,
        629 kHz colour-under delay & vertical smear, FM limiter noise floor, and S-curve contrast crushing)
        which accumulate across generations regardless of TBC!
    """
    h, w = rgb_in.shape[:2]
    scale_w = w / 720.0
    scale_h = h / 480.0
    gens = int(np.clip(generations, 1, 8))

    cur_rgb = np.clip(rgb_in, 0.0, 1.0).astype(np.float32).copy()
    events: list[TransferEvent] = []
    rows = np.arange(h, dtype=np.float32)
    total_jitter_disp = np.zeros(h, dtype=np.float32)
    eff_bw = 3.2

    for g in range(1, gens + 1):
        rng_g = np.random.default_rng((int(seed) * 31337 + int(frame_index) * 997 + g * 7919) & 0xFFFFFFFF)
        y, i, q = rgb_to_yiq(cur_rgb)

        # 1. Irreversible Luma FM Bandwidth Narrowing + Pre-emphasis Overshoot Ringing per generation
        lp_sigma = max(0.55, (0.85 + 0.28 * g) * scale_w)
        y_lp = lowpass(y, lp_sigma, axis=1)
        # Constant-k group-delay edge overshoot on the right of transitions
        y_delayed = horizontal_shift(y_lp, np.full(h, 1.4 * scale_w, dtype=np.float32))
        ring_edge = y_lp - y_delayed
        y = np.clip(y_lp + 0.14 * ring_edge, 0.0, 1.05)
        eff_bw = 3.2 / (g ** 0.46)

        # 2. Irreversible Colour-Under (629 kHz) Heterodyne Delay, Bandwidth Loss, Vertical 1H Comb Average & Phase Drift
        c_sigma = max(1.2, (2.2 + 0.55 * g) * scale_w)
        i = lowpass(i, c_sigma, axis=1)
        q = lowpass(q, c_sigma, axis=1)
        # 1H delay-line vertical chroma averaging & rightward group-delay shift (~1.15 px per generation)
        i = 0.68 * i + 0.32 * np.roll(i, 1, axis=0)
        q = 0.68 * q + 0.32 * np.roll(q, 1, axis=0)
        c_shift = np.full(h, 1.15 * scale_w, dtype=np.float32)
        i = horizontal_shift(i, c_shift)
        q = horizontal_shift(q, c_shift)

        # Subcarrier phase drift & ACC gain compression per generation
        ph_deg = (3.8 if tbc_mode == 0 else 1.8) * float(np.sin(g * 1.1 + frame_index * 0.17))
        ph_rad = np.deg2rad(ph_deg) + rng_g.standard_normal(h).astype(np.float32) * (0.045 if tbc_mode == 0 else 0.018)
        cs = np.cos(ph_rad)[:, None].astype(np.float32)
        sn = np.sin(ph_rad)[:, None].astype(np.float32)
        acc_decay = 0.945 if tbc_mode >= 1 else 0.915
        i, q = (i * cs - q * sn) * acc_decay, (i * sn + q * cs) * acc_decay

        # 3. Time-base Jitter: Cumulative without TBC vs Re-clocked with TBC
        if tbc_mode == 0:
            # Without TBC: each generation adds transport wave + high-frequency jitter that compounds on previous errors
            wave = np.sin(rows / max(6.0, h * 0.14) + frame_index * 0.45 + g * 0.8) * (1.25 * (g ** 0.35) * scale_w)
            jit = rng_g.standard_normal(h).astype(np.float32) * (0.85 * scale_w)
            jit = np.convolve(jit, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same") + wave
            total_jitter_disp = total_jitter_disp + jit
            chain = [f"VTR_SRC_G{g-1}", "COMPOSITE_DA", f"VTR_REC_G{g}_NO_TBC"]
        elif tbc_mode == 1:
            # Line TBC: resets upstream horizontal jitter each pass, leaving only small TBC residual
            jit = rng_g.standard_normal(h).astype(np.float32) * (0.18 * scale_w)
            total_jitter_disp = 0.15 * total_jitter_disp + jit
            chain = [f"VTR_SRC_G{g-1}", "LINE_TBC_CORRECTOR", f"VTR_REC_G{g}"]
        else:
            # Full-frame TBC + Genlock: near-zero horizontal jitter accumulation
            jit = rng_g.standard_normal(h).astype(np.float32) * (0.07 * scale_w)
            total_jitter_disp = 0.05 * total_jitter_disp + jit
            chain = [f"VTR_SRC_G{g-1}", "FRAME_SYNC_TBC_GENLOCK", "PROC_AMP", f"VTR_REC_G{g}"]

        y = horizontal_shift(y, jit if tbc_mode == 0 else total_jitter_disp)
        i = horizontal_shift(i, jit if tbc_mode == 0 else total_jitter_disp)
        q = horizontal_shift(q, jit if tbc_mode == 0 else total_jitter_disp)

        # 4. Irreversible Magnetic S-Curve Nonlinearity (Pedestal Crushing + Highlight Compression) & Tape Noise Floor
        y = np.clip((y - 0.018) * 1.035, 0.0, 1.0)
        y = 0.5 + 0.5 * np.tanh((y - 0.5) * 2.15) / float(np.tanh(0.5 * 2.15))
        luma_noise = lowpass(rng_g.standard_normal((h, w)).astype(np.float32), max(0.6, 1.2 * scale_w), axis=1)
        chroma_noise_i = lowpass(rng_g.standard_normal((h, w)).astype(np.float32), max(1.4, 2.8 * scale_w), axis=1)
        chroma_noise_q = lowpass(rng_g.standard_normal((h, w)).astype(np.float32), max(1.4, 2.8 * scale_w), axis=1)
        y = y + (0.0085 * np.sqrt(g)) * luma_noise
        i = i + (0.0065 * np.sqrt(g)) * chroma_noise_i
        q = q + (0.0065 * np.sqrt(g)) * chroma_noise_q

        # 5. Cumulative bottom Head-Switching zone growth across generations (each VTR adds its own switch line step)
        hs_rows = min(h // 4, int(round((4 + 2 * g) * scale_h)))
        if hs_rows > 0:
            hs_step = (1.8 if tbc_mode == 0 else 0.65) * scale_w * (1.0 if (g % 2 == 1) else -0.8)
            y[-hs_rows:, :] = horizontal_shift(y[-hs_rows:, :], np.full(hs_rows, hs_step, dtype=np.float32))

        cur_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)
        g_metrics = _measure_branch_metrics(
            rgb_in, cur_rgb,
            jitter_px=float(np.std(total_jitter_disp)),
            eff_bw_mhz=eff_bw,
        )
        events.append(TransferEvent(
            source_id=f"tape_gen_{g-1}",
            destination_id=f"tape_gen_{g}",
            operation="multi_gen_tape_dub",
            input_format="VHS_COMPOSITE",
            output_format="VHS_COMPOSITE",
            processing_chain=chain,
            timebase_mode=TBC_MODE_NAMES.get(tbc_mode, "line_tbc"),
            audio_sync_offset=float(av_drift_ms) * g * (0.25 if tbc_mode >= 1 else 0.85),
            generation_index=g,
            metrics=g_metrics,
        ))

    final_metrics = events[-1].metrics if events else _measure_branch_metrics(rgb_in, cur_rgb, 0.0, 3.2)
    return cur_rgb, events, final_metrics


def _measure_8x8_blockiness(rgb: np.ndarray) -> float:
    """Measure 8x8 DCT block-boundary discontinuity relative to interior pixel steps."""
    lum = np.clip(rgb, 0.0, 1.0) @ LUMA
    h, w = lum.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    dx = np.abs(np.diff(lum, axis=1))
    dy = np.abs(np.diff(lum, axis=0))
    bx_cols = np.arange(7, w - 1, 8)
    by_rows = np.arange(7, h - 1, 8)
    if bx_cols.size == 0 or by_rows.size == 0:
        return 0.0
    b_energy = 0.5 * (float(dx[:, bx_cols].mean()) + float(dy[by_rows, :].mean()))
    all_energy = 0.5 * (float(dx.mean()) + float(dy.mean()))
    return float(max(0.0, b_energy - 0.75 * all_energy))


def _simulate_digital_transcode(
    rgb_in: np.ndarray,
    generations: int,
    codec_target: int,
    av_drift_ms: float = 18.0,
    std=None,
    frame_index: int = 0,
) -> tuple[np.ndarray, list[TransferEvent], dict[str, Any]]:
    """
    Workstation C: Digital Format Transcoding (`4:2:2 -> 4:1:1 DV25 -> 4:2:0 MPEG-1 VCD`, Gen 1..8).

    Models:
      - Exact `0.00 px` horizontal timebase jitter (digital sample grid is locked),
      - Chroma subsampling footprint reduction (`4:2:2` vs `4:1:1` horizontal 4x decimation vs
        `4:2:0` 2x2 decimation, and their `1/8` intersection when cascaded),
      - Multi-generation 8x8 DCT re-quantization and block-grid phase shifts across passes.
    """
    from ..core.standards import get_standard
    std_obj = std if std is not None else get_standard("ntsc")
    gens = int(np.clip(generations, 1, 8))
    cur_rgb = np.clip(rgb_in, 0.0, 1.0).astype(np.float32).copy()
    h, w = cur_rgb.shape[:2]
    events: list[TransferEvent] = []

    for g in range(1, gens + 1):
        # Determine codec stage in this generation
        if codec_target == 0:
            # BT.601 4:2:2 8-bit re-sampling pass
            y, cb, cr = rgb_to_ycbcr(cur_rgb)
            if w % 2 == 0:
                cb_s = 0.5 * (cb[:, 0::2] + cb[:, 1::2])
                cr_s = 0.5 * (cr[:, 0::2] + cr[:, 1::2])
                cb = np.repeat(cb_s, 2, axis=1)[:, :w]
                cr = np.repeat(cr_s, 2, axis=1)[:, :w]
            y = np.round(y * 219.0) / 219.0
            cb = np.round(cb * 224.0) / 224.0
            cr = np.round(cr * 224.0) / 224.0
            cur_rgb = np.clip(ycbcr_to_rgb(y, cb, cr), 0.0, 1.0)
            in_fmt = "BT601_YUV422_8BIT"
            out_fmt = "BT601_YUV422_8BIT"
            chain = [f"DEC_422_G{g}", "CSC_BT601_8BIT", f"ENC_422_G{g}"]
            eff_bw = 5.2 - 0.08 * g
        elif codec_target == 1:
            # DV25 4:1:1 (4x1 horizontal chroma subsampling + 8x8 DCT quantization)
            y, cb, cr = rgb_to_ycbcr(cur_rgb)
            # 4:1:1 true 4-pixel horizontal chroma box decimation
            pad_w = (4 - (w % 4)) % 4
            if pad_w:
                cb_p = np.pad(cb, ((0, 0), (0, pad_w)), mode="edge")
                cr_p = np.pad(cr, ((0, 0), (0, pad_w)), mode="edge")
            else:
                cb_p, cr_p = cb, cr
            cb_411 = cb_p.reshape(h, -1, 4).mean(axis=2)
            cr_411 = cr_p.reshape(h, -1, 4).mean(axis=2)
            cb = np.repeat(cb_411, 4, axis=1)[:, :w]
            cr = np.repeat(cr_411, 4, axis=1)[:, :w]
            cur_rgb = np.clip(ycbcr_to_rgb(y, cb, cr), 0.0, 1.0)

            # Slight 1-px block grid shift on generation > 1 to model re-encoding after NLE trim/title composite
            shift_px = (g - 1) % 2
            if shift_px > 0:
                cur_rgb = np.roll(cur_rgb, shift_px, axis=1)
            dv = DVCodec(std_obj, DVParams(
                chroma_mode="411",
                quality=max(0.35, 0.92 / (g ** 0.38)),
                base_step=0.014 * (1.0 + 0.28 * (g - 1)),
            ))
            sig_coded = dv.encode_decode(SignalFrame.from_rgb(cur_rgb, std_obj))
            cur_rgb = sig_coded.to_rgb()
            if shift_px > 0:
                cur_rgb = np.roll(cur_rgb, -shift_px, axis=1)
            in_fmt = "BT601_YUV422" if g == 1 else "DV25_411"
            out_fmt = "DV25_411_25MBPS"
            chain = [f"CHROMA_411_DECIMATE_G{g}", "DCT_8X8_CLASS_QUANT", "DV25_PACK"]
            eff_bw = 4.8 - 0.18 * g
        elif codec_target == 2:
            # MPEG-1 VCD 4:2:0 (352x240 CIF decimation + 4:2:0 chroma + coarse MPEG-1 intra/inter matrix)
            mp = MPEG1Codec(MPEG1CodecParams(
                quality=max(0.30, 0.88 / (g ** 0.45)),
                motion_compensation=1.0,
                width=352,
                height=240,
                bitrate_kbps=max(650.0, 1150.0 - 95.0 * (g - 1)),
                hbi_border=0.35,
            ))
            cur_rgb = mp.encode_decode(cur_rgb, frame_index=int(frame_index))
            in_fmt = "BT601_YUV422" if g == 1 else "MPEG1_VCD_420"
            out_fmt = "MPEG1_VCD_420_1150KBPS"
            chain = [f"CIF_352X240_RESAMPLE_G{g}", "CHROMA_420_SUBSAMPLE", "MPEG1_DCT_ZIGZAG_QUANT"]
            eff_bw = 2.6 - 0.16 * g
        else:
            # Target 3: Cross-Format Cascade (4:2:2 -> DV25 4:1:1 -> MPEG-1 VCD 4:2:0)
            # First apply 4:1:1 horizontal 4x chroma decimation + DV25 DCT
            y, cb, cr = rgb_to_ycbcr(cur_rgb)
            pad_w = (4 - (w % 4)) % 4
            cb_p = np.pad(cb, ((0, 0), (0, pad_w)), mode="edge") if pad_w else cb
            cr_p = np.pad(cr, ((0, 0), (0, pad_w)), mode="edge") if pad_w else cr
            cb_411 = np.repeat(cb_p.reshape(h, -1, 4).mean(axis=2), 4, axis=1)[:, :w]
            cr_411 = np.repeat(cr_p.reshape(h, -1, 4).mean(axis=2), 4, axis=1)[:, :w]
            # Then apply 4:2:0 vertical 2x chroma decimation (resulting in 4x2 = 1/8 chroma footprint!)
            cb_s, cr_s = subsample_420(cb_411, cr_411)
            cb_411_420, cr_411_420 = upsample_420(cb_s, cr_s, y.shape)
            cur_rgb = np.clip(ycbcr_to_rgb(y, cb_411_420, cr_411_420), 0.0, 1.0)

            dv = DVCodec(std_obj, DVParams(
                chroma_mode="411",
                quality=max(0.40, 0.90 / (g ** 0.30)),
                base_step=0.015 * (1.0 + 0.22 * (g - 1)),
            ))
            cur_rgb = dv.encode_decode(SignalFrame.from_rgb(cur_rgb, std_obj)).to_rgb()
            mp = MPEG1Codec(MPEG1CodecParams(
                quality=max(0.35, 0.85 / (g ** 0.40)),
                motion_compensation=1.0,
                width=352,
                height=240,
                bitrate_kbps=max(700.0, 1150.0 - 80.0 * (g - 1)),
                hbi_border=0.45,
            ))
            cur_rgb = mp.encode_decode(cur_rgb, frame_index=int(frame_index))
            in_fmt = "BT601_422" if g == 1 else "DV25_411_INTERMEDIATE"
            out_fmt = "MPEG1_VCD_420_CASCADE"
            chain = [
                f"INGEST_422_G{g}",
                "DV25_411_DCT_ENCODE",
                "411_TO_420_CHROMA_REDECIMATE",
                "MPEG1_CIF_1150K_ENCODE",
            ]
            eff_bw = 2.45 - 0.15 * g

        blk_err = _measure_8x8_blockiness(cur_rgb)
        g_metrics = _measure_branch_metrics(
            rgb_in, cur_rgb,
            jitter_px=0.0,
            eff_bw_mhz=max(1.2, eff_bw),
            dct_block_err=blk_err,
        )
        events.append(TransferEvent(
            source_id=f"digital_asset_g{g-1}",
            destination_id=f"transcode_asset_g{g}",
            operation="digital_format_transcode",
            input_format=in_fmt,
            output_format=out_fmt,
            processing_chain=chain,
            timebase_mode="digital_sample_clock_locked",
            audio_sync_offset=float(av_drift_ms) * g * 0.35,
            generation_index=g,
            metrics=g_metrics,
        ))

    final_metrics = events[-1].metrics if events else _measure_branch_metrics(rgb_in, cur_rgb, 0.0, 4.5)
    return cur_rgb.astype(np.float32), events, final_metrics


def render_replication_comparison_lab(
    rgb_in: np.ndarray,
    spec: TransferWorkstationSpec,
    std=None,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Mode 4: 4-Way Replication Chain Comparison Lab (`README.txt` Section II.2).

    Runs the exact same input frame through:
      1. `GEN-0 MASTER`: Original signal
      2. `PATH A (ANALOG DUB xG NO-TBC)`: Multi-gen tape dub without TBC (cumulative jitter + chroma/bw loss)
      3. `PATH B (ANALOG DUB xG WITH-TBC)`: Multi-gen tape dub with TBC (stable sync, irreversible chroma/bw/noise loss)
      4. `PATH C (DIGITAL 411->420 xG)`: Digital transcode chain (zero jitter, 4:1:1->4:2:0 chroma loss + 8x8 DCT grid)
    and composites a 2x2 laboratory monitor wall with per-quadrant signal metrics and `TransferEvent` provenance.
    """
    h, w = rgb_in.shape[:2]
    gens = int(np.clip(spec.generations, 1, 8))

    # Compute the three degraded comparison branches
    dub_no_tbc_rgb, ev_no_tbc, m_no_tbc = _simulate_analog_tape_branch(
        rgb_in, gens=gens, tbc_mode=0, av_drift_ms=spec.av_drift_ms, frame_index=frame_index, seed=seed
    )
    dub_tbc_rgb, ev_tbc, m_tbc = _simulate_analog_tape_branch(
        rgb_in, gens=gens, tbc_mode=2, av_drift_ms=spec.av_drift_ms, frame_index=frame_index, seed=seed
    )
    dig_rgb, ev_dig, m_dig = _simulate_digital_transcode(
        rgb_in, generations=gens, codec_target=spec.codec_target,
        av_drift_ms=spec.av_drift_ms, std=std, frame_index=frame_index,
    )
    m_ref = _measure_branch_metrics(rgb_in, rgb_in, jitter_px=0.0, eff_bw_mhz=5.5, dct_block_err=0.0)

    corner = int(np.clip(getattr(spec, "quad_corner", 0), 0, 4))
    b_mode = int(np.clip(getattr(spec, "show_border", 1), 0, 2))
    t_mode = int(np.clip(getattr(spec, "show_text", 0), 0, 2))
    sc_txt = max(1, min(w // 380, h // 260))

    quads = [
        (0, 0, rgb_in, f"GEN-0 MASTER REF", m_ref, (0.35, 0.88, 0.48)),
        (0, 1, dub_no_tbc_rgb, f"PATH A: DUB x{gens} NO-TBC", m_no_tbc, (0.95, 0.42, 0.35)),
        (1, 0, dub_tbc_rgb, f"PATH B: DUB x{gens} + TBC", m_tbc, (0.95, 0.78, 0.28)),
        (1, 1, dig_rgb, f"PATH C: DIG 411->420 x{gens}", m_dig, (0.42, 0.78, 0.96)),
    ]

    if corner in (1, 2, 3, 4):
        _, _, q_img, title, met, accent = quads[corner - 1]
        canvas = _resample_nn(np.clip(q_img, 0.0, 1.0), h, w).copy()
        if t_mode >= 1:
            _draw_badge(canvas, title, 6, 6, scale=sc_txt, color=accent, bg=(0.05, 0.07, 0.10))
        if t_mode >= 2:
            stat_line = (
                f"PSNR:{met['psnr_db']:.1f}dB BW:{met['luma_bw_mhz']:.1f}M "
                f"JIT:{met['timebase_jitter_px']:.2f}px CHR-{met['chroma_loss_pct']:.0f}%"
            )
            _draw_badge(canvas, stat_line, 6, max(18, h - 14 * sc_txt), scale=sc_txt, color=(0.92, 0.94, 0.96), bg=(0.04, 0.06, 0.09))
    else:
        footer_h = max(24, int(h * 0.11)) if t_mode >= 2 else 0
        grid_h = max(32, h - footer_h)
        cell_h = grid_h // 2
        cell_w = w // 2
        pad = max(2, min(w, h) // 160) if b_mode == 2 else 0
        canvas = np.full((h, w, 3), 0.05, dtype=np.float32)

        for r_idx, c_idx, q_img, title, met, accent in quads:
            y0 = r_idx * cell_h + pad
            y1 = (grid_h if r_idx == 1 else (r_idx + 1) * cell_h) - pad
            x0 = c_idx * cell_w + pad
            x1 = (w if c_idx == 1 else (c_idx + 1) * cell_w) - pad
            if y1 <= y0 + 4 or x1 <= x0 + 4:
                continue
            tile = _resample_nn(np.clip(q_img, 0.0, 1.0), y1 - y0, x1 - x0)
            canvas[y0:y1, x0:x1] = tile
            if b_mode == 2:
                canvas[y0:y0 + 1, x0:x1] = np.array(accent, dtype=np.float32)
                canvas[y1 - 1:y1, x0:x1] = np.array(accent, dtype=np.float32) * 0.65
            if t_mode >= 1:
                _draw_badge(canvas, title, x0 + 4, y0 + 4, scale=sc_txt, color=accent, bg=(0.05, 0.07, 0.10))
            if t_mode >= 2:
                stat_line = (
                    f"PSNR:{met['psnr_db']:.1f}dB BW:{met['luma_bw_mhz']:.1f}M "
                    f"JIT:{met['timebase_jitter_px']:.2f}px CHR-{met['chroma_loss_pct']:.0f}%"
                )
                _draw_badge(
                    canvas, stat_line,
                    x0 + 4, max(y0 + 14, y1 - 11 * sc_txt),
                    scale=sc_txt, color=(0.92, 0.94, 0.96), bg=(0.04, 0.06, 0.09),
                )

        if b_mode == 1:
            canvas[max(0, cell_h - 1):min(grid_h, cell_h + 1), :] = (0.14, 0.16, 0.20)
            canvas[:grid_h, max(0, cell_w - 1):min(w, cell_w + 1)] = (0.14, 0.16, 0.20)

        if t_mode >= 2 and footer_h > 0:
            fy0 = grid_h
            canvas[fy0:h, :] = np.array([0.07, 0.09, 0.13], dtype=np.float32)
            canvas[fy0:fy0 + 2, :] = np.array([0.28, 0.52, 0.78], dtype=np.float32)
            _draw_badge(
                canvas,
                f"REPLICATION LAB GEN={gens}  TBC_JITTER_SUPPRESSION: {m_no_tbc['timebase_jitter_px']:.2f}px -> {m_tbc['timebase_jitter_px']:.2f}px  "
                f"DIG_DCT_BLK:{m_dig['dct_block_err']:.4f}  AV_SYNC:{spec.av_drift_ms:+.1f}ms",
                6, fy0 + 5, scale=sc_txt, color=(0.95, 0.86, 0.42), bg=None,
            )

    provenance_chain = [e.to_dict() for e in (ev_no_tbc[-1:] + ev_tbc[-1:] + ev_dig[-1:])]
    report = {
        "station_mode": "replication_comparison_lab",
        "generations": gens,
        "branches": {
            "master_ref": m_ref,
            "analog_dub_no_tbc": m_no_tbc,
            "analog_dub_with_tbc": m_tbc,
            "digital_transcode": m_dig,
        },
        "tbc_jitter_reduction_ratio": round(
            float(m_no_tbc["timebase_jitter_px"]) / max(1e-4, float(m_tbc["timebase_jitter_px"])), 2
        ),
        "transfer_events": provenance_chain,
    }
    return np.clip(canvas, 0.0, 1.0).astype(np.float32), report


def _simulate_analog_tape_branch(
    rgb_in: np.ndarray,
    gens: int,
    tbc_mode: int,
    av_drift_ms: float,
    frame_index: int,
    seed: int,
) -> tuple[np.ndarray, list[TransferEvent], dict[str, Any]]:
    return _simulate_multi_gen_tape_dub(
        rgb_in, generations=gens, tbc_mode=tbc_mode,
        av_drift_ms=av_drift_ms, frame_index=frame_index, seed=seed,
    )


def apply_transfer_workstation(
    rgb_in: np.ndarray,
    spec: TransferWorkstationSpec,
    std=None,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute the selected Transfer, Dubbing, or Transcoding Workstation (`transfer_station_mode = 1..4`)
    and return `(output_rgb, transfer_report_dict_with_TransferEvents)`.
    """
    if not spec.is_active():
        return rgb_in, {"active": False, "station_mode": "off", "transfer_events": []}

    mode = int(spec.station_mode)
    if mode == 1:
        out_rgb, events, metrics = _simulate_analog_ingest(
            rgb_in, spec, frame_index=frame_index, seed=seed
        )
        return out_rgb, {
            "active": True,
            "station_mode": STATION_MODE_NAMES[1],
            "generations": 1,
            "tbc_mode": TBC_MODE_NAMES.get(spec.tbc_mode, "line_tbc"),
            "audio_sync_offset_ms": events[-1].audio_sync_offset if events else 0.0,
            "metrics": metrics,
            "transfer_events": [e.to_dict() for e in events],
        }

    if mode == 2:
        out_rgb, events, metrics = _simulate_multi_gen_tape_dub(
            rgb_in,
            generations=spec.generations,
            tbc_mode=spec.tbc_mode,
            av_drift_ms=spec.av_drift_ms,
            frame_index=frame_index,
            seed=seed,
        )
        return out_rgb, {
            "active": True,
            "station_mode": STATION_MODE_NAMES[2],
            "generations": int(spec.generations),
            "tbc_mode": TBC_MODE_NAMES.get(spec.tbc_mode, "line_tbc"),
            "audio_sync_offset_ms": events[-1].audio_sync_offset if events else 0.0,
            "metrics": metrics,
            "transfer_events": [e.to_dict() for e in events],
        }

    if mode == 3:
        out_rgb, events, metrics = _simulate_digital_transcode(
            rgb_in,
            generations=spec.generations,
            codec_target=spec.codec_target,
            av_drift_ms=spec.av_drift_ms,
            std=std,
            frame_index=frame_index,
        )
        return out_rgb, {
            "active": True,
            "station_mode": STATION_MODE_NAMES[3],
            "generations": int(spec.generations),
            "codec_target": CODEC_TARGET_NAMES.get(spec.codec_target, "CASCADE_422_TO_DV411_TO_VCD420"),
            "audio_sync_offset_ms": events[-1].audio_sync_offset if events else 0.0,
            "metrics": metrics,
            "transfer_events": [e.to_dict() for e in events],
        }

    # Mode 4: 4-Way Replication Chain Comparison Lab
    lab_rgb, lab_rep = render_replication_comparison_lab(
        rgb_in, spec, std=std, frame_index=frame_index, seed=seed
    )
    lab_rep["active"] = True
    return lab_rgb, lab_rep


def compute_transfer_audio_effect(
    spec: TransferWorkstationSpec,
    sample_rate: int,
    num_samples: int,
    seed: int = 20240101,
) -> dict[str, Any]:
    """
    Compute audio sync slip (`shift_samples`), cumulative tape dub hiss (`hiss_add`),
    and digital transcode quantization step for the audio simulator.
    """
    if not spec.is_active():
        return {"active": False, "shift_samples": 0, "hiss_add": 0.0, "lp_cutoff_hz": 16000.0}

    gens = 1 if spec.station_mode == 1 else int(np.clip(spec.generations, 1, 8))
    if spec.station_mode == 1:
        scale_tbc = 1.35 if spec.tbc_mode == 0 else (0.45 if spec.tbc_mode == 1 else 0.10)
        offset_ms = float(spec.av_drift_ms) * scale_tbc
        hiss_add = 0.004
        lp_hz = 14000.0
    elif spec.station_mode == 2:
        scale_tbc = 0.85 if spec.tbc_mode == 0 else 0.25
        offset_ms = float(spec.av_drift_ms) * gens * scale_tbc
        hiss_add = 0.012 * np.sqrt(gens)
        lp_hz = max(2400.0, 9500.0 / (gens ** 0.38))
    else:
        offset_ms = float(spec.av_drift_ms) * gens * 0.35
        hiss_add = 0.002 * gens
        lp_hz = 11000.0 if spec.codec_target >= 2 else 15000.0

    shift_samples = int(round((offset_ms / 1000.0) * sample_rate))
    rng = np.random.default_rng((int(seed) * 65537 + gens * 131) & 0xFFFFFFFF)
    noise_vec = rng.standard_normal(max(1, int(num_samples))).astype(np.float32) * hiss_add
    return {
        "active": True,
        "offset_ms": round(offset_ms, 2),
        "shift_samples": shift_samples,
        "hiss_add": round(float(hiss_add), 5),
        "lp_cutoff_hz": round(float(lp_hz), 1),
        "noise_vec": noise_vec,
    }


class TransferStationStage(Stage):
    """
    Node-graph stage (`transfer_station`) implementing Candidate J:
    Analog-to-Digital Ingest, Multi-Generation Tape Dubbing, Digital Transcoding & 4-Way Comparison Lab.
    """
    name = "transfer_station"
    layers = ("player",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("player")
        self.spec = TransferWorkstationSpec.from_params(cfg)
        if not self.spec.is_active():
            # When explicitly placed as a `transfer_station` node in a graph, default to Mode 4 (Comparison Lab)
            self.spec.station_mode = 4

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig = buf.meta["signal"] if "signal" in buf.meta else None
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        out_rgb, rep = apply_transfer_workstation(
            rgb_in, self.spec, std=buf.std, frame_index=int(buf.index), seed=self.ctx.seed
        )
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        comp_out = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63)
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["transfer_workstation"] = rep
        return FrameBuffer(out_rgb, comp_out, buf.index, buf.time, buf.std, meta)

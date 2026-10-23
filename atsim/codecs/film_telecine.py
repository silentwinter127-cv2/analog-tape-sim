"""
电影胶片（16mm / 35mm）化学成像与电视胶转磁系统 (`film_telecine`)
========================================================================

本模块与独立拆出的 **`atsim/codecs/photochemical_film.py` (`01b 胶片光化学与 LOMO 暗房模块`)** 协同工作：
- 光化学乳剂感光、10 种胶片型号 (`FILM_STOCKS[1..10]`)、C-41/E-6 交叉冲印、LOMO 光学/桃心散景/漏光/重曝与片基老化由 `photochemical_film.py` 提供；
- 本模块聚焦于 **Rank Cintel 飞点扫描 / CCD 胶转磁传递 (`24fps -> 29.97i 3:2 Pulldown` 隔行梳状伪影与 `24fps -> 25i PAL +4.167%` 加速升调)** 以及 **`2×2` 跨媒介历史对照实验室 (`胶片 -> 胶转磁 -> 家用VHS -> 流媒体`)**。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass, vertical_shift
from .photochemical_film import (
    FILM_STOCKS,
    FilmStockProfile,
    PhotochemicalFilmSpec,
    simulate_film_aging_and_projector,
    simulate_photochemical_film,
)

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class FilmTelecineSpec:
    enabled: bool = False
    stock: int = 0                  # 0=Off, 1..10 see FILM_STOCKS in photochemical_film.py
    gamma_contrast: float = 1.0     # H&D curve gamma multiplier
    halation: float = 0.35          # Anti-halation backing red-orange scatter
    grain_amount: float = 0.28      # Silver-halide / dye-cloud grain intensity
    grain_size: float = 1.45        # Spatial clumping radius (px)
    dye_fade: float = 0.0           # Asymmetric Cyan/Yellow dye fading (0..1)
    shrinkage: float = 0.0          # Vinegar syndrome acetate shrinkage (0..1)
    scratches_dust: float = 0.15    # Longitudinal scratches + single-frame dust/hair
    gate_weave: float = 0.25        # 2D mechanical projector/telecine gate weave
    shutter_flicker: float = 0.12   # Rotary shutter luminance flicker
    telecine_pulldown: int = 0      # 0=24p Direct, 1=NTSC 3:2 Pulldown, 2=PAL 25fps +4.167%, 3=2x2 Cross-Media Lab
    quad_corner: int = 0            # 0=2x2 Lab, 1=Top-Left(Film), 2=Top-Right(Telecine), 3=Bottom-Left(VHS Dub), 4=Bottom-Right(Stream)
    show_border: int = 1            # 0=Seamless, 1=Thin Crosshair, 2=Framed Bezel
    show_text: int = 0              # 0=No Text (Default), 1=Corner Titles

    @classmethod
    def from_params(cls, cfg: dict[str, Any], fallback_cam: dict[str, Any] | None = None) -> "FilmTelecineSpec":
        pspec = PhotochemicalFilmSpec.from_params(cfg, fallback_cam=fallback_cam)
        return cls(
            enabled=pspec.enabled,
            stock=pspec.stock,
            gamma_contrast=pspec.gamma_contrast,
            halation=pspec.halation,
            grain_amount=pspec.grain_amount,
            grain_size=pspec.grain_size,
            dye_fade=pspec.dye_fade,
            shrinkage=pspec.shrinkage,
            scratches_dust=pspec.scratches_dust,
            gate_weave=pspec.gate_weave,
            shutter_flicker=pspec.shutter_flicker,
            telecine_pulldown=pspec.telecine_pulldown,
            quad_corner=pspec.quad_corner,
            show_border=pspec.show_border,
            show_text=pspec.show_text,
        )


def compute_telecine_pull_mapping(video_frame_index: int, pulldown_mode: int) -> dict[str, Any]:
    """
    Compute which 24fps film frame(s) feed the top (even) and bottom (odd) video fields
    for a given video frame index (`模块` 核心机制):

    * `pulldown_mode == 0` (24p Direct Progressive Scan):
      `top_film_idx = bottom_film_idx = video_frame_index`, `is_combed_frame = False`.
    * `pulldown_mode == 1` (NTSC 24fps -> 29.97i `3:2 Pulldown`, 5-video-frame cycle `AA - BB - BC - CD - DD`):
      - `cadence_pos == 0` (`AA`): Top=4k+0, Bottom=4k+0 (Clean progressive frame)
      - `cadence_pos == 1` (`BB`): Top=4k+1, Bottom=4k+1 (Clean progressive frame)
      - `cadence_pos == 2` (`BC`): Top=4k+1, Bottom=4k+2 (**Interlaced split-frame combing!**)
      - `cadence_pos == 3` (`CD`): Top=4k+2, Bottom=4k+3 (**Interlaced split-frame combing!**)
      - `cadence_pos == 4` (`DD`): Top=4k+3, Bottom=4k+3 (Clean progressive frame)
    * `pulldown_mode == 2` (PAL 24fps -> 25i `+4.167% Speedup`):
      1:1 frame mapping (`2:2 Pulldown`), playback speed = `25 / 24 = 1.04167x`, audio pitch = `+70.67 cents`.
    """
    v_idx = max(0, int(video_frame_index))
    mode = int(pulldown_mode)
    if mode == 1:
        cycle = v_idx // 5
        pos = v_idx % 5
        base_f = cycle * 4
        table = (
            (base_f + 0, base_f + 0, "AA", False),
            (base_f + 1, base_f + 1, "BB", False),
            (base_f + 1, base_f + 2, "BC", True),
            (base_f + 2, base_f + 3, "CD", True),
            (base_f + 3, base_f + 3, "DD", False),
        )
        f_top, f_bot, pat, combed = table[pos]
        return {
            "pulldown_mode": "NTSC_3_2",
            "cadence_phase": pos,
            "cadence_pattern": pat,
            "top_field_film_frame": f_top,
            "bottom_field_film_frame": f_bot,
            "is_combed_frame": combed,
            "speed_ratio": 1.0,
            "audio_pitch_shift_cents": 0.0,
        }
    if mode == 2:
        speed_ratio = 25.0 / 24.0
        cents = float(1200.0 * np.log2(speed_ratio))
        return {
            "pulldown_mode": "PAL_2_2_SPEEDUP",
            "cadence_phase": v_idx % 2,
            "cadence_pattern": "AA",
            "top_field_film_frame": v_idx,
            "bottom_field_film_frame": v_idx,
            "is_combed_frame": False,
            "speed_ratio": round(speed_ratio, 6),
            "audio_pitch_shift_cents": round(cents, 2),
        }
    return {
        "pulldown_mode": "DIRECT_24P",
        "cadence_phase": 0,
        "cadence_pattern": "P",
        "top_field_film_frame": v_idx,
        "bottom_field_film_frame": v_idx,
        "is_combed_frame": False,
        "speed_ratio": 1.0,
        "audio_pitch_shift_cents": 0.0,
    }


def compute_telecine_audio_effect(pulldown_mode: int, optical_track_wear: float = 0.2) -> dict[str, Any]:
    """
    Return the optical-soundtrack & telecine pitch/tempo telemetry (`模块`).
    """
    mapping = compute_telecine_pull_mapping(0, pulldown_mode)
    return {
        "pulldown_mode": mapping["pulldown_mode"],
        "speed_ratio": mapping["speed_ratio"],
        "pitch_shift_cents": mapping["audio_pitch_shift_cents"],
        "optical_bandwidth_hz": round(float(7800.0 * (1.0 - 0.35 * np.clip(optical_track_wear, 0.0, 1.0))), 1),
        "sprocket_hum_96hz": round(float(0.08 * np.clip(optical_track_wear, 0.0, 1.0)), 4),
    }


def apply_film_telecine(
    rgb: np.ndarray,
    spec: FilmTelecineSpec | PhotochemicalFilmSpec,
    frame_index: int = 0,
    seed: int = 20240101,
    secondary_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Run the complete Film Photochemical + Projector + Rank Cintel Telecine pipeline (`模块`).
    """
    if not spec.enabled:
        return rgb, {"enabled": False}

    base_rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = base_rgb.shape[:2]

    pull_info = compute_telecine_pull_mapping(frame_index, spec.telecine_pulldown)
    f_top = int(pull_info["top_field_film_frame"])
    f_bot = int(pull_info["bottom_field_film_frame"])

    # Film Frame A (for Top Field / Even Scanlines)
    film_top, chem_info = simulate_photochemical_film(base_rgb, spec, frame_index=f_top, seed=seed)
    film_top, proj_top = simulate_film_aging_and_projector(film_top, spec, frame_index=f_top, seed=seed)

    if f_bot != f_top:
        # NTSC 3:2 Pulldown split-field frame (`BC` or `CD`):
        # Bottom field comes from the NEXT film frame `f_bot = f_top + 1`!
        if secondary_rgb is not None and secondary_rgb.shape == base_rgb.shape:
            src_bot = np.clip(np.asarray(secondary_rgb, dtype=np.float32), 0.0, 1.0)
        else:
            shift_x = np.full(h, 3.2 * max(0.5, w / 720.0), dtype=np.float32)
            src_bot = np.stack([horizontal_shift(base_rgb[..., c], shift_x) for c in range(3)], axis=-1)
        film_bot, _ = simulate_photochemical_film(src_bot, spec, frame_index=f_bot, seed=seed)
        film_bot, proj_bot = simulate_film_aging_and_projector(film_bot, spec, frame_index=f_bot, seed=seed)

        # Interlace weave: even scanlines from `film_top`, odd scanlines from `film_bot`
        telecine_out = film_top.copy()
        telecine_out[1::2, :, :] = film_bot[1::2, :, :]
    else:
        telecine_out = film_top
        proj_bot = proj_top

    # Flying-Spot / CCD Telecine horizontal aperture correction & mild highlight log-masking
    lp_tc = lowpass(telecine_out, max(0.6, 1.1 * (w / 720.0)), axis=1)
    telecine_out = np.clip(telecine_out + 0.16 * (telecine_out - lp_tc), 0.0, 1.0)

    aud_info = compute_telecine_audio_effect(spec.telecine_pulldown, spec.scratches_dust)
    telemetry = {
        "enabled": True,
        "stock_id": chem_info["stock_id"],
        "stock_name": chem_info["stock_name"],
        "grain_rms": chem_info["grain_rms"],
        "projector": proj_top,
        "telecine": pull_info,
        "audio": aud_info,
    }

    if spec.telecine_pulldown != 3:
        return telecine_out.astype(np.float32), telemetry

    # Mode 3 (`telecine_pulldown == 3`): 2x2 Cross-Media Historical Lineage Lab
    from ..stages.display import _resample_to

    def _draw_text(canvas: np.ndarray, y0: int, x0: int, text: str,
                   color: tuple[float, float, float], scale: int = 1) -> None:
        from .multiview_wall import _draw_bitmap_text
        _draw_bitmap_text(canvas, text, x0, y0, color, scale=scale)

    shift_tc = np.full(h, 3.0 * max(0.5, w / 720.0), dtype=np.float32)
    src_b2 = np.stack([horizontal_shift(base_rgb[..., c], shift_tc) for c in range(3)], axis=-1)
    f_b2, _ = simulate_photochemical_film(src_b2, spec, frame_index=frame_index + 1, seed=seed)
    f_b2, _ = simulate_film_aging_and_projector(f_b2, spec, frame_index=frame_index + 1, seed=seed)
    tc_combed = film_top.copy()
    tc_combed[1::2, :, :] = f_b2[1::2, :, :]

    sig_vhs = SignalFrame.from_rgb(tc_combed, None)  # type: ignore[arg-type]
    y_v = lowpass(sig_vhs.y, max(1.0, 2.2 * (w / 720.0)), axis=1)
    i_v = lowpass(sig_vhs.i, max(2.8, 6.5 * (w / 720.0)), axis=1)
    q_v = lowpass(sig_vhs.q, max(3.2, 7.5 * (w / 720.0)), axis=1)
    i_v = horizontal_shift(i_v, np.full(h, 2.5, dtype=np.float32))
    q_v = horizontal_shift(q_v, np.full(h, 2.5, dtype=np.float32))
    vhs_rgb = SignalFrame(y_v, i_v, q_v).to_rgb()
    vhs_rgb[-10:, :, :] = np.clip(vhs_rgb[-10:, :, :] * 0.85 + 0.12, 0.0, 1.0)

    from .mpeg1 import MPEG1Codec, MPEG1CodecParams
    mp_codec = MPEG1Codec(MPEG1CodecParams(quality=0.55, width=352, height=240))
    stream_rgb = mp_codec.encode_decode(vhs_rgb, frame_index=0)

    corner = int(np.clip(getattr(spec, "quad_corner", 0), 0, 4))
    b_mode = int(np.clip(getattr(spec, "show_border", 1), 0, 2))
    t_mode = int(np.clip(getattr(spec, "show_text", 0), 0, 2))
    branches = [
        (film_top, f"1.FILM 16/35mm ({chem_info['stock_name'][:18]})", (0.98, 0.86, 0.45)),
        (tc_combed, "2.TELECINE 29.97i (3:2 PULLDOWN BC)", (0.45, 0.95, 0.85)),
        (vhs_rgb, "3.FILM -> TELECINE -> HOME VHS DUB", (0.95, 0.65, 0.40)),
        (stream_rgb, "4.VHS -> DIGITAL STREAM RE-ENCODE", (0.80, 0.75, 0.98)),
    ]

    if corner in (1, 2, 3, 4):
        sel_img, sel_lbl, sel_col = branches[corner - 1]
        canvas = _resample_to(sel_img, h, w).copy()
        if t_mode >= 1:
            canvas[:18, :] *= 0.22
            _draw_text(canvas, 4, 6, sel_lbl, sel_col, scale=1)
    else:
        qh, qw = h // 2, w // 2
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        slots = [
            (0, qh, 0, qw, branches[0]),
            (0, qh, qw, w, branches[1]),
            (qh, h, 0, qw, branches[2]),
            (qh, h, qw, w, branches[3]),
        ]
        for y0, y1, x0, x1, (b_img, b_lbl, b_col) in slots:
            q_tile = _resample_to(b_img, y1 - y0, x1 - x0)
            if t_mode >= 1:
                q_tile[:18, :] *= 0.22
                _draw_text(q_tile, 4, 6, b_lbl, b_col, scale=1)
            canvas[y0:y1, x0:x1] = q_tile
        if b_mode >= 1:
            canvas[qh - 1:qh + 1, :] = np.array([0.16, 0.18, 0.22], dtype=np.float32)
            canvas[:, qw - 1:qw + 1] = np.array([0.16, 0.18, 0.22], dtype=np.float32)

    telemetry["cross_media_lab"] = {
        "film_vs_vhs_l1": round(float(np.mean(np.abs(film_top - vhs_rgb))), 4),
        "vhs_vs_stream_l1": round(float(np.mean(np.abs(vhs_rgb - stream_rgb))), 4),
    }
    return np.clip(canvas, 0.0, 1.0).astype(np.float32), telemetry


class FilmTelecineStage(Stage):
    """Node-graph stage for `film_scanner_telecine` (**模块**)."""
    name = "film_scanner_telecine"
    layers = ("film", "camera")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg_film = self.cfg_of("film")
        cfg_cam = self.cfg_of("camera")
        self.spec = PhotochemicalFilmSpec.from_params(cfg_film, fallback_cam=cfg_cam)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.stock == 0:
                self.spec.stock = 2

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        from .photochemical_film import apply_photochemical_film_pipeline
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        sec_rgb = None
        if getattr(self.ctx, "source", None) is not None:
            try:
                sec_rgb = np.clip(self.ctx.source.frame(int(buf.index) + 1), 0.0, 1.0).astype(np.float32)
            except Exception:
                sec_rgb = None

        out_rgb, info = apply_photochemical_film_pipeline(
            rgb_in,
            self.spec,
            frame_index=int(buf.index),
            seed=self.ctx.seed,
            secondary_rgb=sec_rgb,
        )
        meta = dict(buf.meta)
        meta["film_telecine"] = info
        meta["photochemical_film"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

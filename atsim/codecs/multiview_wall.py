"""
Broadcast Multiviewer / Studio Monitor Wall & Independent Per-CRT Unit Aging Array (`atsim.codecs.multiview_wall`).

Implements Candidate F from the Broadcast Television Workstation architecture:

1. **Multi-Source Studio Monitor Wall (`MultiviewWallSpec`, `render_multiview_wall`, `MultiviewWallStage`)**:
   Composites multiple real-time tap points from the signal chain or node graph onto a broadcast control-room
   monitor wall:
     * `layout_mode = 0` (`2x2` Production Control Room Wall):
       - **MON 1 (`CAM / SRC`)**: Camera / Test Generator reference monitor (`Sony PVM-14L5 Trinitron · D65`);
       - **MON 2 (`VTR RAW · TBC OFF`)**: Direct un-corrected tape playback monitor (`JVC TM-H150CG · Underscan`),
         exposing raw horizontal timebase jitter, dropouts, and bottom head-switching seam;
       - **MON 3 (`TBC / ENG PROC`)**: Time-base corrected & comb-separated engineering monitor
         (`Ikegami TM14 · Aged Green-Phosphor / Silver-White Monochrome Check`);
       - **MON 4 (`PGM ON-AIR`)**: Final Program master output monitor (`Consumer 21" 9300K Slot-Mask TV`).
     * `layout_mode = 1` (`3x2` Master Control Room 6-Monitor Bridge):
       -Simultaneously monitors `TEST GEN`, `CAM-1`, `PATCH / 75Ω DA`, `VTR RAW (TBC OFF)`, `TBC / PVW`, and `PGM AIR`.
     * `layout_mode = 2` (`1+3` Director's Asymmetric Wall):
       - Large left `PGM ON-AIR` master monitor (`2/3` width) + 3 stacked right-hand preview monitors (`CAM`, `VTR`, `TBC`).

2. **Independent Per-Monitor CRT Calibration & Aging Divergence (`mv_unit_variance`)**:
   Each monitor on the wall has its own physical CRT tube characteristics (phosphor colour temperature `5400K / 6500K /
   9300K`, cathode aging green/warm cast, shadow-mask pitch, scanline thickness, overscan vs underscan, and EHT
   breathing). Crucially, a monitor's calibration offset changes *what the operator sees on that monitor* without
   altering the underlying video signal passed to downstream equipment.

3. **Physical Signal-Loss & Unlock Responses per Monitor (`mv_fault_channel`, `mv_fault_mode`)**:
   When a monitor's input feed is lost or faulted, it never displays a generic software error popup. Instead, it
   renders the physical hardware response of that receiver/monitor:
     * `1 = black_no_sync`: Floating DC black raster with subtle thermal cathode floor;
     * `2 = rf_snow`: Open-tuner high-gain IF demodulator snow + rolling blanking bar;
     * `3 = freerun_roll`: Unlocked H-Hold / V-Hold free-running oscillator diagonal tear & vertical roll;
     * `4 = frame_freeze`: Digital TBC / frame-synchronizer last-field freeze with interlace stairstepping;
     * `5 = vtr_blue_osd`: Deck character-generator blue screen (`NO SIGNAL`).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass
from ..core.standards import VideoStandard, get_standard
def _resample_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    sy = np.clip((np.arange(height, dtype=np.float32) + 0.5) * (h / height) - 0.5, 0.0, h - 1.0)
    sx = np.clip((np.arange(width, dtype=np.float32) + 0.5) * (w / width) - 0.5, 0.0, w - 1.0)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = (sy - y0)[:, None, None]
    fx = (sx - x0)[None, :, None]
    top = img[y0][:, x0] * (1.0 - fx) + img[y0][:, x1] * fx
    bot = img[y1][:, x0] * (1.0 - fx) + img[y1][:, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

FAULT_MODE_NAMES = {
    0: "auto_by_monitor",
    1: "black_no_sync",
    2: "rf_snow",
    3: "freerun_roll",
    4: "frame_freeze",
    5: "vtr_blue_osd",
}

# Pre-defined bitmap glyphs (5x7) for hardware UMD (Under-Monitor Display) character cells
_FONT_5X7: dict[str, tuple[int, ...]] = {
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x06, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    ":": (0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06),
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
}


@dataclass
class MultiviewWallSpec:
    enabled: bool = False
    layout_mode: int = 0        # 0=2x2 Quad Wall, 1=3x2 Six-Monitor MCR Bridge, 2=1+3 Director Wall
    unit_variance: float = 0.45  # 0..1 independent CRT calibration & aging spread across monitors
    fault_channel: int = 0      # 0=None, 1..6=inject signal loss/unlock on monitor #k
    fault_mode: int = 0         # 0=Auto per monitor, 1=Black/NoSync, 2=RF Snow, 3=FreeRun Roll, 4=Freeze, 5=Blue OSD
    tally_umd: bool = False     # Draw Tally LEDs and UMD text labels (default False = no text)
    quad_corner: int = 0        # 0=Multiview Wall, 1..6=Isolate individual monitor cell #1..#6 full screen
    show_border: int = 1        # 0=No Border (Seamless), 1=Thin Crosshair Border (Default), 2=Studio Rack Bezel

    @classmethod
    def from_params(cls, cfg: dict | None) -> "MultiviewWallSpec":
        c = cfg or {}
        en = float(c.get("mv_enabled", 0.0)) > 0.5
        lay = int(np.clip(round(float(c.get("mv_layout", 0.0))), 0, 2))
        var = float(np.clip(float(c.get("mv_unit_variance", 0.45)), 0.0, 1.0))
        f_ch = int(np.clip(round(float(c.get("mv_fault_channel", 0.0))), 0, 6))
        f_md = int(np.clip(round(float(c.get("mv_fault_mode", 0.0))), 0, 5))
        umd = float(c.get("mv_tally_umd", 0.0)) > 0.5
        q_c = int(np.clip(round(float(c.get("mv_quad_corner", 0.0))), 0, 6))
        b_m = int(np.clip(round(float(c.get("mv_show_border", 1.0))), 0, 2))
        return cls(
            enabled=en,
            layout_mode=lay,
            unit_variance=var,
            fault_channel=f_ch,
            fault_mode=f_md,
            tally_umd=umd,
            quad_corner=q_c,
            show_border=b_m,
        )

    def is_active(self) -> bool:
        return bool(self.enabled)


@dataclass
class MonitorCellProfile:
    """Physical CRT unit identity for one monitor slot in the studio monitor wall."""
    slot_index: int
    model_name: str
    rgb_gain: tuple[float, float, float]
    black_lift: float
    gamma: float
    scanline_depth: float
    overscan: float          # positive = overscan crop, negative = underscan border (VTR check monitor)
    monochrome_mix: float    # >0 for silver-white monochrome check tube
    default_fault_mode: int  # 1=black, 2=rf_snow, 3=freerun_roll, 4=frame_freeze, 5=vtr_blue_osd


def get_monitor_profile(slot_idx: int, variance: float) -> MonitorCellProfile:
    """
    Return the physical CRT personality for monitor slot `slot_idx` (`0..5`), scaled by `variance`.
    At `variance == 0.0`, all monitors are identically D65-calibrated.
    At `variance > 0.0`, each monitor exhibits its authentic studio role & aging signature.
    """
    v = float(np.clip(variance, 0.0, 1.0))
    profiles = (
        # Slot 0: Sony PVM-14L5 Reference Trinitron (D65 calibrated, tight aperture grille, slight overscan)
        MonitorCellProfile(
            slot_index=0, model_name="PVM-14L5 D65",
            rgb_gain=(1.0 - 0.02 * v, 1.0 + 0.01 * v, 1.0 + 0.02 * v),
            black_lift=0.005 * v, gamma=1.0 + 0.04 * v,
            scanline_depth=0.18 * v, overscan=0.015 * v, monochrome_mix=0.0,
            default_fault_mode=1,
        ),
        # Slot 1: JVC TM-H150CG VTR Raw Bridge Monitor (Underscanned to show head-switch & HBI, warm 5400K)
        MonitorCellProfile(
            slot_index=1, model_name="JVC VTR-UNDERSCAN",
            rgb_gain=(1.0 + 0.14 * v, 1.0 + 0.03 * v, 1.0 - 0.16 * v),
            black_lift=0.025 * v, gamma=1.0 - 0.06 * v,
            scanline_depth=0.26 * v, overscan=-0.045 * v, monochrome_mix=0.0,
            default_fault_mode=3,
        ),
        # Slot 2: Ikegami TM14 Aged Engineering Tube (Green cathode drift + high-contrast phosphor)
        MonitorCellProfile(
            slot_index=2, model_name="IKEGAMI TBC-ENG",
            rgb_gain=(1.0 - 0.14 * v, 1.0 + 0.16 * v, 1.0 - 0.22 * v),
            black_lift=0.038 * v, gamma=1.0 + 0.10 * v,
            scanline_depth=0.30 * v, overscan=0.02 * v, monochrome_mix=0.18 * v,
            default_fault_mode=4,
        ),
        # Slot 3: Consumer 21" 9300K Slot-Mask ON-AIR Receiver (Cool 9300K blue push + 5% TV overscan + scanlines)
        MonitorCellProfile(
            slot_index=3, model_name="PGM-AIR 9300K",
            rgb_gain=(1.0 - 0.15 * v, 1.0 - 0.01 * v, 1.0 + 0.24 * v),
            black_lift=0.015 * v, gamma=1.0 + 0.14 * v,
            scanline_depth=0.34 * v, overscan=0.045 * v, monochrome_mix=0.0,
            default_fault_mode=2,
        ),
        # Slot 4: Patchbay / DA Preview Monitor (Slight magenta-warm aging)
        MonitorCellProfile(
            slot_index=4, model_name="PVW-PATCH",
            rgb_gain=(1.0 + 0.10 * v, 1.0 - 0.08 * v, 1.0 + 0.08 * v),
            black_lift=0.020 * v, gamma=1.0 + 0.05 * v,
            scanline_depth=0.22 * v, overscan=0.01 * v, monochrome_mix=0.0,
            default_fault_mode=5,
        ),
        # Slot 5: Silver-White Monochrome Precision Waveform/Focus Check Tube
        MonitorCellProfile(
            slot_index=5, model_name="B&W-MONO CHECK",
            rgb_gain=(0.96, 1.00, 1.04),
            black_lift=0.010 * v, gamma=1.0,
            scanline_depth=0.20 * v, overscan=0.0, monochrome_mix=0.85 * v,
            default_fault_mode=1,
        ),
    )
    return profiles[slot_idx % len(profiles)]


def render_signal_loss_state(
    rgb: np.ndarray,
    fault_mode: int,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, str]:
    """
    Synthesize the physical hardware behavior of a monitor when its input signal is lost or unlocked,
    without ever throwing a software popup error.
    """
    h, w = rgb.shape[:2]
    rng = np.random.default_rng((int(seed) * 1664525 + int(frame_index) * 1013904223 + int(fault_mode) * 97) & 0xFFFFFFFF)
    mode = int(np.clip(fault_mode, 1, 5))

    if mode == 1:
        # 1. Black / No-Sync Floating Bias Raster (unterminated AC-coupled video amp floats slightly above black)
        hum_y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        floating = 0.035 + 0.018 * np.sin(2.0 * np.pi * (hum_y + frame_index * 0.12))
        noise = rng.standard_normal((h, w, 3)).astype(np.float32) * 0.014
        out = np.clip(floating[..., None] + noise, 0.0, 1.0).astype(np.float32)
        return out, "NO SYNC · BLACK"

    if mode == 2:
        # 2. Open-Tuner RF Snow (high-gain IF AGC at maximum gain producing broadband snow + rolling hum bar)
        snow_y = rng.standard_normal((h, w)).astype(np.float32)
        snow_y = lowpass(snow_y, 1.1, axis=1)
        yy = np.arange(h, dtype=np.float32)[:, None]
        roll_bar = 0.5 + 0.5 * np.cos(2.0 * np.pi * (yy / max(8.0, h * 0.45) - frame_index * 0.17))
        luma = np.clip(0.46 + 0.34 * snow_y + 0.12 * roll_bar, 0.0, 1.0)
        chroma_n = rng.standard_normal((h, w, 3)).astype(np.float32) * 0.08
        out = np.clip(luma[..., None] + chroma_n, 0.0, 1.0).astype(np.float32)
        return out, "RF OPEN · SNOW"

    if mode == 3:
        # 3. Unlocked H-Hold / V-Hold Free-Running Oscillator Roll (diagonal shear + VBI bar rolling across screen)
        roll_rows = int((frame_index * 11 + h // 3) % max(1, h))
        rolled = np.roll(np.clip(rgb, 0.0, 1.0), roll_rows, axis=0)
        # Diagonal horizontal free-run shear (2.5 cycles per field)
        rows = np.arange(h, dtype=np.float32)
        shear = (np.mod(rows * 2.3 + frame_index * 17.0, float(w)) - w * 0.5) * 0.65
        out = horizontal_shift(rolled, shear)
        # Draw the physical VBI sync bar at `roll_rows`
        vbi_h = max(4, h // 12)
        r0 = roll_rows % h
        r1 = min(h, r0 + vbi_h)
        out[r0:r1, :] = 0.04
        if r1 - r0 >= 2:
            out[r0 + (r1 - r0) // 2, ::4] = 0.85  # VBI VITC / equalizing pulse teeth
        return np.clip(out, 0.0, 1.0).astype(np.float32), "H/V UNLOCK ROLL"

    if mode == 4:
        # 4. Digital TBC / Frame-Synchronizer Field Freeze (repeats single field with static dropout & comb step)
        even_rows = np.clip(rgb[0::2, :, :], 0.0, 1.0)
        frozen = np.repeat(even_rows, 2, axis=0)[:h, :w, :]
        # Slightly quantized 6-bit field store look
        frozen = np.round(frozen * 32.0) / 32.0
        return np.clip(frozen, 0.0, 1.0).astype(np.float32), "TBC FIELD FREEZE"

    # 5. VTR / Deck Blue Screen OSD ("NO SIGNAL")
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[..., 0] = 0.06
    out[..., 1] = 0.14
    out[..., 2] = 0.78
    _draw_bitmap_text(out, "NO SIGNAL", max(4, w // 3), max(4, h // 2 - 4), color=(0.95, 0.96, 0.95), scale=max(1, h // 90))
    return out, "VTR BLUE MUTE"


def _draw_bitmap_text(
    canvas: np.ndarray,
    text: str,
    x0: int,
    y0: int,
    color: tuple[float, float, float] = (0.92, 0.95, 0.92),
    scale: int = 1,
) -> None:
    """Render crisp 5x7 dot-matrix hardware UMD characters onto `canvas` in-place."""
    h, w = canvas.shape[:2]
    sc = max(1, int(scale))
    col_arr = np.array(color, dtype=np.float32)
    cur_x = int(x0)
    for ch in str(text).upper():
        rows_bits = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for ry, bits in enumerate(rows_bits):
            py0 = y0 + ry * sc
            py1 = min(h, py0 + sc)
            if py0 < 0 or py0 >= h:
                continue
            for rx in range(5):
                if (bits >> (4 - rx)) & 1:
                    px0 = cur_x + rx * sc
                    px1 = min(w, px0 + sc)
                    if 0 <= px0 < w:
                        canvas[py0:py1, px0:px1] = col_arr
        cur_x += 6 * sc
        if cur_x >= w - 4:
            break


def render_monitor_cell(
    rgb: np.ndarray,
    cell_w: int,
    cell_h: int,
    slot_idx: int,
    label: str,
    tally_state: str,
    spec: MultiviewWallSpec,
    frame_index: int = 0,
    seed: int = 20240101,
    is_faulted: bool = False,
    fault_mode_override: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Render a single studio CRT monitor cell of exact shape `(cell_h, cell_w, 3)` with its own
    independent CRT calibration/aging profile, hardware bezel, Tally LED, and UMD source label.
    """
    prof = get_monitor_profile(slot_idx, spec.unit_variance)
    b_mode = int(np.clip(getattr(spec, "show_border", 1), 0, 2))
    if b_mode == 0 and not spec.tally_umd:
        pad = 0
    elif b_mode == 1 and not spec.tally_umd:
        pad = 1
    else:
        pad = max(2, min(cell_w, cell_h) // 36)
    umd_h = max(12, min(22, cell_h // 9)) if spec.tally_umd else 0
    inner_h = max(8, cell_h - 2 * pad - umd_h)
    inner_w = max(8, cell_w - 2 * pad)

    src_resized = _resample_to(np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0), inner_h, inner_w)

    fault_tag = "OK"
    if is_faulted:
        f_mode = fault_mode_override if (fault_mode_override is not None and fault_mode_override > 0) else (
            spec.fault_mode if spec.fault_mode > 0 else prof.default_fault_mode
        )
        scr, fault_tag = render_signal_loss_state(src_resized, f_mode, frame_index=frame_index, seed=seed + slot_idx * 31)
    else:
        scr = src_resized.copy()

    # Apply this monitor's independent CRT calibration & aging (does NOT touch upstream signal!)
    if abs(prof.overscan) > 1e-3:
        if prof.overscan > 0:
            # Overscan crop
            sc_up = 1.0 / max(0.7, 1.0 - 2.0 * prof.overscan)
            bh = max(inner_h, int(round(inner_h * sc_up)))
            bw = max(inner_w, int(round(inner_w * sc_up)))
            big = _resample_to(scr, bh, bw)
            y0_c = (bh - inner_h) // 2
            x0_c = (bw - inner_w) // 2
            scr = big[y0_c:y0_c + inner_h, x0_c:x0_c + inner_w]
        else:
            # Underscan border (exposes full active raster + HBI/VBI edge for VTR inspection)
            shrink = max(0.75, 1.0 + 2.0 * prof.overscan)
            sh = max(4, int(round(inner_h * shrink)))
            sw = max(4, int(round(inner_w * shrink)))
            small = _resample_to(scr, sh, sw)
            framed = np.full_like(scr, 0.025)
            y0_c = (inner_h - sh) // 2
            x0_c = (inner_w - sw) // 2
            # Subtle HBI/VBI sync frame border around the underscanned raster
            framed[max(0, y0_c - 1):min(inner_h, y0_c + sh + 1), max(0, x0_c - 1):min(inner_w, x0_c + sw + 1)] = 0.14
            framed[y0_c:y0_c + sh, x0_c:x0_c + sw] = small
            scr = framed

    # Per-unit RGB gun gain, black lift, gamma & optional silver-white monochrome mix
    gains = np.array(prof.rgb_gain, dtype=np.float32)[None, None, :]
    scr = np.clip(scr * gains + prof.black_lift, 0.0, 1.0)
    if abs(prof.gamma - 1.0) > 1e-3:
        scr = np.power(np.clip(scr, 0.0, 1.0), 1.0 / prof.gamma).astype(np.float32)
    if prof.monochrome_mix > 1e-3:
        lum = (scr @ LUMA_W)[..., None]
        p4_tint = np.array([0.95, 1.00, 1.05], dtype=np.float32)[None, None, :]
        scr = scr * (1.0 - prof.monochrome_mix) + (lum * p4_tint) * prof.monochrome_mix

    # Per-unit scanline grating & subtle CRT corner vignette
    if prof.scanline_depth > 1e-3:
        yy_s = np.arange(inner_h, dtype=np.float32)[:, None, None]
        scan_mod = 1.0 - prof.scanline_depth * (0.5 - 0.5 * np.cos(2.0 * np.pi * yy_s / 2.0))
        scan_mod /= max(1e-6, float(scan_mod.mean()))
        scr = np.clip(scr * scan_mod, 0.0, 1.0)

    # Assemble into studio rack housing (or seamless/thin border depending on show_border)
    if pad == 0:
        cell = scr.copy()
    else:
        cell = np.full((cell_h, cell_w, 3), 0.075, dtype=np.float32)
        if b_mode >= 1:
            cell[0, :] = 0.16 if b_mode == 1 else 0.22
            cell[-1, :] = 0.06 if b_mode == 1 else 0.04
            cell[:, 0] = 0.16 if b_mode == 1 else 0.20
            cell[:, -1] = 0.06 if b_mode == 1 else 0.04
        cell[pad:pad + inner_h, pad:pad + inner_w] = scr

    if spec.tally_umd and umd_h >= 10:
        umd_y0 = pad + inner_h
        cell[umd_y0:cell_h - 1, pad:pad + inner_w] = np.array([0.04, 0.045, 0.055], dtype=np.float32)
        # Tally LED lamp on the top-left of the bezel & left of UMD bar
        if is_faulted:
            tally_col = (0.98, 0.18, 0.12) if (frame_index % 2 == 0) else (0.45, 0.08, 0.06)
        elif tally_state == "PGM":
            tally_col = (0.96, 0.14, 0.14)  # Red ON-AIR Tally
        elif tally_state == "PVW":
            tally_col = (0.15, 0.88, 0.32)  # Green Preview Tally
        else:
            tally_col = (0.95, 0.68, 0.16)  # Amber VTR / Engineering Tally

        led_w = max(8, inner_w // 12)
        cell[umd_y0 + 2:cell_h - 2, pad + 2:pad + 2 + led_w] = np.array(tally_col, dtype=np.float32)
        # Top bezel Tally strip
        tally_strip_w = max(18, inner_w // 4)
        tx0 = pad + (inner_w - tally_strip_w) // 2
        cell[1:max(2, pad), tx0:tx0 + tally_strip_w] = np.array(tally_col, dtype=np.float32)

        # UMD text (`CH1 CAM-1` + status or model)
        font_sc = 2 if (umd_h >= 16 and inner_w >= 210) else 1
        ty0 = umd_y0 + max(1, (umd_h - 7 * font_sc) // 2)
        umd_txt = f"{label}" if not is_faulted else f"{label} {fault_tag}"
        txt_col = (0.95, 0.32, 0.28) if is_faulted else (0.72, 0.94, 0.78)
        _draw_bitmap_text(cell, umd_txt[:22], pad + led_w + 6, ty0, color=txt_col, scale=font_sc)

    info = {
        "slot": slot_idx + 1,
        "label": label,
        "model": prof.model_name,
        "tally": "FAULT" if is_faulted else tally_state,
        "faulted": bool(is_faulted),
        "fault_tag": fault_tag,
        "mean_luma": round(float((scr @ LUMA_W).mean()), 4),
        "r_over_b": round(float(scr[..., 0].mean() / max(1e-4, scr[..., 2].mean())), 4),
    }
    return cell, info


def render_multiview_wall(
    feeds: list[dict],
    spec: MultiviewWallSpec,
    width: int = 720,
    height: int = 480,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict]:
    """
    Composite multiple monitor feeds onto a `2x2` (`layout_mode=0`), `3x2` (`layout_mode=1`),
    or `1+3` (`layout_mode=2`) studio monitor wall.

    Each item in `feeds` is a dict:
      `{"label": str, "rgb": np.ndarray, "tally": "PGM"|"PVW"|"AUX", "faulted": bool}`
    """
    h, w = max(32, int(height)), max(32, int(width))
    wall = np.full((h, w, 3), 0.05, dtype=np.float32)
    cell_reports: list[dict] = []

    lay = int(np.clip(spec.layout_mode, 0, 2))
    if lay == 1:
        # 3x2 Six-Monitor MCR Bridge
        rows, cols = 2, 3
        n_slots = 6
    else:
        # 2x2 Quad Monitor Wall (or 1+3 Director Wall)
        rows, cols = 2, 2
        n_slots = 4

    # Pad feeds if fewer than n_slots were supplied
    padded_feeds = list(feeds[:n_slots])
    while len(padded_feeds) < n_slots:
        fallback_rgb = padded_feeds[-1]["rgb"] if padded_feeds else np.zeros((h // 2, w // 2, 3), dtype=np.float32)
        idx_pad = len(padded_feeds)
        padded_feeds.append({
            "label": f"CH{idx_pad + 1} AUX-{idx_pad + 1}",
            "rgb": fallback_rgb,
            "tally": "AUX",
            "faulted": False,
        })

    corner_sel = int(np.clip(getattr(spec, "quad_corner", 0), 0, n_slots))
    if corner_sel > 0:
        idx_s = corner_sel - 1
        fd = padded_feeds[idx_s]
        is_f = bool(fd.get("faulted", False)) or (spec.fault_channel == idx_s + 1)
        c_img, rep = render_monitor_cell(
            fd["rgb"], w, h, slot_idx=idx_s,
            label=fd.get("label", f"CH{idx_s + 1}"),
            tally_state=fd.get("tally", "PGM" if idx_s == n_slots - 1 else "PVW"),
            spec=spec, frame_index=frame_index, seed=seed, is_faulted=is_f,
        )
        wall[:, :] = c_img
        cell_reports.append(rep)
    elif lay == 2:
        # 1+3 Director's Wall: Left 2/3 is large PGM monitor (feed[-1]), Right 1/3 has 3 stacked preview monitors
        w_left = (w * 2) // 3
        w_right = w - w_left
        pgm_feed = padded_feeds[3]
        f_pgm = bool(pgm_feed.get("faulted", False)) or (spec.fault_channel == 4)
        c_pgm, rep_pgm = render_monitor_cell(
            pgm_feed["rgb"], w_left, h, slot_idx=3,
            label=pgm_feed.get("label", "CH4 PGM-AIR"),
            tally_state="PGM", spec=spec,
            frame_index=frame_index, seed=seed, is_faulted=f_pgm,
        )
        wall[:, :w_left] = c_pgm
        cell_reports.append(rep_pgm)

        h_sub = h // 3
        for idx_r in range(3):
            y0 = idx_r * h_sub
            y1 = h if idx_r == 2 else (idx_r + 1) * h_sub
            fd = padded_feeds[idx_r]
            is_f = bool(fd.get("faulted", False)) or (spec.fault_channel == idx_r + 1)
            c_sub, rep_sub = render_monitor_cell(
                fd["rgb"], w_right, y1 - y0, slot_idx=idx_r,
                label=fd.get("label", f"CH{idx_r + 1}"),
                tally_state=fd.get("tally", "PVW"), spec=spec,
                frame_index=frame_index, seed=seed, is_faulted=is_f,
            )
            wall[y0:y1, w_left:] = c_sub
            cell_reports.append(rep_sub)
    else:
        cell_h = h // rows
        cell_w = w // cols
        for idx_s in range(n_slots):
            r_i = idx_s // cols
            c_i = idx_s % cols
            y0 = r_i * cell_h
            y1 = h if r_i == rows - 1 else (r_i + 1) * cell_h
            x0 = c_i * cell_w
            x1 = w if c_i == cols - 1 else (c_i + 1) * cell_w
            fd = padded_feeds[idx_s]
            is_f = bool(fd.get("faulted", False)) or (spec.fault_channel == idx_s + 1)
            c_img, rep = render_monitor_cell(
                fd["rgb"], x1 - x0, y1 - y0, slot_idx=idx_s,
                label=fd.get("label", f"CH{idx_s + 1}"),
                tally_state=fd.get("tally", "PGM" if idx_s == n_slots - 1 else "PVW"),
                spec=spec, frame_index=frame_index, seed=seed, is_faulted=is_f,
            )
            wall[y0:y1, x0:x1] = c_img
            cell_reports.append(rep)

    return np.clip(wall, 0.0, 1.0).astype(np.float32), {
        "layout_mode": lay,
        "layout_name": {0: "2x2_quad", 1: "3x2_six_bridge", 2: "1plus3_director"}.get(lay, "2x2_quad"),
        "unit_variance": round(float(spec.unit_variance), 3),
        "fault_channel": int(spec.fault_channel),
        "fault_mode": FAULT_MODE_NAMES.get(int(spec.fault_mode), "auto"),
        "monitors": cell_reports,
    }


class MultiviewWallStage(Stage):
    """
    Standalone Node-Graph Sink Stage (`multiview_wall`).
    Accepts up to 4 inputs (`in`, `in_b`, `in_c`, `in_d`) on the node graph and renders a real-time
    broadcast monitor wall with independent per-CRT aging and physical signal-loss responses.
    """
    name = "multiview_wall"
    layers = ("display",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        d = dict(self.cfg_of("display") or {})
        d["mv_enabled"] = 1.0
        self.spec = MultiviewWallSpec.from_params(d)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        ch1_rgb = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0)
        h, w = ch1_rgb.shape[:2]

        # Gather multi-input feeds injected by GraphExecutor (`mv_feeds`) or synthesize tap comparison
        mv_feeds = buf.meta["mv_feeds"] if "mv_feeds" in buf.meta else None
        if mv_feeds and isinstance(mv_feeds, list) and len(mv_feeds) >= 2:
            feeds = mv_feeds
        else:
            feeds = [
                {"label": "CH1 CAM/SRC", "rgb": ch1_rgb, "tally": "PVW", "faulted": False},
                {"label": "CH2 VTR-RAW", "rgb": ch1_rgb, "tally": "AUX", "faulted": False},
                {"label": "CH3 TBC-OUT", "rgb": ch1_rgb, "tally": "PVW", "faulted": False},
                {"label": "CH4 PGM-AIR", "rgb": ch1_rgb, "tally": "PGM", "faulted": False},
            ]

        wall_rgb, wall_info = render_multiview_wall(
            feeds, self.spec, width=w, height=h,
            frame_index=int(buf.index), seed=int(self.ctx.seed) + 83,
        )
        meta = dict(buf.meta)
        meta["multiview_wall"] = wall_info
        meta["signal"] = SignalFrame.from_rgb(wall_rgb, buf.std)
        return FrameBuffer(wall_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

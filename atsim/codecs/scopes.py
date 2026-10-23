"""
Candidate D (Module C): Broadcast Waveform Monitor (IRE Scope), Chroma Vectorscope,
and Node-Level A/B Difference Heatmap & Spectrum Analyzer.

Implements the three core physical broadcast measurement & calibration instruments:

1. **Broadcast Luminance / Composite IRE Waveform Monitor (`compute_waveform_monitor`)**:
   - Converts active video & synthesized Horizontal Blanking Interval (HBI) into standard
     broadcast IRE units (`-40 IRE .. +110 IRE`, where `140 IRE = 1.0 Vpp`, `0 IRE` = blanking,
     `-40 IRE` = sync tip, `7.5 IRE` = NTSC setup / `0 IRE` = PAL black floor, `100 IRE` = peak white).
   - Synthesizes the complete HBI waveform at the left of the scanline:
     * Front porch (`0 IRE`),
     * Horizontal sync tip (`-40 IRE`, dynamically compressed by `sync_clip` and shared power-bus
       voltage sag `battery_sag`),
     * Back porch (`0 IRE`) with 8–10 cycles of phase-accurate **Colour Burst** (`±20 IRE` peak-to-peak
       envelope at `3.58 MHz` 180° for NTSC or `4.43 MHz` ±135° swinging burst for PAL, rotated by
       `burst_phase_error` and scaled by ACC `chroma_gain`).
   - Computes both full-frame 2D persistence density histograms (Composite/Luma IRE and RGB Parade)
     and a high-resolution single-scanline oscilloscope trace (`line_y`) for line-by-line inspection
     of H-Sync crushing, pedestal lift (`switcher_pedestal_add` / `luma_pedestal`), and highlight clipping.

2. **Broadcast Chroma Vectorscope (`compute_vectorscope`)**:
   - Projects frame chroma onto the broadcast polar vector graticule `(U, V)` / `(I, Q)`, normalized
     so 100% saturation maps to the outer circle (`R = 1.0`).
   - Computes exact 75% and 100% standard SMPTE / EBU colour-bar target box coordinates for all six
     primaries & complements (`R, Mg, B, Cy, G, Yl`) plus `-I` and `+Q` axes:
     * `Yl` (167.1°), `R` (103.4°), `Mg` (60.6°), `B` (347.1°), `Cy` (283.4°), `G` (240.6°).
   - Tracks the live Colour Burst vector (`180°` at `20 IRE` nominal for NTSC; `135°` / `225°` dual
     swinging vectors for PAL) and directly measures:
     * `burst_angle_deg` & `hue_rotation_deg` (induced by `burst_phase_error` or switcher hot cuts),
     * `acc_gain_measured` (chroma amplitude relative to nominal),
     * `hanover_split_deg` (line-to-line alternating phase split in PAL when burst phase or delay-line
       comb lags).

3. **Node-Level / Chain A/B Difference Heatmap & Horizontal Power Spectrum (`compute_ab_diff_and_spectrum`)**:
   - Compares any two frames (`sim_rgb` vs `ref_rgb`, or `node_a` vs `node_b` in the node graph):
     * Computes Luma ΔY, Chroma ΔC, combined error heatmap, L1 error, RMSE, and PSNR (dB).
     * Computes the horizontal 1D FFT power spectral density (`0.0 .. 6.5 MHz`) for both Reference
       and Simulated signals (plus Composite spectrum), directly revealing VHS/Hi8 luma bandwidth
       roll-off (`luma_bandwidth` in MHz) and the `3.58 MHz` (NTSC) / `4.43 MHz` (PAL) chroma
       subcarrier peak.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.signal import rgb_to_yiq
from ..core.standards import VideoStandard, get_standard

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Standard broadcast vectorscope angles (degrees, counter-clockwise from +U / 0 deg = right,
# where Colour Burst sits at 180 deg = left) and normalized 75% / 100% radii for SMPTE/EBU bars:
VECTORSCOPE_TARGETS: dict[str, dict[str, float]] = {
    "R":  {"label": "R (红)",   "angle_deg": 103.5, "r_75": 0.632, "r_100": 0.843},
    "Mg": {"label": "Mg (品)",  "angle_deg":  60.7, "r_75": 0.591, "r_100": 0.788},
    "B":  {"label": "B (蓝)",   "angle_deg": 347.1, "r_75": 0.447, "r_100": 0.596},
    "Cy": {"label": "Cy (青)",  "angle_deg": 283.5, "r_75": 0.632, "r_100": 0.843},
    "G":  {"label": "G (绿)",   "angle_deg": 240.7, "r_75": 0.591, "r_100": 0.788},
    "Yl": {"label": "Yl (黄)",  "angle_deg": 167.1, "r_75": 0.447, "r_100": 0.596},
}


def _extract_scope_params(params: dict | None, std: VideoStandard) -> dict[str, float]:
    """Extract physical sync, burst, pedestal, and bandwidth parameters from resolved layer dict."""
    p = params or {}
    fmt = p.get("format") or {}
    ply = p.get("player") or {}
    tpe = p.get("tape") or {}
    con = p.get("connector") or {}

    is_pal = str(getattr(std, "color_system", "NTSC")).upper() == "PAL"
    nominal_setup_ire = 0.0 if is_pal else 7.5

    sync_clip = float(ply.get("sync_clip", 0.0))
    battery_level = float(con.get("battery_level", 1.0))
    battery_sag = float(con.get("battery_sag", 0.0))
    # Sync tip is -40 IRE nominal; sync_clip and severe battery sag compress it toward 0 IRE
    from .patchbay_da import PatchbayDASpec, compute_termination_physics
    da_spec = PatchbayDASpec.from_params(con)
    da_phys = compute_termination_physics(da_spec)
    term_gv = float(da_phys["voltage_gain"]) if da_spec.is_active() else 1.0
    sag_atten = max(0.0, (1.0 - battery_level) * battery_sag * 0.35)
    sync_atten = float(np.clip((1.0 - 0.68 * sync_clip - sag_atten) * term_gv, 0.12, 1.98))
    sync_tip_ire = float(np.clip(-40.0 * sync_atten, -79.0, -5.0))

    burst_phase_err_deg = float(fmt.get("burst_phase_error", 0.0)) + float(ply.get("burst_phase_error", 0.0)) * 0.5
    chroma_gain = float(fmt.get("chroma_gain", 1.0))
    luma_bw_mhz = float(fmt.get("luma_bandwidth", 3.0))
    chroma_bw_mhz = float(fmt.get("chroma_bandwidth", 0.75))

    pedestal_offset_ire = float(tpe.get("luma_pedestal", 0.0)) * 100.0 + float(ply.get("output_bias", 0.0)) * 100.0
    if int(round(float(con.get("switcher_mode", 0.0)))) == 2:
        mix = float(con.get("switcher_mix", 0.0))
        ped_add = float(con.get("switcher_pedestal_add", 0.5))
        pedestal_offset_ire += 7.5 * ped_add * (1.0 - abs(2.0 * mix - 1.0))

    return {
        "is_pal": 1.0 if is_pal else 0.0,
        "nominal_setup_ire": nominal_setup_ire,
        "sync_tip_ire": sync_tip_ire,
        "sync_atten": sync_atten,
        "burst_phase_err_deg": burst_phase_err_deg,
        "chroma_gain": chroma_gain,
        "luma_bw_mhz": luma_bw_mhz,
        "chroma_bw_mhz": chroma_bw_mhz,
        "pedestal_offset_ire": pedestal_offset_ire,
        "termination_ohm": float(da_phys["r_load_ohm"]),
        "termination_gain_db": float(da_phys["gain_db"]),
        "termination_gv": float(term_gv),
    }


def compute_waveform_monitor(
    rgb: np.ndarray,
    std: VideoStandard | None = None,
    params: dict | None = None,
    line_y: float | int | None = None,
    cols_x: int = 192,
    bins_ire: int = 120,
) -> dict[str, Any]:
    """
    Compute Broadcast Waveform Monitor (IRE Scope) telemetry & 2D phosphor density map.

    IRE Scale: `-45 IRE` to `+115 IRE` (spanning `-40 IRE` sync tip, `0 IRE` blanking,
    `7.5 IRE` NTSC black setup, and `100 IRE` reference white).
    The left 16% of the horizontal trace represents the Horizontal Blanking Interval (HBI):
      - `0.00 .. 0.025`: Front porch (`0 IRE`)
      - `0.025 .. 0.085`: Horizontal Sync Pulse (`sync_tip_ire`, nominally `-40 IRE`)
      - `0.085 .. 0.150`: Back porch (`0 IRE`) carrying 9 cycles of Colour Burst (`±20 IRE`)
      - `0.150 .. 1.000`: Active scanline video (`setup_ire .. 100 IRE` + modulated chroma envelope)
    """
    if std is None:
        std = get_standard("ntsc")
    sp = _extract_scope_params(params, std)
    arr = np.clip(np.asarray(rgb, dtype=np.float32), -0.2, 1.25)
    h, w = arr.shape[:2]

    # Select scanline index
    if line_y is None:
        sel_row = h // 2
    elif isinstance(line_y, float) and 0.0 <= line_y <= 1.0:
        sel_row = int(np.clip(round(line_y * (h - 1)), 0, h - 1))
    else:
        sel_row = int(np.clip(int(line_y), 0, h - 1))

    y_luma, i_ch, q_ch = rgb_to_yiq(arr)
    setup_ire = sp["nominal_setup_ire"]
    ped_ire = sp["pedestal_offset_ire"]

    # Active video luma in IRE: 0.0 -> setup_ire + ped_ire, 1.0 -> 100.0 + ped_ire
    span_ire = 100.0 - setup_ire
    luma_ire = setup_ire + ped_ire + y_luma * span_ire

    # Modulated composite chroma carrier amplitude in IRE
    chroma_amp_ire = np.hypot(i_ch, q_ch) * 0.63 * 100.0 * max(0.2, min(2.5, sp["chroma_gain"]))
    phase_grid = np.arange(w, dtype=np.float32)[None, :] * std.subcarrier_rad_per_sample + (
        np.arange(h, dtype=np.float32)[:, None] * np.pi
    )
    comp_ire = luma_ire + chroma_amp_ire * np.cos(phase_grid + np.deg2rad(sp["burst_phase_err_deg"]))

    # Synthesize HBI + Active trace for the full-frame density map and the selected scanline
    hbi_cols = max(24, int(round(cols_x * 0.16)))
    act_cols = cols_x - hbi_cols

    # Resample active columns to act_cols
    x_idx = np.linspace(0, w - 1, act_cols).astype(np.int32)
    luma_act = luma_ire[:, x_idx]
    comp_act = comp_ire[:, x_idx]
    r_act = setup_ire + ped_ire + arr[:, x_idx, 0] * span_ire
    g_act = setup_ire + ped_ire + arr[:, x_idx, 1] * span_ire
    b_act = setup_ire + ped_ire + arr[:, x_idx, 2] * span_ire

    # Construct HBI waveform columns (h, hbi_cols)
    u_hbi = np.linspace(0.0, 1.0, hbi_cols, dtype=np.float32)
    hbi_base = np.zeros((h, hbi_cols), dtype=np.float32)
    sync_mask = (u_hbi >= 0.16) & (u_hbi <= 0.54)
    burst_mask = (u_hbi >= 0.60) & (u_hbi <= 0.92)
    hbi_base[:, sync_mask] = sp["sync_tip_ire"]

    # Colour burst in back porch: 9 cycles of 20 IRE amplitude (40 IRE p-p), scaled by sync_atten
    burst_u = (u_hbi - 0.60) / max(1e-5, 0.92 - 0.60)
    burst_env = np.where(burst_mask, np.maximum(0.0, np.sin(np.pi * np.clip(burst_u, 0.0, 1.0))) ** 0.7, 0.0)
    burst_amp = 20.0 * sp["sync_atten"] * min(1.98, max(0.1, sp["chroma_gain"]))
    row_alt = np.where((np.arange(h) % 2) == 0, 1.0, -1.0)[:, None].astype(np.float32)
    burst_phase = (
        2.0 * np.pi * 9.0 * burst_u[None, :]
        + np.deg2rad(sp["burst_phase_err_deg"])
        + (row_alt * np.deg2rad(45.0) if sp["is_pal"] > 0.5 else 0.0)
    )
    hbi_comp = hbi_base + burst_amp * burst_env[None, :] * np.sin(burst_phase)

    full_luma = np.concatenate([hbi_base, luma_act], axis=1)
    full_comp = np.concatenate([hbi_comp, comp_act], axis=1)

    # Build 2D persistence density histogram (bins_ire x cols_x), IRE range [-45, +115]
    ire_min, ire_max = -82.0 if sp["sync_tip_ire"] < -45.0 else -45.0, 115.0
    density = np.zeros((bins_ire, cols_x), dtype=np.float32)
    bin_idx = np.clip(
        ((ire_max - full_comp) / (ire_max - ire_min) * (bins_ire - 1)).astype(np.int32),
        0,
        bins_ire - 1,
    )
    for c in range(cols_x):
        bc = np.bincount(bin_idx[:, c], minlength=bins_ire).astype(np.float32)
        density[:, c] = bc / max(1.0, float(h))
    # Log-compress density for authentic CRT phosphor persistence glow
    density_norm = np.clip(np.log1p(density * 18.0) / np.log1p(18.0), 0.0, 1.0)

    # Column-wise percentile envelopes (p05, p50, p95) for fast vector canvas drawing
    col_p05 = np.percentile(full_comp, 5, axis=0).astype(np.float32)
    col_p50 = np.percentile(full_comp, 50, axis=0).astype(np.float32)
    col_p95 = np.percentile(full_comp, 95, axis=0).astype(np.float32)
    col_luma_mean = full_luma.mean(axis=0).astype(np.float32)

    # RGB Parade envelopes across active video
    parade_cols = max(32, cols_x // 3)
    px_idx = np.linspace(0, act_cols - 1, parade_cols).astype(np.int32)
    parade = {
        "r_p10": np.round(np.percentile(r_act[:, px_idx], 10, axis=0), 2).tolist(),
        "r_p50": np.round(np.percentile(r_act[:, px_idx], 50, axis=0), 2).tolist(),
        "r_p90": np.round(np.percentile(r_act[:, px_idx], 90, axis=0), 2).tolist(),
        "g_p10": np.round(np.percentile(g_act[:, px_idx], 10, axis=0), 2).tolist(),
        "g_p50": np.round(np.percentile(g_act[:, px_idx], 50, axis=0), 2).tolist(),
        "g_p90": np.round(np.percentile(g_act[:, px_idx], 90, axis=0), 2).tolist(),
        "b_p10": np.round(np.percentile(b_act[:, px_idx], 10, axis=0), 2).tolist(),
        "b_p50": np.round(np.percentile(b_act[:, px_idx], 50, axis=0), 2).tolist(),
        "b_p90": np.round(np.percentile(b_act[:, px_idx], 90, axis=0), 2).tolist(),
    }

    # Selected scanline trace (HBI + active)
    line_comp_trace = full_comp[sel_row].astype(np.float32)
    line_luma_trace = full_luma[sel_row].astype(np.float32)
    line_r_trace = np.concatenate([hbi_base[sel_row], r_act[sel_row]]).astype(np.float32)
    line_g_trace = np.concatenate([hbi_base[sel_row], g_act[sel_row]]).astype(np.float32)
    line_b_trace = np.concatenate([hbi_base[sel_row], b_act[sel_row]]).astype(np.float32)

    active_luma_flat = luma_ire.ravel()
    stats = {
        "sync_tip_ire": round(float(sp["sync_tip_ire"]), 2),
        "termination_ohm": round(float(sp["termination_ohm"]), 2),
        "termination_gain_db": round(float(sp["termination_gain_db"]), 2),
        "blanking_ire": 0.0,
        "setup_ire": round(float(setup_ire + ped_ire), 2),
        "nominal_setup_ire": round(float(setup_ire), 2),
        "burst_pp_ire": round(float(2.0 * burst_amp), 2),
        "active_min_ire": round(float(np.percentile(active_luma_flat, 1)), 2),
        "active_mean_ire": round(float(active_luma_flat.mean()), 2),
        "active_p99_ire": round(float(np.percentile(active_luma_flat, 99)), 2),
        "active_max_ire": round(float(active_luma_flat.max()), 2),
        "clip_white_pct": round(float(np.mean(active_luma_flat >= 99.5) * 100.0), 2),
        "crush_black_pct": round(float(np.mean(active_luma_flat <= setup_ire + 0.5) * 100.0), 2),
        "selected_line": int(sel_row),
        "selected_line_norm": round(float(sel_row / max(1, h - 1)), 4),
        "total_lines": int(h),
        "hbi_fraction": round(float(hbi_cols / cols_x), 4),
    }

    return {
        "ire_min": ire_min,
        "ire_max": ire_max,
        "cols_x": int(cols_x),
        "bins_ire": int(bins_ire),
        "hbi_cols": int(hbi_cols),
        "stats": stats,
        "col_p05": np.round(col_p05, 2).tolist(),
        "col_p50": np.round(col_p50, 2).tolist(),
        "col_p95": np.round(col_p95, 2).tolist(),
        "col_luma_mean": np.round(col_luma_mean, 2).tolist(),
        "line_comp_ire": np.round(line_comp_trace, 2).tolist(),
        "line_luma_ire": np.round(line_luma_trace, 2).tolist(),
        "line_rgb_ire": {
            "r": np.round(line_r_trace, 2).tolist(),
            "g": np.round(line_g_trace, 2).tolist(),
            "b": np.round(line_b_trace, 2).tolist(),
        },
        "parade": parade,
        "density_grid": np.round(density_norm, 3).tolist(),
    }


def compute_vectorscope(
    rgb: np.ndarray,
    std: VideoStandard | None = None,
    params: dict | None = None,
    grid_size: int = 96,
    max_scatter: int = 900,
) -> dict[str, Any]:
    """
    Compute Broadcast Chroma Vectorscope telemetry, target boxes, burst vector, and 2D polar density.

    Coordinate convention (SMPTE / EBU broadcast vectorscope graticule):
      - Horizontal axis `U = 0.492 * (B - Y)` (`0°` = right `+U`, `180°` = left `-U` where NTSC Burst sits)
      - Vertical axis `V = 0.877 * (R - Y)` (`90°` = top `+V`, `270°` = bottom `-V`)
      - Normalized by `R_MAX = 0.632` so 100% colour bars reach their exact standard target boxes
        inside the unit circle `[-1.0, +1.0]`.
    """
    if std is None:
        std = get_standard("ntsc")
    sp = _extract_scope_params(params, std)
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = arr.shape[:2]

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # Broadcast U and V colour-difference signals, scaled to unit graticule radius (R_MAX = 0.632)
    u_norm = (0.492 * (b - y)) / 0.632
    v_norm = (0.877 * (r - y)) / 0.632

    # Measure even vs odd scanline chroma phase to detect PAL Hanover bar splitting
    even_u, even_v = u_norm[0::2], v_norm[0::2]
    odd_u, odd_v = u_norm[1::2], v_norm[1::2]
    even_mag = np.hypot(even_u, even_v)
    odd_mag = np.hypot(odd_u, odd_v)

    hanover_split_deg = 0.0
    if float(even_mag.max()) > 0.06 and float(odd_mag.max()) > 0.06:
        e_mask = even_mag > 0.06
        o_mask = odd_mag > 0.06
        ang_e = float(np.degrees(np.arctan2(even_v[e_mask].mean(), even_u[e_mask].mean())) % 360.0)
        ang_o = float(np.degrees(np.arctan2(odd_v[o_mask].mean(), odd_u[o_mask].mean())) % 360.0)
        diff_eo = abs(((ang_e - ang_o + 180.0) % 360.0) - 180.0)
        hanover_split_deg = round(float(diff_eo), 2)

    # Build 2D polar density histogram on [-1.05, +1.05]^2
    lim = 1.05
    u_flat = np.clip(u_norm.ravel(), -lim, lim)
    v_flat = np.clip(v_norm.ravel(), -lim, lim)
    gx = np.clip(((u_flat + lim) / (2.0 * lim) * (grid_size - 1)).astype(np.int32), 0, grid_size - 1)
    gy = np.clip(((lim - v_flat) / (2.0 * lim) * (grid_size - 1)).astype(np.int32), 0, grid_size - 1)
    hist2d = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(hist2d, (gy, gx), 1.0)
    hist2d /= max(1.0, float(u_flat.size) * 0.008)
    hist2d_norm = np.clip(np.log1p(hist2d * 12.0) / np.log1p(12.0), 0.0, 1.0)

    # Subsample scatter points (u, v, r, g, b) for crisp real-time vector trace rendering
    step = max(1, u_flat.size // max_scatter)
    u_sub = u_flat[::step][:max_scatter]
    v_sub = v_flat[::step][:max_scatter]
    rgb_flat = arr.reshape(-1, 3)[::step][:max_scatter]
    scatter = [
        [round(float(u_sub[k]), 4), round(float(v_sub[k]), 4),
         int(rgb_flat[k, 0] * 255), int(rgb_flat[k, 1] * 255), int(rgb_flat[k, 2] * 255)]
        for k in range(len(u_sub))
    ]

    # Target boxes (75% and 100%)
    targets: dict[str, dict[str, Any]] = {}
    for code, tinfo in VECTORSCOPE_TARGETS.items():
        rad = math.radians(tinfo["angle_deg"])
        targets[code] = {
            "label": tinfo["label"],
            "angle_deg": tinfo["angle_deg"],
            "r_75": tinfo["r_75"],
            "r_100": tinfo["r_100"],
            "u_75": round(tinfo["r_75"] * math.cos(rad), 4),
            "v_75": round(tinfo["r_75"] * math.sin(rad), 4),
            "u_100": round(tinfo["r_100"] * math.cos(rad), 4),
            "v_100": round(tinfo["r_100"] * math.sin(rad), 4),
        }

    # Colour Burst vector(s):
    # NTSC nominal burst is at 180 deg (-U axis), radius 0.285 (20 IRE / 70 IRE).
    # PAL swinging burst alternates between 135 deg and 225 deg.
    #Note: When `burst_phase_error` rotates the picture hues by +theta, the vectorscope shows both
    # the rotated chroma cloud and the phase offset between nominal Burst (180°) and recovered Burst.
    b_err = sp["burst_phase_err_deg"]
    burst_r = round(float(0.285 * sp["sync_atten"] * min(1.5, max(0.1, sp["chroma_gain"]))), 4)
    if sp["is_pal"] > 0.5:
        a1 = (135.0 + b_err) % 360.0
        a2 = (225.0 - b_err) % 360.0
        burst_vectors = [
            {"label": "Burst +135°", "angle_deg": round(a1, 2), "r": burst_r,
             "u": round(burst_r * math.cos(math.radians(a1)), 4),
             "v": round(burst_r * math.sin(math.radians(a1)), 4)},
            {"label": "Burst -135°", "angle_deg": round(a2, 2), "r": burst_r,
             "u": round(burst_r * math.cos(math.radians(a2)), 4),
             "v": round(burst_r * math.sin(math.radians(a2)), 4)},
        ]
        nominal_burst_deg = 135.0
    else:
        a0 = (180.0 + b_err) % 360.0
        burst_vectors = [
            {"label": "Burst 180°", "angle_deg": round(a0, 2), "r": burst_r,
             "u": round(burst_r * math.cos(math.radians(a0)), 4),
             "v": round(burst_r * math.sin(math.radians(a0)), 4)},
        ]
        nominal_burst_deg = 180.0

    # Dominant / mean chroma vector angle & saturation stats
    mag_all = np.hypot(u_norm, v_norm)
    active_chroma_mask = mag_all > 0.04
    if np.any(active_chroma_mask):
        angles_all = (np.degrees(np.arctan2(v_norm[active_chroma_mask], u_norm[active_chroma_mask])) + 360.0) % 360.0
        mean_sat = float(mag_all[active_chroma_mask].mean())
        p95_sat = float(np.percentile(mag_all[active_chroma_mask], 95))
        # Circular mean angle weighted by saturation magnitude
        weights = mag_all[active_chroma_mask]
        mean_u_w = float(np.sum(u_norm[active_chroma_mask] * weights) / max(1e-6, float(np.sum(weights))))
        mean_v_w = float(np.sum(v_norm[active_chroma_mask] * weights) / max(1e-6, float(np.sum(weights))))
        dominant_hue_deg = float((math.degrees(math.atan2(mean_v_w, mean_u_w)) + 360.0) % 360.0)
    else:
        mean_sat = 0.0
        p95_sat = float(np.percentile(mag_all, 95))
        dominant_hue_deg = 0.0

    return {
        "standard": "PAL" if sp["is_pal"] > 0.5 else "NTSC",
        "grid_size": int(grid_size),
        "targets": targets,
        "burst_vectors": burst_vectors,
        "stats": {
            "nominal_burst_deg": nominal_burst_deg,
            "burst_phase_error_deg": round(float(b_err), 2),
            "burst_amplitude_ire": round(float(burst_r * 70.0), 2),
            "acc_chroma_gain": round(float(sp["chroma_gain"]), 3),
            "mean_saturation_pct": round(float(mean_sat * 100.0), 2),
            "p95_saturation_pct": round(float(p95_sat * 100.0), 2),
            "dominant_hue_deg": round(dominant_hue_deg, 2),
            "hanover_split_deg": hanover_split_deg,
            "color_killer_active": bool(p95_sat < 0.025),
        },
        "scatter": scatter,
        "density_grid": np.round(hist2d_norm, 3).tolist(),
    }


def compute_ab_diff_and_spectrum(
    sim_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    std: VideoStandard | None = None,
    params: dict | None = None,
    bins_freq: int = 128,
    heatmap_w: int = 96,
    heatmap_h: int = 72,
    label_a: str = "仿真输出 (Sim)",
    label_b: str = "原始参考 (Source)",
) -> dict[str, Any]:
    """
    Compute Node-Level / Chain A/B Difference Heatmap & Horizontal Power Spectral Density (0..6.5 MHz).
    """
    if std is None:
        std = get_standard("ntsc")
    sp = _extract_scope_params(params, std)

    a = np.clip(np.asarray(sim_rgb, dtype=np.float32), 0.0, 1.0)
    b = np.clip(np.asarray(ref_rgb, dtype=np.float32), 0.0, 1.0)
    if a.shape != b.shape:
        # Nearest-neighbour resize b to match a
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        yi = np.linspace(0, hb - 1, ha).astype(np.int32)
        xi = np.linspace(0, wb - 1, wa).astype(np.int32)
        b = b[yi][:, xi]

    h, w = a.shape[:2]
    ya, ia, qa = rgb_to_yiq(a)
    yb, ib, qb = rgb_to_yiq(b)

    diff_y = np.abs(ya - yb)
    diff_c = np.hypot(ia - ib, qa - qb)
    diff_total = np.clip(0.65 * diff_y + 0.85 * diff_c, 0.0, 1.0)

    mse = float(np.mean((a - b) ** 2))
    rmse = math.sqrt(mse)
    psnr_db = 99.0 if mse < 1e-10 else float(10.0 * math.log10(1.0 / mse))
    l1_luma_ire = float(diff_y.mean() * 100.0)
    l1_chroma_ire = float(diff_c.mean() * 100.0)
    p95_err_ire = float(np.percentile(diff_total, 95) * 100.0)

    # Downsample heatmap to (heatmap_h, heatmap_w) for fast WebUI canvas rendering
    hy_idx = np.linspace(0, h - 1, heatmap_h).astype(np.int32)
    hx_idx = np.linspace(0, w - 1, heatmap_w).astype(np.int32)
    heat_small = diff_total[hy_idx][:, hx_idx]
    heat_luma_small = diff_y[hy_idx][:, hx_idx]
    heat_chroma_small = diff_c[hy_idx][:, hx_idx]

    # Horizontal 1D Power Spectrum (0 .. 6.5 MHz)
    # Using active line duration (~52.6 us NTSC / 52.0 us PAL) to map FFT bins to physical MHz
    active_us = 52.0 if sp["is_pal"] > 0.5 else 52.6
    fs_mhz = w / active_us  # Effective sample rate in MHz

    win = np.hanning(w).astype(np.float32)[None, :]
    ya_ac = (ya - ya.mean(axis=1, keepdims=True)) * win
    yb_ac = (yb - yb.mean(axis=1, keepdims=True)) * win

    # Synthesize modulated composite carrier on `a` so the spectrum also shows the 3.58/4.43 MHz chroma peak
    fsc_mhz = float(std.subcarrier_hz) / 1e6
    t_us = np.linspace(0.0, active_us, w, endpoint=False, dtype=np.float32)[None, :]
    row_alt = (np.arange(h, dtype=np.float32)[:, None] * 0.5)
    sub_ph = 2.0 * np.pi * (fsc_mhz * t_us + row_alt)
    comp_a = ya + 0.63 * sp["chroma_gain"] * (ia * np.cos(sub_ph) + qa * np.sin(sub_ph))
    comp_a_ac = (comp_a - comp_a.mean(axis=1, keepdims=True)) * win

    fft_a = np.fft.rfft(ya_ac, axis=1)
    fft_b = np.fft.rfft(yb_ac, axis=1)
    fft_comp = np.fft.rfft(comp_a_ac, axis=1)
    freqs_mhz = np.fft.rfftfreq(w, d=1.0 / fs_mhz)

    psd_a = np.mean(np.abs(fft_a) ** 2, axis=0) / max(1.0, float(w))
    psd_b = np.mean(np.abs(fft_b) ** 2, axis=0) / max(1.0, float(w))
    psd_comp = np.mean(np.abs(fft_comp) ** 2, axis=0) / max(1.0, float(w))

    ref_peak = max(1e-8, float(psd_b.max()), float(psd_comp.max()))
    db_a = 10.0 * np.log10(np.maximum(psd_a / ref_peak, 1e-8))
    db_b = 10.0 * np.log10(np.maximum(psd_b / ref_peak, 1e-8))
    db_comp = 10.0 * np.log10(np.maximum(psd_comp / ref_peak, 1e-8))

    max_mhz = 6.5
    f_grid = np.linspace(0.05, max_mhz, bins_freq, dtype=np.float32)
    spec_sim_luma_db = np.interp(f_grid, freqs_mhz, db_a).astype(np.float32)
    spec_ref_luma_db = np.interp(f_grid, freqs_mhz, db_b).astype(np.float32)
    spec_sim_comp_db = np.interp(f_grid, freqs_mhz, db_comp).astype(np.float32)

    # Measure high-frequency attenuation at 3.0 MHz (VHS luma limit) and subcarrier peak at fsc_mhz
    atten_3mhz_db = float(
        np.interp(3.0, f_grid, spec_sim_luma_db) - np.interp(3.0, f_grid, spec_ref_luma_db)
    )
    subcarrier_peak_db = float(np.interp(fsc_mhz, f_grid, spec_sim_comp_db))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "heatmap_w": int(heatmap_w),
        "heatmap_h": int(heatmap_h),
        "stats": {
            "psnr_db": round(psnr_db, 2),
            "rmse": round(rmse, 5),
            "l1_luma_ire": round(l1_luma_ire, 2),
            "l1_chroma_ire": round(l1_chroma_ire, 2),
            "p95_error_ire": round(p95_err_ire, 2),
            "luma_bw_nominal_mhz": round(sp["luma_bw_mhz"], 2),
            "chroma_bw_nominal_mhz": round(sp["chroma_bw_mhz"], 2),
            "subcarrier_mhz": round(fsc_mhz, 3),
            "atten_at_3mhz_db": round(atten_3mhz_db, 2),
            "subcarrier_peak_db": round(subcarrier_peak_db, 2),
        },
        "spectrum": {
            "freqs_mhz": np.round(f_grid, 3).tolist(),
            "ref_luma_db": np.round(spec_ref_luma_db, 2).tolist(),
            "sim_luma_db": np.round(spec_sim_luma_db, 2).tolist(),
            "sim_comp_db": np.round(spec_sim_comp_db, 2).tolist(),
        },
        "heatmap": {
            "total": np.round(heat_small, 3).tolist(),
            "luma": np.round(heat_luma_small, 3).tolist(),
            "chroma": np.round(heat_chroma_small, 3).tolist(),
        },
    }


def compute_all_scopes(
    sim_rgb: np.ndarray,
    ref_rgb: np.ndarray | None = None,
    std: VideoStandard | None = None,
    params: dict | None = None,
    line_y: float | int | None = None,
    label_a: str = "仿真输出 (Sim)",
    label_b: str = "原始参考 (Source)",
) -> dict[str, Any]:
    """Compute Waveform Monitor, Vectorscope, and A/B Diff & Spectrum in one unified payload."""
    if std is None:
        std = get_standard("ntsc")
    if ref_rgb is None:
        ref_rgb = sim_rgb
    wfm = compute_waveform_monitor(sim_rgb, std=std, params=params, line_y=line_y)
    vec = compute_vectorscope(sim_rgb, std=std, params=params)
    ab = compute_ab_diff_and_spectrum(
        sim_rgb, ref_rgb, std=std, params=params, label_a=label_a, label_b=label_b
    )
    return {
        "standard": "PAL" if str(getattr(std, "color_system", "NTSC")).upper() == "PAL" else "NTSC",
        "waveform": wfm,
        "vectorscope": vec,
        "ab_compare": ab,
    }

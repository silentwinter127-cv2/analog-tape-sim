"""
Candidate I: Master Control Playout Automation (`master_playout`) & Dual-Bus Preview / Clean Feed / DSK.

Models the final broadcast transmission desk (Master Control Room / MCR Playout):
  1. **Dual-Bus PVW / PGM & Clean Feed Architecture (`playout_output_bus`)**:
     - **Clean Feed (`bus = 1`)**: The switched and color-corrected program video *before* the
       Downstream Keyer (DSK), used for syndication, international feeds, and master tape archiving.
       Guaranteed 100% free of station bugs, clocks, and lower-third tickers.
     - **Program On-Air (`PGM`, `bus = 0`)**: Clean Feed passed through the **Downstream Keyer (DSK)**,
       which keys the station emblem bug (`TVB / HK-MCR1`), real-time HH:MM:SS:FF broadcast clock,
       program classification badge, and bottom crawling news ticker (`playout_dsk_crawl`).
     - **Preview Bus (`PVW`, `bus = 2`)**: Next queued playlist source or VTR preroll leader with
       safe-title reticle, countdown cue overlay, and green `PVW NEXT` tally banner.
     - **MCR Dual-Bus Playout Desk View (`bus = 3`)**: Split-screen operator console displaying
       `PVW (NEXT)` on the left, `PGM (ON-AIR)` on the right, `CLEAN FEED` mini-monitor, and the
       live **Broadcast As-Run Event Log** table along the bottom.
  2. **Master Control Playlist State Machine & Emergency Failover (`playout_state`, `playout_failover_auto`)**:
     - `0 = ON-AIR Program`: Normal playlist execution.
     - `1 = VTR Preroll Leader Countdown`: 5-second servo lock countdown (`PREROLL -00:03 -> LOCKED -> TAKE`)
       with SMPTE leader crosshairs and cue tone indicator before rolling into Program.
     - `2 = Emergency Standby Slate Failover`: When the VTR loses lock, tape snaps/sticks, or manual
       emergency takeover is triggered, the master playout matrix automatically cuts to the station's
       `"PLEASE STAND BY / 请稍候 · 信号传输中断"` calibration slate while keeping the DSK station bug
       and clock alive on PGM and logging a `FAILOVER_STANDBY` entry in the As-Run Log.
     - `3 = Playlist Auto-Transition Cycle`: Cycles through `Preroll -> On-Air -> Next Source` across frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode
from .multiview_wall import _draw_bitmap_text


def _draw_text_5x7(
    canvas: np.ndarray,
    text: str,
    x0: int,
    y0: int,
    scale: int = 1,
    color: tuple[float, float, float] = (0.94, 0.96, 0.94),
    bg: tuple[float, float, float] | None = None,
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
from .test_signal_gen import generate_test_pattern

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

PLAYOUT_BUS_NAMES: dict[int, str] = {
    0: "pgm_on_air",
    1: "clean_feed",
    2: "pvw_next",
    3: "mcr_dual_bus_desk",
}

PLAYOUT_STATE_NAMES: dict[int, str] = {
    0: "on_air_program",
    1: "vtr_preroll_countdown",
    2: "emergency_standby_slate",
    3: "playlist_auto_cycle",
}


@dataclass
class MasterPlayoutSpec:
    """Configuration of the Master Control Playout & DSK subsystem (`overlay` layer)."""
    enabled: bool = False
    output_bus: int = 0          # 0=PGM (with DSK), 1=Clean Feed (no DSK), 2=PVW, 3=MCR Dual-Bus Desk
    playout_state: int = 0       # 0=On-Air, 1=Preroll Countdown, 2=Emergency Standby Slate, 3=Auto-Cycle
    dsk_enabled: bool = True     # Downstream Keyer (Station Bug + Clock)
    dsk_crawl: bool = True       # Bottom news/notice ticker crawl on PGM
    failover_auto: bool = True   # Automatic emergency slate takeover on upstream signal loss

    @classmethod
    def from_params(cls, cfg: dict) -> "MasterPlayoutSpec":
        c = cfg or {}
        return cls(
            enabled=float(c["playout_enabled"]) > 0.5 if "playout_enabled" in c else False,
            output_bus=int(np.clip(round(float(c["playout_output_bus"])) if "playout_output_bus" in c else 0, 0, 3)),
            playout_state=int(np.clip(round(float(c["playout_state"])) if "playout_state" in c else 0, 0, 3)),
            dsk_enabled=float(c["playout_dsk_enabled"]) > 0.5 if "playout_dsk_enabled" in c else True,
            dsk_crawl=float(c["playout_dsk_crawl"]) > 0.5 if "playout_dsk_crawl" in c else True,
            failover_auto=float(c["playout_failover_auto"]) > 0.5 if "playout_failover_auto" in c else True,
        )


def _resample_nn(img: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (h, w) == (h_out, w_out):
        return img
    ys = np.clip((np.arange(h_out) * h) // max(1, h_out), 0, h - 1)
    xs = np.clip((np.arange(w_out) * w) // max(1, w_out), 0, w - 1)
    return img[ys][:, xs]


def render_emergency_standby_slate(width: int, height: int, frame_index: int = 0, std=None) -> np.ndarray:
    """
    Synthesize an authentic 1990s broadcast `"PLEASE STAND BY / 节目信号传输中断 · 请稍候"` emergency
    apology & station ID slate over a dimmed SMPTE/PM5544 engineering backing.
    """
    bg = generate_test_pattern(width, height, mode=1, frame_index=frame_index, std=std)
    slate = bg * 0.28
    # Central broadcast slate placard
    y0, y1 = int(height * 0.20), int(height * 0.80)
    x0, x1 = int(width * 0.12), int(width * 0.88)
    slate[y0:y1, x0:x1] = np.array([0.08, 0.14, 0.34], dtype=np.float32)

    # Double gold/white border frame
    bw = max(2, min(width, height) // 90)
    gold = np.array([0.95, 0.78, 0.24], dtype=np.float32)
    slate[y0:y0 + bw, x0:x1] = gold
    slate[y1 - bw:y1, x0:x1] = gold
    slate[y0:y1, x0:x0 + bw] = gold
    slate[y0:y1, x1 - bw:x1] = gold

    # Color bar reference strip inside the slate header
    bar_y0, bar_y1 = y0 + bw * 3, y0 + int((y1 - y0) * 0.22)
    bar_x0, bar_x1 = x0 + bw * 4, x1 - bw * 4
    if bar_y1 > bar_y0 and bar_x1 > bar_x0:
        slate[bar_y0:bar_y1, bar_x0:bar_x1] = bg[bar_y0:bar_y1, bar_x0:bar_x1] * 0.85

    scale = max(1, min(width // 180, height // 120))
    _draw_text_5x7(
        slate, "PLEASE STAND BY",
        x0 + int((x1 - x0) * 0.16), y0 + int((y1 - y0) * 0.34),
        scale=scale + 1, color=(0.98, 0.96, 0.90), bg=(0.05, 0.09, 0.24),
    )
    _draw_text_5x7(
        slate, "SIGNAL INTERRUPTED - STANDBY BACKUP",
        x0 + int((x1 - x0) * 0.10), y0 + int((y1 - y0) * 0.56),
        scale=max(1, scale), color=(0.95, 0.78, 0.25), bg=(0.05, 0.09, 0.24),
    )
    _draw_text_5x7(
        slate, f"MCR AUTO-FAILOVER ACTIVE  FRM:{int(frame_index):04d}",
        x0 + int((x1 - x0) * 0.10), y0 + int((y1 - y0) * 0.74),
        scale=max(1, scale), color=(0.45, 0.92, 0.65), bg=(0.05, 0.09, 0.24),
    )
    return np.clip(slate, 0.0, 1.0).astype(np.float32)


def render_vtr_preroll_leader(
    base_rgb: np.ndarray,
    frame_index: int = 0,
    fps: float = 29.97,
) -> tuple[np.ndarray, dict]:
    """
    Render a broadcast VTR Preroll Leader (`PREROLL -00:03 -> SERVO LOCK -> TAKE`) with SMPTE clock wipe
    and transport lock status bar.
    """
    h, w = base_rgb.shape[:2]
    # Countdown from -5s to 0s over 10 frames so preview scrubbing shows distinct countdown seconds
    rem_sec = max(1, 5 - (int(frame_index) % 5))
    lock_pct = min(100, int(round(45 + (int(frame_index) % 5) * 13)))
    out = base_rgb * 0.52

    # SMPTE countdown crosshair & circle
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.5, w * 0.5
    rad = min(h, w) * 0.30
    dist = np.hypot(yy - cy, xx - cx)
    ring = (np.abs(dist - rad) < max(1.5, min(h, w) / 140.0)) | (np.abs(dist - rad * 0.78) < max(1.2, min(h, w) / 160.0))
    cross = ((np.abs(yy - cy) <= 1.0) & (dist < rad * 1.15)) | ((np.abs(xx - cx) <= 1.0) & (dist < rad * 1.15))

    # Sweeping radar sector
    ang = (np.arctan2(yy - cy, xx - cx) + np.pi * 0.5) % (2.0 * np.pi)
    sweep_lim = ((int(frame_index) % 5) + 1) * (2.0 * np.pi / 5.0)
    sector = (dist < rad) & (ang <= sweep_lim)
    out[sector] = np.clip(out[sector] * 1.35 + np.array([0.06, 0.12, 0.18], dtype=np.float32), 0.0, 1.0)
    out[ring | cross] = np.array([0.92, 0.94, 0.90], dtype=np.float32)

    scale = max(1, min(w // 160, h // 110))
    # Big countdown numeral in center
    _draw_text_5x7(
        out, f"-00:0{rem_sec}",
        int(cx - 21 * (scale + 1)), int(cy - 4 * (scale + 1)),
        scale=scale + 1, color=(0.98, 0.92, 0.25), bg=(0.06, 0.06, 0.08),
    )
    # Top VTR Preroll status banner
    _draw_text_5x7(
        out, f"VTR-A PREROLL CUE  SERVO:{lock_pct}% LOCKED",
        int(w * 0.08), int(h * 0.08),
        scale=scale, color=(0.35, 0.96, 0.48), bg=(0.04, 0.08, 0.05),
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "preroll_remaining_s": rem_sec,
        "servo_lock_pct": lock_pct,
    }


def apply_dsk_overlay(
    clean_rgb: np.ndarray,
    spec: MasterPlayoutSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    is_standby: bool = False,
) -> np.ndarray:
    """
    Apply the Downstream Keyer (DSK) onto `clean_rgb` to produce the `PGM` on-air bus:
      - Top-right station bug (`HK-TV1 HD/PAL`) + live broadcast timecode clock (`15:03:XX:FF`);
      - Bottom lower-third news/program crawl bar (`playout_dsk_crawl`).
    Crucially returns a new array without modifying `clean_rgb` in place.
    """
    pgm = np.asarray(clean_rgb, dtype=np.float32).copy()
    if not spec.dsk_enabled and not spec.dsk_crawl:
        return pgm

    h, w = pgm.shape[:2]
    scale = max(1, min(w // 200, h // 135))

    if spec.dsk_enabled:
        # 1. Top-Right Broadcast Station Bug + Timecode Clock Box
        sec = int(frame_index / max(1.0, fps))
        ff = int(frame_index % max(1, int(round(fps))))
        bug_str = f"HK-TV1  20:{sec // 60:02d}:{sec % 60:02d}:{ff:02d}"
        bx = max(4, w - (len(bug_str) * 6 * scale) - int(w * 0.04))
        by = max(4, int(h * 0.055))

        # Tri-color station emblem icon (Red/Green/Blue TVB-style bars) to the left of the bug text
        em_w = max(6, 5 * scale)
        em_h = max(7, 7 * scale)
        ex0 = max(2, bx - em_w - 4)
        ey0 = by
        if ey0 + em_h < h and ex0 + em_w < w:
            w3 = max(1, em_w // 3)
            pgm[ey0:ey0 + em_h, ex0:ex0 + w3] = (0.18, 0.55, 0.96)
            pgm[ey0:ey0 + em_h, ex0 + w3:ex0 + 2 * w3] = (0.22, 0.88, 0.38)
            pgm[ey0:ey0 + em_h, ex0 + 2 * w3:ex0 + em_w] = (0.94, 0.24, 0.20)

        _draw_text_5x7(
            pgm, bug_str, bx, by,
            scale=scale, color=(0.94, 0.96, 0.94), bg=(0.06, 0.08, 0.12),
        )

    if spec.dsk_crawl:
        # 2. Bottom Lower-Third News / Master Control Ticker Crawl
        bar_h = max(12, 11 * scale)
        by0 = h - bar_h - max(3, int(h * 0.03))
        by1 = by0 + bar_h
        if by0 > 0:
            # Semi-transparent dark broadcast lower-third ribbon with amber top rule
            pgm[by0:by1, :] = pgm[by0:by1, :] * 0.25 + np.array([0.05, 0.08, 0.16], dtype=np.float32) * 0.75
            pgm[by0:min(h, by0 + max(1, scale)), :] = (0.94, 0.72, 0.18)

            ticker_msg = (
                "MCR ALERT: BACKUP SLATE ON-AIR - VTR SERVO RECOVERY IN PROGRESS   "
                if is_standby else
                "MCR SCHEDULE: 20:00 EVENING NEWS  |  20:30 STUDIO LIVE DRAMA  |  DSK KEYER ACTIVE   "
            )
            # Scroll horizontally by frame_index
            shift_chars = (int(frame_index) // 2) % max(1, len(ticker_msg))
            scrolled = (ticker_msg[shift_chars:] + ticker_msg[:shift_chars])[:max(16, w // (6 * scale) - 2)]
            _draw_text_5x7(
                pgm, scrolled,
                int(w * 0.03), by0 + max(2, (bar_h - 7 * scale) // 2),
                scale=scale, color=(0.98, 0.92, 0.72), bg=None,
            )

    return np.clip(pgm, 0.0, 1.0).astype(np.float32)


def build_as_run_log(
    spec: MasterPlayoutSpec,
    eff_state: int,
    frame_index: int,
    fps: float,
    upstream_fault: bool,
) -> list[dict]:
    """
    Generate the structured Broadcast As-Run Event Log (`as_run_log`) for Master Control verification.
    """
    sec = int(frame_index / max(1.0, fps))
    ff = int(frame_index % max(1, int(round(fps))))
    tc_now = f"20:00:{sec:02d}:{ff:02d}"
    dsk_tag = "DSK1:ON+CRAWL" if (spec.dsk_enabled and spec.dsk_crawl) else ("DSK1:BUG" if spec.dsk_enabled else "DSK1:BYPASS")

    events = [
        {
            "tc": "20:00:00:00",
            "event_id": "EVT-101",
            "type": "PREROLL",
            "source": "VTR-A (VHS SP)",
            "bus": "PVW",
            "dsk": dsk_tag,
            "status": "SERVO LOCKED (-05.0s)",
        },
        {
            "tc": "20:00:00:12",
            "event_id": "EVT-102",
            "type": "TAKE_PGM",
            "source": "CAM-1 / VTR-A",
            "bus": "PGM+CLEAN",
            "dsk": dsk_tag,
            "status": "ON-AIR NORMAL",
        },
    ]
    if eff_state == 1:
        events.append({
            "tc": tc_now,
            "event_id": "EVT-103",
            "type": "COUNTDOWN",
            "source": "VTR-B NEXT",
            "bus": "PVW->PGM",
            "dsk": dsk_tag,
            "status": f"PREROLL CUEING (FRM {frame_index})",
        })
    elif eff_state == 2:
        reason = "UPSTREAM SYNC LOSS" if upstream_fault else "MANUAL STANDBY TAKEOVER"
        events.append({
            "tc": tc_now,
            "event_id": "EVT-999",
            "type": "FAILOVER",
            "source": "SLATE_GEN (STBY)",
            "bus": "PGM+CLEAN",
            "dsk": dsk_tag,
            "status": f"EMERGENCY SLATE ({reason})",
        })
    else:
        events.append({
            "tc": tc_now,
            "event_id": "EVT-104",
            "type": "VERIFY",
            "source": "PGM + CLEAN_FEED",
            "bus": PLAYOUT_BUS_NAMES.get(spec.output_bus, "pgm_on_air").upper(),
            "dsk": dsk_tag,
            "status": "CLEAN FEED ISOLATED (0% DSK SPILL)",
        })
    return events


def render_mcr_dual_bus_desk(
    pvw_rgb: np.ndarray,
    pgm_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    as_run_log: list[dict],
    eff_state: int,
) -> np.ndarray:
    """
    Compose the Master Control Room (MCR) Dual-Bus Operator Console (`output_bus == 3`):
      - Top-Left: `PVW / NEXT` Preview monitor (green tally)
      - Top-Right: `PGM / ON-AIR` Program monitor with DSK (red tally)
      - Bottom: `CLEAN FEED` miniature verification strip + Live `AS-RUN LOG` event table.
    """
    h, w = pgm_rgb.shape[:2]
    desk = np.full((h, w, 3), (0.07, 0.08, 0.10), dtype=np.float32)

    pad = max(3, min(h, w) // 80)
    top_h = int(h * 0.64)
    cell_w = (w - 3 * pad) // 2
    cell_h = top_h - 2 * pad

    if cell_w > 16 and cell_h > 16:
        pvw_cell = _resample_nn(pvw_rgb, cell_h, cell_w)
        pgm_cell = _resample_nn(pgm_rgb, cell_h, cell_w)
        # Green Tally border above PVW, Red Tally border above PGM
        desk[pad:pad + cell_h, pad:pad + cell_w] = pvw_cell
        desk[pad:pad + cell_h, 2 * pad + cell_w:2 * pad + 2 * cell_w] = pgm_cell
        desk[max(0, pad - 3):pad, pad + cell_w // 4:pad + 3 * cell_w // 4] = (0.18, 0.90, 0.32)
        desk[max(0, pad - 3):pad, 2 * pad + cell_w + cell_w // 4:2 * pad + cell_w + 3 * cell_w // 4] = (0.95, 0.18, 0.16)

    # Bottom As-Run Log + Clean Feed verification strip
    bot_y0 = top_h + pad
    bot_h = h - bot_y0 - pad
    if bot_h > 18:
        cf_w = int(w * 0.26)
        cf_cell = _resample_nn(clean_rgb, bot_h, max(8, cf_w - pad))
        desk[bot_y0:bot_y0 + bot_h, pad:pad + cf_cell.shape[1]] = cf_cell
        scale = max(1, min(w // 240, h // 160))
        _draw_text_5x7(
            desk, "CLEAN FEED",
            pad + 3, bot_y0 + 3,
            scale=scale, color=(0.35, 0.90, 0.98), bg=(0.04, 0.06, 0.10),
        )

        # Right side of bottom strip: As-Run Log entries
        lx0 = pad + cf_w + pad
        desk[bot_y0:bot_y0 + bot_h, lx0:w - pad] = (0.04, 0.05, 0.07)
        _draw_text_5x7(
            desk, f"MCR AS-RUN LOG  STATE:{PLAYOUT_STATE_NAMES.get(eff_state, 'ON-AIR').upper()}",
            lx0 + 4, bot_y0 + 3,
            scale=scale, color=(0.96, 0.80, 0.24), bg=None,
        )
        row_step = max(9, 8 * scale + 2)
        for idx_e, ev in enumerate(as_run_log[-3:]):
            ly = bot_y0 + 4 + (idx_e + 1) * row_step
            if ly + 7 * scale >= h - pad:
                break
            line_s = f"{ev['tc']} {ev['type']:8s} {ev['source'][:12]:12s} {ev['status'][:22]}"
            col = (0.95, 0.35, 0.30) if ev["type"] == "FAILOVER" else (0.78, 0.88, 0.80)
            _draw_text_5x7(desk, line_s, lx0 + 4, ly, scale=scale, color=col, bg=None)

    return np.clip(desk, 0.0, 1.0).astype(np.float32)


def apply_master_playout(
    rgb: np.ndarray,
    spec: MasterPlayoutSpec,
    std=None,
    frame_index: int = 0,
    fps: float = 29.97,
    pvw_source_rgb: np.ndarray | None = None,
    upstream_fault: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Execute the Master Control Playout state machine, producing:
      - `clean_feed_rgb` (Clean Feed without DSK),
      - `pgm_rgb` (Program On-Air with DSK bug/clock/crawl),
      - `pvw_rgb` (Preview bus with next source / preroll cue),
      - `as_run_log` (Broadcast As-Run Event Log),
    and returning the user-selected output bus (`spec.output_bus`).
    """
    h, w = rgb.shape[:2]
    base_prog = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)

    # Check if upstream signal is dark/lost or explicitly flagged as faulted
    luma_std = float((base_prog @ LUMA).std())
    auto_fault_detected = bool(upstream_fault or (spec.failover_auto and luma_std < 0.002))

    eff_state = int(spec.playout_state)
    if eff_state == 3:
        # Playlist Auto-Transition cycle across frames:
        # frames 0..1 -> Preroll Countdown; frames 2..4 -> On-Air Program; frame 5 -> Next Source
        cyc = int(frame_index) % 6
        eff_state = 1 if cyc < 2 else 0
    if auto_fault_detected and spec.failover_auto:
        eff_state = 2

    # Build Preview (`pvw_base`) source: either provided `pvw_source_rgb` or PM5544 station chart
    if pvw_source_rgb is not None and pvw_source_rgb.shape == base_prog.shape:
        pvw_base = np.clip(np.asarray(pvw_source_rgb, dtype=np.float32), 0.0, 1.0)
    else:
        pvw_base = generate_test_pattern(w, h, mode=5, frame_index=frame_index, fps=fps, std=std)

    preroll_info = {}
    if eff_state == 1:
        # VTR Preroll Leader Countdown active on Clean Feed / PGM
        clean_feed_rgb, preroll_info = render_vtr_preroll_leader(base_prog, frame_index=frame_index, fps=fps)
    elif eff_state == 2:
        # Emergency Standby Slate active on Clean Feed / PGM
        clean_feed_rgb = render_emergency_standby_slate(w, h, frame_index=frame_index, std=std)
    else:
        clean_feed_rgb = base_prog.copy()

    # Apply Downstream Keyer (DSK) ONLY to `pgm_rgb` — `clean_feed_rgb` stays 100% clean!
    pgm_rgb = apply_dsk_overlay(
        clean_feed_rgb, spec, frame_index=frame_index, fps=fps, is_standby=(eff_state == 2),
    )

    # Build `pvw_rgb` with safe-action reticle and PVW NEXT cue header
    pvw_rgb = pvw_base.copy()
    ry0, ry1 = int(h * 0.10), int(h * 0.90)
    rx0, rx1 = int(w * 0.10), int(w * 0.90)
    pvw_rgb[ry0, rx0:rx1:3] = (0.25, 0.92, 0.40)
    pvw_rgb[ry1 - 1, rx0:rx1:3] = (0.25, 0.92, 0.40)
    pvw_rgb[ry0:ry1:3, rx0] = (0.25, 0.92, 0.40)
    pvw_rgb[ry0:ry1:3, rx1 - 1] = (0.25, 0.92, 0.40)
    scale = max(1, min(w // 200, h // 135))
    _draw_text_5x7(
        pvw_rgb, "PVW NEXT: VTR-B CUE READY",
        int(w * 0.05), int(h * 0.055),
        scale=scale, color=(0.25, 0.96, 0.42), bg=(0.04, 0.08, 0.05),
    )

    as_run_log = build_as_run_log(spec, eff_state, frame_index=frame_index, fps=fps, upstream_fault=auto_fault_detected)

    bus = int(spec.output_bus)
    if bus == 1:
        selected_rgb = clean_feed_rgb
    elif bus == 2:
        selected_rgb = pvw_rgb
    elif bus == 3:
        selected_rgb = render_mcr_dual_bus_desk(pvw_rgb, pgm_rgb, clean_feed_rgb, as_run_log, eff_state)
    else:
        selected_rgb = pgm_rgb

    dsk_diff_l1 = float(np.abs(pgm_rgb - clean_feed_rgb).mean())
    info = {
        "enabled": True,
        "output_bus": bus,
        "output_bus_name": PLAYOUT_BUS_NAMES.get(bus, "pgm_on_air"),
        "playout_state": eff_state,
        "playout_state_name": PLAYOUT_STATE_NAMES.get(eff_state, "on_air_program"),
        "dsk_enabled": bool(spec.dsk_enabled),
        "dsk_crawl": bool(spec.dsk_crawl),
        "failover_triggered": bool(eff_state == 2),
        "pgm_vs_clean_l1": round(dsk_diff_l1, 5),
        "as_run_log": as_run_log,
        **preroll_info,
    }
    return selected_rgb, {
        "report": info,
        "pgm_rgb": pgm_rgb,
        "clean_feed_rgb": clean_feed_rgb,
        "pvw_rgb": pvw_rgb,
    }


class MasterPlayoutStage(Stage):
    """
    Standalone Node-Graph Stage (`master_playout`) for Master Control Playout, DSK, and Clean Feed.
    """
    name = "master_playout"
    layers = ("overlay",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = dict(self.cfg_of("overlay") or {})
        c["playout_enabled"] = 1.0
        self.spec = MasterPlayoutSpec.from_params(c)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig_in: SignalFrame | None = buf.meta.get("signal")
        base_rgb = buf.rgb if (buf.rgb is not None and buf.meta.get("from_display", False)) else (
            sig_in.to_rgb() if sig_in is not None else np.clip(buf.rgb, 0.0, 1.0)
        )
        pvw_src = None
        if "bus_b_yiq" in buf.meta:
            yb, ib, qb = buf.meta["bus_b_yiq"]
            pvw_src = SignalFrame(yb, ib, qb).to_rgb()

        upstream_flt = (buf.meta.get("format") == "no_signal")
        out_rgb, res = apply_master_playout(
            base_rgb, self.spec, std=buf.std,
            frame_index=int(buf.index), fps=self.ctx.fps,
            pvw_source_rgb=pvw_src, upstream_fault=upstream_flt,
        )
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        comp = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63) if buf.composite is not None else None
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["master_playout"] = res["report"]
        meta["playout_buses"] = {
            "pgm_rgb": res["pgm_rgb"],
            "clean_feed_rgb": res["clean_feed_rgb"],
            "pvw_rgb": res["pvw_rgb"],
        }
        return FrameBuffer(out_rgb, comp, buf.index, buf.time, buf.std, meta)

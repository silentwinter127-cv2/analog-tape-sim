"""
Candidate L: Event-Driven & Condition-Driven Broadcast Playout Automation Engine v2 (`playout_automation_v2`).

Implements Section 5 & Section 6 of the Broadcast Infrastructure Specification (`README.txt`):
  1. **Unified Event State Machine (`PlayoutEvent`)**:
     - Normal lifecycle progression:
         `SCHEDULED -> PREPARING (VTR Pre-roll & servo lock) -> READY -> ON_AIR -> COMPLETED`
     - 6 Distinct Hierarchical Exception Branches (never collapsed into a generic "playback failed"):
         * `MISSING_ASSET`: Asset ID missing from media library / tape bin -> immediate skip/fallback
           without VTR transport roll (`CAUSE: ASSET_STORAGE_MISSING`).
         * `NOT_READY`: VTR transport failed to reach servo lock within the pre-roll window (e.g., due
           to tape stiction, belt slip, or slow capstan lock) -> holds previous event or switches to
           standby filler (`CAUSE: VTR_PREROLL_SERVO_TIMEOUT`).
         * `SIGNAL_LOSS`: Active `ON_AIR` source loses sync, drops to black, freezes, or mutes beyond
           `watchdog_sec` -> automatic failover to redundant backup (`CAUSE: WATCHDOG_SIGNAL_LOSS`).
         * `OVERRIDDEN`: Master Control Operator manual crosspoint takeover (`TAKE MANUAL / BREAK-IN`)
           -> forces live/emergency bus to air (`CAUSE: MCR_OPERATOR_OVERRIDE`).
         * `FALLBACK`: Redundant backup chain (`BACKUP_B` or `STANDBY_SLATE`) actively on air after
           failover, tracking failover latency (`failover_gap_ms`) and primary recovery state.
         * `HELD`: Conditional / GPI-driven event waiting for upstream live end-cue or operator confirm,
           rippling subsequent scheduled times (`CAUSE: GPI_CUE_HOLD_OVERRUN`).
  2. **Three Automation Scheduling & Redundancy Paradigms (`auto_v2_mode`)**:
     - `1 = Time-Driven Fixed Playlist`: Strict wall-clock execution (`19:00:00 News -> 19:30:00 Outro
       -> 19:30:15 Commercial Break -> 19:32:15 Program Return`).
     - `2 = Condition / GPI Event-Driven`: Waits for VTR servo lock (`transport_servo`), satellite lock
       (`satellite_ird`), or live studio tally before transitioning; ripples downstream events when held.
     - `3 = Redundant Main/Backup Closed-Loop Failover`: Monitors physical upstream telemetry (black
       frame, freeze frame, audio silence, VTR stiction/unthread, RF/Satellite unlock) and switches to
       the hot-standby backup feed while recording a reverse-traceable `CausalFaultRecord` chain
       (Infrastructure C).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode
from .master_playout import (
    MasterPlayoutSpec,
    _draw_text_5x7,
    _resample_nn,
    apply_dsk_overlay,
    render_emergency_standby_slate,
    render_vtr_preroll_leader,
)
from .test_signal_gen import generate_test_pattern

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

AUTOMATION_MODES: dict[int, str] = {
    0: "off",
    1: "time_driven_playlist",
    2: "condition_gpi_driven",
    3: "redundant_main_backup_failover",
}

FAULT_INJECT_NAMES: dict[int, str] = {
    0: "NONE_PHYSICAL_AUTO",
    1: "MISSING_ASSET",
    2: "NOT_READY",
    3: "SIGNAL_LOSS",
    4: "OVERRIDDEN",
    5: "FALLBACK",
    6: "HELD",
}

VALID_EVENT_STATUSES: tuple[str, ...] = (
    "SCHEDULED",
    "PREPARING",
    "READY",
    "ON_AIR",
    "COMPLETED",
    "MISSING_ASSET",
    "NOT_READY",
    "SIGNAL_LOSS",
    "OVERRIDDEN",
    "FALLBACK",
    "HELD",
)


@dataclass
class PlayoutEvent:
    """
    Single broadcast automation schedule item (`README.txt` §5.2).
    """
    event_id: str
    scheduled_time: float
    duration: float | None
    source_id: str
    action: str
    fallback_id: str | None = "SLATE_STANDBY"
    status: str = "SCHEDULED"
    title: str = ""
    trigger_type: str = "TIME"        # "TIME" | "GPI_CONDITION" | "MANUAL"
    actual_start_time: float | None = None
    servo_lock_pct: float = 100.0
    diagnostic_cause: str = "NOMINAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "scheduled_time": round(float(self.scheduled_time), 3),
            "duration": round(float(self.duration), 3) if self.duration is not None else None,
            "source_id": str(self.source_id),
            "action": str(self.action),
            "fallback_id": str(self.fallback_id) if self.fallback_id else None,
            "status": str(self.status),
            "title": str(self.title),
            "trigger_type": str(self.trigger_type),
            "actual_start_time": round(float(self.actual_start_time), 3) if self.actual_start_time is not None else None,
            "servo_lock_pct": round(float(self.servo_lock_pct), 1),
            "diagnostic_cause": str(self.diagnostic_cause),
        }


@dataclass
class CausalFaultRecord:
    """
    Infrastructure C (`README.txt` §6): Reverse-traceable causal fault & decision record.
    Links physical device anomalies -> watchdog measurements -> automation failover decisions.
    """
    timestamp_tc: str
    event_id: str
    root_device: str
    physical_symptom: str
    watchdog_metric: str
    automation_status: str
    recovery_action: str
    black_gap_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_tc": self.timestamp_tc,
            "event_id": self.event_id,
            "root_device": self.root_device,
            "physical_symptom": self.physical_symptom,
            "watchdog_metric": self.watchdog_metric,
            "automation_status": self.automation_status,
            "recovery_action": self.recovery_action,
            "black_gap_ms": round(float(self.black_gap_ms), 1),
        }


@dataclass
class PlayoutAutomationV2Spec:
    """Configuration for Candidate L (`overlay` layer / `automation_controller` node)."""
    enabled: bool = False
    mode: int = 1                # 0=Off, 1=Time-Driven, 2=Condition/GPI-Driven, 3=Redundant Main/Backup Failover
    fault_inject: int = 0        # 0=Auto from upstream physics, 1=MISSING_ASSET, 2=NOT_READY, 3=SIGNAL_LOSS, 4=OVERRIDDEN, 5=FALLBACK, 6=HELD
    watchdog_sec: float = 1.0    # Black/Freeze/Silence detector threshold (s)
    console_view: bool = False   # False=PGM Output, True=Multi-Event Automation Matrix Console View

    @classmethod
    def from_params(cls, cfg: dict) -> "PlayoutAutomationV2Spec":
        c = cfg or {}
        mode = int(np.clip(round(float(c.get("auto_v2_mode", 0.0))), 0, 3))
        enabled = (float(c.get("auto_v2_enabled", 0.0)) > 0.5) or (mode > 0)
        return cls(
            enabled=enabled,
            mode=mode if mode > 0 else (1 if enabled else 0),
            fault_inject=int(np.clip(round(float(c.get("auto_v2_fault_inject", 0.0))), 0, 6)),
            watchdog_sec=float(np.clip(float(c.get("auto_v2_watchdog_sec", 1.0)), 0.2, 5.0)),
            console_view=float(c.get("auto_v2_console_view", 0.0)) > 0.5,
        )

    def is_active(self) -> bool:
        return bool(self.enabled and self.mode > 0)


def build_default_playlist() -> list[PlayoutEvent]:
    """
    Construct the canonical broadcast playlist from `README.txt` §5.1:
      19:00:00 News Studio -> 19:30:00 News Outro -> 19:30:15 Commercial Break
      -> 19:32:15 Program Return -> 19:32:30 Next Feature VTR.
    """
    return [
        PlayoutEvent(
            event_id="EVT-1900",
            scheduled_time=0.0,
            duration=3.0,
            source_id="STUDIO-A (LIVE CAM)",
            action="SWITCH_STUDIO_LIVE",
            fallback_id="BACKUP-IRD-1",
            status="SCHEDULED",
            title="19:00 EVENING NEWS LIVE",
            trigger_type="TIME",
        ),
        PlayoutEvent(
            event_id="EVT-1930",
            scheduled_time=3.0,
            duration=1.5,
            source_id="VTR-1 (BETACAM SP)",
            action="ROLL_VTR_OUTRO",
            fallback_id="STILL_STORE_1",
            status="SCHEDULED",
            title="19:30 NEWS OUTRO BUMPER",
            trigger_type="GPI_CONDITION",
        ),
        PlayoutEvent(
            event_id="EVT-1931",
            scheduled_time=4.5,
            duration=2.5,
            source_id="SPOT-SRV (MPEG2/DV)",
            action="PLAY_AD_BLOCK",
            fallback_id="PROMO_REEL_B",
            status="SCHEDULED",
            title="19:30:15 COMMERCIAL BREAK",
            trigger_type="TIME",
        ),
        PlayoutEvent(
            event_id="EVT-1933",
            scheduled_time=7.0,
            duration=2.0,
            source_id="MCR-NET-FEED",
            action="RETURN_NETWORK_PGM",
            fallback_id="SLATE_STANDBY",
            status="SCHEDULED",
            title="19:32:15 RETURN TO PGM CHAIN",
            trigger_type="TIME",
        ),
        PlayoutEvent(
            event_id="EVT-1935",
            scheduled_time=9.0,
            duration=5.0,
            source_id="VTR-2 (VHS ARCHIVE)",
            action="PREROLL_AND_TAKE_VTR",
            fallback_id="SLATE_STANDBY",
            status="SCHEDULED",
            title="19:32:30 FEATURE DOCUMENTARY",
            trigger_type="GPI_CONDITION",
        ),
    ]


def inspect_upstream_signal_health(
    rgb: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Infrastructure C Watchdog: inspect incoming video/audio/transport metadata and pixel statistics
    to detect Black Frame, Freeze/Static Snow, Sync Loss, VTR Transport Not-Ready, or Satellite Cliff Drop.
    """
    m = meta or {}
    lum = np.clip(rgb @ LUMA, 0.0, 1.0)
    mean_y = float(lum.mean())
    std_y = float(lum.std())

    fmt_flag = str(m.get("format", ""))
    no_sig = (fmt_flag == "no_signal") or (mean_y < 0.015 and std_y < 0.008)

    # Check VTR transport servo telemetry (`transport_servo`)
    servo_info = m.get("transport_servo") or {}
    t_state = str(servo_info.get("transport_state", "play"))
    servo_locked = bool(servo_info.get("servo_locked", True))
    vtr_not_ready = (t_state in ("pause_still", "stop_unthread")) or (not servo_locked)

    # Check Satellite IRD telemetry (`satellite_ird`)
    sat_info = m.get("satellite_ird") or {}
    sat_service_lost = bool(sat_info.get("cliff_service_lost", False))

    # Check RF link sync slip (`rf`)
    rf_info = m.get("rf") or {}
    rf_cn_db = float(rf_info.get("cn_db", 40.0))
    rf_unlock = rf_cn_db < 14.0

    # Determine root physical symptom
    if sat_service_lost:
        root_dev = "SAT_IRD_RECEIVER"
        symptom = f"DVB-S QPSK CLIFF UNLOCK (C/N={sat_info.get('cn_db', 0.0):.1f}dB, RS_ERR={sat_info.get('rs_uncorrected_pkts', 0)})"
        fault_code = "SIGNAL_LOSS"
    elif no_sig:
        root_dev = "UPSTREAM_FEED_BUS"
        symptom = f"BLACK_OR_CARRIER_LOSS (Y_mean={mean_y:.4f}, Y_std={std_y:.4f})"
        fault_code = "SIGNAL_LOSS"
    elif vtr_not_ready:
        root_dev = "VTR_TRANSPORT_SERVO"
        symptom = f"SERVO_NOT_LOCKED (state={t_state}, locked={servo_locked})"
        fault_code = "NOT_READY"
    elif rf_unlock:
        root_dev = "RF_TUNER_DEMOD"
        symptom = f"RF_CARRIER_BELOW_THRESHOLD (C/N={rf_cn_db:.1f}dB)"
        fault_code = "SIGNAL_LOSS"
    else:
        root_dev = "NONE"
        symptom = "SIGNAL_NOMINAL_LOCKED"
        fault_code = "OK"

    return {
        "healthy": fault_code == "OK",
        "fault_code": fault_code,
        "root_device": root_dev,
        "physical_symptom": symptom,
        "luma_mean": round(mean_y, 4),
        "luma_std": round(std_y, 4),
        "vtr_not_ready": vtr_not_ready,
        "sat_service_lost": sat_service_lost,
        "no_signal": no_sig,
    }


def _render_backup_program_feed(
    primary_rgb: np.ndarray,
    backup_rgb: np.ndarray | None,
    frame_index: int,
    std=None,
    status: str = "FALLBACK",
) -> np.ndarray:
    """
    Generate or select the redundant backup program feed when `FALLBACK`, `MISSING_ASSET`,
    `NOT_READY`, `SIGNAL_LOSS`, or `OVERRIDDEN` is active.
    """
    h, w = primary_rgb.shape[:2]
    if backup_rgb is not None and backup_rgb.shape[:2] == (h, w):
        out = np.asarray(backup_rgb, dtype=np.float32).copy()
    elif status == "OVERRIDDEN":
        # Operator Manual Override camera / studio test feed (PM5544 circle card + live studio tint)
        pm = generate_test_pattern(w, h, mode=5, frame_index=frame_index, std=std)
        out = np.clip(pm * 0.72 + primary_rgb * 0.28, 0.0, 1.0).astype(np.float32)
        scale = max(1, min(w // 180, h // 120))
        _draw_text_5x7(
            out, "MCR OPERATOR MANUAL OVERRIDE - LIVE STUDIO BUS",
            int(w * 0.06), int(h * 0.14),
            scale=scale, color=(0.98, 0.92, 0.22), bg=(0.38, 0.06, 0.06),
        )
    elif status == "NOT_READY":
        # VTR Pre-roll stalled / servo hunting leader
        out, _ = render_vtr_preroll_leader(primary_rgb, frame_index=frame_index)
        scale = max(1, min(w // 180, h // 120))
        _draw_text_5x7(
            out, "VTR NOT READY - SERVO LOCK TIMEOUT (HOLDING PVW)",
            int(w * 0.06), int(h * 0.82),
            scale=scale, color=(0.98, 0.36, 0.24), bg=(0.12, 0.04, 0.04),
        )
    else:
        # Emergency Standby Slate (`MISSING_ASSET`, `SIGNAL_LOSS`, `FALLBACK`)
        out = render_emergency_standby_slate(w, h, frame_index=frame_index, std=std)
        scale = max(1, min(w // 180, h // 120))
        _draw_text_5x7(
            out, f"AUTO-V2 STATUS: {status}",
            int(w * 0.16), int(h * 0.12),
            scale=scale, color=(0.98, 0.84, 0.22), bg=(0.14, 0.06, 0.06),
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def render_automation_v2_console(
    pgm_rgb: np.ndarray,
    pvw_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    playlist: list[PlayoutEvent],
    causal_log: list[CausalFaultRecord],
    active_event: PlayoutEvent,
    health: dict[str, Any],
    spec: PlayoutAutomationV2Spec,
) -> np.ndarray:
    """
    Render the Harris ADC-100 / Grass Valley Master Control Automation v2 Console (`auto_v2_console_view = 1`):
      - Top-Left: `PVW / PREPARING` monitor (with VTR servo lock meter)
      - Top-Center: `PGM / ON-AIR` monitor (with DSK bug & active status tally)
      - Top-Right: `CLEAN FEED / BACKUP` monitor + Watchdog Telemetry (`Y_mean`, `Servo`, `Failover`)
      - Middle: Multi-Event State Machine Playlist Table (`SCHEDULED -> PREPARING -> READY -> ON_AIR -> COMPLETED`
        or `MISSING_ASSET / NOT_READY / SIGNAL_LOSS / OVERRIDDEN / FALLBACK / HELD`)
      - Bottom: Infrastructure C Causal Fault Trace (`ROOT_DEVICE -> SYMPTOM -> ACTION`).
    """
    h, w = pgm_rgb.shape[:2]
    canvas = np.full((h, w, 3), (0.05, 0.06, 0.08), dtype=np.float32)
    pad = max(2, min(h, w) // 90)
    scale = max(1, min(w // 260, h // 175))

    # 1. Top 3 Monitor Windows (PVW, PGM, CLEAN/BACKUP)
    top_h = int(h * 0.46)
    mon_w = (w - 4 * pad) // 3
    mon_h = max(16, top_h - 2 * pad - 10 * scale)
    y_mon = pad + 9 * scale

    if mon_w > 16 and mon_h > 16:
        for idx_m, (img_m, title_m, tally_col) in enumerate([
            (pvw_rgb, "PVW / CUEING", (0.20, 0.88, 0.36)),
            (pgm_rgb, f"PGM [{active_event.status}]", (0.96, 0.20, 0.18)),
            (clean_rgb, "CLEAN / BACKUP", (0.28, 0.82, 0.96)),
        ]):
            x_m = pad + idx_m * (mon_w + pad)
            cell = _resample_nn(img_m, mon_h, mon_w)
            canvas[y_mon:y_mon + mon_h, x_m:x_m + mon_w] = cell
            canvas[pad:pad + 7 * scale + 2, x_m:x_m + mon_w] = (0.10, 0.12, 0.16)
            _draw_text_5x7(canvas, title_m[:18], x_m + 3, pad + 1, scale=scale, color=tally_col)

    # 2. Middle Playlist Event State Machine Table
    mid_y0 = top_h + pad
    mid_y1 = int(h * 0.78)
    canvas[mid_y0:mid_y1, pad:w - pad] = (0.08, 0.09, 0.12)
    mode_tag = AUTOMATION_MODES.get(spec.mode, "time_driven").upper()
    _draw_text_5x7(
        canvas,
        f"PLAYLIST STATE ENGINE ({mode_tag})  WATCHDOG:{'OK' if health['healthy'] else health['fault_code']}",
        pad + 4, mid_y0 + 2,
        scale=scale, color=(0.96, 0.84, 0.25),
    )

    row_h = max(8, 8 * scale + 1)
    status_colors = {
        "COMPLETED": (0.52, 0.56, 0.60),
        "ON_AIR": (0.24, 0.96, 0.42),
        "READY": (0.35, 0.88, 0.96),
        "PREPARING": (0.96, 0.82, 0.24),
        "SCHEDULED": (0.75, 0.78, 0.82),
        "MISSING_ASSET": (0.98, 0.28, 0.28),
        "NOT_READY": (0.98, 0.52, 0.20),
        "SIGNAL_LOSS": (0.98, 0.22, 0.22),
        "OVERRIDDEN": (0.96, 0.40, 0.92),
        "FALLBACK": (0.98, 0.72, 0.18),
        "HELD": (0.92, 0.65, 0.25),
    }
    for i_ev, ev in enumerate(playlist[:5]):
        ry = mid_y0 + 3 + (i_ev + 1) * row_h
        if ry + 7 * scale >= mid_y1:
            break
        col = status_colors.get(ev.status, (0.85, 0.85, 0.85))
        line_s = f"{ev.event_id} {ev.source_id[:12]:<12s} [{ev.status:<13s}] LOCK:{ev.servo_lock_pct:3.0f}% {ev.diagnostic_cause[:18]}"
        _draw_text_5x7(canvas, line_s[:max(20, (w - 2 * pad) // (6 * scale))], pad + 4, ry, scale=scale, color=col)

    # 3. Bottom Causal Fault Trace (Infrastructure C)
    bot_y0 = mid_y1 + pad
    if bot_y0 < h - pad:
        canvas[bot_y0:h - pad, pad:w - pad] = (0.06, 0.05, 0.07)
        _draw_text_5x7(
            canvas, "CAUSAL FAULT TRACE (INFRASTRUCTURE C - REVERSE DIAGNOSTICS):",
            pad + 4, bot_y0 + 2, scale=scale, color=(0.96, 0.52, 0.42),
        )
        for i_c, rec in enumerate(causal_log[-2:]):
            cy = bot_y0 + 3 + (i_c + 1) * row_h
            if cy + 7 * scale >= h - pad:
                break
            trace_line = f"{rec.timestamp_tc} {rec.root_device} -> {rec.physical_symptom[:22]} -> {rec.automation_status} ({rec.recovery_action})"
            _draw_text_5x7(
                canvas, trace_line[:max(20, (w - 2 * pad) // (6 * scale))],
                pad + 4, cy, scale=scale, color=(0.94, 0.88, 0.76),
            )

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def apply_playout_automation_v2(
    primary_rgb: np.ndarray,
    spec: PlayoutAutomationV2Spec,
    std=None,
    frame_index: int = 0,
    fps: float = 29.97,
    upstream_meta: dict[str, Any] | None = None,
    backup_rgb: np.ndarray | None = None,
    upstream_fault: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute the Event-Driven & Condition-Driven Broadcast Playout Automation Engine v2 (`Candidate L`).

    Returns `(output_rgb, automation_result_dict)`.
    """
    src_rgb = np.clip(np.asarray(primary_rgb, dtype=np.float32), 0.0, 1.0)
    if not spec.is_active():
        return src_rgb, {
            "enabled": False,
            "pgm_rgb": src_rgb,
            "clean_feed_rgb": src_rgb,
            "pvw_rgb": src_rgb,
            "report": {"enabled": False},
        }

    h, w = src_rgb.shape[:2]
    t_sec = float(frame_index) / max(1.0, float(fps))
    sec = int(t_sec)
    ff = int(frame_index % max(1, int(round(fps))))
    tc_now = f"19:30:{sec:02d}:{ff:02d}"

    # 1. Inspect physical upstream signal & device health
    eff_meta = dict(upstream_meta or {})
    if upstream_fault and eff_meta.get("format") != "no_signal":
        eff_meta["format"] = "no_signal"
    health = inspect_upstream_signal_health(src_rgb, eff_meta)

    # 2. Build schedule and evaluate state transitions (`SCHEDULED -> PREPARING -> READY -> ON_AIR -> COMPLETED`)
    playlist = build_default_playlist()
    # Map frame_index into active slot (Default: slot 1 `EVT-1930` is active around t=0..4s so single-frame preview shows rich transition)
    active_idx = min(len(playlist) - 2, max(1, int(frame_index // 4) % (len(playlist) - 1)))

    for idx, ev in enumerate(playlist):
        if idx < active_idx:
            ev.status = "COMPLETED"
            ev.actual_start_time = ev.scheduled_time
            ev.servo_lock_pct = 100.0
            ev.diagnostic_cause = "COMPLETED_ON_TIME"
        elif idx == active_idx:
            ev.status = "ON_AIR"
            ev.actual_start_time = ev.scheduled_time
            ev.servo_lock_pct = 100.0
            ev.diagnostic_cause = "LOCKED_ON_AIR"
        elif idx == active_idx + 1:
            # Next event is in PREPARING (VTR pre-roll) or READY depending on frame progress
            sub_f = int(frame_index) % 4
            if sub_f < 2:
                ev.status = "PREPARING"
                ev.servo_lock_pct = 58.0 + 18.0 * sub_f
                ev.diagnostic_cause = "VTR_PREROLL_CUEING"
            else:
                ev.status = "READY"
                ev.servo_lock_pct = 99.5
                ev.diagnostic_cause = "PREROLL_LOCKED_READY"
        else:
            ev.status = "SCHEDULED"
            ev.servo_lock_pct = 0.0
            ev.diagnostic_cause = "AWAITING_CUE_WINDOW"

    active_ev = playlist[active_idx]
    next_ev = playlist[min(len(playlist) - 1, active_idx + 1)]

    # 3. Determine active exception state (from explicit `fault_inject` OR physical upstream health)
    eff_fault = int(spec.fault_inject)
    if eff_fault == 0:
        if health["fault_code"] == "SIGNAL_LOSS":
            eff_fault = 5 if spec.mode == 3 else 3
        elif health["fault_code"] == "NOT_READY":
            eff_fault = 2
        elif spec.mode == 2 and (int(frame_index) % 6 == 5):
            eff_fault = 6  # GPI condition hold demonstration in mode 2

    causal_log: list[CausalFaultRecord] = [
        CausalFaultRecord(
            timestamp_tc="19:30:00:00",
            event_id=playlist[0].event_id,
            root_device="MCR_SPG_GENLOCK",
            physical_symptom="BLACKBURST_LOCKED_0.0DEG",
            watchdog_metric=f"Y={health['luma_mean']:.2f} SERVO=LOCKED",
            automation_status="ON_AIR",
            recovery_action="NOMINAL_PLAYOUT",
            black_gap_ms=0.0,
        )
    ]

    failover_active = False
    failover_gap_ms = 0.0

    if eff_fault == 1:
        # 1. MISSING_ASSET
        active_ev.status = "MISSING_ASSET"
        active_ev.servo_lock_pct = 0.0
        active_ev.diagnostic_cause = f"ASSET_NOT_FOUND ({active_ev.source_id})"
        next_ev.status = "READY"
        failover_active = True
        failover_gap_ms = 0.0  # Detected pre-air in bin check, zero black gap
        clean_rgb = _render_backup_program_feed(src_rgb, backup_rgb, frame_index, std=std, status="MISSING_ASSET")
        causal_log.append(CausalFaultRecord(
            timestamp_tc=tc_now,
            event_id=active_ev.event_id,
            root_device="MEDIA_ASSET_SERVER",
            physical_symptom="FILE_OR_CASSETTE_ID_MISSING",
            watchdog_metric="PREFLIGHT_CHECK=FAIL",
            automation_status="MISSING_ASSET",
            recovery_action=f"SKIP_TO_FALLBACK ({active_ev.fallback_id})",
            black_gap_ms=failover_gap_ms,
        ))
    elif eff_fault == 2:
        # 2. NOT_READY (VTR Pre-roll timeout / tape stiction)
        active_ev.status = "NOT_READY"
        active_ev.servo_lock_pct = 42.0
        active_ev.diagnostic_cause = "VTR_PREROLL_STICTION_TIMEOUT"
        failover_active = True
        failover_gap_ms = 40.0
        clean_rgb = _render_backup_program_feed(src_rgb, backup_rgb, frame_index, std=std, status="NOT_READY")
        causal_log.append(CausalFaultRecord(
            timestamp_tc=tc_now,
            event_id=active_ev.event_id,
            root_device="VTR_CAPSTAN_SERVO",
            physical_symptom=health["physical_symptom"] if health["vtr_not_ready"] else "PREROLL_CTL_PHASE_SLIP_>4FRM",
            watchdog_metric="SERVO_LOCK=42% (<95% REQ)",
            automation_status="NOT_READY",
            recovery_action="HOLD_PREVIOUS_OR_FILLER_SLATE",
            black_gap_ms=failover_gap_ms,
        ))
    elif eff_fault == 3:
        # 3. SIGNAL_LOSS (Active source lost carrier/sync/black)
        active_ev.status = "SIGNAL_LOSS"
        active_ev.servo_lock_pct = 0.0
        active_ev.diagnostic_cause = health["physical_symptom"] if not health["healthy"] else "UPSTREAM_CARRIER_LOSS"
        failover_active = True
        failover_gap_ms = round(float(spec.watchdog_sec) * 1000.0, 1)
        clean_rgb = _render_backup_program_feed(src_rgb, backup_rgb, frame_index, std=std, status="SIGNAL_LOSS")
        causal_log.append(CausalFaultRecord(
            timestamp_tc=tc_now,
            event_id=active_ev.event_id,
            root_device=health["root_device"] if not health["healthy"] else "UPSTREAM_VTR_OR_IRD",
            physical_symptom=active_ev.diagnostic_cause,
            watchdog_metric=f"BLACK/LOSS > {spec.watchdog_sec:.1f}s",
            automation_status="SIGNAL_LOSS",
            recovery_action="ALARM_AND_CUT_EMERGENCY_SLATE",
            black_gap_ms=failover_gap_ms,
        ))
    elif eff_fault == 4:
        # 4. OVERRIDDEN (Operator Manual Takeover)
        active_ev.status = "OVERRIDDEN"
        active_ev.diagnostic_cause = "MCR_DIRECTOR_MANUAL_TAKE"
        next_ev.status = "HELD"
        next_ev.diagnostic_cause = "HELD_BY_MANUAL_OVERRIDE"
        failover_active = True
        failover_gap_ms = 0.0
        clean_rgb = _render_backup_program_feed(src_rgb, backup_rgb, frame_index, std=std, status="OVERRIDDEN")
        causal_log.append(CausalFaultRecord(
            timestamp_tc=tc_now,
            event_id=active_ev.event_id,
            root_device="MCR_CONTROL_PANEL",
            physical_symptom="OPERATOR_MANUAL_CROSSPOINT_TAKE",
            watchdog_metric="AUTO_SEQ=SUSPENDED",
            automation_status="OVERRIDDEN",
            recovery_action="ROUTE_MANUAL_STUDIO_BUS_TO_PGM",
            black_gap_ms=0.0,
        ))
    elif eff_fault == 5:
        # 5. FALLBACK (Redundant Main -> Hot-Standby Backup Switchover)
        active_ev.status = "FALLBACK"
        active_ev.diagnostic_cause = f"FAILOVER_TO_{active_ev.fallback_id}"
        failover_active = True
        failover_gap_ms = 20.0  # half-frame vertical interval clean switch on hot-standby
        clean_rgb = _render_backup_program_feed(src_rgb, backup_rgb, frame_index, std=std, status="FALLBACK")
        causal_log.append(CausalFaultRecord(
            timestamp_tc=tc_now,
            event_id=active_ev.event_id,
            root_device=health["root_device"] if not health["healthy"] else "MAIN_PLAYOUT_CHANNEL_A",
            physical_symptom=health["physical_symptom"] if not health["healthy"] else "PRIMARY_FEED_WATCHDOG_TRIP",
            watchdog_metric="REDUNDANCY_SYNC=HOT_LOCKED",
            automation_status="FALLBACK",
            recovery_action=f"AUTO_SWITCH_BACKUP ({active_ev.fallback_id})",
            black_gap_ms=failover_gap_ms,
        ))
    elif eff_fault == 6:
        # 6. HELD (Condition / GPI Cue Hold & Schedule Ripple)
        active_ev.status = "HELD"
        active_ev.diagnostic_cause = "WAITING_GPI_LIVE_END_TRIGGER"
        next_ev.status = "PREPARING"
        next_ev.diagnostic_cause = "RIPPLED_+00:15s_BY_HOLD"
        for later_ev in playlist[active_idx + 1:]:
            later_ev.scheduled_time += 15.0
        clean_rgb = src_rgb.copy()
        causal_log.append(CausalFaultRecord(
            timestamp_tc=tc_now,
            event_id=active_ev.event_id,
            root_device="STUDIO_GPI_TALLY",
            physical_symptom="LIVE_SEGMENT_OVERRUN_NO_END_CUE",
            watchdog_metric="GPI_IN_1=LOW (HOLD)",
            automation_status="HELD",
            recovery_action="HOLD_CURRENT_AND_RIPPLE_PLAYLIST_+15S",
            black_gap_ms=0.0,
        ))
    else:
        clean_rgb = src_rgb.copy()

    # 4. Build Preview (`pvw_rgb`) and Program (`pgm_rgb` with DSK)
    pvw_rgb, preroll_info = render_vtr_preroll_leader(src_rgb, frame_index=frame_index, fps=fps)
    mp_spec = MasterPlayoutSpec(
        enabled=True,
        output_bus=0,
        playout_state=2 if failover_active else 0,
        dsk_enabled=True,
        dsk_crawl=True,
        failover_auto=True,
    )
    pgm_rgb = apply_dsk_overlay(clean_rgb, mp_spec, frame_index=frame_index, fps=fps, is_standby=failover_active)

    if spec.console_view:
        out_rgb = render_automation_v2_console(
            pgm_rgb=pgm_rgb,
            pvw_rgb=pvw_rgb,
            clean_rgb=clean_rgb,
            playlist=playlist,
            causal_log=causal_log,
            active_event=active_ev,
            health=health,
            spec=spec,
        )
    else:
        out_rgb = pgm_rgb

    status_counts: dict[str, int] = {}
    for ev in playlist:
        status_counts[ev.status] = status_counts.get(ev.status, 0) + 1

    report = {
        "enabled": True,
        "automation_mode": AUTOMATION_MODES.get(spec.mode, "time_driven_playlist"),
        "fault_inject_mode": FAULT_INJECT_NAMES.get(eff_fault, "NONE"),
        "active_event_id": active_ev.event_id,
        "active_event_status": active_ev.status,
        "active_diagnostic_cause": active_ev.diagnostic_cause,
        "next_event_id": next_ev.event_id,
        "next_event_status": next_ev.status,
        "failover_active": failover_active,
        "failover_gap_ms": failover_gap_ms,
        "clean_feed_untouched_by_dsk": True,
        "upstream_health": health,
        "status_counts": status_counts,
        "playlist": [e.to_dict() for e in playlist],
        "causal_trace": [c.to_dict() for c in causal_log],
        "preroll_telemetry": preroll_info,
    }
    return out_rgb, {
        "enabled": True,
        "pgm_rgb": pgm_rgb,
        "clean_feed_rgb": clean_rgb,
        "pvw_rgb": pvw_rgb,
        "report": report,
    }


class AutomationControllerStage(Stage):
    """
    Standalone Node Graph stage (`automation_controller`) for Candidate L.
    Supports `in` (Primary Program Feed) and optional `in_backup` (Redundant Backup Feed),
    and exposes `out` (PGM), `clean` (Clean Feed), and `pvw` (Preview Bus).
    """
    name = "automation_controller"
    layers = ("overlay",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("overlay")
        self.spec = PlayoutAutomationV2Spec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.mode <= 0:
                self.spec.mode = 3

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer, backup_buf: FrameBuffer | None = None) -> FrameBuffer:
        sig = buf.meta.get("signal")
        base_rgb = buf.rgb if (buf.rgb is not None and (buf.meta.get("from_display", False) or sig is None)) else sig.to_rgb()
        bkp_rgb = None
        if backup_buf is not None:
            b_sig = backup_buf.meta.get("signal")
            bkp_rgb = backup_buf.rgb if (backup_buf.rgb is not None and b_sig is None) else (b_sig.to_rgb() if b_sig is not None else None)

        out_rgb, res = apply_playout_automation_v2(
            base_rgb,
            self.spec,
            std=buf.std,
            frame_index=int(buf.index),
            fps=float(self.ctx.fps),
            upstream_meta=buf.meta,
            backup_rgb=bkp_rgb,
        )
        meta = dict(buf.meta)
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        meta["signal"] = sig_out
        meta["playout_automation_v2"] = res["report"]
        meta["playout_buses"] = {
            "pgm_rgb": res["pgm_rgb"],
            "clean_feed_rgb": res["clean_feed_rgb"],
            "pvw_rgb": res["pvw_rgb"],
        }
        causal = list(meta.get("causal_chain") or [])
        causal.extend(res["report"]["causal_trace"])
        meta["causal_chain"] = causal
        comp = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63) if buf.composite is not None else None
        return FrameBuffer(out_rgb, comp, buf.index, buf.time, buf.std, meta)

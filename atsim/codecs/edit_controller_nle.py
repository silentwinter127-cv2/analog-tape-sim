"""
Candidate K: Linear Tape Edit Controller & Early Non-Linear Editing (NLE) System (`edit_controller_nle`)
and Shared Infrastructure A (`EditDecision`, `EditSession`, Timebase & Asset Addressing).

Models the two fundamental editing paradigms described in `README.txt` Section I:

  1. **Linear Tape-to-Tape Editing (`Sony RM-450 / BVE-9100`, `edit_mode = 1, 2`)**:
     - Coordinates a `Player VTR` (Source) and a `Recorder VTR` (Record Master) via **Pre-roll**
       (`edit_preroll_sec`), capstan/drum servo phase lock, and Control Track (`CTL`) / SMPTE timecode addressing.
     - **Mechanical In/Out Point Slip (`edit_point_slip_frames`)**:
       Because pinch-roller engagement and CTL pulse counting have finite mechanical tolerance during
       pre-roll lockup, the actual recorded cut lands `±1..4 frames` away from the ideal EDL cut point.
     - **Mode 1 — Assemble Edit (`edit_mode = 1`)**:
       Switches the full-width stationary erase head, rotating video heads, and CTL control-track head
       into record simultaneously at the In-point. Produces an **In-point erase black flash + H-AFC hook**,
       and leaves a signature **Out-point CTL control-track break & rainbow tear (`tail_ctl_break`)**
       where the newly written control track ends and collides with old tape tracks.
     - **Mode 2 — Insert Edit (`edit_mode = 2`, Flying Erase Head `edit_flying_erase`)**:
       Preserves the existing continuous CTL control track on the destination tape (so there is **no
       Out-point CTL break or vertical roll**).
       * When `edit_flying_erase = 1` (Flying Erase Head on the rotating drum): old helical video tracks
         are erased cleanly track-by-track right ahead of the record head.
       * When `edit_flying_erase = 0` (Stationary erase head only / worn flying erase head): across the
         cut zone, the new video signal overwrites un-erased diagonal remnants of the outgoing shot,
         creating **diagonal ghost overlap + 180-deg subcarrier beat rainbow fringes**.

  2. **Early 2000s Non-Linear Editing (`Canopus DV Storm / Premiere 6.0 NLE`, `edit_mode = 3, 4`)**:
     - Separates non-destructive `EditSession` timeline metadata (`V1`, `V2`, `Title`, `A1`, `A2` tracks
       and `EditDecision` clips) from source media files, with exact `0-frame` addressing slip.
     - **Mode 3 — NLE Real-Time Timeline Preview (`edit_mode = 3`)**:
       During real-time playback over multi-track transitions/titles (`edit_nle_complexity`), CPU decode
       and IDE disk bandwidth saturate (`buffer_underrun`), causing:
         * Dynamic drop to **half/quarter-resolution nearest-neighbor preview proxy**, and
         * **Dropped / repeated frames (`preview_dropped_frame = True`)** where complex frames hold the
           previous frame buffer.
     - **Mode 4 — NLE Final Render Export (`edit_mode = 4`)**:
       Renders every frame at full spatial resolution with **zero dropped frames** (`Preview != Final Export`),
       while applying **2nd-generation 8x8 DCT re-quantization (`DV25 / MPEG-1`)** to rendered transition
       and title-composited segments (`reencoded_segment = True`).
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
from .dv import DVCodec, DVParams
from .multiview_wall import _draw_bitmap_text

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

EDIT_MODE_NAMES: dict[int, str] = {
    0: "off",
    1: "linear_assemble_edit",
    2: "linear_insert_edit",
    3: "nle_realtime_preview",
    4: "nle_final_export",
}


@dataclass
class EditDecision:
    """
    A single Edit Decision List (EDL) / timeline event (`README.txt` Section I.2).
    """
    source_id: str
    source_in: int
    source_out: int
    timeline_in: int
    track_id: str
    transition_id: str | None = None
    edit_type: str = "CUT"
    actual_in_slip_frames: int = 0
    actual_out_slip_frames: int = 0

    @property
    def timeline_out(self) -> int:
        return int(self.timeline_in + max(1, self.source_out - self.source_in))

    @property
    def effective_in(self) -> int:
        return max(0, int(self.timeline_in + self.actual_in_slip_frames))

    @property
    def effective_out(self) -> int:
        return max(self.effective_in + 1, int(self.timeline_out + self.actual_out_slip_frames))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_in": int(self.source_in),
            "source_out": int(self.source_out),
            "timeline_in": int(self.timeline_in),
            "timeline_out": int(self.timeline_out),
            "effective_in": int(self.effective_in),
            "effective_out": int(self.effective_out),
            "track_id": self.track_id,
            "transition_id": self.transition_id,
            "edit_type": self.edit_type,
            "actual_in_slip_frames": int(self.actual_in_slip_frames),
            "actual_out_slip_frames": int(self.actual_out_slip_frames),
        }


@dataclass
class EditSession:
    """
    Shared editing project & timeline model (`README.txt` Section I.2 & Infrastructure A).
    """
    video_tracks: list[str]
    audio_tracks: list[str]
    decisions: list[EditDecision]
    frame_rate: float = 29.97
    timebase: str = "SMPTE_DROP_FRAME"
    engine_mode: str = "linear_assemble_edit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_mode": self.engine_mode,
            "video_tracks": list(self.video_tracks),
            "audio_tracks": list(self.audio_tracks),
            "frame_rate": round(float(self.frame_rate), 3),
            "timebase": self.timebase,
            "decisions": [d.to_dict() for d in self.decisions],
        }


@dataclass
class EditControllerSpec:
    """Configuration of the Linear Tape Edit Controller & Early NLE System (`tape` layer)."""
    edit_mode: int = 0                # 0=Off, 1=Linear Assemble, 2=Linear Insert, 3=NLE Preview, 4=NLE Final Export
    point_slip_frames: int = 2        # 0..4 frames of mechanical CTL/capstan slip at linear In/Out points
    flying_erase: bool = True         # True=Flying Erase Head clean track erase, False=Stationary erase overlap
    preroll_sec: float = 5.0          # 1..8 s dual-VTR servo pre-roll duration
    nle_complexity: float = 0.65      # 0..1 multi-track timeline load (drives preview proxy & dropped frames)
    console_osd: bool = False         # True=Overlay Sony RM-450 / BVE-9100 or NLE Timeline monitor HUD

    @classmethod
    def from_params(cls, cfg: dict | None) -> "EditControllerSpec":
        c = cfg or {}
        return cls(
            edit_mode=int(np.clip(round(float(c["edit_mode"])) if "edit_mode" in c else 0, 0, 4)),
            point_slip_frames=int(np.clip(round(float(c["edit_point_slip_frames"])) if "edit_point_slip_frames" in c else 2, 0, 4)),
            flying_erase=float(c["edit_flying_erase"]) > 0.5 if "edit_flying_erase" in c else True,
            preroll_sec=float(np.clip(float(c["edit_preroll_sec"]) if "edit_preroll_sec" in c else 5.0, 1.0, 8.0)),
            nle_complexity=float(np.clip(float(c["edit_nle_complexity"]) if "edit_nle_complexity" in c else 0.65, 0.0, 1.0)),
            console_osd=float(c["edit_console_osd"]) > 0.5 if "edit_console_osd" in c else False,
        )

    def is_active(self) -> bool:
        return int(self.edit_mode) > 0


def _draw_text_box(
    canvas: np.ndarray,
    text: str,
    x0: int,
    y0: int,
    scale: int = 1,
    color: tuple[float, float, float] = (0.94, 0.96, 0.94),
    bg: tuple[float, float, float] | None = (0.05, 0.07, 0.11),
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


def _synthesize_b_roll_shot(rgb_a: np.ndarray, b_yiq: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """
    Obtain Incoming Shot B (`V2 / Player VTR B-Roll`) either from a connected `in_b` node port
    or by synthesizing a distinct B-Roll camera angle (zoomed + warm amber studio grade + horizontal flip).
    """
    if b_yiq is not None:
        yb, ib, qb = b_yiq
        if yb.shape == rgb_a.shape[:2]:
            return np.clip(yiq_to_rgb(yb, ib, qb), 0.0, 1.0).astype(np.float32)

    h, w = rgb_a.shape[:2]
    # Create a visually & numerically distinct Shot B (center 1.35x crop + warm tint + inverted luma contrast)
    y0, y1 = int(h * 0.12), max(int(h * 0.12) + 4, int(h * 0.88))
    x0, x1 = int(w * 0.12), max(int(w * 0.12) + 4, int(w * 0.88))
    crop = rgb_a[y0:y1, x0:x1, ::-1].copy()  # channel swap / distinct colour signature for clear cut verification
    ys = np.clip((np.arange(h) * crop.shape[0]) // max(1, h), 0, crop.shape[0] - 1)
    xs = np.clip((np.arange(w) * crop.shape[1]) // max(1, w), 0, crop.shape[1] - 1)
    shot_b = crop[ys][:, xs]
    # Add a distinct studio B-Roll grade so Shot A and Shot B differ strongly (> 0.12 L1)
    shot_b = np.clip(shot_b * np.array([1.12, 0.94, 0.78], dtype=np.float32) + np.array([0.06, 0.02, -0.04], dtype=np.float32), 0.0, 1.0)
    return shot_b.astype(np.float32)


def build_default_edit_session(
    spec: EditControllerSpec,
    fps: float = 29.97,
) -> EditSession:
    """
    Construct the canonical `EditSession` with `EditDecision` entries for the current editing mode.
    In linear tape modes (`edit_mode in (1, 2)`), `actual_in_slip_frames = spec.point_slip_frames`;
    in NLE modes (`edit_mode in (3, 4)`), file-indexed timeline cuts have `actual_in_slip_frames = 0`.
    """
    is_linear = spec.edit_mode in (1, 2)
    slip = int(spec.point_slip_frames) if is_linear else 0
    ed_type = (
        "ASSEMBLE" if spec.edit_mode == 1
        else ("INSERT_V" if spec.edit_mode == 2 else ("NLE_PREVIEW" if spec.edit_mode == 3 else "NLE_EXPORT"))
    )
    decisions = [
        EditDecision(
            source_id="VTR_A_MASTER" if is_linear else "CLIP_001_A.AVI",
            source_in=0,
            source_out=2,
            timeline_in=0,
            track_id="V1",
            transition_id=None,
            edit_type="BASE_TRACK",
            actual_in_slip_frames=0,
            actual_out_slip_frames=0,
        ),
        EditDecision(
            source_id="VTR_B_PLAYER" if is_linear else "CLIP_002_B_ROLL.AVI",
            source_in=10,
            source_out=14,
            timeline_in=2,
            track_id="V1" if is_linear else "V2",
            transition_id=None if is_linear else "DISSOLVE_CROSS_2F",
            edit_type=ed_type,
            actual_in_slip_frames=slip,
            actual_out_slip_frames=slip,
        ),
    ]
    return EditSession(
        video_tracks=["V1"] if is_linear else ["V1", "V2", "TITLE_1"],
        audio_tracks=["A1_LINEAR", "A2_LINEAR"] if is_linear else ["A1_PCM", "A2_PCM"],
        decisions=decisions,
        frame_rate=fps,
        timebase="CTL_PULSES_29_97" if is_linear else "NLE_SAMPLE_INDEX",
        engine_mode=EDIT_MODE_NAMES.get(spec.edit_mode, "off"),
    )


def _apply_linear_tape_editor(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    spec: EditControllerSpec,
    session: EditSession,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute Linear Tape-to-Tape Editing (`edit_mode == 1` Assemble Edit or `edit_mode == 2` Insert Edit).
    """
    h, w = rgb_a.shape[:2]
    scale_w = w / 720.0
    scale_h = h / 480.0
    rng = np.random.default_rng((int(seed) * 49157 + int(frame_index) * 613 + spec.edit_mode * 19) & 0xFFFFFFFF)

    insert_dec = session.decisions[1]
    # Periodic 8-frame edit cycle so any scrubbed preview frame or sequence demonstrates the cut window
    cycle_frame = int(frame_index) % 8
    eff_in = int(insert_dec.effective_in)    # ideal_in (2) + point_slip_frames (0..4)
    eff_out = min(7, int(eff_in + 3))

    in_edit_segment = (cycle_frame >= eff_in) and (cycle_frame < eff_out)
    at_in_point = (cycle_frame == eff_in)
    at_out_point = (cycle_frame == eff_out)
    active_shot = "SHOT_B_INCOMING" if in_edit_segment else "SHOT_A_OUTGOING"
    base_rgb = rgb_b.copy() if in_edit_segment else rgb_a.copy()

    rows = np.arange(h, dtype=np.float32)
    yn = rows / max(1.0, h - 1.0)
    assemble_flash_active = False
    tail_ctl_break_active = False
    flying_erase_ghost_active = False
    servo_lock_err_px = 0.0

    if spec.edit_mode == 1:
        # --- MODE 1: ASSEMBLE EDIT (Full-drum + stationary erase + CTL rewrite) ---
        if at_in_point:
            # At Assemble In-point: stationary erase head sits several cm upstream of rotating drum,
            # leaving a momentary black erase gap + capstan/drum phase lock hook on the upper field!
            assemble_flash_active = True
            flash_band = np.clip((0.32 - yn) / 0.06, 0.0, 1.0)[:, None, None]
            base_rgb = base_rgb * (1.0 - 0.92 * flash_band) + 0.02 * flash_band
            hook_shift = np.exp(-yn / 0.14) * (14.0 * scale_w) + rng.standard_normal(h).astype(np.float32) * (1.8 * scale_w)
            y, i, q = rgb_to_yiq(base_rgb)
            y = horizontal_shift(y, hook_shift)
            # Subcarrier burst phase jump at assemble splice
            ph_step = np.exp(-yn / 0.20) * 2.4
            cs, sn = np.cos(ph_step)[:, None].astype(np.float32), np.sin(ph_step)[:, None].astype(np.float32)
            i, q = i * cs - q * sn, i * sn + q * cs
            base_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0)
            servo_lock_err_px = float(np.std(hook_shift))
        elif at_out_point:
            # At Assemble Out-point: stopping an Assemble Edit leaves a physical CTL control-track gap
            # and un-erased track collision ("Assemble Tail Break")!
            tail_ctl_break_active = True
            y, i, q = rgb_to_yiq(base_rgb)
            tear_shift = np.sin(yn * 18.0 + frame_index) * (18.0 * scale_w) + rng.standard_normal(h).astype(np.float32) * (4.5 * scale_w)
            y = horizontal_shift(y, tear_shift)
            i = horizontal_shift(i, tear_shift)
            q = horizontal_shift(q, tear_shift)
            # Band of FM limiter snow where the old and new control tracks collide
            snow_band = np.exp(-0.5 * ((yn - 0.62) / 0.12) ** 2)[:, None]
            snow_noise = lowpass(rng.standard_normal((h, w)).astype(np.float32), max(0.8, 1.6 * scale_w), axis=1)
            y = y * (1.0 - 0.78 * snow_band) + (0.48 + 0.28 * snow_noise) * (0.78 * snow_band)
            i = i * (1.0 - snow_band)
            q = q * (1.0 - snow_band)
            base_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0)
            servo_lock_err_px = float(np.std(tear_shift))
        elif in_edit_segment:
            # Normal assemble-recorded segment (slight pre-roll servo settling residual)
            jit = rng.standard_normal(h).astype(np.float32) * (0.35 * max(0.2, 6.0 - spec.preroll_sec) * scale_w)
            y, i, q = rgb_to_yiq(base_rgb)
            y = horizontal_shift(y, jit)
            base_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0)
            servo_lock_err_px = float(np.std(jit))

    elif spec.edit_mode == 2:
        # --- MODE 2: INSERT EDIT (Preserves existing continuous CTL track; uses Flying Erase Head) ---
        if in_edit_segment or at_out_point:
            if not spec.flying_erase:
                # Without a working Flying Erase Head: new helical video tracks overwrite un-erased
                # diagonal remnants of Shot A, producing diagonal ghost overlap + 180-deg chroma beat rainbow!
                flying_erase_ghost_active = True
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                diag_phase = (yy / max(1.0, h)) * 3.2 + (xx / max(1.0, w)) * 0.85 - (cycle_frame - eff_in) * 0.55
                ghost_env = np.clip(0.55 * (0.5 + 0.5 * np.sin(diag_phase * 2.0 * np.pi)) * np.exp(-0.35 * max(0, cycle_frame - eff_in)), 0.0, 0.65)
                base_rgb = base_rgb * (1.0 - 0.42 * ghost_env[..., None]) + rgb_a * (0.42 * ghost_env[..., None])

                # 180-deg colour-under subcarrier beat rainbow across the un-erased overlap
                y, i, q = rgb_to_yiq(base_rgb)
                beat_angle = (yy * 0.28 + frame_index * 1.1) * ghost_env
                cs = np.cos(beat_angle).astype(np.float32)
                sn = np.sin(beat_angle).astype(np.float32)
                i_b = i * cs - q * sn + 0.14 * ghost_env * np.cos(yy * 0.35)
                q_b = i * sn + q * cs + 0.14 * ghost_env * np.sin(yy * 0.35)
                base_rgb = np.clip(yiq_to_rgb(y, i_b, q_b), 0.0, 1.0)
                servo_lock_err_px = 0.45
            else:
                # Healthy Flying Erase Head + continuous CTL track: clean frame-accurate insert!
                jit = rng.standard_normal(h).astype(np.float32) * (0.12 * scale_w)
                y, i, q = rgb_to_yiq(base_rgb)
                y = horizontal_shift(y, jit)
                base_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0)
                servo_lock_err_px = float(np.std(jit))

    report = {
        "active": True,
        "edit_mode": EDIT_MODE_NAMES[spec.edit_mode],
        "active_shot": active_shot,
        "cycle_frame": cycle_frame,
        "ideal_in_frame": int(insert_dec.timeline_in),
        "effective_in_frame": eff_in,
        "effective_out_frame": eff_out,
        "point_slip_frames": int(insert_dec.actual_in_slip_frames),
        "preroll_sec": round(float(spec.preroll_sec), 2),
        "flying_erase_enabled": bool(spec.flying_erase),
        "assemble_flash_active": assemble_flash_active,
        "tail_ctl_break_active": tail_ctl_break_active,
        "flying_erase_ghost_active": flying_erase_ghost_active,
        "servo_lock_err_px": round(servo_lock_err_px, 3),
        "preview_dropped_frame": False,
        "reencoded_segment": False,
        "session": session.to_dict(),
    }
    return base_rgb.astype(np.float32), report


def _apply_early_nle_timeline(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    spec: EditControllerSpec,
    session: EditSession,
    std=None,
    frame_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute Early 2000s Non-Linear Editor (`edit_mode == 3` Real-Time Preview vs `edit_mode == 4` Final Export).

    Key physical distinction (`README.txt` Section I.1 & I.2):
      - Both modes share the exact same non-destructive `EditSession` timeline (`in_slip == 0`).
      - **Mode 3 (Real-Time Preview)**: When multi-track transitions/titles exceed real-time CPU/disk budget
        (`edit_nle_complexity`), the preview engine drops to coarse proxy resolution and drops/repeats frames
        (`preview_dropped_frame = True`).
      - **Mode 4 (Final Render Export)**: Renders every frame at full spatial resolution with zero dropped
        frames, but re-encodes transition/title segments through 8x8 DV25 DCT quantization (`reencoded_segment = True`).
    """
    from ..core.standards import get_standard
    std_obj = std if std is not None else get_standard("ntsc")
    h, w = rgb_a.shape[:2]
    cycle_frame = int(frame_index) % 8
    dec = session.decisions[1]
    eff_in = int(dec.timeline_in)     # 0 slip in NLE!
    eff_out = int(dec.timeline_out)   # 6

    in_b_segment = (cycle_frame >= eff_in) and (cycle_frame < eff_out)
    in_transition = cycle_frame in (eff_in, eff_in + 1, eff_out - 1)
    complexity = float(np.clip(spec.nle_complexity, 0.0, 1.0))

    # 1. Multi-track timeline compositing (V1 + V2 Dissolve + Title Lower-Third)
    if cycle_frame == eff_in:
        alpha = 0.45
        comp_rgb = rgb_a * (1.0 - alpha) + rgb_b * alpha
    elif cycle_frame == eff_in + 1:
        alpha = 0.82
        comp_rgb = rgb_a * (1.0 - alpha) + rgb_b * alpha
    elif in_b_segment:
        comp_rgb = rgb_b.copy()
    else:
        comp_rgb = rgb_a.copy()

    # Non-destructive NLE Title Track (`TITLE_1`) active across the B-roll segment
    if in_b_segment and h >= 64 and w >= 96:
        ty0, ty1 = int(h * 0.78), int(h * 0.87)
        tx0, tx1 = int(w * 0.08), int(w * 0.68)
        comp_rgb[ty0:ty1, tx0:tx1] = comp_rgb[ty0:ty1, tx0:tx1] * 0.35 + np.array([0.08, 0.16, 0.36], dtype=np.float32) * 0.65
        _draw_text_box(
            comp_rgb, "NLE TRACK V2 + TITLE_1 OVERLAY",
            tx0 + 6, ty0 + max(2, (ty1 - ty0 - 7) // 2),
            scale=max(1, min(w // 320, h // 240)),
            color=(0.96, 0.92, 0.38), bg=None,
        )

    preview_dropped = False
    proxy_factor = 1
    reencoded_seg = False
    dct_err = 0.0

    if spec.edit_mode == 3:
        # --- MODE 3: NLE REAL-TIME TIMELINE PREVIEW ---
        # Buffer underrun & draft proxy resolution under multi-track load
        if complexity >= 0.20 and (in_b_segment or in_transition):
            proxy_factor = 2 if complexity < 0.55 else 4
            # Downsample and nearest-neighbor blocky upsample (Draft Preview Proxy)
            ph = max(8, h // proxy_factor)
            pw = max(8, w // proxy_factor)
            small = comp_rgb[::proxy_factor, ::proxy_factor]
            ys = np.clip((np.arange(h) * small.shape[0]) // max(1, h), 0, small.shape[0] - 1)
            xs = np.clip((np.arange(w) * small.shape[1]) // max(1, w), 0, small.shape[1] - 1)
            comp_rgb = small[ys][:, xs].copy()

            # Disk / CPU preview cache underrun: freeze/hold outgoing frame on transition spikes
            if complexity >= 0.35 and (cycle_frame % 2 == 0):
                preview_dropped = True
                # Repeat the pre-transition cached frame (Shot A proxy) with subtle buffer-stutter line hold
                small_a = rgb_a[::proxy_factor, ::proxy_factor]
                comp_rgb = small_a[ys][:, xs].copy()

            # Unrendered red/yellow timeline indicator line at the very top of the preview monitor
            comp_rgb[:max(2, h // 90), :, :] = (
                np.array([0.92, 0.18, 0.14], dtype=np.float32)
                if preview_dropped else np.array([0.95, 0.78, 0.16], dtype=np.float32)
            )
    else:
        # --- MODE 4: NLE FINAL RENDER EXPORT ---
        # Zero dropped frames, full spatial resolution, but rendered transitions/titles undergo
        # 2nd-generation DV25 8x8 DCT re-compression!
        if in_b_segment or in_transition:
            reencoded_seg = True
            q_eff = max(0.32, 0.85 - 0.45 * complexity)
            dv = DVCodec(std_obj, DVParams(
                chroma_mode="411",
                quality=q_eff,
                base_step=0.018 * (1.0 + 0.95 * complexity),
            ))
            pre_dct = comp_rgb.copy()
            comp_rgb = dv.encode_decode(SignalFrame.from_rgb(comp_rgb, std_obj)).to_rgb()
            dct_err = float(np.mean(np.abs(comp_rgb - pre_dct)))

    report = {
        "active": True,
        "edit_mode": EDIT_MODE_NAMES[spec.edit_mode],
        "active_shot": "SHOT_B_INCOMING" if in_b_segment else "SHOT_A_OUTGOING",
        "cycle_frame": cycle_frame,
        "ideal_in_frame": eff_in,
        "effective_in_frame": eff_in,
        "effective_out_frame": eff_out,
        "point_slip_frames": 0,
        "preroll_sec": 0.0,
        "flying_erase_enabled": True,
        "assemble_flash_active": False,
        "tail_ctl_break_active": False,
        "flying_erase_ghost_active": False,
        "servo_lock_err_px": 0.0,
        "preview_dropped_frame": preview_dropped,
        "preview_proxy_factor": proxy_factor,
        "reencoded_segment": reencoded_seg,
        "reencode_dct_l1": round(dct_err, 5),
        "session": session.to_dict(),
    }
    return np.clip(comp_rgb, 0.0, 1.0).astype(np.float32), report


def _render_edit_console_osd(
    rgb: np.ndarray,
    spec: EditControllerSpec,
    report: dict[str, Any],
    frame_index: int = 0,
) -> np.ndarray:
    """
    Overlay the Sony BVE-9100 / RM-450 Linear Edit Controller HUD (`edit_mode in (1, 2)`)
    or Early 2000s NLE Timeline HUD (`edit_mode in (3, 4)`) when `spec.console_osd` is enabled.
    """
    if not spec.console_osd:
        return rgb
    out = rgb.copy()
    h, w = out.shape[:2]
    if h < 64 or w < 96:
        return out
    sc = max(1, min(w // 340, h // 240))
    bar_h = max(24, int(h * 0.13))
    out[:bar_h, :] = out[:bar_h, :] * 0.22 + np.array([0.05, 0.08, 0.13], dtype=np.float32) * 0.78
    out[bar_h - 1:bar_h, :] = np.array([0.32, 0.62, 0.88], dtype=np.float32)

    tc_str = f"01:00:{int(frame_index // 30):02d}:{int(frame_index % 30):02d}"
    if spec.edit_mode in (1, 2):
        mode_lbl = "SONY BVE-9100 ASSEMBLE" if spec.edit_mode == 1 else "SONY BVE-9100 INSERT"
        feh_lbl = "FEH:ON" if spec.flying_erase else "FEH:OFF(OVERLAP)"
        line1 = f"{mode_lbl}  TC:{tc_str}  SLIP:{report['point_slip_frames']:+d}F  {feh_lbl}"
        _draw_text_box(out, line1, 6, 5, scale=sc, color=(0.42, 0.95, 0.58), bg=None)
    else:
        mode_lbl = "NLE REALTIME PREVIEW" if spec.edit_mode == 3 else "NLE FINAL EXPORT"
        drop_lbl = "DROP:YES(STUTTER)" if report.get("preview_dropped_frame") else f"PROXY:1/{report.get('preview_proxy_factor', 1)}"
        line1 = f"{mode_lbl}  TC:{tc_str}  SLIP:0F  {drop_lbl}"
        _draw_text_box(out, line1, 6, 5, scale=sc, color=(0.95, 0.82, 0.35), bg=None)
    return out


def apply_edit_controller(
    rgb_in: np.ndarray,
    spec: EditControllerSpec,
    std=None,
    frame_index: int = 0,
    fps: float = 29.97,
    seed: int = 20240101,
    bus_b_yiq: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Main entry point for Candidate K: Linear Tape Editor (`edit_mode = 1, 2`)
    and Early Non-Linear Editor (`edit_mode = 3, 4`).
    """
    if not spec.is_active():
        return rgb_in, {"active": False, "edit_mode": "off"}

    session = build_default_edit_session(spec, fps=fps)
    shot_a = np.clip(rgb_in, 0.0, 1.0).astype(np.float32)
    shot_b = _synthesize_b_roll_shot(shot_a, b_yiq=bus_b_yiq)

    if spec.edit_mode in (1, 2):
        out_rgb, rep = _apply_linear_tape_editor(
            shot_a, shot_b, spec, session, frame_index=frame_index, seed=seed
        )
    else:
        out_rgb, rep = _apply_early_nle_timeline(
            shot_a, shot_b, spec, session, std=std, frame_index=frame_index
        )

    out_rgb = _render_edit_console_osd(out_rgb, spec, rep, frame_index=frame_index)
    return out_rgb, rep


class LinearEditorStage(Stage):
    """
    Standalone node-graph stage (`linear_editor`) for Sony RM-450 / BVE-9100 Tape-to-Tape Linear Editing.
    Accepts `in_a` (Recorder Master / Outgoing Shot A) and optional `in_b` (Player Source / Incoming Shot B).
    """
    name = "linear_editor"
    layers = ("tape",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("tape")
        self.spec = EditControllerSpec.from_params(cfg)
        if self.spec.edit_mode not in (1, 2):
            self.spec.edit_mode = 1

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig = buf.meta["signal"] if "signal" in buf.meta else None
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        b_yiq = buf.meta["bus_b_yiq"] if "bus_b_yiq" in buf.meta else None
        out_rgb, rep = apply_edit_controller(
            rgb_in, self.spec, std=buf.std, frame_index=int(buf.index),
            fps=self.ctx.fps, seed=self.ctx.seed, bus_b_yiq=b_yiq,
        )
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        comp_out = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63)
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["edit_controller"] = rep
        return FrameBuffer(out_rgb, comp_out, buf.index, buf.time, buf.std, meta)


class NLETimelineStage(Stage):
    """
    Standalone node-graph stage (`nle_timeline`) for Early 2000s Non-Linear Editing (`Premiere 6 / Canopus DV Storm`).
    Accepts `in_a` (Track V1) and optional `in_b` (Track V2).
    """
    name = "nle_timeline"
    layers = ("tape",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("tape")
        self.spec = EditControllerSpec.from_params(cfg)
        if self.spec.edit_mode not in (3, 4):
            self.spec.edit_mode = 3

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig = buf.meta["signal"] if "signal" in buf.meta else None
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        b_yiq = buf.meta["bus_b_yiq"] if "bus_b_yiq" in buf.meta else None
        out_rgb, rep = apply_edit_controller(
            rgb_in, self.spec, std=buf.std, frame_index=int(buf.index),
            fps=self.ctx.fps, seed=self.ctx.seed, bus_b_yiq=b_yiq,
        )
        sig_out = SignalFrame.from_rgb(out_rgb, buf.std)
        comp_out = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63)
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["edit_controller"] = rep
        return FrameBuffer(out_rgb, comp_out, buf.index, buf.time, buf.std, meta)

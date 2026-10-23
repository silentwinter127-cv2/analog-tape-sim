"""
Candidate H: Studio Camera Control Unit (CCU) Rack & Master Sync Generator (Blackburst SPG / House Genlock).

Models the physical engineering chain of a multi-camera 1990s television studio:
  1. **Multi-Camera Unit Characteristics (`ccu_cam_id`)**:
     Different studio cameras (e.g., Sony BVP-370 reference, Ikegami HL-55 warm ENG unit,
     BTS LDK-90 cool wide camera, or an aged high-dark-current roving camera) exhibit distinct
     colorimetric matrices, pedestal offsets, and highlight knee responses under the same studio lighting.
  2. **Centralized CCU Remote Control Panel (RCP) Painting (`apply_studio_ccu`)**:
     The video engineer (Shader / VOs) remotely trims over the multi-core / triax camera cable:
       - Remote Iris (`ccu_iris_ev`, f-stop exposure);
       - Master Pedestal (`ccu_master_pedestal_ire`, overall black setup in IRE);
       - Red / Blue White Balance Gains (`ccu_r_gain`, `ccu_b_gain`);
       - Red / Blue Black Balance / Optical Flare Compensation (`ccu_r_flare`, `ccu_b_flare`);
       - Broadcast High-Headroom Knee Point & Slope (`ccu_knee_point`, `ccu_knee_slope`),
         compressing 400%-600% studio specular highlights smoothly into the 100 IRE ceiling.
  3. **Master Sync Generator (Blackburst SPG / House Genlock)**:
     Distributes a house Blackburst (BB) reference (H-Sync, V-Sync, and 3.58/4.43 MHz Colour Burst)
     to all cameras and VTRs:
       - **Triax / Coaxial Cable Length Delay (`genlock_cable_len_m`)**:
         Velocity factor $v_p = 0.66 c \\approx 1.98 \\times 10^8\\,\\text{m/s}$ produces a propagation
         delay of $5.05\\,\\text{ns/m}$, shifting subcarrier phase by $6.51^\\circ/\\text{m}$ (NTSC)
         or $8.06^\\circ/\\text{m}$ (PAL) and delaying H-Sync timing by $\\Delta x_{\\text{px}}$,
         while long multi-core cables introduce $\\sqrt{f}$ high-frequency roll-off.
       - **CCU Subcarrier & Horizontal Phase Trim (`genlock_sc_phase_deg`, `genlock_h_phase_px`)**:
         Allows the engineer to phase-align all cameras to $0^\\circ$ at the production switcher input.
       - **Genlock Unlock & Free-Run Crystal Drift (`genlock_mode`)**:
         `0 = Locked & Auto-Timed`, `1 = Cable Delay Uncompensated`,
         `2 = Free-Run Internal Crystal Drift` (continuous H/V crawl + 360-deg burst rotation),
         `3 = Intermittent BNC Genlock Drop` (periodic H-AFC hook & burst shock).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode, horizontal_shift, lowpass, rgb_to_yiq, vertical_shift, yiq_to_rgb
from ..core.standards import VideoStandard

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

CAMERA_UNIT_PROFILES: dict[int, dict] = {
    0: {
        "name": "CAM-1 SONY BVP-370 (Studio D65 Reference)",
        "rgb_gain": (1.00, 1.00, 1.00),
        "rgb_flare": (0.000, 0.000, 0.000),
        "gamma": 1.00,
    },
    1: {
        "name": "CAM-2 Ikegami HL-55 (Warm ENG / High Contrast)",
        "rgb_gain": (1.14, 0.99, 0.86),
        "rgb_flare": (0.024, 0.004, -0.016),
        "gamma": 1.06,
    },
    2: {
        "name": "CAM-3 BTS LDK-90 (Cool Wide / Cyan Shadow Toe)",
        "rgb_gain": (0.88, 1.02, 1.15),
        "rgb_flare": (-0.018, 0.010, 0.028),
        "gamma": 0.95,
    },
    3: {
        "name": "CAM-4 Aged Roving CCD (Elevated Dark Pedestal & Green Drift)",
        "rgb_gain": (0.94, 1.12, 0.90),
        "rgb_flare": (0.032, 0.048, 0.018),
        "gamma": 0.92,
    },
}

GENLOCK_MODE_NAMES: dict[int, str] = {
    0: "bb_locked_autotimed",
    1: "cable_delay_uncompensated",
    2: "freerun_crystal_drift",
    3: "intermittent_bnc_unlock",
}


@dataclass
class StudioCCUSpec:
    """Parameters of the Studio CCU Rack & House Genlock system (`camera` layer)."""
    enabled: bool = False
    cam_id: int = 0
    iris_ev: float = 0.0
    master_pedestal_ire: float = 0.0
    r_gain: float = 1.0
    b_gain: float = 1.0
    r_flare: float = 0.0
    b_flare: float = 0.0
    knee_point: float = 0.82
    knee_slope: float = 0.35
    genlock_mode: int = 0
    cable_len_m: float = 15.0
    sc_phase_deg: float = 0.0
    h_phase_px: float = 0.0

    @classmethod
    def from_params(cls, cfg: dict) -> "StudioCCUSpec":
        c = cfg or {}
        return cls(
            enabled=float(c["ccu_enabled"]) > 0.5 if "ccu_enabled" in c else False,
            cam_id=int(np.clip(round(float(c["ccu_cam_id"])) if "ccu_cam_id" in c else 0, 0, 3)),
            iris_ev=float(np.clip(float(c["ccu_iris_ev"]) if "ccu_iris_ev" in c else 0.0, -2.5, 2.5)),
            master_pedestal_ire=float(np.clip(float(c["ccu_master_pedestal_ire"]) if "ccu_master_pedestal_ire" in c else 0.0, -15.0, 25.0)),
            r_gain=float(np.clip(float(c["ccu_r_gain"]) if "ccu_r_gain" in c else 1.0, 0.5, 1.6)),
            b_gain=float(np.clip(float(c["ccu_b_gain"]) if "ccu_b_gain" in c else 1.0, 0.5, 1.6)),
            r_flare=float(np.clip(float(c["ccu_r_flare"]) if "ccu_r_flare" in c else 0.0, -0.25, 0.25)),
            b_flare=float(np.clip(float(c["ccu_b_flare"]) if "ccu_b_flare" in c else 0.0, -0.25, 0.25)),
            knee_point=float(np.clip(float(c["ccu_knee_point"]) if "ccu_knee_point" in c else 0.82, 0.50, 0.98)),
            knee_slope=float(np.clip(float(c["ccu_knee_slope"]) if "ccu_knee_slope" in c else 0.35, 0.05, 1.0)),
            genlock_mode=int(np.clip(round(float(c["genlock_mode"])) if "genlock_mode" in c else 0, 0, 3)),
            cable_len_m=float(np.clip(float(c["genlock_cable_len_m"]) if "genlock_cable_len_m" in c else 15.0, 0.0, 150.0)),
            sc_phase_deg=float(np.clip(float(c["genlock_sc_phase_deg"]) if "genlock_sc_phase_deg" in c else 0.0, -180.0, 180.0)),
            h_phase_px=float(np.clip(float(c["genlock_h_phase_px"]) if "genlock_h_phase_px" in c else 0.0, -24.0, 24.0)),
        )


def compute_cable_genlock_timing(
    spec: StudioCCUSpec,
    std: VideoStandard,
    width: int,
    frame_index: int = 0,
    fps: float = 29.97,
) -> dict:
    """
    Compute the coaxial/triax cable propagation delay (ns), subcarrier phase rotation (deg),
    and horizontal line timing shift (px) relative to House Blackburst (SPG).
    """
    # Propagation delay in RG-59 / Studio Triax cable (v_p = 0.66c -> 5.05 ns/m)
    tau_cable_ns = float(spec.cable_len_m) * 5.05
    f_sc_hz = float(std.subcarrier_hz)
    # Uncompensated cable subcarrier phase lag (deg): 360 * f_sc * tau
    cable_sc_lag_deg = (360.0 * f_sc_hz * (tau_cable_ns * 1e-9))
    # Uncompensated horizontal timing delay in active pixels
    px_rate_hz = float(std.pixel_rate_hz) * (float(width) / max(1.0, float(std.samples_per_line)))
    cable_h_delay_px = (tau_cable_ns * 1e-9) * px_rate_hz

    mode = int(spec.genlock_mode)
    v_roll_rows = 0
    is_locked = True

    if mode == 0:
        # Locked to House Blackburst with automatic cable timing; only manual RCP phase trims apply
        net_sc_deg = float(spec.sc_phase_deg)
        net_h_px = float(spec.h_phase_px)
    elif mode == 1:
        # Cable delay uncompensated: full cable phase lag + manual RCP trim
        net_sc_deg = float(cable_sc_lag_deg + spec.sc_phase_deg)
        net_h_px = float(cable_h_delay_px + spec.h_phase_px)
    elif mode == 2:
        # Free-Run Crystal Drift: internal camera XO drifts against house SPG
        is_locked = False
        t_s = float(frame_index) / max(1e-6, float(fps))
        net_sc_deg = float((spec.sc_phase_deg + 115.0 * t_s + 48.0 * frame_index) % 360.0 - 180.0)
        net_h_px = float(spec.h_phase_px + 14.0 * np.sin(0.85 * frame_index + 0.4))
        v_roll_rows = int(round((frame_index * 19 + 28) % 120))
    else:
        # Intermittent BNC Genlock unlock (every other frame pair experiences AFC hook & burst slip)
        is_locked = (int(frame_index) % 3 == 0)
        if is_locked:
            net_sc_deg = float(spec.sc_phase_deg)
            net_h_px = float(spec.h_phase_px)
        else:
            net_sc_deg = float(spec.sc_phase_deg + 82.0 * (1.0 if frame_index % 2 == 0 else -1.0))
            net_h_px = float(spec.h_phase_px + 9.5)
            v_roll_rows = 14

    # Wrap subcarrier phase into [-180, 180]
    net_sc_deg = float(((net_sc_deg + 180.0) % 360.0) - 180.0)
    return {
        "genlock_mode": mode,
        "genlock_mode_name": GENLOCK_MODE_NAMES.get(mode, "bb_locked_autotimed"),
        "genlock_locked": bool(is_locked),
        "cable_len_m": round(float(spec.cable_len_m), 2),
        "cable_delay_ns": round(float(tau_cable_ns), 2),
        "cable_sc_lag_deg": round(float(cable_sc_lag_deg), 2),
        "cable_h_delay_px": round(float(cable_h_delay_px), 3),
        "net_sc_phase_deg": round(float(net_sc_deg), 2),
        "net_h_shift_px": round(float(net_h_px), 3),
        "v_roll_rows": int(v_roll_rows),
    }


def apply_studio_ccu(
    rgb: np.ndarray,
    spec: StudioCCUSpec,
    std: VideoStandard,
    frame_index: int = 0,
    fps: float = 29.97,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict]:
    """
    Apply Studio Camera Control Unit (CCU) remote painting and House Blackburst Genlock timing.
    """
    if not spec.enabled:
        return rgb, {"enabled": False}

    h, w = rgb.shape[:2]
    out = np.asarray(rgb, dtype=np.float32).copy()

    # 1. Multi-Camera Unit Baseline Personality (`ccu_cam_id = 0..3`)
    unit = CAMERA_UNIT_PROFILES.get(int(spec.cam_id), CAMERA_UNIT_PROFILES[0])
    u_gain = np.array(unit["rgb_gain"], dtype=np.float32)[None, None, :]
    u_flare = np.array(unit["rgb_flare"], dtype=np.float32)[None, None, :]
    u_gamma = float(unit["gamma"])
    if abs(u_gamma - 1.0) > 1e-3:
        out = np.power(np.clip(out, 0.0, 2.0), u_gamma).astype(np.float32)
    out = out * u_gain + u_flare

    # 2. Remote CCU Iris f-stop control (`ccu_iris_ev`)
    if abs(spec.iris_ev) > 1e-4:
        out = out * float(2.0 ** spec.iris_ev)

    # 3. CCU White Balance Gains (`ccu_r_gain`, `ccu_b_gain`) & Black Balance / Flare (`ccu_r_flare`, `ccu_b_flare`)
    # plus Master Pedestal (`ccu_master_pedestal_ire` in IRE, where 100 IRE = 1.0)
    ped_norm = float(spec.master_pedestal_ire) / 100.0
    wb_gains = np.array([float(spec.r_gain), 1.0, float(spec.b_gain)], dtype=np.float32)[None, None, :]
    flare_offs = np.array([
        ped_norm + float(spec.r_flare),
        ped_norm,
        ped_norm + float(spec.b_flare),
    ], dtype=np.float32)[None, None, :]
    out = out * wb_gains + flare_offs

    # 4. Broadcast CCU Dynamic Highlight Knee (`ccu_knee_point`, `ccu_knee_slope`)
    kp = float(np.clip(spec.knee_point, 0.50, 0.98))
    ks = float(np.clip(spec.knee_slope, 0.05, 1.0))
    over = np.maximum(0.0, out - kp)
    # Smooth hyperbolic knee compression above `knee_point`
    headroom = max(0.02, 1.0 - kp)
    compressed_over = headroom * np.tanh((over * ks) / headroom)
    out = np.where(out <= kp, out, kp + compressed_over)

    # 5. Multi-Core Umbilical Cable High-Frequency Skin-Effect Loss
    if spec.cable_len_m > 20.0:
        sigma_cable = float(np.clip((spec.cable_len_m - 20.0) / 85.0, 0.0, 1.8)) * (w / 720.0)
        if sigma_cable > 0.25:
            out = lowpass(out, sigma_cable, axis=1)

    # 6. House Blackburst Genlock Subcarrier Phase & H/V Timing Alignment
    timing = compute_cable_genlock_timing(spec, std, width=w, frame_index=frame_index, fps=fps)
    sc_deg = float(timing["net_sc_phase_deg"])
    h_px = float(timing["net_h_shift_px"])
    v_roll = int(timing["v_roll_rows"])

    if abs(sc_deg) > 0.15:
        y_p, i_p, q_p = rgb_to_yiq(np.clip(out, 0.0, 1.2))
        rad = np.deg2rad(sc_deg)
        c_p, s_p = float(np.cos(rad)), float(np.sin(rad))
        i_rot = i_p * c_p - q_p * s_p
        q_rot = i_p * s_p + q_p * c_p
        out = yiq_to_rgb(y_p, i_rot, q_rot)

    if abs(h_px) > 0.05 or spec.genlock_mode in (2, 3):
        shifts = np.full(h, h_px, dtype=np.float32)
        if spec.genlock_mode == 3 and not timing["genlock_locked"]:
            # Top H-AFC pull-in hook during intermittent genlock loss
            rows = np.arange(h, dtype=np.float32)
            shifts = shifts + 16.0 * np.exp(-rows / max(4.0, h * 0.14))
        elif spec.genlock_mode == 2:
            # Free-run crystal drift: slight diagonal line phase skew across the field
            rows = np.arange(h, dtype=np.float32)
            shifts = shifts + np.linspace(-5.0, 5.0, h, dtype=np.float32)
        out = horizontal_shift(out, shifts, mode="edge")
        # Expose authentic left/right HBI blanking border when H-phase shifts the active raster
        blank_px = int(min(w // 4, abs(round(h_px))))
        if blank_px >= 1:
            hbi_black = np.clip(0.02 + ped_norm * 0.3, 0.0, 0.12)
            if h_px > 0:
                out[:, :blank_px, :] = hbi_black
            else:
                out[:, w - blank_px:, :] = hbi_black

    if v_roll > 0 and h >= 32:
        out = vertical_shift(out, float(v_roll), mode="wrap")
        # Draw visible VBI sync serration bar at the rolled boundary
        vbi_y0 = v_roll % h
        vbi_h = max(4, min(12, h // 24))
        y1 = min(h, vbi_y0 + vbi_h)
        out[vbi_y0:y1, :, :] = 0.03
        # White VITC timecode pulses on the first rolled VBI row
        rng = np.random.default_rng((int(seed) + int(frame_index) * 37) & 0xFFFFFFFF)
        vitc_bits = (rng.random(max(8, w // 6)) > 0.45).astype(np.float32)
        vitc_row = np.repeat(vitc_bits, 6)[:w]
        out[vbi_y0, :, :] = vitc_row[:, None] * 0.78

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    y_final = out @ LUMA
    info = {
        "enabled": True,
        "cam_id": int(spec.cam_id),
        "cam_name": unit["name"],
        "iris_ev": round(float(spec.iris_ev), 3),
        "master_pedestal_ire": round(float(spec.master_pedestal_ire), 2),
        "r_gain": round(float(spec.r_gain), 3),
        "b_gain": round(float(spec.b_gain), 3),
        "r_flare": round(float(spec.r_flare), 3),
        "b_flare": round(float(spec.b_flare), 3),
        "knee_point": round(float(spec.knee_point), 3),
        "knee_slope": round(float(spec.knee_slope), 3),
        "mean_luma_ire": round(float(y_final.mean() * 100.0), 2),
        "p02_black_ire": round(float(np.percentile(y_final, 2.0) * 100.0), 2),
        "p99_white_ire": round(float(np.percentile(y_final, 99.0) * 100.0), 2),
        "r_over_g": round(float(out[..., 0].mean() / max(1e-5, out[..., 1].mean())), 4),
        "b_over_g": round(float(out[..., 2].mean() / max(1e-5, out[..., 1].mean())), 4),
        **timing,
    }
    return out, info


class CCURackStage(Stage):
    """
    Standalone Node-Graph Stage (`ccu_rack`) for Studio Camera Control Unit & Blackburst Genlock.
    """
    name = "ccu_rack"
    layers = ("camera",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = dict(self.cfg_of("camera") or {})
        c["ccu_enabled"] = 1.0
        self.spec = StudioCCUSpec.from_params(c)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig_in: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig_in.to_rgb() if (sig_in is not None and buf.rgb is None) else np.clip(buf.rgb, 0.0, 1.5)
        rgb_out, info = apply_studio_ccu(
            rgb_in, self.spec, buf.std,
            frame_index=int(buf.index), fps=self.ctx.fps, seed=int(self.ctx.seed) + 101,
        )
        sig_out = SignalFrame.from_rgb(rgb_out, buf.std)
        comp = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63) if buf.composite is not None else None
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["ccu_genlock"] = info
        meta["tap_camera_rgb"] = rgb_out
        return FrameBuffer(rgb_out, comp, buf.index, buf.time, buf.std, meta)

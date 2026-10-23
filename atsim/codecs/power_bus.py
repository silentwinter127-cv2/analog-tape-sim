"""
`atsim.codecs.power_bus` -- Shared Power & Electrical Bus System (Module 3).

Models the physical power supply, battery discharge curve, dynamic load-current voltage sag,
rectifier capacitor ripple, cold startup surge, and brush-motor / flyback EMI crosstalk as a
single shared electrical bus (`PowerBusState`) that couples Camera, Tape Servo, Player Sync,
CRT Display, and Audio together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PowerBusSpec:
    """Electrical bus parameters, stored in the `connector` layer."""
    psu_ripple: float = 0.0          # 0..1: Electrolytic filter cap ESR/aging (50/60Hz & 100/120Hz ripple)
    battery_level: float = 1.0       # 0..1: Battery State of Charge (1.0 = full, 0.0 = depleted)
    battery_sag: float = 0.0         # 0..1: Battery/supply internal resistance R_int
    load_coupling: float = 0.0       # 0..1.5: Cross-device dynamic load current coupling strength
    startup_surge: float = 0.0       # 0..1: Cold power-on inrush & servo spin-up transient (0..2.4s)
    emi_crosstalk: float = 0.0       # 0..1: Motor commutator & flyback EMI rail coupling
    mains_hz: float = 50.0           # 50 Hz (PAL regions) or 60 Hz (NTSC regions)

    @classmethod
    def from_params(cls, params: dict | None) -> "PowerBusSpec":
        p = params or {}
        conn = p.get("connector") if isinstance(p.get("connector"), dict) else p
        return cls(
            psu_ripple=float(np.clip(conn.get("psu_ripple", 0.0), 0.0, 1.0)),
            battery_level=float(np.clip(conn.get("battery_level", 1.0), 0.0, 1.0)),
            battery_sag=float(np.clip(conn.get("battery_sag", 0.0), 0.0, 1.0)),
            load_coupling=float(np.clip(conn.get("load_coupling", 0.0), 0.0, 1.5)),
            startup_surge=float(np.clip(conn.get("startup_surge", 0.0), 0.0, 1.0)),
            emi_crosstalk=float(np.clip(conn.get("emi_crosstalk", 0.0), 0.0, 1.0)),
            mains_hz=float(np.clip(conn.get("mains_hz", 50.0), 45.0, 65.0)),
        )

    def is_active(self) -> bool:
        return (
            self.psu_ripple > 1e-4
            or self.battery_level < 0.999
            or self.battery_sag > 1e-4
            or self.load_coupling > 1e-4
            or self.startup_surge > 1e-4
            or self.emi_crosstalk > 1e-4
        )


@dataclass
class PowerBusState:
    """Evaluated per-frame and per-scanline shared electrical state."""
    active: bool = False
    v_oc: float = 1.0                # Open-circuit supply voltage (nominal 1.0)
    r_int: float = 0.0               # Effective internal resistance (normalized ohms)
    i_load_mean: float = 0.25        # Mean load current across the frame
    v_bus_mean: float = 1.0          # Mean bus voltage after load sag & ripple
    v_drop_mean: float = 0.0         # Mean voltage deficit max(0, 1.0 - v_bus_mean)
    v_bus_row: np.ndarray = field(default_factory=lambda: np.ones(1, np.float32))
    ripple_row: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    emi_row: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    cam_exposure_stops: float = 0.0
    cam_noise_boost: float = 0.0
    cam_warm_drift: float = 0.0
    servo_speed_drop: float = 0.0
    servo_line_shift: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    sync_thresh_shift: float = 0.0
    crt_scale_sag: float = 0.0
    audio_hum_1f: float = 0.0
    audio_buzz_2f: float = 0.0
    audio_emi_whine: float = 0.0
    mains_hz: float = 50.0

    @property
    def audio_pitch_ratio(self) -> float:
        return float(np.clip(1.0 - self.servo_speed_drop, 0.55, 1.15))

    @property
    def audio_hum_amp(self) -> float:
        return float(np.clip(self.audio_hum_1f + self.audio_buzz_2f + 0.5 * self.audio_emi_whine, 0.0, 1.5))

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "v_oc": round(float(self.v_oc), 4),
            "r_int": round(float(self.r_int), 4),
            "i_load_mean": round(float(self.i_load_mean), 4),
            "v_bus_mean": round(float(self.v_bus_mean), 4),
            "v_drop_mean": round(float(self.v_drop_mean), 4),
            "cam_exposure_stops": round(float(self.cam_exposure_stops), 4),
            "servo_speed_drop": round(float(self.servo_speed_drop), 4),
            "sync_thresh_shift": round(float(self.sync_thresh_shift), 4),
            "crt_scale_sag": round(float(self.crt_scale_sag), 4),
        }


def shepherd_battery_curve(soc: float, base_sag: float) -> tuple[float, float]:
    s = float(np.clip(soc, 0.0, 1.0))
    deficit = 1.0 - s
    knee = max(0.0, 0.34 - s) / 0.34
    v_oc = 1.0 - 0.055 * deficit - 0.145 * (knee ** 1.75)
    r_soc = 0.025 * (deficit ** 1.5) + 0.095 * (knee ** 2.0)
    r_int = float(np.clip(base_sag * (0.14 + 0.18 * deficit) + r_soc, 0.0, 0.65))
    return float(np.clip(v_oc, 0.72, 1.05)), r_int


def evaluate_power_bus(
    params: dict | None,
    frame_index: int,
    fps: float,
    height: int,
    width: int = 720,
    seed: int = 20240101,
    row_luma: np.ndarray | None = None,
    stiction_loss: float = 0.0,
    zoom_vel: float = 0.0,
) -> PowerBusState:
    spec = PowerBusSpec.from_params(params)
    H = max(2, int(height))
    if not spec.is_active():
        return PowerBusState(active=False, mains_hz=spec.mains_hz)

    p = params or {}
    cam_cfg = p.get("camera") or {}
    tape_cfg = p.get("tape") or {}
    disp_cfg = p.get("display") or {}

    t_s = float(frame_index) / max(1e-6, float(fps))
    v_oc, r_int = shepherd_battery_curve(spec.battery_level, spec.battery_sag)

    nv_mode = int(round(float(cam_cfg.get("nv_mode", 0.0))))
    ir_lamp = float(cam_cfg.get("nv_illuminator", 0.0)) if nv_mode in (1, 2) else 0.0
    wt_zoom = float(cam_cfg.get("wt_zoom", 0.0))
    zoom_dyn = float(cam_cfg.get("zoom_dynamic_mode", 0.0))
    zoom_trav = float(cam_cfg.get("zoom_travel", 0.0))
    motor_zoom_act = float(np.clip(zoom_vel * 1.5 + (0.45 if zoom_dyn > 0 else 0.0) + 0.35 * zoom_trav + 0.15 * wt_zoom, 0.0, 1.2))
    i_cam = 0.12 + 0.34 * np.clip(ir_lamp, 0.0, 2.0) + 0.24 * motor_zoom_act

    shuttle = float(tape_cfg.get("shuttle_speed", 0.0))
    wear = float(tape_cfg.get("wear", 0.25))
    stiction_r = float(tape_cfg.get("stiction_rate", 0.0))
    breakage = float(tape_cfg.get("base_breakage", 0.0))
    servo_pulse = 0.06 * np.sin(2.0 * np.pi * t_s * 2.3 + 0.7) + 0.04 * np.cos(2.0 * np.pi * t_s * 5.8)
    stiction_spike = float(stiction_loss) * 0.75
    if stiction_spike <= 1e-4 and stiction_r > 1e-3:
        phase_st = (t_s * max(0.3, stiction_r) + 0.37) % 1.0
        if phase_st < 0.18:
            stiction_spike = float(tape_cfg.get("stiction_amount", 0.6)) * 0.55 * (1.0 - phase_st / 0.18)
    # Module 4: Mechanical transport state machine & belt/reel torque load on shared power bus
    t_state = int(round(float(tape_cfg.get("transport_state", 0.0))))
    belt_slip_load = float(np.clip(tape_cfg.get("belt_slip", 0.0), 0.0, 1.5)) * 0.22
    reel_prog = float(np.clip(tape_cfg.get("reel_progress", 0.50), 0.0, 1.0))
    reel_end_torque = abs(reel_prog - 0.50) * 2.0 * float(np.clip(tape_cfg.get("tension_ripple", 0.0), 0.0, 1.5)) * 0.18
    solenoid_load = 0.28 if t_state in (2, 3, 4) else (0.12 if t_state == 1 else 0.0)
    conn_p = params.get("connector") or {}
    sw_mode = int(round(float(conn_p.get("switcher_mode", 0.0))))
    switcher_load = 0.16 if sw_mode == 1 else (0.10 if sw_mode > 1 else 0.0)
    da_mode = int(round(float(conn_p.get("da_termination_mode", 0.0))))
    da_n = max(1, int(round(float(conn_p.get("da_load_count", 2.0)))))
    da_load = (0.055 * da_n) if da_mode == 3 else ((0.075 * (da_n - 1)) if da_mode == 2 else 0.0)
    i_servo_base = (
        0.16
        + 0.36 * np.clip(shuttle / 5.0, 0.0, 1.2)
        + 0.18 * wear
        + 0.28 * breakage
        + stiction_spike
        + belt_slip_load
        + reel_end_torque
        + solenoid_load
        + switcher_load
        + da_load
        + max(0.0, servo_pulse)
    )

    disp_on = float(disp_cfg.get("enabled", 1.0)) > 0.5
    if row_luma is not None and len(row_luma) == H:
        y_lum = np.clip(np.asarray(row_luma, np.float32), 0.0, 1.2)
    else:
        y_lum = np.full(H, 0.38, dtype=np.float32)
    i_crt_row = (0.10 + 0.44 * (y_lum ** 1.25)) if disp_on else (0.04 + 0.08 * y_lum)

    coup = float(np.clip(spec.load_coupling, 0.0, 1.5))
    eff_coup = max(coup, 0.65 if (spec.battery_sag > 1e-3 or spec.battery_level < 0.99) else 0.25)
    i_dynamic_row = i_cam + i_servo_base + i_crt_row
    i_total_row = (0.22 + eff_coup * i_dynamic_row).astype(np.float32)
    i_load_mean = float(i_total_row.mean())

    rows_norm = np.arange(H, dtype=np.float32) / float(H)
    t_row = t_s + rows_norm / max(1e-6, float(fps))
    f1 = float(spec.mains_hz)
    f2 = 2.0 * f1

    ph2 = 2.0 * np.pi * f2 * t_row
    ph1 = 2.0 * np.pi * f1 * t_row + 0.45
    rect_2f = 0.65 * np.sin(ph2) - 0.25 * np.sin(2.0 * ph2 + 0.4) + 0.10 * np.cos(3.0 * ph2)
    rect_1f = np.sin(ph1)
    load_rip_scale = 0.70 + 0.65 * i_total_row
    ripple_amp = spec.psu_ripple * load_rip_scale
    ripple_row = (ripple_amp * (0.68 * rect_2f + 0.32 * rect_1f)).astype(np.float32)

    startup_droop = 0.0
    startup_flash = 0.0
    startup_servo_lag = 0.0
    if spec.startup_surge > 1e-4 and t_s < 3.0:
        su = float(spec.startup_surge)
        env_fast = float(np.exp(-t_s / 0.42))
        env_med = float(np.exp(-t_s / 0.95))
        startup_droop = su * (0.34 * env_fast - 0.14 * env_med * float(np.sin(2.0 * np.pi * 1.4 * t_s)))
        startup_flash = su * 0.45 * env_med * float(np.sin(2.0 * np.pi * 1.6 * t_s))
        startup_servo_lag = su * 0.12 * env_med

    sag_drop_row = i_total_row * r_int
    v_bus_row = np.clip(
        v_oc - sag_drop_row - startup_droop + 0.18 * ripple_row,
        0.28,
        1.15,
    ).astype(np.float32)
    v_bus_mean = float(v_bus_row.mean())
    v_drop_row = np.maximum(0.0, 1.0 - v_bus_row)
    v_drop_mean = float(v_drop_row.mean())

    emi_row = np.zeros(H, dtype=np.float32)
    if spec.emi_crosstalk > 1e-4:
        rng = np.random.default_rng((int(seed) * 1664525 + int(frame_index) * 1013904223) & 0xFFFFFFFF)
        motor_intensity = spec.emi_crosstalk * (0.45 + 0.85 * (motor_zoom_act + stiction_spike + np.clip(shuttle / 5.0, 0.0, 1.0)))
        comm_freq = 18.0 + 14.0 * motor_zoom_act
        comm_wave = np.sin(2.0 * np.pi * rows_norm * comm_freq + frame_index * 1.7)
        burst_gate = (rng.random(H) < (0.18 * min(1.0, motor_intensity))).astype(np.float32)
        emi_row = (motor_intensity * (0.045 * comm_wave + 0.11 * burst_gate * rng.standard_normal(H))).astype(np.float32)

    scale_w = float(width) / 720.0
    cam_exposure_stops = float(np.clip(-1.45 * v_drop_mean + startup_flash, -2.0, 1.2))
    cam_noise_boost = float(np.clip(0.38 * v_drop_mean + 0.24 * spec.psu_ripple + 0.18 * spec.emi_crosstalk, 0.0, 0.85))
    cam_warm_drift = float(np.clip(0.42 * v_drop_mean, 0.0, 0.45))

    servo_speed_drop = float(np.clip(0.16 * v_drop_mean + startup_servo_lag, 0.0, 0.25))
    hum_snake_px = ripple_row * (14.5 * scale_w)
    load_tear_px = (v_drop_row - v_drop_mean) * (28.0 * scale_w) + (startup_servo_lag * 42.0 * scale_w) * np.sin(2.0 * np.pi * rows_norm)
    emi_jitter_px = emi_row * (9.0 * scale_w)
    servo_line_shift = (hum_snake_px + load_tear_px + emi_jitter_px).astype(np.float32)

    sync_thresh_shift = float(np.clip(0.72 * v_drop_mean + 0.38 * spec.psu_ripple + 0.35 * abs(startup_droop), 0.0, 0.95))
    crt_scale_sag = float(np.clip(0.55 * v_drop_mean + 0.20 * spec.psu_ripple, 0.0, 0.85))

    audio_hum_1f = float(np.clip(0.35 * spec.psu_ripple * (0.7 + 0.6 * i_load_mean), 0.0, 1.0))
    audio_buzz_2f = float(np.clip(0.65 * spec.psu_ripple * (0.7 + 0.8 * i_load_mean) + 0.25 * v_drop_mean, 0.0, 1.0))
    audio_emi_whine = float(np.clip(spec.emi_crosstalk * (0.5 + 0.7 * (motor_zoom_act + i_servo_base)), 0.0, 1.0))

    return PowerBusState(
        active=True,
        v_oc=v_oc,
        r_int=r_int,
        i_load_mean=i_load_mean,
        v_bus_mean=v_bus_mean,
        v_drop_mean=v_drop_mean,
        v_bus_row=v_bus_row,
        ripple_row=ripple_row,
        emi_row=emi_row,
        cam_exposure_stops=cam_exposure_stops,
        cam_noise_boost=cam_noise_boost,
        cam_warm_drift=cam_warm_drift,
        servo_speed_drop=servo_speed_drop,
        servo_line_shift=servo_line_shift,
        sync_thresh_shift=sync_thresh_shift,
        crt_scale_sag=crt_scale_sag,
        audio_hum_1f=audio_hum_1f,
        audio_buzz_2f=audio_buzz_2f,
        audio_emi_whine=audio_emi_whine,
        mains_hz=spec.mains_hz,
    )


def apply_camera_power_effects(rgb: np.ndarray, pstate: PowerBusState, frame_index: int, seed: int) -> np.ndarray:
    if not pstate.active:
        return rgb
    out = np.asarray(rgb, np.float32).copy()
    h, w = out.shape[:2]

    if abs(pstate.cam_exposure_stops) > 1e-4 or pstate.cam_warm_drift > 1e-4:
        gain = float(2.0 ** pstate.cam_exposure_stops)
        wd = float(pstate.cam_warm_drift)
        ch_gain = np.array([gain * (1.0 + 0.14 * wd), gain * (1.0 - 0.02 * wd), gain * (1.0 - 0.18 * wd)], dtype=np.float32)
        out = out * ch_gain[None, None, :]

    if len(pstate.ripple_row) == h:
        hum_bar = (0.085 * pstate.ripple_row)[:, None, None]
        out = out + hum_bar

    if len(pstate.emi_row) == h and np.any(np.abs(pstate.emi_row) > 1e-4):
        rng = np.random.default_rng((int(seed) * 214013 + int(frame_index) * 2531011) & 0xFFFFFFFF)
        hash_carrier = np.sin(2.0 * np.pi * np.arange(w, dtype=np.float32)[None, :] / max(3.0, w / 42.0) + rng.uniform(0, 6.28, size=(h, 1)).astype(np.float32))
        emi_2d = (pstate.emi_row[:, None] * hash_carrier)[..., None]
        out = out + emi_2d * np.array([0.85, 1.0, 0.90], dtype=np.float32)

    if pstate.cam_noise_boost > 1e-4:
        rng_n = np.random.default_rng((int(seed) * 747796405 + int(frame_index) * 2891336453) & 0xFFFFFFFF)
        n = rng_n.standard_normal((h, w, 1)).astype(np.float32) * (0.045 * pstate.cam_noise_boost)
        out = out + n

    return np.clip(out, 0.0, 2.0).astype(np.float32)

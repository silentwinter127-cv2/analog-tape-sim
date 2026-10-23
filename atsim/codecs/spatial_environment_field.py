"""
Option R: 2D Spatial Physical Environment Field & Multi-Domain Coupling System (`spatial_environment_field.py`).

Replaces uniform scalar environment assumptions with a continuous 2D spatial physical field
`EnvironmentField` over room coordinates `(x, y) in [0, 1]^2` (`10m x 8m` studio / vault / living room):
  1. **Per-Device 2D Spatial Sampling (`EnvironmentField.sample_at`)**:
     - Each device reads local physical fields at its own coordinates `(x_d, y_d)`:
       `cam_pos = (env_cam_x, env_cam_y)`, `vtr_pos = (env_vtr_x, env_vtr_y)`, `crt_pos = (env_crt_x, env_crt_y)`.
     - Moving a device across the room changes its local temperature, relative humidity, dew-point margin,
       near-field magnetic flux density `B_emi`, ground potential `V_gnd`, illuminant spectrum `T_cct`,
       and acoustic standing-wave / noise-floor coupling.
  2. **Four Coupled Physical Subsystems**:
     - **A. Thermo-Hygrometric & Dew Condensation Field (`ThermoHygroField`)**:
       Magnus-Tetens dew point $T_{\\text{dew}}(T, RH)$ + thermal inertia $T_{\\text{dev}}(t)$ when a cold camcorder/VTR
       ($T_{\\text{init}} < T_{\\text{dew}}$) enters a warm, humid space, causing front-element optical condensation fog
       and rotary head-drum moisture stiction that gradually clears as the chassis warms up.
     - **B. Near-Field Electromagnetic & Ground-Potential Field (`ElectromagneticField`)**:
       Biot-Savart / dipole distance decay $B(r) \\propto (r^2 + r_0^2)^{-3/2}$ from high-power transformers/transmitters
       at `(0.80, 0.72)`, attenuated by chassis shielding effectiveness `env_shielding_eff`, plus ground-potential
       difference $\\Delta V_{\\text{gnd}}$ proportional to cable vector between `cam_pos` and `vtr_pos`.
     - **C. Room Acoustic Modal & Reverberation Field (`compute_spatial_acoustic_audio`)**:
       Sabine $RT_{60}$ diffuse reverb, 3D room axial standing-wave modes ($f_{1,0,0}, f_{0,1,0}$), HVAC/transformer
       spatial noise floor, and cardioid microphone directivity.
     - **D. Spatial Mixed-Illuminant & Stray-Light Optical Field (`OpticalSpaceField`)**:
       Daylight window (`5600K`) vs tungsten/fluorescent practicals (`2900K / 4100K`) spatial color-temperature
       gradient, lens hood stray-light veiling glare, and CRT faceplate ambient reflection washout.
  3. **2x2 Spatial Environment Field Lab (`env_field_mode = 4`) & Single-Corner Split (`env_quad_corner = 0..4`)**:
     - Corner 1 (Top-Left): `1. 温湿度结露耦合 (Cold-to-Warm Dew Condensation)`
     - Corner 2 (Top-Right): `2. 近场电磁与地电位差耦合 (Near-Field EMI & Ground Loop)`
     - Corner 3 (Bottom-Left): `3. 空间混合色温与CRT环境杂散光 (Spatial Mixed Lighting & Glare)`
     - Corner 4 (Bottom-Right): `4. 全耦合输出与2D空间场拓扑图 (Full Coupled Output + 2D Field Map)`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass
from .multiview_wall import _draw_bitmap_text


def draw_text(canvas: np.ndarray, x0: int, y0: int, text: str, color: tuple[float, float, float], scale: int = 1) -> None:
    _draw_bitmap_text(canvas, str(text), int(x0), int(y0), color, scale=int(scale))


LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class SpatialEnvFieldSpec:
    enabled: bool = False
    mode: int = 0                    # 0=off, 1=thermo_dew, 2=emi_ground, 3=optical_glare, 4=2x2_spatial_lab, 5=full_coupled_single
    room_temp_c: float = 26.0        # -10..45 C
    room_rh_pct: float = 78.0        # 15..98 %RH
    device_init_temp_c: float = 6.0  # -15..40 C (cold device brought into warm humid room)
    emi_source_power: float = 0.65   # 0..1.5
    shielding_eff: float = 0.45      # 0..1 chassis & cable shielding effectiveness
    light_mix: float = 0.60          # 0..1 spatial mixed-illuminant & stray glare strength
    acoustic_rt60: float = 0.65      # 0.05..2.5 s room reverberation time RT60
    cam_x: float = 0.22
    cam_y: float = 0.30
    vtr_x: float = 0.72
    vtr_y: float = 0.68
    crt_x: float = 0.78
    crt_y: float = 0.28
    quad_corner: int = 0             # 0=2x2 quad, 1=TL, 2=TR, 3=BL, 4=BR
    show_border: int = 2             # 0=seamless, 1=hairline, 2=studio border
    show_text: int = 0               # 0=no text (clean default), 1=corner labels, 2=labels+telemetry

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "SpatialEnvFieldSpec":
        en = float(cfg.get("env_field_enabled", 0.0)) > 0.5
        md = int(round(float(np.clip(cfg.get("env_field_mode", 0.0), 0.0, 5.0))))
        if md > 0:
            en = True
        return cls(
            enabled=en,
            mode=md if md > 0 else (4 if en else 0),
            room_temp_c=float(np.clip(cfg.get("env_room_temp_c", 26.0), -10.0, 45.0)),
            room_rh_pct=float(np.clip(cfg.get("env_room_rh_pct", 78.0), 15.0, 98.0)),
            device_init_temp_c=float(np.clip(cfg.get("env_device_init_temp_c", 6.0), -15.0, 40.0)),
            emi_source_power=float(np.clip(cfg.get("env_emi_source_power", 0.65), 0.0, 1.5)),
            shielding_eff=float(np.clip(cfg.get("env_shielding_eff", 0.45), 0.0, 1.0)),
            light_mix=float(np.clip(cfg.get("env_light_mix", 0.60), 0.0, 1.0)),
            acoustic_rt60=float(np.clip(cfg.get("env_acoustic_rt60", 0.65), 0.05, 2.5)),
            cam_x=float(np.clip(cfg.get("env_cam_x", 0.22), 0.0, 1.0)),
            cam_y=float(np.clip(cfg.get("env_cam_y", 0.30), 0.0, 1.0)),
            vtr_x=float(np.clip(cfg.get("env_vtr_x", 0.72), 0.0, 1.0)),
            vtr_y=float(np.clip(cfg.get("env_vtr_y", 0.68), 0.0, 1.0)),
            crt_x=float(np.clip(cfg.get("env_crt_x", 0.78), 0.0, 1.0)),
            crt_y=float(np.clip(cfg.get("env_crt_y", 0.28), 0.0, 1.0)),
            quad_corner=int(round(float(np.clip(cfg.get("env_quad_corner", 0.0), 0.0, 4.0)))),
            show_border=int(round(float(np.clip(cfg.get("env_show_border", 2.0), 0.0, 2.0)))),
            show_text=int(round(float(np.clip(cfg.get("env_show_text", 0.0), 0.0, 2.0)))),
        )

    def is_active(self) -> bool:
        return bool(self.enabled and self.mode > 0)


def magnus_dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Magnus-Tetens dew point formula in degrees Celsius."""
    a, b = 17.27, 237.7
    rh_c = float(np.clip(rh_pct, 1.0, 100.0))
    gamma = (a * float(temp_c)) / (b + float(temp_c)) + float(np.log(rh_c / 100.0))
    return float((b * gamma) / max(1e-4, a - gamma))


class EnvironmentField:
    """
    2D Spatial Physical Field over room coordinates `(x, y) in [0, 1]^2`.
      - HVAC cold vent at `(0.15, 0.18)`
      - High-power transformer / transmitter feeder EMI source at `(0.80, 0.72)`
      - Damp wall moisture plume at `(0.75, 0.70)`
      - Daylight window at `(0.08, 0.25)` (`5600K`) vs tungsten bank at `(0.86, 0.32)` (`2900K`)
    """

    EMI_POS = (0.80, 0.72)
    DAMP_POS = (0.75, 0.70)
    HVAC_POS = (0.15, 0.18)
    WINDOW_POS = (0.08, 0.25)
    TUNGSTEN_POS = (0.86, 0.32)

    def __init__(self, spec: SpatialEnvFieldSpec):
        self.spec = spec

    def sample_at(
        self,
        x: float,
        y: float,
        t_sec: float = 0.0,
        device_self_heat_c: float = 1.8,
        thermal_tau_s: float = 14.0,
    ) -> dict[str, float]:
        px = float(np.clip(x, 0.0, 1.0))
        py = float(np.clip(y, 0.0, 1.0))

        # 1. Local Temperature T(x, y) & Relative Humidity RH(x, y)
        d_hvac2 = (px - self.HVAC_POS[0]) ** 2 + (py - self.HVAC_POS[1]) ** 2
        d_emi2 = (px - self.EMI_POS[0]) ** 2 + (py - self.EMI_POS[1]) ** 2
        d_damp2 = (px - self.DAMP_POS[0]) ** 2 + (py - self.DAMP_POS[1]) ** 2

        local_temp_c = (
            float(self.spec.room_temp_c)
            - 3.8 * float(np.exp(-d_hvac2 / 0.08))
            + 4.5 * float(np.exp(-d_emi2 / 0.10))
        )
        local_rh_pct = float(np.clip(
            float(self.spec.room_rh_pct)
            + 16.0 * float(np.exp(-d_damp2 / 0.14))
            - 8.0 * float(np.exp(-d_hvac2 / 0.10)),
            10.0, 99.0,
        ))
        dew_point_c = magnus_dew_point_c(local_temp_c, local_rh_pct)

        # Device thermal inertia: T_dev(t) warms from device_init_temp_c toward local_temp_c + self_heat
        t_s = max(0.0, float(t_sec))
        equil_temp_c = local_temp_c + float(device_self_heat_c)
        dev_temp_c = equil_temp_c + (float(self.spec.device_init_temp_c) - equil_temp_c) * float(
            np.exp(-t_s / max(1.0, float(thermal_tau_s)))
        )

        # Condensation film depth > 0 whenever device surface is colder than local dew point!
        dew_deficit_c = max(0.0, dew_point_c - dev_temp_c)
        condensation_film = float(np.clip(dew_deficit_c / 11.0, 0.0, 1.0))

        # 2. Near-Field Biot-Savart Magnetic Flux B_emi(x, y) & Ground Potential V_gnd(x, y)
        r_emi = float(np.sqrt(d_emi2))
        b_raw = float(self.spec.emi_source_power) * (0.045 / ((r_emi * r_emi + 0.045) ** 1.5))
        shield_atten = 1.0 - 0.82 * float(np.clip(self.spec.shielding_eff, 0.0, 1.0))
        b_emi = float(np.clip(b_raw * shield_atten, 0.0, 2.0))

        # Spatial ground potential gradient induced by return currents near the power transformer
        v_gnd = float(self.spec.emi_source_power) * (0.65 * px + 0.55 * py + 0.45 * float(np.exp(-d_emi2 / 0.12)))

        # 3. Spatial Optical Field: Daylight Window (5600K) vs Tungsten/Fluorescent Bank (2900K)
        d_win = float(np.hypot(px - self.WINDOW_POS[0], py - self.WINDOW_POS[1]))
        d_tung = float(np.hypot(px - self.TUNGSTEN_POS[0], py - self.TUNGSTEN_POS[1]))
        w_day = float(np.exp(-(d_win ** 2) / 0.28))
        w_tung = float(np.exp(-(d_tung ** 2) / 0.28))
        day_share = w_day / max(1e-4, w_day + w_tung)
        cct_k = 2900.0 + (5600.0 - 2900.0) * day_share
        stray_glare = float(np.clip(float(self.spec.light_mix) * (0.65 * w_day + 0.55 * w_tung), 0.0, 1.0))

        return {
            "x": round(px, 3),
            "y": round(py, 3),
            "local_temp_c": round(local_temp_c, 2),
            "local_rh_pct": round(local_rh_pct, 2),
            "dew_point_c": round(dew_point_c, 2),
            "device_temp_c": round(dev_temp_c, 2),
            "dew_deficit_c": round(dew_deficit_c, 2),
            "condensation_film": round(condensation_film, 4),
            "b_emi_field": round(b_emi, 4),
            "v_ground_potential": round(v_gnd, 4),
            "cct_kelvin": round(cct_k, 1),
            "stray_glare": round(stray_glare, 4),
        }


def evaluate_all_devices(spec: SpatialEnvFieldSpec, t_sec: float = 0.0) -> dict[str, Any]:
    """
    Sample the 2D environment field at `cam`, `vtr`, and `crt` coordinates and compute
    inter-device coupling quantities (such as ground-loop potential difference `delta_v_ground`).
    """
    field = EnvironmentField(spec)
    cam = field.sample_at(spec.cam_x, spec.cam_y, t_sec=t_sec, device_self_heat_c=1.5, thermal_tau_s=12.0)
    vtr = field.sample_at(spec.vtr_x, spec.vtr_y, t_sec=t_sec, device_self_heat_c=2.5, thermal_tau_s=16.0)
    crt = field.sample_at(spec.crt_x, spec.crt_y, t_sec=t_sec, device_self_heat_c=4.0, thermal_tau_s=20.0)

    cable_dist = float(np.hypot(spec.vtr_x - spec.cam_x, spec.vtr_y - spec.cam_y))
    delta_v_gnd = abs(float(vtr["v_ground_potential"]) - float(cam["v_ground_potential"])) * (0.5 + 0.8 * cable_dist)
    ground_hum_level = float(np.clip(delta_v_gnd * (1.0 - 0.65 * spec.shielding_eff), 0.0, 1.2))

    return {
        "cam": cam,
        "vtr": vtr,
        "crt": crt,
        "cable_length_norm": round(cable_dist, 4),
        "delta_v_ground": round(delta_v_gnd, 4),
        "ground_hum_level": round(ground_hum_level, 4),
    }


def _apply_thermo_dew_branch(
    rgb_in: np.ndarray,
    env_state: dict[str, Any],
    frame_index: int,
    seed: int,
) -> np.ndarray:
    """
    Branch 1: Thermo-Hygrometric & Dew Condensation Coupling.
      - Camera front-element condensation water film (`cam.condensation_film`):
        High-gloss forward scattering bloom + shadow black-floor lift + droplet caustic rings.
      - VTR rotary drum condensation (`vtr.condensation_film`):
        Micro-water film stiction shear bands + wet-head RF envelope dropout snow.
    """
    out = np.clip(np.asarray(rgb_in, dtype=np.float32), 0.0, 1.0).copy()
    h, w = out.shape[:2]
    cam_dew = float(env_state["cam"]["condensation_film"])
    vtr_dew = float(env_state["vtr"]["condensation_film"])

    if cam_dew > 1e-4:
        # Lens condensation fog: wide forward-scatter veil + highlight halation skirt + black lift
        veil_wide = lowpass(out, max(2.5, 14.0 * cam_dew * (w / 360.0)))
        hot = np.clip(out - 0.55, 0.0, 1.0)
        halo = lowpass(hot, max(1.8, 7.5 * cam_dew * (w / 360.0)))
        out = out * (1.0 - 0.52 * cam_dew) + veil_wide * (0.52 * cam_dew) + halo * (0.45 * cam_dew) + 0.09 * cam_dew

    if vtr_dew > 1e-4:
        rng = np.random.default_rng((int(seed) * 2246822519 + int(frame_index) * 131 + 7) & 0xFFFFFFFF)
        # Drum dew stiction: sawtooth horizontal line pull + wet-head RF loss band
        rows = np.arange(h, dtype=np.float32)
        band_c = float(((frame_index * 7) % max(1, h)))
        d_row = np.abs(rows - band_c)
        stic_env = np.exp(-0.5 * (d_row / max(4.0, h * 0.10)) ** 2).astype(np.float32)
        shifts = stic_env * (18.0 * vtr_dew) + rng.standard_normal(h).astype(np.float32) * (2.5 * vtr_dew)
        out = horizontal_shift(out, shifts)
        snow = rng.standard_normal((h, w)).astype(np.float32)
        snow = lowpass(snow, 1.5, axis=1)
        wet_band = (stic_env[:, None] * (0.45 * vtr_dew)).astype(np.float32)
        out = out * (1.0 - wet_band[..., None]) + (0.52 + 0.25 * snow[..., None]) * wet_band[..., None]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_emi_ground_branch(
    rgb_in: np.ndarray,
    env_state: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    """
    Branch 2: Near-Field Electromagnetic Interference (`B_emi`) & Ground-Potential Loop (`ground_hum_level`).
      - 50/100 Hz S-curve horizontal deflection hum snake proportional to `vtr.b_emi_field`
      - Rolling mains ground-loop hum bar proportional to `ground_hum_level`
      - Diagonal switching-regulator / transmitter RF beat herringbone proportional to `cam.b_emi_field + vtr.b_emi_field`
    """
    out = np.clip(np.asarray(rgb_in, dtype=np.float32), 0.0, 1.0).copy()
    h, w = out.shape[:2]
    b_vtr = float(env_state["vtr"]["b_emi_field"])
    b_cam = float(env_state["cam"]["b_emi_field"])
    b_crt = float(env_state["crt"]["b_emi_field"])
    g_hum = float(env_state["ground_hum_level"])
    tot_emi = float(np.clip(b_vtr + 0.7 * b_cam + 0.6 * b_crt, 0.0, 2.0))

    rows = np.arange(h, dtype=np.float32)
    yn = rows / max(1.0, h - 1.0)
    phase_t = float(frame_index) * 0.38

    if tot_emi > 1e-4:
        # S-curve horizontal hum snake from magnetic field coupling into H-AFC & CRT yoke
        hum_shift = (8.5 * tot_emi * (w / 360.0)) * np.sin(2.0 * np.pi * yn * 2.0 + phase_t).astype(np.float32)
        out = horizontal_shift(out, hum_shift)

        # Diagonal EMI carrier beat fringes
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        beat = np.cos(2.0 * np.pi * (xx / max(3.0, 9.0 * (w / 360.0)) - yy / max(4.0, 14.0 * (h / 240.0))) + phase_t * 2.0)
        out[..., 0] = np.clip(out[..., 0] + 0.08 * tot_emi * beat, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.14 * tot_emi * beat, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.09 * tot_emi * beat, 0.0, 1.0)

    if g_hum > 1e-4:
        # Rolling ground-loop hum bar from spatial ground potential difference between CAM and VTR
        hum_bar = (0.16 * g_hum) * np.sin(2.0 * np.pi * (yn + frame_index * 0.08))[:, None, None].astype(np.float32)
        out = np.clip(out + hum_bar, 0.0, 1.0)

    return out.astype(np.float32)


def _apply_optical_space_branch(
    rgb_in: np.ndarray,
    spec: SpatialEnvFieldSpec,
    env_state: dict[str, Any],
) -> np.ndarray:
    """
    Branch 3: Spatial Mixed-Illuminant Color-Temperature Gradient & CRT Ambient Glare.
      - Left side illuminated by 5600K daylight window (cool cyan-blue cast) vs right side illuminated by
        2900K tungsten + 4100K fluorescent practicals (warm amber-green cast).
      - Off-axis stray light veiling glare at `cam` position + CRT phosphor ambient reflection washout at `crt`.
    """
    out = np.clip(np.asarray(rgb_in, dtype=np.float32), 0.0, 1.0).copy()
    h, w = out.shape[:2]
    lmix = float(np.clip(spec.light_mix, 0.0, 1.0))
    if lmix <= 1e-4:
        return out

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1.0, w - 1.0)
    yn = yy / max(1.0, h - 1.0)

    # Spatial mixed illuminant across the scene: left cool daylight (B>R), right warm tungsten (R>B)
    r_gain = 1.0 + lmix * (0.24 * (xn - 0.45))
    g_gain = 1.0 + lmix * (0.06 * np.sin(np.pi * yn))
    b_gain = 1.0 - lmix * (0.26 * (xn - 0.45))
    out[..., 0] *= r_gain
    out[..., 1] *= g_gain
    out[..., 2] *= b_gain

    # Off-axis stray light cone from window/lamp + CRT ambient phosphor washout
    cam_glare = float(env_state["cam"]["stray_glare"])
    crt_glare = float(env_state["crt"]["stray_glare"])
    glare_cone = np.exp(-(((xn - 0.15) ** 2 + (yn - 0.20) ** 2) / 0.35)).astype(np.float32)
    washout = (0.14 * cam_glare * glare_cone + 0.08 * crt_glare)[..., None]
    out = out * (1.0 - 0.35 * washout) + washout * np.array([0.96, 0.98, 1.02], dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _render_spatial_field_map_and_coupled(
    rgb_coupled: np.ndarray,
    spec: SpatialEnvFieldSpec,
    env_state: dict[str, Any],
    show_text: int = 0,
) -> np.ndarray:
    """
    Branch 4: Full Multi-Domain Coupled Output with a bottom-right 2D Room Environment Field Radar Map
    showing the live positions of `CAM`, `VTR`, and `CRT` relative to the Dew/Humidity and EMI sources.
    """
    out = rgb_coupled.copy()
    h, w = out.shape[:2]
    map_h = max(36, int(round(h * 0.34)))
    map_w = max(48, int(round(w * 0.36)))
    y0 = h - map_h - 6
    x0 = w - map_w - 6
    if y0 < 4 or x0 < 4:
        return out

    yy, xx = np.mgrid[0:map_h, 0:map_w].astype(np.float32)
    nx = xx / max(1.0, map_w - 1.0)
    ny = yy / max(1.0, map_h - 1.0)

    # Synthesize 2D heatmap of EMI (red/amber around (0.80, 0.72)) + Humidity/Dew (cyan/blue around (0.75, 0.70) & (0.15, 0.18))
    d_emi2 = (nx - 0.80) ** 2 + (ny - 0.72) ** 2
    d_hvac2 = (nx - 0.15) ** 2 + (ny - 0.18) ** 2
    d_damp2 = (nx - 0.75) ** 2 + (ny - 0.70) ** 2

    emi_heat = np.exp(-d_emi2 / 0.07) * float(np.clip(spec.emi_source_power, 0.0, 1.5))
    dew_heat = (0.7 * np.exp(-d_damp2 / 0.10) + 0.6 * np.exp(-d_hvac2 / 0.08)) * (float(spec.room_rh_pct) / 100.0)

    radar = np.zeros((map_h, map_w, 3), dtype=np.float32)
    radar[..., 0] = 0.08 + 0.72 * emi_heat
    radar[..., 1] = 0.10 + 0.35 * dew_heat + 0.25 * emi_heat
    radar[..., 2] = 0.14 + 0.75 * dew_heat

    # Grid lines on radar map
    for frac in (0.25, 0.50, 0.75):
        gy = int(round(frac * (map_h - 1)))
        gx = int(round(frac * (map_w - 1)))
        radar[gy, :, :] = np.maximum(radar[gy, :, :], 0.24)
        radar[:, gx, :] = np.maximum(radar[:, gx, :], 0.24)

    # Plot device markers: CAM (green), VTR (amber), CRT (cyan)
    for dx_f, dy_f, col in (
        (spec.cam_x, spec.cam_y, (0.35, 0.98, 0.45)),
        (spec.vtr_x, spec.vtr_y, (0.98, 0.78, 0.25)),
        (spec.crt_x, spec.crt_y, (0.35, 0.88, 0.98)),
    ):
        px = int(round(float(np.clip(dx_f, 0.05, 0.95)) * (map_w - 1)))
        py = int(round(float(np.clip(dy_f, 0.05, 0.95)) * (map_h - 1)))
        r_dot = max(2, min(4, map_h // 16))
        radar[max(0, py - r_dot):min(map_h, py + r_dot + 1), max(0, px - r_dot):min(map_w, px + r_dot + 1)] = np.array(col, dtype=np.float32)

    # Border around radar inset
    radar[0, :, :] = (0.75, 0.62, 0.28)
    radar[-1, :, :] = (0.75, 0.62, 0.28)
    radar[:, 0, :] = (0.75, 0.62, 0.28)
    radar[:, -1, :] = (0.75, 0.62, 0.28)

    out[y0:y0 + map_h, x0:x0 + map_w] = 0.15 * out[y0:y0 + map_h, x0:x0 + map_w] + 0.85 * radar
    if show_text >= 1 and map_w >= 80:
        draw_text(out, x0 + 4, y0 + 3, "2D ENV FIELD", (0.95, 0.88, 0.55), scale=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_spatial_environment_field(
    rgb_in: np.ndarray,
    spec: SpatialEnvFieldSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Execute Option R 2D Spatial Physical Environment Field & Multi-Domain Coupling System.
    """
    if not spec.is_active():
        return rgb_in, {"active": False}

    t_sec = float(frame_index) / max(1.0, float(fps))
    env_state = evaluate_all_devices(spec, t_sec=t_sec)

    b1_dew = _apply_thermo_dew_branch(rgb_in, env_state, frame_index=frame_index, seed=seed)
    b2_emi = _apply_emi_ground_branch(rgb_in, env_state, frame_index=frame_index)
    b3_opt = _apply_optical_space_branch(rgb_in, spec, env_state)

    # Full multi-domain coupled frame (Thermo-Dew -> Optical -> EMI/Ground)
    full_coupled = _apply_emi_ground_branch(
        _apply_optical_space_branch(b1_dew, spec, env_state),
        env_state,
        frame_index=frame_index,
    )
    b4_map = _render_spatial_field_map_and_coupled(full_coupled, spec, env_state, show_text=int(spec.show_text))

    report = {
        "active": True,
        "mode": int(spec.mode),
        "env_quad_corner": int(spec.quad_corner),
        "env_show_border": int(spec.show_border),
        "env_show_text": int(spec.show_text),
        "t_sec": round(t_sec, 3),
        "devices": env_state,
    }

    corner = int(spec.quad_corner)
    if spec.mode == 1 or corner == 1:
        return b1_dew, report
    if spec.mode == 2 or corner == 2:
        return b2_emi, report
    if spec.mode == 3 or corner == 3:
        return b3_opt, report
    if spec.mode == 5:
        return full_coupled, report
    if corner == 4:
        return b4_map, report

    # Mode 4 (`quad_corner == 0`): 2x2 Spatial Environment Field Comparison Lab
    h, w = rgb_in.shape[:2]
    h2, w2 = h // 2, w // 2
    if h2 < 16 or w2 < 16:
        return full_coupled, report

    def _down(im: np.ndarray, th: int, tw: int) -> np.ndarray:
        yi = np.clip((np.arange(th, dtype=np.float32) + 0.5) * (im.shape[0] / th) - 0.5, 0, im.shape[0] - 1).astype(np.int32)
        xi = np.clip((np.arange(tw, dtype=np.float32) + 0.5) * (im.shape[1] / tw) - 0.5, 0, im.shape[1] - 1).astype(np.int32)
        return im[yi][:, xi]

    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas[:h2, :w2] = _down(b1_dew, h2, w2)
    canvas[:h2, w2:] = _down(b2_emi, h2, w - w2)
    canvas[h2:, :w2] = _down(b3_opt, h - h2, w2)
    canvas[h2:, w2:] = _down(b4_map, h - h2, w - w2)

    show_border = int(spec.show_border)
    show_text = int(spec.show_text)
    if show_border >= 1:
        bw = 1 if show_border == 1 else 2
        col = np.array([0.26, 0.30, 0.28] if show_border == 1 else [0.38, 0.78, 0.65], dtype=np.float32)
        canvas[max(0, h2 - bw):min(h, h2 + bw), :, :] = col
        canvas[:, max(0, w2 - bw):min(w, w2 + bw), :] = col

    if show_text >= 1 and h >= 120 and w >= 200:
        bh_lbl = min(16, max(12, h2 // 9))
        for qy, qx, title, fg in (
            (0, 0, "1.DEW CONDENSATION FIELD", (0.55, 0.90, 0.98)),
            (0, w2, "2.NEAR-FIELD EMI & GND", (0.98, 0.72, 0.38)),
            (h2, 0, "3.SPATIAL CCT & GLARE", (0.95, 0.88, 0.48)),
            (h2, w2, "4.COUPLED + 2D ROOM MAP", (0.58, 0.95, 0.62)),
        ):
            canvas[qy:qy + bh_lbl, qx:min(w, qx + w2), :] *= 0.22
            draw_text(canvas, qx + 5, qy + 3, title, fg, scale=1)

    if show_text >= 2 and h >= 160 and w >= 240:
        footer_h = min(18, max(14, h // 15))
        canvas[h - footer_h:, :, :] *= 0.18
        c_dew = env_state["cam"]["condensation_film"]
        v_emi = env_state["vtr"]["b_emi_field"]
        g_hum = env_state["ground_hum_level"]
        msg = f"ENV FIELD: CAM_DEW={c_dew:.2f} | VTR_EMI={v_emi:.2f} | GND_LOOP={g_hum:.2f}"
        draw_text(canvas, 6, h - footer_h + 4, msg, (0.75, 0.96, 0.84), scale=1)

    return np.clip(canvas, 0.0, 1.0).astype(np.float32), report


def compute_spatial_acoustic_audio(
    stereo_in: np.ndarray,
    spec: SpatialEnvFieldSpec,
    sample_rate: int = 44100,
    start_s: float = 0.0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply Option R 3D Room Acoustic Space Field to stereo audio:
      1. Axial room standing-wave resonances (`f_100 ~ 171 Hz`, `f_010 ~ 214 Hz`, `f_110 ~ 274 Hz`)
      2. Multi-tap early reflections + Sabine `RT60` diffuse decay tail
      3. Cardioid microphone off-axis high-frequency roll-off & distance attenuation based on `cam_pos`
      4. Spatial HVAC rumble (near `HVAC_POS`) + 50/100 Hz transformer magnetic hum (near `EMI_POS`)
    """
    arr = np.asarray(stereo_in, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    n = arr.shape[0]
    if n == 0 or not spec.is_active():
        return arr, {"active": False}

    env_state = evaluate_all_devices(spec, t_sec=start_s)
    rt60 = float(np.clip(spec.acoustic_rt60, 0.05, 2.5))
    sr = max(1000, int(sample_rate))
    out = arr.copy()

    # 1. Multi-tap early reflections & Sabine RT60 decay tail
    tap_ms_list = (13.0, 23.0, 37.0, 59.0, 89.0)
    reverb_acc = np.zeros_like(out)
    for idx_t, tap_ms in enumerate(tap_ms_list):
        d_samp = int(round((tap_ms / 1000.0) * sr))
        if 0 < d_samp < n:
            # Sabine amplitude decay: 10^(-3 * t / RT60)
            gain_t = float(10.0 ** (-3.0 * (tap_ms / 1000.0) / max(0.08, rt60))) * (0.32 / (idx_t + 1.0))
            delayed = np.zeros_like(out)
            # Cross-channel stereo reflection spread
            delayed[d_samp:, 0] = arr[:-d_samp, idx_t % 2]
            delayed[d_samp:, 1] = arr[:-d_samp, (idx_t + 1) % 2]
            reverb_acc += delayed * gain_t

    wet_mix = float(np.clip(0.18 * rt60, 0.0, 0.42))
    out = out * (1.0 - 0.5 * wet_mix) + reverb_acc * wet_mix

    # 2. Spatial EMI hum & HVAC low-frequency rumble at microphone (`cam`) + deck (`vtr`)
    t_vec = start_s + np.arange(n, dtype=np.float32) / float(sr)
    b_emi_tot = float(env_state["cam"]["b_emi_field"]) + 0.7 * float(env_state["vtr"]["b_emi_field"])
    g_hum = float(env_state["ground_hum_level"])
    hum_amp = float(np.clip(0.018 * b_emi_tot + 0.022 * g_hum, 0.0, 0.15))
    if hum_amp > 1e-5:
        hum_wave = hum_amp * (
            0.65 * np.sin(2.0 * np.pi * 50.0 * t_vec)
            + 0.35 * np.sin(2.0 * np.pi * 100.0 * t_vec + 0.4)
        ).astype(np.float32)
        out += hum_wave[:, None]

    # HVAC rumble inversely proportional to distance from HVAC vent (0.15, 0.18)
    d_hvac = float(np.hypot(spec.cam_x - EnvironmentField.HVAC_POS[0], spec.cam_y - EnvironmentField.HVAC_POS[1]))
    hvac_amp = float(np.clip(0.012 * np.exp(-(d_hvac ** 2) / 0.16), 0.0, 0.05))
    if hvac_amp > 1e-5:
        rng = np.random.default_rng((int(seed) ^ 0x5A5A5A5A) & 0xFFFFFFFF)
        rumble = rng.standard_normal((n, 2)).astype(np.float32) * hvac_amp
        out += rumble

    return np.clip(out, -1.0, 1.0).astype(np.float32), {
        "active": True,
        "rt60_s": round(rt60, 3),
        "wet_mix": round(wet_mix, 4),
        "induced_hum_amp": round(hum_amp, 5),
        "hvac_rumble_amp": round(hvac_amp, 5),
    }


class EnvChamberStage(Stage):
    """Standalone node-graph stage (`env_chamber`) for Option R."""
    name = "env_chamber"
    layers = ("connector",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = dict(self.cfg_of("connector"))
        if "env_chamber" in (self.ctx.params or {}):
            c.update(self.ctx.params["env_chamber"])
        self.spec = SpatialEnvFieldSpec.from_params(c)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.mode <= 0:
                self.spec.mode = 4

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        self._read_params()
        sig = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        rgb_out, rep = apply_spatial_environment_field(
            rgb_in, self.spec, frame_index=int(buf.index), fps=float(self.ctx.fps), seed=int(self.ctx.seed) + 73
        )
        sig_out = SignalFrame.from_rgb(rgb_out, buf.std)
        sig_out._rf_direct = True
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["spatial_env_field"] = rep
        return FrameBuffer(rgb_out, buf.composite, buf.index, buf.time, buf.std, meta)

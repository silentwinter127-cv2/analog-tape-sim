"""
Module 1 Extension (Candidate A) -- Virtual Camera Reshooting CRT Off-Screen System
(`atsim.codecs.crt_reshoot`).

Models the physical optical & temporal chain when a secondary camera (smartphone CMOS,
digital camera, or camcorder CCD) photographs a live CRT television or broadcast monitor
("把视频信号真正送进虚拟 CRT，再用虚拟相机重新拍摄，而不是直接对数字画面叠加 CRT 滤镜"):

1. **Shutter–Field Scanning Beat Roll Bar & CMOS Rolling Shutter
   (`reshoot_enabled`, `reshoot_shutter_hz`, `reshoot_fps_offset`, `reshoot_rolling_shutter`)**:
   - A CRT does not illuminate the whole screen at once; the electron beam sweeps vertically
     at field rate $f_{\text{CRT}}$ (59.94 Hz NTSC / 50.0 Hz PAL). Each scanline $v \in [0, 1)$
     flashes when the beam strikes at $t_{\text{hit}}(k, v) = (k + v) / f_{\text{CRT}}$ and decays
     exponentially with P22 phosphor time constants ($\tau_R > \tau_G > \tau_B$).
   - The reshooting camera integrates light over $[t_{\text{open}}(v), t_{\text{open}}(v) + \tau_{\text{shutter}}]$,
     where $\tau_{\text{shutter}} = 1 / f_{\text{shutter}}$ and $t_{\text{open}}(v)$ includes the
     CMOS rolling-shutter readout skew $\tau_{\text{readout}} \cdot v$.
   - Integrating the pulsed phosphor decay over the shutter window produces an authentic
     rolling bright/dark scan band (with a bright leading beam edge, warm-red trailing P22
     afterglow, or double-sweep bright band when $\tau_{\text{shutter}} > T_{\text{field}}$)
     that crawls vertically at the beat rate $\Delta f = f_{\text{cam}} - f_{\text{CRT}}$.

2. **2D Sensor Pixel Lattice vs. CRT Shadow-Mask & Scanline Spatial Aliasing Moiré
   (`reshoot_moire`, `reshoot_tilt_deg`)**:
   - Projecting the curved CRT RGB phosphor triad lattice (Aperture Grille / Slot Mask / Dot Mask)
     and horizontal scanlines onto the reshooting camera's discrete Bayer/CFA pixel grid under a
     slight roll tilt $\theta_{\text{tilt}}$ and radial lens distortion generates 2D curved
     chromatic Moiré interference fringes ($\Delta \mathbf{k} = \mathbf{k}_{\text{CRT}} - \mathbf{k}_{\text{cam}}$).

3. **Off-Axis Camera Perspective, Ambient Room Glare & 9300K Phosphor White-Balance Shift
   (`reshoot_perspective`, `reshoot_ambient_glare`, `reshoot_wb_shift`)**:
   - 3D off-axis projective foreshortening (tilt/yaw keystone + recessed bezel framing + shallow DOF),
   - Diffuse ambient room light reflecting off the unlit grey-white phosphor powder (lifting the
     black floor) plus curved leaded-glass specular window/lamp reflection,
   - D93 (9300K) consumer CRT phosphor cool cyan-blue cast & auto-exposure highlight bloom/clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.signal import lowpass
from ..core.standards import VideoStandard

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class CRTReshootSpec:
    """Parameters of the Virtual Camera Reshooting CRT Off-Screen System."""
    enabled: bool = False
    shutter_hz: float = 60.0            # 1 / T_shutter in Hz (e.g. 50, 60, 100, 125, 250 Hz)
    fps_offset: float = 0.60            # f_cam - f_CRT beat frequency in Hz
    rolling_shutter: float = 0.35       # CMOS rolling-shutter line-readout fraction (0=CCD global, >0=CMOS rolling)
    moire: float = 0.0                  # 2D sensor-vs-shadow-mask spatial aliasing Moiré intensity (0..1.5)
    tilt_deg: float = 0.0               # Camera roll tilt angle relative to CRT raster (deg, -15..15)
    perspective: float = 0.0            # Off-axis camera perspective & shallow DOF (-1..1)
    ambient_glare: float = 0.0          # Room ambient light phosphor reflection & glass glare (0..1)
    wb_shift: float = 0.0               # 9300K CRT phosphor cool cast & AE highlight overexposure (-1..1)

    @classmethod
    def from_params(cls, params: dict | None, force_enable: bool = False) -> "CRTReshootSpec":
        p = params or {}
        d = p.get("display") if isinstance(p.get("display"), dict) else p
        en = force_enable or (float(d.get("reshoot_enabled", 0.0)) > 0.5)
        return cls(
            enabled=bool(en),
            shutter_hz=float(np.clip(d.get("reshoot_shutter_hz", 60.0), 15.0, 600.0)),
            fps_offset=float(np.clip(d.get("reshoot_fps_offset", 0.60), -25.0, 25.0)),
            rolling_shutter=float(np.clip(d.get("reshoot_rolling_shutter", 0.35), 0.0, 1.0)),
            moire=float(np.clip(d.get("reshoot_moire", 0.0), 0.0, 1.5)),
            tilt_deg=float(np.clip(d.get("reshoot_tilt_deg", 0.0), -20.0, 20.0)),
            perspective=float(np.clip(d.get("reshoot_perspective", 0.0), -1.0, 1.0)),
            ambient_glare=float(np.clip(d.get("reshoot_ambient_glare", 0.0), 0.0, 1.0)),
            wb_shift=float(np.clip(d.get("reshoot_wb_shift", 0.0), -1.0, 1.0)),
        )

    def is_active(self) -> bool:
        return (
            self.enabled
            or self.moire > 1e-4
            or abs(self.perspective) > 1e-4
            or self.ambient_glare > 1e-4
            or abs(self.wb_shift) > 1e-4
        )


def compute_shutter_scan_envelope(
    height: int,
    spec: CRTReshootSpec,
    std: VideoStandard,
    frame_index: int,
    fps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute the exact per-row 3-channel `(H, 3)` exposure envelope from integrating
    the CRT vertical electron-beam sweep + P22 phosphor decay over the camera's shutter window.

    Let $T_{\text{field}} = 1 / f_{\text{CRT}}$ be the CRT vertical field period (`1/59.94s` NTSC
    or `1/50s` PAL). Normalized to field period units:
      * Duty cycle $D = \tau_{\text{shutter}} / T_{\text{field}} = f_{\text{CRT}} / f_{\text{shutter}}$.
      * At row $v = y / H \in [0, 1)$, the camera shutter opens at normalized phase:
            $\phi_{\text{open}}(v) = \phi_0(n) + \alpha_{\text{RS}} \cdot v$
        and closes at $\phi_{\text{close}}(v) = \phi_{\text{open}}(v) + D$.
      * The CRT electron beam excites row $v$ at phases $v + k$ ($k \in \mathbb{Z}$).
        Following each strike at $s = v + k$, the P22 phosphor radiance is a combination of
        an intense prompt excitation pulse ($\sigma_p \approx 0.018$ field) plus an exponential
        afterglow tail $\exp(-(\phi - s) / \tau_c) \cdot u(\phi - s)$ with $\tau_R > \tau_G > \tau_B$.
    """
    h = max(2, int(height))
    f_crt = float(getattr(std, "field_hz", 59.94 if std.color_system == "NTSC" else 50.0))
    f_shutter = max(15.0, float(spec.shutter_hz))
    duty = float(f_crt / f_shutter)  # fraction of a CRT field integrated during one shutter opening

    # Phase crawl per frame from beat frequency delta_f = spec.fps_offset
    # Plus natural mismatch if f_shutter != f_crt
    beat_hz = float(spec.fps_offset)
    if abs(beat_hz) < 1e-3 and abs(f_shutter - f_crt) > 0.5:
        beat_hz = (f_shutter - f_crt) * 0.12
    phase_step = beat_hz / max(1.0, float(fps))
    phi0 = float((frame_index * phase_step + 0.18) % 1.0)

    v = np.arange(h, dtype=np.float32) / float(h)
    rs_skew = float(np.clip(spec.rolling_shutter, 0.0, 1.0)) * 0.65

    # Relative phase of the electron beam strike relative to the shutter opening at row v:
    # Beam hits row v at phase `v (mod 1)`. Shutter opens at `phi0 + rs_skew * v`.
    # Because rolling shutter advances in the SAME downward direction as the CRT beam,
    # the effective vertical phase is `u(v) = (v * (1.0 - rs_skew) - phi0) % 1.0`!
    eff_slope = max(0.22, 1.0 - rs_skew)
    u = np.mod(v * eff_slope - phi0, 1.0).astype(np.float32)

    # Number of full fields integrated + fractional remainder `rem_duty`
    n_full = int(np.floor(duty))
    rem = float(duty - n_full)

    # Smooth analytical integration of prompt beam flash + P22 phosphor decay over [0, rem]
    # For a single field pass (u in [0, 1)):
    #   - rows with `0 <= u <= rem` are directly struck by the electron beam DURING the open shutter!
    #   - rows just after `rem` (`u > rem`) were struck BEFORE the shutter opened and only contribute
    #     their decaying P22 phosphor tail `exp(-(1 - u) / tau_c)`!
    sigma_edge = 0.028  # soft transition from finite beam spot + vertical blanking retrace
    if rem < 1e-3:
        in_window = np.zeros_like(u)
        leading_hot = np.zeros_like(u)
        decay_dist = np.zeros_like(u)
    elif rem > 0.999:
        in_window = np.ones_like(u)
        leading_hot = np.zeros_like(u)
        decay_dist = np.zeros_like(u)
    else:
        # Smooth boxcar over u in [0, rem] on the unit circle
        rise = 0.5 * (1.0 + np.tanh(u / sigma_edge))
        fall = 0.5 * (1.0 + np.tanh((rem - u) / sigma_edge))
        wrap_rise = 0.5 * (1.0 + np.tanh((u - 1.0 + rem) / sigma_edge)) * float(rem > 0.85)
        in_window = np.clip(rise * fall + wrap_rise, 0.0, 1.0)

        # Bright phosphor strike crest right near the active beam front (u ~ rem)
        d_front = np.minimum(np.abs(u - rem), np.abs(u - rem + 1.0))
        leading_hot = np.exp(-0.5 * (d_front / 0.038) ** 2) * float(abs(rem - 0.5) < 0.46)

        # Distance behind the shutter window for phosphor persistence tail
        decay_dist = np.mod(u - rem, 1.0)

    # Channel-dependent P22 phosphor decay constants in fractions of a field:
    # Red (Y2O2S:Eu) decays slowest (~0.24 field), Green (ZnS:Cu,Al) medium (~0.16), Blue (ZnS:Ag) fastest (~0.12)
    taus = (0.25, 0.16, 0.12)
    env_channels = []
    for ch_idx, tau in enumerate(taus):
        afterglow = np.exp(-decay_dist / tau) * (1.0 - in_window)
        # Total integrated light for this row:
        # `n_full` complete sweeps + `in_window` active sweep + `0.32 * afterglow` + `0.24 * leading_hot`
        raw_e = float(n_full) + in_window + 0.30 * afterglow + (0.24 if ch_idx != 0 else 0.20) * leading_hot
        env_channels.append(raw_e.astype(np.float32))

    env = np.stack(env_channels, axis=-1)  # (H, 3)

    # Normalize against nominal 1-field exposure while preserving the contrast between
    # swept and unswept bands (and allowing camera auto-exposure partial compensation)
    norm_ref = max(0.42, float(duty) if duty <= 1.0 else float(duty) * 0.88)
    env = env / norm_ref

    # Ensure dark unswept trough has a realistic ambient/phosphor floor rather than dead 0.0
    floor = 0.18 if duty < 0.95 else 0.65
    env = np.clip(floor + (1.0 - floor) * env, 0.12, 1.65).astype(np.float32)

    info = {
        "crt_field_hz": round(f_crt, 2),
        "shutter_hz": round(f_shutter, 2),
        "duty_fields": round(duty, 4),
        "beat_hz": round(beat_hz, 3),
        "scan_bar_phase": round(phi0, 4),
        "env_min": round(float(env.min()), 4),
        "env_max": round(float(env.max()), 4),
        "env_row_ptp": round(float(env.mean(axis=1).max() - env.mean(axis=1).min()), 4),
    }
    return env, info


def apply_subpixel_moire(
    rgb: np.ndarray,
    spec: CRTReshootSpec,
    mask_pitch: float = 2.8,
    scan_pitch: float = 2.0,
    curvature: float = 0.22,
    frame_index: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Synthesize 2D spatial aliasing Moiré between the CRT's curved RGB phosphor shadow mask +
    scanline lattice and the reshooting camera's Bayer/CFA sensor grid.

    Why CRT off-screen Moiré forms curved rainbow rings & hyperbolic ripples:
      1. The CRT phosphor stripes/slots sit on a curved glass dome (`curvature`), so their
         projected spatial frequency $k_{\text{CRT}}(x, y)$ varies radially across the camera lens.
      2. The camera sensor is tilted by $\theta = \text{tilt\_deg}$ and samples R, G, B at shifted
         Bayer subpixel phases.
      3. Beating $\mathbf{k}_{\text{CRT}}(x, y)$ against $\mathbf{k}_{\text{sensor}}$ yields a low-frequency
         difference phase $\Delta \Phi(x, y)$ whose contours are concentric ellipses / hyperbolas!
    """
    m_amt = float(np.clip(spec.moire, 0.0, 1.5))
    if m_amt <= 1e-4:
        return rgb, 0.0

    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
    ny = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny

    theta = np.deg2rad(float(spec.tilt_deg) if abs(spec.tilt_deg) > 0.05 else 1.85)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    # Camera sensor coordinates rotated by camera tilt + off-axis perspective foreshortening
    persp = float(spec.perspective)
    scale_grad = 1.0 + 0.08 * persp * nx + 0.05 * abs(persp) * ny
    u_cam = (nx * cos_t - ny * sin_t) * scale_grad
    v_cam = (nx * sin_t + ny * cos_t) * scale_grad

    # CRT curved faceplate magnification mismatch -> quadratic phase curvature
    curv_k = 0.18 + 0.32 * float(np.clip(curvature, 0.0, 1.2))
    u_crt = nx * (1.0 + curv_k * r2)
    v_crt = ny * (1.0 + curv_k * 0.85 * r2)

    # Spatial beat frequencies (difference between CRT mask/scanline pitch and sensor pixel pitch):
    eff_mp = max(1.8, float(mask_pitch) if mask_pitch > 0.1 else 2.8)
    eff_sp = max(1.5, float(scan_pitch) if scan_pitch > 0.1 else 2.0)
    # Vertical phosphor triad beat (~8..22 cycles across width) + horizontal scanline beat (~6..16 cycles)
    freq_x = (w / eff_mp) * 0.085
    freq_y = (h / eff_sp) * 0.065

    # Difference phase Delta_Phi(x, y) between curved CRT lattice and rotated camera sensor lattice
    phase_v_grille = 2.0 * np.pi * freq_x * (u_crt - 0.92 * u_cam)
    phase_h_scan = 2.0 * np.pi * freq_y * (v_crt - 0.94 * v_cam)
    # 2D cross-term beat (diagonal slot/dot triad interaction with Bayer diagonal)
    phase_diag1 = 0.65 * phase_v_grille + 0.75 * phase_h_scan
    phase_diag2 = 0.65 * phase_v_grille - 0.75 * phase_h_scan

    # Subtle micro-vibration phase shift if camera is handheld or between frames
    micro_ph = float((frame_index * 0.07) % (2.0 * np.pi))

    # Bayer CFA subpixels sample R, G, B at phases (0, 2pi/3, 4pi/3) of the beat!
    lum = np.clip(rgb @ LUMA, 0.0, 1.5)
    # Moiré is most visible on mid-to-bright phosphor regions (an unlit black phosphor emits no light to alias)
    vis_gate = np.clip((lum + 0.06) / 0.65, 0.08, 1.15)

    out = rgb.copy()
    ch_phases = (-1.05, 0.0, +1.05)  # R, G, B Bayer subpixel phase offsets (rad)
    ch_weights = (0.22, 0.17, 0.24)
    for ch, (c_ph, c_w) in enumerate(zip(ch_phases, ch_weights)):
        fringe = (
            0.55 * np.cos(phase_v_grille + c_ph + micro_ph)
            + 0.30 * np.cos(phase_h_scan - 0.5 * c_ph)
            + 0.25 * np.cos(phase_diag1 + 1.2 * c_ph)
            + 0.15 * np.cos(phase_diag2 - 0.8 * c_ph)
        ).astype(np.float32)
        mod = 1.0 + (m_amt * c_w) * fringe * vis_gate
        out[..., ch] = out[..., ch] * mod

    moire_rms = float(np.std(out - rgb))
    return np.clip(out, 0.0, 1.8).astype(np.float32), moire_rms


def apply_offaxis_perspective_and_room(
    rgb: np.ndarray,
    spec: CRTReshootSpec,
    curvature: float = 0.22,
) -> np.ndarray:
    """
    Apply:
      1. Ambient room light diffuse phosphor reflection (black-level veiling) + curved glass
         window/lamp specular reflection (`reshoot_ambient_glare`),
      2. 9300K CRT phosphor cool cyan-blue camera white-balance shift & auto-exposure highlight
         bloom/clipping (`reshoot_wb_shift`),
      3. Off-axis camera tilt & perspective keystone geometry + monitor chassis/bezel surround +
         off-axis shallow depth-of-field (`reshoot_perspective`, `reshoot_tilt_deg`).
    """
    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32, copy=True)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
    ny = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny

    # 1. Ambient Room Light Glare & Unlit Phosphor Diffuse Reflection (`ambient_glare`)
    glare = float(np.clip(spec.ambient_glare, 0.0, 1.0))
    if glare > 1e-4:
        # Unlit P22 phosphor layer is a light grey-greenish powder behind tinted glass:
        # room light lifts the black floor into a warm/cool ambient veil
        phosphor_body_refl = np.array([0.095, 0.105, 0.102], dtype=np.float32)[None, None, :]
        room_grad = (0.75 + 0.25 * (0.5 - 0.4 * ny - 0.2 * nx))[..., None].astype(np.float32)
        out = out * (1.0 - 0.18 * glare) + (glare * room_grad) * phosphor_body_refl

        # Soft ceiling lamp / window reflection on the curved leaded faceplate glass
        curv_f = 1.0 + 0.25 * (0.3 + float(np.clip(curvature, 0.0, 1.0))) * r2
        rx = nx * curv_f
        ry = ny * curv_f
        # Window/softbox quadrilateral + ceiling bulb highlight
        win_x = np.exp(-0.5 * ((rx + 0.36) / 0.24) ** 2)
        win_y = np.exp(-0.5 * ((ry + 0.34) / 0.19) ** 2)
        lamp2 = 0.45 * np.exp(-0.5 * (((rx - 0.44) / 0.16) ** 2 + ((ry + 0.28) / 0.14) ** 2))
        fresnel = 0.30 + 0.70 * np.clip(r2 / 1.4, 0.0, 1.0)
        spec_glare = (win_x * win_y + lamp2) * fresnel
        spec_tint = np.array([1.02, 0.99, 0.92], dtype=np.float32)[None, None, :]
        out = out + (0.28 * glare) * spec_glare[..., None] * spec_tint

    # 2. 9300K CRT Phosphor White Balance Cast & Reshoot Camera AE Overexposure (`wb_shift`)
    wb = float(np.clip(spec.wb_shift, -1.0, 1.0))
    if abs(wb) > 1e-4:
        # Positive wb_shift = 9300K CRT screen shot by a 5200K daylight/indoor camera -> cool cyan-blue cast
        # Negative wb_shift = warm tungsten-balanced cast
        gains = np.array([1.0 - 0.18 * wb, 1.0 + 0.04 * wb, 1.0 + 0.26 * wb], dtype=np.float32)
        out = out * gains[None, None, :]
        if wb > 0.05:
            # Camera auto-exposure meters the dark bezel/room and overexposes the bright CRT centre
            lum = out @ LUMA
            hot = np.clip((lum - 0.62) / 0.35, 0.0, 1.0)
            halo = lowpass(hot, max(1.5, w / 90.0))
            out = out * (1.0 + 0.16 * wb) + (0.22 * wb) * halo[..., None]

    # 3. Off-Axis Camera Perspective & Tilt (`perspective`, `tilt_deg`)
    persp = float(np.clip(spec.perspective, -1.0, 1.0))
    tilt = float(np.clip(spec.tilt_deg, -20.0, 20.0))
    if abs(persp) > 1e-4 or abs(tilt) > 0.15:
        theta = np.deg2rad(tilt * 0.65)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        # Rotate & apply projective foreshortening (homography denominator `denom`)
        zoom_out = 1.0 + 0.075 * abs(persp) + 0.025 * (abs(tilt) / 10.0)
        rx = (nx * cos_t + ny * sin_t) * zoom_out
        ry = (-nx * sin_t + ny * cos_t) * zoom_out
        denom = np.maximum(0.55, 1.0 - 0.14 * persp * rx - 0.06 * abs(persp) * ry)
        sx = rx / denom
        sy = ry / denom

        src_x = (sx * 0.5 + 0.5) * (w - 1.0)
        src_y = (sy * 0.5 + 0.5) * (h - 1.0)
        inside = (sx >= -1.0) & (sx <= 1.0) & (sy >= -1.0) & (sy <= 1.0)

        x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 1)
        y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = np.clip(src_x - x0, 0.0, 1.0)[..., None].astype(np.float32)
        fy = np.clip(src_y - y0, 0.0, 1.0)[..., None].astype(np.float32)

        top = out[y0, x0] * (1.0 - fx) + out[y0, x1] * fx
        bot = out[y1, x0] * (1.0 - fx) + out[y1, x1] * fx
        warped = (top * (1.0 - fy) + bot * fy).astype(np.float32)

        # Dark plastic/metal CRT monitor casing & bezel surround outside [-1, 1]^2
        dist_out = np.maximum(np.abs(sx) - 1.0, np.abs(sy) - 1.0)
        bezel_edge = np.clip(1.0 - np.maximum(0.0, dist_out) / 0.015, 0.0, 1.0)[..., None]
        bezel_plastic = (0.025 + 0.035 * glare) * np.array([0.92, 0.94, 0.98], dtype=np.float32)[None, None, :]
        bezel_lip = np.exp(-0.5 * ((dist_out - 0.012) / 0.008) ** 2)[..., None] * 0.08
        warped = warped * bezel_edge + (1.0 - bezel_edge) * (bezel_plastic + bezel_lip)

        # Shallow depth-of-field blur on the far side of the tilted CRT screen
        if abs(persp) > 0.08:
            soft = lowpass(warped, max(0.8, 2.2 * abs(persp) * (w / 480.0)))
            dof_w = np.clip((np.abs(nx * np.sign(persp) + 0.35 * ny) - 0.35) / 0.65, 0.0, 1.0) * abs(persp)
            warped = warped * (1.0 - 0.65 * dof_w[..., None]) + soft * (0.65 * dof_w[..., None])

        out = warped

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_crt_reshoot(
    rgb: np.ndarray,
    spec: CRTReshootSpec,
    std: VideoStandard,
    frame_index: int,
    fps: float,
    mask_pitch: float = 2.8,
    scan_pitch: float = 2.0,
    curvature: float = 0.22,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Full Virtual Camera Reshooting CRT Off-Screen pipeline:
      1. Shutter-Field Scanning Beat Roll Bar & P22 Phosphor Decay Integration,
      2. 2D Sensor Pixel Lattice vs. CRT Shadow-Mask & Scanline Spatial Aliasing Moiré,
      3. Ambient Room Glare, 9300K Phosphor White-Balance Cast & Off-Axis Camera Perspective.
    """
    if not spec.is_active():
        return rgb, {"active": False}

    out = np.asarray(rgb, dtype=np.float32).copy()
    h, w = out.shape[:2]
    info: dict[str, Any] = {"active": True}

    # 1. Temporal Shutter-Field Beat Roll Bar (when reshoot_enabled is True)
    if spec.enabled:
        env, scan_info = compute_shutter_scan_envelope(h, spec, std, int(frame_index), float(fps))
        out = out * env[:, None, :]
        # Bright active electron-beam scanline crest also adds a subtle phosphor flare veil
        crest = np.maximum(0.0, env - 1.05)[:, None, :]
        if float(crest.max()) > 1e-4:
            out = out + 0.08 * crest
        info.update(scan_info)
    else:
        info["scan_bar_phase"] = 0.0
        info["env_row_ptp"] = 0.0

    # 2. 2D Spatial Aliasing Moiré between CRT Shadow Mask / Scanlines and Camera Sensor
    if spec.moire > 1e-4:
        out, moire_rms = apply_subpixel_moire(
            out, spec, mask_pitch=mask_pitch, scan_pitch=scan_pitch,
            curvature=curvature, frame_index=int(frame_index),
        )
        info["moire_rms"] = round(moire_rms, 5)
    else:
        info["moire_rms"] = 0.0

    # 3. Ambient Room Glare, 9300K Phosphor WB Shift & Off-Axis Perspective
    out = apply_offaxis_perspective_and_room(out, spec, curvature=curvature)
    info["ambient_glare"] = round(float(spec.ambient_glare), 3)
    info["perspective"] = round(float(spec.perspective), 3)
    info["tilt_deg"] = round(float(spec.tilt_deg), 2)
    return np.clip(out, 0.0, 1.0).astype(np.float32), info

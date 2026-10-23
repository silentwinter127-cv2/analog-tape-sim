"""
`atsim.codecs.sync_clock` -- Sync, Blanking (HBI/VBI), Sampling Clock & Cross-Standard System (Module 2).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.signal import SignalFrame, horizontal_shift, lowpass, rgb_to_yiq, yiq_to_rgb


@dataclass
class SyncClockSpec:
    """Parameters of the Sync & Clock subsystem (stored in the `player` layer)."""
    sync_clip: float = 0.0           # 0..1: Sync tip attenuation / slicer threshold compression
    h_hold: float = 0.0              # -1..1: Horizontal oscillator free-run offset / flagging
    v_hold: float = 0.0              # -1..1: Vertical oscillator hold offset / rolling
    clock_drift_ppm: float = 0.0     # -500..500 ppm: Crystal clock frequency offset
    clock_jitter: float = 0.0        # 0..1: PLL / ADC aperture phase jitter
    std_mismatch: int = 0            # 0=Match, 1=PAL->NTSC (50->60Hz), 2=NTSC->PAL (60->50Hz)
    pulse_cross: bool = False        # True: Center HBI/VBI pulse cross with lifted monitor pedestal

    @classmethod
    def from_params(cls, params: dict | None) -> "SyncClockSpec":
        p = params or {}
        ply = p.get("player") if isinstance(p.get("player"), dict) else p
        return cls(
            sync_clip=float(np.clip(ply.get("sync_clip", 0.0), 0.0, 1.0)),
            h_hold=float(np.clip(ply.get("h_hold", 0.0), -1.0, 1.0)),
            v_hold=float(np.clip(ply.get("v_hold", 0.0), -1.0, 1.0)),
            clock_drift_ppm=float(np.clip(ply.get("clock_drift_ppm", 0.0), -500.0, 500.0)),
            clock_jitter=float(np.clip(ply.get("clock_jitter", 0.0), 0.0, 1.0)),
            std_mismatch=int(round(float(np.clip(ply.get("std_mismatch", 0.0), 0.0, 2.0)))),
            pulse_cross=float(ply.get("pulse_cross", 0.0)) > 0.5,
        )

    def is_active(self, power_sync_shift: float = 0.0) -> bool:
        return (
            self.sync_clip > 1e-4
            or abs(self.h_hold) > 1e-4
            or abs(self.v_hold) > 1e-4
            or abs(self.clock_drift_ppm) > 1e-3
            or self.clock_jitter > 1e-4
            or self.std_mismatch > 0
            or self.pulse_cross
            or power_sync_shift > 1e-4
        )


def build_hbi_strip(
    height: int,
    hbi_w: int,
    sync_margin: float,
    burst_amp: float,
    burst_phase_rad: np.ndarray,
    is_pal: bool,
    frame_index: int,
    lifted_pedestal: bool = False,
) -> np.ndarray:
    h = max(2, int(height))
    bw = max(6, int(hbi_w))
    ped = 0.22 if lifted_pedestal else 0.05
    tip_depth = (0.20 if lifted_pedestal else 0.048) * float(np.clip(sync_margin, 0.05, 1.0))

    u = np.linspace(0.0, 1.0, bw, dtype=np.float32)[None, :]
    sync_gate = np.clip((u - 0.16) / 0.05, 0.0, 1.0) * np.clip((0.56 - u) / 0.05, 0.0, 1.0)
    y_hbi = np.full((h, bw), ped, dtype=np.float32) - tip_depth * sync_gate

    burst_env = np.clip((u - 0.60) / 0.04, 0.0, 1.0) * np.clip((0.88 - u) / 0.04, 0.0, 1.0)
    burst_carrier = np.sin(2.0 * np.pi * 9.0 * ((u - 0.60) / 0.28) + burst_phase_rad[:, None])
    b_amp = (0.14 if lifted_pedestal else 0.08) * float(np.clip(burst_amp, 0.0, 2.0))

    if is_pal:
        v_sign = np.where((np.arange(h) % 2) == 0, 1.0, -1.0)[:, None].astype(np.float32)
        i_hbi = -0.707 * b_amp * burst_env * burst_carrier
        q_hbi = 0.707 * v_sign * b_amp * burst_env * burst_carrier
    else:
        i_hbi = -b_amp * burst_env * burst_carrier
        q_hbi = 0.25 * b_amp * burst_env * np.cos(2.0 * np.pi * 9.0 * ((u - 0.60) / 0.28) + burst_phase_rad[:, None])

    if lifted_pedestal:
        y_hbi = y_hbi + 0.45 * b_amp * burst_env * burst_carrier

    rgb_hbi = np.clip(yiq_to_rgb(y_hbi, i_hbi, q_hbi), 0.0, 1.0).astype(np.float32)
    return rgb_hbi


def build_vbi_band(
    vbi_h: int,
    width: int,
    sync_margin: float,
    frame_index: int,
    seed: int,
    lifted_pedestal: bool = False,
) -> np.ndarray:
    vh = max(6, int(vbi_h))
    w = max(16, int(width))
    ped = 0.22 if lifted_pedestal else 0.055
    tip_low = max(0.01, ped - (0.20 if lifted_pedestal else 0.050) * float(np.clip(sync_margin, 0.05, 1.0)))

    band = np.full((vh, w, 3), ped, dtype=np.float32)
    u = np.linspace(0.0, 1.0, w, dtype=np.float32)

    r_pre = max(1, int(round(vh * 0.28)))
    r_ser = max(r_pre + 1, int(round(vh * 0.58)))
    r_post = max(r_ser + 1, int(round(vh * 0.80)))

    eq_mask = ((u < 0.045) | ((u >= 0.50) & (u < 0.545))).astype(np.float32)
    eq_line = ped * (1.0 - eq_mask) + tip_low * eq_mask
    band[:r_pre, :, :] = eq_line[None, :, None]
    band[r_ser:r_post, :, :] = eq_line[None, :, None]

    ser_mask = (((u >= 0.0) & (u < 0.44)) | ((u >= 0.50) & (u < 0.94))).astype(np.float32)
    ser_line = ped * (1.0 - ser_mask) + tip_low * ser_mask
    band[r_pre:r_ser, :, :] = ser_line[None, :, None]

    if r_post < vh:
        rng = np.random.default_rng((int(seed) * 314159 + int(frame_index) * 271828) & 0xFFFFFFFF)
        n_data = vh - r_post
        for dr in range(n_data):
            row_idx = r_post + dr
            if dr == n_data - 1 and n_data >= 2:
                ref_line = np.full((w, 3), ped, dtype=np.float32)
                c0, c1 = int(0.15 * w), int(0.45 * w)
                c2, c3 = int(0.50 * w), int(0.82 * w)
                ref_line[c0:c1] = 0.68 if lifted_pedestal else 0.52
                ref_line[c2:c3, 0] = 0.42
                ref_line[c2:c3, 1] = 0.55
                ref_line[c2:c3, 2] = 0.22
                band[row_idx] = ref_line
            else:
                bits = rng.integers(0, 2, size=64).astype(np.float32)
                bit_idx = np.clip((u * 64.0).astype(np.int32), 0, 63)
                vitc = ped + (0.38 if lifted_pedestal else 0.26) * bits[bit_idx] * ((u >= 0.12) & (u <= 0.92)).astype(np.float32)
                band[row_idx, :, :] = vitc[:, None]

    return band


def apply_sync_and_clock(
    rgb: np.ndarray,
    spec: SyncClockSpec,
    std,
    frame_index: int,
    fps: float,
    seed: int = 20240101,
    power_sync_shift: float = 0.0,
    tbc_on: bool = False,
    tbc_strength: float = 0.85,
) -> tuple[np.ndarray, dict]:
    if not spec.is_active(power_sync_shift):
        return rgb, {"locked": True}

    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    scale_h = h / 480.0
    out = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0).copy()
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 19937 + 7) & 0xFFFFFFFF)
    is_pal = str(getattr(std, "color_system", "NTSC")).upper() == "PAL"

    eff_clip = float(np.clip(spec.sync_clip + 0.85 * power_sync_shift, 0.0, 1.0))
    if tbc_on:
        eff_clip *= (1.0 - 0.72 * float(np.clip(tbc_strength, 0.0, 1.0)))
    sync_margin = float(np.clip(1.0 - eff_clip, 0.02, 1.0))

    rows_norm = np.arange(h, dtype=np.float32) / max(1.0, float(h - 1))
    y_plane, i_plane, q_plane = rgb_to_yiq(out)

    ppm = float(spec.clock_drift_ppm)
    jit = float(spec.clock_jitter)
    if tbc_on:
        jit *= (1.0 - 0.75 * float(np.clip(tbc_strength, 0.0, 1.0)))

    fifo_slip_lines = 0
    if abs(ppm) > 1e-3 or jit > 1e-4:
        frac_rate = ppm * 1e-6 * 45.0
        xx = np.arange(w, dtype=np.float32)[None, :]
        frame_drift_px = ((frame_index * ppm * 0.018) % 24.0) * scale_w
        row_skew_px = (rows_norm * ppm * 0.028 * scale_w)[:, None]
        jitter_px = (rng.standard_normal((h, 1)).astype(np.float32) * (jit * 3.8 * scale_w)) if jit > 1e-4 else 0.0

        slip_offset = np.zeros((h, 1), dtype=np.float32)
        if abs(ppm) > 80.0 and ((int(frame_index) + int(abs(ppm) // 40)) % 3 == 0):
            slip_row = int((0.25 + 0.55 * ((frame_index * 0.37) % 1.0)) * h)
            slip_span = max(2, int(round(12 * scale_h)))
            r1 = min(h, slip_row + slip_span)
            slip_offset[slip_row:r1, 0] = np.sign(ppm) * (14.0 * scale_w)
            fifo_slip_lines = r1 - slip_row

        src_x = np.clip(xx * (1.0 + frac_rate) + row_skew_px + frame_drift_px + jitter_px + slip_offset, 0.0, w - 1.001)
        x0 = np.floor(src_x).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        fx = (src_x - x0).astype(np.float32)
        r_idx = np.arange(h, dtype=np.int32)[:, None]

        y_plane = y_plane[r_idx, x0] * (1.0 - fx) + y_plane[r_idx, x1] * fx
        i_plane = i_plane[r_idx, x0] * (1.0 - fx) + i_plane[r_idx, x1] * fx
        q_plane = q_plane[r_idx, x0] * (1.0 - fx) + q_plane[r_idx, x1] * fx

        ph_err = (xx / float(w)) * (ppm * 0.0042) + (rng.standard_normal((h, 1)).astype(np.float32) * (jit * 0.55))
        cs, sn = np.cos(ph_err).astype(np.float32), np.sin(ph_err).astype(np.float32)
        i_plane, q_plane = i_plane * cs - q_plane * sn, i_plane * sn + q_plane * cs

    h_off = float(spec.h_hold)
    h_shift = np.zeros(h, dtype=np.float32)
    h_locked = True

    if eff_clip > 1e-4 or abs(h_off) > 1e-4:
        flag_amp = (h_off * 48.0 + eff_clip * 26.0 * (1.0 if h_off >= 0 else -1.0)) * scale_w
        tau_flag = 0.09 + 0.10 * eff_clip
        flagging = flag_amp * np.exp(-rows_norm / tau_flag) * np.cos(2.0 * np.pi * rows_norm * 1.5)

        row_luma = y_plane.mean(axis=1)
        bright_crush = np.clip(row_luma - 0.42, 0.0, 1.0) * (eff_clip * 42.0 * scale_w)
        slicer_noise = rng.standard_normal(h).astype(np.float32) * ((eff_clip ** 1.4) * 16.0 * scale_w)

        lock_limit = 0.28 * sync_margin
        if abs(h_off) > lock_limit:
            h_locked = False
            excess_h = h_off - np.sign(h_off) * lock_limit
            diag_shear = ((rows_norm * excess_h * 4.5 + frame_index * excess_h * 0.45) % 1.0) * w * 0.45 * np.sign(h_off)
        else:
            diag_shear = h_off * 18.0 * scale_w * (rows_norm - 0.5)

        h_shift = (flagging + bright_crush + slicer_noise + diag_shear).astype(np.float32)
        y_plane = horizontal_shift(y_plane, h_shift)
        i_plane = horizontal_shift(i_plane, h_shift)
        q_plane = horizontal_shift(q_plane, h_shift)

        burst_slip = np.clip(np.abs(h_shift) / max(1.0, 22.0 * scale_w), 0.0, 1.0)
        b_theta = (h_shift / max(1.0, 14.0 * scale_w)) * 0.65
        cs_b, sn_b = np.cos(b_theta)[:, None].astype(np.float32), np.sin(b_theta)[:, None].astype(np.float32)
        atten_c = (1.0 - 0.65 * (eff_clip ** 1.5) * burst_slip)[:, None].astype(np.float32)
        i_plane, q_plane = (i_plane * cs_b - q_plane * sn_b) * atten_c, (i_plane * sn_b + q_plane * cs_b) * atten_c

    mismatch = int(spec.std_mismatch)
    mismatch_roll_frac = 0.0
    if mismatch > 0:
        if mismatch == 1:
            mismatch_roll_frac = (frame_index * (9.94 / max(1.0, fps)) + 0.28) % 1.0
            slant_px = ((rows_norm - 0.5) * 28.0 * scale_w).astype(np.float32)
            y_plane = horizontal_shift(y_plane, slant_px)
            i_plane = horizontal_shift(i_plane, slant_px)
            q_plane = horizontal_shift(q_plane, slant_px)
            yy_m, xx_m = np.mgrid[0:h, 0:w].astype(np.float32)
            beat_wave = np.sin(2.0 * np.pi * (xx_m / max(2.0, 5.2 * scale_w) + yy_m / max(2.0, 7.5 * scale_h)) + frame_index * 1.3)
            i_plane = i_plane * 0.25 + 0.14 * beat_wave
            q_plane = q_plane * 0.25 + 0.14 * np.cos(2.0 * np.pi * (xx_m / max(2.0, 5.2 * scale_w) - yy_m / max(2.0, 7.5 * scale_h)))
            y_plane = y_plane + 0.045 * beat_wave * np.hypot(i_plane, q_plane)
        elif mismatch == 2:
            mismatch_roll_frac = (-frame_index * (9.94 / max(1.0, fps)) + 0.62) % 1.0
            slant_px = (-(rows_norm - 0.5) * 24.0 * scale_w).astype(np.float32)
            y_plane = horizontal_shift(y_plane, slant_px)
            i_plane = horizontal_shift(i_plane, slant_px)
            q_plane = horizontal_shift(q_plane, slant_px)
            alt_sign = np.where((np.arange(h) % 2) == 0, 1.0, -1.0)[:, None].astype(np.float32)
            q_plane = q_plane * alt_sign * 0.58
            i_plane = i_plane * (0.78 + 0.22 * alt_sign)

    out = np.clip(yiq_to_rgb(y_plane, i_plane, q_plane), 0.0, 1.0).astype(np.float32)

    v_off = float(spec.v_hold)
    v_locked = True
    v_roll_rows = 0

    v_lock_thresh = 0.18 * sync_margin
    if abs(v_off) > v_lock_thresh or eff_clip > 0.62 or mismatch > 0:
        v_locked = False
        if mismatch > 0 and abs(v_off) <= v_lock_thresh:
            roll_phase = mismatch_roll_frac
        else:
            drive = v_off if abs(v_off) > 1e-3 else (0.28 * (eff_clip - 0.55))
            raw_ph = (frame_index * drive * 0.22 + 0.35 * abs(drive)) % 1.0
            roll_phase = float((raw_ph + 0.14 * (1.0 - np.cos(2.0 * np.pi * raw_ph)) / (2.0 * np.pi)) % 1.0)

        v_roll_rows = int(round(roll_phase * h)) % h
        if v_roll_rows > 0:
            vbi_h = max(8, int(round(0.075 * h)))
            vbi_band = build_vbi_band(vbi_h, w, sync_margin, frame_index, seed, lifted_pedestal=spec.pulse_cross)
            rolled = np.roll(out, v_roll_rows, axis=0)
            r0 = max(0, v_roll_rows - vbi_h // 2)
            r1 = min(h, r0 + vbi_h)
            rolled[r0:r1, :, :] = vbi_band[: (r1 - r0), :, :]
            out = rolled
    elif abs(v_off) > 1e-4:
        shift_rows = int(round(v_off * 28.0 * scale_h))
        if abs(shift_rows) >= 1:
            vbi_h = min(h // 4, abs(shift_rows))
            vbi_band = build_vbi_band(max(6, vbi_h), w, sync_margin, frame_index, seed, lifted_pedestal=spec.pulse_cross)
            if shift_rows > 0:
                out = np.roll(out, shift_rows, axis=0)
                out[:vbi_h, :, :] = vbi_band[:vbi_h, :, :]
            else:
                out = np.roll(out, shift_rows, axis=0)
                out[h - vbi_h :, :, :] = vbi_band[:vbi_h, :, :]

    if spec.pulse_cross:
        out = np.clip(out * 0.84 + 0.14, 0.0, 1.0)

        hbi_w = max(12, int(round(0.14 * w)))
        vbi_h = max(10, int(round(0.11 * h)))
        burst_ph = (np.arange(h, dtype=np.float32) * 0.35 + h_shift * 0.18).astype(np.float32)
        burst_amp = (0.25 if mismatch == 1 else 1.0) * float(np.clip(sync_margin * 1.1, 0.1, 1.3))

        hbi_strip = build_hbi_strip(
            h, hbi_w, sync_margin, burst_amp, burst_ph, is_pal=is_pal,
            frame_index=frame_index, lifted_pedestal=True,
        )
        vbi_band = build_vbi_band(
            vbi_h, w, sync_margin, frame_index=frame_index, seed=seed, lifted_pedestal=True,
        )

        half_x = w // 2
        half_y = h // 2
        crossed = np.roll(np.roll(out, half_x, axis=1), half_y, axis=0)

        c0 = max(0, half_x - hbi_w // 2)
        c1 = min(w, c0 + hbi_w)
        for r in range(h):
            dx_r = int(round(float(h_shift[r]) * 0.5))
            rc0 = int(np.clip(c0 + dx_r, 0, w - (c1 - c0)))
            rc1 = rc0 + (c1 - c0)
            crossed[r, rc0:rc1, :] = hbi_strip[r, : (c1 - c0), :]

        r0 = max(0, half_y - vbi_h // 2)
        r1 = min(h, r0 + vbi_h)
        crossed[r0:r1, :, :] = vbi_band[: (r1 - r0), :, :]
        crossed[r0:r1, c0:c1, :] = np.minimum(crossed[r0:r1, c0:c1, :], hbi_strip[r0:r1, : (c1 - c0), :])
        out = crossed

    info = {
        "sync_margin": round(sync_margin, 4),
        "h_locked": h_locked,
        "v_locked": v_locked,
        "v_roll_rows": int(v_roll_rows),
        "fifo_slip_lines": int(fifo_slip_lines),
        "std_mismatch": mismatch,
        "pulse_cross": bool(spec.pulse_cross),
    }
    return np.clip(out, 0.0, 1.0).astype(np.float32), info

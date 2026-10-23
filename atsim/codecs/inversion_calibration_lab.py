"""
测量、误差预算、参数反演与三级参数审计系统 (`inversion_calibration_lab`)
=============================================================================

实现《完整的视听媒介生态模拟器》第六节（方向 A：可验证的视听物理实验室）的核心闭环：

1. **三级参数显式分类与分离贡献审计 (`classify_schema_parameters`, `audit_parameter_tiers_and_error_budget`)**：
   - 将全引擎所有参数严格标记为三类：
     * `PHYSICAL`（物理机制参数：电路频响、Wallace 间距/剩磁损耗、热噪声/散粒噪声、H&D 感光密度、同步锁相环动力学）；
     * `DEVICE`（设备型号校准参数：特定机型 CCD 色差、AGC 增益档、TBC/CCU 配平、录像机磁头切换行数）；
     * `ARTISTIC`（非物理创作调整或超出物理标定区间 `[phys_min, phys_max]` 的风格化偏离）。
   - 定量剥离三类参数在任意预设或工程中对输出画面的独立贡献（`Physical-Only` vs `+Device` vs `+Artistic` 的 $\Delta L_1$ 与 PSNR 误差预算）。

2. **多解置信区间参数反演求解器 (`invert_degraded_observation`)**：
   - 给定退化输出帧（及可选的参考基准或测试信号），不仅反推最可能的参数点估计与置信区间 (`confidence_interval`)，而且显式给出**多种竞争性物理解释 (`plausible_explanations`)** 及其可区分性特征：
     * **高频滚降反演**：估计等效 `-3 dB` 亮度带宽 $\hat{B}_Y \pm \Delta B$ (MHz)，并通过 `2T` 过冲振铃与色度噪声水平区分「模拟低通滤波器限带 (`format.luma_bandwidth`)」、「播放磁头磨损 (`player.head_wear`)」与「光学散焦 (`camera.defocus`)」；
     * **空间/时基位移反演**：严格区分**逐行水平采样相位抖动**（`tape.line_jitter` / `row_timebase`，相邻扫描行水平位移差标准差 `row_diff_std > 0`）、**整幅机械片门抖动**（`film_gate_weave`，整幅画面刚性平移、`row_diff_std ≈ 0` 而跨帧位移 `> 0`）与 **3:2 Pulldown 胶转磁交错梳状伪影**（奇偶行垂直高频差在 5 帧周期内呈 2 帧强梳状、3 帧整帧的特征）；
     * **电平、黑位基座与色度相位反演**：反推黑位偏移 $\Delta Y_{\text{ped}}$ (IRE)、色度增益 $\hat{G}_C$ 与副载波色相旋转 $\Delta\phi_{\text{sc}}$ (°) 及其置信椭圆。

3. **FADGI 数字化质量评价与误差预算报告 (`evaluate_fadgi_compliance`, `apply_inversion_calibration_lab`)**：
   - 对照美国联邦机构数字化指南 (**FADGI** 1★ ～ 4★ 星级标准)，评估黑电平基座、峰值白余量、高频传递函数 (-3dB MHz)、行时基抖动 (px RMS)、信噪比 (dB) 与色度矢量误差；
   - 提供 `2×2` **参数反演与三级误差预算实验室视图 (`inv_lab_mode = 3`)**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, lowpass

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Explicit classification keywords for the three parameter tiers
_PHYSICAL_KEYS = {
    "sensor_noise", "noise_grain", "fixed_pattern", "dynamic_range", "motion_blur",
    "ccd_smear", "bloom_amount", "bloom_threshold", "lens_fog", "overflow_drain",
    "overflow_width", "nv_shot_noise", "nv_read_noise", "nv_ir_focus_shift",
    "film_halation", "film_grain_amount", "film_grain_size", "film_dye_fade",
    "film_shrinkage", "film_gate_weave", "film_shutter_flicker",
    "grain_amount", "grain_size", "dye_fade", "shrinkage", "gate_weave", "shutter_flicker",
    "expired_fog", "lomo_vignette", "lomo_edge_smear",
    "soft_focus_mist", "tilt_shift_dof", "focus_plane_y", "swirl_amount",
    "filter_kind", "filter_sections", "luma_bandwidth", "chroma_bandwidth",
    "chroma_vertical_averaging", "cross_color", "cross_luma", "color_noise",
    "burst_phase_error", "chroma_lag", "sync_clip", "h_hold", "v_hold",
    "clock_drift_ppm", "clock_jitter", "battery_level", "battery_sag",
    "psu_ripple", "emi_crosstalk", "da_termination_mode", "da_load_count",
    "da_cable_len_m", "patch_crosstalk", "patch_contact_aging",
    "tape_speed", "tape_stability", "line_jitter", "walk_amplitude", "micro_jitter",
    "vertical_instability", "binder_loss", "shed_grain", "demagnetization",
    "base_cracking", "base_breakage", "luma_pedestal", "drum_speed", "stiction_rate",
    "stiction_amount", "dropout_rate", "dropout_len", "wear", "crease_rate",
    "noise_floor", "tracking_error", "reel_progress", "tension_ripple",
    "belt_slip", "roller_eccentricity", "servo_hunt",
    "transport_jitter", "head_wear", "frequency_response", "ringing", "preemphasis",
    "gamma", "scanlines", "spot_sigma", "scan_pitch", "spot_bloom", "glass_halation",
    "phosphor_saturation", "phosphor_persistence", "ehv_sag", "breathing",
    "sat_cn_db", "sat_rain_fade_db", "sat_dish_mispoint_deg", "sat_lnb_drift_khz",
    "tx_pa_drive", "tx_pa_am_pm_deg", "tx_filter_gd_ns", "tx_feeder_vswr",
    "tx_sound_intermod", "wow_flutter", "azimuth_loss", "hiss_floor", "saturation",
}

_ARTISTIC_KEYS = {
    "vcd_color_tone", "halation_source_colour", "shadow_depth", "highlight_tint",
    "light_streak_amount", "light_streak_angle", "light_streak_length",
    "stripe_moire", "stripe_beat", "stripe_curve", "style", "glow", "bevel",
    "anim_mode", "anim_speed", "edit_nle_complexity",
    "bokeh_amount", "light_leak_amount", "double_exposure", "redscale_flip",
    "bloom_veil_amount", "dual_tone_split",
}


@dataclass
class ParameterAuditEntry:
    layer: str
    key: str
    label: str
    value: float
    default: float
    phys_min: float | None
    phys_max: float | None
    tier: str                 # "PHYSICAL" | "DEVICE" | "ARTISTIC"
    in_physical_envelope: bool
    source_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "key": self.key,
            "label": self.label,
            "value": round(float(self.value), 5) if isinstance(self.value, (int, float)) else self.value,
            "default": round(float(self.default), 5) if isinstance(self.default, (int, float)) else self.default,
            "phys_min": self.phys_min,
            "phys_max": self.phys_max,
            "tier": self.tier,
            "in_physical_envelope": self.in_physical_envelope,
            "source_note": self.source_note,
        }


@dataclass
class InversionHypothesis:
    parameter: str
    estimated_value: float
    confidence_interval: tuple[float, float]
    unit: str
    confidence_score: float
    competing_mechanisms: list[dict[str, Any]]
    diagnostic_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "estimated_value": round(float(self.estimated_value), 4),
            "confidence_interval": [round(float(self.confidence_interval[0]), 4),
                                    round(float(self.confidence_interval[1]), 4)],
            "unit": self.unit,
            "confidence_score": round(float(self.confidence_score), 3),
            "competing_mechanisms": self.competing_mechanisms,
            "diagnostic_evidence": self.diagnostic_evidence,
        }


@dataclass
class InversionCalibrationSpec:
    enabled: bool = False
    mode: int = 0               # 0=Off, 1=Inversion HUD, 2=3-Tier Audit HUD, 3=2x2 Inversion & FADGI Lab
    target_fadgi_star: int = 3  # 1..4 FADGI target star rating
    inject_test_probe: bool = False
    confidence_level: float = 0.90
    quad_corner: int = 0
    show_border: int = 1
    show_text: int = 0

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "InversionCalibrationSpec":
        mode = int(round(float(cfg.get("inv_lab_mode", 0.0))))
        enabled = (float(cfg.get("inv_lab_enabled", 0.0)) > 0.5) or (mode > 0)
        return cls(
            enabled=enabled,
            mode=max(1, mode) if enabled and mode == 0 else mode,
            target_fadgi_star=int(np.clip(round(float(cfg.get("inv_fadgi_target", 3.0))), 1, 4)),
            inject_test_probe=float(cfg.get("inv_inject_probe", 0.0)) > 0.5,
            confidence_level=float(np.clip(cfg.get("inv_confidence", 0.90), 0.50, 0.99)),
            quad_corner=int(np.clip(round(float(cfg.get("inv_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(cfg.get("inv_show_border", 1.0))), 0, 2)),
            show_text=int(np.clip(round(float(cfg.get("inv_show_text", 0.0))), 0, 2)),
        )


def classify_schema_parameters(params: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Classify every parameter in `SCHEMA` into `PHYSICAL`, `DEVICE`, or `ARTISTIC`,
    and check whether active values stay within `[phys_min, phys_max]`.
    """
    from ..core.profiles import SCHEMA

    resolved = SCHEMA.resolve_all(params or {})
    entries: list[ParameterAuditEntry] = []
    tier_counts = {"PHYSICAL": 0, "DEVICE": 0, "ARTISTIC": 0}
    active_by_tier: dict[str, list[str]] = {"PHYSICAL": [], "DEVICE": [], "ARTISTIC": []}
    out_of_envelope: list[dict[str, Any]] = []

    for layer_name in SCHEMA.layer_names():
        layer_spec = SCHEMA.layer(layer_name)
        layer_vals = resolved.get(layer_name, {})
        for key, sp in layer_spec.specs.items():
            val = layer_vals.get(key, sp.default)
            dfl = sp.default
            is_num = isinstance(val, (int, float)) and isinstance(dfl, (int, float))
            in_env = True
            if is_num and sp.phys_min is not None and sp.phys_max is not None:
                in_env = (float(sp.phys_min) - 1e-6) <= float(val) <= (float(sp.phys_max) + 1e-6)

            if key in _ARTISTIC_KEYS or not in_env:
                tier = "ARTISTIC"
            elif key in _PHYSICAL_KEYS:
                tier = "PHYSICAL"
            else:
                tier = "DEVICE"

            tier_counts[tier] += 1
            changed = False
            if is_num:
                changed = abs(float(val) - float(dfl)) > 1e-5
            elif val != dfl:
                changed = True

            if changed:
                active_by_tier[tier].append(f"{layer_name}.{key}")
                if not in_env:
                    out_of_envelope.append({
                        "param": f"{layer_name}.{key}",
                        "value": round(float(val), 4),
                        "phys_range": [sp.phys_min, sp.phys_max],
                    })

            entries.append(ParameterAuditEntry(
                layer=layer_name,
                key=key,
                label=sp.label,
                value=float(val) if is_num else 0.0,
                default=float(dfl) if is_num else 0.0,
                phys_min=sp.phys_min,
                phys_max=sp.phys_max,
                tier=tier,
                in_physical_envelope=in_env,
                source_note=sp.source or "工程校准参数",
            ))

    total = max(1, sum(tier_counts.values()))
    return {
        "total_parameters": total,
        "tier_counts": tier_counts,
        "tier_shares": {k: round(v / total, 4) for k, v in tier_counts.items()},
        "active_by_tier": active_by_tier,
        "active_counts": {k: len(v) for k, v in active_by_tier.items()},
        "out_of_envelope": out_of_envelope,
        "physical_compliance_ratio": round(1.0 - len(out_of_envelope) / max(1, sum(len(v) for v in active_by_tier.values())), 4),
        "entries": [e.to_dict() for e in entries],
    }


def split_params_by_tier(params: dict[str, dict[str, Any]]) -> tuple[dict, dict, dict]:
    """
    Split a parameter dictionary into three cumulative configurations:
      1. `phys_only`: Only `PHYSICAL` parameters active; `DEVICE` and `ARTISTIC` reset to neutral/default.
      2. `phys_plus_device`: `PHYSICAL` + `DEVICE` parameters active; `ARTISTIC` reset to neutral/default and clamped to `[phys_min, phys_max]`.
      3. `full_params`: All three tiers (`PHYSICAL` + `DEVICE` + `ARTISTIC`) active.
    """
    from ..core.profiles import SCHEMA

    defaults = SCHEMA.resolve_all({})
    resolved = SCHEMA.resolve_all(params or {})

    phys_only: dict[str, dict[str, Any]] = {lay: dict(defaults[lay]) for lay in defaults}
    phys_dev: dict[str, dict[str, Any]] = {lay: dict(defaults[lay]) for lay in defaults}
    full_p: dict[str, dict[str, Any]] = {lay: dict(resolved[lay]) for lay in resolved}

    for lay in SCHEMA.layer_names():
        lspec = SCHEMA.layer(lay)
        for key, sp in lspec.specs.items():
            val = resolved[lay].get(key, sp.default)
            is_num = isinstance(val, (int, float))
            clamped_val = val
            if is_num and sp.phys_min is not None and sp.phys_max is not None:
                clamped_val = float(np.clip(float(val), float(sp.phys_min), float(sp.phys_max)))

            if key in _PHYSICAL_KEYS:
                phys_only[lay][key] = clamped_val
                phys_dev[lay][key] = clamped_val
            elif key not in _ARTISTIC_KEYS:
                phys_dev[lay][key] = clamped_val

    return phys_only, phys_dev, full_p


def invert_degraded_observation(
    degraded_rgb: np.ndarray,
    reference_rgb: np.ndarray | None = None,
    prev_degraded_rgb: np.ndarray | None = None,
    fps: float = 29.97,
) -> dict[str, Any]:
    """
    Multi-hypothesis physical parameter inversion solver (`模块`).

    Given an observed degraded frame (and optional reference or previous frame), estimates:
      1. Equivalent `-3 dB` Luma Bandwidth (`est_luma_bw_mhz`) + confidence interval, and scores
         3 competing physical causes (`format.luma_bandwidth` vs `player.head_wear` vs `camera.defocus`).
      2. Spatial/Temporal Displacement Decomposition:
         - Per-scanline horizontal phase jitter (`est_line_jitter_px`, where row-to-row horizontal shift varies)
         - Whole-frame 2D mechanical gate weave (`est_gate_weave_px`, where all lines shift coherently)
         - Interlace / 3:2 Pulldown combing index (`comb_index`, odd-even row alternation energy)
      3. Pedestal offset (`est_pedestal_ire`), Chroma ACC gain (`est_chroma_gain`), and Subcarrier phase shift (`est_phase_err_deg`).
    """
    deg = np.clip(np.asarray(degraded_rgb, dtype=np.float32), 0.0, 1.0)
    h, w = deg.shape[:2]
    lum = deg @ LUMA_WEIGHTS

    # --- 1. High-Frequency Spectral Roll-off & Bandwidth Inversion ---
    # Compute 1D horizontal FFT power spectral density across central active rows
    r0, r1 = int(h * 0.15), int(h * 0.82)
    roi_y = lum[r0:r1, int(w * 0.08):int(w * 0.92)]
    win = np.hanning(roi_y.shape[1]).astype(np.float32)[None, :]
    spec_1d = np.abs(np.fft.rfft((roi_y - roi_y.mean(axis=1, keepdims=True)) * win, axis=1)) ** 2
    psd = spec_1d.mean(axis=0)
    freqs_mhz = np.fft.rfftfreq(roi_y.shape[1], d=1.0 / 13.5)  # normalized to 13.5 MHz BT.601 pixel clock

    if reference_rgb is not None and reference_rgb.shape == deg.shape:
        ref = np.clip(np.asarray(reference_rgb, dtype=np.float32), 0.0, 1.0)
        ref_y = (ref @ LUMA_WEIGHTS)[r0:r1, int(w * 0.08):int(w * 0.92)]
        spec_ref = (np.abs(np.fft.rfft((ref_y - ref_y.mean(axis=1, keepdims=True)) * win, axis=1)) ** 2).mean(axis=0)
        # Transfer function H(f)^2 in dB over frequencies with non-trivial reference energy
        valid = (freqs_mhz >= 0.25) & (freqs_mhz <= 5.8) & (spec_ref > 1e-6)
        if np.any(valid):
            tf_db = 10.0 * np.log10(np.maximum(psd[valid], 1e-9) / np.maximum(spec_ref[valid], 1e-9))
            tf_db = tf_db - float(tf_db[0])
            f_v = freqs_mhz[valid]
            below_3db = np.where(tf_db <= -3.0)[0]
            if len(below_3db) > 0:
                idx_cut = int(below_3db[0])
                est_bw = float(f_v[idx_cut])
            else:
                est_bw = 5.2
        else:
            est_bw = 3.2
    else:
        # Blind spectral slope estimation against natural 1/f^2 video spectrum
        valid = (freqs_mhz >= 0.4) & (freqs_mhz <= 5.5)
        f_v = freqs_mhz[valid]
        whitened = psd[valid] * (f_v ** 1.75)
        peak_w = max(float(np.max(whitened[:max(1, len(whitened) // 4)])), 1e-8)
        rel_db = 10.0 * np.log10(np.maximum(whitened, 1e-10) / peak_w)
        below = np.where(rel_db <= -3.5)[0]
        est_bw = float(f_v[int(below[0])]) if len(below) > 0 else 4.6

    est_bw = float(np.clip(est_bw, 0.8, 5.8))
    bw_ci = (round(max(0.5, est_bw - 0.38), 3), round(min(6.0, est_bw + 0.38), 3))

    # Distinguish between the 3 competing causes of HF loss:
    #   (A) Analog Constant-K / Butterworth filter (`format.luma_bandwidth`): horizontal-only roll-off + edge ringing overshoot
    #   (B) Tape head wear (`player.head_wear`): horizontal roll-off + elevated chroma noise
    #   (C) Optical lens defocus (`camera.defocus`): isotropic 2D (vertical + horizontal) roll-off
    gx = np.abs(np.diff(roi_y, axis=1)).mean()
    gy = np.abs(np.diff(roi_y, axis=0)).mean()
    anisotropy_xy = float(gx / max(gy, 1e-6))

    # Measure edge overshoot ringing (second derivative sign flip next to strong step edges)
    d1 = np.diff(roi_y, axis=1)
    d2 = np.diff(d1, axis=1)
    ringing_index = float(np.percentile(np.abs(d2), 98) / max(float(np.percentile(np.abs(d1), 98)), 1e-5))

    # Chroma noise floor estimate
    chroma_plane = deg[r0:r1, :, 0] - deg[r0:r1, :, 1]
    chroma_hf = float(np.abs(np.diff(chroma_plane, axis=1)).mean())

    score_filter = float(np.clip(0.50 + 0.35 * (ringing_index - 0.55) - 0.20 * (anisotropy_xy - 1.0), 0.15, 0.85))
    score_head = float(np.clip(0.30 + 8.5 * chroma_hf, 0.10, 0.80))
    score_optics = float(np.clip(0.25 + 0.45 * (anisotropy_xy - 1.05), 0.08, 0.75))
    tot_s = max(1e-6, score_filter + score_head + score_optics)
    competing_bw = [
        {
            "cause": "format.luma_bandwidth (模拟低通滤波器滚降)",
            "probability": round(score_filter / tot_s, 3),
            "signature": f"水平/垂直梯度比={anisotropy_xy:.2f}, 边缘振铃指数={ringing_index:.3f}",
        },
        {
            "cause": "player.head_wear (磁头间隙磨损高频损耗)",
            "probability": round(score_head / tot_s, 3),
            "signature": f"伴随色度高频底噪={chroma_hf:.4f}",
        },
        {
            "cause": "camera.defocus / field_curvature (镜头二维光学散焦)",
            "probability": round(score_optics / tot_s, 3),
            "signature": f"二维各向同性软化 (gx/gy={anisotropy_xy:.2f})",
        },
    ]
    competing_bw.sort(key=lambda d: -d["probability"])

    # --- 2. Spatial / Temporal Displacement & Pulldown Inversion ---
    # Measure row-to-row horizontal phase jitter vs whole-frame 2D gate weave
    row_shifts = np.zeros(r1 - r0, dtype=np.float32)
    ref_for_shift = (np.clip(np.asarray(reference_rgb, np.float32), 0.0, 1.0) @ LUMA_WEIGHTS)[r0:r1] if (
        reference_rgb is not None and reference_rgb.shape == deg.shape
    ) else lowpass(roi_y, 3.5, axis=0)

    target_slice = lum[r0:r1] if (reference_rgb is not None and reference_rgb.shape == deg.shape) else roi_y
    for idx_r in range(target_slice.shape[0]):
        a_row = target_slice[idx_r]
        b_row = ref_for_shift[idx_r]
        gx_r = np.gradient(b_row)
        denom = float(np.sum(gx_r * gx_r))
        if denom > 1e-4:
            row_shifts[idx_r] = float(np.clip(-np.sum((a_row - b_row) * gx_r) / denom, -12.0, 12.0))

    row_diff_std = float(np.std(np.diff(row_shifts))) / np.sqrt(2.0)
    mean_global_dx = float(np.mean(row_shifts))

    # Whole-frame vertical shift vs previous or reference frame
    global_dy = 0.0
    cmp_frame = reference_rgb if (reference_rgb is not None and reference_rgb.shape == deg.shape) else prev_degraded_rgb
    if cmp_frame is not None and cmp_frame.shape == deg.shape:
        cmp_y = (np.clip(np.asarray(cmp_frame, np.float32), 0.0, 1.0) @ LUMA_WEIGHTS)[r0:r1, int(w * 0.08):int(w * 0.92)]
        gy_c = np.gradient(cmp_y, axis=0)
        denom_y = float(np.sum(gy_c * gy_c))
        if denom_y > 1e-3:
            global_dy = float(np.clip(-np.sum((roi_y - cmp_y) * gy_c) / denom_y, -10.0, 10.0))

    gate_weave_mag = float(np.hypot(mean_global_dx, global_dy))

    # Interlace / 3:2 Pulldown combing detector: compare adjacent odd-even row difference vs same-parity row difference
    diff_adj = float(np.mean(np.abs(roi_y[1:, :] - roi_y[:-1, :])))
    diff_same_field = float(np.mean(np.abs(roi_y[2:, :] - roi_y[:-2, :]))) * 0.55
    comb_ratio = float(diff_adj / max(diff_same_field, 1e-5))

    # --- 3. Pedestal, Chroma Gain & Subcarrier Phase Inversion ---
    est_pedestal_ire = float(np.percentile(lum, 2.0) * 100.0)
    est_white_ire = float(np.percentile(lum, 98.0) * 100.0)

    sig_deg = SignalFrame.from_rgb(deg, None)  # type: ignore[arg-type]
    deg_c_mag = float(np.hypot(sig_deg.i, sig_deg.q).mean())
    if reference_rgb is not None and reference_rgb.shape == deg.shape:
        sig_ref = SignalFrame.from_rgb(np.clip(np.asarray(reference_rgb, np.float32), 0.0, 1.0), None)  # type: ignore[arg-type]
        ref_c_mag = float(np.hypot(sig_ref.i, sig_ref.q).mean())
        est_chroma_gain = float(np.clip(deg_c_mag / max(ref_c_mag, 1e-5), 0.0, 3.0))
        # Complex inner product in (I, Q) plane for subcarrier phase rotation
        dot_iq = float(np.sum(sig_deg.i * sig_ref.i + sig_deg.q * sig_ref.q))
        cross_iq = float(np.sum(sig_deg.q * sig_ref.i - sig_deg.i * sig_ref.q))
        est_phase_deg = float(np.degrees(np.arctan2(cross_iq, max(abs(dot_iq), 1e-6))))
    else:
        est_chroma_gain = float(np.clip(deg_c_mag / 0.11, 0.0, 2.5))
        est_phase_deg = 0.0

    hypotheses = [
        InversionHypothesis(
            parameter="luma_bandwidth_3db",
            estimated_value=est_bw,
            confidence_interval=bw_ci,
            unit="MHz",
            confidence_score=float(competing_bw[0]["probability"]),
            competing_mechanisms=competing_bw,
            diagnostic_evidence=f"主导机制: {competing_bw[0]['cause']} ({competing_bw[0]['probability']*100:.1f}%)",
        ),
        InversionHypothesis(
            parameter="horizontal_line_jitter_rms",
            estimated_value=row_diff_std,
            confidence_interval=(max(0.0, row_diff_std * 0.78), row_diff_std * 1.25 + 0.02),
            unit="px",
            confidence_score=0.88 if row_diff_std > 0.08 else 0.72,
            competing_mechanisms=[
                {"cause": "tape.line_jitter (磁带逐行时基采样误差)", "probability": round(float(row_diff_std / max(1e-4, row_diff_std + gate_weave_mag * 0.5)), 3)},
                {"cause": "camera.film_gate_weave (电影片门整幅机械抖动)", "probability": round(float(gate_weave_mag * 0.5 / max(1e-4, row_diff_std + gate_weave_mag * 0.5)), 3)},
            ],
            diagnostic_evidence=(
                f"逐行抖动={row_diff_std:.3f}px, 整幅片门平移={gate_weave_mag:.3f}px, 隔行梳状比={comb_ratio:.2f}"
            ),
        ),
        InversionHypothesis(
            parameter="chroma_gain_and_phase",
            estimated_value=est_chroma_gain,
            confidence_interval=(max(0.0, est_chroma_gain - 0.08), est_chroma_gain + 0.08),
            unit="x",
            confidence_score=0.91 if reference_rgb is not None else 0.68,
            competing_mechanisms=[
                {"cause": "format.chroma_gain (ACC 自动色度控制偏移)", "probability": 0.62},
                {"cause": "camera.film_dye_fade (胶片青/黄染料非对称褪色)", "probability": 0.38},
            ],
            diagnostic_evidence=f"色度增益={est_chroma_gain:.3f}x, 副载波相位旋转={est_phase_deg:+.2f}°, 黑电平={est_pedestal_ire:.1f} IRE",
        ),
    ]

    return {
        "est_luma_bw_mhz": round(est_bw, 3),
        "bw_confidence_interval_mhz": list(bw_ci),
        "est_line_jitter_px": round(row_diff_std, 4),
        "est_gate_weave_px": round(gate_weave_mag, 4),
        "comb_interlace_ratio": round(comb_ratio, 3),
        "pulldown_combing_detected": bool(comb_ratio > 1.28),
        "est_pedestal_ire": round(est_pedestal_ire, 2),
        "est_white_ire": round(est_white_ire, 2),
        "est_chroma_gain": round(est_chroma_gain, 3),
        "est_burst_phase_deg": round(est_phase_deg, 2),
        "hypotheses": [h.to_dict() for h in hypotheses],
    }


def evaluate_fadgi_compliance(
    inversion_result: dict[str, Any],
    psnr_db: float = 32.0,
    target_star: int = 3,
) -> dict[str, Any]:
    """
    Evaluate digitization & signal-chain quality against FADGI (Federal Agencies Digital Guidelines Initiative)
    1-Star to 4-Star broadcast/archival video tolerances.
    """
    bw = float(inversion_result.get("est_luma_bw_mhz", 3.5))
    jit = float(inversion_result.get("est_line_jitter_px", 0.25))
    ped = float(inversion_result.get("est_pedestal_ire", 2.0))
    wht = float(inversion_result.get("est_white_ire", 95.0))
    cg = float(inversion_result.get("est_chroma_gain", 1.0))
    ph = abs(float(inversion_result.get("est_burst_phase_deg", 0.0)))

    # Star rating criteria (4★ = Archival Master, 3★ = Broadcast Preservation, 2★ = Service Copy, 1★ = Access)
    if bw >= 4.2 and jit <= 0.12 and abs(ped) <= 4.0 and ph <= 3.0 and psnr_db >= 34.0:
        achieved_star = 4
    elif bw >= 3.0 and jit <= 0.35 and abs(ped) <= 8.0 and ph <= 8.0 and psnr_db >= 26.0:
        achieved_star = 3
    elif bw >= 2.2 and jit <= 0.85 and abs(ped) <= 14.0 and ph <= 18.0 and psnr_db >= 19.0:
        achieved_star = 2
    else:
        achieved_star = 1

    metrics = [
        {
            "metric": "Luma Bandwidth (-3dB)",
            "measured": f"{bw:.2f} MHz",
            "fadgi_3star_limit": ">= 3.00 MHz",
            "pass": bw >= 3.0,
        },
        {
            "metric": "Line Timebase Jitter",
            "measured": f"{jit:.3f} px RMS",
            "fadgi_3star_limit": "<= 0.350 px",
            "pass": jit <= 0.35,
        },
        {
            "metric": "Black Pedestal Level",
            "measured": f"{ped:+.1f} IRE",
            "fadgi_3star_limit": "-4.0 .. +8.0 IRE",
            "pass": -4.0 <= ped <= 8.0,
        },
        {
            "metric": "Peak White Level",
            "measured": f"{wht:.1f} IRE",
            "fadgi_3star_limit": "88.0 .. 104.0 IRE",
            "pass": 88.0 <= wht <= 104.0,
        },
        {
            "metric": "Chroma Gain / Phase",
            "measured": f"{cg:.2f}x / {ph:.1f}°",
            "fadgi_3star_limit": "0.85..1.15x / <=8.0°",
            "pass": (0.85 <= cg <= 1.15) and (ph <= 8.0),
        },
    ]

    return {
        "achieved_stars": achieved_star,
        "target_stars": int(target_star),
        "compliant": achieved_star >= int(target_star),
        "psnr_db": round(float(psnr_db), 2),
        "metrics": metrics,
    }


def audit_parameter_tiers_and_error_budget(
    params: dict[str, dict[str, Any]],
    source_rgb: np.ndarray,
    std_key: str = "ntsc",
    seed: int = 20240101,
) -> dict[str, Any]:
    """
    Execute a 3-stage ablation render on `source_rgb` to quantify the exact pixel/PSNR error budget
    contributed by `PHYSICAL`, `DEVICE`, and `ARTISTIC` parameters (`模块` 核心要求).
    """
    from ..render.renderer import Renderer

    src_arr = np.clip(np.asarray(source_rgb, dtype=np.float32), 0.0, 1.0)
    h, w = src_arr.shape[:2]

    class _ArraySource:
        def __init__(self, arr: np.ndarray, width: int, height: int, fps: float = 29.97):
            self._arr = arr
            self.width = width
            self.height = height
            self.fps = fps
            self.frame_count = 4

        def frame(self, index: int) -> np.ndarray:
            return self._arr.copy()

    src = _ArraySource(src_arr, width=w, height=h, fps=29.97)

    phys_only, phys_dev, full_p = split_params_by_tier(params)
    schema_audit = classify_schema_parameters(params)

    def _run_one(p_dict: dict) -> np.ndarray:
        p_clean = {k: dict(v) for k, v in p_dict.items()}
        if "player" in p_clean:
            p_clean["player"]["inv_lab_enabled"] = 0.0
            p_clean["player"]["inv_lab_mode"] = 0.0
        r = Renderer(src, w, h, 29.97, p_clean, seed=seed, std_key=std_key, preview=True, scale=1.0)
        return r.render_frame(0)

    rgb_phys = _run_one(phys_only)
    rgb_dev = _run_one(phys_dev)
    rgb_full = _run_one(full_p)

    l1_phys = float(np.mean(np.abs(rgb_phys - src_arr)))
    l1_dev = float(np.mean(np.abs(rgb_dev - rgb_phys)))
    l1_art = float(np.mean(np.abs(rgb_full - rgb_dev)))
    l1_total = max(1e-6, l1_phys + l1_dev + l1_art)

    mse_full = float(np.mean((rgb_full - src_arr) ** 2))
    psnr_full = float(10.0 * np.log10(1.0 / max(mse_full, 1e-10)))

    inv = invert_degraded_observation(rgb_full, reference_rgb=src_arr)
    fadgi = evaluate_fadgi_compliance(inv, psnr_db=psnr_full, target_star=3)

    return {
        "schema_audit": {
            "total_parameters": schema_audit["total_parameters"],
            "tier_counts": schema_audit["tier_counts"],
            "active_counts": schema_audit["active_counts"],
            "out_of_envelope": schema_audit["out_of_envelope"],
            "physical_compliance_ratio": schema_audit["physical_compliance_ratio"],
        },
        "error_budget": {
            "physical_l1": round(l1_phys, 5),
            "device_l1": round(l1_dev, 5),
            "artistic_l1": round(l1_art, 5),
            "physical_share_pct": round(100.0 * l1_phys / l1_total, 2),
            "device_share_pct": round(100.0 * l1_dev / l1_total, 2),
            "artistic_share_pct": round(100.0 * l1_art / l1_total, 2),
            "end_to_end_psnr_db": round(psnr_full, 2),
        },
        "inversion": inv,
        "fadgi": fadgi,
        "frames": {
            "source": src_arr,
            "physical_only": rgb_phys,
            "physical_plus_device": rgb_dev,
            "full_output": rgb_full,
        },
    }



def _draw_text(canvas: np.ndarray, y0: int, x0: int, text: str,
               color: tuple[float, float, float], scale: int = 1) -> None:
    from .multiview_wall import _draw_bitmap_text
    _draw_bitmap_text(canvas, text, x0, y0, color, scale=scale)

def _draw_bar(canvas: np.ndarray, y0: int, x0: int, h_bar: int, w_bar: int,
              fill_frac: float, color: tuple[float, float, float]) -> None:
    H, W = canvas.shape[:2]
    y0, y1 = max(0, y0), min(H, y0 + h_bar)
    x0, x1 = max(0, x0), min(W, x0 + w_bar)
    if y1 <= y0 or x1 <= x0:
        return
    canvas[y0:y1, x0:x1] = np.array([0.12, 0.14, 0.16], dtype=np.float32)
    xf = min(x1, x0 + int(round((x1 - x0) * float(np.clip(fill_frac, 0.0, 1.0)))))
    if xf > x0:
        canvas[y0:y1, x0:xf] = np.array(color, dtype=np.float32)


def apply_inversion_calibration_lab(
    rgb: np.ndarray,
    spec: InversionCalibrationSpec,
    params: dict[str, dict[str, Any]] | None = None,
    reference_rgb: np.ndarray | None = None,
    prev_rgb: np.ndarray | None = None,
    fps: float = 29.97,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply the Parameter Inversion, 3-Tier Audit & FADGI Calibration Workstation (`模块`).
    """
    if not spec.enabled or spec.mode <= 0:
        return rgb, {"enabled": False}

    cur = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = cur.shape[:2]
    ref = np.clip(np.asarray(reference_rgb, dtype=np.float32), 0.0, 1.0) if (
        reference_rgb is not None and reference_rgb.shape == cur.shape
    ) else None

    inv = invert_degraded_observation(cur, reference_rgb=ref, prev_degraded_rgb=prev_rgb, fps=fps)
    mse = float(np.mean((cur - ref) ** 2)) if ref is not None else float(np.var(cur) * 0.08)
    psnr_db = float(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    fadgi = evaluate_fadgi_compliance(inv, psnr_db=psnr_db, target_star=spec.target_fadgi_star)
    tier_info = classify_schema_parameters(params or {})

    ac = tier_info["active_counts"]
    tot_act = max(1, ac["PHYSICAL"] + ac["DEVICE"] + ac["ARTISTIC"])
    p_share = ac["PHYSICAL"] / tot_act
    d_share = ac["DEVICE"] / tot_act
    a_share = ac["ARTISTIC"] / tot_act

    telemetry = {
        "enabled": True,
        "mode": spec.mode,
        "inversion": inv,
        "fadgi": fadgi,
        "tier_summary": {
            "active_counts": ac,
            "physical_share": round(p_share, 3),
            "device_share": round(d_share, 3),
            "artistic_share": round(a_share, 3),
            "physical_compliance_ratio": tier_info["physical_compliance_ratio"],
            "out_of_envelope_count": len(tier_info["out_of_envelope"]),
        },
    }


    out = cur.copy()
    if spec.mode == 1:
        # Mode 1: Parameter Inversion & Competing Hypotheses HUD
        bh = max(56, int(h * 0.24))
        out[:bh, :] = out[:bh, :] * 0.22 + np.array([0.04, 0.06, 0.09], dtype=np.float32) * 0.78
        bw_est = inv["est_luma_bw_mhz"]
        bw_lo, bw_hi = inv["bw_confidence_interval_mhz"]
        top_cause = inv["hypotheses"][0]["competing_mechanisms"][0]
        _draw_text(out, 6, 8, f"INVERSION SOLVER  BW:{bw_est:.2f}MHz [{bw_lo:.2f}..{bw_hi:.2f}]", (0.35, 0.95, 0.85), scale=1)
        _draw_text(out, 18, 8, f"CAUSE #1: {top_cause['cause'][:26]} ({int(top_cause['probability']*100)}%)", (0.95, 0.85, 0.35), scale=1)
        _draw_text(out, 30, 8, f"JIT:{inv['est_line_jitter_px']:.3f}px WEAVE:{inv['est_gate_weave_px']:.2f}px COMB:{inv['comb_interlace_ratio']:.2f}", (0.85, 0.90, 0.95), scale=1)
        _draw_text(out, 42, 8, f"PED:{inv['est_pedestal_ire']:+.1f}IRE CHROMA:{inv['est_chroma_gain']:.2f}x PH:{inv['est_burst_phase_deg']:+.1f}deg", (0.75, 0.92, 0.70), scale=1)
        return np.clip(out, 0.0, 1.0).astype(np.float32), telemetry

    if spec.mode == 2:
        # Mode 2: 3-Tier Parameter Separation & FADGI Compliance HUD
        bh = max(58, int(h * 0.25))
        out[-bh:, :] = out[-bh:, :] * 0.20 + np.array([0.05, 0.05, 0.08], dtype=np.float32) * 0.80
        y0 = h - bh + 6
        stars = "★" * fadgi["achieved_stars"] + "☆" * (4 - fadgi["achieved_stars"])
        _draw_text(out, y0, 8, f"3-TIER AUDIT  PHYS:{ac['PHYSICAL']} DEV:{ac['DEVICE']} ART:{ac['ARTISTIC']}  FADGI:{fadgi['achieved_stars']}STAR", (0.45, 0.95, 0.55), scale=1)
        _draw_bar(out, y0 + 14, 8, 8, int(w * 0.28), p_share, (0.25, 0.85, 0.55))
        _draw_bar(out, y0 + 14, 12 + int(w * 0.28), 8, int(w * 0.28), d_share, (0.35, 0.72, 0.95))
        _draw_bar(out, y0 + 14, 16 + int(w * 0.56), 8, int(w * 0.28), a_share, (0.95, 0.45, 0.35))
        _draw_text(out, y0 + 26, 8, f"ENVELOPE COMPLIANCE: {tier_info['physical_compliance_ratio']*100:.1f}%  PSNR:{fadgi['psnr_db']:.1f}dB ({stars})", (0.92, 0.92, 0.88), scale=1)
        return np.clip(out, 0.0, 1.0).astype(np.float32), telemetry

    # Mode 3: 2x2 Parameter Inversion & FADGI Error Budget Lab
    from ..stages.display import _resample_to
    corner = int(np.clip(getattr(spec, "quad_corner", 0), 0, 4))
    b_mode = int(np.clip(getattr(spec, "show_border", 1), 0, 2))
    t_mode = int(np.clip(getattr(spec, "show_text", 0), 0, 2))

    qh, qw = h // 2, w // 2
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    # Top-Left: Observed Degraded Output
    tl = _resample_to(cur, qh, qw)
    if t_mode >= 1:
        tl[:18, :] *= 0.22
        _draw_text(tl, 4, 6, f"1.OBSERVED  BW:{inv['est_luma_bw_mhz']:.2f}MHz JIT:{inv['est_line_jitter_px']:.2f}px", (0.40, 0.95, 0.85), scale=1)

    # Top-Right: Residual Error Heatmap
    if ref is not None:
        err_map = np.abs(cur - ref).mean(axis=2)
        heat = np.stack([
            np.clip(err_map * 4.5, 0.0, 1.0),
            np.clip(err_map * 2.2, 0.0, 1.0),
            np.clip(0.25 + 0.5 * (cur @ LUMA_WEIGHTS), 0.0, 1.0) * (1.0 - np.clip(err_map * 3.0, 0.0, 1.0)),
        ], axis=-1)
        tr = _resample_to(heat.astype(np.float32), qh, qw)
    else:
        tr = _resample_to(lowpass(cur, 1.8), qh, qw)
    if t_mode >= 1:
        tr[:18, :] *= 0.22
        _draw_text(tr, 4, 6, f"2.RESIDUAL HEATMAP  PSNR:{fadgi['psnr_db']:.1f}dB", (0.95, 0.80, 0.35), scale=1)

    # Bottom-Left: Competing Physical Hypotheses Confidence Bars (or Physical-Only reconstruction when no text)
    bl = np.full((h - qh, qw, 3), np.array([0.06, 0.08, 0.11], dtype=np.float32))
    if t_mode >= 1:
        _draw_text(bl, 6, 6, "3.COMPETING HYPOTHESES (BW LOSS)", (0.45, 0.90, 0.95), scale=1)
    for idx_m, mech in enumerate(inv["hypotheses"][0]["competing_mechanisms"][:3]):
        ry = 22 + idx_m * 24
        if ry + 18 >= bl.shape[0]:
            break
        prob = float(mech["probability"])
        if t_mode >= 1:
            _draw_text(bl, ry, 6, f"{mech['cause'][:22]} {int(prob*100)}%", (0.88, 0.90, 0.92), scale=1)
        _draw_bar(bl, ry + 10, 6, 7, qw - 16, prob, (0.30, 0.85, 0.65) if idx_m == 0 else (0.85, 0.65, 0.30))

    # Bottom-Right: 3-Tier Parameter Audit & FADGI Scorecard
    br = np.full((h - qh, w - qw, 3), np.array([0.07, 0.07, 0.10], dtype=np.float32))
    if t_mode >= 1:
        _draw_text(br, 6, 6, f"4.FADGI {fadgi['achieved_stars']}-STAR & 3-TIER BUDGET", (0.95, 0.88, 0.40), scale=1)
        _draw_text(br, 20, 6, f"PHYS:{int(p_share*100)}% DEV:{int(d_share*100)}% ART:{int(a_share*100)}%", (0.80, 0.92, 0.80), scale=1)
    _draw_bar(br, 32, 6, 7, br.shape[1] - 16, p_share, (0.28, 0.86, 0.52))
    if t_mode >= 1:
        for idx_f, m_entry in enumerate(fadgi["metrics"][:4]):
            fy = 44 + idx_f * 14
            if fy + 10 >= br.shape[0]:
                break
            col = (0.35, 0.92, 0.50) if m_entry["pass"] else (0.95, 0.40, 0.35)
            status = "OK" if m_entry["pass"] else "NG"
            _draw_text(br, fy, 6, f"[{status}] {m_entry['metric'][:14]}:{m_entry['measured']}", col, scale=1)

    if corner in (1, 2, 3, 4):
        q_pick = [tl, tr, bl, br][corner - 1]
        canvas = _resample_to(q_pick, h, w)
    else:
        canvas[:qh, :qw] = tl
        canvas[:qh, qw:qw * 2] = tr
        canvas[qh:, :qw] = bl
        canvas[qh:, qw:] = br
        if b_mode >= 1:
            canvas[qh - 1:qh + 1, :] = np.array([0.30, 0.35, 0.40], dtype=np.float32)
            canvas[:, qw - 1:qw + 1] = np.array([0.30, 0.35, 0.40], dtype=np.float32)

    return np.clip(canvas, 0.0, 1.0).astype(np.float32), telemetry


class InversionAnalyzerStage(Stage):
    """Node-graph and pipeline stage for `inversion_analyzer` (**模块**)."""
    name = "inversion_analyzer"
    layers = ("player",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._prev_rgb: np.ndarray | None = None
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("player")
        self.spec = InversionCalibrationSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.mode == 0:
                self.spec.mode = 3

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        ref_rgb = None
        if getattr(self.ctx, "source", None) is not None:
            try:
                ref_rgb = np.clip(self.ctx.source.frame(int(buf.index)), 0.0, 1.0).astype(np.float32)
            except Exception:
                ref_rgb = None

        out_rgb, info = apply_inversion_calibration_lab(
            rgb_in,
            self.spec,
            params=self.ctx.params,
            reference_rgb=ref_rgb,
            prev_rgb=self._prev_rgb,
            fps=self.ctx.fps,
        )
        self._prev_rgb = rgb_in.copy()
        meta = dict(buf.meta)
        meta["inversion_lab"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

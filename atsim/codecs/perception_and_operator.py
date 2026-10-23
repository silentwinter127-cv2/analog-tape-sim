"""
人类视觉感知与机房操作者行为闭环系统 (`perception_and_operator`)
==========================================================================

实现《完整的视听媒介生态模拟器》第五节（方向 A+C：设备输出 ≠ 观众体验与有人值守生产）：

1. **观众视觉感知心理物理模型 (`compute_viewer_perception`)**：
   - **视角空间频率转换 ($\text{cycles per degree, cpd}$)**：
     给定观看距离与画面高度比 $D_{\text{view}} = d / H$（如近距检视 `1.5×H` vs 客厅沙发观看 `6.0×H`），垂直视角为 $\theta_v = 2 \arctan(1 / (2 D_{\text{view}}))$（度），像素空间频率 $f_{\text{px}}$ 映射为视网膜空间频率 $f_{\text{cpd}} = f_{\text{px}} \cdot H / \theta_v$；
   - **Mannos–Sakrison / Barten 人眼对比敏感度函数 $\text{CSF}(f_{\text{cpd}}, L_{\text{nit}}, E_{\text{lux}})$**：
     $$A(f) = 2.6 \,(0.0192 + 0.114 f)\,\exp\!\left(-(0.114 f)^{1.1}\right)$$
     峰值位于 `3.5–5.0 cpd`；在远距离观看 (`6–8×H`) 时，CRT 扫描线、荫罩三色点阵与 VHS 高频雪花落入 `>22 cpd` 的人眼高频截止区而被视觉系统自然融合成柔和细腻的画面；而在近距离趴屏 (`1.5×H`) 时，这些结构正好落入 `4–8 cpd` 的峰值敏感带，产生强烈的「颗粒/网格可见度 (JND)」；
   - **Ferry–Porter 临界闪烁融合频率 ($\text{CFF}$) 与主观评分**：
     根据屏幕亮度 `percept_display_nits` 与环境照度 `percept_ambient_lux` 计算 $\text{CFF} = 12.5 \log_{10}(L_{\text{eff}}) + 18.0\text{ Hz}$，输出 **物理实测误差 (`physical_rmse`)** vs **人眼可察觉误差 (`perceived_jnd_mean`)**、**主观清晰度 (`subjective_sharpness_mos`)**、**年代真实感 (`nostalgia_authenticity_mos`)** 与 **视觉舒适度 (`visual_comfort_mos`)**。

2. **机房操作者认知与操控行为闭环 (`simulate_human_operator_loop`)**：
   - 模拟值班工程师盯着波形监视器与预监窗，对磁带循迹误差 (Tracking)、副载波色相偏移与黑电平漂移进行手动闭环修正；
   - **技能水平 (`operator_skill`) 与时间压力/疲劳度 (`operator_stress`)**：
     - 高技能、低压力下：工程师平滑收敛并将残余跟踪/色相误差压低 `75%+`；
     - 高压力/疲劳 (`operator_stress > 0.6`) 下：认知反应延迟 $\tau_r$ 拉长，手动旋钮阻尼比 $\zeta$ 下降，引发经典的 **二阶欠阻尼过调震荡 (Overshoot Hunting — 旋钮拧过头再往回拧的周期摆动)**，并在极端压力下触发 **误切事故 (`Mis-Cut to Unready Bus`)**。

3. **`2×2` 人类视觉感知与机房操作者闭环实验室 (`percept_mode = 4`)**：
   - 同屏对比 `1. 设备物理输出 vs 观看距离 CSF 视觉感知融合画面`、`2. 人眼 JND (Just-Noticeable Difference) 可察觉伪影热力图`、`3. 机房操作者手动 Tracking/CCU 闭环修正与过调震荡画面`、`4. 心理物理 CSF 曲线与操作者控制轨迹仪表盘`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass
from ..stages.display import _resample_to
from .multiview_wall import _draw_bitmap_text


def draw_text(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    text: str,
    color: tuple[float, float, float],
    scale: int = 1,
) -> None:
    _draw_bitmap_text(canvas, str(text), int(x0), int(y0), color, scale=int(scale))


@dataclass
class PerceptionOperatorSpec:
    enabled: bool = False
    mode: int = 0
    viewing_dist_h: float = 3.5
    display_nits: float = 120.0
    ambient_lux: float = 45.0
    operator_skill: float = 0.78
    operator_stress: float = 0.35
    quad_corner: int = 0
    show_border: int = 2
    show_text: int = 0

    def is_active(self) -> bool:
        return bool(self.enabled and self.mode > 0)

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "PerceptionOperatorSpec":
        en = float(cfg.get("percept_enabled", 0.0)) > 0.5
        md = int(round(float(cfg.get("percept_mode", 0.0))))
        if md > 0:
            en = True
        elif en and md == 0:
            md = 4
        return cls(
            enabled=en,
            mode=int(np.clip(md, 0, 4)),
            viewing_dist_h=float(np.clip(cfg.get("percept_viewing_dist_h", 3.5), 1.0, 10.0)),
            display_nits=float(np.clip(cfg.get("percept_display_nits", 120.0), 20.0, 600.0)),
            ambient_lux=float(np.clip(cfg.get("percept_ambient_lux", 45.0), 1.0, 1000.0)),
            operator_skill=float(np.clip(cfg.get("operator_skill", 0.78), 0.1, 1.0)),
            operator_stress=float(np.clip(cfg.get("operator_stress", 0.35), 0.0, 1.0)),
            quad_corner=int(np.clip(round(float(cfg.get("percept_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(cfg.get("percept_show_border", 2.0))), 0, 2)),
            show_text=int(np.clip(round(float(cfg.get("percept_show_text", 0.0))), 0, 2)),
        )


def mannos_sakrison_csf(f_cpd: np.ndarray | float, lum_nits: float = 120.0, ambient_lux: float = 45.0) -> np.ndarray | float:
    """
    Mannos-Sakrison / Barten human Contrast Sensitivity Function CSF(f_cpd),
    normalized so peak sensitivity near 4 cpd is ~1.0 under optimal luminance.
    """
    f = np.maximum(0.1, np.asarray(f_cpd, dtype=np.float32))
    # Luminance adaptation shift: higher luminance shifts peak slightly higher and boosts sensitivity
    lum_factor = float(np.clip(np.log10(max(10.0, lum_nits)) / np.log10(120.0), 0.55, 1.25))
    # Ambient glare reduces dark contrast sensitivity via Weber veil
    glare_loss = float(1.0 / (1.0 + 0.0008 * max(0.0, ambient_lux - 30.0)))
    csf = 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))
    return csf * lum_factor * glare_loss


def compute_viewer_perception(
    rgb: np.ndarray,
    spec: PerceptionOperatorSpec,
    reference_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Model what the human visual system actually perceives at viewing distance `viewing_dist_h`
    and compute the JND (Just-Noticeable Difference) heatmap.
    """
    img = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = img.shape[:2]
    dist_h = float(np.clip(spec.viewing_dist_h, 1.0, 10.0))
    # Vertical visual angle subtended by screen height (in degrees)
    theta_deg = float(np.degrees(2.0 * np.arctan(1.0 / (2.0 * dist_h))))
    raster_lines = max(480.0, float(h))
    px_per_deg = raster_lines / max(0.5, theta_deg)
    nyquist_cpd = 0.5 * px_per_deg

    # Effective human eye optical + neural low-pass sigma in pixels:
    # At close distance (1.2xH), px_per_deg is small (~12 lines/deg) -> eye resolves every scanline (sigma_px ~ 0.15)
    # At far distance (7.0xH), px_per_deg is large (~58 lines/deg) -> eye fuses fine scanlines/noise above 18 cpd (sigma_px ~ 2.2)
    eye_cutoff_cpd = 16.0 * float(np.clip(np.log10(max(15.0, spec.display_nits)) / 2.0, 0.7, 1.2))
    sigma_eye_480 = float(np.clip(px_per_deg / max(4.0, eye_cutoff_cpd * 1.15) - 0.55, 0.05, 4.5))
    sigma_eye_px = float(np.clip(sigma_eye_480 * max(0.45, h / 480.0), 0.05, 4.5))

    # Ambient room light screen reflection veil (Weber black lift)
    veil = float(np.clip(0.00018 * spec.ambient_lux, 0.0, 0.14))
    img_veiled = np.clip(img * (1.0 - 0.5 * veil) + veil, 0.0, 1.0)

    if sigma_eye_px > 0.18:
        perceived_rgb = lowpass(lowpass(img_veiled, sigma_eye_px, axis=1), sigma_eye_px * 0.85, axis=0)
    else:
        perceived_rgb = img_veiled.copy()

    # Compute local high-frequency artifact energy (scanlines / shadow-mask / dot crawl) at ~0.35 cycles/line
    artifact_freq_cpd = 0.35 * px_per_deg
    csf_at_artifact = float(mannos_sakrison_csf(artifact_freq_cpd, spec.display_nits, spec.ambient_lux))

    ref = np.clip(np.asarray(reference_rgb, dtype=np.float32), 0.0, 1.0) if reference_rgb is not None else lowpass(img, 1.8)
    raw_diff = np.mean(np.abs(img - ref), axis=-1)
    physical_rmse = float(np.sqrt(np.mean((img - ref) ** 2)))

    # Perceived JND map weights raw high-frequency error by CSF(artifact_freq_cpd)
    perceived_diff = np.mean(np.abs(perceived_rgb - ref), axis=-1) * (0.35 + 0.85 * csf_at_artifact)
    jnd_mean = float(np.mean(perceived_diff) * 18.0)

    # Ferry-Porter critical flicker fusion frequency (Hz)
    cff_hz = float(12.5 * np.log10(max(5.0, spec.display_nits)) + 18.0)
    flicker_visibility = float(np.clip((cff_hz - 50.0) / 25.0, 0.0, 1.0))

    # Subjective Mean Opinion Scores (1.0 .. 5.0)
    sharpness_mos = float(np.clip(4.6 - 6.5 * float(np.mean(np.abs(perceived_rgb - ref))) - 0.12 * max(0.0, sigma_eye_px - 1.2), 1.2, 4.95))
    comfort_mos = float(np.clip(4.8 - 0.55 * jnd_mean - 0.65 * flicker_visibility - 1.5 * veil, 1.1, 4.95))
    nostalgia_mos = float(np.clip(3.2 + 1.4 * np.exp(-((dist_h - 3.8) ** 2) / 6.0) - 0.4 * abs(jnd_mean - 0.8), 1.5, 4.98))

    # Build false-color JND heatmap (Blue -> Cyan -> Yellow -> Red)
    j_norm = np.clip(perceived_diff * 4.5, 0.0, 1.0)
    jnd_heatmap = np.stack(
        [
            np.clip(2.0 * j_norm - 0.3, 0.0, 1.0),
            np.clip(1.8 * j_norm * (1.0 - 0.6 * j_norm), 0.0, 1.0),
            np.clip(0.85 * (1.0 - j_norm) + 0.15, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)

    telemetry = {
        "viewing_dist_h": round(dist_h, 2),
        "visual_angle_deg": round(theta_deg, 2),
        "px_per_deg": round(px_per_deg, 2),
        "nyquist_cpd": round(nyquist_cpd, 2),
        "artifact_freq_cpd": round(artifact_freq_cpd, 2),
        "csf_at_artifact": round(csf_at_artifact, 4),
        "sigma_eye_px": round(sigma_eye_px, 3),
        "cff_hz": round(cff_hz, 2),
        "physical_rmse": round(physical_rmse, 4),
        "perceived_jnd_mean": round(jnd_mean, 4),
        "subjective_sharpness_mos": round(sharpness_mos, 2),
        "visual_comfort_mos": round(comfort_mos, 2),
        "nostalgia_authenticity_mos": round(nostalgia_mos, 2),
    }
    return perceived_rgb.astype(np.float32), jnd_heatmap, telemetry


def simulate_human_operator_loop(
    rgb: np.ndarray,
    spec: PerceptionOperatorSpec,
    frame_index: int = 0,
    fps: float = 29.97,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Simulate a human broadcast/VTR operator monitoring scopes and adjusting Tracking,
    CCU white balance, and emergency bypass under skill and stress constraints.
    """
    skill = float(np.clip(spec.operator_skill, 0.1, 1.0))
    stress = float(np.clip(spec.operator_stress, 0.0, 1.0))
    t_sec = float(frame_index) / max(1.0, float(fps))

    # Reaction delay tau_r (seconds) and second-order damping ratio zeta
    reaction_delay_s = float(0.25 + 1.15 * (1.0 - skill) + 1.40 * (stress ** 1.4))
    damping_zeta = float(np.clip(0.92 * skill - 0.68 * stress, 0.08, 0.95))
    omega_n = float(2.2 + 2.8 * stress)

    # Initial plant disturbance (tracking error + chroma phase offset at t=0)
    # Operator begins correcting after `reaction_delay_s`
    dt = max(0.0, t_sec - reaction_delay_s * 0.25)
    env_decay = float(np.exp(-damping_zeta * omega_n * dt))
    # Under high stress (low damping_zeta), operator overshoots and hunts back and forth!
    hunting_osc = float(np.cos(omega_n * np.sqrt(max(0.04, 1.0 - damping_zeta ** 2)) * dt))
    steady_residual = float(0.28 * (1.0 - skill) + 0.18 * stress)
    tracking_err_Knob = float(env_decay * hunting_osc * (0.45 + 0.65 * stress) + steady_residual * np.sin(t_sec * 1.9))
    overshoot_peak = float(np.exp(-np.pi * damping_zeta / np.sqrt(max(0.04, 1.0 - damping_zeta ** 2))))

    # Mis-cut accident under high stress & low skill
    miscut_active = bool(stress >= 0.72 and skill <= 0.55 and (int(frame_index) % 6 == 2))

    out = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0).copy()
    h, w = out.shape[:2]

    # Apply residual tracking wave / overshoot hunting to lower rows
    if abs(tracking_err_Knob) > 0.01:
        rows = np.arange(h, dtype=np.float32)
        bot_w = np.clip((rows - h * 0.62) / max(1.0, h * 0.38), 0.0, 1.0) ** 1.5
        shifts = bot_w * tracking_err_Knob * 14.0 * np.sin(rows * 0.25 + t_sec * 6.0)
        for ch in range(3):
            out[..., ch] = horizontal_shift(out[..., ch], shifts)

    # CCU color balance trim vs overshoot cast
    wb_trim_err = tracking_err_Knob * 0.18
    out[..., 0] = np.clip(out[..., 0] * (1.0 + wb_trim_err), 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] * (1.0 - wb_trim_err), 0.0, 1.0)

    if miscut_active:
        # Operator accidentally punches an un-synced auxiliary bus for a split second
        out = np.roll(out[:, ::-1, :], shift=h // 5, axis=0)
        out[:8, :] = np.array([0.95, 0.20, 0.20], dtype=np.float32)

    op_info = {
        "operator_skill": round(skill, 3),
        "operator_stress": round(stress, 3),
        "reaction_delay_s": round(reaction_delay_s, 3),
        "damping_zeta": round(damping_zeta, 3),
        "overshoot_ratio": round(overshoot_peak, 4),
        "instantaneous_knob_error": round(abs(tracking_err_Knob), 4),
        "steady_residual_error": round(steady_residual, 4),
        "miscut_accident": miscut_active,
        "operator_state": "PANIC_OVERSHOOT_HUNTING" if stress > 0.65 else ("SKILLFUL_LOCKED" if skill > 0.7 else "MANUAL_TRIMMING"),
    }
    return out, op_info


def _render_dashboard_tile(
    h: int,
    w: int,
    percept_info: dict[str, Any],
    op_info: dict[str, Any],
    show_text: int = 2,
) -> np.ndarray:
    """Render Corner 4: CSF curve & Operator closed-loop response dashboard."""
    panel = np.zeros((h, w, 3), dtype=np.float32)
    panel[:, :] = np.array([0.07, 0.09, 0.12], dtype=np.float32)

    # Draw CSF curve in upper half
    xs = np.linspace(0.5, 32.0, w - 16, dtype=np.float32)
    csf_vals = mannos_sakrison_csf(xs, percept_info.get("display_nits", 120.0))
    csf_vals = np.clip(csf_vals / max(1e-5, float( np.max(csf_vals))), 0.0, 1.0)
    y_top0, y_top1 = int(h * 0.18), int(h * 0.52)
    span_y = max(8, y_top1 - y_top0)
    for ix, val in enumerate(csf_vals):
        px = 8 + ix
        py = int(round(y_top1 - val * (span_y - 4)))
        if 0 <= py < h and 0 <= px < w:
            panel[max(0, py - 1) : min(h, py + 2), px] = np.array([0.35, 0.92, 0.78], dtype=np.float32)

    # Mark current viewing-distance artifact cpd on the CSF curve
    art_cpd = float(percept_info.get("artifact_freq_cpd", 6.0))
    art_x = int(np.clip(8 + (art_cpd / 32.0) * (w - 16), 8, w - 8))
    panel[y_top0:y_top1, max(0, art_x - 1) : min(w, art_x + 1)] = np.array([0.96, 0.78, 0.28], dtype=np.float32)

    # Draw Operator step response & overshoot trajectory in lower half
    y_bot0, y_bot1 = int(h * 0.60), int(h * 0.92)
    y_mid = (y_bot0 + y_bot1) // 2
    panel[y_mid : y_mid + 1, 8 : w - 8] = np.array([0.25, 0.30, 0.36], dtype=np.float32)
    zeta = float(op_info.get("damping_zeta", 0.5))
    t_grid = np.linspace(0.0, 4.0, w - 16, dtype=np.float32)
    resp = np.exp(-zeta * 3.0 * t_grid) * np.cos(3.2 * np.sqrt(max(0.04, 1.0 - zeta * zeta)) * t_grid)
    for ix, rv in enumerate(resp):
        px = 8 + ix
        py = int(np.clip(y_mid - rv * ((y_bot1 - y_bot0) * 0.42), y_bot0, y_bot1 - 1))
        panel[max(0, py - 1) : min(h, py + 2), px] = np.array([0.95, 0.52, 0.38], dtype=np.float32)

    if show_text >= 1:
        draw_text(panel, 6, 4, f"4.CSF & OPERATOR (D={percept_info['viewing_dist_h']}xH)", (0.95, 0.88, 0.42), scale=1)
        if show_text >= 2 and h >= 90:
            draw_text(
                panel,
                6,
                y_top1 + 2,
                f"JND:{percept_info['perceived_jnd_mean']:.2f} MOS:{percept_info['subjective_sharpness_mos']:.1f} OVR:{int(op_info['overshoot_ratio']*100)}%",
                (0.78, 0.92, 0.85),
                scale=1,
            )
    return panel


def apply_perception_and_operator(
    rgb: np.ndarray,
    spec: PerceptionOperatorSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    reference_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Option T Human Visual Perception & Operator Closed-Loop System."""
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    perceived_rgb, jnd_heatmap, percept_info = compute_viewer_perception(src, spec, reference_rgb=reference_rgb)
    op_rgb, op_info = simulate_human_operator_loop(src, spec, frame_index=frame_index, fps=fps)

    report = {
        "active": bool(spec.enabled),
        "mode": int(spec.mode),
        "percept_quad_corner": int(spec.quad_corner),
        "percept_show_border": int(spec.show_border),
        "percept_show_text": int(spec.show_text),
        "perception": percept_info,
        "operator": op_info,
    }
    if not spec.enabled or spec.mode <= 0:
        return src, report

    h, w = src.shape[:2]
    dash_rgb = _render_dashboard_tile(h, w, percept_info, op_info, show_text=max(1, spec.show_text))

    if spec.mode == 1:
        out = perceived_rgb
    elif spec.mode == 2:
        out = op_rgb
    elif spec.mode == 3:
        # High-stress operator mode
        stress_spec = PerceptionOperatorSpec(
            enabled=True,
            mode=3,
            viewing_dist_h=spec.viewing_dist_h,
            display_nits=spec.display_nits,
            ambient_lux=spec.ambient_lux,
            operator_skill=min(spec.operator_skill, 0.45),
            operator_stress=max(spec.operator_stress, 0.82),
        )
        out, op_info_stress = simulate_human_operator_loop(src, stress_spec, frame_index=frame_index, fps=fps)
        report["operator"] = op_info_stress
    else:
        c_sel = int(spec.quad_corner)
        if c_sel in (1, 2, 3, 4):
            out = {1: perceived_rgb, 2: jnd_heatmap, 3: op_rgb, 4: dash_rgb}[c_sel].copy()
        else:
            qh, qw = h // 2, w // 2
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            quads = [
                (0, qh, 0, qw, perceived_rgb, f"1.CSF PERCEIVED ({spec.viewing_dist_h:.1f}xH)", (0.45, 0.95, 0.82)),
                (0, qh, qw, w, jnd_heatmap, f"2.JND HEATMAP ({percept_info['perceived_jnd_mean']:.2f})", (0.95, 0.82, 0.38)),
                (qh, h, 0, qw, op_rgb, f"3.OPERATOR ({op_info['operator_state'][:10]})", (0.52, 0.88, 0.96)),
                (qh, h, qw, w, dash_rgb, "4.CSF & OVERSHOOT CURVE", (0.95, 0.62, 0.45)),
            ]
            for y0, y1, x0, x1, img_q, lbl, col in quads:
                tile = _resample_to(img_q, y1 - y0, x1 - x0)
                if spec.show_text >= 1:
                    tile[:16, :] *= 0.22
                    draw_text(tile, 4, 4, lbl, col, scale=1)
                canvas[y0:y1, x0:x1] = tile
            if spec.show_border >= 1:
                canvas[qh - 1 : qh + 1, :] = np.array([0.20, 0.24, 0.30], dtype=np.float32)
                canvas[:, qw - 1 : qw + 1] = np.array([0.20, 0.24, 0.30], dtype=np.float32)
            out = canvas

    return np.clip(out, 0.0, 1.0).astype(np.float32), report


class HumanOperatorStage(Stage):
    """Node-graph stage for `human_operator` (**模块**)."""
    name = "human_operator"
    layers = ("display",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("display")
        self.spec = PerceptionOperatorSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.mode <= 0:
                self.spec.mode = 4

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        ref_rgb = buf.meta.get("tap_camera_rgb")
        out_rgb, info = apply_perception_and_operator(
            rgb_in,
            self.spec,
            frame_index=int(buf.index),
            fps=float(self.ctx.fps),
            reference_rgb=ref_rgb,
        )
        meta = dict(buf.meta)
        meta["perception_operator"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

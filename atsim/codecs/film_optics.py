"""
独立胶片光学与抗晕光物理模块 (`atsim/codecs/film_optics.py`)
=============================================================================

将「胶片镜头光学、片基-乳剂层间光传播、抗晕光背层 (Remjet / AHU) 与散景光晕」从光化学显影暗房模块中
完全独立拆出，保持简洁清晰的物理边界：

1. **物理区分三类高光扩散机制（README 第四节）**：
   - **Halation（胶片晕光）**：强光穿透感蓝/感绿/感红乳剂层后，在醋酸纤维素/聚酯片基或背层界面发生反射与散射，
     从背面二次曝光感红层（及部分感绿层），在点光源周围形成特征性的红橙色二次曝光光环。
   - **Lens Flare / Veiling Glare（镜头炫光与杂散光）**：发生在摄影机镜头光学系统内部（如无镀膜新月透镜前向米氏散射、
     逆光琥珀金阳炎洪流 `bloom_veil_mode=1`、高调奶雾眩光 `bloom_veil_mode=2`、柔焦镜 `soft_focus_mist`）。
   - **Blooming（传感器电荷溢出）**：发生在电子摄像管或 CCD 传感器势井饱和时（归属 `camera` 模块，与胶片晕光严格区分）。

2. **胶片层透射/反射与抗晕光结构 (`remjet_backing`, `aged_backing_dye`)**：
   - **Remjet 碳黑抗晕光背层 (`remjet_backing = 0..1`)**：Kodak Vision3 500T 等电影负片背面涂有碳黑 Remjet 层，
     可吸收约 88% 穿透乳剂的强光以抑制晕光；若在冲洗前去除 Remjet（如 CineStill 工艺，`remjet_backing = 0`），
     片基界面全反射将产生显著的红橙色点光源光晕环。
   - **醋酸片基老化后背层抗晕光染料重显 (`aged_backing_dye = 0..1`, FilmCare.org 案例)**：
     部分胶片采用冲洗时脱色的抗晕光底层/背层染料（AHU）；当醋酸片基老化产酸（醋酸综合征）时，
     原本无色的背层染料分子结构逆转并重新显色，对透射光产生洋红/粉红或青绿色偏贡献。

3. **摄影机与 LOMO 几何/衍射光学系统**：
   - 塑胶新月透镜 $\\cos^4\\theta$ 暗角 (`lomo_vignette`) 与匹兹伐场曲边缘像散色散 (`lomo_edge_smear`)
   - 匹兹伐切向旋焦猫眼散景 (`swirl_amount`)
   - 沙姆定律移轴/微距焦平面浅景深 (`tilt_shift_dof`, `focus_plane_y`)
   - 异形光圈与球差过校正肥皂泡散景核 (`bokeh_shape = 0..6`, `bokeh_amount`)
   - 机壳海绵老化漏光 (`light_leak_amount`, `light_leak_pattern`) 与未过片双重曝光 (`double_exposure`)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, lowpass

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class FilmOpticsSpec:
    """独立胶片光学模块参数规范（涵盖镜头光学、片基散射晕光、Remjet 背层与老化背层染料重显）。"""
    enabled: bool = True
    # 抗晕光结构与片基界面光学传播 (README Section IV)
    halation: float = 0.35          # 基础片基反射晕光系数 (0..1.5)
    remjet_backing: float = 0.0     # Remjet 碳黑抗晕光背层吸收率 (0=无背层/强红晕, 1=完整ECN-2碳黑背层抑制晕光)
    aged_backing_dye: float = 0.0   # FilmCare.org 醋酸片基老化引发背层抗晕光染料重新显色 (0..1)
    # 镜头杂散光、逆光阳炎与柔焦雾化
    bloom_veil_mode: int = 0        # 0=标准背散射, 1=逆光琥珀金阳炎洪流, 2=高调奶油柔焦眩光, 3=点光源双层色散光冕, 4=对角冷暖双色温光学纱幕
    bloom_veil_amount: float = 0.0  # 光学眩光/阳炎/纱幕强度 (0..1.5)
    soft_focus_mist: float = 0.0    # Duto / Pro-Mist 柔焦镜漫射雾化 (0..1.2)
    dual_tone_split: float = 0.0    # 对角空间/明暗双色温光学分离 (0..1.2)
    # 焦平面、旋焦、场曲与异形散景
    tilt_shift_dof: float = 0.0     # 移轴/微距焦平面浅景深强度 (0..1.2)
    focus_plane_y: float = 0.65     # 合焦平面纵向中心位置 [0.1, 0.9]
    swirl_amount: float = 0.0       # 匹兹伐切向旋焦猫眼散景强度 (0..1.2)
    lomo_vignette: float = 0.0      # 新月透镜 cos^4(theta) + 镜筒暗角 (0..1.5)
    lomo_edge_smear: float = 0.0    # 离轴场曲与径向色散边缘拖尾 (0..1.5)
    bokeh_shape: int = 0            # 0=圆形, 1=桃心, 2=六芒星, 3=折返环, 4=菱形, 5=Trioplan肥皂泡亮环, 6=六边形光斑
    bokeh_amount: float = 0.0       # 异形散景光斑强度 (0..1.5)
    # 暗盒漏光与未过片重曝
    light_leak_amount: float = 0.0  # 机壳漏光强度 (0..1.5)
    light_leak_pattern: int = 0     # 0=自动, 1=后盖红橙, 2=粉紫斜光束, 3=右上暖金, 4=宝丽来底边药液漏光
    double_exposure: float = 0.0    # 线性曝光域双重曝光叠加 (0..1.0)

    @classmethod
    def from_params(cls, cfg: dict[str, Any], fallback_cam: dict[str, Any] | None = None) -> "FilmOpticsSpec":
        c = dict(fallback_cam or {})
        c.update(cfg or {})

        def _g(k_film: str, k_cam: str, default: float) -> float:
            if k_film in (cfg or {}):
                return float(cfg[k_film])
            if k_cam in c:
                return float(c[k_cam])
            return float(default)

        return cls(
            enabled=True,
            halation=float(np.clip(_g("halation", "film_halation", 0.35), 0.0, 1.5)),
            remjet_backing=float(np.clip(_g("remjet_backing", "film_remjet_backing", 0.0), 0.0, 1.0)),
            aged_backing_dye=float(np.clip(_g("aged_backing_dye", "film_aged_backing_dye", 0.0), 0.0, 1.0)),
            bloom_veil_mode=int(np.clip(round(_g("bloom_veil_mode", "film_bloom_veil_mode", 0.0)), 0, 5)),
            bloom_veil_amount=float(np.clip(_g("bloom_veil_amount", "film_bloom_veil_amount", 0.0), 0.0, 1.5)),
            soft_focus_mist=float(np.clip(_g("soft_focus_mist", "film_soft_focus_mist", 0.0), 0.0, 1.2)),
            dual_tone_split=float(np.clip(_g("dual_tone_split", "film_dual_tone_split", 0.0), 0.0, 1.2)),
            tilt_shift_dof=float(np.clip(_g("tilt_shift_dof", "film_tilt_shift_dof", 0.0), 0.0, 1.2)),
            focus_plane_y=float(np.clip(_g("focus_plane_y", "film_focus_plane_y", 0.65), 0.1, 0.9)),
            swirl_amount=float(np.clip(_g("swirl_amount", "film_swirl_amount", 0.0), 0.0, 1.2)),
            lomo_vignette=float(np.clip(_g("lomo_vignette", "film_lomo_vignette", 0.0), 0.0, 1.5)),
            lomo_edge_smear=float(np.clip(_g("lomo_edge_smear", "film_lomo_edge_smear", 0.0), 0.0, 1.5)),
            bokeh_shape=int(np.clip(round(_g("bokeh_shape", "film_bokeh_shape", 0.0)), 0, 6)),
            bokeh_amount=float(np.clip(_g("bokeh_amount", "film_bokeh_amount", 0.0), 0.0, 1.5)),
            light_leak_amount=float(np.clip(_g("light_leak_amount", "film_light_leak_amount", 0.0), 0.0, 1.5)),
            light_leak_pattern=int(np.clip(round(_g("light_leak_pattern", "film_light_leak_pattern", 0.0)), 0, 4)),
            double_exposure=float(np.clip(_g("double_exposure", "film_double_exposure", 0.0), 0.0, 1.0)),
        )


def build_shaped_bokeh_kernel(shape_id: int, radius_px: int = 9) -> np.ndarray:
    """
    构造归一化的 2D 物理光圈孔径与球差散景 PSF 核 (`bokeh_shape = 0..6`)：
      * `0 = Circle` (标准圆形光圈)
      * `1 = Heart` (桃心形散景插片)
      * `2 = Star` (六芒星衍射光圈)
      * `3 = Annular Donut` (折返镜头环形散景)
      * `4 = Holga Diamond` (双叶片菱形光圈)
      * `5 = Soap-Bubble Ring` (Trioplan 球差过校正肥皂泡亮边圆环散景)
      * `6 = Hexagonal Fairy-Light` (六叶片正六边形光圈散景)
    """
    r = max(3, int(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    nx = xx / float(r)
    ny = yy / float(r)

    if shape_id == 1:
        u = nx * 1.20
        v = -ny * 1.22 + 0.15
        f_heart = (u * u + v * v - 0.82) ** 3 - (u * u) * (v ** 3)
        kern = np.clip(0.18 - f_heart, 0.0, 1.0) ** 0.55
    elif shape_id == 2:
        rad = np.hypot(nx, ny)
        ang = np.arctan2(ny, nx)
        star_r = 0.52 + 0.42 * np.cos(6.0 * ang)
        kern = np.clip((star_r - rad) * 4.0, 0.0, 1.0)
    elif shape_id == 3:
        rad = np.hypot(nx, ny)
        kern = np.exp(-0.5 * ((rad - 0.68) / 0.18) ** 2)
    elif shape_id == 4:
        diamond_d = np.abs(nx) * 0.85 + np.abs(ny) * 1.10
        kern = np.clip((0.92 - diamond_d) * 3.5, 0.0, 1.0)
    elif shape_id == 5:
        rad = np.hypot(nx, ny)
        disc_in = np.clip((0.96 - rad) * 5.0, 0.0, 1.0)
        bright_rim = np.exp(-0.5 * ((rad - 0.86) / 0.085) ** 2)
        kern = disc_in * (0.16 + 1.15 * bright_rim)
    elif shape_id == 6:
        rad = np.hypot(nx, ny)
        ang = np.arctan2(ny, nx)
        sector = np.pi / 3.0
        rel_ang = np.mod(ang + sector * 0.5, sector) - sector * 0.5
        hex_r = 0.88 * np.cos(sector * 0.5) / np.maximum(np.cos(rel_ang), 1e-3)
        kern = np.clip((hex_r - rad) * 4.5, 0.0, 1.0) * (0.78 + 0.22 * np.exp(-0.5 * ((rad - 0.65) / 0.22) ** 2))
    else:
        rad = np.hypot(nx, ny)
        kern = np.clip((0.95 - rad) * 3.5, 0.0, 1.0)

    total = float(kern.sum())
    if total > 1e-6:
        kern /= total
    return kern.astype(np.float32)


def apply_lomo_optics_and_leaks(
    rgb: np.ndarray,
    spec: FilmOpticsSpec | Any,
    frame_index: int = 0,
    seed: int = 20240101,
    secondary_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    在到达胶片乳剂前的线性曝光域中模拟摄影机与 LOMO 镜头光学过程：
      1. 未过片双重曝光 (`double_exposure`)
      2. 移轴/微距焦平面浅景深 (`tilt_shift_dof`, `focus_plane_y`)
      3. 匹兹伐切向旋焦与猫眼像散 (`swirl_amount`)
      4. 离轴场曲与边缘径向色散拖尾 (`lomo_edge_smear`)
      5. 异形光圈与球差过校正肥皂泡散景 (`bokeh_shape = 0..6`, `bokeh_amount`)
      6. 塑胶新月透镜 `cos^4(theta)` 暗角 (`lomo_vignette`)
      7. 机壳海绵封条老化漏光 (`light_leak_amount`, `light_leak_pattern`)
    """
    out = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.5).copy()
    h, w = out.shape[:2]
    scale_w = max(0.5, w / 720.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    rad = np.sqrt(r2)

    # 1. 双重曝光 (`double_exposure`)
    dbl = float(getattr(spec, "double_exposure", 0.0))
    if dbl > 1e-4:
        if secondary_rgb is not None and secondary_rgb.shape == out.shape:
            ghost_exp = np.clip(np.asarray(secondary_rgb, dtype=np.float32), 0.0, 1.5)
        else:
            ghost_exp = np.roll(out[:, ::-1, :], shift=max(8, int(w * 0.14)), axis=1)
        out = out * (1.0 - 0.22 * dbl) + ghost_exp * (0.68 * dbl)

    # 2. 移轴/微距焦平面浅景深 (`tilt_shift_dof`, `focus_plane_y`)
    ts_dof = float(getattr(spec, "tilt_shift_dof", 0.0))
    f_plane_y = float(getattr(spec, "focus_plane_y", 0.65))
    dof_blur_ratio = 1.0
    if ts_dof > 1e-4:
        y_frac = yy / max(1.0, float(h - 1))
        plane_dist = np.abs(y_frac - f_plane_y + 0.05 * nx)
        defocus_w = np.clip((plane_dist - 0.08) / 0.42, 0.0, 1.0) ** 1.35
        defocus_w = np.clip(defocus_w * min(1.2, ts_dof * 1.15), 0.0, 0.96)[..., None]

        blur_mid = np.stack([gaussian_filter(out[..., c], sigma=max(1.5, 4.2 * ts_dof * scale_w)) for c in range(3)], axis=-1)
        blur_far = np.stack([gaussian_filter(out[..., c], sigma=max(3.5, 10.5 * ts_dof * scale_w)) for c in range(3)], axis=-1)
        lum_ts = out @ LUMA_W
        hot_ts = np.clip((lum_ts - 0.55) / 0.35, 0.0, 1.2) ** 1.6
        orb_bloom = lowpass(hot_ts, max(3.0, 8.5 * ts_dof * scale_w))[..., None]
        far_mix = np.clip((defocus_w - 0.35) / 0.65, 0.0, 1.0)
        dof_field = blur_mid * (1.0 - far_mix) + blur_far * far_mix + (0.28 * ts_dof) * orb_bloom * defocus_w
        out = out * (1.0 - defocus_w) + dof_field * defocus_w
        dof_blur_ratio = float(1.0 + 3.5 * defocus_w.mean())

    # 3. 匹兹伐切向旋焦与猫眼像散 (`swirl_amount`)
    swirl = float(getattr(spec, "swirl_amount", 0.0))
    if swirl > 1e-4:
        tx = -ny / np.maximum(rad, 1e-4)
        ty = nx / np.maximum(rad, 1e-4)
        swirl_gate = np.clip((rad - 0.24) / 0.85, 0.0, 1.0) ** 1.45
        arc_span = (4.5 + 12.5 * swirl) * scale_w * swirl_gate
        swirl_acc = out.copy() * 1.25
        w_swirl = np.full((h, w, 1), 1.25, dtype=np.float32)
        for frac in (-1.0, -0.66, -0.33, 0.33, 0.66, 1.0):
            d_arc = arc_span * frac
            sy = np.clip(np.round(yy + ty * d_arc - 0.12 * ny * (frac * frac) * swirl).astype(np.int32), 0, h - 1)
            sx = np.clip(np.round(xx + tx * d_arc - 0.12 * nx * (frac * frac) * swirl).astype(np.int32), 0, w - 1)
            sample = out[sy, sx]
            s_lum = (sample @ LUMA_W)[..., None]
            wt = (swirl_gate[..., None] * (0.45 + 0.85 * np.clip(s_lum - 0.35, 0.0, 1.0))) * float(1.1 - 0.35 * abs(frac))
            swirl_acc += sample * wt
            w_swirl += wt
        out = swirl_acc / w_swirl

    # 4. 离轴场曲与边缘径向色散拖尾 (`lomo_edge_smear`)
    smear = float(getattr(spec, "lomo_edge_smear", 0.0))
    if smear > 1e-4:
        ux = nx / np.maximum(rad, 1e-4)
        uy = ny / np.maximum(rad, 1e-4)
        edge_w = np.clip((r2 - 0.12) / 1.15, 0.0, 1.0) ** 1.25
        reach = (3.5 + 9.0 * smear) * scale_w
        acc = out.copy()
        w_tot = np.ones((h, w, 1), dtype=np.float32)
        for step_k in (0.35, 0.70, 1.00):
            for sign in (-1.0, 1.0):
                sy = np.clip(np.round(yy + sign * uy * reach * step_k * edge_w).astype(np.int32), 0, h - 1)
                sx = np.clip(np.round(xx + sign * ux * reach * step_k * edge_w).astype(np.int32), 0, w - 1)
                wt = (0.42 * smear * edge_w * (1.15 - 0.4 * step_k))[..., None]
                acc += out[sy, sx] * wt
                w_tot += wt
        out = acc / w_tot
        r_sh = np.clip(np.round(xx - nx * 2.2 * smear * scale_w).astype(np.int32), 0, w - 1)
        b_sh = np.clip(np.round(xx + nx * 2.2 * smear * scale_w).astype(np.int32), 0, w - 1)
        row_idx = np.arange(h, dtype=np.int32)[:, None]
        out[..., 0] = out[..., 0] * (1.0 - 0.45 * edge_w) + out[row_idx, r_sh, 0] * (0.45 * edge_w)
        out[..., 2] = out[..., 2] * (1.0 - 0.45 * edge_w) + out[row_idx, b_sh, 2] * (0.45 * edge_w)

    # 5. 异形光圈与肥皂泡散景 (`bokeh_shape = 0..6`, `bokeh_amount`)
    bokeh_amt = float(getattr(spec, "bokeh_amount", 0.0))
    bokeh_sh = int(getattr(spec, "bokeh_shape", 0))
    bokeh_energy = 0.0
    if bokeh_amt > 1e-4:
        lum = out @ LUMA_W
        hot = (np.clip((lum - 0.62) / 0.32, 0.0, 1.5) ** 1.8).copy()
        rng_bk = np.random.default_rng((int(seed) * 31337 + int(frame_index) * 17 + 7) & 0xFFFFFFFF)

        if bokeh_sh == 1:
            n_pts = 9
            for p_i in range(n_pts):
                t_p = (p_i + 0.5) / n_pts
                px = int(np.clip((0.08 + 0.84 * t_p + 0.03 * rng_bk.standard_normal()) * w, 4, w - 5))
                py = int(np.clip((0.52 - 0.16 * np.sin(t_p * 2.0 * np.pi) + 0.04 * rng_bk.standard_normal()) * h, 4, h - 5))
                hot[max(0, py - 1):min(h, py + 2), max(0, px - 1):min(w, px + 2)] += 0.95 * bokeh_amt
        elif bokeh_sh == 5:
            n_bubbles = 14
            for _ in range(n_bubbles):
                bx = int(np.clip(rng_bk.uniform(0.32, 0.96) * w, 6, w - 7))
                by = int(np.clip(rng_bk.uniform(0.06, 0.62) * h, 6, h - 7))
                hot[max(0, by - 1):min(h, by + 2), max(0, bx - 1):min(w, bx + 2)] += float(rng_bk.uniform(0.75, 1.25)) * bokeh_amt
        elif bokeh_sh == 6:
            n_lights = 12
            for _ in range(n_lights):
                lx = int(np.clip(rng_bk.uniform(0.06, 0.94) * w, 6, w - 7))
                ly = int(np.clip(rng_bk.uniform(0.08, 0.58) * h, 6, h - 7))
                hot[max(0, ly - 1):min(h, ly + 2), max(0, lx - 1):min(w, lx + 2)] += float(rng_bk.uniform(0.85, 1.35)) * bokeh_amt

        if float(hot.max()) > 1e-4:
            k_rad = max(4, int(round((8.0 + 5.0 * bokeh_amt) * scale_w)))
            kern = build_shaped_bokeh_kernel(bokeh_sh if bokeh_sh > 0 else 1, radius_px=k_rad)
            bokeh_field = convolve(hot, kern, mode="nearest")
            b_max = max(float(bokeh_field.max()), 1e-4)
            bokeh_norm = np.clip(bokeh_field / b_max, 0.0, 1.2)
            if bokeh_sh == 1:
                bk_tint = np.array([1.00, 0.58, 0.78], dtype=np.float32)[None, None, :]
            elif bokeh_sh == 5:
                bk_tint = np.array([1.02, 0.84, 0.74], dtype=np.float32)[None, None, :]
            elif bokeh_sh == 6:
                bk_tint = np.array([1.04, 0.90, 0.52], dtype=np.float32)[None, None, :]
            else:
                bk_tint = np.array([1.00, 0.92, 0.78], dtype=np.float32)[None, None, :]
            out = out + (0.62 * bokeh_amt) * bokeh_norm[..., None] * bk_tint
            bokeh_energy = float(bokeh_norm.mean())

    # 6. 新月透镜 cos^4(theta) 暗角 (`lomo_vignette`)
    vig = float(getattr(spec, "lomo_vignette", 0.0))
    vig_ratio = 1.0
    if vig > 1e-4:
        cos_theta2 = 1.0 / (1.0 + 0.42 * vig * r2)
        cos4 = cos_theta2 * cos_theta2
        barrel_cut = np.clip(1.0 - 0.24 * vig * (rad ** 2.6), 0.18, 1.0)
        vig_mask = (cos4 * barrel_cut).astype(np.float32)
        out = out * vig_mask[..., None]
        vig_ratio = float(vig_mask[0, 0] / max(float(vig_mask[h // 2, w // 2]), 1e-4))

    # 7. 机壳海绵老化漏光 (`light_leak_amount`, `light_leak_pattern`)
    leak_amt = float(getattr(spec, "light_leak_amount", 0.0))
    leak_pat = int(getattr(spec, "light_leak_pattern", 0))
    if leak_amt > 1e-4:
        pat_eff = leak_pat if leak_pat > 0 else ((int(seed) + int(frame_index)) % 4 + 1)
        if pat_eff == 1:
            band1 = np.exp(-0.5 * ((nx + 0.68) / 0.22) ** 2) * (0.75 + 0.25 * np.cos(ny * 2.5))
            band2 = 0.55 * np.exp(-0.5 * ((nx - 0.74) / 0.18) ** 2)
            leak_rgb = (
                band1[..., None] * np.array([1.15, 0.36, 0.06], dtype=np.float32)
                + band2[..., None] * np.array([1.05, 0.62, 0.12], dtype=np.float32)
            )
        elif pat_eff == 2:
            diag_d = (nx * 0.55 + ny * 0.83)
            streak = np.exp(-0.5 * (diag_d / 0.28) ** 2) * (0.65 + 0.35 * np.sin(nx * 4.0))
            leak_rgb = streak[..., None] * np.array([1.05, 0.42, 0.76], dtype=np.float32)
        elif pat_eff == 3:
            d_ur2 = (nx - 0.65) ** 2 + (ny + 0.68) ** 2
            sun_veil = np.exp(-d_ur2 / 0.95)
            leak_rgb = sun_veil[..., None] * np.array([1.12, 0.86, 0.42], dtype=np.float32)
        else:
            bot_spread = np.exp(-0.5 * ((ny - 0.88) / 0.18) ** 2) * (0.6 + 0.4 * np.cos(nx * 5.0))
            top_corner = np.exp(-((nx + 0.78) ** 2 + (ny + 0.78) ** 2) / 0.35)
            leak_rgb = (bot_spread + 0.65 * top_corner)[..., None] * np.array([1.04, 0.82, 0.52], dtype=np.float32)

        out = out + (0.52 * leak_amt) * leak_rgb

    return np.clip(out, 0.0, 2.0).astype(np.float32), {
        "vignette_corner_to_center": round(vig_ratio, 4),
        "bokeh_shape": bokeh_sh,
        "bokeh_energy": round(bokeh_energy, 5),
        "tilt_shift_dof": round(ts_dof, 3),
        "dof_blur_ratio": round(dof_blur_ratio, 3),
        "swirl_amount": round(swirl, 3),
        "light_leak_amount": round(leak_amt, 3),
        "light_leak_pattern": int(getattr(spec, "light_leak_pattern", 0)),
        "double_exposure": round(dbl, 3),
    }


def apply_lomo_halation_and_veiling_glare(
    rgb: np.ndarray,
    spec: FilmOpticsSpec | Any,
    stock_prof: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    模拟胶片乳剂-片基界面晕光 (Halation)、Remjet 碳黑抗晕光背层吸收、醋酸片基老化背层染料重显
    以及镜头光学眩光/逆光阳炎/奶雾柔焦 (`halation`, `remjet_backing`, `aged_backing_dye`,
    `bloom_veil_mode`, `bloom_veil_amount`, `soft_focus_mist`, `dual_tone_split`)。
    """
    out = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 2.0).copy()
    h, w = out.shape[:2]
    scale_w = max(0.5, w / 720.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0

    hal = float(getattr(spec, "halation", 0.35))
    remjet = float(np.clip(getattr(spec, "remjet_backing", 0.0), 0.0, 1.0))
    aged_dye = float(np.clip(getattr(spec, "aged_backing_dye", 0.0), 0.0, 1.0))
    bv_mode = int(getattr(spec, "bloom_veil_mode", 0))
    bv_amt = float(getattr(spec, "bloom_veil_amount", 0.0))
    sf_mist = float(getattr(spec, "soft_focus_mist", 0.0))
    dt_split = float(getattr(spec, "dual_tone_split", 0.0))
    if bv_mode > 0 and bv_amt < 1e-4:
        bv_amt = 0.78

    # Remjet 碳黑背层吸收穿透乳剂的光子，抑制片基界面反射晕光 (README Section IV)
    remjet_transmission = 1.0 - 0.86 * remjet
    eff_hal = hal * remjet_transmission

    lum = out @ LUMA_W
    halo_energy = 0.0
    hal_tint_tuple = getattr(stock_prof, "halation_tint", (1.00, 0.38, 0.14)) if stock_prof is not None else (1.00, 0.38, 0.14)

    # 1. 片基界面反射晕光 (`halation` × `remjet_transmission`)
    if eff_hal > 1e-4:
        hot = np.clip((lum - 0.58) / 0.36, 0.0, 1.2) ** 1.4
        if float(hot.max()) > 1e-4:
            r_tight = max(1.2, 3.2 * scale_w)
            r_wide = max(3.0, 9.5 * scale_w)
            halo_field = 0.65 * lowpass(hot, r_tight) + 0.35 * lowpass(hot, r_wide)
            ht = np.array(hal_tint_tuple, dtype=np.float32)[None, None, :]
            out = out + (0.42 * eff_hal) * halo_field[..., None] * ht
            halo_energy += float(halo_field.mean()) * eff_hal

    # 1b. FilmCare.org 醋酸片基老化后抗晕光背层染料重新显色 (`aged_backing_dye`)
    if aged_dye > 1e-4:
        # 脱色抗晕光染料在酸性片基环境中恢复发色团，选择性吸收绿/黄波段，使片基呈现特征性洋红/粉紫或青蓝底色
        backing_stain = np.array(
            [1.0 + 0.08 * aged_dye, 1.0 - 0.24 * aged_dye, 1.0 + 0.05 * aged_dye],
            dtype=np.float32,
        )[None, None, :]
        out = out * backing_stain + aged_dye * np.array([0.05, -0.03, 0.04], dtype=np.float32)[None, None, :]

    # 2. 镜头杂散光、逆光琥珀阳炎、高调奶雾眩光与点光源双层色散光冕 (`bloom_veil_mode = 1..4`)
    if bv_amt > 1e-4 and bv_mode in (1, 2, 3, 4):
        if bv_mode == 1:
            d_sun1 = ((nx + 0.10) / 0.52) ** 2 + ((ny + 0.76) / 0.46) ** 2
            d_sun2 = ((nx - 0.62) / 0.30) ** 2 + ((ny + 0.70) / 0.32) ** 2
            sun_core = np.exp(-d_sun1 * 1.45) + 0.48 * np.exp(-d_sun2 * 1.8)
            mie_flood = np.exp(-d_sun1 * 0.32) + 0.35 * np.exp(-d_sun2 * 0.45)
            hot_scene = lowpass(np.clip((lum - 0.48) / 0.42, 0.0, 1.2), max(2.5, 8.0 * scale_w))
            core_rgb = np.array([1.05, 0.96, 0.38], dtype=np.float32)[None, None, :]
            flood_rgb = np.array([0.96, 0.54, 0.08], dtype=np.float32)[None, None, :]
            cocoa_shadow = np.array([0.42, 0.24, 0.09], dtype=np.float32)[None, None, :]
            lower_shade = np.clip((ny + 0.15) / 1.15, 0.0, 1.0)[..., None]
            out = out * (1.0 - 0.38 * bv_amt * lower_shade) + (out * cocoa_shadow) * (0.38 * bv_amt * lower_shade)
            out = (
                out
                + (0.62 * bv_amt) * (mie_flood + 0.45 * hot_scene)[..., None] * flood_rgb
                + (0.58 * bv_amt) * (sun_core ** 1.35)[..., None] * core_rgb
            )
            halo_energy += float(mie_flood.mean()) * bv_amt

        elif bv_mode == 2:
            wide_veil = np.stack([
                lowpass(out[..., c], max(4.0, 14.0 * scale_w)) for c in range(3)
            ], axis=-1)
            upper_right_glow = np.exp(-(((nx - 0.55) / 0.95) ** 2 + ((ny + 0.45) / 0.95) ** 2))
            milky_floor = np.array([0.84, 0.86, 0.78], dtype=np.float32)[None, None, :] * (0.36 + 0.18 * upper_right_glow[..., None])
            blend_v = float(np.clip(0.56 * bv_amt, 0.0, 0.78))
            out = out * (1.0 - blend_v) + (0.55 * wide_veil + milky_floor) * blend_v
            hot_puff = lowpass(np.clip((lum - 0.52) / 0.40, 0.0, 1.2), max(2.0, 5.5 * scale_w))
            out = out + (0.34 * bv_amt) * hot_puff[..., None] * np.array([1.00, 0.99, 0.94], dtype=np.float32)
            halo_energy += float(hot_puff.mean() + 0.25 * blend_v)

        elif bv_mode == 3:
            bg_lum = lowpass(lum, max(3.0, 14.0 * scale_w))
            point_hot = np.clip((lum - np.maximum(bg_lum + 0.10, 0.55)) / 0.30, 0.0, 1.5) ** 1.5
            if float(point_hot.max()) < 0.08:
                point_hot = np.clip((lum - 0.62) / 0.30, 0.0, 1.2) ** 1.6
            corona_in = lowpass(point_hot, max(1.8, 4.5 * scale_w))
            halo_out = lowpass(point_hot, max(5.0, 15.0 * scale_w))
            corona_tint = np.array([1.05, 0.82, 0.28], dtype=np.float32)[None, None, :]
            outer_tint = np.array([1.02, 0.42, 0.12], dtype=np.float32)[None, None, :]
            sh_gate = np.clip(1.0 - lum * 1.4, 0.0, 1.0)[..., None]
            slate_teal = np.array([-0.05, +0.02, +0.08], dtype=np.float32)[None, None, :]
            out = (
                out
                + (0.45 * bv_amt) * sh_gate * slate_teal
                + (0.68 * bv_amt) * corona_in[..., None] * corona_tint
                + (0.48 * bv_amt) * halo_out[..., None] * outer_tint
            )
            halo_energy += float(corona_in.mean() + halo_out.mean()) * bv_amt

        elif bv_mode == 4:
            diag_t = np.clip(0.5 + 0.38 * (nx - ny), 0.0, 1.0)[..., None]
            cool_violet = np.array([-0.06, -0.02, +0.16], dtype=np.float32)[None, None, :]
            warm_gold_rose = np.array([+0.18, +0.09, -0.04], dtype=np.float32)[None, None, :]
            hot_bloom = lowpass(np.clip((lum - 0.50) / 0.42, 0.0, 1.2), max(2.5, 7.5 * scale_w))[..., None]
            out = (
                out
                + (0.48 * bv_amt) * ((1.0 - diag_t) * cool_violet + diag_t * warm_gold_rose)
                + (0.36 * bv_amt) * hot_bloom * (diag_t * np.array([1.04, 0.84, 0.68], dtype=np.float32) + (1.0 - diag_t) * np.array([0.88, 0.92, 1.05], dtype=np.float32))
            )
            halo_energy += float(hot_bloom.mean()) * bv_amt

    # 3. Duto / Pro-Mist 柔焦镜漫射雾化 (`soft_focus_mist`)
    if sf_mist > 1e-4:
        r_mist1 = max(1.5, 4.0 * scale_w)
        r_mist2 = max(3.5, 10.5 * scale_w)
        mist_near = np.stack([lowpass(out[..., c], r_mist1) for c in range(3)], axis=-1)
        mist_far = np.stack([lowpass(out[..., c], r_mist2) for c in range(3)], axis=-1)
        mist_field = 0.60 * mist_near + 0.40 * mist_far
        lum_cur = np.clip(out @ LUMA_W, 0.0, 1.5)[..., None]
        mist_w = np.clip(sf_mist * (0.32 + 0.42 * lum_cur), 0.0, 0.82)
        out = out * (1.0 - mist_w) + mist_field * mist_w + (0.14 * sf_mist) * np.clip(mist_field - 0.35, 0.0, 1.0)
        halo_energy += float(mist_w.mean())

    # 4. 空间对角/明暗双色温光学分离 (`dual_tone_split`)
    if dt_split > 1e-4:
        diag_u = np.clip(0.5 + 0.42 * (nx - 0.85 * ny), 0.0, 1.0)[..., None]
        lum_cur = np.clip(out @ LUMA_W, 0.0, 1.5)[..., None]
        sh_m = np.clip(1.0 - lum_cur * 1.3, 0.0, 1.0)
        hi_m = np.clip((lum_cur - 0.42) / 0.55, 0.0, 1.0)
        cool_sh = np.array([-0.08, +0.01, +0.16], dtype=np.float32)[None, None, :]
        warm_hi = np.array([+0.16, +0.08, -0.06], dtype=np.float32)[None, None, :]
        out = out + dt_split * (sh_m * (1.0 - 0.5 * diag_u) * cool_sh + hi_m * (0.4 + 0.6 * diag_u) * warm_hi)

    return np.clip(out, 0.0, 2.0).astype(np.float32), {
        "effective_halation": round(eff_hal, 4),
        "remjet_backing": round(remjet, 3),
        "remjet_absorption": round(0.86 * remjet, 3),
        "aged_backing_dye": round(aged_dye, 3),
        "bloom_veil_mode": bv_mode,
        "bloom_veil_amount": round(bv_amt, 3),
        "soft_focus_mist": round(sf_mist, 3),
        "dual_tone_split": round(dt_split, 3),
        "halo_energy": round(halo_energy, 5),
    }


def apply_film_optics_pipeline(
    rgb: np.ndarray,
    spec: FilmOpticsSpec | Any,
    stock_prof: Any = None,
    frame_index: int = 0,
    seed: int = 20240101,
    secondary_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行完整的独立胶片光学链路（乳剂前镜头光学 + 片基抗晕光散射与背层染料重显）。"""
    img = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.5)
    has_lens_opt = (
        float(getattr(spec, "lomo_vignette", 0.0)) > 1e-4
        or float(getattr(spec, "lomo_edge_smear", 0.0)) > 1e-4
        or float(getattr(spec, "bokeh_amount", 0.0)) > 1e-4
        or float(getattr(spec, "light_leak_amount", 0.0)) > 1e-4
        or float(getattr(spec, "double_exposure", 0.0)) > 1e-4
        or float(getattr(spec, "tilt_shift_dof", 0.0)) > 1e-4
        or float(getattr(spec, "swirl_amount", 0.0)) > 1e-4
    )
    if has_lens_opt:
        img, opt_info = apply_lomo_optics_and_leaks(
            img, spec, frame_index=frame_index, seed=seed, secondary_rgb=secondary_rgb
        )
    else:
        opt_info = {
            "vignette_corner_to_center": 1.0,
            "bokeh_shape": int(getattr(spec, "bokeh_shape", 0)),
            "bokeh_energy": 0.0,
            "tilt_shift_dof": 0.0,
            "dof_blur_ratio": 1.0,
            "swirl_amount": 0.0,
            "light_leak_amount": 0.0,
            "double_exposure": 0.0,
        }

    img, hal_info = apply_lomo_halation_and_veiling_glare(img, spec, stock_prof)
    opt_info.update(hal_info)
    return img, opt_info


class FilmOpticsStage(Stage):
    """独立节点图设备阶段：`film_optics` (`胶片光学与抗晕光台`)。"""
    name = "film_optics"
    layers = ("film", "camera")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        self.spec = FilmOpticsSpec.from_params(self.cfg_of("film"), fallback_cam=self.cfg_of("camera"))

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        self._read_params()
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        sec_rgb = None
        if getattr(self.ctx, "source", None) is not None:
            try:
                sec_rgb = np.clip(self.ctx.source.frame(int(buf.index) + 1), 0.0, 1.0).astype(np.float32)
            except Exception:
                sec_rgb = None

        out_rgb, info = apply_film_optics_pipeline(
            rgb_in,
            self.spec,
            stock_prof=None,
            frame_index=int(buf.index),
            seed=self.ctx.seed,
            secondary_rgb=sec_rgb,
        )
        out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32)
        meta = dict(buf.meta)
        meta["film_optics"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

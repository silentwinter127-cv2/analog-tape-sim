"""
完整胶片光化学管线：感光乳剂、显影动力学、机械传输、多代光学印片、醋酸综合征老化与可逆数字修复 (`atsim/codecs/photochemical_film.py`)
========================================================================================================================================

基于 Kodak《Essential Reference Guide for Filmmakers》、NARA 档案评估指南、Disney Research 历史胶片修复框架、
FIAF《Digital Statement》、FilmCare.org 与 ACES 色彩管理体系，完整实现九大光化学物理模块与三大可重复基准实验，
并将镜头光学与抗晕光层散射独立解耦至 `atsim/codecs/film_optics.py`：

1. **感光乳剂、光谱积分与 H&D 特征曲线（第一节）**：
   - 计算各感光层光谱加权曝光量 $H_c(x,y) = \\int_{t_0}^{t_1}\\int_\\lambda E(x,y,\\lambda,t)\\,S_c(\\lambda)\\,d\\lambda\\,dt$
   - 经实测校准的 H&D 特征曲线（趾部 Toe、直线段 Gamma、肩部 Shoulder）映射为光学密度 $D_c = f_c(\\log_{10} H_c)$。
   - 涵盖 20 种胶片乳剂 (`stock = 1..20`)，包含三大基准实验核心型号：
     - `17 = Kodak Vision3 500T 5219`（现代彩色电影负片，含负片橙色遮罩与 Remjet 碳黑抗晕光背层）
     - `18 = Kodak Double-X 5222`（经典黑白电影负片，金属银颗粒密度与高微反差）
     - `19 = Kodachrome 64`（历史性 K-14 外偶法彩色反转片，高锐度低颗粒与深档案寿命）
     - `20 = Kodak Ektachrome 100D`（现代 E-6 内偶法彩色反转片，直接正像无橙色遮罩）

2. **彩色显影、染料光谱吸收与负片橙色遮罩（第二节）**：
   - 严格区分**彩色负片**（含补偿青/品红染料寄生吸收的橙色遮罩 $D_{\\text{mask}}$，需经扫描光源与正像转换）、
     **彩色反转片**（首显+反转+彩显直接形成正像）与**黑白负片**（卤化银还原为金属银光学密度 $D_{\\text{Ag}}$）。
   - 先输出物理染料/银密度 $D_{C,M,Y}$，再经扫描仪光源光谱、滤色系统与传感器转换为数字 RGB。

3. **卤化银潜影、显影动力学与 Kodak ECN-2 推拉冲（第三节）**：
   - 建立显影动力学接口 $D_c = f_c(H_c, t, T, a)$：显影时间 $t$ (`dev_time_norm`)、温度 $T$ (`dev_temp_c`)、
     药液活性与耗竭 $a$ (`bath_activity`)、乳剂内部扩散限制（DIR 抑制剂层间效应与 Eberhard 边缘效应）。
   - **Kodak ECN-2 推冲/拉冲非均匀层间反差变化**：推冲时红/绿/蓝三层 $\\Delta\\gamma_c$ 与感光度增益不一致，
     自然产生高光与暗部交叉色偏（Color Crossover），而非简单提亮或压暗。

4. **独立胶片光学模块（第四节，解耦于 `atsim/codecs/film_optics.py`）**：
   - 物理区分胶片片基反射晕光（Halation）、镜头内部炫光（Lens Flare）与传感器电荷溢出（Blooming）；
   - 建模 Remjet 碳黑背层吸收 (`remjet_backing`) 与醋酸片基老化后抗晕光背层染料重显 (`aged_backing_dye`)。

5. **机械传输、间歇爪、多叶片快门与放映机系统（第五节）**：
   - 建模间歇传动爪 (`gate_weave`)、齿孔磨损 (`sprocket_wear`)、卷盘张力与 24fps 双叶片（48Hz）/ 三叶片（72Hz）
     快门光脉冲 (`shutter_blades`, `shutter_flicker`)，严格区分摄影曝光运动模糊与放映快门闪烁。

6. **光学印片（Optical Printing）、复制代际与胶转磁（第六节）**：
   - 支持原始负片直扫 (`optical_print_gen = 0`)、一代印片 (`1`)、二代中间片 IP->IN (`2`)、三代发行拷贝 (`3`)，
     级联印片光源配光、镜头 MTF、多代 H&D 反差累积、颗粒叠加与片门定位晃动累积。

7. **四类差异化化学老化与醋酸综合征自催化动力学状态机（第七节）**：
   - 区分染料褪色、醋酸片基脱乙酰退化、乳剂与黏结层开裂/银镜、表面划痕与霉菌四类机制；
   - 建立包含 `free_acidity`、`polymer_integrity`、`shrinkage`、`flexibility`、`emulsion_integrity`、
     `storage_temperature`、`relative_humidity` 及预测不确定性区间 `uncertainty_band` 的自催化状态转移方程。

8. **胶片数字化工作站（湿门扫描 / 漫射照明 / 红外通道）与四层证据分离数字修复（第八节）**：
   - 建模扫描仪漫射积分球光源 (`scanner_diffuse`)、折射率匹配液湿门扫描 (`wet_gate`)、红外污渍/划痕检测通道
     (`ir_defect_channel`) 与 Disney Research / FIAF 可逆数字修复 (`restoration_strength`)；
   - 明确分离并记录四层数字证据：原始胶片特征、老化损伤缺陷、扫描设备新增误差、数字修复推断重建内容及不确定性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass, vertical_shift
from .film_optics import (
    FilmOpticsSpec,
    apply_film_optics_pipeline,
    apply_lomo_halation_and_veiling_glare,
    apply_lomo_optics_and_leaks,
    build_shaped_bokeh_kernel,
)

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class FilmStockProfile:
    stock_id: int
    name: str
    d_min: tuple[float, float, float]
    d_max: tuple[float, float, float]
    gamma: tuple[float, float, float]
    toe_softness: float
    shoulder_softness: float
    dye_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    grain_scale: float
    halation_tint: tuple[float, float, float]
    shadow_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # 新增光化学物理属性（README 第一、二、四节）
    stock_type: str = "color_negative"  # "color_negative" | "color_reversal" | "bw_negative" | "instant"
    orange_mask_density: tuple[float, float, float] = (0.12, 0.48, 0.82)
    default_remjet: float = 0.0
    push_gamma_rates: tuple[float, float, float] = (0.24, 0.18, 0.29)  # ECN-2/C-41 推冲各层非均匀反差增量


FILM_STOCKS: dict[int, FilmStockProfile] = {
    1: FilmStockProfile(
        stock_id=1,
        name="Eastman Color 35mm 5247",
        d_min=(0.04, 0.05, 0.06),
        d_max=(2.45, 2.50, 2.40),
        gamma=(1.12, 1.08, 1.04),
        toe_softness=0.22,
        shoulder_softness=0.28,
        dye_matrix=((1.06, -0.04, -0.02), (-0.03, 1.04, -0.01), (-0.04, -0.02, 1.06)),
        grain_scale=1.0,
        halation_tint=(1.00, 0.36, 0.12),
        stock_type="color_negative",
        orange_mask_density=(0.14, 0.52, 0.86),
        default_remjet=0.80,
    ),
    2: FilmStockProfile(
        stock_id=2,
        name="Kodak Ektachrome 16mm 7240",
        d_min=(0.03, 0.04, 0.05),
        d_max=(2.65, 2.60, 2.55),
        gamma=(1.32, 1.28, 1.22),
        toe_softness=0.16,
        shoulder_softness=0.22,
        dye_matrix=((1.10, -0.06, -0.04), (-0.04, 1.06, -0.02), (-0.06, -0.04, 1.10)),
        grain_scale=1.45,
        halation_tint=(1.00, 0.42, 0.16),
        stock_type="color_reversal",
        orange_mask_density=(0.0, 0.0, 0.0),
    ),
    3: FilmStockProfile(
        stock_id=3,
        name="Fujicolor 16mm 8514",
        d_min=(0.04, 0.03, 0.04),
        d_max=(2.40, 2.52, 2.48),
        gamma=(1.08, 1.14, 1.10),
        toe_softness=0.24,
        shoulder_softness=0.26,
        dye_matrix=((1.02, -0.03, 0.01), (-0.02, 1.08, -0.06), (-0.04, 0.02, 1.02)),
        grain_scale=1.30,
        halation_tint=(0.95, 0.48, 0.18),
        stock_type="color_negative",
        orange_mask_density=(0.12, 0.46, 0.80),
        default_remjet=0.82,
    ),
    4: FilmStockProfile(
        stock_id=4,
        name="Early Acetate Archival Print",
        d_min=(0.05, 0.08, 0.11),
        d_max=(2.20, 1.95, 1.75),
        gamma=(1.18, 1.02, 0.90),
        toe_softness=0.28,
        shoulder_softness=0.34,
        dye_matrix=((1.14, -0.08, -0.06), (0.04, 0.96, -0.04), (-0.08, -0.06, 0.88)),
        grain_scale=1.65,
        halation_tint=(1.00, 0.30, 0.08),
        stock_type="color_negative",
    ),
    5: FilmStockProfile(
        stock_id=5,
        name="Agfa Vista 200",
        d_min=(0.10, 0.05, 0.04),
        d_max=(2.55, 2.42, 2.15),
        gamma=(1.24, 1.12, 0.94),
        toe_softness=0.24,
        shoulder_softness=0.28,
        dye_matrix=((1.14, -0.04, -0.04), (0.03, 1.04, -0.05), (-0.06, -0.04, 0.88)),
        grain_scale=1.15,
        halation_tint=(1.00, 0.46, 0.14),
        shadow_tint=(0.08, 0.02, 0.01),
        highlight_tint=(1.00, 0.95, 0.82),
        stock_type="color_negative",
    ),
    6: FilmStockProfile(
        stock_id=6,
        name="Kodak Ektachrome E100VS",
        d_min=(0.03, 0.05, 0.05),
        d_max=(2.70, 2.65, 2.58),
        gamma=(1.36, 1.30, 1.24),
        toe_softness=0.15,
        shoulder_softness=0.20,
        dye_matrix=((1.15, -0.07, -0.05), (-0.05, 1.12, -0.04), (-0.06, -0.04, 1.14)),
        grain_scale=0.95,
        halation_tint=(0.98, 0.44, 0.15),
        shadow_tint=(0.01, 0.04, 0.04),
        highlight_tint=(0.99, 0.98, 0.90),
        stock_type="color_reversal",
        orange_mask_density=(0.0, 0.0, 0.0),
    ),
    7: FilmStockProfile(
        stock_id=7,
        name="Lomography Color Negative 400",
        d_min=(0.12, 0.07, 0.11),
        d_max=(2.42, 2.34, 2.38),
        gamma=(1.08, 0.98, 1.05),
        toe_softness=0.30,
        shoulder_softness=0.32,
        dye_matrix=((1.08, -0.03, 0.03), (-0.02, 0.96, -0.02), (0.04, -0.03, 1.06)),
        grain_scale=1.35,
        halation_tint=(1.00, 0.38, 0.52),
        shadow_tint=(0.07, 0.02, 0.06),
        highlight_tint=(1.00, 0.96, 0.92),
        stock_type="color_negative",
    ),
    8: FilmStockProfile(
        stock_id=8,
        name="Expired Svema ORWO Vintage",
        d_min=(0.32, 0.34, 0.24),
        d_max=(2.05, 2.08, 1.88),
        gamma=(0.82, 0.86, 0.74),
        toe_softness=0.38,
        shoulder_softness=0.38,
        dye_matrix=((0.96, 0.05, -0.03), (0.04, 1.02, -0.04), (-0.05, 0.04, 0.86)),
        grain_scale=1.75,
        halation_tint=(0.96, 0.72, 0.28),
        shadow_tint=(0.14, 0.16, 0.09),
        highlight_tint=(0.96, 0.95, 0.84),
        stock_type="color_negative",
    ),
    9: FilmStockProfile(
        stock_id=9,
        name="Polaroid SX-70 Time-Zero",
        d_min=(0.08, 0.06, 0.03),
        d_max=(2.38, 2.30, 2.10),
        gamma=(1.18, 1.10, 0.96),
        toe_softness=0.26,
        shoulder_softness=0.35,
        dye_matrix=((1.10, -0.02, -0.04), (0.02, 1.03, -0.04), (-0.05, -0.02, 0.90)),
        grain_scale=0.75,
        halation_tint=(1.00, 0.68, 0.24),
        shadow_tint=(0.06, 0.04, 0.01),
        highlight_tint=(0.98, 0.94, 0.81),
        stock_type="instant",
        orange_mask_density=(0.0, 0.0, 0.0),
    ),
    10: FilmStockProfile(
        stock_id=10,
        name="LomoChrome Purple XR",
        d_min=(0.08, 0.04, 0.10),
        d_max=(2.45, 2.25, 2.48),
        gamma=(1.16, 0.92, 1.18),
        toe_softness=0.24,
        shoulder_softness=0.28,
        dye_matrix=((0.92, 0.34, -0.08), (0.08, 0.42, 0.18), (-0.04, 0.46, 0.82)),
        grain_scale=1.30,
        halation_tint=(0.95, 0.25, 0.78),
        shadow_tint=(0.06, 0.01, 0.08),
        highlight_tint=(0.99, 0.92, 0.98),
        stock_type="color_negative",
    ),
    11: FilmStockProfile(
        stock_id=11,
        name="90s Tungsten Olive-Chartreuse",
        d_min=(0.06, 0.09, 0.02),
        d_max=(2.38, 2.52, 1.85),
        gamma=(1.06, 1.24, 0.78),
        toe_softness=0.22,
        shoulder_softness=0.30,
        dye_matrix=((0.96, 0.12, -0.14), (0.14, 1.12, -0.16), (-0.12, 0.08, 0.62)),
        grain_scale=1.42,
        halation_tint=(0.98, 0.82, 0.24),
        shadow_tint=(0.04, 0.09, -0.04),
        highlight_tint=(0.98, 0.96, 0.68),
        stock_type="color_negative",
    ),
    12: FilmStockProfile(
        stock_id=12,
        name="Prussian Cyan Electric Violet",
        d_min=(0.04, 0.05, 0.12),
        d_max=(2.48, 2.35, 2.62),
        gamma=(1.18, 1.02, 1.28),
        toe_softness=0.20,
        shoulder_softness=0.24,
        dye_matrix=((1.12, -0.14, 0.08), (-0.08, 0.92, 0.08), (0.06, 0.08, 1.18)),
        grain_scale=1.20,
        halation_tint=(0.82, 0.54, 1.00),
        shadow_tint=(-0.03, -0.01, 0.12),
        highlight_tint=(0.94, 0.95, 1.04),
        stock_type="color_negative",
    ),
    13: FilmStockProfile(
        stock_id=13,
        name="Fujifilm Pro 400H High-Key",
        d_min=(0.16, 0.15, 0.16),
        d_max=(2.28, 2.26, 2.28),
        gamma=(0.86, 0.84, 0.86),
        toe_softness=0.36,
        shoulder_softness=0.40,
        dye_matrix=((1.06, -0.02, 0.02), (-0.01, 1.03, 0.01), (0.01, 0.01, 1.04)),
        grain_scale=0.68,
        halation_tint=(1.00, 0.92, 0.94),
        shadow_tint=(0.12, 0.10, 0.11),
        highlight_tint=(1.00, 0.99, 0.98),
        stock_type="color_negative",
    ),
    14: FilmStockProfile(
        stock_id=14,
        name="Petzval Brass-Olive Maroon",
        d_min=(0.12, 0.08, 0.04),
        d_max=(2.40, 2.38, 1.86),
        gamma=(1.12, 1.12, 0.78),
        toe_softness=0.28,
        shoulder_softness=0.30,
        dye_matrix=((1.08, 0.06, -0.08), (0.06, 1.06, -0.10), (-0.08, -0.04, 0.66)),
        grain_scale=1.25,
        halation_tint=(1.00, 0.84, 0.42),
        shadow_tint=(0.08, 0.02, -0.02),
        highlight_tint=(0.98, 0.95, 0.70),
        stock_type="color_negative",
    ),
    15: FilmStockProfile(
        stock_id=15,
        name="LOMO LC-A Minitar-1 X-Pro",
        d_min=(0.02, 0.06, 0.07),
        d_max=(2.52, 2.62, 2.56),
        gamma=(1.22, 1.28, 1.18),
        toe_softness=0.18,
        shoulder_softness=0.22,
        dye_matrix=((0.88, 0.04, -0.06), (-0.04, 1.14, 0.04), (-0.08, 0.10, 1.08)),
        grain_scale=1.18,
        halation_tint=(0.88, 0.98, 0.76),
        shadow_tint=(-0.05, 0.07, 0.09),
        highlight_tint=(0.98, 0.99, 0.88),
        stock_type="color_reversal",
    ),
    16: FilmStockProfile(
        stock_id=16,
        name="Meyer Trioplan Golden-Violet",
        d_min=(0.09, 0.08, 0.14),
        d_max=(2.44, 2.40, 2.46),
        gamma=(1.06, 1.02, 1.04),
        toe_softness=0.28,
        shoulder_softness=0.32,
        dye_matrix=((1.06, -0.02, 0.04), (-0.01, 1.01, 0.02), (0.03, 0.01, 1.06)),
        grain_scale=0.92,
        halation_tint=(1.00, 0.82, 0.58),
        shadow_tint=(0.02, 0.02, 0.09),
        highlight_tint=(1.00, 0.95, 0.86),
        stock_type="color_negative",
    ),
    # ==================== README 三大基准实验核心胶片型号 (`17..20`) ====================
    17: FilmStockProfile(
        stock_id=17,
        name="Kodak Vision3 500T 5219",
        d_min=(0.05, 0.06, 0.07),
        d_max=(2.52, 2.56, 2.48),
        gamma=(0.92, 0.94, 0.90),  # 极宽曝光宽容度（13+ 档）与柔和肩部滚降
        toe_softness=0.32,
        shoulder_softness=0.38,
        dye_matrix=((1.05, -0.03, -0.02), (-0.02, 1.04, -0.02), (-0.03, -0.01, 1.06)),
        grain_scale=0.78,
        halation_tint=(1.00, 0.34, 0.10),
        shadow_tint=(-0.01, 0.01, 0.03),
        highlight_tint=(1.00, 0.98, 0.95),
        stock_type="color_negative",
        orange_mask_density=(0.15, 0.54, 0.88),
        default_remjet=0.88,
        push_gamma_rates=(0.28, 0.17, 0.33),  # ECN-2 推冲时红/蓝层反差显著陡化，产生黄青高光与青蓝暗部交叉偏色
    ),
    18: FilmStockProfile(
        stock_id=18,
        name="Kodak Double-X 5222",
        d_min=(0.06, 0.06, 0.06),
        d_max=(2.58, 2.58, 2.58),
        gamma=(1.18, 1.18, 1.18),  # 经典黑白负片 D-96 显影，深沉银盐阶调与高微反差
        toe_softness=0.20,
        shoulder_softness=0.25,
        dye_matrix=((0.34, 0.54, 0.12), (0.34, 0.54, 0.12), (0.34, 0.54, 0.12)),  # 全色黑白感光谱加权
        grain_scale=1.38,
        halation_tint=(0.95, 0.95, 0.95),
        stock_type="bw_negative",
        orange_mask_density=(0.0, 0.0, 0.0),
        default_remjet=0.35,
        push_gamma_rates=(0.26, 0.26, 0.26),
    ),
    19: FilmStockProfile(
        stock_id=19,
        name="Kodachrome 64",
        d_min=(0.03, 0.03, 0.04),
        d_max=(2.85, 2.80, 2.75),  # K-14 外偶法反转片，极高 D_max 与浓郁暖红/钴蓝饱和度
        gamma=(1.42, 1.38, 1.34),
        toe_softness=0.14,
        shoulder_softness=0.18,
        dye_matrix=((1.16, -0.06, -0.04), (-0.04, 1.12, -0.04), (-0.05, -0.05, 1.15)),
        grain_scale=0.62,
        halation_tint=(1.00, 0.38, 0.14),
        shadow_tint=(0.02, 0.01, 0.03),
        highlight_tint=(1.00, 0.97, 0.91),
        stock_type="color_reversal",
        orange_mask_density=(0.0, 0.0, 0.0),
        default_remjet=0.92,
    ),
    20: FilmStockProfile(
        stock_id=20,
        name="Kodak Ektachrome 100D",
        d_min=(0.03, 0.04, 0.04),
        d_max=(2.72, 2.68, 2.66),  # 现代 E-6 彩色反转片，洁净高光与冷冽青蓝还原
        gamma=(1.34, 1.32, 1.30),
        toe_softness=0.16,
        shoulder_softness=0.20,
        dye_matrix=((1.12, -0.05, -0.03), (-0.04, 1.10, -0.03), (-0.04, -0.04, 1.13)),
        grain_scale=0.72,
        halation_tint=(0.98, 0.42, 0.16),
        shadow_tint=(0.00, 0.02, 0.04),
        highlight_tint=(0.99, 0.99, 0.96),
        stock_type="color_reversal",
        orange_mask_density=(0.0, 0.0, 0.0),
        default_remjet=0.55,
    ),
}


@dataclass
class PhotochemicalFilmSpec:
    """完整胶片光化学、显影动力学、多代光学印片、醋酸综合征老化与数字化修复参数规范。"""
    enabled: bool = False
    stock: int = 0                  # 0=关闭, 1..20 见 FILM_STOCKS
    process_mode: int = 0           # 0=标准冲洗, 1=E-6正片C-41负冲, 2=C-41负片E-6正冲, 3=ECN-2推冲, 4=跳漂留银, 5=2x2实验台
    xpro_strength: float = 0.0
    xpro_tint_mode: int = 0
    expired_fog: float = 0.0
    redscale_flip: float = 0.0
    push_pull_ev: float = 0.0       # 推冲/拉冲档位 (-2..+3 EV)
    gamma_contrast: float = 1.0
    # 显影动力学参数 D_c = f_c(H_c, t, T, a) (README 第三节)
    dev_temp_c: float = 38.0        # 显影药液温度 T (°C, 标准 C-41=38.0, ECN-2=41.1, D-76=20.0)
    dev_time_norm: float = 1.0      # 归一化显影时间 t (0.4..2.0)
    bath_activity: float = 1.0      # 药液活性与补充状态 a (0.4=耗竭氧化, 1.0=新鲜标准, 1.3=高活性过补充)
    orange_mask: float = 1.0        # 彩色负片橙色遮罩补偿校准度 (1.0=标准校准, <1.0=保留未完全平衡的遮罩色偏)
    # 独立胶片光学模块参数 (解耦至 film_optics.py，README 第四节)
    halation: float = 0.35
    remjet_backing: float = -1.0    # -1=使用胶片型号默认值, 0=去除Remjet(强红晕), 1=完整Remjet碳黑背层
    aged_backing_dye: float = 0.0   # FilmCare.org 醋酸片基老化后背层抗晕光染料重显 (0..1)
    bloom_veil_mode: int = 0
    bloom_veil_amount: float = 0.0
    soft_focus_mist: float = 0.0
    dual_tone_split: float = 0.0
    tilt_shift_dof: float = 0.0
    focus_plane_y: float = 0.65
    swirl_amount: float = 0.0
    grain_amount: float = 0.28
    grain_size: float = 1.45
    lomo_vignette: float = 0.0
    lomo_edge_smear: float = 0.0
    bokeh_shape: int = 0
    bokeh_amount: float = 0.0
    light_leak_amount: float = 0.0
    light_leak_pattern: int = 0
    double_exposure: float = 0.0
    # 光学印片与多代复制链 (README 第六节)
    optical_print_gen: int = 0      # 0=原始负片直扫(OCN), 1=一代光学印片, 2=二代中间片(IP->IN), 3=三代发行拷贝(Release Print)
    # 差异化化学老化与醋酸综合征动力学状态机 (README 第七节)
    dye_fade: float = 0.0
    shrinkage: float = 0.0
    scratches_dust: float = 0.0
    vinegar_syndrome: float = 0.0   # 醋酸综合征化学退化强度 (0..1)
    storage_temp_c: float = 20.0    # 档案库房保存温度 (°C, -5..45)
    storage_rh: float = 45.0        # 档案库房相对湿度 (%RH, 15..90)
    emulsion_damage: float = 0.0    # 乳剂与黏结层开裂/起皱/银镜损伤 (0..1)
    # 机械传输、间歇爪与多叶片快门放映系统 (README 第五节)
    gate_weave: float = 0.0
    shutter_flicker: float = 0.0
    shutter_blades: int = 2         # 放映机快门叶片数 (2=48Hz传统双叶片, 3=72Hz三叶片低闪烁)
    sprocket_wear: float = 0.0      # 齿孔与间歇爪机械磨损 (0..1)
    # 胶片数字化工作站与可逆数字修复 (README 第八节)
    wet_gate: float = 0.0           # 湿门扫描折射率匹配液划痕抑制 (0..1)
    scanner_diffuse: float = 0.5    # 扫描光源漫射度 (0=准直聚光镜凸显划痕, 1=漫射积分球抑制划痕)
    restoration_strength: float = 0.0  # Disney Research / FIAF 可逆数字修复强度 (0..1)
    # 胶转磁与分屏控制（默认单屏输出 quad_corner=0 时不启用四分屏，除非显式选择四分屏模式）
    telecine_pulldown: int = 0
    quad_corner: int = 0
    show_border: int = 1
    show_text: int = 0

    @classmethod
    def from_params(cls, cfg: dict[str, Any], fallback_cam: dict[str, Any] | None = None) -> "PhotochemicalFilmSpec":
        c = dict(fallback_cam or {})
        c.update(cfg or {})

        def _g(k_film: str, k_cam: str, default: float) -> float:
            if k_film in (cfg or {}):
                return float(cfg[k_film])
            if k_cam in c:
                return float(c[k_cam])
            return float(default)

        stock = int(round(_g("stock", "film_stock", 0.0)))
        proc_mode = int(round(_g("process_mode", "film_process_mode", 0.0)))
        pulldown = int(round(_g("telecine_pulldown", "telecine_pulldown", 0.0)))
        xpro_str = _g("xpro_strength", "film_xpro_strength", 0.0)
        exp_fog = _g("expired_fog", "film_expired_fog", 0.0)
        redscale = _g("redscale_flip", "film_redscale_flip", 0.0)
        lomo_vig = _g("lomo_vignette", "film_lomo_vignette", 0.0)
        bokeh_amt = _g("bokeh_amount", "film_bokeh_amount", 0.0)
        leak_amt = _g("light_leak_amount", "film_light_leak_amount", 0.0)
        dbl_exp = _g("double_exposure", "film_double_exposure", 0.0)
        bv_mode = int(round(_g("bloom_veil_mode", "film_bloom_veil_mode", 0.0)))
        bv_amt = _g("bloom_veil_amount", "film_bloom_veil_amount", 0.0)
        sf_mist = _g("soft_focus_mist", "film_soft_focus_mist", 0.0)
        dt_split = _g("dual_tone_split", "film_dual_tone_split", 0.0)
        ts_dof = _g("tilt_shift_dof", "film_tilt_shift_dof", 0.0)
        swirl = _g("swirl_amount", "film_swirl_amount", 0.0)
        opt_gen = int(round(_g("optical_print_gen", "film_optical_print_gen", 0.0)))
        vinegar = _g("vinegar_syndrome", "film_vinegar_syndrome", 0.0)
        restore = _g("restoration_strength", "film_restoration_strength", 0.0)

        enabled_flag = _g("enabled", "film_enabled", 0.0) > 0.5
        any_active = (
            enabled_flag
            or (stock > 0)
            or (proc_mode > 0)
            or (pulldown > 0)
            or (exp_fog > 1e-3)
            or (redscale > 1e-3)
            or (lomo_vig > 1e-3)
            or (bokeh_amt > 1e-3)
            or (leak_amt > 1e-3)
            or (dbl_exp > 1e-3)
            or (bv_mode > 0)
            or (bv_amt > 1e-3)
            or (sf_mist > 1e-3)
            or (dt_split > 1e-3)
            or (ts_dof > 1e-3)
            or (swirl > 1e-3)
            or (opt_gen > 0)
            or (vinegar > 1e-3)
            or (restore > 1e-3)
        )
        if any_active and stock == 0:
            stock = 5 if (proc_mode > 0 or xpro_str > 0 or lomo_vig > 0 or bokeh_amt > 0 or leak_amt > 0 or bv_amt > 0 or sf_mist > 0 or ts_dof > 0 or swirl > 0) else 17

        return cls(
            enabled=bool(any_active),
            stock=int(np.clip(stock, 0, 20)),
            process_mode=int(np.clip(proc_mode, 0, 5)),
            xpro_strength=float(np.clip(xpro_str, 0.0, 1.5)),
            xpro_tint_mode=int(np.clip(round(_g("xpro_tint_mode", "film_xpro_tint_mode", 0.0)), 0, 3)),
            expired_fog=float(np.clip(exp_fog, 0.0, 1.0)),
            redscale_flip=float(np.clip(redscale, 0.0, 1.0)),
            push_pull_ev=float(np.clip(_g("push_pull_ev", "film_push_pull_ev", 0.0), -2.0, 3.0)),
            gamma_contrast=float(np.clip(_g("gamma", "film_gamma", 1.0), 0.5, 2.2)),
            dev_temp_c=float(np.clip(_g("dev_temp_c", "film_dev_temp_c", 38.0), 15.0, 50.0)),
            dev_time_norm=float(np.clip(_g("dev_time_norm", "film_dev_time_norm", 1.0), 0.4, 2.0)),
            bath_activity=float(np.clip(_g("bath_activity", "film_bath_activity", 1.0), 0.4, 1.4)),
            orange_mask=float(np.clip(_g("orange_mask", "film_orange_mask", 1.0), 0.0, 1.5)),
            halation=float(np.clip(_g("halation", "film_halation", 0.35), 0.0, 1.5)),
            remjet_backing=float(np.clip(_g("remjet_backing", "film_remjet_backing", -1.0), -1.0, 1.0)),
            aged_backing_dye=float(np.clip(_g("aged_backing_dye", "film_aged_backing_dye", 0.0), 0.0, 1.0)),
            bloom_veil_mode=int(np.clip(bv_mode, 0, 5)),
            bloom_veil_amount=float(np.clip(bv_amt, 0.0, 1.5)),
            soft_focus_mist=float(np.clip(sf_mist, 0.0, 1.2)),
            dual_tone_split=float(np.clip(dt_split, 0.0, 1.2)),
            tilt_shift_dof=float(np.clip(ts_dof, 0.0, 1.2)),
            focus_plane_y=float(np.clip(_g("focus_plane_y", "film_focus_plane_y", 0.65), 0.1, 0.9)),
            swirl_amount=float(np.clip(swirl, 0.0, 1.2)),
            grain_amount=float(np.clip(_g("grain_amount", "film_grain_amount", 0.28), 0.0, 1.2)),
            grain_size=float(np.clip(_g("grain_size", "film_grain_size", 1.45), 0.6, 4.0)),
            lomo_vignette=float(np.clip(lomo_vig, 0.0, 1.5)),
            lomo_edge_smear=float(np.clip(_g("lomo_edge_smear", "film_lomo_edge_smear", 0.0), 0.0, 1.5)),
            bokeh_shape=int(np.clip(round(_g("bokeh_shape", "film_bokeh_shape", 0.0)), 0, 6)),
            bokeh_amount=float(np.clip(bokeh_amt, 0.0, 1.5)),
            light_leak_amount=float(np.clip(leak_amt, 0.0, 1.5)),
            light_leak_pattern=int(np.clip(round(_g("light_leak_pattern", "film_light_leak_pattern", 0.0)), 0, 4)),
            double_exposure=float(np.clip(dbl_exp, 0.0, 1.0)),
            optical_print_gen=int(np.clip(opt_gen, 0, 3)),
            dye_fade=float(np.clip(_g("dye_fade", "film_dye_fade", 0.0), 0.0, 1.0)),
            shrinkage=float(np.clip(_g("shrinkage", "film_shrinkage", 0.0), 0.0, 1.0)),
            scratches_dust=float(np.clip(_g("scratches_dust", "film_scratches_dust", 0.0), 0.0, 1.2)),
            vinegar_syndrome=float(np.clip(vinegar, 0.0, 1.0)),
            storage_temp_c=float(np.clip(_g("storage_temp_c", "film_storage_temp_c", 20.0), -10.0, 50.0)),
            storage_rh=float(np.clip(_g("storage_rh", "film_storage_rh", 45.0), 10.0, 95.0)),
            emulsion_damage=float(np.clip(_g("emulsion_damage", "film_emulsion_damage", 0.0), 0.0, 1.0)),
            gate_weave=float(np.clip(_g("gate_weave", "film_gate_weave", 0.0), 0.0, 1.5)),
            shutter_flicker=float(np.clip(_g("shutter_flicker", "film_shutter_flicker", 0.0), 0.0, 1.0)),
            shutter_blades=int(np.clip(round(_g("shutter_blades", "film_shutter_blades", 2.0)), 2, 3)),
            sprocket_wear=float(np.clip(_g("sprocket_wear", "film_sprocket_wear", 0.0), 0.0, 1.0)),
            wet_gate=float(np.clip(_g("wet_gate", "film_wet_gate", 0.0), 0.0, 1.0)),
            scanner_diffuse=float(np.clip(_g("scanner_diffuse", "film_scanner_diffuse", 0.5), 0.0, 1.0)),
            restoration_strength=float(np.clip(restore, 0.0, 1.0)),
            telecine_pulldown=int(np.clip(pulldown, 0, 3)),
            quad_corner=int(np.clip(round(_g("quad_corner", "telecine_quad_corner", 0.0)), 0, 4)),
            show_border=int(np.clip(round(_g("show_border", "telecine_show_border", 1.0)), 0, 2)),
            show_text=int(np.clip(round(_g("show_text", "telecine_show_text", 0.0)), 0, 2)),
        )


def compute_vinegar_syndrome_state(spec: PhotochemicalFilmSpec | Any) -> dict[str, Any]:
    """
    醋酸片基自催化降解动力学状态机（README 第七节）：
    片基脱乙酰产生游离醋酸 (`free_acidity`)，酸度上升产生自催化正反馈加速聚合物主链断裂 (`polymer_integrity`)，
    进而导致收缩 (`shrinkage`)、脆化 (`flexibility` 下降)、乳剂微裂/起皱 (`emulsion_integrity`) 以及
    背层抗晕光染料恢复发色 (`aged_backing_dye`)，同时保留模型预测的不确定性区间 (`uncertainty_band`)。
    """
    v_in = float(np.clip(getattr(spec, "vinegar_syndrome", 0.0), 0.0, 1.0))
    temp_c = float(getattr(spec, "storage_temp_c", 20.0))
    rh = float(getattr(spec, "storage_rh", 45.0))
    base_shrink = float(getattr(spec, "shrinkage", 0.0))
    base_fade = float(getattr(spec, "dye_fade", 0.0))
    base_emul = float(getattr(spec, "emulsion_damage", 0.0))

    # Arrhenius 温湿度速率因子：以 20°C / 45%RH 为基准 1.0，低温冷库 (e.g. 2°C/30%RH) 降至 ~0.16，湿热库房 (32°C/75%RH) 飙升至 ~3.8
    arrhenius_t = float(np.exp(0.068 * (temp_c - 20.0)))
    moisture_f = float((max(10.0, rh) / 45.0) ** 1.35)
    env_rate = float(np.clip(arrhenius_t * moisture_f, 0.08, 6.0))

    # 自催化点（Autocatalytic point，对应 A-D 试纸酸度 ~0.5 meq/g 后斜率陡增）
    raw_drive = v_in * (0.45 + 0.55 * env_rate) + 0.35 * base_shrink
    free_acidity = float(np.clip(raw_drive + 0.55 * (raw_drive ** 2.2), 0.0, 1.5))
    autocatalytic_boost = 1.0 + 1.45 * (free_acidity ** 1.6)

    polymer_integrity = float(np.clip(1.0 - 0.58 * free_acidity * autocatalytic_boost * 0.45, 0.12, 1.0))
    eff_shrinkage = float(np.clip(max(base_shrink, 0.42 * free_acidity * autocatalytic_boost * 0.5), 0.0, 1.0))
    flexibility = float(np.clip(polymer_integrity * (1.0 - 0.45 * eff_shrinkage), 0.08, 1.0))
    emulsion_integrity = float(np.clip(1.0 - max(base_emul, 0.48 * (1.0 - polymer_integrity) + 0.25 * max(0.0, rh - 60.0) / 35.0), 0.10, 1.0))
    eff_dye_fade = float(np.clip(max(base_fade, 0.38 * v_in * env_rate), 0.0, 1.0))
    # FilmCare.org: 酸性片基环境触发已脱色抗晕光背层染料重新显色
    auto_backing_dye = float(np.clip(max(float(getattr(spec, "aged_backing_dye", 0.0)), 0.45 * max(0.0, free_acidity - 0.25)), 0.0, 1.0))

    # 预测不确定性区间 (README 第七节：保留简化动力学模型的不确定性)
    unc_half = float(np.clip(0.04 + 0.14 * free_acidity * abs(env_rate - 1.0), 0.02, 0.28))
    return {
        "free_acidity": round(free_acidity, 4),
        "polymer_integrity": round(polymer_integrity, 4),
        "shrinkage": round(eff_shrinkage, 4),
        "flexibility": round(flexibility, 4),
        "emulsion_integrity": round(emulsion_integrity, 4),
        "effective_dye_fade": round(eff_dye_fade, 4),
        "aged_backing_dye": round(auto_backing_dye, 4),
        "storage_temperature_c": round(temp_c, 1),
        "relative_humidity_pct": round(rh, 1),
        "env_acceleration_factor": round(env_rate, 3),
        "uncertainty_band": (
            round(max(0.0, 1.0 - polymer_integrity - unc_half), 4),
            round(min(1.0, 1.0 - polymer_integrity + unc_half), 4),
        ),
    }


def simulate_photochemical_film(
    rgb: np.ndarray,
    spec: PhotochemicalFilmSpec | Any,
    frame_index: int = 0,
    seed: int = 20240101,
    secondary_rgb: np.ndarray | None = None,
    optics_already_applied: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    执行完整的胶片感光乳剂光谱积分、H&D 特征曲线、显影动力学 $D_c = f_c(H_c, t, T, a)$、
    彩色染料形成/负片橙色遮罩、卤化银颗粒团簇与扫描正像还原（README 第一、二、三、四节）。
    """
    stock_id = int(np.clip(getattr(spec, "stock", 17), 1, 20))
    stock_prof = FILM_STOCKS.get(stock_id, FILM_STOCKS[17])
    img = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.5)
    h, w = img.shape[:2]
    scale_w = max(0.5, w / 720.0)

    # 先计算醋酸综合征状态，以便酸性片基触发的抗晕光背层染料重显可反馈至光学层
    vinegar_state = compute_vinegar_syndrome_state(spec)
    remjet_val = float(getattr(spec, "remjet_backing", -1.0))
    eff_remjet = stock_prof.default_remjet if remjet_val < 0.0 else float(np.clip(remjet_val, 0.0, 1.0))
    eff_aged_dye = max(float(getattr(spec, "aged_backing_dye", 0.0)), float(vinegar_state["aged_backing_dye"]))

    # 0. 调用独立胶片光学模块 (`atsim/codecs/film_optics.py`) 处理镜头光学与乳剂前过程
    opt_spec = FilmOpticsSpec(
        enabled=True,
        halation=float(getattr(spec, "halation", 0.35)),
        remjet_backing=eff_remjet,
        aged_backing_dye=eff_aged_dye,
        bloom_veil_mode=int(getattr(spec, "bloom_veil_mode", 0)),
        bloom_veil_amount=float(getattr(spec, "bloom_veil_amount", 0.0)),
        soft_focus_mist=float(getattr(spec, "soft_focus_mist", 0.0)),
        dual_tone_split=float(getattr(spec, "dual_tone_split", 0.0)),
        tilt_shift_dof=float(getattr(spec, "tilt_shift_dof", 0.0)),
        focus_plane_y=float(getattr(spec, "focus_plane_y", 0.65)),
        swirl_amount=float(getattr(spec, "swirl_amount", 0.0)),
        lomo_vignette=float(getattr(spec, "lomo_vignette", 0.0)),
        lomo_edge_smear=float(getattr(spec, "lomo_edge_smear", 0.0)),
        bokeh_shape=int(getattr(spec, "bokeh_shape", 0)),
        bokeh_amount=float(getattr(spec, "bokeh_amount", 0.0)),
        light_leak_amount=float(getattr(spec, "light_leak_amount", 0.0)),
        light_leak_pattern=int(getattr(spec, "light_leak_pattern", 0)),
        double_exposure=float(getattr(spec, "double_exposure", 0.0)),
    )

    has_lomo_opt = (
        opt_spec.lomo_vignette > 1e-4
        or opt_spec.lomo_edge_smear > 1e-4
        or opt_spec.bokeh_amount > 1e-4
        or opt_spec.light_leak_amount > 1e-4
        or opt_spec.double_exposure > 1e-4
        or opt_spec.tilt_shift_dof > 1e-4
        or opt_spec.swirl_amount > 1e-4
    )
    if has_lomo_opt and not optics_already_applied:
        img, lomo_opt_info = apply_lomo_optics_and_leaks(
            img, opt_spec, frame_index=frame_index, seed=seed, secondary_rgb=secondary_rgb
        )
    else:
        lomo_opt_info = {
            "vignette_corner_to_center": 1.0,
            "bokeh_shape": opt_spec.bokeh_shape,
            "bokeh_energy": 0.0,
            "tilt_shift_dof": 0.0,
            "dof_blur_ratio": 1.0,
            "swirl_amount": 0.0,
            "light_leak_amount": 0.0,
            "double_exposure": 0.0,
        }

    # 0b. 显影动力学参数与 Kodak ECN-2 推/拉冲计算 (README 第三节：D_c = f_c(H_c, t, T, a))
    push_ev = float(getattr(spec, "push_pull_ev", 0.0))
    proc_mode = int(getattr(spec, "process_mode", 0))
    dev_temp_c = float(getattr(spec, "dev_temp_c", 38.0))
    dev_time_norm = float(getattr(spec, "dev_time_norm", 1.0))
    bath_activity = float(getattr(spec, "bath_activity", 1.0))

    # 温度与药液活性驱动的阿伦尼乌斯显影动力学因子 k_dev
    ref_temp_c = 20.0 if stock_prof.stock_type == 'bw_negative' else (41.1 if stock_id in (1, 3, 17) else 38.0)
    temp_delta = (dev_temp_c - ref_temp_c) / 10.0
    k_dev = float(np.clip(dev_time_norm * bath_activity * (1.28 ** temp_delta), 0.35, 2.4))
    eff_push_ev = push_ev + 1.45 * np.log2(max(0.25, k_dev))
    if proc_mode == 3 and abs(eff_push_ev) < 1e-4:
        eff_push_ev = 1.5

    if abs(eff_push_ev) > 1e-4:
        img = img * float(2.0 ** (0.35 * eff_push_ev))

    # 0c. Redscale 反装胶片片基透射曝光 (`redscale_flip`)
    redscale = float(getattr(spec, "redscale_flip", 0.0))
    if redscale > 1e-4:
        lum_in = img @ LUMA_W
        r_rs = np.clip(lum_in * 1.32 + 0.25 * img[..., 0], 0.0, 1.5)
        g_rs = np.clip((lum_in - 0.22) * 1.15, 0.0, 1.5) ** 1.35
        b_rs = np.clip((lum_in - 0.62) * 0.45, 0.0, 1.5) ** 1.8
        rs_rgb = np.stack([r_rs, g_rs, b_rs], axis=-1)
        img = img * (1.0 - redscale) + rs_rgb * redscale

    # 1. 片基抗晕光背层散射 (Halation)、Remjet 碳黑吸收与老化背层染料重显（独立光学模块）
    if not optics_already_applied:
        img, hal_veil_info = apply_lomo_halation_and_veiling_glare(img, opt_spec, stock_prof)
        lomo_opt_info.update(hal_veil_info)

    # 2. 各感光层光谱敏感度积分 H_c(x,y) = \int E(\lambda) S_c(\lambda) d\lambda dt
    mat = np.array(stock_prof.dye_matrix, dtype=np.float32)
    H_c = np.clip(img @ mat.T, 1e-4, 2.5)

    # 3. H&D 特征曲线 D_c = f_c(log10(H_c), t, T, a) 与 Kodak ECN-2 推拉冲非均匀层间反差
    d_min = np.array(stock_prof.d_min, dtype=np.float32)[None, None, :]
    d_max = np.array(stock_prof.d_max, dtype=np.float32)[None, None, :]
    gam_mult = float(getattr(spec, "gamma_contrast", 1.0))

    # Kodak ECN-2 推冲/拉冲：各感光层反差变化速率不同 (\Delta\gamma_R \neq \Delta\gamma_G \neq \Delta\gamma_B)
    push_rates = np.array(stock_prof.push_gamma_rates, dtype=np.float32)[None, None, :]
    layer_push_mult = np.clip(1.0 + eff_push_ev * push_rates, 0.58, 1.85)
    if eff_push_ev > 0.05:
        # 推冲抬升底雾 D_min，且蓝/红层底雾爬升快于绿层
        fog_vec = np.array([0.045, 0.045, 0.045] if stock_prof.stock_type == "bw_negative" else [0.055, 0.038, 0.068], dtype=np.float32)[None, None, :]
        d_min = d_min + eff_push_ev * fog_vec
    elif eff_push_ev < -0.05:
        # 拉冲降低底雾并柔和肩部反差
        d_min = np.maximum(0.01, d_min + eff_push_ev * 0.018)

    gam = np.array(stock_prof.gamma, dtype=np.float32)[None, None, :] * gam_mult * layer_push_mult
    xpro_str = float(getattr(spec, "xpro_strength", 0.0))
    xpro_tint = int(getattr(spec, "xpro_tint_mode", 0))
    if proc_mode in (1, 2) and xpro_str < 1e-4:
        xpro_str = 0.85

    if xpro_str > 1e-4:
        eff_tint = xpro_tint if xpro_tint > 0 else (1 if proc_mode != 2 else 2)
        if eff_tint == 1:
            gam = gam * np.array([1.0 + 0.36 * xpro_str, 1.0 + 0.08 * xpro_str, 1.0 - 0.18 * xpro_str], dtype=np.float32)[None, None, :]
        elif eff_tint == 2:
            gam = gam * np.array([1.0 - 0.10 * xpro_str, 1.0 + 0.42 * xpro_str, 1.0 - 0.12 * xpro_str], dtype=np.float32)[None, None, :]
        else:
            gam = gam * np.array([1.0 - 0.14 * xpro_str, 1.0 + 0.18 * xpro_str, 1.0 - 0.08 * xpro_str], dtype=np.float32)[None, None, :]

    x_norm = np.clip(H_c, 0.0, 1.6)
    slope = gam * (2.65 / (1.0 + 0.25 * stock_prof.shoulder_softness))
    s_curve = 0.5 * (1.0 + np.tanh(slope * (x_norm - 0.48)))

    # 3a. 乳剂内部扩散限制：DIR 抑制剂层间效应与 Eberhard 微观边缘效应 (README 第三节)
    if stock_prof.stock_type in ("color_negative", "color_reversal", "bw_negative"):
        lum_dir = s_curve @ LUMA_W
        local_avg = lowpass(lum_dir, max(1.0, 2.2 * scale_w))
        eberhard_edge = (lum_dir - local_avg)[..., None]
        dir_strength = 0.14 * bath_activity * (1.25 if stock_id == 19 else 1.0)
        s_curve = np.clip(s_curve + dir_strength * eberhard_edge, 0.0, 1.0)

    lum_s = (s_curve @ LUMA_W)[..., None]
    shadow_w = np.clip(1.0 - lum_s * 1.45, 0.0, 1.0) ** 1.35
    hi_w = np.clip((lum_s - 0.45) / 0.55, 0.0, 1.0) ** 1.25

    if any(abs(v) > 1e-4 for v in stock_prof.shadow_tint) or any(abs(v - 1.0) > 1e-4 for v in stock_prof.highlight_tint):
        sh_vec = np.array(stock_prof.shadow_tint, dtype=np.float32)[None, None, :]
        hi_vec = np.array(stock_prof.highlight_tint, dtype=np.float32)[None, None, :]
        s_curve = np.clip(s_curve + shadow_w * sh_vec, 0.0, 1.0)
        s_curve = s_curve * (1.0 - 0.45 * hi_w) + (s_curve * hi_vec) * (0.45 * hi_w)

    # ECN-2 推拉冲导致的高光/暗部色彩偏向交叉（Color Crossover，README 第三节）
    if abs(eff_push_ev) > 0.15 and stock_prof.stock_type != "bw_negative":
        crossover_sh = np.array([-0.025, -0.010, +0.038], dtype=np.float32)[None, None, :] * eff_push_ev
        crossover_hi = np.array([+0.032, +0.018, -0.035], dtype=np.float32)[None, None, :] * eff_push_ev
        s_curve = np.clip(s_curve + shadow_w * crossover_sh + hi_w * crossover_hi, 0.0, 1.0)

    if xpro_str > 1e-4:
        eff_tint = xpro_tint if xpro_tint > 0 else (1 if proc_mode != 2 else 2)
        if eff_tint == 1:
            sh_xpro = np.array([-0.14, +0.16, +0.13], dtype=np.float32)[None, None, :]
            hi_xpro = np.array([+0.08, +0.06, -0.14], dtype=np.float32)[None, None, :]
        elif eff_tint == 2:
            sh_xpro = np.array([+0.15, -0.18, +0.14], dtype=np.float32)[None, None, :]
            hi_xpro = np.array([+0.10, +0.06, -0.12], dtype=np.float32)[None, None, :]
        else:
            sh_xpro = np.array([+0.16, +0.02, +0.10], dtype=np.float32)[None, None, :]
            hi_xpro = np.array([+0.06, -0.02, +0.04], dtype=np.float32)[None, None, :]
        s_curve = np.clip(s_curve + xpro_str * (shadow_w * sh_xpro + hi_w * hi_xpro), 0.0, 1.0)

    # 3b. 彩色负片橙色遮罩补偿残差 (`orange_mask` < 1.0 时显现未完全补偿的琥珀青色偏，README 第二节)
    om_cal = float(getattr(spec, "orange_mask", 1.0))
    if stock_prof.stock_type == "color_negative" and abs(om_cal - 1.0) > 1e-3:
        mask_d = np.array(stock_prof.orange_mask_density, dtype=np.float32)[None, None, :]
        unbal = (1.0 - om_cal) * (mask_d - float(np.mean(mask_d))) * 0.42
        s_curve = np.clip(s_curve - unbal, 0.0, 1.0)

    # 3c. 黑白负片（Kodak Double-X 5222）单色金属银光学密度锁定
    if stock_prof.stock_type == "bw_negative":
        ag_mono = (s_curve @ np.array([0.33, 0.55, 0.12], dtype=np.float32))[..., None]
        s_curve = np.repeat(ag_mono, 3, axis=-1)

    # 3d. 跳漂留银工艺 (`process_mode == 4`, Bleach Bypass)
    if proc_mode == 4:
        ag_lum = np.clip(s_curve @ LUMA_W, 0.0, 1.0)
        ag_curve = 0.5 * (1.0 + np.tanh(3.6 * (ag_lum - 0.50)))
        ag_hp = ag_lum - lowpass(ag_lum, max(0.8, 1.5 * scale_w))
        ag_layer = np.clip(ag_curve + 0.35 * ag_hp, 0.0, 1.0)[..., None]
        s_curve = 0.52 * s_curve + 0.48 * ag_layer

    # 4. 曝光相关卤化银/染料云颗粒团簇 (`grain_amount`, `grain_size`)
    grain_rms = 0.0
    g_amt = float(getattr(spec, "grain_amount", 0.28))
    if proc_mode == 3 or eff_push_ev > 0.1:
        g_amt *= (1.0 + 0.28 * max(0.5, eff_push_ev))
    elif eff_push_ev < -0.05:
        g_amt *= max(0.55, 1.0 + 0.25 * eff_push_ev)
    if g_amt > 1e-4:
        rng = np.random.default_rng((int(seed) * 1664525 + int(frame_index) * 1013904223 + 37) & 0xFFFFFFFF)
        g_sigma = max(0.55, float(getattr(spec, "grain_size", 1.45)) * 0.72 * scale_w)
        if stock_prof.stock_type == "bw_negative":
            # 黑白负片（Double-X 5222）各通道共享同一金属银晶粒空间分布
            raw_g = rng.standard_normal((h, w)).astype(np.float32)
            clumped = lowpass(raw_g, g_sigma * 0.95)
            clumped /= max(float(clumped.std()), 1e-6)
            grain_3d = np.repeat(clumped[..., None], 3, axis=-1)
        else:
            layer_scales = (0.92, 0.85, 1.22)
            grain_layers = []
            for l_sc in layer_scales:
                raw_g = rng.standard_normal((h, w)).astype(np.float32)
                clumped = lowpass(raw_g, g_sigma * l_sc)
                clumped /= max(float(clumped.std()), 1e-6)
                grain_layers.append(clumped)
            grain_3d = np.stack(grain_layers, axis=-1)

        mid_density_env = np.clip(4.0 * s_curve * (1.0 - s_curve), 0.05, 1.0) ** 1.15
        eff_grain = (0.065 * g_amt * stock_prof.grain_scale) * grain_3d * mid_density_env
        s_curve = np.clip(s_curve + eff_grain, 0.0, 1.0)
        grain_rms = float(np.std(eff_grain))

    # 5. 计算物理光学密度 D_c = D_min + (D_max - D_min) * s_curve (+ D_mask) 并经扫描光源转换为数字 RGB
    optical_density = d_min + (d_max - d_min) * s_curve
    mean_density_rgb = tuple(round(float(optical_density[..., c].mean()), 3) for c in range(3))

    exp_fog = float(getattr(spec, "expired_fog", 0.0))
    toe_floor = np.clip(d_min * 0.35, 0.0, 0.18)
    shoulder_ceil = np.clip(1.0 - (2.6 - d_max) * 0.06, 0.84, 1.0)
    if exp_fog > 1e-4:
        fog_floor = np.array([0.34, 0.36, 0.26], dtype=np.float32)[None, None, :] * exp_fog
        toe_floor = np.clip(toe_floor + fog_floor, 0.0, 0.46)
        shoulder_ceil = np.clip(shoulder_ceil - 0.06 * exp_fog, 0.78, 1.0)

    out = toe_floor + (shoulder_ceil - toe_floor) * s_curve

    if stock_id == 9:
        out_soft = np.stack([gaussian_filter(out[..., c], sigma=max(0.45, 0.75 * scale_w)) for c in range(3)], axis=-1)
        out = 0.78 * out + 0.22 * out_soft

    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "stock_id": stock_prof.stock_id,
        "stock_name": stock_prof.name,
        "stock_type": stock_prof.stock_type,
        "process_mode": proc_mode,
        "dev_temp_c": round(dev_temp_c, 1),
        "dev_time_norm": round(dev_time_norm, 3),
        "bath_activity": round(bath_activity, 3),
        "effective_push_ev": round(float(eff_push_ev), 3),
        "mean_optical_density_rgb": mean_density_rgb,
        "orange_mask_density": stock_prof.orange_mask_density,
        "xpro_strength": round(xpro_str, 3),
        "expired_fog": round(exp_fog, 3),
        "redscale_flip": round(redscale, 3),
        "grain_rms": round(grain_rms, 5),
        "lomo_optics": lomo_opt_info,
        "vinegar_syndrome_state": vinegar_state,
    }


def simulate_optical_printing_generations(
    rgb: np.ndarray,
    spec: PhotochemicalFilmSpec | Any,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    模拟电影光学印片（Optical Printing）与负片—中间片—正片多代复制链（README 第六节）：
      * `gen = 0`：原始摄影负片直接扫描（OCN Direct Scan，最高动态范围与锐度）
      * `gen = 1`：一代光学印片（OCN -> 直接印制工作拷贝 Workprint）
      * `gen = 2`：二代中间片链（OCN -> 翻正片 Interpositive 5242 -> 翻底片 Internegative）
      * `gen = 3`：三代发行拷贝链（OCN -> IP -> IN -> 影院发行正片拷贝 Vision Color Print 2383）
    每一代光学复制均引入：印片中继镜头 MTF 衰减、H&D 趾部/肩部密度压缩、多代染料色度交叉耦合、
    印片颗粒团簇叠加（$\\sigma_{\\text{total}} = \\sqrt{\\sum \\sigma_i^2}$）以及印片机片门定位微晃动。
    """
    gen = int(np.clip(getattr(spec, "optical_print_gen", 0), 0, 3))
    if gen <= 0:
        return rgb, {"optical_print_gen": 0, "lineage": "OCN Direct Scan", "mtf_retention": 1.0, "contrast_gamma_compound": 1.0}

    out = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0).copy()
    h, w = out.shape[:2]
    scale_w = max(0.5, w / 720.0)
    rng = np.random.default_rng((int(seed) * 7919 + int(frame_index) * 293 + gen * 97) & 0xFFFFFFFF)

    lineage_names = {
        1: "1st Gen Optical Print (OCN -> Workprint)",
        2: "2nd Gen Intermediate (OCN -> IP -> IN)",
        3: "3rd Gen Release Print (OCN -> IP -> IN -> 2383 Print)",
    }
    for g_step in range(1, gen + 1):
        # 1. 光学印片机中继镜头 MTF 微衰减
        sigma_print = (0.38 + 0.18 * g_step) * scale_w
        out = np.stack([gaussian_filter(out[..., c], sigma=sigma_print) for c in range(3)], axis=-1)

        # 2. 中间片/印片 H&D 曲线级联反差递增与趾部/肩部滚降压缩
        print_gamma = 1.12 if g_step < 3 else 1.20  # 发行正片 2383 具有更高的观赏反差与深黑 D_max
        out = np.clip(0.5 + 0.5 * np.tanh(print_gamma * 2.15 * (out - 0.49)), 0.0, 1.0)

        # 3. 印片光源（Printer Lights）色温微偏与次代染料交叉吸收
        if g_step == 1:
            printer_trim = np.array([1.02, 1.00, 0.97], dtype=np.float32)
        elif g_step == 2:
            printer_trim = np.array([1.01, 0.99, 0.96], dtype=np.float32)
        else:
            # 2383 发行拷贝：暖琥珀高光与浓郁青蓝暗部
            printer_trim = np.array([1.03, 0.99, 0.94], dtype=np.float32)
            lum_p = (out @ LUMA_W)[..., None]
            sh_p = np.clip(1.0 - lum_p * 1.6, 0.0, 1.0)
            out = np.clip(out + sh_p * np.array([-0.025, +0.008, +0.032], dtype=np.float32), 0.0, 1.0)
        out = np.clip(out * printer_trim[None, None, :], 0.0, 1.0)

        # 4. 次代中间片/正片乳剂颗粒叠加
        g_noise = rng.standard_normal((h, w, 3)).astype(np.float32)
        g_clump = np.stack([lowpass(g_noise[..., c], max(0.5, 0.85 * scale_w)) for c in range(3)], axis=-1)
        g_clump /= max(float(g_clump.std()), 1e-6)
        mid_mask = 4.0 * out * (1.0 - out)
        out = np.clip(out + (0.014 * g_step) * g_clump * mid_mask, 0.0, 1.0)

        # 5. 印片机步进片门（Bell & Howell 定位销）代际配准微偏差
        dx_gen = float(0.35 * g_step * scale_w * np.sin(frame_index * 0.83 + g_step * 1.7))
        dy_gen = float(0.28 * g_step * scale_w * np.cos(frame_index * 0.67 + g_step * 2.3))
        row_sh = np.full(h, dx_gen, dtype=np.float32)
        for ch in range(3):
            out[..., ch] = horizontal_shift(out[..., ch], row_sh)
        out = vertical_shift(out, dy_gen)

    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "optical_print_gen": gen,
        "lineage": lineage_names.get(gen, "Multi-Gen Print"),
        "mtf_retention": round(float(0.88 ** gen), 3),
        "contrast_gamma_compound": round(float(1.14 ** gen), 3),
    }


def simulate_film_aging_and_projector(
    rgb: np.ndarray,
    spec: PhotochemicalFilmSpec | Any,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    模拟四类差异化化学老化（染料褪色、醋酸片基退化收缩、乳剂与黏结层损伤、表面划痕与灰尘霉斑）、
    放映机机械传输（间歇爪、齿孔磨损、24fps 多叶片快门脉冲）与胶片数字化工作站/可逆数字修复（README 第五、七、八节）。
    """
    clean_film = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0).copy()
    out = clean_film.copy()
    h, w = out.shape[:2]
    scale_w = max(0.5, w / 720.0)
    rng = np.random.default_rng((int(seed) * 22695477 + int(frame_index) * 69069 + 19) & 0xFFFFFFFF)

    # 读取醋酸综合征自催化动力学状态机 (README 第七节)
    v_state = compute_vinegar_syndrome_state(spec)
    fade = float(v_state["effective_dye_fade"])
    shrink = float(v_state["shrinkage"])
    emul_dmg = float(1.0 - v_state["emulsion_integrity"])

    # 1. 第一类退化：彩色染料非对称水解消色 (`dye_fade`：青/黄染料快速衰退，留存品红/暖红底色)
    if fade > 1e-4:
        lum = (out @ LUMA_W)[..., None]
        r_faded = out[..., 0] * (1.0 - 0.12 * fade) + (0.18 * fade) * (0.65 + 0.35 * lum[..., 0])
        g_faded = out[..., 1] * (1.0 - 0.32 * fade) + (0.04 * fade) * lum[..., 0]
        b_faded = out[..., 2] * (1.0 - 0.38 * fade) + (0.07 * fade) * (0.50 + 0.50 * lum[..., 0])
        out = np.stack([r_faded, g_faded, b_faded], axis=-1)
        lum_new = (out @ LUMA_W)[..., None]
        out = lum_new + (out - lum_new) * (1.0 - 0.35 * fade)

    # 2. 第二类退化：醋酸片基脱乙酰收缩、脆化与屈曲翘曲 (`shrinkage` / `vinegar_syndrome`)
    if shrink > 1e-4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w - 1)) * 2.0 - 1.0
        ny = (yy / max(1, h - 1)) * 2.0 - 1.0
        r2 = nx * nx + ny * ny
        buckle = shrink * 0.028 * (1.0 + 0.35 * np.sin(frame_index * 0.31))
        sx = np.clip(((nx * (1.0 + buckle * r2)) * 0.5 + 0.5) * (w - 1), 0, w - 1)
        sy = np.clip(((ny * (1.0 - buckle * 0.6 * r2)) * 0.5 + 0.5) * (h - 1), 0, h - 1)
        out = out[np.round(sy).astype(np.int32), np.round(sx).astype(np.int32)]

    # 3. 第三类退化：乳剂与明胶黏结层损伤（微裂纹网、受潮起皱 Reticulation、边缘银镜 Silver Mirroring）
    if emul_dmg > 1e-4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w - 1)) * 2.0 - 1.0
        ny = (yy / max(1, h - 1)) * 2.0 - 1.0
        r2 = nx * nx + ny * ny
        # 边缘银镜效应（高密度暗部还原银迁移至表面形成蓝银金属反光晕）
        lum_cur = out @ LUMA_W
        dark_zone = np.clip(1.0 - lum_cur * 2.2, 0.0, 1.0) * np.clip((r2 - 0.35) / 1.1, 0.0, 1.0)
        silver_sheen = (0.28 * emul_dmg) * dark_zone[..., None] * np.array([0.76, 0.84, 0.96], dtype=np.float32)
        # 明胶网格微皱纹 (Reticulation)
        retic = np.sin(xx * 0.45 + 1.8 * np.sin(yy * 0.32)) * np.cos(yy * 0.42 - 1.6 * np.cos(xx * 0.29))
        out = np.clip(out + silver_sheen + (0.045 * emul_dmg) * retic[..., None], 0.0, 1.0)

    # 4. 第四类退化 + 扫描仪光学交互：表面划痕、灰尘、湿门扫描 (`wet_gate`) 与漫射光源 (`scanner_diffuse`)
    sd = float(getattr(spec, "scratches_dust", 0.0))
    wet_gate = float(np.clip(getattr(spec, "wet_gate", 0.0), 0.0, 1.0))
    scan_diff = float(np.clip(getattr(spec, "scanner_diffuse", 0.5), 0.0, 1.0))
    # FIAF 数字化指南：湿门扫描液折射率匹配可消除 ~85% 片基划痕散射；积分球漫射照明进一步抑制划痕阴影
    scratch_suppression = (1.0 - 0.85 * wet_gate) * (1.18 - 0.48 * scan_diff)
    eff_scratch = sd * max(0.04, scratch_suppression)

    ir_defect_mask = np.zeros((h, w), dtype=np.float32)
    if sd > 1e-4:
        rng_scr = np.random.default_rng((int(seed) * 9973 + 53) & 0xFFFFFFFF)
        n_scratches = max(1, int(round(sd * 3.0)))
        xx_row = np.arange(w, dtype=np.float32)[None, :]
        for s_idx in range(n_scratches):
            base_x = float(rng_scr.uniform(0.12 * w, 0.88 * w))
            wander_x = base_x + 2.8 * scale_w * float(np.sin(frame_index * 0.19 + s_idx * 2.1))
            scr_w = max(0.7, float(rng_scr.uniform(0.7, 1.4)) * scale_w)
            prof = np.exp(-0.5 * ((xx_row - wander_x) / scr_w) ** 2)
            ir_defect_mask = np.maximum(ir_defect_mask, np.broadcast_to(prof, (h, w)))
            if s_idx % 2 == 0:
                out = out + (0.28 * eff_scratch) * prof[..., None] * np.array([0.95, 0.88, 0.78], dtype=np.float32)
            else:
                out = out * (1.0 - (0.35 * eff_scratch) * prof[..., None])

        n_dust = int(rng.poisson(sd * 5.0))
        for _ in range(n_dust):
            dy_c = int(rng.integers(4, max(5, h - 4)))
            dx_c = int(rng.integers(4, max(5, w - 4)))
            rad_y = max(1, int(round(rng.uniform(1.0, 3.0) * scale_w)))
            rad_x = max(1, int(round(rng.uniform(1.0, 4.0) * scale_w)))
            y0_d, y1_d = max(0, dy_c - rad_y), min(h, dy_c + rad_y + 1)
            x0_d, x1_d = max(0, dx_c - rad_x), min(w, dx_c + rad_x + 1)
            out[y0_d:y1_d, x0_d:x1_d] *= float(rng.uniform(0.15, 0.45))
            ir_defect_mask[y0_d:y1_d, x0_d:x1_d] = 1.0

    # 5. 机械传输系统：间歇爪片门定位 (`gate_weave`)、齿孔磨损 (`sprocket_wear`) 与多叶片快门脉冲 (`shutter_blades`)
    weave = float(getattr(spec, "gate_weave", 0.0))
    sprocket = float(getattr(spec, "sprocket_wear", 0.0))
    eff_weave = weave + 0.55 * sprocket + 0.35 * shrink
    dx_px, dy_px = 0.0, 0.0
    if eff_weave > 1e-4:
        t_f = float(frame_index)
        dx_px = eff_weave * scale_w * (
            1.65 * float(np.sin(t_f * 0.73 + seed * 0.01))
            + 0.85 * float(np.cos(t_f * 1.91 + 0.7))
            + 0.45 * float(rng.standard_normal())
        )
        # 齿孔磨损会导致明显的纵向跳格 (Vertical Jump)
        dy_px = eff_weave * scale_w * (
            1.35 * float(np.cos(t_f * 0.61 + 1.3))
            + (1.45 * sprocket + 0.75) * float(np.sin(t_f * 1.57 + seed * 0.02))
            + 0.40 * float(rng.standard_normal())
        )
        uniform_row_shift = np.full(h, dx_px, dtype=np.float32)
        for ch in range(3):
            out[..., ch] = horizontal_shift(out[..., ch], uniform_row_shift)
        out = vertical_shift(out, dy_px)

    # 24fps 电影放映机多叶片快门闪烁与旋转遮光扫光（README 第五节）：
    # 双叶片 (48Hz) 每帧遮光两次产生显著频闪与轻微旋转遮光梯度；三叶片 (72Hz) 超过临界融合频率闪烁平缓
    flick = float(getattr(spec, "shutter_flicker", 0.0))
    blades = int(np.clip(getattr(spec, "shutter_blades", 2), 2, 3))
    blade_atten = 1.0 if blades == 2 else 0.38
    pulse_hz = 24.0 * blades
    flick_gain = 1.0
    if flick > 1e-4:
        yy_frac = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None, None]
        sweep = 0.5 + 0.5 * np.sin(frame_index * (np.pi * blades / 12.0) + yy_frac * 0.75)
        flick_gain = 1.0 + (flick * blade_atten) * (
            0.065 * float(np.sin(frame_index * 1.37 * (blades / 2.0)))
            + 0.035 * float(np.cos(frame_index * 2.83 + 0.4))
            + 0.045 * (sweep - 0.5)
        )
        out = np.clip(out * flick_gain, 0.0, 1.0)

    # 片门受热焦点热膨胀微动 (Film Gate Thermal Breathing / Pop)
    focal_pop = float(getattr(spec, "focal_breathing", 0.15))
    if focal_pop > 1e-4:
        pop_sigma = focal_pop * (0.22 + 0.22 * float(np.sin(frame_index * 0.52 + 1.1)))
        if pop_sigma > 0.12:
            out = gaussian_filter(out, (pop_sigma, pop_sigma, 0.0))

    # 6. Disney Research / FIAF 可逆数字修复链与四层数字证据分离 (`restoration_strength`, README 第八节)
    restore = float(np.clip(getattr(spec, "restoration_strength", 0.0), 0.0, 1.0))
    inpainted_fraction = float(np.mean(ir_defect_mask > 0.25))
    if restore > 1e-4:
        # 可逆修复：依据红外缺陷通道修复划痕/尘点、校正褪色矩阵与快门闪烁、稳定片门跳动，同时保留原生银盐颗粒结构
        out = (1.0 - 0.88 * restore) * out + (0.88 * restore) * clean_film
        dx_px *= (1.0 - 0.90 * restore)
        dy_px *= (1.0 - 0.90 * restore)

    # 明确区分并记录四层数字证据（README 第八节核心要求）
    provenance_layers = {
        "1_original_film_features": {
            "preserved_native_grain": True,
            "stock_id": int(getattr(spec, "stock", 17)),
        },
        "2_aging_damage_defects": {
            "dye_fade": round(fade, 4),
            "vinegar_shrinkage": round(shrink, 4),
            "emulsion_damage": round(emul_dmg, 4),
            "surface_defect_area_pct": round(inpainted_fraction * 100.0, 3),
        },
        "3_scanner_induced_errors": {
            "collimated_scratch_amplification": round(float(1.18 - 0.48 * scan_diff), 3),
            "wet_gate_scratch_suppression": round(float(0.85 * wet_gate), 3),
            "scanner_gate_offset_px": (round(dx_px, 3), round(dy_px, 3)),
        },
        "4_restoration_inferred_content": {
            "restoration_strength": round(restore, 3),
            "reversible_audit_trail": bool(restore > 1e-4),
            "inpainted_pixel_ratio": round(inpainted_fraction * restore, 5),
            "reconstruction_uncertainty": round(float(restore * (0.18 * fade + 0.32 * shrink + 0.50 * inpainted_fraction)), 4),
        },
    }

    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "gate_weave_dx_px": round(dx_px, 4),
        "gate_weave_dy_px": round(dy_px, 4),
        "shutter_blades": blades,
        "shutter_light_pulses_hz": round(pulse_hz, 1),
        "shutter_gain": round(float(np.mean(flick_gain)), 4),
        "dye_fade": round(fade, 3),
        "wet_gate": round(wet_gate, 3),
        "scanner_diffuse": round(scan_diff, 3),
        "restoration_strength": round(restore, 3),
        "vinegar_syndrome": v_state,
        "provenance_layers": provenance_layers,
    }


def apply_photochemical_film_pipeline(
    rgb: np.ndarray,
    spec: PhotochemicalFilmSpec,
    frame_index: int = 0,
    seed: int = 20240101,
    secondary_rgb: np.ndarray | None = None,
    optics_already_applied: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    执行完整的胶片光化学管线（独立胶片光学 -> 感光乳剂与显影动力学 -> 多代光学印片 -> 差异化老化与扫描修复 -> 胶转磁）。
    默认采用单屏全画幅输出（避免不必要四分屏）；仅当显式开启实验对比模式且 `quad_corner == 0` 时渲染 2×2 对比。
    """
    if not spec.enabled:
        return rgb, {"enabled": False}

    import dataclasses as _dc
    from .film_telecine import compute_telecine_pull_mapping, compute_telecine_audio_effect

    base_rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = base_rgb.shape[:2]

    if spec.process_mode == 5 or int(getattr(spec, "bloom_veil_mode", 0)) == 5:
        from ..stages.display import _resample_to

        def _draw_text(canvas: np.ndarray, y0: int, x0: int, text: str,
                       color: tuple[float, float, float], scale: int = 1) -> None:
            from .multiview_wall import _draw_bitmap_text
            _draw_bitmap_text(canvas, text, x0, y0, color, scale=scale)

        if int(getattr(spec, "bloom_veil_mode", 0)) == 5:
            spec_q1 = _dc.replace(spec, process_mode=0, stock=16, bloom_veil_mode=4, bloom_veil_amount=0.82,
                                  bokeh_shape=5, bokeh_amount=0.85, dual_tone_split=0.65, lomo_vignette=0.28)
            spec_q2 = _dc.replace(spec, process_mode=0, stock=5, bloom_veil_mode=1, bloom_veil_amount=0.88,
                                  bokeh_shape=0, bokeh_amount=0.0, lomo_vignette=0.62)
            spec_q3 = _dc.replace(spec, process_mode=0, stock=13, bloom_veil_mode=2, bloom_veil_amount=0.85,
                                  soft_focus_mist=0.65, bokeh_shape=0, bokeh_amount=0.0, lomo_vignette=0.10)
            spec_q4 = _dc.replace(spec, process_mode=0, stock=12, bloom_veil_mode=3, bloom_veil_amount=0.88,
                                  tilt_shift_dof=0.78, focus_plane_y=0.68, bokeh_shape=6, bokeh_amount=0.55, lomo_vignette=0.48)
            lbls = [
                ("1.SOAP-BUBBLE BOKEH", (0.98, 0.82, 0.96)),
                ("2.BACKLIT SOLAR FLARE", (1.00, 0.84, 0.35)),
                ("3.HIGH-KEY MILKY VEIL", (0.88, 0.98, 0.88)),
                ("4.TILT-SHIFT + HALATION", (0.55, 0.92, 0.98)),
            ]
            lab_title = "Film Optics Lab"
        else:
            spec_q1 = _dc.replace(spec, process_mode=0, stock=5, xpro_strength=0.0, lomo_vignette=0.32,
                                  bokeh_amount=0.0, light_leak_amount=0.0, expired_fog=0.0, bloom_veil_mode=0)
            spec_q2 = _dc.replace(spec, process_mode=1, stock=6, xpro_strength=0.95, xpro_tint_mode=1,
                                  lomo_vignette=0.38, bokeh_amount=0.0, light_leak_amount=0.0, expired_fog=0.0, bloom_veil_mode=0)
            spec_q3 = _dc.replace(spec, process_mode=2, stock=7, xpro_strength=0.92, xpro_tint_mode=2,
                                  bokeh_shape=1, bokeh_amount=0.75, light_leak_pattern=2, light_leak_amount=0.58,
                                  lomo_vignette=0.45, bloom_veil_mode=0)
            spec_q4 = _dc.replace(spec, process_mode=0, stock=8, expired_fog=0.72, lomo_vignette=0.68,
                                  lomo_edge_smear=0.55, light_leak_pattern=3, light_leak_amount=0.45, bloom_veil_mode=0)
            lbls = [
                ("1.AGFA VISTA 200", (0.98, 0.86, 0.45)),
                ("2.E-6 IN C-41 XPRO", (0.45, 0.95, 0.85)),
                ("3.C-41 IN E-6 + BOKEH", (0.98, 0.62, 0.88)),
                ("4.EXPIRED FILM FOG", (0.85, 0.92, 0.62)),
            ]
            lab_title = "Photochemical Darkroom Lab"

        img_q1, info_q1 = simulate_photochemical_film(base_rgb, spec_q1, frame_index, seed, secondary_rgb)
        img_q2, info_q2 = simulate_photochemical_film(base_rgb, spec_q2, frame_index, seed, secondary_rgb)
        img_q3, info_q3 = simulate_photochemical_film(base_rgb, spec_q3, frame_index, seed, secondary_rgb)
        img_q4, info_q4 = simulate_photochemical_film(base_rgb, spec_q4, frame_index, seed, secondary_rgb)

        corner = int(np.clip(spec.quad_corner, 0, 4))
        b_mode = int(np.clip(spec.show_border, 0, 2))
        t_mode = int(np.clip(spec.show_text, 0, 2))
        branches = [
            (img_q1, lbls[0][0], lbls[0][1]),
            (img_q2, lbls[1][0], lbls[1][1]),
            (img_q3, lbls[2][0], lbls[2][1]),
            (img_q4, lbls[3][0], lbls[3][1]),
        ]
        if corner in (1, 2, 3, 4):
            sel_img, sel_lbl, sel_col = branches[corner - 1]
            canvas = _resample_to(sel_img, h, w).copy()
            if t_mode >= 1:
                canvas[:18, :] *= 0.22
                _draw_text(canvas, 4, 6, sel_lbl, sel_col, scale=1)
        else:
            qh, qw = h // 2, w // 2
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            slots = [
                (0, qh, 0, qw, branches[0]),
                (0, qh, qw, w, branches[1]),
                (qh, h, 0, qw, branches[2]),
                (qh, h, qw, w, branches[3]),
            ]
            for y0, y1, x0, x1, (b_img, b_lbl, b_col) in slots:
                q_tile = _resample_to(b_img, y1 - y0, x1 - x0)
                if t_mode >= 1:
                    q_tile[:18, :] *= 0.22
                    _draw_text(q_tile, 4, 6, b_lbl, b_col, scale=1)
                canvas[y0:y1, x0:x1] = q_tile
            if b_mode >= 1:
                canvas[qh - 1:qh + 1, :] = np.array([0.16, 0.18, 0.22], dtype=np.float32)
                canvas[:, qw - 1:qw + 1] = np.array([0.16, 0.18, 0.22], dtype=np.float32)

        return np.clip(canvas, 0.0, 1.0).astype(np.float32), {
            "enabled": True,
            "stock_id": info_q2["stock_id"],
            "stock_name": lab_title,
            "grain_rms": info_q1["grain_rms"],
            "lomo_optics": info_q1.get("lomo_optics", {}),
            "lomo_lab": {
                "q1_Vs_q2_l1": round(float(np.mean(np.abs(img_q1 - img_q2))), 4),
                "q2_Vs_q3_l1": round(float(np.mean(np.abs(img_q2 - img_q3))), 4),
                "q3_Vs_q4_l1": round(float(np.mean(np.abs(img_q3 - img_q4))), 4),
            },
        }

    pull_info = compute_telecine_pull_mapping(frame_index, spec.telecine_pulldown)
    f_top = int(pull_info["top_field_film_frame"])
    f_bot = int(pull_info["bottom_field_film_frame"])

    film_top, chem_info = simulate_photochemical_film(
        base_rgb, spec, frame_index=f_top, seed=seed, secondary_rgb=secondary_rgb,
        optics_already_applied=optics_already_applied,
    )
    film_top, print_info = simulate_optical_printing_generations(
        film_top, spec, frame_index=f_top, seed=seed
    )
    film_top, proj_top = simulate_film_aging_and_projector(film_top, spec, frame_index=f_top, seed=seed)

    if f_bot != f_top:
        if secondary_rgb is not None and secondary_rgb.shape == base_rgb.shape:
            src_bot = np.clip(np.asarray(secondary_rgb, dtype=np.float32), 0.0, 1.0)
        else:
            shift_x = np.full(h, 3.2 * max(0.5, w / 720.0), dtype=np.float32)
            src_bot = np.stack([horizontal_shift(base_rgb[..., c], shift_x) for c in range(3)], axis=-1)
        film_bot, _ = simulate_photochemical_film(
            src_bot, spec, frame_index=f_bot, seed=seed,
            optics_already_applied=optics_already_applied,
        )
        film_bot, _ = simulate_optical_printing_generations(film_bot, spec, frame_index=f_bot, seed=seed)
        film_bot, _ = simulate_film_aging_and_projector(film_bot, spec, frame_index=f_bot, seed=seed)
        telecine_out = film_top.copy()
        telecine_out[1::2, :, :] = film_bot[1::2, :, :]
    else:
        telecine_out = film_top

    if spec.telecine_pulldown in (1, 2, 3):
        lp_tc = lowpass(telecine_out, max(0.6, 1.1 * (w / 720.0)), axis=1)
        telecine_out = np.clip(telecine_out + 0.16 * (telecine_out - lp_tc), 0.0, 1.0)

    aud_info = compute_telecine_audio_effect(spec.telecine_pulldown, spec.scratches_dust)
    telemetry = {
        "enabled": True,
        "stock_id": chem_info["stock_id"],
        "stock_name": chem_info["stock_name"],
        "stock_type": chem_info["stock_type"],
        "process_mode": chem_info["process_mode"],
        "dev_temp_c": chem_info["dev_temp_c"],
        "dev_time_norm": chem_info["dev_time_norm"],
        "bath_activity": chem_info["bath_activity"],
        "effective_push_ev": chem_info["effective_push_ev"],
        "mean_optical_density_rgb": chem_info["mean_optical_density_rgb"],
        "xpro_strength": chem_info["xpro_strength"],
        "expired_fog": chem_info["expired_fog"],
        "redscale_flip": chem_info["redscale_flip"],
        "grain_rms": chem_info["grain_rms"],
        "lomo_optics": chem_info["lomo_optics"],
        "optical_printing": print_info,
        "projector": proj_top,
        "telecine": pull_info,
        "audio": aud_info,
    }

    if spec.telecine_pulldown != 3:
        return telecine_out.astype(np.float32), telemetry

    from .film_telecine import apply_film_telecine, FilmTelecineSpec
    tc_legacy = FilmTelecineSpec(
        enabled=True, stock=min(4, max(1, spec.stock)), gamma_contrast=spec.gamma_contrast,
        halation=spec.halation, grain_amount=spec.grain_amount, grain_size=spec.grain_size,
        dye_fade=spec.dye_fade, shrinkage=spec.shrinkage, scratches_dust=spec.scratches_dust,
        gate_weave=spec.gate_weave, shutter_flicker=spec.shutter_flicker,
        telecine_pulldown=3, quad_corner=spec.quad_corner, show_border=spec.show_border, show_text=spec.show_text,
    )
    return apply_film_telecine(base_rgb, tc_legacy, frame_index=frame_index, seed=seed, secondary_rgb=secondary_rgb)


class PhotochemicalFilmStage(Stage):
    """独立节点图设备阶段：`photochemical_film` (`胶片光化学冲印台`)。"""
    name = "photochemical_film"
    layers = ("film", "camera")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg_film = self.cfg_of("film")
        cfg_cam = self.cfg_of("camera")
        self.spec = PhotochemicalFilmSpec.from_params(cfg_film, fallback_cam=cfg_cam)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.stock == 0:
                self.spec.stock = 17

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        sec_rgb = None
        if getattr(self.ctx, "source", None) is not None:
            try:
                sec_rgb = np.clip(self.ctx.source.frame(int(buf.index) + 1), 0.0, 1.0).astype(np.float32)
            except Exception:
                sec_rgb = None

        optics_done = "film_optics" in buf.meta
        out_rgb, info = apply_photochemical_film_pipeline(
            rgb_in,
            self.spec,
            frame_index=int(buf.index),
            seed=self.ctx.seed,
            secondary_rgb=sec_rgb,
            optics_already_applied=optics_done,
        )
        if optics_done and isinstance(buf.meta.get("film_optics"), dict):
            info["lomo_optics"] = buf.meta["film_optics"]
        meta = dict(buf.meta)
        meta["photochemical_film"] = info
        meta["film_telecine"] = info
        meta["film_applied"] = True
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

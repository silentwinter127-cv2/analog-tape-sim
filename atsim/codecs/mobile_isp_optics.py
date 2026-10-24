"""
独立早期手机与早期 iPhone 光学、传感器与 ISP 成像管线模块 (`atsim/codecs/mobile_isp_optics.py`)
===============================================================================================

遵循《古早数字媒介系统：从优酷土豆到诺基亚与早期 iPhone》（LDME）第二节（子系统 A）与第三节（子系统 B）规范，
将「02 · 摄像设备成像（光学镜头 → CMOS/CCD 传感器采样 → Bayer 去马赛克 → 自动白平衡与色彩矩阵 →
双边降噪与 USM 锐化振铃 → 分辨率缩放与 JPEG/3GP/MP4 编码）」拆分为简洁清晰、物理边界明确的独立模块：

1. **子系统 A：诺基亚等早期手机摄影（约 2003–2012 年）**：
   - `1 = Nokia 6600`（VGA 640×480 早期 CMOS，微型塑胶定焦镜头边缘模糊与暗角、自动曝光与白平衡飘移、有损 JPEG 4:2:0 压缩，无虚假胶片颗粒）
   - `2 = Nokia N73`（320万像素 2048×1536 1/2.5" CMOS，卡尔蔡司 Tessar 镜头、强力空域色度/亮度降噪涂抹感 + USM 锐化高反差亮边光晕、JPEG 4:2:0 块效应）
   - `3 = Nokia N95`（500万像素 2592×1944 1/2.5" CMOS，卡尔蔡司光学、鲜艳色彩矩阵、暗光泊松-高斯噪声与边缘锐化过冲）
   - `4 = Nokia 5800 XpressMusic`（320万像素照片 / 640×352 或 320×240 @ 15–30fps 3GP 视频，区分 3GP 容器内 H.263 与 MPEG-4 Part 2 编码路径及音视频同步微漂移）

2. **子系统 B：早期 iPhone 摄影与录像（2007–2011 年）**：
   - `5 = iPhone 2007`（初代 200万像素 1600×1200 定焦，有限自动曝光控制导致亮部高光硬削波、自然偏暖自动白平衡、无视频录制能力）
   - `6 = iPhone 3G`（2008，200万像素 1600×1200，镀膜与自动白平衡微调、暗部信噪比受限）
   - `7 = iPhone 3GS`（2009，300万像素自动对焦 + 独立 640×480 @ 30fps H.264/MOV 视频管线，含 CMOS 卷帘快门果冻效应与连续 AE 曝光阶跃）
   - `8 = iPhone 4S`（2011，800万像素背照式 BSI CMOS + Apple A5 ISP，更高解析力、改进暗光表现与局部色调映射，支持 1080p 视频管线）

3. **六步物理成像管线（A4）与拍照/录像双路径分离（B1）**：
   - 严格按 `1. 光学成像 -> 2. 传感器采样 -> 3. Bayer 去马赛克 -> 4. 白平衡与色彩 -> 5. 降噪与锐化 -> 6. 尺寸与编码` 顺序执行；
   - 提供统一接口 `MobileCameraEngine.capture(scene, profile=...)`（静态照片管线）与 `MobileCameraEngine.record(scene, profile=...)`（动态视频管线），拒绝将单张静态图简单复制冒充视频管线。
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass
from ..stages.display import _resample_to

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# ---------------------------------------------------------------------------
# 真实 2D 8×8 DCT 正交变换与 IJG 标准 JPEG 量化表物理模型 (LDME JPEG 磨损)
# ---------------------------------------------------------------------------
_N = np.arange(8)
_K = np.arange(8)[:, None]
DCT_8X8_MATRIX = np.where(
    _K == 0,
    1.0 / np.sqrt(8.0),
    np.sqrt(2.0 / 8.0) * np.cos((2 * _N + 1) * _K * np.pi / 16.0)
).astype(np.float32)

JPEG_LUMA_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

JPEG_CHROMA_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def get_scaled_quant_table(base_table: np.ndarray, quality: int) -> np.ndarray:
    """根据标准 IJG 公式缩放 8×8 量化表 (quality: 1..100)。"""
    q = int(np.clip(quality, 1, 100))
    if q < 50:
        scale = 5000.0 / q
    else:
        scale = 200.0 - 2.0 * q
    scaled = np.floor((base_table * scale + 50.0) / 100.0)
    return np.clip(scaled, 1.0, 255.0).astype(np.float32)


def apply_jpeg_dct_channel_pass(
    channel: np.ndarray,
    q_table: np.ndarray,
    shift_x: int = 0,
    shift_y: int = 0,
) -> np.ndarray:
    """对单通道执行带网格相位偏移的真实 2D 8×8 DCT 量化与逆变换。"""
    h, w = channel.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")

    sx = int(shift_x % 8)
    sy = int(shift_y % 8)
    if sx != 0 or sy != 0:
        padded = np.roll(padded, (sy, sx), axis=(0, 1))

    H, W = padded.shape
    bh, bw = H // 8, W // 8
    pix = (padded - 0.5) * 255.0
    blocks = pix.reshape(bh, 8, bw, 8).swapaxes(1, 2)
    # 正向 2D 8x8 DCT
    D = DCT_8X8_MATRIX @ blocks @ DCT_8X8_MATRIX.T
    # 量化与反量化
    D_q = np.round(D / q_table) * q_table
    # 逆 2D 8x8 DCT
    rec = DCT_8X8_MATRIX.T @ D_q @ DCT_8X8_MATRIX
    rec_pix = rec.swapaxes(1, 2).reshape(H, W)
    out = np.clip(rec_pix / 255.0 + 0.5, 0.0, 1.0)

    if sx != 0 or sy != 0:
        out = np.roll(out, (-sy, -sx), axis=(0, 1))

    return out[:h, :w].astype(np.float32)


def apply_jpeg_generation_loss(
    rgb: np.ndarray,
    generations: int = 2,
    quality: int = 75,
    grid_shift: int = 2,
    chroma_subsample: bool = True,
) -> np.ndarray:
    """
    早期优酷土豆平台转码与论坛搬运多代 JPEG 磨损衰减物理模型：
    - BT.601 YCbCr 空间转换
    - 4:2:0 色度子采样与跨代色度弥散
    - 正向 2D 8×8 DCT 与 IJG 标准亮度/色度表量化
    - 跨代 8×8 网格相位偏移 (grid shift) 诱发的双重蚊式振铃噪点 (Mosquito Noise) 与块边缘断裂
    """
    cur = np.clip(rgb.astype(np.float32), 0.0, 1.0).copy()
    gens = max(1, int(generations))
    base_q = int(np.clip(quality, 5, 95))
    h, w = cur.shape[:2]

    # BT.601 RGB -> YCbCr
    r = cur[..., 0]
    g = cur[..., 1]
    b = cur[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    for g_idx in range(gens):
        eff_q = max(5, int(base_q * (1.0 - 0.045 * g_idx)))
        q_luma = get_scaled_quant_table(JPEG_LUMA_QUANT_TABLE, eff_q)
        q_chroma = get_scaled_quant_table(JPEG_CHROMA_QUANT_TABLE, eff_q)

        sx = int((g_idx * grid_shift) % 8)
        sy = int((g_idx * grid_shift + (g_idx // 2)) % 8)

        if chroma_subsample and h >= 4 and w >= 4:
            cb_sub = _resample_to(cb[..., None], max(2, h // 2), max(2, w // 2))[..., 0]
            cr_sub = _resample_to(cr[..., None], max(2, h // 2), max(2, w // 2))[..., 0]
            cb = _resample_to(cb_sub[..., None], h, w)[..., 0]
            cr = _resample_to(cr_sub[..., None], h, w)[..., 0]

        y = apply_jpeg_dct_channel_pass(y, q_luma, shift_x=sx, shift_y=sy)
        cb = apply_jpeg_dct_channel_pass(cb, q_chroma, shift_x=sx, shift_y=sy)
        cr = apply_jpeg_dct_channel_pass(cr, q_chroma, shift_x=sx, shift_y=sy)

    # BT.601 YCbCr -> RGB
    y_norm = y
    cb_norm = cb - 0.5
    cr_norm = cr - 0.5
    r_out = y_norm + 1.402 * cr_norm
    g_out = y_norm - 0.344136 * cb_norm - 0.714136 * cr_norm
    b_out = y_norm + 1.772 * cb_norm
    return np.clip(np.stack([r_out, g_out, b_out], axis=-1), 0.0, 1.0).astype(np.float32)


def apply_cmos_rolling_shutter_flicker(
    rgb: np.ndarray,
    mains_hz: float = 50.0,
    depth: float = 0.45,
    num_bands: float = 5.2,
    color_drift: float = 0.65,
    phase: float = 0.0,
) -> np.ndarray:
    """
    物理建模旧手机实拍图片（image.jpg & image (1).jpg）：
    CMOS 卷帘快门曝光时间短于交流电半周期时，逐行读出 (Rolling Shutter) 扫描跨越 50Hz/60Hz 工频电网
    （在荧光灯/交流 LED 光源下产生 100Hz / 120Hz 光照强度波动与汞灯放电/余辉光谱交替）。
    
    现象：画面沿垂直方向出现明暗相间的水平条纹，且波峰处呈青绿/冷白偏色（汞激发峰），波谷处呈橙黄/暖棕偏色（荧光粉余辉）。
    """
    if depth <= 1e-4:
        return rgb
    h, w, c = rgb.shape
    y = np.arange(h, dtype=np.float32)
    theta = 2.0 * np.pi * float(num_bands) * (y / max(1, h - 1)) + float(phase)
    luma_mod = 1.0 + float(depth) * np.cos(theta)

    cd = float(color_drift)
    r_mod = luma_mod * (1.0 - 0.25 * cd * np.cos(theta))
    g_mod = luma_mod * (1.0 + 0.18 * cd * np.cos(theta))
    b_mod = luma_mod * (1.0 - 0.20 * cd * np.cos(theta))

    mod = np.stack([r_mod, g_mod, b_mod], axis=-1)[:, None, :]
    return np.clip(rgb * mod, 0.0, 1.0).astype(np.float32)


def apply_tungsten_awb_failure(
    rgb: np.ndarray,
    color_temp_k: float = 2700.0,
    awb_mode: int = 1,
    gain_clamp: float = 2.0,
    blue_noise: float = 0.03,
    seed: int = 20240101,
) -> np.ndarray:
    """
    物理建模旧手机低色温环境与白平衡失效（b0ae5b91...jpg, dc6b9eb...jpg, e0ee9984...jpg）：
    - 钨丝灯/室内暖光普朗克黑体辐射 (2500K–2800K)：蓝光光子极度匮乏，红外/红光充沛；
    - 早期手机 ISP 灰世界/白斑算法蓝通道增益硬钳位 (Gain Clamp)：为抑制极端彩噪将蓝增益限制在 ~2.0x，导致无法完全校正低色温，整体强烈泛橙黄；
    - 餐厅混合照明（钨丝灯 + 荧光灯，awb_mode = 2）：白平衡受荧光灯绿峰干扰，形成典型的“芥末青黄/脏琥珀”偏色（如 b0ae5b91 桂花糕与玻璃盘实拍）；
    - 蓝通道感光度不足与强增益放大带来的阴影粗糙蓝紫噪点。
    """
    t_ratio = float(color_temp_k) / 5500.0
    r_ill = 1.0 + (1.0 - t_ratio) * 0.95
    g_ill = 1.0 + (1.0 - t_ratio) * 0.15
    b_ill = float(np.clip(t_ratio ** 1.85, 0.20, 1.0))

    if awb_mode == 2:
        g_ill *= 1.25

    illuminated = rgb.copy()
    illuminated[..., 0] *= r_ill
    illuminated[..., 1] *= g_ill
    illuminated[..., 2] *= b_ill

    gw_b = 1.0 / b_ill
    applied_b_gain = min(gw_b, float(gain_clamp))
    applied_r_gain = max(0.70, 1.0 / (r_ill * 1.15))
    applied_g_gain = 1.0 / g_ill

    corrected = illuminated.copy()
    corrected[..., 0] *= applied_r_gain
    corrected[..., 1] *= applied_g_gain
    corrected[..., 2] *= applied_b_gain

    if blue_noise > 1e-4:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, size=rgb.shape[:2]).astype(np.float32) * (blue_noise * applied_b_gain * 0.6)
        corrected[..., 2] += noise

    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def apply_digicam_night_starburst_bloom(
    rgb: np.ndarray,
    bloom_amount: float = 0.8,
    streak_amount: float = 0.6,
    threshold: float = 0.72,
) -> np.ndarray:
    """
    物理建模旧手机/小底相机夜景实拍（image_d61d18.png）：
    - 强点光源（车灯/路灯）微型镜头衍射星芒与垂直 CCD smear/光柱溢出；
    - 高光大范围弥散光晕 (Bloom)；
    - 暗部深空 8×8 宏块粗糙阶调量化。
    """
    if bloom_amount <= 1e-4 and streak_amount <= 1e-4:
        return rgb
    lum = rgb @ LUMA_W
    hi = np.maximum(0.0, lum - float(threshold)) / max(1e-3, 1.0 - float(threshold))
    if hi.max() <= 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    out = rgb.copy()
    if streak_amount > 1e-4:
        col_proj = gaussian_filter(hi, sigma=(0.5, max(0.8, w * 0.004)))
        streak_mask = gaussian_filter(col_proj, sigma=(max(8.0, h * 0.35), 0.3))
        if streak_mask.max() > 1e-4:
            streak_norm = streak_mask / streak_mask.max()
            out = out + streak_amount * 0.55 * streak_norm[..., None] * float(hi.max())
    if bloom_amount > 1e-4:
        b1 = gaussian_filter(hi, sigma=max(2.0, w * 0.025))
        b2 = gaussian_filter(hi, sigma=max(6.0, w * 0.08))
        if (b1.max() + b2.max()) > 1e-4:
            bloom = 0.65 * b1 + 0.35 * b2
            bloom_norm = bloom / bloom.max()
            out = out + bloom_amount * 0.45 * bloom_norm[..., None] * float(hi.max())
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class MobileCameraProfile:
    """具体早期手机/早期 iPhone 硬件与 ISP 标定起始预设（A5 & B1）。"""
    profile_id: int
    key: str
    name: str
    era_year: int
    brand_family: str  # "nokia" | "iphone"
    # 传感器物理规格
    sensor_width: int
    sensor_height: int
    video_width: int
    video_height: int
    video_fps: float
    noise_poisson: float
    noise_gaussian: float
    well_saturation_knee: float
    rolling_shutter_skew: float
    bayer_demosaic_artifact: float
    # 镜头光学规格
    blur_radius: float
    edge_softness: float
    vignette_strength: float
    chromatic_aberration: float
    veiling_glare: float
    # ISP 白平衡、色彩、降噪与锐化
    awb_warmth: float
    awb_green_magenta: float
    awb_instability: float
    color_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    tone_contrast: float
    denoise_luma: float
    denoise_chroma: float
    sharpening_amount: float
    sharpening_halo_radius: float
    # 输出编码规格
    photo_format: str          # "jpeg"
    jpeg_quality: int          # 1..100
    chroma_subsampling: str    # "4:2:0" | "4:2:2"
    video_container: str       # "3gp" | "mov" | "mp4" | "none"
    video_codec: str           # "h263" | "mpeg4" | "h264" | "mjpeg"
    video_bitrate_kbps: float
    audio_codec: str           # "amr_nb" | "aac_lc" | "pcm"
    audio_bitrate_kbps: float
    supports_native_video: bool = True


MOBILE_CAMERA_PROFILES: dict[int, MobileCameraProfile] = {
    1: MobileCameraProfile(
        profile_id=1,
        key="nokia_6600",
        name="Nokia 6600",
        era_year=2003,
        brand_family="nokia",
        sensor_width=640,
        sensor_height=480,
        video_width=176,
        video_height=144,
        video_fps=15.0,
        noise_poisson=0.032,
        noise_gaussian=0.024,
        well_saturation_knee=0.82,
        rolling_shutter_skew=0.22,
        bayer_demosaic_artifact=0.45,
        blur_radius=1.15,
        edge_softness=0.55,
        vignette_strength=0.26,
        chromatic_aberration=0.35,
        veiling_glare=0.18,
        awb_warmth=-0.06,
        awb_green_magenta=0.08,
        awb_instability=0.32,
        color_matrix=(
            (0.96, 0.03, 0.01),
            (0.02, 0.98, 0.00),
            (0.01, 0.05, 0.94),
        ),
        tone_contrast=1.14,
        denoise_luma=0.12,
        denoise_chroma=0.28,
        sharpening_amount=0.22,
        sharpening_halo_radius=1.4,
        photo_format="jpeg",
        jpeg_quality=68,
        chroma_subsampling="4:2:0",
        video_container="3gp",
        video_codec="h263",
        video_bitrate_kbps=96.0,
        audio_codec="amr_nb",
        audio_bitrate_kbps=12.2,
    ),
    2: MobileCameraProfile(
        profile_id=2,
        key="nokia_n73",
        name="Nokia N73",
        era_year=2006,
        brand_family="nokia",
        sensor_width=2048,
        sensor_height=1536,
        video_width=352,
        video_height=288,
        video_fps=15.0,
        noise_poisson=0.022,
        noise_gaussian=0.015,
        well_saturation_knee=0.88,
        rolling_shutter_skew=0.16,
        bayer_demosaic_artifact=0.25,
        blur_radius=0.60,
        edge_softness=0.24,
        vignette_strength=0.08,
        chromatic_aberration=0.18,
        veiling_glare=0.08,
        awb_warmth=0.04,
        awb_green_magenta=-0.03,
        awb_instability=0.12,
        color_matrix=(
            (1.08, -0.05, -0.03),
            (-0.03, 1.06, -0.03),
            (-0.04, -0.05, 1.09),
        ),
        tone_contrast=1.18,
        denoise_luma=0.34,
        denoise_chroma=0.48,
        sharpening_amount=0.46,
        sharpening_halo_radius=1.6,
        photo_format="jpeg",
        jpeg_quality=78,
        chroma_subsampling="4:2:0",
        video_container="3gp",
        video_codec="mpeg4",
        video_bitrate_kbps=384.0,
        audio_codec="aac_lc",
        audio_bitrate_kbps=48.0,
    ),
    3: MobileCameraProfile(
        profile_id=3,
        key="nokia_n95",
        name="Nokia N95",
        era_year=2007,
        brand_family="nokia",
        sensor_width=2592,
        sensor_height=1944,
        video_width=640,
        video_height=480,
        video_fps=30.0,
        noise_poisson=0.018,
        noise_gaussian=0.011,
        well_saturation_knee=0.90,
        rolling_shutter_skew=0.14,
        bayer_demosaic_artifact=0.18,
        blur_radius=0.48,
        edge_softness=0.18,
        vignette_strength=0.06,
        chromatic_aberration=0.14,
        veiling_glare=0.06,
        awb_warmth=0.05,
        awb_green_magenta=-0.02,
        awb_instability=0.08,
        color_matrix=(
            (1.11, -0.06, -0.05),
            (-0.03, 1.08, -0.05),
            (-0.04, -0.06, 1.10),
        ),
        tone_contrast=1.16,
        denoise_luma=0.28,
        denoise_chroma=0.42,
        sharpening_amount=0.42,
        sharpening_halo_radius=1.4,
        photo_format="jpeg",
        jpeg_quality=82,
        chroma_subsampling="4:2:0",
        video_container="mp4",
        video_codec="mpeg4",
        video_bitrate_kbps=1200.0,
        audio_codec="aac_lc",
        audio_bitrate_kbps=96.0,
    ),
    4: MobileCameraProfile(
        profile_id=4,
        key="nokia_5800_3gp",
        name="Nokia 5800 XpressMusic",
        era_year=2008,
        brand_family="nokia",
        sensor_width=2048,
        sensor_height=1536,
        video_width=640,
        video_height=352,
        video_fps=25.0,
        noise_poisson=0.025,
        noise_gaussian=0.018,
        well_saturation_knee=0.86,
        rolling_shutter_skew=0.19,
        bayer_demosaic_artifact=0.28,
        blur_radius=0.68,
        edge_softness=0.28,
        vignette_strength=0.10,
        chromatic_aberration=0.20,
        veiling_glare=0.10,
        awb_warmth=0.02,
        awb_green_magenta=0.04,
        awb_instability=0.15,
        color_matrix=(
            (1.04, -0.02, -0.02),
            (-0.01, 1.03, -0.02),
            (-0.02, -0.02, 1.04),
        ),
        tone_contrast=1.15,
        denoise_luma=0.30,
        denoise_chroma=0.44,
        sharpening_amount=0.38,
        sharpening_halo_radius=1.5,
        photo_format="jpeg",
        jpeg_quality=76,
        chroma_subsampling="4:2:0",
        video_container="3gp",
        video_codec="h263",
        video_bitrate_kbps=320.0,
        audio_codec="aac_lc",
        audio_bitrate_kbps=48.0,
    ),
    5: MobileCameraProfile(
        profile_id=5,
        key="iphone_2007",
        name="iPhone 2007",
        era_year=2007,
        brand_family="iphone",
        sensor_width=1600,
        sensor_height=1200,
        video_width=320,
        video_height=240,
        video_fps=15.0,
        noise_poisson=0.026,
        noise_gaussian=0.016,
        well_saturation_knee=0.79,  # 早期 iPhone 动态范围受限，亮部容易提前过曝硬削波
        rolling_shutter_skew=0.18,
        bayer_demosaic_artifact=0.22,
        blur_radius=0.72,
        edge_softness=0.32,
        vignette_strength=0.14,
        chromatic_aberration=0.16,
        veiling_glare=0.14,
        awb_warmth=0.11,            # 早期 Apple ISP 自动白平衡偏暖自然肤色倾向
        awb_green_magenta=-0.02,
        awb_instability=0.10,
        color_matrix=(
            (1.07, -0.03, -0.04),
            (-0.02, 1.03, -0.01),
            (-0.04, -0.03, 0.96),
        ),
        tone_contrast=1.20,
        denoise_luma=0.18,          # 比诺基亚 N 系列涂抹感轻，保留更多细颗粒噪声但锐化克制
        denoise_chroma=0.32,
        sharpening_amount=0.24,
        sharpening_halo_radius=1.1,
        photo_format="jpeg",
        jpeg_quality=80,
        chroma_subsampling="4:2:0",
        video_container="none",
        video_codec="mjpeg",
        video_bitrate_kbps=280.0,
        audio_codec="pcm",
        audio_bitrate_kbps=64.0,
        supports_native_video=False,
    ),
    6: MobileCameraProfile(
        profile_id=6,
        key="iphone_3g",
        name="iPhone 3G",
        era_year=2008,
        brand_family="iphone",
        sensor_width=1600,
        sensor_height=1200,
        video_width=320,
        video_height=240,
        video_fps=15.0,
        noise_poisson=0.024,
        noise_gaussian=0.015,
        well_saturation_knee=0.81,
        rolling_shutter_skew=0.17,
        bayer_demosaic_artifact=0.20,
        blur_radius=0.66,
        edge_softness=0.28,
        vignette_strength=0.12,
        chromatic_aberration=0.15,
        veiling_glare=0.12,
        awb_warmth=0.08,
        awb_green_magenta=-0.01,
        awb_instability=0.09,
        color_matrix=(
            (1.06, -0.03, -0.03),
            (-0.02, 1.04, -0.02),
            (-0.03, -0.03, 0.98),
        ),
        tone_contrast=1.18,
        denoise_luma=0.20,
        denoise_chroma=0.35,
        sharpening_amount=0.26,
        sharpening_halo_radius=1.1,
        photo_format="jpeg",
        jpeg_quality=81,
        chroma_subsampling="4:2:0",
        video_container="none",
        video_codec="mjpeg",
        video_bitrate_kbps=320.0,
        audio_codec="pcm",
        audio_bitrate_kbps=64.0,
        supports_native_video=False,
    ),
    7: MobileCameraProfile(
        profile_id=7,
        key="iphone_3gs_video",
        name="iPhone 3GS",
        era_year=2009,
        brand_family="iphone",
        sensor_width=2048,
        sensor_height=1536,
        video_width=640,
        video_height=480,
        video_fps=30.0,
        noise_poisson=0.019,
        noise_gaussian=0.012,
        well_saturation_knee=0.86,
        rolling_shutter_skew=0.21,  # VGA 30fps 录像时明显的 CMOS 卷帘快门斜向摆动与连续 AE 阶跃
        bayer_demosaic_artifact=0.16,
        blur_radius=0.52,
        edge_softness=0.20,
        vignette_strength=0.09,
        chromatic_aberration=0.12,
        veiling_glare=0.09,
        awb_warmth=0.07,
        awb_green_magenta=-0.01,
        awb_instability=0.07,
        color_matrix=(
            (1.08, -0.04, -0.04),
            (-0.02, 1.05, -0.03),
            (-0.03, -0.04, 1.02),
        ),
        tone_contrast=1.15,
        denoise_luma=0.22,
        denoise_chroma=0.36,
        sharpening_amount=0.30,
        sharpening_halo_radius=1.2,
        photo_format="jpeg",
        jpeg_quality=84,
        chroma_subsampling="4:2:0",
        video_container="mov",
        video_codec="h264",
        video_bitrate_kbps=1500.0,
        audio_codec="aac_lc",
        audio_bitrate_kbps=64.0,
        supports_native_video=True,
    ),
    8: MobileCameraProfile(
        profile_id=8,
        key="iphone_4s",
        name="iPhone 4S",
        era_year=2011,
        brand_family="iphone",
        sensor_width=3264,
        sensor_height=2448,
        video_width=1920,
        video_height=1080,
        video_fps=30.0,
        noise_poisson=0.011,
        noise_gaussian=0.007,
        well_saturation_knee=0.93,
        rolling_shutter_skew=0.10,
        bayer_demosaic_artifact=0.09,
        blur_radius=0.34,
        edge_softness=0.11,
        vignette_strength=0.05,
        chromatic_aberration=0.07,
        veiling_glare=0.05,
        awb_warmth=0.04,
        awb_green_magenta=0.00,
        awb_instability=0.04,
        color_matrix=(
            (1.09, -0.05, -0.04),
            (-0.02, 1.07, -0.05),
            (-0.03, -0.05, 1.08),
        ),
        tone_contrast=1.12,
        denoise_luma=0.20,
        denoise_chroma=0.38,
        sharpening_amount=0.32,
        sharpening_halo_radius=1.0,
        photo_format="jpeg",
        jpeg_quality=88,
        chroma_subsampling="4:2:0",
        video_container="mov",
        video_codec="h264",
        video_bitrate_kbps=5000.0,
        audio_codec="aac_lc",
        audio_bitrate_kbps=128.0,
        supports_native_video=True,
    ),
}

MOBILE_PROFILE_BY_KEY: dict[str, MobileCameraProfile] = {
    p.key: p for p in MOBILE_CAMERA_PROFILES.values()
}
MOBILE_PROFILE_BY_KEY.update({
    "nokia_6600_like": MOBILE_CAMERA_PROFILES[1],
    "nokia_n73_like": MOBILE_CAMERA_PROFILES[2],
    "nokia_n95_like": MOBILE_CAMERA_PROFILES[3],
    "nokia_5800_like": MOBILE_CAMERA_PROFILES[4],
    "iphone_2007_like": MOBILE_CAMERA_PROFILES[5],
    "iphone_3g_like": MOBILE_CAMERA_PROFILES[6],
    "iphone_3gs_video_like": MOBILE_CAMERA_PROFILES[7],
    "iphone_4s_like": MOBILE_CAMERA_PROFILES[8],
})


def resolve_mobile_profile(profile: int | str | MobileCameraProfile) -> MobileCameraProfile:
    if isinstance(profile, MobileCameraProfile):
        return profile
    if isinstance(profile, str):
        k = profile.strip().lower()
        if k in MOBILE_PROFILE_BY_KEY:
            return MOBILE_PROFILE_BY_KEY[k]
        if k.isdigit() and int(k) in MOBILE_CAMERA_PROFILES:
            return MOBILE_CAMERA_PROFILES[int(k)]
    idx = int(np.clip(round(float(profile)), 1, 8))
    return MOBILE_CAMERA_PROFILES[idx]


@dataclass
class MobileISPOpticsSpec:
    """参数化早期手机 / iPhone 摄像成像配置，支持在机型预设基础上单独微调各物理环节。"""
    enabled: bool = False
    profile_id: int = 0          # 0=关闭, 1=Nokia 6600, 2=Nokia N73, 3=Nokia N95, 4=Nokia 5800, 5=iPhone 2007, 6=iPhone 3G, 7=iPhone 3GS, 8=iPhone 4S
    capture_mode: int = 0        # 0=静态照片管线 capture(), 1=动态视频管线 record()
    optics_blur_scale: float = 1.0
    vignette_scale: float = 1.0
    sensor_noise_scale: float = 1.0
    bayer_demosaic_scale: float = 1.0
    denoise_scale: float = 1.0
    sharpen_scale: float = 1.0
    jpeg_quality_override: int = 0   # 0=使用机型默认画质, >0=覆盖 JPEG/视频量化品质
    use_real_pillow_jpeg: bool = False

    # 物理建模：工频荧光灯频闪、低色温白平衡失效、JPEG 磨损衰减与夜景星芒 (旧手机照片实测标定)
    flicker_enabled: bool = False
    flicker_mains_hz: float = 50.0
    flicker_depth: float = 0.0
    flicker_color_drift: float = 0.65
    flicker_num_bands: float = 5.2

    tungsten_enabled: bool = False
    color_temp_k: float = 5500.0
    awb_mode: int = 0

    jpeg_decay_enabled: bool = False
    jpeg_decay_gens: float = 1.0
    jpeg_decay_quality: float = 75.0
    jpeg_decay_grid_shift: float = 2.0

    night_bloom: float = 0.0

    def is_active(self) -> bool:
        return bool(
            getattr(self, "enabled", False)
            and (
                getattr(self, "profile_id", 0) > 0
                or getattr(self, "jpeg_decay_enabled", False)
                or getattr(self, "jpeg_decay_gens", 1.0) > 1.5
                or getattr(self, "flicker_enabled", False)
                or getattr(self, "flicker_depth", 0.0) > 0.01
                or getattr(self, "tungsten_enabled", False)
                or getattr(self, "awb_mode", 0) > 0
                or getattr(self, "night_bloom", 0.0) > 0.01
            )
        )

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "MobileISPOpticsSpec":
        prof = int(round(float(cfg.get("ldme_mobile_profile", 0.0))))
        decay_en = float(cfg.get("ldme_jpeg_decay_enabled", 0.0)) > 0.5 or float(cfg.get("ldme_jpeg_decay_gens", 1.0)) > 1.5
        flk_en = float(cfg.get("ldme_flicker_enabled", 0.0)) > 0.5 or float(cfg.get("ldme_flicker_depth", 0.0)) > 0.01
        tung_en = float(cfg.get("ldme_tungsten_enabled", 0.0)) > 0.5 or float(cfg.get("ldme_awb_mode", 0.0)) > 0.5
        bloom_en = float(cfg.get("ldme_night_bloom", 0.0)) > 0.01

        en = float(cfg.get("ldme_mobile_enabled", 0.0)) > 0.5 or (prof > 0) or decay_en or flk_en or tung_en or bloom_en
        if en and prof <= 0:
            prof = 2  # 默认 Nokia N73
        return cls(
            enabled=en,
            profile_id=int(np.clip(prof, 0, 8)),
            capture_mode=int(np.clip(round(float(cfg.get("ldme_capture_mode", 0.0))), 0, 1)),
            optics_blur_scale=float(np.clip(cfg.get("ldme_optics_blur", 1.0), 0.0, 2.5)),
            vignette_scale=float(np.clip(cfg.get("ldme_vignette", 1.0), 0.0, 2.5)),
            sensor_noise_scale=float(np.clip(cfg.get("ldme_sensor_noise", 1.0), 0.0, 3.0)),
            bayer_demosaic_scale=float(np.clip(cfg.get("ldme_bayer_demosaic", 1.0), 0.0, 2.5)),
            denoise_scale=float(np.clip(cfg.get("ldme_isp_denoise", 1.0), 0.0, 2.5)),
            sharpen_scale=float(np.clip(cfg.get("ldme_isp_sharpen", 1.0), 0.0, 2.5)),
            jpeg_quality_override=int(np.clip(round(float(cfg.get("ldme_jpeg_quality", 0.0))), 0, 100)),
            use_real_pillow_jpeg=bool(float(cfg.get("ldme_use_real_jpeg", 0.0)) > 0.5),

            flicker_enabled=bool(flk_en),
            flicker_mains_hz=float(cfg.get("ldme_flicker_mains_hz", 50.0)),
            flicker_depth=float(np.clip(cfg.get("ldme_flicker_depth", 0.0), 0.0, 1.0)),
            flicker_color_drift=float(np.clip(cfg.get("ldme_flicker_color_drift", 0.65), 0.0, 1.0)),
            flicker_num_bands=float(np.clip(cfg.get("ldme_flicker_num_bands", 5.2), 0.5, 20.0)),

            tungsten_enabled=bool(tung_en),
            color_temp_k=float(np.clip(cfg.get("ldme_color_temp_k", 5500.0), 1800.0, 9500.0)),
            awb_mode=int(np.clip(round(float(cfg.get("ldme_awb_mode", 0.0))), 0, 3)),

            jpeg_decay_enabled=bool(decay_en),
            jpeg_decay_gens=float(np.clip(cfg.get("ldme_jpeg_decay_gens", 1.0), 1.0, 10.0)),
            jpeg_decay_quality=float(np.clip(cfg.get("ldme_jpeg_decay_quality", 75.0), 5.0, 95.0)),
            jpeg_decay_grid_shift=float(np.clip(cfg.get("ldme_jpeg_decay_grid_shift", 2.0), 0.0, 7.0)),

            night_bloom=float(np.clip(cfg.get("ldme_night_bloom", 0.0), 0.0, 2.5)),
        )


# ---------------------------------------------------------------------------
# 六步手机摄影与录像物理管线（A4）
# ---------------------------------------------------------------------------

def _stage1_mobile_optics(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    spec: MobileISPOpticsSpec,
) -> np.ndarray:
    """
    1. 光学成像 (Optics):
       微型手机镜头中心解析力模糊、离轴边缘场曲软化、cos^4(theta) 机械暗角、
       横向色差以及树脂镜片逆光杂散光 (Veiling Glare)。
    """
    out = np.clip(rgb.astype(np.float32), 0.0, 1.5).copy()
    h, w = out.shape[:2]
    scale_w = max(0.5, w / 640.0)

    blur_r = float(prof.blur_radius * spec.optics_blur_scale)
    if blur_r > 1e-3:
        center_blur = np.stack([gaussian_filter(out[..., c], sigma=max(0.25, blur_r * 0.65 * scale_w)) for c in range(3)], axis=-1)
    else:
        center_blur = out

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny

    # 离轴边缘模糊（微型塑料非球面镜片边缘像散）
    edge_soft = float(prof.edge_softness * spec.optics_blur_scale)
    if edge_soft > 1e-3:
        edge_blur = np.stack([gaussian_filter(center_blur[..., c], sigma=max(0.6, (blur_r + 1.8 * edge_soft) * scale_w)) for c in range(3)], axis=-1)
        edge_w = np.clip((r2 - 0.18) / 1.15, 0.0, 1.0) ** 1.35
        out = center_blur * (1.0 - edge_w[..., None]) + edge_blur * edge_w[..., None]
    else:
        out = center_blur

    # 微型镜头横向色散（红蓝通道边缘微位移）
    ca = float(prof.chromatic_aberration * spec.optics_blur_scale)
    if ca > 1e-3:
        shift_x = np.clip(np.round(xx - nx * 1.6 * ca * scale_w).astype(np.int32), 0, w - 1)
        shift_b = np.clip(np.round(xx + nx * 1.6 * ca * scale_w).astype(np.int32), 0, w - 1)
        row_idx = np.arange(h, dtype=np.int32)[:, None]
        out[..., 0] = out[row_idx, shift_x, 0]
        out[..., 2] = out[row_idx, shift_b, 2]

    # 树脂镜片逆光杂散光（Veiling Glare）
    if prof.veiling_glare > 1e-3:
        lum = out @ LUMA_W
        hi = np.clip((lum - 0.68) / 0.32, 0.0, 1.0)
        glare = gaussian_filter(hi, sigma=max(4.0, 14.0 * scale_w))[..., None]
        out = out + prof.veiling_glare * 0.28 * glare

    # 镜头暗角
    vig = float(prof.vignette_strength * spec.vignette_scale)
    if vig > 1e-3:
        v_mask = np.clip(1.0 - vig * 0.55 * r2, 0.25, 1.0)[..., None]
        out = out * v_mask

    return np.clip(out, 0.0, 1.2).astype(np.float32)


def _stage2_sensor_sampling(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    spec: MobileISPOpticsSpec,
    is_video: bool,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    2. 传感器采样 (Sensor Sampling):
       按真实传感器/视频工作模式的宽高比与有效采样密度进行空间采样，
       模拟自动曝光不稳定性 / 视频连续 AE 阶跃、满井电荷饱和硬削波（Highlight Clipping）、
       CMOS 卷帘快门斜向摆动（视频模式）以及信号相关的泊松光子散粒噪声 + 高斯读出噪声。
    """
    h, w = rgb.shape[:2]
    # 根据传感器/录像分辨率计算工作栅格尺寸（避免不必要的百万像素全尺寸缓慢重采样，同时精确还原空间采样上限）
    target_w = prof.video_width if is_video else min(w, prof.sensor_width)
    target_h = prof.video_height if is_video else min(h, prof.sensor_height)
    work_w = max(64, min(w, int(target_w)))
    work_h = max(48, min(h, int(target_h)))

    sampled = _resample_to(rgb, work_h, work_w)

    # 自动曝光波动（早期手机与视频连续 AE 阶跃）
    if prof.awb_instability > 1e-3 or is_video:
        phase = float(frame_index) * (0.28 if is_video else 0.12)
        ae_step = np.round(np.sin(phase) * 2.0) / 2.0 if is_video else np.sin(phase)
        ae_gain = 1.0 + (0.09 * prof.awb_instability) * float(ae_step)
        sampled = sampled * ae_gain

    # 视频模式下的 CMOS 卷帘快门行间相位延迟 (Rolling Shutter Skew)
    if is_video and prof.rolling_shutter_skew > 1e-3:
        t_phase = float(frame_index) * 0.35
        y_norm = np.linspace(-1.0, 1.0, work_h, dtype=np.float32)
        row_dx = (prof.rolling_shutter_skew * 2.8 * np.sin(t_phase)) * y_norm
        sampled = horizontal_shift(sampled, row_dx)

    # 泊松散粒噪声 + 高斯读出噪声
    n_scale = float(spec.sensor_noise_scale)
    if n_scale > 1e-4 and (prof.noise_poisson > 1e-5 or prof.noise_gaussian > 1e-5):
        rng = np.random.default_rng((int(seed) * 1664525 + int(frame_index) * 1013904223 + prof.profile_id * 97) & 0xFFFFFFFF)
        shot_std = np.sqrt(np.clip(sampled, 0.002, 1.0)) * (prof.noise_poisson * n_scale)
        read_std = prof.noise_gaussian * n_scale
        noise = rng.normal(0.0, 1.0, size=sampled.shape).astype(np.float32) * (shot_std + read_std)
        sampled = sampled + noise

    # 传感器势井饱和与高光保护膝点（早期 iPhone / VGA 手机亮部容易发生通道相继饱和与高光平顶削波）
    knee = float(np.clip(prof.well_saturation_knee, 0.65, 0.99))
    over = np.maximum(0.0, sampled - knee)
    sampled = np.where(sampled <= knee, sampled, knee + (1.0 - knee) * np.tanh(over / max(1e-3, 1.0 - knee) * 1.8))
    return np.clip(sampled, 0.0, 1.0).astype(np.float32), (work_h, work_w)


def _stage3_bayer_demosaic(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    spec: MobileISPOpticsSpec,
) -> np.ndarray:
    """
    3. 去马赛克 (Bayer CFA Demosaic):
       对 RGGB 拜耳阵列进行单通道马赛克抽样并用早期手机低复杂度双线性插值重建，
       在斜向高对比度边缘自然产生拜耳拉链伪影 (Zippering) 与伪彩摩尔纹，而非人为贴纹理。
    """
    strength = float(np.clip(prof.bayer_demosaic_artifact * spec.bayer_demosaic_scale, 0.0, 1.0))
    if strength <= 1e-3:
        return rgb

    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return rgb

    # 构建 RGGB 拜耳采样网格：
    # 偶行偶列 = R，偶行奇列 + 奇行偶列 = G，奇行奇列 = B
    cfa = np.zeros_like(rgb)
    cfa[0::2, 0::2, 0] = rgb[0::2, 0::2, 0]
    cfa[0::2, 1::2, 1] = rgb[0::2, 1::2, 1]
    cfa[1::2, 0::2, 1] = rgb[1::2, 0::2, 1]
    cfa[1::2, 1::2, 2] = rgb[1::2, 1::2, 2]

    k_rb = np.array([[0.25, 0.50, 0.25],
                     [0.50, 1.00, 0.50],
                     [0.25, 0.50, 0.25]], dtype=np.float32)
    k_g = np.array([[0.00, 0.25, 0.00],
                    [0.25, 1.00, 0.25],
                    [0.00, 0.25, 0.00]], dtype=np.float32)

    r_rec = convolve(cfa[..., 0], k_rb, mode="nearest")
    g_rec = convolve(cfa[..., 1], k_g, mode="nearest")
    b_rec = convolve(cfa[..., 2], k_rb, mode="nearest")
    demosaiced = np.stack([r_rec, g_rec, b_rec], axis=-1)

    return np.clip(rgb * (1.0 - strength) + demosaiced * strength, 0.0, 1.0).astype(np.float32)


def _stage4_awb_and_color_matrix(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    frame_index: int = 0,
) -> np.ndarray:
    """
    4. 白平衡与色彩 (AWB & Color Matrix):
       灰世界自动白平衡估计与机型固有偏移（如 Nokia 6600 室内荧光灯偏青绿/品红摆动、
       初代 iPhone 偏暖肤色还原、Nokia N73/N95 高饱和蔡司色彩矩阵）及 ISP S 型色调曲线。
    """
    out = rgb.copy()
    # 灰世界白平衡残差 + 机型色温倾向
    mean_rgb = np.maximum(out.mean(axis=(0, 1)), 0.08)
    lum_mean = float(mean_rgb @ LUMA_W)
    gw_gain = np.clip(lum_mean / mean_rgb, 0.82, 1.22)
    # 早期手机 AWB 并不能完全校正色温，保留一定比例的环境偏色并叠加机型固有白平衡偏移
    drift = prof.awb_instability * 0.08 * np.cos(float(frame_index) * 0.19 + prof.profile_id)
    r_gain = (0.65 + 0.35 * gw_gain[0]) * (1.0 + prof.awb_warmth + drift)
    g_gain = (0.65 + 0.35 * gw_gain[1]) * (1.0 + prof.awb_green_magenta * 0.5)
    b_gain = (0.65 + 0.35 * gw_gain[2]) * (1.0 - prof.awb_warmth * 0.85 - drift)
    out[..., 0] *= r_gain
    out[..., 1] *= g_gain
    out[..., 2] *= b_gain

    # 3x3 色彩校正矩阵 (CCM)
    ccm = np.asarray(prof.color_matrix, dtype=np.float32)
    out = np.einsum("hwc,dc->hwd", out, ccm)
    out = np.clip(out, 0.0, 1.0)

    # ISP 色调曲线与对比度重映射
    c = float(prof.tone_contrast)
    if abs(c - 1.0) > 1e-3:
        centered = out - 0.5
        out = np.clip(0.5 + centered * c, 0.0, 1.0)
    return out.astype(np.float32)


def _stage5_denoise_and_sharpen(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    spec: MobileISPOpticsSpec,
) -> np.ndarray:
    """
    5. 降噪与锐化 (Denoise & Sharpening):
       - 色度强低通涂抹 + 亮度保边降噪（重现 Nokia N73/N95 暗部及平缓区域的塑料涂抹感）
       - USM 孔径锐化与高反差边缘过冲振铃亮边（Sharpening Halo / Overshoot）
    """
    y = rgb @ LUMA_W
    cb = rgb[..., 2] - y
    cr = rgb[..., 0] - y

    dn_luma = float(prof.denoise_luma * spec.denoise_scale)
    dn_chroma = float(prof.denoise_chroma * spec.denoise_scale)

    # 色度空间低通降噪（抹除彩色噪斑但造成细小色彩边缘渗色）
    if dn_chroma > 1e-3:
        sigma_c = 0.8 + 2.6 * dn_chroma
        cb = gaussian_filter(cb, sigma=sigma_c)
        cr = gaussian_filter(cr, sigma=sigma_c)

    # 亮度保边软阈值双边近似降噪（抹除低幅值纹理细节，保留强边缘）
    if dn_luma > 1e-3:
        y_smooth = gaussian_filter(y, sigma=0.75 + 1.6 * dn_luma)
        detail = y - y_smooth
        thresh = 0.025 + 0.065 * dn_luma
        edge_preserve = np.clip((np.abs(detail) - thresh) / max(1e-4, thresh * 1.5), 0.0, 1.0)
        y = y_smooth + detail * edge_preserve

    # USM 锐化与过冲亮边光晕 (Halo Ringing)
    sh_amt = float(prof.sharpening_amount * spec.sharpen_scale)
    if sh_amt > 1e-3:
        halo_r = max(0.7, float(prof.sharpening_halo_radius))
        y_blur = gaussian_filter(y, sigma=halo_r)
        hp = y - y_blur
        # 早期手机 ISP 锐化器带有限幅过冲，在强阶跃边缘形成明显白边/黑边光晕
        overshoot = np.clip(hp * (1.45 * sh_amt), -0.18, 0.22)
        y = np.clip(y + overshoot, 0.0, 1.0)

    r = y + cr
    b = y + cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def _stage6_resize_and_encode(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    spec: MobileISPOpticsSpec,
    out_h: int,
    out_w: int,
    is_video: bool,
    frame_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    6. 尺寸与编码 (Resize & Encoding):
       - 静态照片路径 (`capture`)：执行 JPEG 4:2:0 色度子采样与 8×8 DCT 块量化（或真实 Pillow JPEG 编解码），生成 EXIF 元数据；
       - 动态视频路径 (`record`)：区分 3GP (`H.263` / `MPEG-4 Part 2`) 与 MOV/MP4 (`H.264`) 的帧率抽样、宏块量化及 P 帧运动预测伪影。
    """
    quality = int(spec.jpeg_quality_override) if spec.jpeg_quality_override > 0 else int(prof.jpeg_quality)
    quality = int(np.clip(quality, 10, 98))

    if not is_video:
        # 静态照片 JPEG 编码路径
        if spec.use_real_pillow_jpeg:
            try:
                from PIL import Image
                u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(u8, mode="RGB")
                bio = io.BytesIO()
                subsampling_flag = 2 if prof.chroma_subsampling == "4:2:0" else 1
                pil_img.save(bio, format="JPEG", quality=quality, subsampling=subsampling_flag)
                bio.seek(0)
                dec_img = Image.open(bio).convert("RGB")
                coded = np.asarray(dec_img, dtype=np.float32) / 255.0
            except Exception:
                coded = _fast_jpeg_420_quantize(rgb, quality=quality, subsampling=prof.chroma_subsampling)
        else:
            coded = _fast_jpeg_420_quantize(rgb, quality=quality, subsampling=prof.chroma_subsampling)

        # 早期优酷土豆与网络论坛搬运多代 JPEG 磨损衰减 (JPEG Wear / Generation Loss)
        if getattr(spec, "jpeg_decay_enabled", False) or getattr(spec, "jpeg_decay_gens", 1) > 1:
            decay_q = int(getattr(spec, "jpeg_decay_quality", 65.0) if getattr(spec, "jpeg_decay_quality", 65.0) > 0 else quality)
            coded = apply_jpeg_generation_loss(
                coded,
                generations=max(1, int(getattr(spec, "jpeg_decay_gens", 1))),
                quality=decay_q,
                grid_shift=int(getattr(spec, "jpeg_decay_grid_shift", False)),
                chroma_subsample=True,
            )

        exif_meta = {
            "Make": "Nokia" if prof.brand_family == "nokia" else "Apple",
            "Model": prof.name,
            "Year": prof.era_year,
            "Software": f"{prof.name} Firmware ISP",
            "Width": prof.sensor_width,
            "Height": prof.sensor_height,
            "ColorSpace": "sRGB",
            "YCbCrSubSampling": prof.chroma_subsampling,
            "JPEGQuality": quality,
        }
        out = _resample_to(coded, out_h, out_w)
        return out, {
            "path_mode": "photo_capture",
            "container": "JPEG",
            "codec": "jpeg",
            "pixel_format": prof.chroma_subsampling,
            "encoded_size": f"{prof.sensor_width}x{prof.sensor_height}",
            "jpeg_quality": quality,
            "exif": exif_meta,
        }

    # 动态视频录制路径 (`record`)：注意 3GP 是容器，内部可承载 H.263 或 MPEG-4 Part 2；早期 iPhone 3GS/4S 为 MOV H.264
    v_codec = prof.video_codec
    v_kbps = float(prof.video_bitrate_kbps)
    # 映射视频码率到宏块量化步长
    q_step = float(np.clip(0.085 * ((350.0 / max(48.0, v_kbps)) ** 0.55), 0.010, 0.14))
    gop_len = 12 if v_codec == "h263" else (15 if v_codec == "mpeg4" else 25)
    gop_pos = int(frame_index) % gop_len
    # P 帧预测误差随 GOP 位置累积
    p_frame_factor = 1.0 + (0.38 if v_codec == "h263" else 0.22) * (gop_pos / max(1.0, float(gop_len - 1)))
    coded = _fast_video_macroblock_quantize(
        rgb,
        codec=v_codec,
        q_step=q_step * p_frame_factor,
        frame_index=frame_index,
    )
    if getattr(spec, "jpeg_decay_enabled", False) or getattr(spec, "jpeg_decay_gens", 1) > 1:
        decay_q = int(getattr(spec, "jpeg_decay_quality", 65.0) if getattr(spec, "jpeg_decay_quality", 65.0) > 0 else 60)
        coded = apply_jpeg_generation_loss(
            coded,
            generations=max(1, int(getattr(spec, "jpeg_decay_gens", 1))),
            quality=decay_q,
            grid_shift=int(getattr(spec, "jpeg_decay_grid_shift", False)),
            chroma_subsample=True,
        )
    out = _resample_to(coded, out_h, out_w)
    return out, {
        "path_mode": "video_record",
        "container": prof.video_container.upper(),
        "codec": v_codec,
        "pixel_format": "4:2:0",
        "encoded_size": f"{prof.video_width}x{prof.video_height}",
        "video_fps": prof.video_fps,
        "video_bitrate_kbps": v_kbps,
        "audio_codec": prof.audio_codec,
        "audio_bitrate_kbps": prof.audio_bitrate_kbps,
        "supports_native_video": prof.supports_native_video,
    }


def _fast_jpeg_420_quantize(
    rgb: np.ndarray,
    quality: int = 78,
    subsampling: str = "4:2:0",
) -> np.ndarray:
    """快速确定性 8×8 块量化与 4:2:0 色度子采样模型。"""
    h, w = rgb.shape[:2]
    y = rgb @ LUMA_W
    cb = rgb[..., 2] - y
    cr = rgb[..., 0] - y

    cs_y = 2 if subsampling == "4:2:0" else 1
    cs_x = 2
    if h >= 4 and w >= 4:
        cb_s = _resample_to(cb[..., None], max(2, h // cs_y), max(2, w // cs_x))[..., 0]
        cr_s = _resample_to(cr[..., None], max(2, h // cs_y), max(2, w // cs_x))[..., 0]
        cb = _resample_to(cb_s[..., None], h, w)[..., 0]
        cr = _resample_to(cr_s[..., None], h, w)[..., 0]

    q_norm = float(np.clip((100 - quality) / 85.0, 0.02, 1.0))
    q_step = 0.006 + 0.075 * (q_norm ** 1.35)
    gh, gw = max(2, h // 8), max(2, w // 8)
    y_dc = _resample_to(_resample_to(y[..., None], gh, gw), h, w)[..., 0]
    y_ac = y - y_dc
    y_dc_q = np.round(y_dc / q_step) * q_step
    y_ac_q = np.round(y_ac / (q_step * 1.5)) * (q_step * 1.5)
    y_out = (0.45 * q_norm) * y_dc_q + (1.0 - 0.45 * q_norm) * (y_dc + y_ac_q)

    r = y_out + cr
    b = y_out + cb
    g = (y_out - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def _fast_video_macroblock_quantize(
    rgb: np.ndarray,
    codec: str = "h263",
    q_step: float = 0.04,
    frame_index: int = 0,
) -> np.ndarray:
    """区分 H.263 (16×16/8×8 无环路去块强网格)、MPEG-4 Part 2 与早期 H.264 (含轻量环路去块滤波) 的视频编码损伤。"""
    h, w = rgb.shape[:2]
    y = rgb @ LUMA_W
    cb = rgb[..., 2] - y
    cr = rgb[..., 0] - y

    if h >= 4 and w >= 4:
        cb = _resample_to(_resample_to(cb[..., None], max(2, h // 2), max(2, w // 2)), h, w)[..., 0]
        cr = _resample_to(_resample_to(cr[..., None], max(2, h // 2), max(2, w // 2)), h, w)[..., 0]

    bs = 16 if codec == "h263" else 8
    gh, gw = max(2, h // bs), max(2, w // bs)
    y_dc = _resample_to(_resample_to(y[..., None], gh, gw), h, w)[..., 0]
    y_ac = y - y_dc
    y_dc_q = np.round(y_dc / max(1e-4, q_step)) * q_step
    y_ac_q = np.round(y_ac / max(1e-4, q_step * 1.7)) * (q_step * 1.7)
    y_out = 0.62 * y_dc_q + 0.38 * (y_dc + y_ac_q)

    if codec == "h264":
        # H.264 帧内环路去块滤波平滑块边界但略微柔化微纹理
        y_out = 0.72 * y_out + 0.28 * gaussian_filter(y_out, sigma=0.85)

    r = y_out + cr
    b = y_out + cb
    g = (y_out - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


class MobileCameraEngine:
    """
    统一早期手机与早期 iPhone 摄像引擎接口（B1）：
      - `camera.capture(scene, profile="iphone_2007")` -> 静态照片成像管线
      - `camera.record(scene, profile="iphone_3gs_video")` -> 动态视频录制管线
    """

    @staticmethod
    def capture(
        scene: np.ndarray,
        profile: int | str | MobileCameraProfile = "nokia_n73",
        spec: MobileISPOpticsSpec | None = None,
        frame_index: int = 0,
        seed: int = 20240101,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prof = resolve_mobile_profile(profile)
        eff_spec = spec or MobileISPOpticsSpec(enabled=True, profile_id=prof.profile_id, capture_mode=0)
        return run_mobile_isp_pipeline(scene, prof, eff_spec, is_video=False, frame_index=frame_index, seed=seed)

    @staticmethod
    def record(
        scene: np.ndarray,
        profile: int | str | MobileCameraProfile = "iphone_3gs_video",
        spec: MobileISPOpticsSpec | None = None,
        frame_index: int = 0,
        seed: int = 20240101,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        prof = resolve_mobile_profile(profile)
        eff_spec = spec or MobileISPOpticsSpec(enabled=True, profile_id=prof.profile_id, capture_mode=1)
        return run_mobile_isp_pipeline(scene, prof, eff_spec, is_video=True, frame_index=frame_index, seed=seed)


def run_mobile_isp_pipeline(
    rgb: np.ndarray,
    prof: MobileCameraProfile,
    spec: MobileISPOpticsSpec,
    is_video: bool = False,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行完整 6 步手机摄影/视频成像管线并返回像素与可追溯元数据。"""
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = src.shape[:2]

    # Step 1: 光学成像
    s1 = _stage1_mobile_optics(src, prof, spec)
    if getattr(spec, "night_bloom", 0.0) > 1e-4:
        s1 = apply_digicam_night_starburst_bloom(
            s1,
            bloom_amount=getattr(spec, "night_bloom", 0.0),
            streak_amount=getattr(spec, "night_bloom", 0.0) * 0.75,
            threshold=0.72,
        )

    # Step 2: 传感器采样、噪声与饱和
    s2, work_hw = _stage2_sensor_sampling(s1, prof, spec, is_video=is_video, frame_index=frame_index, seed=seed)
    if getattr(spec, "flicker_enabled", False) or getattr(spec, "flicker_depth", 0.28) > 1e-4:
        s2 = apply_cmos_rolling_shutter_flicker(
            s2,
            mains_hz=getattr(spec, "flicker_mains_hz", 50.0),
            depth=getattr(spec, "flicker_depth", 0.28) if getattr(spec, "flicker_depth", 0.28) > 1e-4 else (0.45 if getattr(spec, "flicker_enabled", False) else 0.0),
            num_bands=getattr(spec, "flicker_num_bands", 3.2),
            color_drift=getattr(spec, "flicker_color_drift", 0.22),
            phase=float(frame_index) * (0.35 if is_video else 0.0),
        )

    # Step 3: 拜耳去马赛克
    s3 = _stage3_bayer_demosaic(s2, prof, spec)

    # Step 4: 自动白平衡与色彩矩阵
    s4 = _stage4_awb_and_color_matrix(s3, prof, frame_index=frame_index)
    if getattr(spec, "tungsten_enabled", False) or getattr(spec, "awb_mode", 1) > 0 or abs(getattr(spec, "color_temp_k", 2700.0) - 5500.0) > 100.0:
        s4 = apply_tungsten_awb_failure(
            s4,
            color_temp_k=getattr(spec, "color_temp_k", 2700.0),
            awb_mode=getattr(spec, "awb_mode", 1) if getattr(spec, "awb_mode", 1) > 0 else (1 if getattr(spec, "tungsten_enabled", False) else 0),
            seed=seed,
        )

    # Step 5: 双边降噪涂抹与 USM 锐化光晕
    s5 = _stage5_denoise_and_sharpen(s4, prof, spec)

    # Step 6: 尺寸与 JPEG / 3GP / MOV 编码
    out_rgb, enc_info = _stage6_resize_and_encode(
        s5, prof, spec, out_h=h, out_w=w, is_video=is_video, frame_index=frame_index
    )

    info = {
        "active": True,
        "profile_id": prof.profile_id,
        "profile_key": prof.key,
        "device_name": prof.name,
        "era_year": prof.era_year,
        "brand_family": prof.brand_family,
        "pipeline_stages": [
            "1.optics",
            "2.sensor_poisson_gaussian",
            "3.bayer_demosaic",
            "4.awb_color_matrix",
            "5.denoise_and_usm_sharpen",
            "6.resize_and_encode",
        ],
        "working_resolution": f"{work_hw[1]}x{work_hw[0]}",
        "encoding": enc_info,
    }
    return out_rgb, info


class MobileCameraISPStage(Stage):
    """独立节点图阶段 `mobile_camera_isp`：02 · 早期手机与早期 iPhone 摄像设备成像系统。"""
    name = "mobile_camera_isp"
    layers = ("camera",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("camera")
        self.spec = MobileISPOpticsSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.profile_id <= 0:
                self.spec.profile_id = 2

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        prof = resolve_mobile_profile(self.spec.profile_id)
        out_rgb, info = run_mobile_isp_pipeline(
            rgb_in,
            prof,
            self.spec,
            is_video=bool(self.spec.capture_mode == 1),
            frame_index=int(buf.index),
            seed=int(self.ctx.seed) + 307,
        )
        meta = dict(buf.meta)
        meta["mobile_camera_isp"] = info
        meta["tap_camera_rgb"] = out_rgb
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

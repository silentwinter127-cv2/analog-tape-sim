"""
古早数字媒介系统：从优酷土豆到诺基亚与早期 iPhone (`atsim/codecs/legacy_digital_media.py`)
==========================================================================================

完整实现《古早数字媒介系统：从优酷土豆到诺基亚与早期 iPhone》（Legacy Digital Media Engine, LDME）
全部六个子系统与三组标准基准测试，并与独立手机成像模块 `atsim/codecs/mobile_isp_optics.py` 协同工作：

- **01 · 统一数据接口 (`MediaState` & `TransferEvent`)**：
  记录图像/视频帧像素、色彩空间、位深、分辨率、帧率、像素格式 (`4:4:4`/`4:2:2`/`4:2:0`)、编码器、码率、EXIF/容器元数据与完整传播历史链 `history`。
- **02 · 摄像设备成像（解耦于 `atsim/codecs/mobile_isp_optics.py`）**：
  覆盖 Nokia 6600、Nokia N73、Nokia N95、Nokia 5800 XpressMusic 与 iPhone 2007、iPhone 3G、iPhone 3GS、iPhone 4S 的六步物理成像与 `capture()` / `record()` 双路径。
- **03 · 文件与传输 (`Subsystem D`)**：
  严格区分无损文件复制（`蓝牙传输`、`USB/电脑导入`，仅根据策略保留或变更文件系统时间戳/OBEX属性，绝不凭空添加图像压缩）与有损应用处理（`彩信/早期社交软件发送` 的强制缩放、重编码与 EXIF 剥离；`电脑剪辑软件导出 + 视频网站多代转码` $V_0 \\to V_1 \\to V_2 \\to V_3$）。
- **04 · 平台处理 (`Subsystem C`：优酷 / 土豆 2006–2015 流媒体转码管线)**：
  - 阶段一（2006–2009 早期 Flash 视频）：FLV 容器、`flv1` (Sorenson Spark / H.263)、`320×240` @ `25fps`、`350 kbps` 视频 + `48 kbps` MP3；
  - 阶段二（2009–2012 多档转码与 H.264 普及）：`480×360` 标清档 (`h264`, `700 kbps`, `64 kbps` AAC, `25fps`) 与 `320×240` 流畅档；
  - 阶段三（2012–2015 移动互联网与多终端适配）：`640×360` / `854×480` / `1280×720` H.264 MP4/HLS 多屏自适应；
  - 严格区分**空间退化**（真实缩放、4:2:0 色度子采样、块效应与振铃）与**时间退化**（低帧率抽样、GOP 关键帧呼吸、运动区域预测误差累积），并区分「原始低清直传」与「高清源被平台降采样转码为低清」。支持调用真实 FFmpeg 进行编解码实验。
- **05 · 播放与显示 (`Subsystem E`：早期 LCD、Flash 播放器缩放与缓冲系统)**：
  严格区分 **LCD** 与 **CRT** 物理机制：建模早期手机 16-bit RGB565 TFT 液晶与 2007 桌面 6-bit TN 液晶面板（响应时间拖影、6-bit+FRC 灰阶断层、视角伽马反转、RGB 竖条子像素晶格、CCFL 冷白背光），以及 Flash 播放器非整数倍窗口缩放、4:3 拉伸 16:9 与网络缓冲区耗尽卡顿状态机。
- **06 · 三组标准基准测试 (`Subsystem F`)**：
  - `测试一：同一场景，不同手机`
  - `测试二：同一视频，不同转码链`
  - `测试三：同一文件，不同播放设备`
"""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass
from ..stages.display import _resample_to
from .mobile_isp_optics import (
    DCT_8X8_MATRIX,
    JPEG_LUMA_QUANT_TABLE,
    JPEG_CHROMA_QUANT_TABLE,
    get_scaled_quant_table,
    apply_jpeg_dct_channel_pass,
    apply_jpeg_generation_loss,
    apply_cmos_rolling_shutter_flicker,
    apply_tungsten_awb_failure,
    apply_digicam_night_starburst_bloom,
    MOBILE_CAMERA_PROFILES,
    MobileCameraEngine,
    MobileCameraProfile,
    MobileISPOpticsSpec,
    resolve_mobile_profile,
    run_mobile_isp_pipeline,
)
from .multiview_wall import _draw_bitmap_text

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _draw_text(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    text: str,
    color: tuple[float, float, float],
    scale: int = 1,
) -> None:
    _draw_bitmap_text(canvas, str(text), int(x0), int(y0), color, scale=int(scale))


# ---------------------------------------------------------------------------
# 一、统一媒体描述结构与传输事件 (`MediaState` & `TransferEvent`)
# ---------------------------------------------------------------------------

@dataclass
class TransferEvent:
    """记录文件在设备、编辑软件与平台之间每一次传输或重编码的溯源事件。"""
    source_device: str
    destination: str
    method: str                 # "bluetooth_copy" | "usb_copy" | "mms_send" | "social_im" | "desktop_nle_export" | "platform_transcode"
    pixel_transform: str | None = None
    reencoded: bool = False
    metadata_policy: str = "preserve"  # "preserve" | "filesystem_retime" | "strip_exif" | "container_rewrap"
    codec_after: str | None = None
    bitrate_kbps_after: float | None = None
    resolution_after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_device": self.source_device,
            "destination": self.destination,
            "method": self.method,
            "pixel_transform": self.pixel_transform,
            "reencoded": self.reencoded,
            "metadata_policy": self.metadata_policy,
            "codec_after": self.codec_after,
            "bitrate_kbps_after": self.bitrate_kbps_after,
            "resolution_after": self.resolution_after,
        }


@dataclass
class MediaState:
    """
    LDME 统一媒体描述结构（第一节）：
    每个子系统节点接收并输出统一的 `MediaState`，完整保留像素、编码格式、元数据与传播历史。
    """
    pixels: np.ndarray
    color_space: str = "RGB"            # "RGB" | "YUV_BT601" | "YUV_BT709"
    bit_depth: int = 8                  # 6 | 8 | 10
    width: int = 640
    height: int = 480
    frame_rate: float = 25.0
    pixel_format: str = "4:2:0"         # "4:4:4" | "4:2:2" | "4:2:0"
    codec: str = "raw"                  # "jpeg" | "h263" | "mpeg4" | "flv1" | "h264"
    bitrate: float = 700.0              # kbps
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_rgb(
        cls,
        rgb: np.ndarray,
        fps: float = 25.0,
        codec: str = "raw",
        bitrate: float = 1500.0,
        pixel_format: str = "4:4:4",
        metadata: dict[str, Any] | None = None,
    ) -> "MediaState":
        arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        h, w = arr.shape[:2]
        return cls(
            pixels=arr,
            color_space="RGB",
            bit_depth=8,
            width=int(w),
            height=int(h),
            frame_rate=float(fps),
            pixel_format=pixel_format,
            codec=codec,
            bitrate=float(bitrate),
            metadata=dict(metadata or {"EXIF_Present": True, "Container": "Original"}),
            history=[],
        )

    def append_event(self, ev: TransferEvent | dict[str, Any]) -> None:
        self.history.append(ev.to_dict() if isinstance(ev, TransferEvent) else dict(ev))

    def summary_dict(self) -> dict[str, Any]:
        return {
            "color_space": self.color_space,
            "bit_depth": self.bit_depth,
            "width": self.width,
            "height": self.height,
            "frame_rate": round(float(self.frame_rate), 2),
            "pixel_format": self.pixel_format,
            "codec": self.codec,
            "bitrate_kbps": round(float(self.bitrate), 1),
            "metadata": copy.deepcopy(self.metadata),
            "history_count": len(self.history),
            "history": copy.deepcopy(self.history),
        }


# ---------------------------------------------------------------------------
# 二、子系统 D（03 · 文件与传输）：蓝牙、USB、彩信与多代有损级联
# ---------------------------------------------------------------------------

@dataclass
class FileTransferSpec:
    """
    03 · 旧手机与视频网站之间的文件传输配置：
      - `method = 0`: 关闭 / 直通
      - `method = 1`: 蓝牙传输 (`bluetooth_copy`，位级无损复制文件本身，不添加伪造压缩；按策略保留或调整 OBEX 文件时间戳)
      - `method = 2`: USB / 电脑导入 (`usb_copy`，位级无损复制，记录 PC 导入目录、时间戳与缩略图缓存元数据变化)
      - `method = 3`: 彩信 / 早期社交软件发送 (`mms_social_send`，强制缩放至 320×240 或 640×480、重编码压缩并剥离 EXIF)
      - `method = 4`: 电脑编辑软件导出 + 多代有损重编码链 ($V_0 \\to V_1 \\to V_2 \\to V_3$)
    """
    enabled: bool = False
    method: int = 0
    generations: int = 1
    mms_max_width: int = 320
    mms_quality: int = 52
    nle_bitrate_kbps: float = 420.0
    jpeg_quality: int = 65
    jpeg_grid_shift: int = 2
    metadata_policy: str = "auto"

    def is_active(self) -> bool:
        return bool(self.enabled and self.method > 0)

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "FileTransferSpec":
        m = int(round(float(cfg.get("ldme_transfer_method", 0.0))))
        en = float(cfg.get("ldme_transfer_enabled", 0.0)) > 0.5 or (m > 0)
        if en and m <= 0:
            m = 4
        return cls(
            enabled=en,
            method=int(np.clip(m, 0, 5)),
            generations=int(np.clip(round(float(cfg.get("ldme_transfer_generations", 2.0))), 1, 8)),
            mms_max_width=int(np.clip(round(float(cfg.get("ldme_mms_width", 320.0))), 128, 1280)),
            mms_quality=int(np.clip(round(float(cfg.get("ldme_mms_quality", 52.0))), 10, 95)),
            nle_bitrate_kbps=float(np.clip(cfg.get("ldme_nle_bitrate_kbps", 420.0), 60.0, 4000.0)),
            jpeg_quality=int(np.clip(round(float(cfg.get("ldme_jpeg_decay_quality", 65.0))), 5, 95)),
            jpeg_grid_shift=int(np.clip(round(float(cfg.get("ldme_jpeg_decay_grid_shift", 2.0))), 0, 7)),
        )


def apply_file_transfer_node(
    state: MediaState,
    spec: FileTransferSpec,
    frame_index: int = 0,
) -> MediaState:
    """
    执行 03 · 文件与传输节点：
    严格遵循规范：蓝牙与 USB 传输绝不凭空加入图像压缩，而彩信/社交软件与桌面剪辑导出则执行真实的缩放与重编码。
    """
    if not spec.is_active():
        return state

    h, w = state.pixels.shape[:2]
    src_dev = str(state.metadata.get("Model", state.codec or "mobile_device"))

    if spec.method == 1:
        # 蓝牙 OBEX 传输：位级复制，像素零改动
        state.metadata = dict(state.metadata)
        state.metadata["TransferChannel"] = "Bluetooth 2.0 + EDR OBEX Object Push"
        state.metadata["FilesystemTimestamp"] = "Modified on Receiver Inbox"
        state.append_event(
            TransferEvent(
                source_device=src_dev,
                destination="receiver_phone_or_pc",
                method="bluetooth_copy",
                pixel_transform=None,
                reencoded=False,
                metadata_policy="preserve_exif_update_fs_mtime",
                codec_after=state.codec,
                bitrate_kbps_after=state.bitrate,
                resolution_after=f"{state.width}x{state.height}",
            )
        )
        return state

    if spec.method == 2:
        # USB 数据线 / 读卡器导入电脑：位级复制，像素零改动，更新文件系统目录与 Thumbs.db 记录
        state.metadata = dict(state.metadata)
        state.metadata["TransferChannel"] = "USB 2.0 Mass Storage / Nokia PC Suite"
        state.metadata["PCImportFolder"] = "C:\\My Documents\\My Pictures\\Phone_Import"
        state.metadata["ThumbnailSidecar"] = "Thumbs.db cached"
        state.append_event(
            TransferEvent(
                source_device=src_dev,
                destination="desktop_workstation",
                method="usb_copy",
                pixel_transform=None,
                reencoded=False,
                metadata_policy="preserve_exif_add_sidecar",
                codec_after=state.codec,
                bitrate_kbps_after=state.bitrate,
                resolution_after=f"{state.width}x{state.height}",
            )
        )
        return state

    if spec.method == 3:
        # 彩信 (MMS <= 300KB) 或早期社交软件发送：强制降分辨率 + 强 4:2:0 重编码 + 剥离全部 EXIF
        target_w = min(w, max(160, int(spec.mms_max_width)))
        target_h = max(120, int(round(h * (target_w / max(1.0, float(w))))))
        small = _resample_to(state.pixels, min(h, target_h), min(w, target_w))
        q_norm = float(np.clip((100 - spec.mms_quality) / 85.0, 0.08, 1.0))
        q_step = 0.012 + 0.085 * (q_norm ** 1.25)
        coded_small = _dct_420_pass(small, block_size=8, q_step=q_step, chroma_subsample=2)
        state.pixels = _resample_to(coded_small, h, w)
        state.width = target_w
        state.height = target_h
        state.pixel_format = "4:2:0"
        state.codec = "jpeg_mms" if state.codec in ("raw", "jpeg") else "3gp_mms_h263"
        state.bitrate = min(state.bitrate, 128.0)
        # 剥离原始 EXIF
        state.metadata = {
            "EXIF_Present": False,
            "StrippedBy": "MMS / Early Social Gateway",
            "MaxPayloadKB": 100,
        }
        state.append_event(
            TransferEvent(
                source_device=src_dev,
                destination="mms_or_social_gateway",
                method="mms_social_compress",
                pixel_transform=f"downscale_{target_w}x{target_h}_420_reencode",
                reencoded=True,
                metadata_policy="strip_exif",
                codec_after=state.codec,
                bitrate_kbps_after=state.bitrate,
                resolution_after=f"{target_w}x{target_h}",
            )
        )
        return state

    if spec.method == 5:
        # method == 5: 早期优酷土豆/论坛截图搬运多代 JPEG 磨损衰减 (I_0 -> I_1 -> ... -> I_k)
        gens = max(1, int(spec.generations))
        q = int(spec.jpeg_quality) if spec.jpeg_quality > 0 else 65
        s = int(spec.jpeg_grid_shift) if spec.jpeg_grid_shift >= 0 else 2
        decayed = apply_jpeg_generation_loss(
            state.pixels,
            generations=gens,
            quality=q,
            grid_shift=s,
            chroma_subsample=True,
        )
        state.pixels = decayed
        state.codec = "jpeg_wear_multigen"
        state.pixel_format = "4:2:0"
        state.append_event(
            TransferEvent(
                source_device=src_dev,
                destination="forum_or_youku_reupload",
                method="jpeg_generation_decay",
                pixel_transform=f"dct_8x8_ijg_q{q}_gen{gens}_shift{s}",
                reencoded=True,
                metadata_policy="strip_exif",
                codec_after="jpeg",
                bitrate_kbps_after=None,
                resolution_after=f"{state.width}x{state.height}",
            )
        )
        return state

    # method == 4: 电脑剪辑软件导出 + 多代有损压缩级联 (V_0 -> V_1 -> ... -> V_k)
    gens = max(1, int(spec.generations))
    cur = state.pixels.copy()
    cur_w, cur_h = state.width, state.height
    cur_kbps = float(spec.nle_bitrate_kbps)

    for g_idx in range(gens):
        scale_g = max(0.36, 0.80 - 0.12 * g_idx)
        gw = max(160, int(round(w * scale_g)))
        gh = max(120, int(round(h * scale_g)))
        sub = _resample_to(cur, gh, gw)
        q_step = float(np.clip(0.024 * ((450.0 / max(60.0, cur_kbps)) ** 0.55) * (1.0 + 0.28 * g_idx), 0.012, 0.12))
        # 每代重编码引入不同的块网格相位偏移（模拟重缩放后的非对齐 8x8 宏块网格重采样损伤）
        shift_px = int(g_idx % 3)
        if shift_px > 0:
            sub = np.roll(sub, shift=(shift_px, shift_px), axis=(0, 1))
        sub_coded = _dct_420_pass(sub, block_size=8 if g_idx % 2 == 0 else 16, q_step=q_step, chroma_subsample=2)
        if shift_px > 0:
            sub_coded = np.roll(sub_coded, shift=(-shift_px, -shift_px), axis=(0, 1))
        # 早期桌面转码软件（如格式工厂/早期会声会影/Windows Movie Maker）常见的 BT.601/709 矩阵微偏与锐化残影
        hp = sub_coded - lowpass(sub_coded, 1.0, axis=1)
        sub_coded = np.clip(sub_coded + 0.14 * hp, 0.0, 1.0)
        cur = _resample_to(sub_coded, h, w)
        cur_w, cur_h = gw, gh
        cur_kbps = max(140.0, cur_kbps * 0.78)
        stage_name = "desktop_nle_export" if g_idx == 0 else f"reupload_transcode_gen_{g_idx + 1}"
        state.append_event(
            TransferEvent(
                source_device=src_dev if g_idx == 0 else f"gen_{g_idx}_stream",
                destination="desktop_editor_mp4" if g_idx == 0 else f"video_platform_gen_{g_idx + 1}",
                method=stage_name,
                pixel_transform=f"scale_{gw}x{gh}_q{q_step:.3f}",
                reencoded=True,
                metadata_policy="container_rewrap",
                codec_after="mpeg4_nle" if g_idx == 0 else "flv1_or_h264",
                bitrate_kbps_after=round(cur_kbps, 1),
                resolution_after=f"{gw}x{gh}",
            )
        )

    state.pixels = np.clip(cur, 0.0, 1.0).astype(np.float32)
    state.width = cur_w
    state.height = cur_h
    state.pixel_format = "4:2:0"
    state.codec = "multi_gen_transcode"
    state.bitrate = cur_kbps
    return state


def _dct_420_pass(
    rgb: np.ndarray,
    block_size: int = 8,
    q_step: float = 0.03,
    chroma_subsample: int = 2,
) -> np.ndarray:
    """基础 YUV 4:2:0 色度子采样 + 宏块量化算子。"""
    h, w = rgb.shape[:2]
    bs = max(2, int(block_size))
    y = rgb @ LUMA_W
    cb = rgb[..., 2] - y
    cr = rgb[..., 0] - y

    cs = max(1, int(chroma_subsample))
    if cs > 1 and h >= cs * 2 and w >= cs * 2:
        cb = _resample_to(_resample_to(cb[..., None], max(2, h // cs), max(2, w // cs)), h, w)[..., 0]
        cr = _resample_to(_resample_to(cr[..., None], max(2, h // cs), max(2, w // cs)), h, w)[..., 0]

    gh, gw = max(2, h // bs), max(2, w // bs)
    y_dc = _resample_to(_resample_to(y[..., None], gh, gw), h, w)[..., 0]
    y_ac = y - y_dc
    if q_step > 1e-4:
        y_dc_q = np.round(y_dc / q_step) * q_step
        y_ac_q = np.round(y_ac / (q_step * 1.65)) * (q_step * 1.65)
        y_out = 0.68 * y_dc_q + 0.32 * (y_dc + y_ac_q)
    else:
        y_out = y

    r = y_out + cr
    b = y_out + cb
    g = (y_out - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 三、子系统 C（04 · 平台处理）：优酷 / 土豆 2006–2015 流媒体转码管线
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamingPlatformTier:
    """2006–2015 中文视频网站（优酷/土豆等）历史实验清晰度档位配置（C1 & C4）。"""
    tier_id: int
    key: str
    name: str
    era_stage: str             # "stage1_flash_flv" | "stage2_h264_multitier" | "stage3_mobile_multiscreen"
    era_years: str
    container: str             # "FLV" | "MP4" | "HLS_TS"
    video_codec: str           # "flv1" | "h264"
    width: int
    height: int
    fps: float
    video_bitrate_kbps: float
    audio_codec: str           # "mp3" | "aac"
    audio_bitrate_kbps: float
    gop_len: int
    watermark_label: str


PLATFORM_TRANSCODE_TIERS: dict[int, StreamingPlatformTier] = {
    1: StreamingPlatformTier(
        tier_id=1,
        key="youku_tudou_2007_flv",
        name="早期 Flash 流畅档 320×240 FLV",
        era_stage="stage1_flash_flv",
        era_years="2006-2009",
        container="FLV",
        video_codec="flv1",
        width=320,
        height=240,
        fps=25.0,
        video_bitrate_kbps=350.0,
        audio_codec="mp3",
        audio_bitrate_kbps=48.0,
        gop_len=15,
        watermark_label="YOUKU/TUDOU FLV 240P",
    ),
    2: StreamingPlatformTier(
        tier_id=2,
        key="platform_2010_sd_h264",
        name="标清 H.264 档 480×360",
        era_stage="stage2_h264_multitier",
        era_years="2009-2012",
        container="FLV/MP4",
        video_codec="h264",
        width=480,
        height=360,
        fps=25.0,
        video_bitrate_kbps=700.0,
        audio_codec="aac",
        audio_bitrate_kbps=64.0,
        gop_len=25,
        watermark_label="SD 360P H.264",
    ),
    3: StreamingPlatformTier(
        tier_id=3,
        key="platform_2011_hd_h264",
        name="高清 H.264 档 640×480",
        era_stage="stage2_h264_multitier",
        era_years="2009-2012",
        container="MP4",
        video_codec="h264",
        width=640,
        height=480,
        fps=25.0,
        video_bitrate_kbps=1100.0,
        audio_codec="aac",
        audio_bitrate_kbps=96.0,
        gop_len=30,
        watermark_label="HD 480P H.264",
    ),
    4: StreamingPlatformTier(
        tier_id=4,
        key="platform_2013_mobile_mp4",
        name="移动多屏适配档 640×360 MP4",
        era_stage="stage3_mobile_multiscreen",
        era_years="2012-2015",
        container="MP4/HLS",
        video_codec="h264",
        width=640,
        height=360,
        fps=25.0,
        video_bitrate_kbps=600.0,
        audio_codec="aac",
        audio_bitrate_kbps=64.0,
        gop_len=25,
        watermark_label="MOBILE 360P MP4",
    ),
}


def encode_decode_ffmpeg_frame(
    rgb: np.ndarray,
    width: int = 320,
    height: int = 240,
    video_codec: str = "flv1",
    bitrate_kbps: float = 350.0,
) -> np.ndarray | None:
    """
    真实 FFmpeg 编解码器后端（C2 & 第八节）：
    当启用真实 FFmpeg 实验模式且环境中存在 `ffmpeg` 时，将图像帧通过真实的 `flv1` (Sorenson Spark)、
    `h263p`、`mpeg4` 或 `libx264` 编码为目标容器并解码回 RGB。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None
    h, w = rgb.shape[:2]
    tw = max(32, (int(width) // 2) * 2)
    th = max(32, (int(height) // 2) * 2)
    u8 = np.clip(np.round(_resample_to(rgb, th, tw) * 255.0), 0, 255).astype(np.uint8)

    codec_map = {
        "flv1": ("flv", ".flv", ["-q:v", str(int(np.clip(round(3800.0 / max(80.0, bitrate_kbps)), 2, 24)))]),
        "h263": ("h263p", ".3gp", ["-b:v", f"{int(bitrate_kbps)}k"]),
        "mpeg4": ("mpeg4", ".mp4", ["-b:v", f"{int(bitrate_kbps)}k"]),
        "h264": ("libx264", ".mp4", ["-b:v", f"{int(bitrate_kbps)}k", "-preset", "ultrafast"]),
    }
    ff_codec, ext, extra_args = codec_map.get(video_codec, codec_map["flv1"])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_in = os.path.join(tmpdir, "in.raw")
            vid_path = os.path.join(tmpdir, f"encoded{ext}")
            raw_out = os.path.join(tmpdir, "out.raw")
            u8.tofile(raw_in)
            cmd_enc = [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tw}x{th}", "-r", "25",
                "-i", raw_in,
                "-c:v", ff_codec, "-pix_fmt", "yuv420p",
                *extra_args,
                vid_path,
            ]
            subprocess.run(cmd_enc, check=True, timeout=4.0)
            cmd_dec = [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-i", vid_path,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-vframes", "1",
                raw_out,
            ]
            subprocess.run(cmd_dec, check=True, timeout=4.0)
            dec_bytes = np.fromfile(raw_out, dtype=np.uint8)
            if dec_bytes.size == th * tw * 3:
                dec_rgb = dec_bytes.reshape((th, tw, 3)).astype(np.float32) / 255.0
                return _resample_to(dec_rgb, h, w)
    except Exception:
        return None
    return None


def apply_platform_transcode(
    state: MediaState,
    tier_id: int = 1,
    source_origin: int = 0,          # 0=高清源文件被平台降为低清, 1=上传时原本就是低清源文件
    bitrate_override_kbps: float = 0.0,
    gop_override: int = 0,
    fps_override: float = 0.0,
    platform_watermark: bool = False,
    use_real_ffmpeg: bool = False,
    frame_index: int = 0,
) -> MediaState:
    """
    执行 04 · 平台处理（优酷/土豆 2006–2015 流媒体转码管线）：
      - 区分「上传时原本就是低清 (`source_origin=1`)」与「高清源文件被平台降采样转码为低清 (`source_origin=0`)」
      - 区分**空间退化**（真实缩放、4:2:0 色度子采样、宏块与振铃）与**时间退化**（低帧率抽样、GOP 关键帧预测误差累积、运动区域与静止背景的差异化损伤）
    """
    tier = PLATFORM_TRANSCODE_TIERS.get(int(np.clip(tier_id, 1, 4)), PLATFORM_TRANSCODE_TIERS[1])
    rgb = state.pixels.copy()
    h, w = rgb.shape[:2]

    eff_kbps = float(bitrate_override_kbps) if bitrate_override_kbps > 10.0 else float(tier.video_bitrate_kbps)
    eff_gop = int(gop_override) if gop_override >= 4 else int(tier.gop_len)
    eff_fps = float(fps_override) if fps_override >= 5.0 else float(tier.fps)

    # 时间采样退化：当目标帧率低于源帧率（例如 15fps 3GP 上传转 25fps FLV）时发生帧重复/顿挫
    fps_ratio = max(1.0, state.frame_rate / max(5.0, eff_fps))
    effective_frame_idx = int(round(np.floor(frame_index / fps_ratio) * fps_ratio))
    gop_pos = effective_frame_idx % max(4, eff_gop)
    gop_pulse = float(np.exp(-1.15 * (gop_pos / max(1.0, float(eff_gop - 1)))))

    # 区分「原本就是低清上传」与「高清源被平台下采样为低清」
    if source_origin == 1:
        # 源本身在进入平台前已是低清（如 Nokia 6600/5800 320x240/176x144 源），带着传感器噪点与早期块伪影进入转码器
        pre_w, pre_h = min(w, 240), min(h, 180)
        pre_low = _resample_to(rgb, pre_h, pre_w)
        pre_low = _dct_420_pass(pre_low, block_size=8, q_step=0.036, chroma_subsample=2)
        work_in = _resample_to(pre_low, h, w)
    else:
        # 高清源在平台转码器内先经过抗混叠前置低通与双线性下采样
        work_in = lowpass(rgb, 0.75, axis=1)

    # 若启用真实 FFmpeg 编解码实验，直接调用 FFmpeg
    ff_out = None
    if use_real_ffmpeg:
        ff_out = encode_decode_ffmpeg_frame(
            work_in,
            width=tier.width,
            height=tier.height,
            video_codec=tier.video_codec,
            bitrate_kbps=eff_kbps,
        )

    if ff_out is not None:
        out_rgb = ff_out
    else:
        # 参数化高保真仿真：真实分辨率缩放 + 空间/时间差异化宏块压缩
        tw = max(64, min(w, int(tier.width)))
        th = max(48, min(h, int(tier.height)))
        scaled = _resample_to(work_in, th, tw)

        # 估计局部空间梯度与帧间运动复杂度（运动中的主体/复杂边缘比静止墙壁承受更重的 P 帧量化与运动补偿块伪影）
        lum = scaled @ LUMA_W
        gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
        activity = np.clip((gx + gy) * 3.2, 0.0, 1.0)
        motion_wave = 0.5 + 0.5 * np.sin(effective_frame_idx * 0.37 + np.linspace(0.0, 3.14, th)[:, None])
        temporal_stress = np.clip(activity * (1.25 - 0.55 * gop_pulse) * (0.7 + 0.3 * motion_wave), 0.0, 1.0)

        base_q = float(np.clip(0.038 * ((400.0 / max(48.0, eff_kbps)) ** 0.62) * (1.30 - 0.45 * gop_pulse), 0.008, 0.14))
        bs = 8 if tier.video_codec == "flv1" else 16
        coded_low = _dct_420_pass(scaled, block_size=bs, q_step=base_q, chroma_subsample=2)
        coded_heavy = _dct_420_pass(scaled, block_size=16, q_step=base_q * 1.85, chroma_subsample=2)
        blended = coded_low * (1.0 - temporal_stress[..., None] * 0.65) + coded_heavy * (temporal_stress[..., None] * 0.65)

        if tier.video_codec == "flv1":
            # Sorenson Spark / H.263 缺乏 H.264 环路去块滤波，块边界硬朗且伴随蚊式振铃噪点 (Mosquito Noise)
            edge_ring = (blended - lowpass(blended, 1.1, axis=1)) * (0.24 * activity[..., None])
            blended = np.clip(blended + edge_ring, 0.0, 1.0)
        else:
            # H.264 帧内环路去块滤波（Deblocking Filter）：抹平 16x16 块边界，但在低码率下产生局部柔化
            deblock = np.stack([gaussian_filter(blended[..., c], sigma=0.75) for c in range(3)], axis=-1)
            blended = 0.65 * blended + 0.35 * deblock

        out_rgb = _resample_to(blended, h, w)

    if platform_watermark:
        wm_h = max(12, int(h * 0.065))
        wm_w = max(64, int(w * 0.26))
        y0, x0 = 8, max(4, w - wm_w - 10)
        y1, x1 = min(h, y0 + wm_h), min(w, x0 + wm_w)
        out_rgb[y0:y1, x0:x1] = out_rgb[y0:y1, x0:x1] * 0.65 + np.array([0.18, 0.58, 0.92], dtype=np.float32) * 0.35
        _draw_text(out_rgb, x0 + 4, y0 + 3, tier.watermark_label[:16], (0.96, 0.98, 1.0), scale=1)

    state.pixels = np.clip(out_rgb, 0.0, 1.0).astype(np.float32)
    state.width = tier.width
    state.height = tier.height
    state.frame_rate = eff_fps
    state.pixel_format = "4:2:0"
    state.codec = tier.video_codec
    state.bitrate = eff_kbps
    state.metadata["PlatformTier"] = tier.name
    state.metadata["PlatformEra"] = tier.era_years
    state.metadata["Container"] = tier.container
    state.metadata["AudioCodec"] = f"{tier.audio_codec.upper()} {int(tier.audio_bitrate_kbps)}kbps"
    state.metadata["SourceOrigin"] = "low_res_direct_upload" if source_origin == 1 else "hd_downscaled_by_platform"
    state.append_event(
        TransferEvent(
            source_device="uploader_client",
            destination=f"platform_{tier.key}",
            method="platform_transcode",
            pixel_transform=f"scale_{tier.width}x{tier.height}_{tier.video_codec}_{int(eff_kbps)}kbps",
            reencoded=True,
            metadata_policy="strip_exif_rewrap_flv_mp4",
            codec_after=tier.video_codec,
            bitrate_kbps_after=eff_kbps,
            resolution_after=f"{tier.width}x{tier.height}",
        )
    )
    return state


# ---------------------------------------------------------------------------
# 四、子系统 E（05 · 播放与显示）：早期 LCD、Flash 播放器缩放与缓冲状态机
# ---------------------------------------------------------------------------

@dataclass
class EarlyLCDFlashSpec:
    """
    05 · 早期数字显示与 Flash 播放器分层模型（第六节）：
    严格区分 **LCD** 与 **CRT** 的物理机制（LCD 具有液晶响应时间拖影、6-bit/RGB565 量化阶梯、
    TN 视角伽马偏移、RGB 竖条子像素晶格与 CCFL 背光，绝不混入 CRT 扫描线或电子束弯曲）：
      - `display_device = 1`: 早期手机 TFT 液晶屏（176×208 / 240×320，16-bit RGB565 65K 色量化，28ms 慢响应，窄视角反相）
      - `display_device = 2`: 2007 桌面 6-bit TN 液晶显示器 + Flash Player 播放器缩放
      - `display_device = 3`: 经典桌面 CRT 监视器（仅供基准测试三与 LCD 进行同源对照）
    """
    enabled: bool = False
    display_device: int = 2          # 1=早期手机 TFT LCD, 2=2007 桌面 6-bit TN LCD, 3=CRT 对照监视器
    lcd_bit_depth: int = 6           # 5(RGB565) | 6(6-bit TN + FRC) | 8(8-bit IPS)
    lcd_response_ms: float = 16.0    # 液晶分子上升/下降响应时间 (ms)，产生像素保持型拖影而非荧光粉余辉
    lcd_viewing_angle_deg: float = 12.0  # 垂直偏轴观看角度（TN 面板向下看变暗反相、向上看泛白）
    lcd_subpixel_grid: float = 0.35  # RGB 竖条子像素晶格可见度 (0..1)
    lcd_backlight_ccfl: float = 0.25 # 早期 CCFL 冷阴极荧光灯管背光色温偏冷青白 (0..1)
    flash_scale_mode: int = 1        # 0=原始比例居中, 1=Flash 窗口非整数双线性放大, 2=最近邻像素化放大, 3=4:3 强行拉伸满屏 16:9
    flash_chrome_overlay: bool = False   # 是否显示早期 Flash Player 底部控制栏与缓冲进度条
    network_throughput_kbps: float = 520.0  # 客户端网络吞吐量（低于视频码率时触发缓冲区耗尽与暂停转圈）
    show_text: int = 0               # 文字叠加显隐控制 (0=不要文字纯净画面, 1=仅标签, 2=完整遥测)

    def is_active(self) -> bool:
        return bool(self.enabled and self.display_device > 0)

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "EarlyLCDFlashSpec":
        dev = int(round(float(cfg.get("ldme_display_device", 0.0))))
        en = float(cfg.get("ldme_display_enabled", 0.0)) > 0.5 or (dev > 0)
        if en and dev <= 0:
            dev = 2
        return cls(
            enabled=en,
            display_device=int(np.clip(dev, 0, 3)),
            lcd_bit_depth=int(np.clip(round(float(cfg.get("ldme_lcd_bits", 6.0))), 5, 8)),
            lcd_response_ms=float(np.clip(cfg.get("ldme_lcd_response_ms", 16.0), 2.0, 45.0)),
            lcd_viewing_angle_deg=float(np.clip(cfg.get("ldme_lcd_view_angle", 12.0), -35.0, 35.0)),
            lcd_subpixel_grid=float(np.clip(cfg.get("ldme_lcd_subpixel", 0.35), 0.0, 1.0)),
            lcd_backlight_ccfl=float(np.clip(cfg.get("ldme_lcd_ccfl", 0.25), 0.0, 1.0)),
            flash_scale_mode=int(np.clip(round(float(cfg.get("ldme_flash_scale_mode", 1.0))), 0, 3)),
            flash_chrome_overlay=bool(float(cfg.get("ldme_flash_chrome", 0.0)) > 0.5),
            network_throughput_kbps=float(np.clip(cfg.get("ldme_net_throughput_kbps", 520.0), 40.0, 5000.0)),
            show_text=int(round(float(cfg.get("ldme_flash_show_text", cfg.get("show_text", cfg.get("stream_show_text", 0.0)))))),
        )


def apply_early_lcd_and_flash_player(
    state: MediaState,
    spec: EarlyLCDFlashSpec,
    frame_index: int = 0,
    prev_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    执行 05 · 早期数字显示与 Flash 播放：
      1. 播放器与浏览器缩放层（非整数倍窗口缩放、4:3 -> 16:9 宽高比拉伸、独立网络缓冲状态机）
      2. 物理显示设备层（严格区分早期手机 RGB565 TFT LCD、2007 桌面 6-bit TN LCD 与经典 CRT）
    """
    rgb = np.clip(state.pixels.astype(np.float32), 0.0, 1.0).copy()
    h, w = rgb.shape[:2]

    # --- E2. 播放器缩放与网络缓冲时间系统 ---
    if spec.flash_scale_mode == 2:
        # 最近邻像素化放大（Flash Player `smoothing = false` 时的硬锯齿像素块）
        sh, sw = max(64, int(h * 0.42)), max(80, int(w * 0.42))
        small = _resample_to(rgb, sh, sw)
        yi = np.clip((np.arange(h) * sh) // h, 0, sh - 1)
        xi = np.clip((np.arange(w) * sw) // w, 0, sw - 1)
        rgb = small[yi[:, None], xi[None, :]]
    elif spec.flash_scale_mode == 3:
        # 4:3 视频在 16:9 宽屏显示器全屏观看时被强行水平拉伸 1.33x
        crop_w = int(round(w * 0.76))
        x0 = (w - crop_w) // 2
        rgb = _resample_to(rgb[:, x0:x0 + crop_w], h, w)
    elif spec.flash_scale_mode == 1:
        # 浏览器非整数倍窗口双线性缩放（例如 320x240 放大到 448x336 播放窗口造成的二次插值柔化）
        sh, sw = max(80, int(h * 0.62)), max(100, int(w * 0.62))
        rgb = _resample_to(_resample_to(rgb, sh, sw), h, w)

    # 独立网络缓冲区水位计算
    demand_kbps = max(64.0, float(state.bitrate))
    throughput_kbps = float(spec.network_throughput_kbps)
    t_sec = float(frame_index) / max(1.0, float(state.frame_rate))
    net_wave = 0.85 + 0.25 * np.sin(t_sec * 1.1)
    eff_tp = throughput_kbps * net_wave
    fill_ratio = eff_tp / demand_kbps
    buffer_level_s = float(np.clip(2.0 * fill_ratio - 0.85 + 0.35 * np.cos(t_sec * 0.9), 0.0, 5.0))
    is_rebuffering = bool(buffer_level_s < 0.25 and fill_ratio < 0.88)

    # --- E1. 显示设备物理层 ---
    dev_id = int(spec.display_device)
    if dev_id == 1:
        # 1. 早期功能机 TFT 彩屏 (176x208 / 240x320, RGB565 16-bit 65K 色, 28ms 慢液晶响应, 强视角偏移)
        lcd_h, lcd_w = min(h, 208), min(w, 176)
        small = _resample_to(rgb, lcd_h, lcd_w)
        # RGB565 量化 (R: 32级, G: 64级, B: 32级)
        r_q = np.round(small[..., 0] * 31.0) / 31.0
        g_q = np.round(small[..., 1] * 63.0) / 63.0
        b_q = np.round(small[..., 2] * 31.0) / 31.0
        q_rgb = np.stack([r_q, g_q, b_q], axis=-1)
        # 还原到输出画布尺寸（保留手机液晶颗粒晶格感）
        yi = np.clip((np.arange(h) * lcd_h) // h, 0, lcd_h - 1)
        xi = np.clip((np.arange(w) * lcd_w) // w, 0, lcd_w - 1)
        out = q_rgb[yi[:, None], xi[None, :]].astype(np.float32)
        # 慢速液晶响应拖影（Sample-and-Hold 保持型拖尾）
        out = _apply_lcd_response_smear(out, response_ms=max(24.0, spec.lcd_response_ms), prev_rgb=prev_rgb)
        # 手机小屏 TN 垂直视角伽马与对比度偏移
        out = _apply_tn_viewing_angle_shift(out, angle_deg=max(16.0, spec.lcd_viewing_angle_deg))
        # 手机 TFT 方格像素黑矩阵网格（非扫描线！）
        out = _apply_lcd_subpixel_matrix(out, strength=max(0.45, spec.lcd_subpixel_grid), pitch_px=3)
        dev_name = "Feature Phone 65K TFT LCD (RGB565)"

    elif dev_id == 2:
        # 2. 2007 桌面 6-bit TN 液晶显示器 (64级/通道 + FRC 抖动, 液晶响应拖影, 视角伽马变化, CCFL 背光)
        out = rgb.copy()
        bits = int(np.clip(spec.lcd_bit_depth, 5, 8))
        levels = float((1 << bits) - 1)
        # 2x2 Bayer Hi-FRC 空间抖动矩阵 + 6-bit 量化
        bayer_2x2 = np.array([[0.0, 0.5], [0.75, 0.25]], dtype=np.float32) - 0.375
        dither = np.tile(bayer_2x2, (h // 2 + 1, w // 2 + 1))[:h, :w, None] / levels
        out = np.round(np.clip(out + dither * 0.65, 0.0, 1.0) * levels) / levels

        # 液晶分子响应时间拖影 (Sample-and-Hold motion smear)
        out = _apply_lcd_response_smear(out, response_ms=spec.lcd_response_ms, prev_rgb=prev_rgb)
        # TN 面板垂直视角伽马偏移
        out = _apply_tn_viewing_angle_shift(out, angle_deg=spec.lcd_viewing_angle_deg)
        # 早期 CCFL 冷白背光与暗部漏光（Black Level Backlight Bleed：LCD 暗部无法像 CRT 一样纯黑）
        ccfl = float(spec.lcd_backlight_ccfl)
        if ccfl > 1e-3:
            black_floor = 0.028 * ccfl
            out = black_floor + out * (1.0 - black_floor)
            out[..., 0] *= (1.0 - 0.05 * ccfl)
            out[..., 1] *= (1.0 + 0.02 * ccfl)
            out[..., 2] *= (1.0 + 0.09 * ccfl)
        # RGB 竖条子像素晶格
        out = _apply_lcd_subpixel_matrix(out, strength=spec.lcd_subpixel_grid, pitch_px=2)
        dev_name = f"2007 Desktop {bits}-bit TN LCD Monitor (CCFL)"

    else:
        # 3. 经典 CRT 监视器对照模式（扫描线、荧光粉非线性余辉与玻璃球面光晕）
        out = rgb.copy()
        rows = np.arange(h, dtype=np.float32)[:, None, None]
        scan_mask = 0.82 + 0.18 * np.cos(rows * np.pi)
        bloom = np.stack([gaussian_filter(out[..., c], sigma=2.2) for c in range(3)], axis=-1)
        out = np.clip((out ** 1.12) * scan_mask + 0.18 * bloom, 0.0, 1.0)
        dev_name = "Classic CRT Monitor (Scanline & Phosphor Reference)"

    # 若处于网络缓冲耗尽状态且开启了文字叠加，绘制缓冲状态/控制栏；若不要文字 (show_text == 0) 则保持画面纯净，绝不输出 BUFFERING... 文字
    if is_rebuffering and spec.show_text >= 1:
        out = out * 0.55
        _draw_text(out, max(8, w // 2 - 58), max(8, h // 2 - 6), "BUFFERING... 0%", (0.95, 0.96, 0.98), scale=1)

    if spec.flash_chrome_overlay and spec.show_text >= 1:
        bar_h = max(14, int(h * 0.065))
        out[h - bar_h :, :] = out[h - bar_h :, :] * 0.25 + np.array([0.16, 0.17, 0.19], dtype=np.float32) * 0.75
        # 播放进度条与已缓冲灰条
        prog_y0 = h - bar_h + 2
        prog_y1 = min(h - 2, prog_y0 + 3)
        buf_x1 = int(np.clip(w * min(1.0, 0.35 + buffer_level_s * 0.15), 10, w - 4))
        play_x1 = int(np.clip(w * 0.28, 6, buf_x1))
        out[prog_y0:prog_y1, 4:buf_x1] = np.array([0.45, 0.48, 0.52], dtype=np.float32)
        out[prog_y0:prog_y1, 4:play_x1] = np.array([0.18, 0.62, 0.95], dtype=np.float32)
        _draw_text(
            out, 8, h - bar_h + 6,
            f"FLASH PLAYER 10 | {state.width}x{state.height} {state.codec.upper()} | BUF:{buffer_level_s:.1f}s",
            (0.86, 0.88, 0.92), scale=1,
        )

    info = {
        "display_device_id": dev_id,
        "display_device_name": dev_name,
        "lcd_bit_depth": spec.lcd_bit_depth if dev_id != 3 else 8,
        "lcd_response_ms": round(float(spec.lcd_response_ms), 1) if dev_id != 3 else 0.0,
        "flash_scale_mode": int(spec.flash_scale_mode),
        "buffer_level_s": round(buffer_level_s, 3),
        "rebuffering_stall": is_rebuffering,
        "is_crt": bool(dev_id == 3),
    }
    return np.clip(out, 0.0, 1.0).astype(np.float32), info


def _apply_lcd_response_smear(
    rgb: np.ndarray,
    response_ms: float = 16.0,
    prev_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """
    模拟早期液晶面板响应时间（灰阶到灰阶 GtG 12–30ms 非对称上升/下降时间）：
    与 CRT 指数型荧光粉余辉不同，LCD 是 Sample-and-Hold（样本保持）型器件，其运动模糊表现为
    沿水平/运动方向的阶跃保持拖影。
    """
    if response_ms <= 3.0:
        return rgb
    alpha = float(np.clip((response_ms - 3.0) / 48.0, 0.0, 0.65))
    if prev_rgb is not None and prev_rgb.shape == rgb.shape:
        # 液晶分子暗到亮与亮到暗响应速度非对称
        rising = rgb > prev_rgb
        w_prev = np.where(rising, alpha * 0.75, alpha * 1.10)
        return np.clip(rgb * (1.0 - w_prev) + prev_rgb * w_prev, 0.0, 1.0).astype(np.float32)
    # 单帧预览下用轻微非对称水平样本保持拖尾表征液晶慢响应
    shifted1 = horizontal_shift(rgb, np.full(rgb.shape[0], 1.6, dtype=np.float32))
    shifted2 = horizontal_shift(rgb, np.full(rgb.shape[0], 3.2, dtype=np.float32))
    return np.clip(rgb * (1.0 - alpha * 0.55) + shifted1 * (alpha * 0.35) + shifted2 * (alpha * 0.20), 0.0, 1.0).astype(np.float32)


def _apply_tn_viewing_angle_shift(rgb: np.ndarray, angle_deg: float = 12.0) -> np.ndarray:
    """
    模拟早期 TN（Twisted Nematic）液晶面板的垂直视角伽马与对比度偏移：
    屏幕顶部与底部相对人眼视线存在垂直夹角差异，导致上方偏亮泛白、下方暗部伽马压暗甚至轻微灰阶反相。
    """
    if abs(angle_deg) <= 1.0:
        return rgb
    h = rgb.shape[0]
    y_norm = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None, None]
    ang_factor = float(np.clip(angle_deg / 35.0, -1.0, 1.0))
    # 局部有效伽马随垂直位置偏移
    local_gamma = np.clip(1.0 + 0.38 * ang_factor * y_norm, 0.65, 1.55)
    out = np.clip(rgb, 1e-4, 1.0) ** local_gamma
    # 极端视角下暗部轻微泛灰/反相
    inversion_lift = np.clip(0.08 * abs(ang_factor) * np.maximum(0.0, y_norm), 0.0, 0.12)
    out = out * (1.0 - inversion_lift) + (1.0 - out) * inversion_lift
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_lcd_subpixel_matrix(rgb: np.ndarray, strength: float = 0.35, pitch_px: int = 2) -> np.ndarray:
    """模拟 LCD 规整的矩形 RGB 竖条子像素与像素间薄黑矩阵网格（绝无 CRT 隔行扫描线或荫罩圆点）。"""
    if strength <= 1e-3:
        return rgb
    h, w = rgb.shape[:2]
    p = max(2, int(pitch_px))
    grid = np.ones((h, w, 3), dtype=np.float32)
    # 像素行/列薄晶格边界
    grid[0::p, :, :] *= (1.0 - 0.14 * strength)
    grid[:, 0::p, :] *= (1.0 - 0.14 * strength)
    # RGB 竖条子像素微强调
    sub_mod = np.ones((1, w, 3), dtype=np.float32)
    sub_mod[:, 0::3, 0] += 0.06 * strength
    sub_mod[:, 1::3, 1] += 0.06 * strength
    sub_mod[:, 2::3, 2] += 0.06 * strength
    out = rgb * grid * sub_mod
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 五、子系统 F（06 · 参数校准与三组标准基准测试 + 全链路组合器）
# ---------------------------------------------------------------------------

def run_ldme_full_chain(
    rgb: np.ndarray,
    camera_profile: int = 2,          # 0=跳过手机摄像, 1..8=Nokia/iPhone 机型
    capture_mode: int = 0,            # 0=capture() 照片, 1=record() 视频
    transfer_method: int = 0,         # 0=直通, 1=蓝牙, 2=USB, 3=彩信/社交, 4=桌面剪辑+多代重编码
    transfer_generations: int = 2,
    platform_tier: int = 0,           # 0=跳过平台转码, 1=2007 FLV 240p, 2=2010 H.264 360p, 3=2011 480p, 4=2013 移动多屏
    source_origin: int = 0,           # 0=高清下采样, 1=原生低清上传
    display_device: int = 0,          # 0=跳过显示层, 1=手机 TFT LCD, 2=2007 桌面 TN LCD, 3=CRT 对照
    flash_scale_mode: int = 1,
    flash_chrome: bool = False,
    frame_index: int = 0,
    fps: float = 25.0,
    seed: int = 20240101,
) -> tuple[np.ndarray, MediaState, dict[str, Any]]:
    """
    串联 LDME 完整数字媒介历史管线（第一节至第六节）：
    `01 场景输入 -> 02 摄像设备成像 -> 03 文件与传输 -> 04 平台处理 -> 05 播放与显示 -> 06 输出与历史记录`
    """
    state = MediaState.from_rgb(rgb, fps=fps)

    # 02 · 摄像设备成像
    cam_info: dict[str, Any] = {"active": False}
    if camera_profile > 0:
        prof = resolve_mobile_profile(camera_profile)
        cam_spec = MobileISPOpticsSpec(enabled=True, profile_id=prof.profile_id, capture_mode=capture_mode)
        cam_rgb, cam_info = run_mobile_isp_pipeline(
            state.pixels, prof, cam_spec, is_video=bool(capture_mode == 1), frame_index=frame_index, seed=seed
        )
        state.pixels = cam_rgb
        state.width = prof.video_width if capture_mode == 1 else prof.sensor_width
        state.height = prof.video_height if capture_mode == 1 else prof.sensor_height
        state.frame_rate = prof.video_fps if capture_mode == 1 else fps
        state.pixel_format = prof.chroma_subsampling
        state.codec = prof.video_codec if capture_mode == 1 else prof.photo_format
        state.bitrate = prof.video_bitrate_kbps if capture_mode == 1 else 1200.0
        state.metadata.update(cam_info.get("encoding", {}).get("exif", {}))
        state.metadata["Model"] = prof.name
        state.metadata["CaptureMode"] = "video_record" if capture_mode == 1 else "photo_capture"
        state.append_event(
            TransferEvent(
                source_device="optical_scene",
                destination=prof.key,
                method="record" if capture_mode == 1 else "capture",
                pixel_transform=f"{state.width}x{state.height}_{state.codec}",
                reencoded=True,
                metadata_policy="write_exif_or_box",
                codec_after=state.codec,
                bitrate_kbps_after=state.bitrate,
                resolution_after=f"{state.width}x{state.height}",
            )
        )

    # 03 · 文件与传输
    if transfer_method > 0:
        tr_spec = FileTransferSpec(
            enabled=True,
            method=transfer_method,
            generations=transfer_generations,
        )
        state = apply_file_transfer_node(state, tr_spec, frame_index=frame_index)

    # 04 · 平台处理
    if platform_tier > 0:
        state = apply_platform_transcode(
            state,
            tier_id=platform_tier,
            source_origin=source_origin,
            platform_watermark=False,
            frame_index=frame_index,
        )

    # 05 · 播放与显示
    disp_info: dict[str, Any] = {"active": False}
    out_rgb = state.pixels
    if display_device > 0:
        lcd_spec = EarlyLCDFlashSpec(
            enabled=True,
            display_device=display_device,
            flash_scale_mode=flash_scale_mode,
            flash_chrome_overlay=flash_chrome,
        )
        out_rgb, disp_info = apply_early_lcd_and_flash_player(state, lcd_spec, frame_index=frame_index)

    report = {
        "camera": cam_info,
        "media_state": state.summary_dict(),
        "display": disp_info,
    }
    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32), state, report


def run_ldme_benchmark_experiment(
    rgb: np.ndarray,
    benchmark_id: int = 1,
    variant_id: int = 1,
    split_view: bool = False,
    frame_index: int = 0,
    fps: float = 25.0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    执行第七节 F1 三组标准基准测试（默认单画面全幅输出 `split_view=False`，绝不强制四分屏）：
      - `benchmark_id = 1`（测试一：同一场景，不同手机）：
        `variant_id`: 1=Nokia 6600, 2=Nokia N73, 3=Nokia 5800 3GP, 4=iPhone 2007, 5=iPhone 3GS, 6=iPhone 4S
      - `benchmark_id = 2`（测试二：同一视频，不同转码链）：
        `variant_id`: 1=单次 FLV 320×240 转码, 2=多代转码链 (NLE 导出 -> 优酷 FLV -> 二次下载重传), 3=高清源转标清 H.264 480×360, 4=低帧率 15fps 3GP 转 FLV
      - `benchmark_id = 3`（测试三：同一文件，不同播放设备）：
        `variant_id`: 1=早期手机 TFT 液晶屏 (RGB565), 2=2007 桌面 6-bit TN 液晶显示器 (Flash 缩放), 3=经典 CRT 监视器对照
    """
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    h, w = src.shape[:2]

    if benchmark_id == 1:
        # 测试一：同一场景，不同手机
        prof_map = {1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 8}
        p_id = prof_map.get(int(np.clip(variant_id, 1, 6)), 2)
        cap_mode = 1 if p_id in (4, 7) else 0
        if not split_view:
            out, st, rep = run_ldme_full_chain(
                src, camera_profile=p_id, capture_mode=cap_mode, frame_index=frame_index, fps=fps, seed=seed
            )
            rep["benchmark"] = "test1_same_scene_different_phones"
            rep["variant_id"] = variant_id
            return out, rep
        # 若用户显式请求分屏对照，生成 2x2 代表机型对照
        tiles = [
            run_ldme_full_chain(src, camera_profile=1, capture_mode=0, frame_index=frame_index, seed=seed)[0],
            run_ldme_full_chain(src, camera_profile=2, capture_mode=0, frame_index=frame_index, seed=seed)[0],
            run_ldme_full_chain(src, camera_profile=5, capture_mode=0, frame_index=frame_index, seed=seed)[0],
            run_ldme_full_chain(src, camera_profile=7, capture_mode=1, frame_index=frame_index, seed=seed)[0],
        ]
        return _compose_2x2_grid(tiles, h, w), {"benchmark": "test1_grid"}

    if benchmark_id == 2:
        # 测试二：同一视频，不同转码链
        v_id = int(np.clip(variant_id, 1, 4))
        if v_id == 1:
            # 单次压缩：早期 Flash FLV 320x240 350kbps
            out, st, rep = run_ldme_full_chain(
                src, camera_profile=0, transfer_method=0, platform_tier=1, source_origin=0, frame_index=frame_index, fps=fps
            )
        elif v_id == 2:
            # 多代有损级联：手机拍摄 -> 桌面剪辑软件导出 -> 视频网站多代重编码
            out, st, rep = run_ldme_full_chain(
                src, camera_profile=2, capture_mode=1, transfer_method=4, transfer_generations=3,
                platform_tier=1, source_origin=1, frame_index=frame_index, fps=fps
            )
        elif v_id == 3:
            # 高清源被平台降为标清 480x360 H.264 700kbps
            out, st, rep = run_ldme_full_chain(
                src, camera_profile=0, transfer_method=0, platform_tier=2, source_origin=0, frame_index=frame_index, fps=fps
            )
        else:
            # 原生低帧率 15fps 3GP 视频上传平台转码（时间采样顿挫 + 空间块效应叠加）
            out, st, rep = run_ldme_full_chain(
                src, camera_profile=4, capture_mode=1, transfer_method=0, platform_tier=1, source_origin=1, frame_index=frame_index, fps=fps
            )
        rep["benchmark"] = "test2_same_video_different_transcode_chains"
        rep["variant_id"] = v_id
        return out, rep

    # benchmark_id == 3: 测试三：同一文件，不同播放设备（固定同一份解码后的标清流媒体文件，分别输出到手机 LCD、电脑 TN LCD 与 CRT）
    v_id = int(np.clip(variant_id, 1, 3))
    out, st, rep = run_ldme_full_chain(
        src,
        camera_profile=0,
        transfer_method=0,
        platform_tier=2,          # 固定解码源：标清 480x360 H.264 文件
        display_device=v_id,      # 1=手机 TFT LCD, 2=电脑 6-bit TN LCD, 3=CRT 监视器
        flash_scale_mode=1 if v_id == 2 else 0,
        frame_index=frame_index,
        fps=fps,
    )
    rep["benchmark"] = "test3_same_file_different_displays"
    rep["variant_id"] = v_id
    return out, rep


def _compose_2x2_grid(tiles: list[np.ndarray], h: int, w: int) -> np.ndarray:
    qh, qw = h // 2, w // 2
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    coords = [(0, qh, 0, qw), (0, qh, qw, w), (qh, h, 0, qw), (qh, h, qw, w)]
    for (y0, y1, x0, x1), img in zip(coords, tiles):
        canvas[y0:y1, x0:x1] = _resample_to(img, y1 - y0, x1 - x0)
    return canvas


# ---------------------------------------------------------------------------
# 六、独立节点图 Stage 封装 (`file_transfer_node`, `lcd_flash_player`)
# ---------------------------------------------------------------------------

class FileTransferStage(Stage):
    """节点图阶段 `file_transfer_node`：03 · 旧手机与视频网站之间的文件传输与多代压缩链。"""
    name = "file_transfer_node"
    layers = ("codec",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("codec")
        self.spec = FileTransferSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.method <= 0:
                self.spec.method = 4

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        state = MediaState.from_rgb(rgb_in, fps=float(self.ctx.fps))
        if isinstance(buf.meta.get("ldme_media_state"), dict):
            state.history = list(buf.meta["ldme_media_state"].get("history", []))
        state = apply_file_transfer_node(state, self.spec, frame_index=int(buf.index))
        out_rgb = state.pixels
        meta = dict(buf.meta)
        meta["ldme_media_state"] = state.summary_dict()
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)


class LCDFlashPlayerStage(Stage):
    """节点图阶段 `lcd_flash_player`：05 · 早期 LCD 显示设备与 Flash 播放器缩放/缓冲引擎。"""
    name = "lcd_flash_player"
    layers = ("display",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._prev_rgb: np.ndarray | None = None
        self._read_params()

    def _read_params(self) -> None:
        cfg = dict(self.cfg_of("display"))
        if hasattr(self.ctx, "params") and self.ctx.params:
            for cand in ("codec", "film", "overlay"):
                sub = self.ctx.params.get(cand)
                if isinstance(sub, dict):
                    if "stream_show_text" in sub:
                        cfg.setdefault("stream_show_text", sub["stream_show_text"])
                    if "show_text" in sub:
                        cfg.setdefault("show_text", sub["show_text"])
        self.spec = EarlyLCDFlashSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.display_device <= 0:
                self.spec.display_device = 2

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        state = MediaState.from_rgb(rgb_in, fps=float(self.ctx.fps))
        out_rgb, info = apply_early_lcd_and_flash_player(
            state, self.spec, frame_index=int(buf.index), prev_rgb=self._prev_rgb
        )
        self._prev_rgb = rgb_in.copy()
        meta = dict(buf.meta)
        meta["ldme_lcd_flash"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

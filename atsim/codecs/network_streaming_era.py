"""
网络视听传播、平台转码与流媒体协议系统 (`network_streaming_era`)
==========================================================================

实现《完整的视听媒介生态模拟器》第四节（方向 B+C：从模拟广播进入互联网视频时代）：

1. **早期互联网流媒体与现代自适应流媒体编解码器 (`simulate_streaming_codec_era`)**：
   - **Era 1 (`mode = 1`, 1999 RealVideo G2 / Windows Media Video 拨号窄带流 `28.8–128 kbps`)**：
     - 160×120 / 240p 强空间下采样、极重帧内环路去块滤波（涂抹水彩感）、色彩饱和度坍塌，以及在每个 GOP 关键帧 (`frame_index % gop_len == 0`) 处清晰度突然跃升、随后在 P 帧预测中迅速糊化的 **关键帧呼吸脉冲 (Keyframe Breathing Pulse)**；
   - **Era 2 (`mode = 2`, 2006 Flash VP6 / 早期土豆优酷 H.264 平台上传二次转码 `360p 4:2:0`)**：
     - 建模 `广播电视 → 录像带 → 采集卡 → 视频网站多次上传重压 (stream_platform_generations)`：多代 `4:2:0` 色度抽样台阶、BT.601↔BT.709 矩阵误转色偏、平台固定 CRF 8×8/16×16 宏块伪影，以及历代平台在右上/右下叠加的半透明残影水印台标；
   - **Era 3 (`mode = 3`, 现代 HLS / DASH ABR 自适应码率流媒体)**：
     - 基于分组网络吞吐量波动 (`stream_bw_fluctuation`) 与客户端播放缓冲区水位 `buffer_level_s` 状态机，动态在 `1080p -> 720p -> 480p -> 240p` 阶梯间触发 **ABR 自适应分辨率阶跃切换 (ABR Resolution Step)**；当缓冲区耗尽 (`buffer_level_s <= 0`) 时触发 **TCP/HLS 预缓冲卡顿转圈 (Rebuffering Stall)** 画面冻结与恢复后的音画同步追赶；
   - **Era 4 (`mode = 4`, UDP / RTP 实时流媒体丢包切片撕裂与参考帧误差扩散)**：
     - 在无重传的 UDP/RTP 实时传输中，分组丢包 (`stream_packet_loss`) 表现为水平宏块条带切片丢失 (**Slice Tear / Green-Magenta Macroblock Corruption**)，并在下一个 I 帧到来前沿运动矢量向后续 P 帧持续扩散 (**Inter-Frame Error Propagation**)。

2. **`2×2` 跨时代网络视听传播与多平台重编码对照实验室 (`stream_mode = 5`)**：
   - 同屏实时对比 `1. 1999 RealVideo 拨号窄带流 (关键帧呼吸脉冲)`、`2. 2006 视频平台多代转码与叠加水印`、`3. HLS/DASH ABR 自适应分辨率阶跃与缓冲卡顿`、`4. UDP/RTP 分组丢包切片撕裂与误差扩散`，完整支持 `stream_quad_corner = 0..4`、`stream_show_border = 0..2` 与 `stream_show_text = 0..2`。
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
class NetworkStreamingSpec:
    enabled: bool = False
    mode: int = 0
    bitrate_kbps: float = 380.0
    bw_fluctuation: float = 0.45
    rtt_ms: float = 85.0
    packet_loss: float = 0.08
    gop_len: int = 15
    platform_generations: int = 2
    quad_corner: int = 0
    show_border: int = 2
    show_text: int = 0

    def is_active(self) -> bool:
        return bool(self.enabled and self.mode > 0)

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "NetworkStreamingSpec":
        en = float(cfg.get("stream_enabled", 0.0)) > 0.5
        md = int(round(float(cfg.get("stream_mode", 0.0))))
        if md > 0:
            en = True
        elif en and md == 0:
            md = 5
        return cls(
            enabled=en,
            mode=int(np.clip(md, 0, 5)),
            bitrate_kbps=float(np.clip(cfg.get("stream_bitrate_kbps", 380.0), 16.0, 8000.0)),
            bw_fluctuation=float(np.clip(cfg.get("stream_bw_fluctuation", 0.45), 0.0, 1.0)),
            rtt_ms=float(np.clip(cfg.get("stream_rtt_ms", 85.0), 5.0, 800.0)),
            packet_loss=float(np.clip(cfg.get("stream_packet_loss", 0.08), 0.0, 0.50)),
            gop_len=max(4, int(round(float(cfg.get("stream_gop_len", 15.0))))),
            platform_generations=int(np.clip(round(float(cfg.get("stream_platform_generations", 2.0))), 0, 5)),
            quad_corner=int(np.clip(round(float(cfg.get("stream_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(cfg.get("stream_show_border", 2.0))), 0, 2)),
            show_text=int(np.clip(round(float(cfg.get("stream_show_text", 0.0))), 0, 2)),
        )


def simulate_network_transport_and_abr(
    spec: NetworkStreamingSpec,
    frame_index: int = 0,
    fps: float = 29.97,
) -> dict[str, Any]:
    """
    Compute packet-switched network throughput, TCP/UDP transport dynamics,
    GOP keyframe breathing phase, ABR ladder selection, and client buffer state.
    """
    t_sec = float(frame_index) / max(1.0, float(fps))
    gop = max(4, int(spec.gop_len))
    gop_pos = int(frame_index) % gop
    is_keyframe = (gop_pos == 0)
    # Keyframe quality decay across the GOP: 1.0 at I-frame, decaying toward P-frames before next I-frame
    gop_decay = float(np.exp(-1.35 * (gop_pos / max(1.0, float(gop - 1)))))

    # Instantaneous available network bandwidth (kbps) with sinusoidal + burst congestion drops
    fluct = float(np.clip(spec.bw_fluctuation, 0.0, 1.0))
    wave = 0.55 * np.sin(2.0 * np.pi * 0.22 * t_sec + 0.4) + 0.45 * np.cos(2.0 * np.pi * 0.47 * t_sec)
    # TCP Mathis throughput reduction under RTT and packet loss
    loss = float(np.clip(spec.packet_loss, 0.0, 0.50))
    rtt = float(np.clip(spec.rtt_ms, 5.0, 800.0))
    tcp_penalty = 1.0 / (1.0 + 3.2 * np.sqrt(loss) * (rtt / 80.0))
    inst_bw_kbps = float(
        max(
            18.0,
            spec.bitrate_kbps
            * (1.0 - 0.68 * fluct * (0.5 - 0.5 * wave))
            * tcp_penalty,
        )
    )

    # ABR resolution ladder selection
    if inst_bw_kbps >= 1800.0:
        abr_profile = "1080p_HIGH"
        abr_scale = 1.00
    elif inst_bw_kbps >= 650.0:
        abr_profile = "720p_MED"
        abr_scale = 0.72
    elif inst_bw_kbps >= 260.0:
        abr_profile = "360p_LOW"
        abr_scale = 0.44
    else:
        abr_profile = "240p_EMERGENCY"
        abr_scale = 0.24

    # Client playback buffer level (seconds) & Rebuffering Stall state
    # Drain rate exceeds fill rate when inst_bw_kbps < target bitrate or loss/fluctuation is high
    fill_ratio = inst_bw_kbps / max(32.0, float(spec.bitrate_kbps))
    buffer_level_s = float(np.clip(2.4 * fill_ratio - 1.8 * fluct - 3.5 * loss + 0.45 * np.sin(t_sec * 1.7), 0.0, 4.0))
    rebuffering_stall = bool(buffer_level_s < 0.38 and (fluct > 0.55 or loss > 0.14))

    # A/V sync drift & catch-up speed multiplier after stall/jitter
    av_sync_offset_ms = float(round((rtt * 0.35 + loss * 420.0) * (1.2 if rebuffering_stall else 0.4), 2))
    catchup_speed = 1.08 if (not rebuffering_stall and av_sync_offset_ms > 45.0) else (0.0 if rebuffering_stall else 1.0)

    return {
        "frame_index": int(frame_index),
        "gop_len": int(gop),
        "gop_pos": int(gop_pos),
        "is_keyframe": bool(is_keyframe),
        "keyframe_quality_pulse": round(gop_decay, 4),
        "nominal_bitrate_kbps": round(float(spec.bitrate_kbps), 1),
        "instantaneous_bw_kbps": round(inst_bw_kbps, 1),
        "rtt_ms": round(rtt, 1),
        "packet_loss_rate": round(loss, 4),
        "abr_profile": abr_profile,
        "abr_resolution_scale": round(abr_scale, 3),
        "buffer_level_s": round(buffer_level_s, 3),
        "rebuffering_stall": rebuffering_stall,
        "av_sync_offset_ms": av_sync_offset_ms,
        "catchup_speed": round(catchup_speed, 3),
    }


def _quantize_macroblocks(
    rgb: np.ndarray,
    block_size: int,
    q_step: float,
    chroma_subsample: int = 2,
) -> np.ndarray:
    """Fast block-grid DCT/spatial quantization + 4:2:0 chroma subsampling."""
    h, w = rgb.shape[:2]
    bs = max(2, int(block_size))
    # Luma/Chroma split
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cb = rgb[..., 2] - y
    cr = rgb[..., 0] - y

    # Chroma subsampling (4:2:0 or 4:1:0)
    cs = max(1, int(chroma_subsample))
    if cs > 1 and h >= cs * 2 and w >= cs * 2:
        cb_small = _resample_to(cb[..., None], max(2, h // cs), max(2, w // cs))[..., 0]
        cr_small = _resample_to(cr[..., None], max(2, h // cs), max(2, w // cs))[..., 0]
        cb = _resample_to(cb_small[..., None], h, w)[..., 0]
        cr = _resample_to(cr_small[..., None], h, w)[..., 0]

    # Block average + quantized AC residual
    gh = max(2, h // bs)
    gw = max(2, w // bs)
    y_blk = _resample_to(_resample_to(y[..., None], gh, gw), h, w)[..., 0]
    ac = y - y_blk
    if q_step > 1e-4:
        y_blk_q = np.round(y_blk / q_step) * q_step
        ac_q = np.round(ac / (q_step * 1.65)) * (q_step * 1.65)
        y_out = 0.72 * y_blk_q + 0.28 * (y_blk + ac_q)
    else:
        y_out = y

    r = y_out + cr
    b = y_out + cb
    g = (y_out - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def _render_branch_1_realvideo_1999(
    rgb: np.ndarray,
    spec: NetworkStreamingSpec,
    net: dict[str, Any],
) -> np.ndarray:
    """Era 1: 1999 RealVideo G2 / WMV dial-up stream with heavy loop filter & keyframe breathing."""
    h, w = rgb.shape[:2]
    pulse = float(net["keyframe_quality_pulse"])  # 1.0 at I-frame, ~0.26 at end of GOP
    bitrate_factor = float(np.clip(spec.bitrate_kbps / 450.0, 0.12, 1.0))
    # Spatial resolution and block quantization breathe with the GOP keyframe pulse
    eff_scale = float(np.clip(0.24 + 0.22 * bitrate_factor + 0.22 * pulse, 0.18, 0.72))
    sh = max(16, int(round(h * eff_scale)))
    sw = max(16, int(round(w * eff_scale)))
    small = _resample_to(rgb, sh, sw)
    q_step = float(np.clip((0.11 - 0.06 * pulse) * (1.15 - 0.55 * bitrate_factor), 0.015, 0.16))
    blk = _quantize_macroblocks(small, block_size=4, q_step=q_step, chroma_subsample=3)
    up = _resample_to(blk, h, w)
    # RealVideo G2 heavy in-loop smoothing between keyframes (waxy/watercolor look on P-frames)
    blur_sigma = float(np.clip((1.85 - 1.35 * pulse) * (1.2 - 0.5 * bitrate_factor), 0.2, 2.8))
    smoothed = lowpass(up, blur_sigma, axis=1)
    smoothed = lowpass(smoothed, blur_sigma * 0.7, axis=0)
    # Dial-up color saturation loss
    luma = (0.299 * smoothed[..., 0] + 0.587 * smoothed[..., 1] + 0.114 * smoothed[..., 2])[..., None]
    out = luma + (smoothed - luma) * (0.62 + 0.25 * pulse)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _render_branch_2_platform_retranscode(
    rgb: np.ndarray,
    spec: NetworkStreamingSpec,
) -> np.ndarray:
    """Era 2: Multi-generation platform upload transcode (4:2:0 + BT.601/709 matrix shift + watermarks)."""
    out = rgb.copy()
    h, w = out.shape[:2]
    gens = max(1, int(spec.platform_generations))
    for g_idx in range(gens):
        scale_g = max(0.32, 0.78 - 0.10 * g_idx)
        sub = _resample_to(out, max(16, int(h * scale_g)), max(16, int(w * scale_g)))
        out = _resample_to(
            _quantize_macroblocks(sub, block_size=8, q_step=0.028 + 0.014 * g_idx, chroma_subsample=2),
            h,
            w,
        )
        # Subtle BT.601 -> BT.709 matrix mismatch & platform unsharp halo per generation
        hp = out - lowpass(out, 1.1, axis=1)
        out = np.clip(out + 0.18 * hp, 0.0, 1.0)
        out[..., 0] = np.clip(out[..., 0] * 1.02 - 0.008, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] * 0.985 + 0.006, 0.0, 1.0)

        # Stacked semi-transparent platform corner watermarks from successive uploads
        wm_h = max(10, int(h * 0.065))
        wm_w = max(36, int(w * 0.22))
        if g_idx % 2 == 0:
            y0, x0 = 6 + g_idx * 3, max(4, w - wm_w - 6)
        else:
            y0, x0 = max(4, h - wm_h - 6 - g_idx * 2), max(4, w - wm_w - 8)
        y1, x1 = min(h, y0 + wm_h), min(w, x0 + wm_w)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * 0.68 + np.array([0.92, 0.88, 0.78], dtype=np.float32) * 0.32
            if spec.show_text >= 1:
                draw_text(out, x0 + 3, y0 + 2, f"TV-WEB G{g_idx+1}", (0.15, 0.15, 0.18), scale=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _render_branch_3_abr_and_stall(
    rgb: np.ndarray,
    spec: NetworkStreamingSpec,
    net: dict[str, Any],
) -> np.ndarray:
    """Era 3: Modern HLS/DASH ABR resolution stepping & TCP Rebuffering Stall spinner."""
    h, w = rgb.shape[:2]
    abr_scale = float(net["abr_resolution_scale"])
    sh = max(16, int(round(h * abr_scale)))
    sw = max(16, int(round(w * abr_scale)))
    sub = _resample_to(rgb, sh, sw)
    q_step = 0.010 if abr_scale >= 0.7 else (0.032 if abr_scale >= 0.4 else 0.068)
    out = _resample_to(_quantize_macroblocks(sub, block_size=8, q_step=q_step, chroma_subsample=2), h, w)

    if net["rebuffering_stall"]:
        # Dim frozen frame and draw TCP/HLS buffering ring in the center
        out = out * 0.48
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = h * 0.5, w * 0.5
        rad = np.hypot(yy - cy, xx - cx)
        r_ring = max(10.0, min(h, w) * 0.095)
        ring = np.abs(rad - r_ring) <= max(1.8, min(h, w) * 0.015)
        ang = np.arctan2(yy - cy, xx - cx)
        spin_phase = (float(net["frame_index"]) * 0.45) % (2.0 * np.pi)
        arc = ((ang - spin_phase + np.pi) % (2.0 * np.pi)) < (1.45 * np.pi)
        mask = ring & arc
        out[mask] = np.array([0.95, 0.96, 0.98], dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _render_branch_4_udp_packet_loss(
    rgb: np.ndarray,
    spec: NetworkStreamingSpec,
    net: dict[str, Any],
    seed: int = 20240101,
) -> np.ndarray:
    """Era 4: UDP/RTP real-time stream packet-loss slice tearing & P-frame motion vector error propagation."""
    h, w = rgb.shape[:2]
    out = _quantize_macroblocks(rgb, block_size=16, q_step=0.024, chroma_subsample=2)
    loss = float(spec.packet_loss)
    if loss <= 1e-4:
        return out

    gop_pos = int(net["gop_pos"])
    # Packet loss errors accumulate across P-frames until the next I-frame clears the reference buffer
    accum_factor = 1.0 + 0.18 * float(gop_pos)
    rng = np.random.default_rng((int(seed) * 1664525 + int(net["frame_index"]) * 1013904223) & 0xFFFFFFFF)
    slice_h = max(8, h // 15)
    n_slices = h // slice_h
    for s_idx in range(n_slices):
        if rng.random() < min(0.82, loss * accum_factor):
            y0 = s_idx * slice_h
            y1 = min(h, (s_idx + 1) * slice_h)
            x_tear = int(rng.integers(0, max(1, w // 2)))
            dx = float(rng.integers(-28, 29))
            # Displaced P-frame motion vector tear + corrupted chroma macroblocks (green/magenta block flash)
            shifted = horizontal_shift(out[y0:y1], np.full(y1 - y0, dx, dtype=np.float32))
            out[y0:y1, x_tear:] = shifted[:, x_tear:]
            if rng.random() < 0.55:
                tint = np.array([0.18, 0.88, 0.24], dtype=np.float32) if (s_idx % 2 == 0) else np.array([0.86, 0.20, 0.82], dtype=np.float32)
                out[y0:y1, x_tear:] = 0.55 * out[y0:y1, x_tear:] + 0.45 * tint[None, None, :]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_network_streaming_era(
    rgb: np.ndarray,
    spec: NetworkStreamingSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Option S Network Streaming & Multi-Platform Transcode Pipeline."""
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    net = simulate_network_transport_and_abr(spec, frame_index=frame_index, fps=fps)
    if not spec.enabled or spec.mode <= 0:
        return src, {"active": False, "network": net}

    b1 = _render_branch_1_realvideo_1999(src, spec, net)
    b2 = _render_branch_2_platform_retranscode(src, spec)
    b3 = _render_branch_3_abr_and_stall(src, spec, net)
    b4 = _render_branch_4_udp_packet_loss(src, spec, net, seed=seed)

    # Measure keyframe vs non-keyframe sharpness on Branch 1 for telemetry
    hf_b1 = float(np.mean(np.abs(np.diff(b1, axis=1))))
    hf_src = float(np.mean(np.abs(np.diff(src, axis=1))))

    report = {
        "active": True,
        "mode": int(spec.mode),
        "stream_quad_corner": int(spec.quad_corner),
        "stream_show_border": int(spec.show_border),
        "stream_show_text": int(spec.show_text),
        "network": net,
        "metrics": {
            "source_hf": round(hf_src, 5),
            "realvideo_hf": round(hf_b1, 5),
            "platform_generations": int(spec.platform_generations),
            "udp_vs_src_l1": round(float(np.mean(np.abs(b4 - src))), 4),
            "abr_vs_src_l1": round(float(np.mean(np.abs(b3 - src))), 4),
        },
    }

    if spec.mode == 1:
        out = b1
    elif spec.mode == 2:
        out = b2
    elif spec.mode == 3:
        out = b3
    elif spec.mode == 4:
        out = b4
    else:
        # Mode 5: 2x2 Quad Lab (or full-frame single corner when quad_corner in 1..4)
        h, w = src.shape[:2]
        c_sel = int(spec.quad_corner)
        if c_sel in (1, 2, 3, 4):
            chosen = {1: b1, 2: b2, 3: b3, 4: b4}[c_sel]
            out = chosen.copy()
            if spec.show_text >= 1:
                titles = {
                    1: f"1.1999 REALVIDEO {int(spec.bitrate_kbps)}kbps GOP:{net['gop_pos']}/{net['gop_len']}",
                    2: f"2.PLATFORM RE-TRANSCODE GEN:{spec.platform_generations} (4:2:0+WM)",
                    3: f"3.HLS/DASH ABR:{net['abr_profile']} BUF:{net['buffer_level_s']:.1f}s",
                    4: f"4.UDP/RTP LOSS:{int(spec.packet_loss*100)}% RTT:{int(spec.rtt_ms)}ms",
                }
                out[:18, :] *= 0.25
                draw_text(out, 6, 5, titles[c_sel], (0.45, 0.95, 0.85), scale=1)
        else:
            qh, qw = h // 2, w // 2
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            quads = [
                (0, qh, 0, qw, b1, f"1.1999 REALVIDEO {int(spec.bitrate_kbps)}K", (0.95, 0.85, 0.40)),
                (0, qh, qw, w, b2, f"2.WEB RE-ENCODE x{spec.platform_generations}", (0.45, 0.92, 0.95)),
                (qh, h, 0, qw, b3, f"3.ABR:{net['abr_profile'][:4]} BUF:{net['buffer_level_s']:.1f}s", (0.50, 0.95, 0.55)),
                (qh, h, qw, w, b4, f"4.UDP LOSS:{int(spec.packet_loss*100)}% SLICE", (0.95, 0.55, 0.45)),
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
                if spec.show_border >= 2:
                    canvas[:2, :] = 0.28
                    canvas[-2:, :] = 0.28
                    canvas[:, :2] = 0.28
                    canvas[:, -2:] = 0.28
            out = canvas

    return np.clip(out, 0.0, 1.0).astype(np.float32), report


def compute_streaming_audio_effect(
    stereo_in: np.ndarray,
    spec: NetworkStreamingSpec,
    sample_rate: int = 44100,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Simulate low-bitrate streaming audio (RealAudio / MP3 / HE-AAC spectral quantization,
    rebuffering silence gap, and UDP packet-loss concealment chirps).
    """
    arr = np.asarray(stereo_in, dtype=np.float32)
    if not spec.enabled or spec.mode <= 0 or arr.size == 0:
        return arr, {"active": False}

    n_samples = arr.shape[0]
    net = simulate_network_transport_and_abr(spec, frame_index=int(round(start_s * 30.0)), fps=30.0)
    out = arr.copy()

    # Low-bitrate MDCT subband quantization + HF cutoff ("swirly" MP3/RealAudio effect)
    bw_ratio = float(np.clip(spec.bitrate_kbps / 640.0, 0.08, 1.0))
    decim = max(1, int(round(6.0 * (1.0 - 0.82 * bw_ratio))))
    if decim > 1 and n_samples > decim * 2:
        for ch in range(out.shape[1]):
            sub = out[::decim, ch]
            out[:, ch] = np.interp(
                np.arange(n_samples, dtype=np.float32),
                np.arange(0, n_samples, decim, dtype=np.float32)[: len(sub)],
                sub,
            )
    q_levels = max(12.0, 180.0 * bw_ratio)
    out = np.round(out * q_levels) / q_levels

    # Rebuffering stall mutes segment in middle of window
    if net["rebuffering_stall"] and spec.mode in (3, 5):
        s0 = int(n_samples * 0.30)
        s1 = int(n_samples * 0.65)
        out[s0:s1, :] *= 0.02

    return np.clip(out, -1.0, 1.0).astype(np.float32), {
        "active": True,
        "decim_factor": decim,
        "rebuffering_stall": net["rebuffering_stall"],
        "abr_profile": net["abr_profile"],
    }


class StreamCDNClientStage(Stage):
    """Node-graph stage for `stream_cdn_client` (**模块**)."""
    name = "stream_cdn_client"
    layers = ("codec",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("codec")
        self.spec = NetworkStreamingSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.mode <= 0:
                self.spec.mode = 5

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        out_rgb, info = apply_network_streaming_era(
            rgb_in,
            self.spec,
            frame_index=int(buf.index),
            fps=float(self.ctx.fps),
            seed=int(self.ctx.seed),
        )
        meta = dict(buf.meta)
        meta["network_streaming"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

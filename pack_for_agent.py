#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 D:\\analog-tape-sim 的源码与 tools\\calib\\clip 实拍素材打包为 <9 MB 的轻量压缩包，
以便直接拖入对话框上传或放入 Google Drive 读取（绕过单文件 10 MB 限制）。

用法（在 D:\\analog-tape-sim 目录下执行）：
    python pack_for_agent.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "tools" / "calib" / "clip"
OUT_DIR = ROOT / "_agent_upload"

CODE_EXTENSIONS = {".py", ".md", ".txt", ".json", ".js", ".css", ".html", ".bat"}
SKIP_DIRS = {".git", "__pycache__", ".server", "runs", "_agent_upload", ".venv", "venv", "node_modules"}
VIDEO_EXTENSIONS = {".mp4", ".mpg", ".mpeg", ".avi", ".mov", ".mkv", ".m4v", ".dv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MAX_ZIP_BYTES = 8_500_000  # 严格控制在 8.5 MB 以内，确保低于 10 MB 限制


def pack_code_archive() -> Path:
    """打包全部源码、文档与核心校准参考帧（不含大体积视频与渲染输出）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "atsim_code.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            rel_dir = Path(dirpath).relative_to(ROOT)

            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                rel_path = rel_dir / fname
                ext = fpath.suffix.lower()

                # 收录所有源码与文档文件（单文件不超过 2 MB）
                if ext in CODE_EXTENSIONS and fpath.stat().st_size < 2_000_000:
                    zf.write(fpath, rel_path.as_posix())

        # 额外收录 verify.py / nv_frame_fit_figure.py 依赖的关键参考图（自动转为高质量大图或原图）
        ref_candidates = [
            ROOT / "tools" / "nightshot" / "user_real.png",
            ROOT / "tools" / "nightshot" / "user_now.png",
            ROOT / "tools" / "nightshot" / "ref_ns2_a.png",
            ROOT / "tools" / "nightshot" / "ref_ns2_b.png",
            ROOT / "tools" / "nightshot" / "ref_c0147.png",
            ROOT / "tools" / "nightshot" / "sc1515" / "d_nv3.png",
        ]
        for ref in ref_candidates:
            if ref.exists() and ref.stat().st_size < 3_500_000:
                if zip_path.stat().st_size < MAX_ZIP_BYTES - 1_500_000:
                    zf.write(ref, ref.relative_to(ROOT).as_posix())

    return zip_path


def probe_video(vpath: Path) -> Dict:
    """使用 ffprobe 获取视频元数据。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-show_entries", "format=duration,size",
        "-of", "json",
        str(vpath),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout)
    except Exception as exc:
        return {"error": str(exc)}


def extract_frames_ffmpeg(vpath: Path, max_frames: int = 40) -> List[np.ndarray]:
    """用 ffmpeg 读取视频前 N 帧或均匀抽样帧（RGB24）。"""
    meta = probe_video(vpath)
    streams = meta.get("streams", [{}])
    w = int(streams[0].get("width", 0) or 0)
    h = int(streams[0].get("height", 0) or 0)
    if w <= 0 or h <= 0:
        return []

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(vpath),
        "-frames:v", str(max_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    raw = proc.stdout
    frame_bytes = w * h * 3
    n = len(raw) // frame_bytes
    frames = []
    for i in range(n):
        arr = np.frombuffer(raw[i * frame_bytes : (i + 1) * frame_bytes], dtype=np.uint8).reshape((h, w, 3))
        frames.append(arr)
    return frames


def compute_frame_stats(rgb_u8: np.ndarray, crop_osd: bool = True) -> Dict:
    """计算单帧统计量（默认排除上下边缘 OSD 区域，同时记录全画幅统计）。"""
    arr = rgb_u8.astype(np.float32) / 255.0
    h, w, _ = arr.shape
    if crop_osd and h > 100 and w > 100:
        # 排除顶部/底部 14% 与左右 10% 的 OSD 图标区域
        y0, y1 = int(h * 0.14), int(h * 0.86)
        x0, x1 = int(w * 0.10), int(w * 0.90)
        roi = arr[y0:y1, x0:x1]
    else:
        roi = arr

    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    g_mean = float(np.mean(g)) + 1e-8
    r_mean = float(np.mean(r))
    b_mean = float(np.mean(b))

    # 逐像素色度幅度 (std across RGB channels or max-min)
    chroma_std = float(np.mean(np.std(roi, axis=-1)))

    return {
        "mean": round(float(np.mean(luma)), 5),
        "std": round(float(np.std(luma)), 5),
        "p05": round(float(np.percentile(luma, 5)), 5),
        "p25": round(float(np.percentile(luma, 25)), 5),
        "p50": round(float(np.percentile(luma, 50)), 5),
        "p75": round(float(np.percentile(luma, 75)), 5),
        "p95": round(float(np.percentile(luma, 95)), 5),
        "p99": round(float(np.percentile(luma, 99)), 5),
        "R_mean": round(r_mean, 5),
        "G_mean": round(g_mean, 5),
        "B_mean": round(b_mean, 5),
        "R_over_G": round(r_mean / g_mean, 5),
        "B_over_G": round(b_mean / g_mean, 5),
        "chroma_std": round(chroma_std, 5),
        "dark_lt_015_pct": round(float(np.mean(luma < 0.15) * 100.0), 3),
        "pinned_gt_095_pct": round(float(np.mean(luma > 0.95) * 100.0), 3),
    }


def pack_calib_archive() -> Path:
    """扫描 tools/calib 下所有实拍视频与图片，生成多帧统计 JSON + 连续抽帧样本包 (<8.5 MB)。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "calib_samples.zip"

    calib_root = ROOT / "tools" / "calib"
    if not calib_root.exists():
        calib_root = CALIB_DIR

    manifest = {"calib_root": str(calib_root), "files": []}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        if calib_root.exists():
            for dirpath, _, filenames in os.walk(calib_root):
                for fname in sorted(filenames):
                    fpath = Path(dirpath) / fname
                    rel_path = fpath.relative_to(ROOT).as_posix()
                    ext = fpath.suffix.lower()

                    if ext in CODE_EXTENSIONS and fpath.stat().st_size < 1_000_000:
                        zf.write(fpath, rel_path)
                        continue

                    if ext in IMAGE_EXTENSIONS:
                        try:
                            im = Image.open(fpath).convert("RGB")
                            arr = np.asarray(im)
                            info = {
                                "path": rel_path,
                                "type": "image",
                                "shape": list(arr.shape),
                                "stats_osd_masked": compute_frame_stats(arr, crop_osd=True),
                                "stats_full": compute_frame_stats(arr, crop_osd=False),
                            }
                            manifest["files"].append(info)
                            # 保存高质量 JPEG 副本以控制体积
                            buf = io.BytesIO()
                            im.save(buf, format="JPEG", quality=95)
                            zf.writestr(f"samples/{fpath.stem}.jpg", buf.getvalue())
                        except Exception as exc:
                            manifest["files"].append({"path": rel_path, "error": str(exc)})

                    elif ext in VIDEO_EXTENSIONS:
                        meta = probe_video(fpath)
                        frames = extract_frames_ffmpeg(fpath, max_frames=36)
                        file_entry = {
                            "path": rel_path,
                            "type": "video",
                            "size_bytes": fpath.stat().st_size,
                            "probe": meta,
                            "sampled_frame_count": len(frames),
                        }
                        if frames:
                            per_frame = [compute_frame_stats(f, crop_osd=True) for f in frames]
                            # 计算相邻帧差分（用于判断慢快门帧累积 / 拖影 / 帧重复周期）
                            diffs = []
                            for i in range(1, len(frames)):
                                d = np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))) / 255.0
                                diffs.append(round(float(d), 6))

                            # 汇总均值
                            keys = list(per_frame[0].keys())
                            avg_stats = {k: round(float(np.mean([pf[k] for pf in per_frame])), 5) for k in keys}
                            file_entry["avg_stats_osd_masked"] = avg_stats
                            file_entry["avg_stats_full"] = compute_frame_stats(frames[len(frames) // 2], crop_osd=False)
                            file_entry["frame_to_frame_l1_diffs"] = diffs
                            file_entry["near_zero_diff_ratio"] = round(
                                float(np.mean([1.0 if d < 0.002 else 0.0 for d in diffs])), 4
                            ) if diffs else 0.0

                            # 导出前 6 帧连续序列（供慢快门/时间累积验证）+ 中间帧与末尾帧
                            sample_indices = sorted(set([0, 1, 2, 3, 4, 5, len(frames) // 2, len(frames) - 1]))
                            for idx in sample_indices:
                                if 0 <= idx < len(frames):
                                    im = Image.fromarray(frames[idx])
                                    buf = io.BytesIO()
                                    im.save(buf, format="JPEG", quality=95)
                                    zf.writestr(f"samples/{fpath.stem}_f{idx:03d}.jpg", buf.getvalue())

                        manifest["files"].append(file_entry)

        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        zf.writestr("calib_manifest.json", manifest_bytes)
        (OUT_DIR / "calib_manifest.json").write_bytes(manifest_bytes)

    return zip_path


def main() -> int:
    print(f"[1/2] 正在打包源码与校准脚本 -> {OUT_DIR / 'atsim_code.zip'} ...")
    code_zip = pack_code_archive()
    print(f"      完成: {code_zip.name} ({code_zip.stat().st_size / (1024*1024):.2f} MB)")

    print(f"[2/2] 正在扫描并抽样 {CALIB_DIR} 实拍素材 -> {OUT_DIR / 'calib_samples.zip'} ...")
    calib_zip = pack_calib_archive()
    print(f"      完成: {calib_zip.name} ({calib_zip.stat().st_size / (1024*1024):.2f} MB)")

    print("\n请将 _agent_upload 目录下的这两个文件直接拖入聊天窗口（或上传到 Google Drive）：")
    print(f"  1. {code_zip}")
    print(f"  2. {calib_zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
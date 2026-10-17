"""
Frame sources: anything the renderer can pull frames from by index.

`ClipSource` reads a real video through FFmpeg and caches decoded frames in a
small LRU.  The cache matters because the renderer legitimately asks for a frame
twice: once when the chain needs its input and once for the camera's motion blur
(the previous field).  `SyntheticSource` generates test patterns with no I/O, so
the engine can be exercised and benchmarked without a clip.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np

from .ffmpeg import VideoInfo, VideoReader, probe


class ClipSource:
    def __init__(self, path: str | Path, scale: float = 1.0,
                 cache_frames: int = 24, info: VideoInfo | None = None):
        self.path = str(path)
        self.info = info or probe(path)
        self.scale = float(scale)
        self.cache_frames = int(max(2, cache_frames))
        self.width = max(2, int(round(self.info.width * self.scale / 2)) * 2)
        self.height = max(2, int(round(self.info.height * self.scale / 2)) * 2)
        self.fps = self.info.fps or 25.0
        self.frame_count = self.info.frame_count
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def frame(self, index: int) -> np.ndarray:
        index = int(max(0, index))
        with self._lock:
            hit = self._cache.get(index)
            if hit is not None:
                self._cache.move_to_end(index)
                return hit
        with VideoReader(self.path, self.info, self.scale, start=index, count=1,
                         width=self.width, height=self.height) as r:
            got = None
            for f in r.frames():
                got = f
                break
        if got is None:
            raise IndexError(f"frame {index} beyond end of {self.path}")
        with self._lock:
            self._cache[index] = got
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        return got

    def prefetch(self, start: int, count: int) -> None:
        """Warm the cache sequentially -- far faster than random seeks."""
        with VideoReader(self.path, self.info, self.scale, start=start, count=count,
                         width=self.width, height=self.height) as r:
            for i, f in enumerate(r.frames()):
                with self._lock:
                    self._cache[start + i] = f
                    while len(self._cache) > self.cache_frames:
                        self._cache.popitem(last=False)

    def warm(self, count: int = 8) -> None:
        try:
            self.prefetch(0, count)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
class StillSource:
    """
    A single image presented as a clip.

    Imports and exports of stills are a first-class path, not a special case: a
    photograph is simply a clip of length one, so making it look like a normal
    source means every stage, the A/B reference, the render job and the export
    paths all work on it unchanged.  `frame()` returns the same picture whatever
    index is asked for, which is also what a paused deck does.
    """

    def __init__(self, path: str | Path, scale: float = 1.0,
                 fps: float = 25.0, frames: int = 1):
        from .imagecodec import load_image
        self.path = str(path)
        rgb = load_image(self.path).astype(np.float32)
        if scale and abs(scale - 1.0) > 1e-6:
            h0, w0 = rgb.shape[:2]
            nh = max(2, int(round(h0 * scale)))
            nw = max(2, int(round(w0 * scale)))
            try:
                from PIL import Image as _PILImage
                pil = _PILImage.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
                rgb = np.asarray(pil.resize((nw, nh), _PILImage.LANCZOS), dtype=np.float32) / 255.0
            except Exception:
                step = max(1, int(round(1.0 / max(1e-6, scale))))
                rgb = rgb[::step, ::step]
        self._rgb = np.ascontiguousarray(rgb)
        self.height, self.width = self._rgb.shape[:2]
        self.fps = float(fps)
        self.frame_count = int(max(1, frames))
        self.scale = 1.0
        self.info = VideoInfo(path=self.path, width=self.width, height=self.height,
                              fps=self.fps, frame_count=self.frame_count,
                              duration=self.frame_count / max(1e-6, self.fps),
                              has_audio=False, codec="still")

    def frame(self, index: int) -> np.ndarray:
        return self._rgb.copy()

    def prefetch(self, start: int = 0, count: int = 8) -> None:
        return None

    def warm(self, count: int = 8) -> None:
        return None

    def clear(self) -> None:
        return None

    @property
    def is_still(self) -> bool:
        return True


# ---------------------------------------------------------------------------
class SyntheticSource:
    """
    A procedurally generated clip.  Used for the self-test and for the in-app
    "no file yet" demo project, so the pipeline can always be demonstrated.
    """

    def __init__(self, width: int = 720, height: int = 480, frames: int = 150,
                 fps: float = 30000.0 / 1001.0, variant: str = "chart"):
        self.width, self.height = int(width), int(height)
        self.frame_count = int(frames)
        self.fps = float(fps)
        self.variant = variant
        self.info = VideoInfo(path="<synthetic>", width=self.width, height=self.height,
                              fps=self.fps, frame_count=self.frame_count,
                              duration=self.frame_count / self.fps, has_audio=False,
                              codec="synthetic")

    def frame(self, index: int) -> np.ndarray:
        t = index / max(1e-6, self.fps)
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = xx / max(1, w - 1)
        ny = yy / max(1, h - 1)
        img = np.zeros((h, w, 3), dtype=np.float32)

        if self.variant == "chart":
            # grey staircase + colour bars + fine detail: the right test signal
            # for bandwidth, cross-colour and chroma resolution
            bars = np.array([
                [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.15, 0.15, 0.15],
            ], dtype=np.float32)
            seg = np.clip((nx * 8).astype(np.int32), 0, 7)
            img[: int(h * 0.45)] = bars[seg][: int(h * 0.45)]
            step = ((nx[..., None] > 0.5) * 1.0).repeat(3, axis=2)
            img[int(h * 0.45): int(h * 0.62)] = step[int(h * 0.45): int(h * 0.62)]
            # zone plate: the classic way to see cross-colour and dot crawl
            r = np.sqrt((nx - 0.5) ** 2 + ((ny - 0.78) * w / h) ** 2)
            plate = 0.5 + 0.5 * np.cos(2 * np.pi * 60.0 * r ** 2)
            zone = plate[int(h * 0.62):, :, None] * np.array([1.0, 1.0, 1.0], dtype=np.float32)
            img[int(h * 0.62):] = zone * 0.9
            # moving marker so temporal behaviour is visible
            cx = int((0.1 + 0.8 * (0.5 + 0.5 * np.sin(t * 2 * np.pi))) * (w - 1))
            cy = int(0.70 * h)
            img[max(0, cy - 8):cy + 8, max(0, cx - 8):cx + 8] = np.array(
                [0.05, 0.05, 0.05], dtype=np.float32)
        else:
            ramp = (0.5 + 0.5 * np.sin(2 * np.pi * (nx * 3 + t * 0.4)))
            img[..., 0] = ramp
            img[..., 1] = 0.5 + 0.4 * np.cos(2 * np.pi * (ny * 5 - t * 0.3))
            img[..., 2] = 0.6 * ramp + 0.2
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def prefetch(self, start: int, count: int) -> None:
        pass

    def warm(self, count: int = 8) -> None:
        pass

    def clear(self) -> None:
        pass

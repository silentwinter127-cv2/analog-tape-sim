"""
Render a synthetic *photographic* test clip and a comparison sheet.

Colour bars are excellent for measuring bandwidth but useless for judging
whether a look is believable.  This builds a scene with the things that reveal
medium signatures: skin tones, a sky gradient, foliage texture, a hard
highlight, a dark area, fine repeating detail (window blinds, a fence) and a
moving object.

Run:  python tools/testimage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atsim.core.profiles import (CAMERA_PROFILES, DISPLAY_PROFILES,
                                 FACTORY_PRESETS, MEDIA_PROFILES,
                                 PLAYER_PROFILES, SCHEMA)
from atsim.core.rng import derive_seed
from atsim.core.params import merge_values
from atsim.io.source import SyntheticSource
from atsim.render.renderer import Renderer

OUT = ROOT / "tools" / "out"


class SceneSource:
    """A synthetic 'photograph' with a moving subject, animated over time."""

    def __init__(self, width: int = 720, height: int = 480, frames: int = 120,
                 fps: float = 30000.0 / 1001.0):
        self.width, self.height = int(width), int(height)
        self.frame_count = int(frames)
        self.fps = float(fps)
        from atsim.io.ffmpeg import VideoInfo
        self.info = VideoInfo(path="<scene>", width=self.width, height=self.height,
                              fps=self.fps, frame_count=self.frame_count,
                              duration=self.frame_count / self.fps, codec="synthetic")
        self._cache: dict[int, np.ndarray] = {}
        self._base = self._build_base()

    # ------------------------------------------------------------------
    def _build_base(self) -> np.ndarray:
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = xx / (w - 1)
        ny = yy / (h - 1)
        img = np.zeros((h, w, 3), dtype=np.float32)

        # sky: vertical gradient with a warm haze near the horizon
        horizon = 0.52
        sky = np.clip(1.0 - ny / horizon, 0.0, 1.0)
        img[..., 0] = 0.35 + 0.45 * sky + 0.12 * (1.0 - sky)
        img[..., 1] = 0.45 + 0.40 * sky + 0.06 * (1.0 - sky)
        img[..., 2] = 0.62 + 0.33 * sky - 0.10 * (1.0 - sky)

        ground = ny > horizon
        gy = (ny - horizon) / max(1e-6, 1.0 - horizon)
        # ground: grass / dirt with texture
        rng = np.random.default_rng(7)
        tex = rng.standard_normal((h, w)).astype(np.float32)
        from scipy.ndimage import gaussian_filter
        tex = gaussian_filter(tex, 1.6)
        tex /= max(1e-6, tex.std())
        fine = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.7)
        fine /= max(1e-6, fine.std())
        g_r = (0.30 + 0.22 * gy) + 0.05 * tex + 0.03 * fine
        g_g = (0.36 + 0.26 * gy) + 0.06 * tex + 0.03 * fine
        g_b = (0.16 + 0.14 * gy) + 0.04 * tex + 0.02 * fine
        img[..., 0] = np.where(ground, g_r, img[..., 0])
        img[..., 1] = np.where(ground, g_g, img[..., 1])
        img[..., 2] = np.where(ground, g_b, img[..., 2])

        # a house wall: flat mid-tone with a hard specular highlight (blown out)
        wr0, wr1 = int(0.30 * h), int(0.52 * h)
        wc0, wc1 = int(0.05 * w), int(0.30 * w)
        img[wr0:wr1, wc0:wc1] = np.array([0.78, 0.72, 0.62], dtype=np.float32)
        img[wr0:wr0 + 6, wc0:wc1] = 0.55
        # window: very fine repeating vertical bars (a resolution killer)
        wr0b, wr1b = int(0.33 * h), int(0.44 * h)
        wc0b, wc1b = int(0.09 * w), int(0.20 * w)
        bars = ((np.arange(wc0b, wc1b) // 2) % 2).astype(np.float32)
        img[wr0b:wr1b, wc0b:wc1b] = (0.12 + 0.72 * bars)[None, :, None]
        # blown-out sky patch / lamp: clipping test
        cy, cx = int(0.20 * h), int(0.62 * w)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        lamp = np.clip(1.0 - rr / (0.09 * h), 0.0, 1.0) ** 0.7
        img = img + lamp[..., None] * np.array([1.0, 0.98, 0.92], dtype=np.float32)

        # skin tone patch (a face) - the hardest thing to get right
        fy, fx = int(0.58 * h), int(0.62 * w)
        fr = 0.10 * h
        rr2 = np.sqrt(((yy - fy) / fr) ** 2 + ((xx - fx) / (fr * 0.82)) ** 2)
        face = np.clip(1.0 - rr2, 0.0, 1.0) ** 0.5
        skin = np.array([0.86, 0.68, 0.56], dtype=np.float32)
        img = img * (1.0 - face[..., None]) + skin * face[..., None]
        # dark hair above the face
        hy = fy - fr * 0.95
        hair = np.clip(1.0 - np.sqrt(((yy - hy) / (fr * 0.95)) ** 2
                                     + ((xx - fx) / fr) ** 2), 0.0, 1.0) ** 0.6
        img = img * (1.0 - 0.9 * hair[..., None]) \
            + np.array([0.07, 0.05, 0.05], dtype=np.float32) * (0.9 * hair[..., None])

        # a fence: fine high-contrast vertical detail in front of the grass
        fx0, fx1 = int(0.35 * h), int(0.50 * h)
        for k in range(int(0.72 * w), int(0.99 * w), 9):
            img[fx0:fx1, k:k + 3] = 0.30
            img[fx0:fx1, k + 3:k + 4] = 0.62

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def frame(self, index: int) -> np.ndarray:
        key = int(index)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        t = index / max(1e-6, self.fps)
        img = self._base.copy()
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # a ball crossing the frame: motion + a bright specular highlight
        bx = 0.5 + 0.34 * np.sin(t * 1.1)
        by = 0.70 + 0.05 * np.sin(t * 2.3)
        r = 0.055 * h
        d = np.sqrt((yy - by * h) ** 2 + (xx - bx * w) ** 2)
        ball = np.clip(1.0 - d / r, 0.0, 1.0)
        spec = np.clip(1.0 - np.sqrt((yy - (by * h - r * 0.35)) ** 2
                                     + (xx - (bx * w - r * 0.35)) ** 2) / (r * 0.35),
                       0.0, 1.0) ** 1.5
        col = np.array([0.92, 0.86, 0.30], dtype=np.float32)
        img = img * (1.0 - ball[..., None]) + col * ball[..., None]
        img = np.clip(img + spec[..., None] * 0.9, 0.0, 1.0)
        self._cache.clear()
        self._cache[key] = img.astype(np.float32)
        return self._cache[key]

    def prefetch(self, start: int, count: int) -> None:
        pass

    def warm(self, count: int = 8) -> None:
        pass

    def clear(self) -> None:
        self._cache.clear()


PRESETS = {
    "00_source": None,
    "01_camera_only": merge_values(
        CAMERA_PROFILES["ccd_camcorder_1990"]["layers"],
        DISPLAY_PROFILES["none"]["layers"]),
    "02_family_1992": FACTORY_PRESETS["family_archive_1992"]["layers"],
    "03_rental_worn": FACTORY_PRESETS["rental_tape_horror"]["layers"],
    "04_pro_transfer": FACTORY_PRESETS["pro_transfer"]["layers"],
    "05_hi8_camcorder": merge_values(
        MEDIA_PROFILES["camcorder_hi8"]["layers"],
        merge_values(PLAYER_PROFILES["vcr_consumer"]["layers"],
                     DISPLAY_PROFILES["crt_consumer"]["layers"])),
    "06_no_display": merge_values(
        FACTORY_PRESETS["family_archive_1992"]["layers"],
        DISPLAY_PROFILES["none"]["layers"]),
    "07_crt_only": merge_values(
        CAMERA_PROFILES["clean_digital"]["layers"],
        merge_values(MEDIA_PROFILES["clean_passthrough"]["layers"],
                     DISPLAY_PROFILES["crt_consumer"]["layers"])),
}


def save(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def main() -> int:
    src = SceneSource()
    frame_index = 26
    clean = src.frame(frame_index)
    save(clean, OUT / "scene_source.png")
    tiles = [clean]
    for name, layers in PRESETS.items():
        if layers is None:
            continue
        r = Renderer(src, src.width, src.height, src.fps, layers, seed=4242)
        frame = r.render_frame(frame_index)
        save(frame, OUT / f"scene_{name}.png")
        tiles.append(frame)
        print(f"{name:18s} mean={frame.mean():.3f} std={frame.std():.3f}")
    sheet = np.concatenate([np.concatenate(tiles[0:4], axis=1),
                            np.concatenate(tiles[4:8], axis=1)], axis=0)
    save(sheet, OUT / "scene_sheet.png")
    print(f"\nwrote {OUT/'scene_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Stage 01b -- Standalone Photochemical Film & LOMO Darkroom System.

Placed after camera stage (01) and before format stage (02), giving photochemical film
its own architectural stage and parameter namespace (`film`), while maintaining full
backward-compatibility with legacy `camera.film_*` and `camera.process_mode` settings.
"""
from __future__ import annotations

import numpy as np

from ..codecs.photochemical_film import (
    PhotochemicalFilmSpec,
    apply_photochemical_film_pipeline,
)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame


class FilmStage(Stage):
    """01b 胶片光化学独立阶段 (FilmStage)。"""
    name = "film"
    layers = ("film", "camera")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg_film = self.cfg_of("film")
        cfg_cam = self.cfg_of("camera")
        self.spec = PhotochemicalFilmSpec.from_params(cfg_film, fallback_cam=cfg_cam)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if not self.spec.enabled:
            return buf

        # Avoid redundant double application if already applied by a dedicated node
        if buf.meta.get("film_applied"):
            return buf

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
            seed=int(self.ctx.seed) + 101,
            secondary_rgb=sec_rgb,
            optics_already_applied=optics_done,
        )

        meta = dict(buf.meta)
        meta["photochemical_film"] = info
        meta["film_applied"] = True
        if sig is not None:
            meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)

        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

"""
Stage 01b -- Photochemical Film & LOMO Darkroom Stage (`film`).

Decoupled from `CameraStage` (`01 成像系统`) so that photochemical silver-halide
emulsion response, LOMO cross-processing (`E-6 in C-41` / `C-41 in E-6`),
expired film base fog (`D_min`), Redscale reverse-loading, LOMO plastic meniscus
vignette, shaped heart bokeh, light leaks, double exposure, projector mechanics,
and Rank Cintel flying-spot telecine operate as a dedicated, standalone stage
and layer (`01b 胶片光化学 (胶片/LOMO)`).

Backward compatibility:
  If `film.enabled` is off (`0`), `FilmStage` automatically checks whether the
  legacy `camera.film_enabled` parameter is active and seamlessly runs the
  photochemical/telecine pipeline from `camera` layer parameters.
"""
from __future__ import annotations

import numpy as np

from ..codecs.photochemical_film import (
    PhotochemicalFilmSpec,
    apply_photochemical_film_pipeline,
)
from ..core.pipeline import FrameBuffer, RenderContext, Stage


class FilmStage(Stage):
    name = "film"
    layers = ("film", "camera")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        film_cfg = self.cfg_of("film")
        cam_cfg = self.cfg_of("camera")
        self.spec = PhotochemicalFilmSpec.from_params(film_cfg, fallback_cam=cam_cfg)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        self._read_params()
        if not self.spec.enabled:
            return buf
        rgb = np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        sec_rgb = None
        if getattr(self.ctx, "source", None) is not None:
            try:
                sec_rgb = np.clip(self.ctx.source.frame(int(buf.index) + 1), 0.0, 1.0).astype(np.float32)
            except Exception:
                sec_rgb = None
        out_rgb, info = apply_photochemical_film_pipeline(
            rgb,
            self.spec,
            frame_index=int(buf.index),
            seed=self.ctx.seed,
            secondary_rgb=sec_rgb,
        )
        meta = dict(buf.meta)
        meta["photochemical_film"] = info
        meta["film_telecine"] = info
        meta["tap_film_rgb"] = out_rgb
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)

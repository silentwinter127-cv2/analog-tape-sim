"""
Stage 02b -- the digital coding branch (DV / MiniDV).

Placed after the format stage and before the medium, because that is where a
recorder sits: the imaging system produces a component signal and the codec
stores it.  For the analog branch this stage does nothing, which is itself the
point -- the two branches must not share a core model.

What it adds, and why it cannot be folded into the analog path:

  * 4:1:1 (525/60) or 4:2:0 (625/50) chroma subsampling -- colour resolution is
    lost before any tape is involved, so it cannot be a "tape parameter",
  * 8x8 DCT with a fixed bit budget spread across macroblocks -- the artefacts are
    block-shaped mosquito noise, worst where detail is highest,
  * no colour subcarrier at all, so a DV frame has no hue to rotate and no
    colour-under bandwidth to lose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..codecs.dv import DVCodec, DVParams
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame


@dataclass
class DVSettings:
    enabled: bool = False
    chroma_mode: str = "411"
    quality: float = 1.0
    base_step: float = 0.0125


class CodecStage(Stage):
    name = "codec"
    layers = ("codec", "format")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        cfg = self.cfg_of("codec")
        self.enabled = float(cfg.get("dv_enabled", 0.0)) > 0.5
        mode = str(cfg.get("chroma_mode", 1.0))
        # the parameter is a 0/1 switch so it clamps consistently with the rest
        # of the schema; PAL uses 4:2:0, 525/60 uses 4:1:1
        if mode in ("1", "1.0"):
            self.chroma_mode = "420" if ctx.std.color_system == "PAL" else "411"
        else:
            self.chroma_mode = "411"
        self.quality = float(np.clip(cfg.get("dv_quality", 1.0), 0.25, 2.0))
        self.codec = DVCodec(ctx.std, DVParams(
            chroma_mode=self.chroma_mode,
            quality=self.quality,
            base_step=float(cfg.get("dv_base_step", 0.0125)),
        ))
        self._last_error = 0.0

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if not self.enabled:
            return buf
        sig: SignalFrame | None = buf.meta.get("signal")
        if sig is None:
            return buf

        coded = self.codec.encode_decode(sig)
        y = coded.y * (1.0 / max(1e-6, self.quality)) ** 0.0  # level preserved

        self._last_error = float(np.abs(coded.y - sig.y).mean())
        meta = dict(buf.meta)
        meta["signal"] = SignalFrame(coded.y.astype(np.float32),
                                     coded.i.astype(np.float32),
                                     coded.q.astype(np.float32))
        meta["format"] = "dv"
        meta["dv"] = {
            "chroma_mode": self.chroma_mode,
            "quality": self.quality,
            "mean_abs_error": round(self._last_error, 5),
        }
        # a DV frame has no composite carrier: the medium stage must not treat it
        # as RF, so the composite plane is dropped rather than faked
        return FrameBuffer(buf.rgb, None, buf.index, buf.time, buf.std, meta)

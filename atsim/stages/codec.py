"""
Stage 02b -- the digital coding branch (DV / MiniDV).

Placed after the format stage and before the medium, because that is where a
recorder sits: the imaging system produces a component signal and the codec
stores it.  For the analog branch this stage does nothing, which is itself the
point -- the two branches must not share a core model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..codecs.dv import DVCodec, DVParams
from ..codecs.mpeg1 import MPEG1Codec, MPEG1CodecParams
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
        self.mpeg1 = None
        self._last_error = 0.0
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("codec")
        self.enabled = float(cfg.get("dv_enabled", 0.0)) > 0.5
        mode = str(cfg.get("chroma_mode", 1.0))
        if mode in ("1", "1.0"):
            self.chroma_mode = "420" if self.ctx.std.color_system == "PAL" else "411"
        else:
            self.chroma_mode = "411"
        self.quality = float(np.clip(cfg.get("dv_quality", 1.0), 0.25, 2.0))
        kind = float(cfg.get("codec_kind", 0.0))
        if kind <= 0.0 and self.enabled:
            kind = 1.0
        self.codec_kind = int(round(kind))
        if self.codec_kind == 2:
            p = MPEG1CodecParams(
                quality=float(np.clip(cfg.get("vcd_quality", 1.0), 0.25, 2.0)),
                motion_compensation=float(np.clip(
                    cfg.get("vcd_motion_comp", 1.0), 0.0, 1.0)),
                width=int(round(float(cfg.get("vcd_width", 352.0)))),
                height=int(round(float(cfg.get("vcd_height", 240.0)))),
                bitrate_kbps=float(cfg.get("vcd_bitrate_kbps", 1150.0)),
            )
            if self.mpeg1 is None:
                self.mpeg1 = MPEG1Codec(p)
            else:
                self.mpeg1.p = p
        else:
            self.mpeg1 = None
        self.codec = DVCodec(self.ctx.std, DVParams(
            chroma_mode=self.chroma_mode,
            quality=self.quality,
            base_step=float(cfg.get("dv_base_step", 0.0125)),
        ))

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if self.codec_kind <= 0:
            return buf
        sig: SignalFrame | None = buf.meta.get("signal")
        if sig is None:
            return buf

        if self.codec_kind == 2 and self.mpeg1 is not None:
            rgb = sig.to_rgb()
            decoded = self.mpeg1.encode_decode(rgb, int(buf.index))
            coded = SignalFrame.from_rgb(decoded, buf.std)
            self._last_error = float(np.abs(decoded - rgb).mean())
            meta = dict(buf.meta)
            meta["signal"] = coded
            meta["format"] = "vcd"
            meta["vcd"] = {
                "quality": self.mpeg1.p.quality,
                "motion_compensation": self.mpeg1.p.motion_compensation,
                "mean_abs_error": round(self._last_error, 5),
            }
            return FrameBuffer(buf.rgb, None, buf.index, buf.time, buf.std, meta)
        coded = self.codec.encode_decode(sig)
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
        return FrameBuffer(buf.rgb, None, buf.index, buf.time, buf.std, meta)

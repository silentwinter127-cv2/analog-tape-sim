"""
The virtual signal pipeline.

Architecture note (this is the core design decision of the whole project):

    The chain is split into *stages*, and each stage may pass on a different
    kind of signal:

        camera  : RGB  ->  RGB (linear-ish light)
        format  : RGB  ->  SignalFrame  (+ optional composite plane)
        tape    : composite plane       (this is what the medium can damage)
        player  : composite -> YIQ      (separation, TBC, amps)
        display : YIQ -> RGB            (CRT, then letterbox/CROP)

    Because `tape` operates on the *composite* signal rather than on a picture,
    a dropout produces lost FM carrier (white streak + chroma noise), a
    time-base error shifts luma and chroma TOGETHER but displaces them against
    the previous line, and head switching tears the bottom of the field.  Those
    artefacts are consequences of where in the chain the operation happens,
    which is exactly the "media differ by mechanism, not by overlay" rule.

A `Chain` is stateless with respect to video data: it builds deterministic
*plans* (defect timelines, transport drift) once per clip from the seed, then
applies them frame by frame.  Same seed + same parameters => same video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .standards import VideoStandard


# ---------------------------------------------------------------------------
# the frame buffer travelling down the chain
# ---------------------------------------------------------------------------
@dataclass
class FrameBuffer:
    rgb: np.ndarray                      # (H, W, 3) float32, 0..1 nominal
    composite: np.ndarray | None = None  # (H, W) float32, present after encoding
    index: int = 0                       # absolute frame index in the clip
    time: float = 0.0                    # seconds
    std: VideoStandard | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def copy(self) -> "FrameBuffer":
        return FrameBuffer(
            self.rgb.copy(),
            None if self.composite is None else self.composite.copy(),
            self.index, self.time, self.std, dict(self.meta),
        )


# ---------------------------------------------------------------------------
# rendering context handed to every stage
# ---------------------------------------------------------------------------
@dataclass
class RenderContext:
    std: VideoStandard
    params: dict[str, dict[str, float]]      # layer -> key -> value
    seed: int
    frame_count: int
    fps: float
    width: int
    height: int
    preview: bool = False                     # reduced-accuracy mode
    scale: float = 1.0                        # render scale versus source
    temporal_cb: Callable[[np.ndarray], np.ndarray] | None = None
    # The clip being simulated.  Most stages never need it, but a few mechanisms
    # are *about* the passage of time rather than about one frame -- shuttle-mode
    # error concealment substitutes macroblocks from a frame the transport passed
    # earlier, so it has to be able to ask for a different frame than the one
    # being rendered.  Optional so tests and synthetic runs can omit it.
    source: Any = None

    def layer(self, name: str) -> dict[str, float]:
        return self.params.get(name, {})

    @property
    def source_width(self) -> int:
        return int(round(self.width / max(1e-6, self.scale)))

    @property
    def source_height(self) -> int:
        return int(round(self.height / max(1e-6, self.scale)))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
class Stage:
    """
    Base class.  A stage receives the config layers it cares about and may
    build a per-clip `plan` (deterministic timeline) before the first frame.
    """

    name = "stage"
    layers: tuple[str, ...] = ()

    def __init__(self, ctx: RenderContext):
        self.ctx = ctx
        self.cfg = {layer: dict(ctx.layer(layer)) for layer in self.layers}

    def cfg_of(self, layer: str) -> dict[str, float]:
        return self.cfg.get(layer, {})

    def prepare(self) -> None:      # build plans; called once per clip
        pass

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        raise NotImplementedError

    def timeline(self) -> dict:
        """Optional: structured description of generated defects (for the UI)."""
        return {}


class Chain:
    """An ordered list of stages sharing one render context."""

    def __init__(self, ctx: RenderContext):
        self.ctx = ctx
        self.stages: list[Stage] = []
        self._plans_ready = False

    def add(self, stage: Stage) -> "Chain":
        self.stages.append(stage)
        return self

    def prepare(self) -> "Chain":
        for st in self.stages:
            st.prepare()
        self._plans_ready = True
        return self

    def run(self, buf: FrameBuffer) -> FrameBuffer:
        if not self._plans_ready:
            self.prepare()
        for st in self.stages:
            buf = st.apply(buf)
        return buf

    def stage_named(self, name: str) -> Stage | None:
        for st in self.stages:
            if st.name == name:
                return st
        return None

    def defect_timeline(self) -> list[dict]:
        out = []
        for st in self.stages:
            tl = st.timeline()
            if tl:
                out.append({"stage": st.name, "layers": list(st.layers), **tl})
        return out

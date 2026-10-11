"""
The character generator as a stage: a camera OSD, or a titler recording onto tape.

One stage serves both, because the two differ only in *where* in the chain it runs -- which is expressed by
the chain's order, not by two implementations. `osd.at` decides whether the chain places it last (a camera's
own overlay) or before the medium (a titler), and the renderer reads that when it builds the chain.

Drawing is delegated to `codecs.osd`, which owns the cell layout and the rasterising; this class owns the
frame plumbing and the position in the signal path.
"""
from __future__ import annotations

import numpy as np

from ..codecs.osd import CharacterGenerator, OSDPlan, parse_lines
from ..core.pipeline import FrameBuffer, RenderContext, Stage


class OverlayStage(Stage):
    """Text drawn by a character generator."""

    name = "overlay"
    layers = ("overlay",)

    #: which slot this instance occupies: 0 = after the display, 1 = after the camera
    SLOT_TAIL = 0
    SLOT_PRE_TAPE = 1

    def __init__(self, ctx: RenderContext, slot: int = 0):
        super().__init__(ctx)
        cfg = self.cfg_of("overlay")
        self.enabled = float(cfg.get("enabled", 0.0)) > 0.5
        # where the generator writes: 0 after the picture is formed (a camera's own OSD), 1 before the
        # medium (a titler recording onto tape).  The renderer places this stage in the matching slot and
        # sets `slot` accordingly; a stage whose slot does not match does nothing.
        self.at = int(np.clip(round(float(cfg.get("at", 0.0))), 0, 1))
        self.slot = int(slot)
        self.text = str(cfg.get("text", ""))
        self.font_kind = float(cfg.get("font", 0.0))
        self.cell_h = int(max(8, round(float(cfg.get("cell_h", 24)))))
        self.col = int(round(float(cfg.get("col", 0))))
        self.halign = int(np.clip(round(float(cfg.get("halign", 0.0))), 0, 2))
        self.valign = int(np.clip(round(float(cfg.get("valign", 0.0))), 0, 2))
        # `pos` is a convenience that sets both, so a person does not have to set two parameters to get
        # "bottom right" -- and 8 is the placement a home recording's OSD usually has
        _pos = int(np.clip(round(float(cfg.get("pos", 0.0))), 0, 8))
        if float(cfg.get("pos", 0.0)) > 1e-6 or _pos:
            self.halign = (0, 1, 2, 0, 1, 2, 0, 1, 2)[_pos]
            self.valign = (0, 0, 0, 1, 1, 1, 2, 2, 2)[_pos]
        self.row = int(round(float(cfg.get("row", 0))))
        self.margin = float(cfg.get("margin", 0.03))
        self.shadow = float(cfg.get("shadow", 0.5))
        self.opacity = float(cfg.get("opacity", 1.0))
        # A camera's OSD draws over the picture and does not black out a rectangle of it -- on a real
        # camcorder that would hide the scene being framed.  A titler recording onto tape is a different
        # machine and a box is normal there.  One parameter cannot have two schema defaults, so the stage
        # decides from the slot, and an explicit setting still wins because the parameter is a value.
        self.bg_on = float(cfg.get("background", 0.0)) > 0.5
        # An outline is what makes text readable on an unknown picture without covering any of it, which is
        # what a camera's OSD does.  A box is available but is not the default, because on a camera it would
        # black out a rectangle of the scene being framed.
        self.outline = float(cfg.get("outline", 1.0))
        self.colour = (float(cfg.get("red", 0.92)), float(cfg.get("green", 0.94)),
                       float(cfg.get("blue", 0.90)))
        self._gen = None
        self.report: dict = {}

    # the font list is fixed at import, and the index selects from it
    FONTS = ("osd", "lyusung", "apos")

    def _generator(self, width: int, height: int):
        key = self.FONTS[int(np.clip(round(self.font_kind), 0, len(self.FONTS) - 1))]
        if self._gen is not None and self._gen.font_key == key:
            return self._gen
        margin_x = max(0, int(round(self.margin * width)))
        margin_y = max(0, int(round(self.margin * height)))
        plan = OSDPlan(lines=parse_lines(self.text), cell_h=self.cell_h,
                       margin_x=margin_x, margin_y=margin_y,
                       halign=self.halign, valign=self.valign,
                       colour=self.colour, shadow=self.shadow, opacity=self.opacity,
                       outline=self.outline,
                       background=(0.0, 0.0, 0.0) if self.bg_on else None)
        # the row and column offsets apply to every line
        for line in plan.lines:
            line.row += self.row
            line.col += self.col
        try:
            self._gen = CharacterGenerator(key, plan)
        except Exception as exc:                                      # noqa: BLE001
            # a font that cannot be loaded must not take the render down: the overlay is optional and
            # the rest of the chain has already done its work
            self._gen = None
            self.report = {"drawn": 0, "missing": [], "error": str(exc)[:120],
                           "font": key, "lines": len(plan.lines)}
        return self._gen

    def prepare(self) -> None:
        pass

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if not self.enabled or not self.text:
            return buf
        if self.at != self.slot:
            # this instance is in the slot the parameter did not choose; the other one draws
            return buf
        rgb = buf.rgb.astype(np.float32, copy=False)
        gen = self._generator(rgb.shape[1], rgb.shape[0])
        if gen is None:
            return buf
        out, rep = gen.draw(rgb)
        self.report = rep
        return FrameBuffer(out.astype(np.float32), None, buf.index, buf.time, buf.std,
                           dict(buf.meta))

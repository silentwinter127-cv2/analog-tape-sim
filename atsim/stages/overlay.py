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


def _available_font_keys() -> tuple[str, ...]:
    """
    The fonts the panel may offer, in the order the panel lists them.

    Read from the registry rather than written out here.  The tuple this replaces held three names while
    the registry held fifteen, so every index above 2 was clamped to the third: measured over the API,
    fourteen distinct indices were posted and the server drew `lyusung` every time.  The panel lists the
    registry, so the two disagreed.

    `font_catalogue()` filters entries whose file is missing, so the dropdown and the engine agree about
    availability as well.
    """
    from ..codecs.fonts import font_catalogue
    keys = tuple(d["key"] for d in font_catalogue() if d.get("available"))
    return keys or ("osd",)


class OverlayStage(Stage):
    """Text drawn by a character generator."""

    name = "overlay"
    layers = ("overlay",)

    #: which slot this instance occupies: 0 = after the display, 1 = before the tape
    SLOT_TAIL = 0
    SLOT_PRE_TAPE = 1
    #: reference vertical resolution that `cell_h`, `letter_gap`, `line_gap`, `shadow_x/y` are specified in.
    #: Scaling all pixel-unit OSD parameters by `height / REF_HEIGHT` makes the preview (e.g. 0.5x = 240p/276p)
    #: and the full-resolution export (1.0x = 480p/552p/1080p) have the *exact same visual proportion*.
    REF_HEIGHT = 240.0

    def __init__(self, ctx: RenderContext, slot: int = 0):
        super().__init__(ctx)
        self.slot = int(slot)
        self._gen = None
        self._gen_sig = None
        self.report: dict = {}
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("overlay")
        self.enabled = float(cfg.get("enabled", 0.0)) > 0.5
        self.at = int(np.clip(round(float(cfg.get("at", 0.0))), 0, 1))
        self.text = str(cfg.get("text", ""))
        self.font_kind = float(cfg.get("font", 0.0))
        self.cell_h = float(max(6.0, float(cfg.get("cell_h", 24.0))))
        self.col = int(round(float(cfg.get("col", 0.0))))
        self.halign = int(np.clip(round(float(cfg.get("halign", 0.0))), 0, 2))
        self.valign = int(np.clip(round(float(cfg.get("valign", 0.0))), 0, 2))
        _pos = int(np.clip(round(float(cfg.get("pos", 0.0))), 0, 8))
        if float(cfg.get("pos", 0.0)) > 1e-6 or _pos:
            self.halign = (0, 1, 2, 0, 1, 2, 0, 1, 2)[_pos]
            self.valign = (0, 0, 0, 1, 1, 1, 2, 2, 2)[_pos]
        self.row = int(round(float(cfg.get("row", 0.0))))
        self.margin = float(cfg.get("margin", 0.03))
        self.shadow = float(cfg.get("shadow", 0.5))
        self.opacity = float(cfg.get("opacity", 1.0))
        self.bg_on = float(cfg.get("background", 0.0)) > 0.5
        self.outline = float(cfg.get("outline", 1.0))
        self.outline_colour = (float(cfg.get("outline_red", 0.02)),
                               float(cfg.get("outline_green", 0.03)),
                               float(cfg.get("outline_blue", 0.05)))
        self.italic = float(cfg.get("italic", 0.0))
        self.stretch_x = float(cfg.get("stretch_x", 1.0))
        self.stretch_y = float(cfg.get("stretch_y", 1.0))
        self.letter_gap = float(cfg.get("letter_gap", 0.0))
        self.line_gap = float(cfg.get("line_gap", 0.0))
        self.shadow_x = float(cfg.get("shadow_x", 2.0))
        self.shadow_y = float(cfg.get("shadow_y", 2.0))
        self.colour = (float(cfg.get("red", 0.92)), float(cfg.get("green", 0.94)),
                       float(cfg.get("blue", 0.90)))
        self._apply_style()

    def on_params_changed(self) -> None:
        self._read_params()

    # **Derived from the registry, not repeated.**
    FONTS = _available_font_keys()

    def _generator(self, width: int, height: int):
        fonts = _available_font_keys()
        key = fonts[int(np.clip(round(self.font_kind), 0, len(fonts) - 1))]
        # Scale all pixel-denominated OSD parameters proportionally to the frame height relative to REF_HEIGHT (240p).
        # At 240p (standard 0.5x preview of 480p), ratio is 1.0; at 480p/552p/1080p export, cell_h, gaps, shadow
        # offset, and outline thickness scale proportionally so preview and export have identical visual proportions.
        scale_r = max(0.25, float(height) / self.REF_HEIGHT)
        eff_cell_h = max(6, int(round(self.cell_h * scale_r)))
        eff_letter_gap = int(round(self.letter_gap * scale_r))
        eff_line_gap = int(round(self.line_gap * scale_r))
        eff_shadow_x = int(round(self.shadow_x * scale_r)) if abs(self.shadow_x) > 1e-4 else 0
        eff_shadow_y = int(round(self.shadow_y * scale_r)) if abs(self.shadow_y) > 1e-4 else 0
        eff_outline_px = max(1, int(round(scale_r)))
        margin_x = max(0, int(round(self.margin * width)))
        margin_y = max(0, int(round(self.margin * height)))

        sig = (key, self.text, eff_cell_h, margin_x, margin_y, self.halign, self.valign,
               self.row, self.col, self.colour, round(self.shadow, 4), round(self.opacity, 4),
               round(self.outline, 4), eff_outline_px, self.outline_colour, round(self.italic, 4),
               eff_shadow_x, eff_shadow_y, round(self.stretch_x, 4), round(self.stretch_y, 4),
               eff_letter_gap, eff_line_gap, self.bg_on)
        if self._gen is not None and self._gen_sig == sig:
            return self._gen

        plan = OSDPlan(lines=parse_lines(self.text), cell_h=eff_cell_h,
                       margin_x=margin_x, margin_y=margin_y,
                       halign=self.halign, valign=self.valign,
                       colour=self.colour, shadow=self.shadow, opacity=self.opacity,
                       outline=self.outline, outline_px=eff_outline_px,
                       outline_colour=self.outline_colour,
                       italic=self.italic, shadow_x=eff_shadow_x, shadow_y=eff_shadow_y,
                       stretch_x=self.stretch_x, stretch_y=self.stretch_y,
                       letter_gap=eff_letter_gap, line_gap=eff_line_gap,
                       background=(0.0, 0.0, 0.0) if self.bg_on else None)
        for line in plan.lines:
            line.row += self.row
            line.col += self.col
        try:
            self._gen = CharacterGenerator(key, plan)
            self._gen_sig = sig
        except Exception as exc:                                      # noqa: BLE001
            self._gen = None
            self._gen_sig = None
            self.report = {"drawn": 0, "missing": [], "error": str(exc)[:120],
                           "font": key, "lines": len(plan.lines)}
        return self._gen

    #: (glyph colour, outline colour, italic) for each template.  Index 0 is "custom".
    STYLE_PRESETS = (
        None,
        ((0.35, 0.95, 0.45), (0.03, 0.05, 0.04), 0.0),     # green inside near-black: the OSD classic
        ((0.92, 0.20, 0.22), (0.05, 0.03, 0.03), 0.0),     # red inside near-black: the REC classic
        ((0.96, 0.96, 0.96), (0.03, 0.03, 0.04), 0.0),     # white inside near-black
        ((0.98, 0.86, 0.20), (0.04, 0.03, 0.02), 0.0),     # yellow inside near-black
        ((0.30, 0.92, 0.95), (0.02, 0.04, 0.05), 0.0),     # cyan inside near-black
    )

    def _apply_style(self) -> None:
        """Set the colour, outline and slant from the template, leaving them alone for index 0."""
        st = int(round(float(self.cfg_of("overlay").get("style", 0.0))))
        if st <= 0 or st >= len(self.STYLE_PRESETS):
            return
        colour, outline, italic = self.STYLE_PRESETS[st]
        self.colour = tuple(float(v) for v in colour)
        self.outline_colour = tuple(float(v) for v in outline)
        if self.italic == 0.0:
            self.italic = float(italic)

    def prepare(self) -> None:
        pass

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if not self.enabled or not self.text:
            return buf
        if self.at != self.slot:
            # this instance is in the slot the parameter did not choose; the other one draws
            return buf
        from ..core.signal import SignalFrame, composite_encode

        sig_in: SignalFrame | None = buf.meta.get("signal")
        # Draw onto the current signal's reconstructed picture when upstream has already formed a signal
        # (e.g. at SLOT_PRE_TAPE after format/codec/connector), NEVER onto the stale raw camera `buf.rgb`.
        # Drawing onto `buf.rgb` and re-encoding `SignalFrame.from_rgb(out)` discarded every change that
        # `FormatStage` (luma/chroma bandwidth, chroma_gain, color_noise) and `CodecStage` (MPEG-1 VCD 352x240
        # or MiniDV 4:1:1) had applied, causing the entire picture look to shift when `at=1` was selected!
        if self.slot == self.SLOT_PRE_TAPE and sig_in is not None:
            base_rgb = sig_in.to_rgb()
        else:
            base_rgb = buf.rgb.astype(np.float32, copy=False)

        gen = self._generator(base_rgb.shape[1], base_rgb.shape[0])
        if gen is None:
            return buf
        out, rep, mask = gen.draw_with_mask(base_rgb)
        self.report = rep
        if not np.any(mask > 0.0):
            return buf

        meta = dict(buf.meta)
        sig_osd = SignalFrame.from_rgb(out, buf.std)
        if self.slot == self.SLOT_PRE_TAPE and sig_in is not None:
            # Replace Y, I, Q ONLY on the pixels where the character generator actually drew (glyphs,
            # outline, shadow, or background box), preserving the upstream encoded Y, I, Q everywhere else
            # bit-for-bit.
            y_new = np.where(mask > 0.5, sig_osd.y, sig_in.y).astype(np.float32)
            i_new = np.where(mask > 0.5, sig_osd.i, sig_in.i).astype(np.float32)
            q_new = np.where(mask > 0.5, sig_osd.q, sig_in.q).astype(np.float32)
            sig = SignalFrame(y_new, i_new, q_new)
            if getattr(sig_in, "_rf_direct", False):
                sig._rf_direct = True
            meta["signal"] = sig
            comp = buf.composite
            if comp is not None:
                comp_osd = composite_encode(sig_osd.y, sig_osd.i, sig_osd.q, buf.std, chroma_gain=0.63)
                comp = np.where(mask > 0.5, comp_osd, comp).astype(np.float32)
            return FrameBuffer(out.astype(np.float32), comp, buf.index, buf.time, buf.std, meta)

        meta["signal"] = sig_osd
        comp = buf.composite
        if comp is not None:
            comp = composite_encode(sig_osd.y, sig_osd.i, sig_osd.q, buf.std, chroma_gain=0.63)
        return FrameBuffer(out.astype(np.float32), comp, buf.index, buf.time, buf.std, meta)

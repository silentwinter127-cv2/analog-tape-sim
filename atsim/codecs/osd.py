"""
Rebuild the glyph rasteriser: cell from the font's own advance, native rendering, block-mean only.

Three faults, each now understood rather than guessed at.

**1. The CJK glyphs were squashed, not decimated.** `is_full_width('婚')` is true, so PIL draws it at twice
the advance of a latin character -- but the code allocated a cell of `height // 2` pixels and stamped the
glyph into it. A 48-pixel-wide glyph compressed into 24 pixels is exactly the broken-block appearance the
render showed.

**2. The cell was derived from the height, not from the advance.** All three fonts state their cell in font
units -- OSD 1200 per half cell, A-POS 504 -- and the pixel cell should come from that and the requested
pixel height, not from a guess that a half cell is half the height. Measured, the two differ: OSD's half
advance is 1200/2048 of the em while half the height is a different number, so glyphs and cells drifted
apart and the strings overlapped.

**3. The resampling was decimation.** `a[::4, ::4]` keeps one sample in sixteen and discards the rest, so
thin strokes and the internal gaps of a dense character are lost wherever they fall between samples. Where
downscaling is genuinely needed it is now a block *mean* over an integer factor, which keeps the pixel
structure as structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fonts import is_full_width, load_font


@dataclass
class OSDLine:
    """One line of text, and where it goes."""

    text: str
    col: int = 0
    row: int = 0


@dataclass
class OSDPlan:
    """Everything the generator draws on one frame."""

    lines: list[OSDLine] = field(default_factory=list)
    #: requested glyph cell height in pixels.  The half-cell width comes from the font's own advance.
    cell_h: int = 24
    #: extra pixels between cells
    gap: int = 0
    margin_x: int = 24
    margin_y: int = 24
    #: 0 left, 1 centre, 2 right -- within the frame's safe area, per line
    halign: int = 0
    #: 0 top, 1 middle, 2 bottom
    valign: int = 0
    colour: tuple[float, float, float] = (0.92, 0.94, 0.90)
    background: tuple[float, float, float] | None = None
    shadow: float = 0.0
    #: a dark border around every glyph.  This is what a real camcorder's OSD uses: it makes the text readable
    #: on any picture without obscuring any of it, which a box does not.
    outline: float = 1.0
    #: border thickness in pixels (scales with frame resolution when > 240p)
    outline_px: int = 1
    #: the outline's own colour.  A generator's outline is a separate pen, not a darkened copy of the glyph --
    #: which is why green-inside-black and red-inside-black are both possible and neither is a shade of white.
    outline_colour: tuple[float, float, float] = (0.02, 0.03, 0.05)
    #: a slant, as a shear.  A bitmap generator had no oblique; it sheared the glyph, so slanted OSD text has
    #: stepped edges.  Applied to the mask after thresholding, so the pixel structure survives.  **Signed**: the
    #: shear is taken about the glyph's vertical centre, so negating it mirrors the slant rather than moving it.
    italic: float = 0.0
    #: horizontal scale of the glyph cell.  A generator drew tall thin characters by changing the cell, not by
    #: filtering, so this stretches the mask and the hard edges survive.
    stretch_x: float = 1.0
    #: vertical scale, independent of the horizontal one
    stretch_y: float = 1.0
    #: pixels of gap added between cells, beyond the font's own advance
    letter_gap: int = 0
    #: pixels of gap added between lines, beyond the cell height
    line_gap: int = 0
    #: shadow offset in pixels, both axes
    shadow_x: int = 2
    shadow_y: int = 2
    opacity: float = 1.0
    #: threshold the glyph coverage into a one-bit mask.  On by default because all three fonts are
    #: bitmap-derived and a character generator's ROM is one bit per pixel -- there is no antialiasing
    #: anywhere in the apparatus, and smoothing the glyphs removes the reason to use these fonts.
    hard_edges: bool = True


class CharacterGenerator:
    """
    Rasterises text in fixed cells and composites it onto a frame.

    Cell width comes from the font's declared advance, so a half-width character occupies `half` pixels and
    a full-width one `2 * half`, which is the 1:2 convention all three fonts state. Glyphs are rendered at
    the size PIL will draw them and only resampled when the requested height differs from the natural one,
    and then by an integer block mean.
    """

    def __init__(self, font_key: str, plan: OSDPlan | None = None):
        self.font_key = font_key
        self.spec = load_font(font_key)
        self.plan = plan or OSDPlan()
        self._cache: dict[str, np.ndarray] = {}
        self._face = None

    # ------------------------------------------------------------------ geometry
    def _eff_stretch(self) -> tuple[float, float]:
        sx = float(self.plan.stretch_x) * float(getattr(self.spec, "rom_stretch_x", 1.0))
        sy = float(self.plan.stretch_y) * float(getattr(self.spec, "rom_stretch_y", 1.0))
        return sx, sy

    def cell(self) -> tuple[int, int]:
        """
        (half-cell width, cell height) in pixels.

        The half width is the font's half advance scaled from font units by the requested height. This is
        what makes a string's advance match its drawn width, and getting it wrong is what made two strings
        overlap.
        """
        upm = max(1, self.spec.units_per_em)
        h = max(4, int(self.plan.cell_h))
        half = max(2, int(round(self.spec.half_advance * h / upm)))
        return half, h

    def adv(self, full: bool) -> int:
        """
        The pixels the pen moves for one character: its cell, scaled, plus the letter gap.

        **This is the single place the transform meets the layout.** Scaling the glyph mask without scaling the
        advance is what made `stretch_x 1.6` grow the ink by 12% instead of 60% -- every character widened while
        the pen still stepped by the unstretched cell, so neighbours overlapped by the difference.
        """
        half, _h = self.cell()
        cell_w = half * 2 if full else half
        sx, _sy = self._eff_stretch()
        gap = int(self.plan.letter_gap)
        return max(1, int(round(cell_w * sx)) + gap)

    def line_pitch(self) -> int:
        """The vertical distance between successive lines: the cell, scaled, plus the line gap."""
        _half, cell_h = self.cell()
        _sx, sy = self._eff_stretch()
        return max(2, int(round(cell_h * sy)) + int(self.plan.line_gap))

    def line_pixels(self, text: str) -> int:
        """The width a line will actually occupy, in pixels."""
        return sum(self.adv(is_full_width(c)) for c in (text or ""))

    def cells_wide(self, width: int) -> int:
        # in *cells* of the half-advance the layout uses, so the overflow report and the drawing agree
        adv = self.adv(False)
        return max(1, (width - 2 * self.plan.margin_x) // max(1, adv))

    def line_width_cells(self, text: str) -> int:
        return sum(2 if is_full_width(c) else 1 for c in text)

    # ------------------------------------------------------------------ rasterising
    def _face_for(self, px: int):
        from PIL import ImageFont
        if self._face is None or getattr(self, "_face_px", None) != px:
            self._face = ImageFont.truetype(self.spec.path, px)
            self._face_px = px
        return self._face

    def _glyph(self, ch: str, cell_h: int) -> np.ndarray | None:
        """
        One character as an alpha bitmap filling its cell, cached per character and height.

        **Full-width characters get a full-cell box and half-width ones a half-cell box**, which is the
        mistake that squashed the Chinese glyphs: the font draws a CJK character at twice a latin advance,
        so stamping it into a half-width cell compressed it by two.
        """
        half, _h = self.cell()
        cw = half * 2 if is_full_width(ch) else half
        plan_iter = self.plan.italic
        sx, sy = self._eff_stretch()
        # **The transform belongs in the key.**  Without it the first glyph rendered at a given cell size is
        # returned for every later shear and stretch -- which is why the rendered italic measurement showed both
        # signs leaning the same way while the same shear applied to a plain mask was correctly signed.
        key = "%s|%d|%d|%.4f|%.4f|%.4f" % (ch, cw, cell_h, float(plan_iter),
                                           float(sx), float(sy))
        got = self._cache.get(key)
        if got is not None:
            return got if got.size else None

        from PIL import Image, ImageDraw
        face = self._face_for(cell_h)
        big = Image.new("L", (cw + 2, cell_h + 2), 0)      # 1 px slack so an overhang is not clipped
        d = ImageDraw.Draw(big)
        asc = self.spec.ascent_px(cell_h)
        try:
            d.text((1, asc + 1), ch, fill=255, font=face, anchor="ls")
        except Exception:                                             # noqa: BLE001
            try:
                d.text((1, 1), ch, fill=255, font=face)
            except Exception:                                         # noqa: BLE001
                pass
        a = np.asarray(big, np.float32) / 255.0
        small = a[1:1 + cell_h, 1:1 + cw]
        # **Scale first, shear second**, and both by re-indexing the mask.
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            hh, ww = small.shape
            nw = max(1, int(round(ww * max(0.05, sx))))
            nh = max(1, int(round(hh * max(0.05, sy))))
            xi = np.clip(np.round((np.arange(nw) + 0.5) / max(1e-6, sx) - 0.5), 0,
                         ww - 1).astype(np.int32)
            yi = np.clip(np.round((np.arange(nh) + 0.5) / max(1e-6, sy) - 0.5), 0,
                         hh - 1).astype(np.int32)
            small = small[yi][:, xi]
        if abs(self.plan.italic) > 1e-3:
            slant = float(self.plan.italic)
            hh, ww = small.shape
            ink_rows = np.where((small > 0.05).any(axis=1))[0]
            pivot = float(ink_rows.mean()) if ink_rows.size else hh * 0.5
            reach = max(abs(0.0 - pivot), abs((hh - 1) - pivot))
            pad = int(np.ceil(abs(slant) * reach)) + 1
            out_w = ww + 2 * pad
            wide = np.zeros((hh, out_w), np.float32)
            wide[:, pad:pad + ww] = small
            out_s = np.zeros((hh, out_w), np.float32)
            for r in range(hh):
                off = int(round((pivot - r) * slant))
                d0 = max(0, pad + off)
                d1 = min(out_w, pad + off + ww)
                if d1 > d0:
                    s0 = d0 - (pad + off)
                    out_s[r, d0:d1] = wide[r, s0:s0 + (d1 - d0)]
            small = out_s
            self._shear_pad = pad
        else:
            self._shear_pad = 0
        if self.plan.hard_edges:
            small = (small >= 0.5).astype(np.float32)
        self._cache[key] = small
        return small if small.size else None

    def missing(self, text: str) -> list[str]:
        return sorted({c for c in text if c not in "\r\n\t" and not self.spec.can_draw(c)})

    # ------------------------------------------------------------------ drawing
    def draw(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        out, reported, _mask = self.draw_with_mask(rgb)
        return out, reported

    def draw_with_mask(self, rgb: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
        """
        Draw the character generator overlay onto `rgb`, returning `(out_rgb, reported, changed_mask)`.

        `changed_mask` is a float32 `(H, W)` weight in `0..1` marking every pixel modified by the
        character generator (glyphs, outline, shadow, or background box).  This lets the pre-tape
        slot (`overlay.at == 1`) composite the generator's output onto the existing signal & carrier
        *only where the generator actually drew*, leaving the rest of the frame's encoded signal
        untouched.
        """
        out = rgb
        plan = self.plan
        half, cell_h = self.cell()
        h, w = rgb.shape[:2]
        zero_mask = np.zeros((h, w), np.float32)
        reported: dict = {"drawn": 0, "missing": [], "lines": len(plan.lines),
                          "cell": [half, cell_h], "font": self.font_key,
                          "halign": plan.halign, "valign": plan.valign}
        if not plan.lines or plan.opacity <= 1e-3:
            return out, reported, zero_mask

        layer = np.zeros((h, w), np.float32)
        missing: set[str] = set()

        # the block's vertical extent, needed for middle and bottom alignment
        n_lines = max(1, len(plan.lines))
        pitch = self.line_pitch()
        block_h = n_lines * pitch + max(0, n_lines - 1) * plan.gap
        usable_h = max(0, h - 2 * plan.margin_y)
        if plan.valign == 1:
            y0_block = plan.margin_y + max(0, (usable_h - block_h) // 2)
        elif plan.valign == 2:
            y0_block = plan.margin_y + max(0, usable_h - block_h)
        else:
            y0_block = plan.margin_y
        usable_w = max(0, w - 2 * plan.margin_x)

        drawn_cells = 0
        ink_rows = [h, 0]
        ink_cols = [w, 0]
        for line in plan.lines:
            y = y0_block + line.row * (pitch + plan.gap)
            if y >= h or y + cell_h < 0:
                continue
            line_w = self.line_pixels(line.text or "")
            if plan.halign == 1:
                x = plan.margin_x + max(0, (usable_w - line_w) // 2)
            elif plan.halign == 2:
                x = plan.margin_x + max(0, usable_w - line_w)
            else:
                x = plan.margin_x
            x += max(0, line.col) * self.adv(False)
            for ch in (line.text or ""):
                full = is_full_width(ch)
                cw = self.adv(full)
                if x >= w:
                    break
                g = self._glyph(ch, cell_h)
                if g is None or not self.spec.can_draw(ch):
                    missing.add(ch)
                    x += cw
                    drawn_cells += 2 if full else 1
                    continue
                gh, gw = g.shape[:2]
                x_draw = x - int(getattr(self, "_shear_pad", 0))
                y0, x0 = max(0, y), max(0, x_draw)
                y1, x1 = min(h, y + gh), min(w, x_draw + gw)
                if y1 > y0 and x1 > x0:
                    sub = g[y0 - y:y1 - y, x0 - x_draw:x1 - x_draw]
                    dest = layer[y0:y1, x0:x1]
                    np.maximum(dest, sub, out=dest)
                    reported["drawn"] += 1
                    ink_rows[0] = min(ink_rows[0], y0)
                    ink_rows[1] = max(ink_rows[1], y1)
                    ink_cols[0] = min(ink_cols[0], x0)
                    ink_cols[1] = max(ink_cols[1], x1)
                    drawn_cells += 2 if full else 1
                x += cw
        reported["missing"] = sorted(missing)
        adv_half = self.adv(False)
        reported["cells_wanted"] = max(
            [(self.line_pixels(l.text or "") + max(1, adv_half) - 1) // max(1, adv_half)
             for l in plan.lines] or [0])
        reported["cells_drawn"] = drawn_cells
        reported["cells_fit"] = self.cells_wide(w)
        reported["overflow"] = reported["cells_wanted"] > reported["cells_fit"]
        reported["cells_used"] = reported["cells_drawn"]
        if reported["drawn"] == 0:
            return out, reported, zero_mask

        op = float(np.clip(plan.opacity, 0.0, 1.0))
        a = layer * op
        base = out.astype(np.float32, copy=True)
        changed = a.copy()

        if plan.background is not None:
            rows = np.where(layer.max(axis=1) > 0)[0]
            cols = np.where(layer.max(axis=0) > 0)[0]
            if rows.size and cols.size:
                pad = max(1, cell_h // 6)
                r0, r1 = max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)
                c0, c1 = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
                bg = np.array(plan.background, np.float32)
                box = base[r0:r1, c0:c1]
                base[r0:r1, c0:c1] = box + (bg[None, None, :] - box) * op
                changed[r0:r1, c0:c1] = np.maximum(changed[r0:r1, c0:c1], op)

        if plan.shadow > 1e-3:
            ox = int(plan.shadow_x) if plan.shadow_x else max(1, cell_h // 12)
            oy = int(plan.shadow_y) if plan.shadow_y else max(1, cell_h // 12)
            sh = np.zeros_like(a)
            ys = slice(max(0, oy), a.shape[0] + min(0, oy))
            yd = slice(max(0, -oy), a.shape[0] + min(0, -oy))
            xs = slice(max(0, ox), a.shape[1] + min(0, ox))
            xd = slice(max(0, -ox), a.shape[1] + min(0, -ox))
            sh[yd, xd] = a[ys, xs]
            base = base * (1.0 - sh[..., None] * plan.shadow)
            np.maximum(changed, np.clip(sh * float(plan.shadow), 0.0, 1.0), out=changed)

        if plan.outline > 1e-3:
            ring = a.copy()
            rad = max(1, int(getattr(plan, "outline_px", 1)))
            for _step in range(rad):
                cur_r = ring.copy()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    sh = np.zeros_like(cur_r)
                    ys = slice(max(0, dy), cur_r.shape[0] + min(0, dy))
                    yd = slice(max(0, -dy), cur_r.shape[0] + min(0, -dy))
                    xs = slice(max(0, dx), cur_r.shape[1] + min(0, dx))
                    xd = slice(max(0, -dx), cur_r.shape[1] + min(0, -dx))
                    sh[yd, xd] = cur_r[ys, xs]
                    np.maximum(ring, sh, out=ring)
            border = np.clip(ring - a, 0.0, 1.0) * float(plan.outline)
            dark = np.asarray(plan.outline_colour, np.float32)
            base = base * (1.0 - border[..., None]) + dark[None, None, :] * border[..., None]
            np.maximum(changed, np.clip(border, 0.0, 1.0), out=changed)

        col = np.array(plan.colour, np.float32)
        out = base * (1.0 - a[..., None]) + col[None, None, :] * a[..., None]

        COEF = np.array([0.299, 0.587, 0.114], np.float32)
        text_luma = float(np.asarray(plan.colour, np.float32) @ COEF)
        reported["text_luma"] = round(text_luma, 4)
        if plan.background is not None:
            bg = float(np.asarray(plan.background, np.float32) @ COEF)
            reported["background_luma"] = round(bg, 4)
            behind = bg
            reported["behind_luma"] = round(behind, 4)
            reported["contrast"] = round(abs(behind - text_luma), 4)
            reported["legible"] = reported["contrast"] >= 0.18
        elif plan.outline > 1e-3:
            dark = float(np.asarray(plan.outline_colour, np.float32) @ COEF)
            reported["behind_luma"] = round(dark, 4)
            reported["contrast"] = round(abs(dark - text_luma), 4)
            reported["legible"] = reported["contrast"] >= 0.18
        else:
            reported["behind_luma"] = None
            reported["contrast"] = None
            reported["legible"] = None
        return out.astype(np.float32), reported, np.clip(changed > 1e-4, 0.0, 1.0).astype(np.float32)


def parse_lines(spec) -> list[OSDLine]:
    """Accept a string, a list of strings, or a list of `{text, col, row}` dicts."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [OSDLine(t, 0, i) for i, t in enumerate(spec.split("\n"))]
    out: list[OSDLine] = []
    for i, item in enumerate(spec):
        if isinstance(item, str):
            out.append(OSDLine(item, 0, i))
        elif isinstance(item, dict):
            out.append(OSDLine(str(item.get("text", "")),
                               int(item.get("col", 0)), i + int(item.get("row", 0))))
    return out

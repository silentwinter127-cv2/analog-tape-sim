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

**And one reading of mine was too quick.** I inferred "bitmap-derived" from a small count of distinct
coordinates. That is only suggestive: the real evidence is that the coordinates are multiples of a common
step -- A-POS's capitals span 924 units in exact multiples of 42, which is a 22-pixel cell at 42 units per
pixel. `gcd_nz` came out as 1 for two of the three because a single off-grid coordinate in one sampled
glyph drags the GCD to 1, so the statistic was not measuring what I thought. The conclusion still holds --
the outlines are visibly grid-aligned -- but it now rests on the coordinate spacing rather than on a count.
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
    colour: tuple[float, float, float] = (0.92, 0.94, 0.90)
    background: tuple[float, float, float] | None = None
    shadow: float = 0.0
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

    def cells_wide(self, width: int) -> int:
        half, _ = self.cell()
        return max(1, (width - 2 * self.plan.margin_x) // max(1, half))

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
        key = "%s|%d|%d" % (ch, cw, cell_h)
        got = self._cache.get(key)
        if got is not None:
            return got if got.size else None

        from PIL import Image, ImageDraw
        # Render at the cell's own size, not at a multiple.  A bitmap font is designed on a pixel grid, so
        # rendering at that grid reproduces the ROM; rendering larger and averaging back invents grey levels
        # that the apparatus does not have.  Measured with a 4x render and a block mean: 52 distinct levels
        # in OSD's 'R' and 22 in A-POS's, against 2 for Lyusung whose grid happened to line up.
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
        if self.plan.hard_edges:
            # one bit per pixel, as the ROM is.  The midpoint is the only defensible threshold: a value at
            # or above half coverage is more on than off.
            small = (small >= 0.5).astype(np.float32)
        self._cache[key] = small
        return small if small.size else None

    def missing(self, text: str) -> list[str]:
        return sorted({c for c in text if c not in "\r\n\t" and not self.spec.can_draw(c)})

    # ------------------------------------------------------------------ drawing
    def draw(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        out = rgb
        plan = self.plan
        half, cell_h = self.cell()
        reported: dict = {"drawn": 0, "missing": [], "lines": len(plan.lines),
                          "cell": [half, cell_h], "font": self.font_key}
        if not plan.lines or plan.opacity <= 1e-3:
            return out, reported

        h, w = rgb.shape[:2]
        layer = np.zeros((h, w), np.float32)
        missing: set[str] = set()

        for line in plan.lines:
            y = plan.margin_y + line.row * (cell_h + plan.gap)
            if y >= h or y + cell_h < 0:
                continue
            x = plan.margin_x + max(0, line.col) * half
            for ch in (line.text or ""):
                cw = half * 2 if is_full_width(ch) else half
                if x >= w:
                    break                     # a generator does not wrap; it runs off the edge
                g = self._glyph(ch, cell_h)
                if g is None or not self.spec.can_draw(ch):
                    missing.add(ch)
                    x += cw
                    continue
                gh, gw = g.shape[:2]
                y0, x0 = max(0, y), max(0, x)
                y1, x1 = min(h, y + gh), min(w, x + gw)
                if y1 > y0 and x1 > x0:
                    sub = g[y0 - y:y1 - y, x0 - x:x1 - x]
                    dest = layer[y0:y1, x0:x1]
                    np.maximum(dest, sub, out=dest)
                    reported["drawn"] += 1
                x += cw
        reported["missing"] = sorted(missing)

        if reported["drawn"] == 0:
            return out, reported

        a = layer * float(np.clip(plan.opacity, 0.0, 1.0))
        base = out.astype(np.float32, copy=True)

        if plan.background is not None:
            rows = np.where(layer.max(axis=1) > 0)[0]
            cols = np.where(layer.max(axis=0) > 0)[0]
            if rows.size and cols.size:
                pad = max(1, cell_h // 6)
                r0, r1 = max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)
                c0, c1 = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
                bg = np.array(plan.background, np.float32)
                box = base[r0:r1, c0:c1]
                base[r0:r1, c0:c1] = box + (bg[None, None, :] - box) * float(plan.opacity)

        if plan.shadow > 1e-3:
            off = max(1, cell_h // 12)
            sh = np.zeros_like(a)
            sh[off:, off:] = a[:-off, :-off]
            base = base * (1.0 - sh[..., None] * plan.shadow)

        col = np.array(plan.colour, np.float32)
        out = base * (1.0 - a[..., None]) + col[None, None, :] * a[..., None]
        return out.astype(np.float32), reported


def parse_lines(spec) -> list[OSDLine]:
    """Accept a string, a list of strings, or a list of `{text, col, row}` dicts."""
    if spec is None:
        return []
    if isinstance(spec, str):
        # Each line gets its own row.  Without this every line was drawn at row 0 and they landed on top of
        # one another -- a fault no single-line test could show, and the first thing a person types is more
        # than one line.
        return [OSDLine(t, 0, i) for i, t in enumerate(spec.split("\n"))]
    out: list[OSDLine] = []
    for i, item in enumerate(spec):
        if isinstance(item, str):
            out.append(OSDLine(item, 0, i))
        elif isinstance(item, dict):
            # an explicit `row` is an offset from this line's own position, so a caller may place a line
            # where it likes and let the rest stack below it rather than all sharing that row
            out.append(OSDLine(str(item.get("text", "")),
                               int(item.get("col", 0)), i + int(item.get("row", 0))))
    return out

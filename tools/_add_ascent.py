"""
Give `FontSpec` the ascender, because placing a glyph in a cell needs the baseline.

A character generator's ROM is indexed by cell and the glyph is positioned on a baseline; PIL positions text
by its own ascent, so the two have to be reconciled or the glyph sits partly outside its cell -- which is
what cut the first render's text in half.

`ascent_px` converts the font's `hhea.ascent` from font units to the requested pixel height, which is the
same conversion used for the advance, so the baseline and the cell come from one source and cannot disagree.
"""
from __future__ import annotations

import ast
import os

P = os.path.join("atsim", "codecs", "fonts.py")
src = open(P, encoding="utf-8").read()

# --- the fields ----------------------------------------------------------
OLD = """    units_per_em: int = 0
    glyph_count: int = 0
    covered: set[int] = field(default_factory=set)"""
NEW = """    units_per_em: int = 0
    glyph_count: int = 0
    #: `hhea.ascent` in font units, which is where the baseline sits in a cell
    ascent: int = 0
    descent: int = 0
    covered: set[int] = field(default_factory=set)

    def ascent_px(self, cell_h: int) -> int:
        \"\"\"
        Where the baseline sits, in pixels, for a cell of `cell_h`.

        The same conversion the advance uses, so the baseline and the cell width come from one source.
        A font whose ascent exceeds its em -- and all three here do to some degree -- places the baseline
        below the top of the cell by this amount, and a glyph is drawn upward from it.
        \"\"\"
        upm = max(1, self.units_per_em)
        if not self.ascent:
            return int(cell_h * 0.8)
        return int(round(self.ascent * cell_h / upm))"""

assert OLD in src, "the field block was not found"
src = src.replace(OLD, NEW, 1)

# --- read them -----------------------------------------------------------
OLD2 = """        spec.units_per_em = int(f["head"].unitsPerEm)
        spec.glyph_count = int(f["maxp"].numGlyphs)"""
NEW2 = """        spec.units_per_em = int(f["head"].unitsPerEm)
        spec.glyph_count = int(f["maxp"].numGlyphs)
        try:
            hhea = f["hhea"]
            spec.ascent = int(hhea.ascent)
            spec.descent = int(hhea.descent)
        except Exception:                                             # noqa: BLE001
            spec.ascent = int(spec.units_per_em * 0.8)
            spec.descent = 0"""
assert OLD2 in src, "the read block was not found"
src = src.replace(OLD2, NEW2, 1)

# --- report it -----------------------------------------------------------
OLD3 = '''        return {"key": self.key, "label": self.label, "scripts": list(self.scripts),
                "units_per_em": self.units_per_em, "glyph_count": self.glyph_count,
                "half_advance": self.half_advance, "full_advance": self.full_advance,
                "covers": len(self.covered)}'''
NEW3 = '''        return {"key": self.key, "label": self.label, "scripts": list(self.scripts),
                "units_per_em": self.units_per_em, "glyph_count": self.glyph_count,
                "half_advance": self.half_advance, "full_advance": self.full_advance,
                "ascent": self.ascent, "covers": len(self.covered)}'''
if OLD3 in src:
    src = src.replace(OLD3, NEW3, 1)

ast.parse(src)
open(P, "w", encoding="utf-8").write(src)
print("FontSpec now carries the ascender")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atsim.codecs.fonts import load_font
for k in ("osd", "lyusung", "apos"):
    s = load_font(k)
    half = round(s.half_advance * 24 / s.units_per_em)
    print("   %-8s upm %5d ascent %5d -> half cell %2d px at 24 px, baseline %2d px"
          % (k, s.units_per_em, s.ascent, half, s.ascent_px(24)))

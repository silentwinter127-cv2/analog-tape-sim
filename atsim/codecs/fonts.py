"""
The font registry: what each typeface can draw, and what its cell is.

A **character generator is not a typesetting engine.** An OSD chip has a font ROM, a fixed cell per
character, and a position register; it draws glyphs into cells and does not measure, kern, wrap or
reflow. That distinction decides everything here, and it is not a simplification -- it is what the
apparatus actually does, and it is why camcorder overlay text looks the way it does: evenly spaced,
fixed size, never proportional.

**All three of the user's fonts are bitmap-derived**, which the inspection established rather than the
file names:

    VCR-OSD-Mono   204 glyphs, ASCII only, one advance (1200), 10 distinct x values per glyph
    Lyusung       7577 glyphs, 6763 CJK, latin 1200 and CJK 2400, 12-13 distinct x values
    A-POS        22169 glyphs, 20902 CJK, latin 504 and CJK 1008, 11 distinct x values, 48% empty

Two facts follow and both matter:

  * **the advance ratio is exactly 1:2** in all three (1200:2400, 504:1008), so a cell is half-width for
    latin and full-width for CJK. That is the half/full-width convention of the era, and it is why a
    subtitle can mix scripts without a layout pass: every character occupies one or two half-cells;
  * **the outlines are grid-aligned**, so they must be rasterised with **nearest-neighbour** scaling. Any
    interpolation softens the pixel steps and the glyph stops looking like the character generator it came
    from. This is not a preference; it is the whole visual character of the font.

Coverage is a hard constraint the user gave, so it is recorded here and enforced: the OSD font has no CJK
at all, so Chinese in it must be reported rather than silently rendered as boxes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from fontTools.ttLib import TTFont


@dataclass
class FontSpec:
    """One typeface, as a character generator would see it: cells, coverage, and a file."""

    key: str
    label: str
    path: str
    #: which scripts this font can draw. Enforced, not advisory: a font with no CJK must not be
    #: silently asked for Chinese.
    scripts: tuple[str, ...]
    #: units of advance for a half-width cell, from the font itself
    half_advance: int = 0
    full_advance: int = 0
    units_per_em: int = 0
    glyph_count: int = 0
    #: `hhea.ascent` in font units, which is where the baseline sits in a cell
    ascent: int = 0
    descent: int = 0
    covered: set[int] = field(default_factory=set)

    def ascent_px(self, cell_h: int) -> int:
        """
        Where the baseline sits, in pixels, for a cell of `cell_h`.

        The same conversion the advance uses, so the baseline and the cell width come from one source.
        A font whose ascent exceeds its em -- and all three here do to some degree -- places the baseline
        below the top of the cell by this amount, and a glyph is drawn upward from it.
        """
        upm = max(1, self.units_per_em)
        if not self.ascent:
            return int(cell_h * 0.8)
        return int(round(self.ascent * cell_h / upm))

    def can_draw(self, ch: str) -> bool:
        return ord(ch) in self.covered

    @property
    def missing_scripts(self) -> tuple[str, ...]:
        return tuple(s for s in ("latin", "cjk") if s not in self.scripts)

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "scripts": list(self.scripts),
                "units_per_em": self.units_per_em, "glyph_count": self.glyph_count,
                "half_advance": self.half_advance, "full_advance": self.full_advance,
                "ascent": self.ascent, "covers": len(self.covered)}


#: The three fonts, with the paths the user gave. `scripts` records the *coverage rule* they stated:
#: the OSD font is English and digits only, the other two also do Chinese.
FONTS: dict[str, FontSpec] = {
    "osd": FontSpec(
        key="osd", label="VCR OSD Mono（英文/数字）",
        path=r"C:\Users\Administrator\Downloads\字体\字体\VCR-OSD-Mono\VCR-OSD-Mono\VCR-OSD-MONO-1-001-1.ttf",
        scripts=("latin",)),
    "lyusung": FontSpec(
        key="lyusung", label="柳宋（中英文数字）",
        path=r"C:\Users\Administrator\Downloads\字体\字体\柳宋_猫啃网\柳宋\Lyusung-210618.ttf",
        scripts=("latin", "cjk")),
    "apos": FontSpec(
        key="apos", label="A-POS 点阵（中英文数字）",
        path=r"C:\USERS\ADMINISTRATOR\APPDATA\LOCAL\MICROSOFT\WINDOWS\FONTS\A-POS点阵修改后.TTF",
        scripts=("latin", "cjk")),
    "ocr_a": FontSpec(
        key="ocr_a", label="OCR-A（机器可读，OSD 经典）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\OCR-A BT.ttf"), scripts=('latin',)),
    "ocr_b": FontSpec(
        key="ocr_b", label="OCR-B（机器可读）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\OCR-B 10 BT.ttf"), scripts=('latin',)),
    "ming": FontSpec(
        key="ming", label="Ming-Imperial（中文点阵风格）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\Ming-Imperial-1.ttf"), scripts=('latin', 'cjk')),
}

#: which character class a codepoint belongs to, since the cell size depends on it
CJK_RANGES = ((0x3000, 0x30FF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))


def is_full_width(ch: str) -> bool:
    """
    Whether this character takes a full cell rather than a half one.

    Determined from the ranges rather than from the font's advance, because the cell is a property of the
    character generator's ROM layout: full-width for CJK and the full-width forms, half for latin. Using
    the font's own advance would work too, but it makes the cell layout depend on which font is loaded,
    and a generator's layout does not change when you swap the ROM.
    """
    c = ord(ch)
    return any(lo <= c <= hi for lo, hi in CJK_RANGES)



#: the height at which the cell is measured.  Large enough that the rounded pixel advance is accurate to
#: better than one per cent, and small enough to be fast.
PROBE_H = 200


def _measure_cell(spec: "FontSpec") -> tuple[int, int, int]:
    """
    (half advance, full advance, ascent) in font units, measured from what the font draws.

    The probe is a run of Latin characters at `PROBE_H` pixels: its pixel width divided by the count gives the
    half-advance in pixels, and the same figure converted back to font units is what the rest of the model
    expects.  A CJK character is then drawn at twice that, which is the half/full convention all three fonts
    follow -- and it is checked rather than assumed, because a font whose CJK advance is not exactly double
    would otherwise be silently mis-laid-out.

    Falls back to the `hmtx` table only if the rasteriser cannot measure, so a font that cannot be drawn is
    still described.
    """
    from PIL import ImageFont
    upm = max(1, spec.units_per_em)
    try:
        face = ImageFont.truetype(spec.path, PROBE_H)
        probes = [("HHHHHHHH", 1), ("婚婚婚婚", 2)]
        widths = {}
        for text, cells in probes:
            try:
                w = float(face.getlength(text))
            except Exception:                                         # noqa: BLE001
                w = float(face.getsize(text)[0])
            per = w / max(1, len(text))
            widths[cells] = per
        half_px = widths.get(1, 0.0)
        full_px = widths.get(2, 0.0)
        if half_px <= 1e-6:
            raise ValueError("no latin width")
        half = int(round(half_px * upm / PROBE_H))
        # the CJK cell should be twice the latin one; where it is not, the larger of the two is used so
        # a wide glyph cannot overflow into its neighbour
        full = int(round(full_px * upm / PROBE_H)) if full_px > 1e-6 else half * 2
        if full < half * 1.5:
            full = half * 2
        ascent = int(round(face.getmetrics()[0] * upm / PROBE_H))
        return half, full, ascent
    except Exception:                                                 # noqa: BLE001
        # the fallback: the smallest advance that occurs more than once, doubled for full width
        try:
            f = TTFont(spec.path, fontNumber=0, lazy=True)
            cmap = f.getBestCmap()
            hmtx = f["hmtx"]
            widths: dict[int, int] = {}
            for c in range(0x20, 0x7F):
                g = cmap.get(c)
                if g and g in hmtx.metrics:
                    widths[hmtx[g][0]] = widths.get(hmtx[g][0], 0) + 1
            half = min(widths) if widths else upm // 2
            full = half * 2
            asc = int(f["hhea"].ascent)
            f.close()
            return half, full, asc
        except Exception:                                             # noqa: BLE001
            return upm // 2, upm, int(upm * 0.8)


def load_font(key: str) -> FontSpec:
    """
    Read a font's coverage and cell metrics, tolerating a malformed table.

    `Lyusung` declares an `OS/2` table at format 5 and truncates it, so `fontTools` raises while
    decompiling. A rasteriser only needs `cmap`, `glyf` and `hmtx`, so the font is perfectly usable while
    being unreadable for metadata -- two different questions, and a model that treats the second as fatal
    cannot use a file a user actually has.
    """
    spec = FONTS[key]
    if spec.covered:
        return spec
    if not os.path.exists(spec.path):
        raise FileNotFoundError(spec.path)
    f = TTFont(spec.path, fontNumber=0, lazy=True)
    try:
        spec.units_per_em = int(f["head"].unitsPerEm)
        spec.glyph_count = int(f["maxp"].numGlyphs)
        try:
            hhea = f["hhea"]
            spec.descent = int(hhea.descent)
            # `ascent` was measured with the cell; only fall back to the table when the measurement did
            # not produce one
            if not spec.ascent:
                spec.ascent = int(hhea.ascent)
        except Exception:                                             # noqa: BLE001
            spec.descent = 0
            if not spec.ascent:
                spec.ascent = int(spec.units_per_em * 0.8)
        cmap = f.getBestCmap()
        spec.covered = set(cmap.keys())
        # Measure the cell instead of reading the advance table.  `hmtx` describes *layout* -- how far to
        # move the pen -- and in 柳宋 and A-POS its most common value belongs to the CJK glyphs, so taking it
        # as the Latin advance doubled every Latin cell and put a gap after every character.  What the
        # generator needs is what the font draws, so the font is asked by drawing it.
        spec.half_advance, spec.full_advance, spec.ascent = _measure_cell(spec)
    finally:
        f.close()
    return spec


def coverage_report(key: str, text: str) -> tuple[list[str], list[str]]:
    """
    Which characters of `text` this font can draw, and which it cannot.

    Returned rather than assumed so the caller can report a missing glyph instead of drawing a box: a
    subtitle that silently loses its Chinese is worse than one that says it cannot.
    """
    spec = load_font(key)
    ok, missing = [], []
    for ch in text:
        if ch in "\r\n\t":
            continue
        (ok if ord(ch) in spec.covered else missing).append(ch)
    return ok, sorted(set(missing))


def font_catalogue() -> list[dict]:
    """Every font, with its coverage resolved, for the UI to offer."""
    out = []
    for key in FONTS:
        try:
            spec = load_font(key)
            d = spec.to_dict()
            d["available"] = True
        except Exception as exc:                                      # noqa: BLE001
            d = FONTS[key].to_dict()
            d["available"] = False
            d["error"] = str(exc)[:80]
        out.append(d)
    return out

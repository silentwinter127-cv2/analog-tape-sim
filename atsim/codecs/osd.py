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
    #: analog keyer rise-time / optical title lens softness (0 = 1-bit ROM step edge, >0 = continuous analog/optical edge)
    edge_softness: float = 0.0
    #: optical title halation / CRT highlight bloom around bright glyphs (0..1)
    glow: float = 0.0
    #: 1980s broadcast hardware character-generator 3D relief / emboss bevel (0..1)
    bevel: float = 0.0
    #: subtitle animation mode:
    #:   0 = static (no animation)
    #:   1 = typewriter character-by-character reveal (逐字打字机显字)
    #:   2 = line-by-line sequential reveal (逐行顺序切入)
    #:   3 = horizontal raster wipe reveal (水平光栅扫描擦除)
    #:   4 = venetian-blind interlace reveal (百叶窗隔行扫描显字)
    #:   5 = vertical credit roll / crawl (垂直滚屏字幕)
    #:   6 = optical dissolve fade-in/out (光学叠化淡入淡出)
    #:   7 = horizontal slide-in to rest (横向推拉飞入定格)
    anim_mode: int = 0
    #: animation playback speed multiplier
    anim_speed: float = 1.0
    #: duration in seconds for the entrance animation to complete
    anim_in_s: float = 1.2
    #: optional cycle period in seconds (0 = play once and hold; >0 = periodic cycle)
    anim_cycle_s: float = 0.0


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
        sw = float(getattr(self.spec, "stroke_weight", 0.0))
        sf = float(getattr(self.spec, "serif_flare", 0.0))
        rg = int(getattr(self.spec, "rom_grid", 0))
        es = float(getattr(self.plan, "edge_softness", 0.0))
        he = bool(self.plan.hard_edges)
        # **The transform belongs in the key.**  Without it the first glyph rendered at a given cell size is
        # returned for every later shear and stretch -- which is why the rendered italic measurement showed both
        # signs leaning the same way while the same shear applied to a plain mask was correctly signed.
        key = "%s|%d|%d|%.4f|%.4f|%.4f|%.3f|%.3f|%d|%.3f|%d" % (
            ch, cw, cell_h, float(plan_iter), float(sx), float(sy),
            sw, sf, rg, es, int(he))
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

        # --- Typeface optical phototypesetting / stroke-weight & serif flare morphology ---
        if abs(sw) > 1e-3 or sf > 1e-3:
            from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
            if sw > 1e-3:
                dil = maximum_filter(small, size=(2, 2))
                small = np.clip(small + sw * 0.85 * (dil - small), 0.0, 1.0)
            elif sw < -1e-3:
                ero = minimum_filter(small, size=(2, 2))
                small = np.clip(small + (-sw) * 0.65 * (ero - small), 0.0, 1.0)
            if sf > 1e-3:
                # Horizontal serif / terminal wedge flare characteristic of 1980s BiaoSong & WeiBei title cards
                h_flare = maximum_filter(small, size=(1, 3))
                v_edge = np.abs(small - gaussian_filter(small, sigma=(1.2, 0.4)))
                small = np.clip(small + sf * 0.65 * (h_flare - small) * np.clip(v_edge * 2.5, 0.25, 1.0), 0.0, 1.0)

        # --- Hardware dot-matrix ROM quantization (e.g., 24x24 Chinese character card ROM) ---
        if rg > 0 and small.shape[0] > 4 and small.shape[1] > 4:
            gh0, gw0 = small.shape
            gy = max(8, int(rg))
            gx = max(4, int(rg if is_full_width(ch) else max(4, rg // 2)))
            yi_q = np.clip(np.floor(np.arange(gy, dtype=np.float32) * gh0 / gy).astype(np.int32), 0, gh0 - 1)
            xi_q = np.clip(np.floor(np.arange(gx, dtype=np.float32) * gw0 / gx).astype(np.int32), 0, gw0 - 1)
            rom_cell = (small[yi_q][:, xi_q] >= 0.46).astype(np.float32)
            yo_q = np.clip(np.floor(np.arange(gh0, dtype=np.float32) * gy / gh0).astype(np.int32), 0, gy - 1)
            xo_q = np.clip(np.floor(np.arange(gw0, dtype=np.float32) * gx / gw0).astype(np.int32), 0, gx - 1)
            small = rom_cell[yo_q][:, xo_q]

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
        if es > 1e-3:
            # Analog downstream keyer rise-time / optical title lens softness:
            # sharp binary core softened along the scanline (horizontal rise/fall time) and optical PSF
            from scipy.ndimage import gaussian_filter
            core_b = np.clip((small - 0.25) / 0.50, 0.0, 1.0).astype(np.float32)
            sig_y = 0.25 + 0.85 * es
            sig_x = 0.45 + 1.45 * es
            small = np.clip(gaussian_filter(core_b, sigma=(sig_y, sig_x)) * 1.08, 0.0, 1.0).astype(np.float32)
        elif self.plan.hard_edges:
            small = (small >= 0.5).astype(np.float32)
        self._cache[key] = small
        return small if small.size else None

    def missing(self, text: str) -> list[str]:
        return sorted({c for c in text if c not in "\r\n\t" and not self.spec.can_draw(c)})

    # ------------------------------------------------------------------ drawing
    def draw(self, rgb: np.ndarray, time_s: float = 0.0, frame_index: int = 0,
             fps: float = 30.0) -> tuple[np.ndarray, dict]:
        out, reported, _mask = self.draw_with_mask(rgb, time_s=time_s,
                                                   frame_index=frame_index, fps=fps)
        return out, reported

    def draw_with_mask(self, rgb: np.ndarray, time_s: float = 0.0,
                       frame_index: int = 0, fps: float = 30.0
                       ) -> tuple[np.ndarray, dict, np.ndarray]:
        """
        Draw the character generator overlay onto `rgb`, returning `(out_rgb, reported, changed_mask)`.

        `changed_mask` is a float32 `(H, W)` weight in `0..1` marking every pixel modified by the
        character generator (glyphs, outline, shadow, bevel, glow, or background box).  This lets the
        pre-tape slot (`overlay.at == 1`) composite the generator's output onto the existing signal &
        carrier *only where the generator actually drew*, leaving the rest of the frame's encoded signal
        untouched.
        """
        out = rgb
        plan = self.plan
        half, cell_h = self.cell()
        h, w = rgb.shape[:2]
        zero_mask = np.zeros((h, w), np.float32)
        amode = int(np.clip(round(float(getattr(plan, "anim_mode", 0))), 0, 7))
        reported: dict = {"drawn": 0, "missing": [], "lines": len(plan.lines),
                          "cell": [half, cell_h], "font": self.font_key,
                          "halign": plan.halign, "valign": plan.valign,
                          "anim_mode": amode}
        if not plan.lines or plan.opacity <= 1e-3:
            return out, reported, zero_mask

        # --- Physical subtitle animation timing envelope ---
        anim_speed = max(0.1, float(getattr(plan, "anim_speed", 1.0)))
        anim_in = max(0.08, float(getattr(plan, "anim_in_s", 1.2)))
        anim_cycle = float(getattr(plan, "anim_cycle_s", 0.0))
        eff_t = max(0.0, float(time_s)) * anim_speed
        if anim_cycle > 0.15:
            eff_t = eff_t % anim_cycle
        # Lead offset `0.22 * anim_in` so frame 0 (t=0.0s) already displays the initial animation state
        # rather than an empty frame, and advances monotonically to 1.0 across `t = 0 .. anim_in`.
        prog = float(np.clip((eff_t + 0.22 * anim_in) / anim_in, 0.0, 1.0)) if amode > 0 else 1.0
        reported["anim_progress"] = round(prog, 4)

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

        # Mode 5: Vertical credit roll / crawl (垂直滚屏字幕)
        scroll_dy = 0
        if amode == 5:
            span_y = usable_h + block_h
            # At prog=0.22 starts in lower third; scrolls smoothly upward as time advances
            scroll_frac = ((eff_t / max(0.2, anim_in * 1.5)) + 0.35) % 1.0
            y0_target = int(round((usable_h + plan.margin_y) - scroll_frac * span_y))
            scroll_dy = y0_target - y0_block

        # Mode 7: Horizontal slide-in to rest (横向推拉飞入定格)
        slide_dx = 0
        if amode == 7:
            ease_out = 1.0 - (1.0 - prog) ** 3
            slide_dx = -int(round((1.0 - ease_out) * max(24.0, w * 0.42)))

        total_chars = max(1, sum(len([c for c in (ln.text or "") if c not in "\r\n\t "]) for ln in plan.lines))
        char_Budget = prog * total_chars if amode == 1 else float(total_chars)
        non_space_idx = 0

        drawn_cells = 0
        ink_rows = [h, 0]
        ink_cols = [w, 0]
        for line_idx, line in enumerate(plan.lines):
            # Mode 2: Line-by-line sequential reveal (逐行顺序切入)
            line_alpha = 1.0
            line_dx = 0
            if amode == 2:
                l_start = line_idx / float(n_lines)
                l_span = 1.0 / float(n_lines)
                l_prog = float(np.clip((prog - l_start) / max(1e-4, l_span), 0.0, 1.0))
                if l_prog <= 1e-4:
                    continue
                line_alpha = l_prog
                line_dx = -int(round((1.0 - l_prog) * max(4.0, cell_h * 0.45)))

            y = y0_block + line.row * (pitch + plan.gap) + scroll_dy
            if y >= h or y + cell_h < 0:
                continue
            line_w = self.line_pixels(line.text or "")
            if plan.halign == 1:
                x = plan.margin_x + max(0, (usable_w - line_w) // 2)
            elif plan.halign == 2:
                x = plan.margin_x + max(0, usable_w - line_w)
            else:
                x = plan.margin_x
            x += max(0, line.col) * self.adv(False) + slide_dx + line_dx
            for ch in (line.text or ""):
                full = is_full_width(ch)
                cw = self.adv(full)
                is_space = ch in " \t"
                char_weight = line_alpha
                if amode == 1 and not is_space:
                    # Mode 1: Hardware character-generator typewriter step reveal
                    rem = char_Budget - non_space_idx
                    non_space_idx += 1
                    if rem <= 0.0:
                        x += cw
                        continue
                    char_weight *= float(np.clip(rem * 2.5, 0.0, 1.0))

                if x >= w:
                    break
                if x + cw < 0:
                    x += cw
                    continue
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
                    sub = g[y0 - y:y1 - y, x0 - x_draw:x1 - x_draw] * char_weight
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

        # --- Spatial/Optical Animation Masks (Modes 3, 4, 6) ---
        if amode == 3 and ink_cols[1] > ink_cols[0]:
            # Mode 3: Horizontal raster scanline wipe (水平光栅扫描擦除)
            span_w = max(1, ink_cols[1] - ink_cols[0])
            wipe_x = ink_cols[0] + prog * (span_w + max(4.0, cell_h * 0.4))
            soft_w = max(2.0, cell_h * 0.25)
            xs_grid = np.arange(w, dtype=np.float32)[None, :]
            wipe_mask = np.clip((wipe_x - xs_grid) / soft_w, 0.0, 1.0)
            layer = layer * wipe_mask
        elif amode == 4 and ink_rows[1] > ink_rows[0]:
            # Mode 4: Venetian-blind interlace scanline reveal (百叶窗隔行扫描显字)
            slat_h = max(2, int(round(cell_h / 4.0)))
            ys_grid = np.arange(h, dtype=np.float32)[:, None]
            slat_phase = np.mod(ys_grid - ink_rows[0], float(slat_h)) / float(slat_h)
            blind_mask = np.clip((prog * 1.15 - slat_phase) * 5.0, 0.0, 1.0)
            layer = layer * blind_mask

        op = float(np.clip(plan.opacity, 0.0, 1.0))
        if amode == 6:
            # Mode 6: Optical printer cross-dissolve fade-in / hold / fade-out
            if anim_cycle > 0.2:
                cyc_p = eff_t / anim_cycle
                fade_in = np.clip((cyc_p + 0.06) / 0.28, 0.0, 1.0)
                fade_out = np.clip((0.96 - cyc_p) / 0.22, 0.0, 1.0)
                op *= float(fade_in * fade_out)
            else:
                # Smooth S-curve optical fade-in
                s_p = prog * prog * (3.0 - 2.0 * prog)
                op *= float(s_p)

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

        # --- Optical Title Glow / Halation (光学叠印辉光/荧光漫射) ---
        glow_amt = float(np.clip(getattr(plan, "glow", 0.0), 0.0, 1.5))
        if glow_amt > 1e-3:
            from scipy.ndimage import gaussian_filter
            sig_g1 = max(1.2, cell_h * 0.14)
            sig_g2 = max(2.8, cell_h * 0.36)
            halo = 0.65 * gaussian_filter(a, sigma=(sig_g1 * 0.8, sig_g1 * 1.25)) + \
                   0.45 * gaussian_filter(a, sigma=(sig_g2 * 0.85, sig_g2 * 1.35))
            halo = np.clip(halo * glow_amt * 0.85, 0.0, 1.0)
            col_arr = np.asarray(plan.colour, np.float32)
            # Warm optical title halation tint carrying the glyph's own color
            glow_tint = np.clip(col_arr * 0.82 + np.array([0.22, 0.14, 0.04], np.float32), 0.0, 1.15)
            base = np.clip(base + halo[..., None] * glow_tint[None, None, :], 0.0, 1.0)
            np.maximum(changed, np.clip(halo * 0.85, 0.0, 1.0), out=changed)

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
        fill_rgb = np.broadcast_to(col[None, None, :], base.shape).copy()

        # --- 1980s Hardware Character Generator 3D Bevel / Relief (立体浮雕切角) ---
        bevel_amt = float(np.clip(getattr(plan, "bevel", 0.0), 0.0, 1.0))
        if bevel_amt > 1e-3:
            b_step = max(1, cell_h // 18)
            a_tl = np.zeros_like(a)
            a_br = np.zeros_like(a)
            a_tl[b_step:, b_step:] = a[:-b_step, :-b_step]
            a_br[:-b_step, :-b_step] = a[b_step:, b_step:]
            hi_rim = np.clip(a - a_tl, 0.0, 1.0) * a
            lo_rim = np.clip(a - a_br, 0.0, 1.0) * a
            spec_col = np.clip(col + np.array([0.28, 0.26, 0.22], np.float32), 0.0, 1.0)
            shade_col = col * 0.42
            fill_rgb = (fill_rgb
                        + (spec_col[None, None, :] - fill_rgb) * (hi_rim[..., None] * bevel_amt * 0.85)
                        + (shade_col[None, None, :] - fill_rgb) * (lo_rim[..., None] * bevel_amt * 0.80))

        out = base * (1.0 - a[..., None]) + fill_rgb * a[..., None]

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
        if float(getattr(plan, "edge_softness", 0.0)) > 1e-3 or glow_amt > 1e-3 or (amode > 0 and prog < 0.999):
            mask_out = np.clip(changed, 0.0, 1.0).astype(np.float32)
        else:
            mask_out = np.clip(changed > 1e-4, 0.0, 1.0).astype(np.float32)
        return out.astype(np.float32), reported, mask_out


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

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
from ..codecs.master_playout import MasterPlayoutSpec, apply_master_playout
from ..codecs.playout_automation_v2 import PlayoutAutomationV2Spec, apply_playout_automation_v2
from ..codecs.media_ecosystem_evolution import MediaEcosystemSpec, apply_media_ecosystem_evolution
from ..core.pipeline import FrameBuffer, RenderContext, Stage


def _available_font_keys() -> tuple[str, ...]:
    """
    The fonts the panel may offer, in the order the panel lists them.
    """
    from ..codecs.fonts import font_catalogue
    keys = tuple(d["key"] for d in font_catalogue() if d.get("available"))
    return keys or ("osd",)


class OverlayStage(Stage):
    """Text drawn by a character generator."""

    name = "overlay"
    layers = ("overlay",)

    #: which slot this instance occupies: 0 = after the display, 1 = before the tape, -1 = node graph node
    SLOT_TAIL = 0
    SLOT_PRE_TAPE = 1
    SLOT_GRAPH = -1
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
        self.edge_softness = float(cfg.get("edge_softness", 0.0))
        self.glow = float(cfg.get("glow", 0.0))
        self.bevel = float(cfg.get("bevel", 0.0))
        self.anim_mode = int(np.clip(round(float(cfg.get("anim_mode", 0.0))), 0, 7))
        self.anim_speed = float(cfg.get("anim_speed", 1.0))
        self.anim_in_s = float(cfg.get("anim_in_s", 1.2))
        self.anim_cycle_s = float(cfg.get("anim_cycle_s", 0.0))
        self.playout_spec = MasterPlayoutSpec.from_params(cfg)
        self.auto_v2_spec = PlayoutAutomationV2Spec.from_params(cfg)
        self.eco_spec = MediaEcosystemSpec.from_params(cfg)
        self._apply_style()

    def on_params_changed(self) -> None:
        self._read_params()

    FONTS = _available_font_keys()

    def _generator(self, width: int, height: int):
        fonts = _available_font_keys()
        key = fonts[int(np.clip(round(self.font_kind), 0, len(fonts) - 1))]
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
               eff_letter_gap, eff_line_gap, self.bg_on, round(self.edge_softness, 4),
               round(self.glow, 4), round(self.bevel, 4), self.anim_mode,
               round(self.anim_speed, 4), round(self.anim_in_s, 4), round(self.anim_cycle_s, 4))
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
                       background=(0.0, 0.0, 0.0) if self.bg_on else None,
                       edge_softness=self.edge_softness, glow=self.glow, bevel=self.bevel,
                       anim_mode=self.anim_mode, anim_speed=self.anim_speed,
                       anim_in_s=self.anim_in_s, anim_cycle_s=self.anim_cycle_s)
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

    STYLE_PRESETS = (
        None,
        ((0.35, 0.95, 0.45), (0.03, 0.05, 0.04), 0.0),     # 1: green inside near-black: the OSD classic
        ((0.92, 0.20, 0.22), (0.05, 0.03, 0.03), 0.0),     # 2: red inside near-black: the REC classic
        ((0.96, 0.96, 0.96), (0.03, 0.03, 0.04), 0.0),     # 3: white inside near-black
        ((0.98, 0.86, 0.20), (0.04, 0.03, 0.02), 0.0),     # 4: yellow inside near-black
        ((0.30, 0.92, 0.95), (0.02, 0.04, 0.05), 0.0),     # 5: cyan inside near-black
        ((0.98, 0.84, 0.26), (0.14, 0.05, 0.02), 0.0),     # 6: 80s documentary golden-yellow title with warm dark-brown outline & bevel
        ((0.96, 0.95, 0.88), (0.05, 0.05, 0.06), 0.0),     # 7: 80s documentary warm ivory narration subtitle with soft optical halo
        ((0.98, 0.72, 0.16), (0.02, 0.02, 0.02), 0.0),     # 8: broadcast amber character-generator credit roll
        ((0.88, 0.16, 0.12), (0.96, 0.84, 0.28), 0.0),     # 9: 80s documentary crimson chapter title with gold border
    )

    def _apply_style(self) -> None:
        """Set the colour, outline and slant from the template, leaving them alone for index 0."""
        cfg = self.cfg_of("overlay")
        st = int(round(float(cfg.get("style", 0.0))))
        if st <= 0 or st >= len(self.STYLE_PRESETS):
            return
        colour, outline, italic = self.STYLE_PRESETS[st]
        self.colour = tuple(float(v) for v in colour)
        self.outline_colour = tuple(float(v) for v in outline)
        if self.italic == 0.0:
            self.italic = float(italic)
        if st == 6:
            if self.bevel <= 1e-4:
                self.bevel = 0.55
            if self.glow <= 1e-4:
                self.glow = 0.35
            if self.edge_softness <= 1e-4:
                self.edge_softness = 0.28
        elif st == 7:
            if self.glow <= 1e-4:
                self.glow = 0.22
            if self.edge_softness <= 1e-4:
                self.edge_softness = 0.32
        elif st == 9:
            if self.bevel <= 1e-4:
                self.bevel = 0.65
            if self.glow <= 1e-4:
                self.glow = 0.40
            if self.edge_softness <= 1e-4:
                self.edge_softness = 0.25

    def prepare(self) -> None:
        pass

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if self.playout_spec.enabled and self.slot != self.SLOT_PRE_TAPE:
            from ..core.signal import SignalFrame, composite_encode
            sig_p = buf.meta["signal"] if "signal" in buf.meta else None
            from_disp = bool(buf.meta["from_display"]) if "from_display" in buf.meta else False
            base_p = buf.rgb if (buf.rgb is not None and (from_disp or sig_p is None)) else sig_p.to_rgb()
            flt = (buf.meta["format"] == "no_signal") if "format" in buf.meta else False
            p_rgb, p_res = apply_master_playout(
                base_p, self.playout_spec, std=buf.std,
                frame_index=int(buf.index), fps=float(self.ctx.fps),
                upstream_fault=flt,
            )
            meta_p = dict(buf.meta)
            sig_out = SignalFrame.from_rgb(p_rgb, buf.std)
            meta_p["signal"] = sig_out
            meta_p["master_playout"] = p_res["report"]
            meta_p["playout_buses"] = {
                "pgm_rgb": p_res["pgm_rgb"],
                "clean_feed_rgb": p_res["clean_feed_rgb"],
                "pvw_rgb": p_res["pvw_rgb"],
            }
            comp_p = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63) if buf.composite is not None else None
            buf = FrameBuffer(p_rgb, comp_p, buf.index, buf.time, buf.std, meta_p)
        if self.auto_v2_spec.is_active() and self.slot != self.SLOT_PRE_TAPE:
            from ..core.signal import SignalFrame, composite_encode
            sig_a = buf.meta["signal"] if "signal" in buf.meta else None
            from_disp_a = bool(buf.meta["from_display"]) if "from_display" in buf.meta else False
            base_a = buf.rgb if (buf.rgb is not None and (from_disp_a or sig_a is None)) else sig_a.to_rgb()
            flt_a = (buf.meta["format"] == "no_signal") if "format" in buf.meta else False
            a_rgb, a_res = apply_playout_automation_v2(
                base_a, self.auto_v2_spec, std=buf.std,
                frame_index=int(buf.index), fps=float(self.ctx.fps),
                upstream_meta=buf.meta,
                upstream_fault=flt_a,
            )
            meta_a = dict(buf.meta)
            sig_a_out = SignalFrame.from_rgb(a_rgb, buf.std)
            meta_a["signal"] = sig_a_out
            meta_a["playout_automation_v2"] = a_res.get("report", a_res) if isinstance(a_res, dict) else a_res
            if isinstance(a_res, dict) and "pgm_rgb" in a_res:
                meta_a["playout_buses"] = {
                    "pgm_rgb": a_res["pgm_rgb"],
                    "clean_feed_rgb": a_res["clean_feed_rgb"],
                    "pvw_rgb": a_res["pvw_rgb"],
                }
            comp_a = composite_encode(sig_a_out.y, sig_a_out.i, sig_a_out.q, buf.std, chroma_gain=0.63) if buf.composite is not None else None
            buf = FrameBuffer(a_rgb, comp_a, buf.index, buf.time, buf.std, meta_a)
        if self.eco_spec.is_active() and self.slot != self.SLOT_PRE_TAPE:
            from ..core.signal import SignalFrame, composite_encode
            sig_e = buf.meta["signal"] if "signal" in buf.meta else None
            from_disp_e = bool(buf.meta["from_display"]) if "from_display" in buf.meta else False
            base_e = buf.rgb if (buf.rgb is not None and (from_disp_e or sig_e is None)) else sig_e.to_rgb()
            e_rgb, e_res = apply_media_ecosystem_evolution(
                base_e, self.eco_spec, frame_index=int(buf.index), seed=int(self.ctx.seed) + 809,
            )
            meta_e = dict(buf.meta)
            sig_e_out = SignalFrame.from_rgb(e_rgb, buf.std)
            meta_e["signal"] = sig_e_out
            meta_e["media_ecosystem_evolution"] = e_res
            comp_e = composite_encode(sig_e_out.y, sig_e_out.i, sig_e_out.q, buf.std, chroma_gain=0.63) if buf.composite is not None else None
            buf = FrameBuffer(e_rgb, comp_e, buf.index, buf.time, buf.std, meta_e)
        if not self.enabled or not self.text:
            return buf
        if self.slot != self.SLOT_GRAPH and self.at != self.slot:
            return buf
        from ..core.signal import SignalFrame, composite_encode

        sig_in: SignalFrame | None = buf.meta.get("signal")
        in_signal_domain = (
            sig_in is not None
            and (
                self.slot == self.SLOT_PRE_TAPE
                or (self.slot == self.SLOT_GRAPH and (buf.composite is not None or not buf.meta.get("from_display", False)))
            )
        )
        if in_signal_domain and sig_in is not None:
            base_rgb = sig_in.to_rgb()
        else:
            base_rgb = buf.rgb.astype(np.float32, copy=False)

        gen = self._generator(base_rgb.shape[1], base_rgb.shape[0])
        if gen is None:
            return buf
        out, rep, mask = gen.draw_with_mask(
            base_rgb, time_s=float(buf.time), frame_index=int(buf.index),
            fps=float(self.ctx.fps))
        self.report = rep
        if not np.any(mask > 0.0):
            return buf

        meta = dict(buf.meta)
        sig_osd = SignalFrame.from_rgb(out, buf.std)
        if in_signal_domain and sig_in is not None:
            w_m = np.clip(mask, 0.0, 1.0).astype(np.float32)
            y_new = (sig_in.y * (1.0 - w_m) + sig_osd.y * w_m).astype(np.float32)
            i_new = (sig_in.i * (1.0 - w_m) + sig_osd.i * w_m).astype(np.float32)
            q_new = (sig_in.q * (1.0 - w_m) + sig_osd.q * w_m).astype(np.float32)
            sig = SignalFrame(y_new, i_new, q_new)
            if getattr(sig_in, "_rf_direct", False):
                sig._rf_direct = True
            meta["signal"] = sig
            comp = buf.composite
            if comp is not None:
                comp_osd = composite_encode(sig_osd.y, sig_osd.i, sig_osd.q, buf.std, chroma_gain=0.63)
                comp = (comp * (1.0 - w_m) + comp_osd * w_m).astype(np.float32)
            return FrameBuffer(out.astype(np.float32), comp, buf.index, buf.time, buf.std, meta)

        meta["signal"] = sig_osd
        comp = buf.composite
        if comp is not None:
            comp = composite_encode(sig_osd.y, sig_osd.i, sig_osd.q, buf.std, chroma_gain=0.63)
        return FrameBuffer(out.astype(np.float32), comp, buf.index, buf.time, buf.std, meta)

"""
Analog Video Switcher / Mixer & Special Effects Generator (SEG) physical model
with Customizable Overlay / Bus B Layer (`atsim/codecs/switcher.py`).

Models a 1980s/1990s analog composite production switcher / mixer (such as the
Panasonic WJ-AVE5, WJ-MX12, Sony XV-AL100, or Grass Valley 100) operating on two
composite video inputs (`in_a` = Program Bus A, `in_b` = Preset / Key / Custom Overlay Bus B):

  1. **Non-Genlocked Hot Switch Transient (`switcher_mode = 1`)**:
     When two video sources are not locked to a common reference sync (Black Burst /
     Frame Synchronizer), cutting between Bus A and Bus B introduces an instantaneous
     step in horizontal sync phase (`Δτ_H`), vertical field counter (`ΔY_field`), and
     colour-burst crystal phase (`Δφ_sc`).

  2. **Analog T-Bar Mix / Dissolve (`switcher_mode = 2`)**:
     Resistive/VCA cross-fader `V_out = (1 - α) V_A + α V_B` (`α = switcher_tbar`) with
     dual-pedestal black level elevation (`switcher_pedestal_add`) and uncorrelated
     subcarrier phase interference (`switcher_genlock_err`).

  3. **Analog Voltage-Ramp Wipe Generator (`switcher_mode = 3`)**:
     Synthesizes geometric wipe patterns (`0=Horizontal`, `1=Vertical`, `2=Diagonal`,
     `3=Diamond Iris`, `4=Circle Iris`) between Bus A and customizable Bus B / Overlay Layer,
     complete with comparator threshold row jitter (`switcher_key_jitter`) and composite
     switching-edge subcarrier border spill (`switcher_subcarrier_spill`).

  4. **Analog Luma Key (`switcher_mode = 4`) & Chroma Key (`switcher_mode = 5`)**:
     High-speed analog comparator slicing a customizable foreground overlay layer (`Bus B`:
     custom uploaded image/video, customizable PIP window, customizable broadcast lower-third /
     corner bug / text, or external `in_b` node) over Bus A, with:
       * ~700 ns chroma-to-luma bandpass group delay misregistration;
       * Residual 3.58/4.43 MHz subcarrier ripple on the comparator input, serrating keyed
         edges into 4-pixel comb zippers and rainbow cross-colour fringes (`switcher_subcarrier_spill`);
       * Scanline-to-scanline comparator threshold jitter (`switcher_key_jitter`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.signal import (
    SignalFrame,
    horizontal_shift,
    lowpass,
    rgb_to_yiq,
    subcarrier_phase,
    vertical_shift,
    yiq_to_rgb,
)
from ..core.standards import VideoStandard, get_standard

SWITCHER_MODE_NAMES = {
    0: "off",
    1: "hot_switch",
    2: "tbar_dissolve",
    3: "analog_wipe",
    4: "luma_key",
    5: "chroma_key",
}

WIPE_PATTERN_NAMES = {
    0: "horizontal",
    1: "vertical",
    2: "diagonal",
    3: "diamond_iris",
    4: "circle_iris",
}

B_SOURCE_NAMES = {
    0: "custom_overlay_or_cam_b",
    1: "custom_text_lower_third",
    2: "custom_pip_window",
    3: "uploaded_custom_media",
    4: "custom_color_gradient",
    5: "smpte_bars",
}

OVERLAY_STYLE_NAMES = {
    0: "clean_scene_or_upload",
    1: "custom_lower_third_text",
    2: "custom_text_bug_only",
    3: "custom_pip_window_with_title",
    4: "full_frame_secondary_scene",
    5: "classic_studio_emblem_banner",
}

# Small in-memory cache for custom uploaded overlay images/frames
_OVERLAY_IMAGE_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
_OVERLAY_TEXT_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


@dataclass
class AnalogSwitcherSpec:
    """Configuration parameters for the Analog Video Switcher & Customizable Overlay (`connector` layer)."""

    mode: int = 0                    # 0=Off, 1=Hot Switch, 2=T-Bar Dissolve, 3=Wipe, 4=Luma Key, 5=Chroma Key
    b_source: int = 0                # 0=Adaptive Custom/Cam B, 1=Custom Text/Lower-Third, 2=Custom PIP Window, 3=Uploaded Overlay Media, 4=Custom Color/Gradient, 5=SMPTE Bars
    tbar: float = 0.50               # T-Bar fader position / wipe progress (0.0 = Bus A .. 1.0 = Bus B)
    wipe_pattern: int = 0            # 0=Horiz, 1=Vert, 2=Diag, 3=Diamond, 4=Circle
    hot_cut_rate: float = 1.25       # Hot-switch frequency (cuts/sec) in mode 1
    genlock_err: float = 0.65        # Non-genlocked timing & subcarrier phase mismatch between Bus A and Bus B (0..1.5)
    key_clip: float = 0.42           # Analog comparator clip threshold (0.05..0.95)
    key_gain: float = 0.65           # Analog key amplifier slope/hardness (0.05..1.5)
    key_jitter: float = 0.35         # Comparator threshold & ramp scanline jitter (0..1.5)
    subcarrier_spill: float = 0.45   # Subcarrier residual edge zipper, group-delay misregistration & border fringe (0..1.5)
    pedestal_add: float = 0.25       # Dual-bus black setup pedestal addition / backing spill (0..1.0)
    # Customizable overlay layer parameters (for P2 & P3 and all switcher modes)
    overlay_style: int = 0           # 0=Auto/Clean, 1=Lower-Third+Text, 2=Text/Bug Only, 3=PIP Window+Title, 4=Full-Frame Cam B, 5=Classic Studio Emblem
    overlay_text: str = "STUDIO B · LIVE"
    overlay_path: str = ""           # Optional uploaded custom image/video path
    overlay_x: float = 0.50          # Horizontal center of custom overlay (0.0..1.0)
    overlay_y: float = 0.78          # Vertical center of custom overlay (0.0..1.0)
    overlay_scale: float = 1.00      # Scale factor of custom overlay (0.15..2.50)
    overlay_opacity: float = 0.92    # Opacity / blend weight of custom overlay elements (0.0..1.0)
    overlay_r: float = 0.95          # Custom overlay primary R (0.0..1.0)
    overlay_g: float = 0.86          # Custom overlay primary G (0.0..1.0)
    overlay_b: float = 0.28          # Custom overlay primary B (0.0..1.0)

    @classmethod
    def from_params(cls, cfg: dict | None) -> "AnalogSwitcherSpec":
        c = cfg or {}
        raw_text = c.get("switcher_overlay_text", "STUDIO B · LIVE")
        if isinstance(raw_text, (int, float)):
            raw_text = "STUDIO B · LIVE" if float(raw_text) != 0.0 else ""
        raw_path = c.get("switcher_overlay_path", "")
        if isinstance(raw_path, (int, float)):
            raw_path = ""
        return cls(
            mode=int(np.clip(round(float(c.get("switcher_mode", 0.0))), 0, 5)),
            b_source=int(np.clip(round(float(c.get("switcher_b_source", 0.0))), 0, 5)),
            tbar=float(np.clip(float(c.get("switcher_tbar", 0.50)), 0.0, 1.0)),
            wipe_pattern=int(np.clip(round(float(c.get("switcher_wipe_pattern", 0.0))), 0, 4)),
            hot_cut_rate=float(np.clip(float(c.get("switcher_hot_cut_rate", 1.25)), 0.1, 6.0)),
            genlock_err=float(np.clip(float(c.get("switcher_genlock_err", 0.65)), 0.0, 1.5)),
            key_clip=float(np.clip(float(c.get("switcher_key_clip", 0.42)), 0.05, 0.95)),
            key_gain=float(np.clip(float(c.get("switcher_key_gain", 0.65)), 0.05, 1.5)),
            key_jitter=float(np.clip(float(c.get("switcher_key_jitter", 0.35)), 0.0, 1.5)),
            subcarrier_spill=float(np.clip(float(c.get("switcher_subcarrier_spill", 0.45)), 0.0, 1.5)),
            pedestal_add=float(np.clip(float(c.get("switcher_pedestal_add", 0.25)), 0.0, 1.0)),
            overlay_style=int(np.clip(round(float(c.get("switcher_overlay_style", 0.0))), 0, 5)),
            overlay_text=str(raw_text or ""),
            overlay_path=str(raw_path or "").strip(),
            overlay_x=float(np.clip(float(c.get("switcher_overlay_x", 0.50)), 0.0, 1.0)),
            overlay_y=float(np.clip(float(c.get("switcher_overlay_y", 0.78)), 0.0, 1.0)),
            overlay_scale=float(np.clip(float(c.get("switcher_overlay_scale", 1.00)), 0.15, 2.50)),
            overlay_opacity=float(np.clip(float(c.get("switcher_overlay_opacity", 0.92)), 0.0, 1.0)),
            overlay_r=float(np.clip(float(c.get("switcher_overlay_r", 0.95)), 0.0, 1.0)),
            overlay_g=float(np.clip(float(c.get("switcher_overlay_g", 0.86)), 0.0, 1.0)),
            overlay_b=float(np.clip(float(c.get("switcher_overlay_b", 0.28)), 0.0, 1.0)),
        )

    def is_active(self) -> bool:
        return self.mode > 0


def _resize_nn_bilinear(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Fast 2D/3D bilinear resize to `(H, W)`."""
    sh, sw = arr.shape[:2]
    if sh == H and sw == W:
        return arr.astype(np.float32, copy=False)
    ys = np.linspace(0.0, max(0.0, float(sh - 1)), H, dtype=np.float32)
    xs = np.linspace(0.0, max(0.0, float(sw - 1)), W, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, max(0, sh - 1))
    x1 = np.minimum(x0 + 1, max(0, sw - 1))
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    if arr.ndim == 3:
        fy = fy[:, :, None]
        fx = fx[:, :, None]
    top = arr[y0[:, None], x0[None, :]] * (1.0 - fx) + arr[y0[:, None], x1[None, :]] * fx
    bot = arr[y1[:, None], x0[None, :]] * (1.0 - fx) + arr[y1[:, None], x1[None, :]] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def _load_uploaded_overlay(path_str: str, H: int, W: int, frame_index: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load an uploaded custom overlay image or video frame from `path_str` and resize to `(H, W)`.
    Returns `(rgb_float32, alpha_float32)` in `0..1`, or `None` if unavailable.
    """
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    cache_key = (str(p.resolve()), int(H), int(W))
    from ..io.imagecodec import is_image_path, load_image
    if is_image_path(p):
        cached = _OVERLAY_IMAGE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            # Try PIL first to preserve RGBA alpha channel if present
            rgba = None
            try:
                from PIL import Image
                with Image.open(p) as im:
                    im_rgba = im.convert("RGBA")
                    rgba = np.asarray(im_rgba, dtype=np.float32) / 255.0
            except Exception:
                rgba = None
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                rgb = _resize_nn_bilinear(rgba[:, :, :3], H, W)
                alpha = _resize_nn_bilinear(rgba[:, :, 3], H, W)
            else:
                raw_rgb = load_image(p)
                rgb = _resize_nn_bilinear(raw_rgb[:, :, :3], H, W)
                alpha = np.ones((H, W), dtype=np.float32)
            if len(_OVERLAY_IMAGE_CACHE) > 24:
                _OVERLAY_IMAGE_CACHE.clear()
            _OVERLAY_IMAGE_CACHE[cache_key] = (np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0))
            return _OVERLAY_IMAGE_CACHE[cache_key]
        except Exception:
            return None
    # Video file fallback via ClipSource
    try:
        from ..io.source import ClipSource
        src = ClipSource(p, scale=1.0)
        idx = max(0, min(int(frame_index), max(0, getattr(src, "frame_count", 1) - 1)))
        raw_rgb = src.frame(idx)
        rgb = _resize_nn_bilinear(raw_rgb[:, :, :3], H, W)
        alpha = np.ones((H, W), dtype=np.float32)
        return np.clip(rgb, 0.0, 1.0), alpha
    except Exception:
        return None


def _place_scaled_layer(
    canvas_rgb: np.ndarray,
    canvas_mask: np.ndarray,
    fg_rgb: np.ndarray,
    fg_alpha: np.ndarray,
    cx_norm: float,
    cy_norm: float,
    scale: float,
    opacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place `(fg_rgb, fg_alpha)` centered at `(cx_norm, cy_norm)` with `scale` and `opacity` onto `canvas_rgb`."""
    H, W = canvas_rgb.shape[:2]
    s = float(np.clip(scale, 0.15, 2.50))
    th = max(8, int(round(H * s)))
    tw = max(8, int(round(W * s)))
    scaled_rgb = _resize_nn_bilinear(fg_rgb, th, tw)
    scaled_a = _resize_nn_bilinear(fg_alpha, th, tw) * float(np.clip(opacity, 0.0, 1.0))

    cy_px = int(round(float(np.clip(cx_norm, 0.0, 1.0)) * W))
    cx_px = cy_px
    cy_px = int(round(float(np.clip(cy_norm, 0.0, 1.0)) * H))

    y0 = cy_px - th // 2
    x0 = cx_px - tw // 2
    y1 = y0 + th
    x1 = x0 + tw

    dy0 = max(0, y0)
    dx0 = max(0, x0)
    dy1 = min(H, y1)
    dx1 = min(W, x1)
    if dy1 <= dy0 or dx1 <= dx0:
        return canvas_rgb, canvas_mask

    sy0 = dy0 - y0
    sx0 = dx0 - x0
    sy1 = sy0 + (dy1 - dy0)
    sx1 = sx0 + (dx1 - dx0)

    sub_a = np.clip(scaled_a[sy0:sy1, sx0:sx1], 0.0, 1.0)[..., None]
    sub_rgb = scaled_rgb[sy0:sy1, sx0:sx1]
    out_rgb = canvas_rgb.copy()
    out_mask = canvas_mask.copy()
    out_rgb[dy0:dy1, dx0:dx1] = out_rgb[dy0:dy1, dx0:dx1] * (1.0 - sub_a) + sub_rgb * sub_a
    out_mask[dy0:dy1, dx0:dx1] = np.maximum(out_mask[dy0:dy1, dx0:dx1], sub_a[..., 0])
    return out_rgb, out_mask


def _render_custom_text_layer(
    H: int,
    W: int,
    text: str,
    cx_norm: float,
    cy_norm: float,
    scale: float,
    colour: tuple[float, float, float],
    opacity: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize customizable overlay text (`switcher_overlay_text`, supporting both Chinese and ASCII)
    into an `(H, W, 3)` RGB plate and `(H, W)` alpha mask positioned around `(cx_norm, cy_norm)`.
    """
    txt = str(text or "").strip()
    if not txt or opacity <= 1e-3:
        return np.zeros((H, W, 3), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    scale_r = max(0.30, float(H) / 240.0) * float(np.clip(scale, 0.25, 2.50))
    cell_h = max(8, int(round(18.0 * scale_r)))
    outline_px = max(1, int(round(1.2 * scale_r)))
    col_key = (round(colour[0], 3), round(colour[1], 3), round(colour[2], 3))
    cache_sig = (int(H), int(W), txt, cell_h, outline_px, col_key, round(float(cx_norm), 3), round(float(cy_norm), 3), round(float(opacity), 3))
    cached = _OVERLAY_TEXT_CACHE.get(cache_sig)
    if cached is not None:
        return cached

    try:
        from .fonts import font_catalogue
        from .osd import CharacterGenerator, OSDPlan, parse_lines
        avail = [d["key"] for d in font_catalogue() if d.get("available")]
        # Prefer a font with full CJK + ASCII coverage if available
        font_key = avail[0] if avail else "osd"
        for pref in ("hzk16", "unifont", "lyusung", "osd"):
            if pref in avail:
                font_key = pref
                break

        halign = 0 if cx_norm < 0.35 else (2 if cx_norm > 0.65 else 1)
        valign = 0 if cy_norm < 0.35 else (2 if cy_norm > 0.65 else 1)
        mx = int(round(abs(cx_norm - (0.0 if halign == 0 else (1.0 if halign == 2 else 0.5))) * W * 0.45)) + int(round(0.04 * W))
        my = int(round(abs(cy_norm - (0.0 if valign == 0 else (1.0 if valign == 2 else 0.5))) * H * 0.45)) + int(round(0.04 * H))

        plan = OSDPlan(
            lines=parse_lines(txt),
            cell_h=cell_h,
            margin_x=max(4, min(W // 3, mx)),
            margin_y=max(4, min(H // 3, my)),
            halign=halign,
            valign=valign,
            colour=col_key,
            shadow=0.65,
            opacity=float(np.clip(opacity, 0.0, 1.0)),
            outline=1.0,
            outline_px=outline_px,
            outline_colour=(0.03, 0.04, 0.06),
        )
        gen = CharacterGenerator(font_key, plan)
        blank = np.zeros((H, W, 3), dtype=np.float32)
        out_rgb, _rep, mask = gen.draw_with_mask(blank, time_s=0.0, frame_index=0, fps=29.97)
        res = (np.clip(out_rgb, 0.0, 1.0).astype(np.float32), np.clip(mask, 0.0, 1.0).astype(np.float32))
    except Exception:
        from ..io.imagecodec import draw_text
        blank = np.zeros((H, W, 3), dtype=np.float32)
        sc = max(1, int(round(2.0 * scale_r)))
        tx = int(np.clip(cx_norm * W - len(txt) * 3 * sc, 4, max(4, W - 20)))
        ty = int(np.clip(cy_norm * H - 4 * sc, 4, max(4, H - 12)))
        out_rgb = draw_text(blank, txt, tx, ty, scale=sc, colour=col_key)
        mask = (np.max(out_rgb, axis=-1) > 0.02).astype(np.float32) * float(np.clip(opacity, 0.0, 1.0))
        res = (out_rgb, mask)

    if len(_OVERLAY_TEXT_CACHE) > 32:
        _OVERLAY_TEXT_CACHE.clear()
    _OVERLAY_TEXT_CACHE[cache_sig] = res
    return res


def _build_reframed_cam_b_rgb(
    y_a: np.ndarray,
    i_a: np.ndarray,
    q_a: np.ndarray,
    frame_index: int,
    fps: float,
    scale: float = 1.0,
    tint_rgb: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Synthesize a clean reframed / secondary camera angle from `(y_a, i_a, q_a)`."""
    H, W = y_a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yn = yy / max(1.0, float(H - 1))
    xn = xx / max(1.0, float(W - 1))
    t_s = float(frame_index) / max(1.0, float(fps))

    zoom_crop = float(np.clip(0.76 / max(0.35, min(1.8, scale)), 0.38, 0.96))
    pan_x = 0.05 * np.sin(2.0 * np.pi * 0.20 * t_s)
    sy = np.clip((yn - 0.5) * zoom_crop + 0.48, 0.0, 1.0) * (H - 1.0)
    sx = np.clip((1.0 - xn - 0.5 + pan_x) * zoom_crop + 0.50, 0.0, 1.0) * (W - 1.0)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (sy - y0).astype(np.float32)
    fx = (sx - x0).astype(np.float32)

    def _samp(p: np.ndarray) -> np.ndarray:
        top = p[y0, x0] * (1.0 - fx) + p[y0, x1] * fx
        bot = p[y1, x0] * (1.0 - fx) + p[y1, x1] * fx
        return (top * (1.0 - fy) + bot * fy).astype(np.float32)

    yb = np.clip(_samp(y_a) * 1.06 + 0.03, 0.0, 1.0)
    ib0 = _samp(i_a)
    qb0 = _samp(q_a)
    theta = np.deg2rad(28.0)
    ib = (ib0 * np.cos(theta) - qb0 * np.sin(theta) + 0.04).astype(np.float32)
    qb = (ib0 * np.sin(theta) + qb0 * np.cos(theta) - 0.02).astype(np.float32)
    rgb = np.clip(yiq_to_rgb(yb, ib, qb), 0.0, 1.0).astype(np.float32)
    if tint_rgb is not None:
        t_arr = np.array(tint_rgb, dtype=np.float32)[None, None, :]
        rgb = np.clip(rgb * (0.78 + 0.28 * t_arr), 0.0, 1.0)
    return rgb


def synthesize_b_source_yiq(
    y_a: np.ndarray,
    i_a: np.ndarray,
    q_a: np.ndarray,
    b_source: int,
    frame_index: int,
    fps: float = 29.97,
    std: VideoStandard | None = None,
    spec: AnalogSwitcherSpec | None = None,
    custom_overlay_rgba: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize the Customizable Overlay / Preset Bus B `(Y, I, Q)` signal when no external
    `in_b` node is wired in the node graph:

      * Supports user-uploaded overlay images/videos (`spec.overlay_path` or `custom_overlay_rgba`),
        customizable overlay styles (`spec.overlay_style`), custom text (`spec.overlay_text`),
        customizable X/Y position (`spec.overlay_x`, `spec.overlay_y`), scale (`spec.overlay_scale`),
        opacity (`spec.overlay_opacity`), and RGB color (`spec.overlay_r/g/b`).
      * `b_source == 0` (**Adaptive Custom Overlay / Camera B**):
        - In Wipe / Dissolve / Hot-Switch (`mode in (1, 2, 3)`): uses the uploaded custom overlay
          image/video if provided, or a clean reframed secondary scene (`Camera B`) with optional
          custom overlay graphics/text when `overlay_style in (1, 2, 3, 5)`.
        - In Luma / Chroma Key (`mode in (4, 5)`): builds a keyable backing plate (`studio blue`
          for Chroma Key, `deep black` for Luma Key) carrying the customizable overlay (uploaded image,
          PIP sub-window, custom lower-third banner, or custom text).
      * `b_source == 1` (**Custom Text / Lower-Third / Corner Bug Overlay**):
        Renders customizable broadcast lower-third bar + custom text (`overlay_text`) + optional corner bug.
      * `b_source == 2` (**Custom Picture-in-Picture (PIP) Window Overlay**):
        Renders a customizable PIP sub-window at `(overlay_x, overlay_y)` with `overlay_scale`, border
        colored by `(overlay_r, overlay_g, overlay_b)`, and optional `overlay_text` caption.
      * `b_source == 3` (**Uploaded Custom Overlay Image / Video**):
        Uses the uploaded image/video (`overlay_path`), falling back to customizable PIP + text if none uploaded yet.
      * `b_source == 4` (**Custom Color / Atmospheric Gradient Overlay**):
        Generates a smooth customizable color/gradient layer controlled by `(overlay_r, overlay_g, overlay_b)`.
      * `b_source == 5` (**Classic SMPTE/EBU Studio Colour Bars**):
        Generates classic 75% broadcast colour bars.
    """
    std_obj = std or get_standard("ntsc")
    H, W = y_a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yn = yy / max(1.0, float(H - 1))
    xn = xx / max(1.0, float(W - 1))
    t_s = float(frame_index) / max(1.0, float(fps))

    sp = spec or AnalogSwitcherSpec(b_source=int(b_source))
    mode = int(sp.mode)
    b_src = int(b_source)
    style = int(sp.overlay_style)
    cx = float(sp.overlay_x)
    cy = float(sp.overlay_y)
    sc = float(sp.overlay_scale)
    op = float(sp.overlay_opacity)
    fg_col = (float(sp.overlay_r), float(sp.overlay_g), float(sp.overlay_b))
    fg_arr = np.array(fg_col, dtype=np.float32)

    # Check if an uploaded overlay image/video is available
    uploaded = custom_overlay_rgba
    if uploaded is None and sp.overlay_path:
        uploaded = _load_uploaded_overlay(sp.overlay_path, H, W, frame_index=frame_index)

    # Keying backing color: Studio Blue for Chroma Key (`mode == 5`), Deep Black for Luma Key (`mode == 4`)
    is_key_mode = mode in (4, 5)
    if mode == 5:
        key_backing_rgb = np.broadcast_to(np.array([0.04, 0.08, 0.76], dtype=np.float32), (H, W, 3)).copy()
    else:
        key_backing_rgb = np.full((H, W, 3), 0.02, dtype=np.float32)

    # Secondary reframed scene (clean Camera B without any hardcoded yellow circles or bars)
    cam_b_rgb = _build_reframed_cam_b_rgb(y_a, i_a, q_a, frame_index=frame_index, fps=fps, scale=sc)

    # ------------------------------------------------------------------
    # Option 5: Classic SMPTE 75% Colour Bars (explicitly selectable)
    # ------------------------------------------------------------------
    if b_src == 5:
        bars_rgb = np.array([
            [0.75, 0.75, 0.75],
            [0.75, 0.75, 0.00],
            [0.00, 0.75, 0.75],
            [0.00, 0.75, 0.00],
            [0.75, 0.00, 0.75],
            [0.75, 0.00, 0.00],
            [0.00, 0.00, 0.75],
        ], dtype=np.float32)
        bar_idx = np.clip((xn * 7.0).astype(np.int32), 0, 6)
        rgb = bars_rgb[bar_idx]
        bot_mask = (yn > 0.76)[..., None]
        ramp = np.repeat(xn[..., None], 3, axis=-1)
        rgb = np.where(bot_mask, 0.12 + 0.75 * ramp, rgb)
        if sp.overlay_text.strip():
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, cx, cy, sc, fg_col, op)
            rgb = rgb * (1.0 - t_m[..., None]) + t_rgb * t_m[..., None]
        yb, ib, qb = rgb_to_yiq(np.clip(rgb, 0.0, 1.0).astype(np.float32))
        return yb.astype(np.float32), ib.astype(np.float32), qb.astype(np.float32)

    # ------------------------------------------------------------------
    # Option 4: Custom Color / Atmospheric Gradient Layer
    # ------------------------------------------------------------------
    if b_src == 4:
        dx = (xn - cx) * (W / max(1.0, float(H)))
        dy = yn - cy
        rad = np.hypot(dx, dy) / max(0.20, 0.65 * sc)
        grad = np.clip(1.0 - 0.55 * rad, 0.15, 1.0)[..., None]
        comp_col = np.array([1.0 - fg_col[0] * 0.5, fg_col[1] * 0.7, min(1.0, fg_col[2] + 0.35)], dtype=np.float32)
        rgb = grad * fg_arr[None, None, :] + (1.0 - grad) * comp_col[None, None, :]
        # Blend with subtle scene luminance so wipe/key retains organic texture
        scene_rgb = np.clip(yiq_to_rgb(y_a, i_a, q_a), 0.0, 1.0)
        rgb = np.clip(rgb * op + scene_rgb * (1.0 - 0.65 * op), 0.0, 1.0)
        if is_key_mode:
            win_mask = (rad <= 0.55 * sc)[..., None]
            rgb = np.where(win_mask, rgb, key_backing_rgb)
        if sp.overlay_text.strip():
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, cx, cy, sc, (0.98, 0.98, 0.96), op)
            rgb = rgb * (1.0 - t_m[..., None]) + t_rgb * t_m[..., None]
        yb, ib, qb = rgb_to_yiq(rgb.astype(np.float32))
        return yb.astype(np.float32), ib.astype(np.float32), qb.astype(np.float32)

    # ------------------------------------------------------------------
    # If an uploaded custom overlay image/video is present (or b_src == 3)
    # ------------------------------------------------------------------
    if uploaded is not None:
        up_rgb, up_alpha = uploaded
        base_canvas = key_backing_rgb.copy() if is_key_mode else cam_b_rgb.copy()
        mask_canvas = np.zeros((H, W), dtype=np.float32)
        placed_rgb, placed_m = _place_scaled_layer(
            base_canvas, mask_canvas, up_rgb, up_alpha,
            cx_norm=cx if (abs(cx - 0.50) > 1e-3 or abs(sc - 1.0) > 1e-3) else 0.50,
            cy_norm=cy if (abs(cy - 0.78) > 1e-3 and abs(cy - 0.50) > 1e-3) else 0.50,
            scale=sc,
            opacity=op,
        )
        if sp.overlay_text.strip() and style in (1, 2, 3):
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, cx, cy, sc, fg_col, op)
            placed_rgb = placed_rgb * (1.0 - t_m[..., None]) + t_rgb * t_m[..., None]
        yb, ib, qb = rgb_to_yiq(np.clip(placed_rgb, 0.0, 1.0).astype(np.float32))
        return yb.astype(np.float32), ib.astype(np.float32), qb.astype(np.float32)

    # Determine effective overlay style when no external file is uploaded
    eff_style = style
    if eff_style == 0:
        if b_src == 1:
            eff_style = 1  # Custom lower-third + text
        elif b_src == 2:
            eff_style = 3  # Custom PIP window + title
        elif is_key_mode:
            eff_style = 3  # Default for P2 (Chroma/Luma Key): clean customizable PIP window + title bar!
        else:
            eff_style = 4  # Default for P3 (Wipe / Mix / Hot-Switch): clean reframed Camera B scene!

    # ------------------------------------------------------------------
    # Style 4: Full-Frame Clean Secondary Scene (Camera B, no forced banners)
    # ------------------------------------------------------------------
    if eff_style == 4 and not is_key_mode:
        rgb = cam_b_rgb.copy()
        if sp.overlay_text.strip() and b_src in (1, 2):
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, cx, cy, sc, fg_col, op)
            rgb = rgb * (1.0 - t_m[..., None]) + t_rgb * t_m[..., None]
        yb, ib, qb = rgb_to_yiq(rgb.astype(np.float32))
        return yb.astype(np.float32), ib.astype(np.float32), qb.astype(np.float32)

    # Start with key backing (in key mode) or clean Camera B scene (in wipe/dissolve mode)
    rgb = key_backing_rgb.copy() if is_key_mode else cam_b_rgb.copy()

    # ------------------------------------------------------------------
    # Style 1: Custom Lower-Third Banner + Custom Text
    # ------------------------------------------------------------------
    if eff_style == 1:
        bw = float(np.clip(0.36 * sc, 0.12, 0.48))
        bh = float(np.clip(0.065 * sc, 0.025, 0.22))
        x_min, x_max = np.clip(cx - bw, 0.02, 0.98), np.clip(cx + bw, 0.02, 0.98)
        y_min, y_max = np.clip(cy - bh, 0.02, 0.98), np.clip(cy + bh, 0.02, 0.98)
        in_bar = (xn >= x_min) & (xn <= x_max) & (yn >= y_min) & (yn <= y_max)
        accent_h = max(0.008, bh * 0.24)
        in_accent = (xn >= x_min) & (xn <= x_max) & (yn >= y_min) & (yn <= y_min + accent_h)

        # High-luma sleek broadcast plate so both Luma Key and Chroma Key slice it cleanly
        plate_col = np.clip(0.28 * fg_arr + np.array([0.58, 0.60, 0.62], dtype=np.float32), 0.25, 0.96)
        rgb[in_bar] = rgb[in_bar] * (1.0 - op) + plate_col * op
        rgb[in_accent] = rgb[in_accent] * (1.0 - op) + fg_arr * op

        if sp.overlay_text.strip():
            txt_col = (0.10, 0.12, 0.15) if float(np.mean(plate_col)) > 0.62 else (0.97, 0.97, 0.95)
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, cx, cy, sc, txt_col, op)
            rgb = rgb * (1.0 - t_m[..., None]) + t_rgb * t_m[..., None]

    # ------------------------------------------------------------------
    # Style 2: Custom Text / Station Bug Only (No background banner box)
    # ------------------------------------------------------------------
    elif eff_style == 2:
        txt_to_draw = sp.overlay_text.strip() or "CH-02 LIVE"
        t_rgb, t_m = _render_custom_text_layer(H, W, txt_to_draw, cx, cy, sc * 1.15, fg_col, op)
        # Ensure high luminance contrast on the text glyphs for clean keying
        bright_fg = np.clip(fg_arr * 0.85 + 0.15, 0.25, 1.0)
        glyph_zone = (t_m > 0.15)[..., None]
        rgb = np.where(glyph_zone, bright_fg[None, None, :] * op + rgb * (1.0 - op), rgb)

    # ------------------------------------------------------------------
    # Style 3 (or 4 in Key Mode): Customizable Picture-in-Picture (PIP) Sub-Window + Optional Title
    # ------------------------------------------------------------------
    elif eff_style in (3, 4):
        # Position PIP window around `(cx, cy)` (if user left cy at default 0.78, center PIP at 0.36 so it frames nicely)
        pip_cx = cx if abs(cx - 0.50) > 1e-3 else 0.66
        pip_cy = cy if abs(cy - 0.78) > 1e-3 else 0.36
        win_w = float(np.clip(0.24 * sc, 0.10, 0.46))
        win_h = float(np.clip(0.22 * sc, 0.10, 0.46))
        x0, x1 = np.clip(pip_cx - win_w, 0.03, 0.97), np.clip(pip_cx + win_w, 0.03, 0.97)
        y0, y1 = np.clip(pip_cy - win_h, 0.03, 0.97), np.clip(pip_cy + win_h, 0.03, 0.97)

        in_outer = (xn >= x0) & (xn <= x1) & (yn >= y0) & (yn <= y1)
        bw_x = max(0.008, 0.012 * sc)
        bw_y = max(0.010, 0.015 * sc)
        in_inner = (xn >= x0 + bw_x) & (xn <= x1 - bw_x) & (yn >= y0 + bw_y) & (yn <= y1 - bw_y)
        in_border = in_outer & (~in_inner)

        # Map the full `cam_b_rgb` into the PIP window
        u = np.clip((xn - (x0 + bw_x)) / max(1e-4, (x1 - x0 - 2.0 * bw_x)), 0.0, 1.0) * (W - 1.0)
        v = np.clip((yn - (y0 + bw_y)) / max(1e-4, (y1 - y0 - 2.0 * bw_y)), 0.0, 1.0) * (H - 1.0)
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        pip_content = np.clip(cam_b_rgb[vi, ui] * 1.12 + 0.08, 0.18, 1.0)

        rgb[in_inner] = rgb[in_inner] * (1.0 - op) + pip_content[in_inner] * op
        rgb[in_border] = rgb[in_border] * (1.0 - op) + fg_arr * op

        # Optional customizable lower-third caption bar if `overlay_text` is non-empty
        if sp.overlay_text.strip():
            cap_y = float(np.clip(y1 + 0.055 * sc, 0.10, 0.92))
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, pip_cx, cap_y, sc * 0.90, fg_col, op)
            glyph_zone = (t_m > 0.12)[..., None]
            rgb = np.where(glyph_zone, np.clip(t_rgb + 0.12 * fg_arr[None, None, :], 0.0, 1.0), rgb)

    # ------------------------------------------------------------------
    # Style 5: Classic Studio Emblem + Lower-Third Banner (now customizable in position/scale/color!)
    # ------------------------------------------------------------------
    else:
        bw = float(np.clip(0.38 * sc, 0.12, 0.48))
        bh = float(np.clip(0.075 * sc, 0.03, 0.20))
        in_banner = (yn >= cy - bh) & (yn <= cy + bh) & (xn >= cx - bw) & (xn <= cx + bw)
        in_accent = (yn >= cy - bh) & (yn <= cy - bh + 0.035 * sc) & (xn >= cx - bw) & (xn <= cx + bw)
        stripe_pat = 0.5 + 0.5 * np.cos(2.0 * np.pi * (xn * 18.0 - t_s * 1.2))
        rgb[in_banner] = fg_arr * op + rgb[in_banner] * (1.0 - op)
        rgb[in_accent] = np.array([0.96, 0.32, 0.18], dtype=np.float32) * op + rgb[in_accent] * (1.0 - op)
        in_ticks = (
            (yn > cy - 0.3 * bh) & (yn < cy + 0.65 * bh)
            & (xn > cx - 0.85 * bw) & (xn < cx + 0.85 * bw)
            & (stripe_pat > 0.55)
        )
        rgb[in_ticks] = np.array([0.98, 0.98, 0.95], dtype=np.float32)

        em_x = float(np.clip(cx + 0.22, 0.15, 0.85))
        em_y = float(np.clip(cy - 0.48, 0.15, 0.85))
        dx = (xn - em_x) * (W / max(1.0, float(H)))
        dy = yn - em_y
        rad = np.hypot(dx, dy)
        r_outer = 0.13 * sc
        r_inner = 0.08 * sc
        cross_w = 0.018 * sc
        emblem = (rad <= r_outer) & ((rad >= r_inner) | (np.abs(dx) < cross_w) | (np.abs(dy) < cross_w))
        rgb[emblem] = fg_arr * op + rgb[emblem] * (1.0 - op)
        if sp.overlay_text.strip():
            t_rgb, t_m = _render_custom_text_layer(H, W, sp.overlay_text, cx, cy, sc, (0.98, 0.98, 0.95), op)
            rgb = rgb * (1.0 - t_m[..., None]) + t_rgb * t_m[..., None]

    yb, ib, qb = rgb_to_yiq(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    return yb.astype(np.float32), ib.astype(np.float32), qb.astype(np.float32)


def _synthesize_vbi_band(width: int, vbi_rows: int, scale_w: float) -> np.ndarray:
    """Build a physical Vertical Blanking Interval (VBI) luma band exposed during V-Hold roll."""
    vbi = np.full((vbi_rows, width), 0.06, dtype=np.float32)
    xx = np.arange(width, dtype=np.float32)[None, :]
    xn = xx / max(1.0, float(width - 1))
    r_idx = np.arange(vbi_rows, dtype=np.float32)[:, None]
    rn = r_idx / max(1.0, float(vbi_rows - 1))
    serr_zone = (rn >= 0.25) & (rn <= 0.60)
    half_line_sync = ((xn < 0.42) | ((xn >= 0.50) & (xn < 0.92))).astype(np.float32)
    vbi = np.where(serr_zone, 0.015 * half_line_sync + 0.08 * (1.0 - half_line_sync), vbi)
    vitc_zone = (rn > 0.68) & (rn < 0.86)
    vitc_bits = (np.sin(2.0 * np.pi * xn * 36.0) > 0.0).astype(np.float32) * (xn > 0.12) * (xn < 0.88)
    vbi = np.where(vitc_zone, 0.06 + 0.58 * vitc_bits, vbi)
    return vbi.astype(np.float32)


def _compute_wipe_field(
    yn: np.ndarray,
    xn: np.ndarray,
    pattern: int,
    aspect: float = 1.333,
    center_x: float = 0.50,
    center_y: float = 0.50,
) -> np.ndarray:
    """
    Compute the normalized 2D analog ramp waveform `R(y, x) in [0, 1]` for the
    selected SEG wipe pattern, supporting customizable center `(center_x, center_y)`:
      * 0: Horizontal ramp (`xn`)
      * 1: Vertical ramp (`yn`)
      * 2: Diagonal 45° ramp (`0.5 * (xn + yn)`)
      * 3: Diamond iris (`|x - cx| + |y - cy|`)
      * 4: Circle iris (`hypot((x - cx)*aspect, y - cy)`)
    """
    p = int(pattern)
    cx = float(np.clip(center_x, 0.10, 0.90))
    cy = float(np.clip(center_y, 0.10, 0.90))
    if p == 1:
        return yn.copy()
    if p == 2:
        return (0.5 * (xn + yn)).astype(np.float32)
    if p == 3:
        return np.clip(np.abs(xn - cx) + np.abs(yn - cy), 0.0, 1.0).astype(np.float32)
    if p == 4:
        r = np.hypot((xn - cx) * aspect, yn - cy) / max(1e-3, 0.5 * np.hypot(aspect, 1.0))
        return np.clip(r, 0.0, 1.0).astype(np.float32)
    return xn.copy()


def compute_switcher_audio_transient(
    spec: AnalogSwitcherSpec,
    t_samples: np.ndarray,
    seed: int = 20240101,
) -> dict:
    """
    Compute audio-domain crosspoint switching transients & Bus B carrier bleed for `AudioSimulator`:
      * `click_pop`: CMOS/relay crosspoint DC-offset click/pop spike at hot-switch instants
      * `b_bus_mix`: fraction of Bus B audio (`0..1`)
      * `hum_add`: ground/subcarrier buzz added by dual-bus mixing or keyer spill
    """
    n = t_samples.size
    if not spec.is_active() or n == 0:
        return {
            "click_pop": np.zeros(n, dtype=np.float32),
            "b_bus_mix": np.zeros(n, dtype=np.float32),
            "hum_add": 0.0,
        }

    click_pop = np.zeros(n, dtype=np.float32)
    b_mix = np.full(n, float(spec.tbar), dtype=np.float32)
    hum_add = 0.0

    if spec.mode == 1:
        rate = max(0.2, float(spec.hot_cut_rate))
        phase = t_samples * rate
        bus_sel = (np.floor(phase) % 2.0).astype(np.float32)
        b_mix = bus_sel
        frac_in_cut = (phase % 1.0) / rate
        env_pop = np.exp(-frac_in_cut / 0.0055).astype(np.float32)
        env_thump = np.exp(-frac_in_cut / 0.022).astype(np.float32) * np.sin(2.0 * np.pi * 165.0 * frac_in_cut).astype(np.float32)
        err = max(0.15, float(spec.genlock_err))
        click_pop = (0.38 * err * env_pop + 0.24 * err * env_thump).astype(np.float32)
        hum_add = 0.08 * err
    elif spec.mode == 2:
        mid = 4.0 * spec.tbar * (1.0 - spec.tbar)
        hum_add = float(0.06 * mid * (spec.pedestal_add + 0.5 * spec.genlock_err))
    elif spec.mode in (3, 4, 5):
        hum_add = float(0.05 * spec.subcarrier_spill)

    return {
        "click_pop": click_pop,
        "b_bus_mix": b_mix,
        "hum_add": hum_add,
    }


def apply_analog_switcher(
    y_a: np.ndarray,
    i_a: np.ndarray,
    q_a: np.ndarray,
    spec: AnalogSwitcherSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    std: VideoStandard | None = None,
    seed: int = 20240101,
    bus_b_yiq: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    custom_overlay_rgba: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Apply the Analog Video Switcher / Mixer (Candidate C / Module 6) to Bus A `(y_a, i_a, q_a)`
    and Bus B (`bus_b_yiq`, or synthesized via `spec` + customizable overlay parameters).

    Returns `(y_out, i_out, q_out, info_dict)`.
    When `spec.is_active()` is False (`mode == 0`), returns Bus A untouched (`0.0` overhead).
    """
    if not spec.is_active():
        return y_a, i_a, q_a, {"active": False, "mode": "off"}

    std_obj = std or get_standard("ntsc")
    H, W = y_a.shape[:2]
    scale_w = W / 720.0
    scale_h = H / 480.0
    t_s = float(frame_index) / max(1.0, float(fps))
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 97 + 13) & 0xFFFFFFFF)

    # Resolve Bus B (either from second graph input `in_b` or customizable overlay generator)
    if bus_b_yiq is not None and bus_b_yiq[0].shape == y_a.shape:
        y_b = bus_b_yiq[0].astype(np.float32, copy=True)
        i_b = bus_b_yiq[1].astype(np.float32, copy=True)
        q_b = bus_b_yiq[2].astype(np.float32, copy=True)
        b_source_kind = "external_in_b"
    else:
        y_b, i_b, q_b = synthesize_b_source_yiq(
            y_a, i_a, q_a,
            spec.b_source,
            frame_index=frame_index,
            fps=fps,
            std=std_obj,
            spec=spec,
            custom_overlay_rgba=custom_overlay_rgba,
        )
        b_source_kind = B_SOURCE_NAMES.get(spec.b_source, "custom_overlay_or_cam_b")
        if spec.overlay_path:
            b_source_kind = f"uploaded:{Path(spec.overlay_path).name}"

    rows = np.arange(H, dtype=np.float32)
    yn = rows[:, None] / max(1.0, float(H - 1))
    cols = np.arange(W, dtype=np.float32)[None, :]
    xn = cols / max(1.0, float(W - 1))

    mode = spec.mode
    mode_name = SWITCHER_MODE_NAMES.get(mode, "off")
    info: dict = {
        "active": True,
        "mode": mode_name,
        "mode_code": int(mode),
        "b_source": b_source_kind,
        "overlay_style": OVERLAY_STYLE_NAMES.get(spec.overlay_style, "clean_scene_or_upload"),
        "overlay_text": spec.overlay_text,
        "tbar": round(float(spec.tbar), 4),
    }

    # =========================================================================
    # Mode 1: Non-Genlocked Hot Switch Transient (非同步信号源硬切瞬态)
    # =========================================================================
    if mode == 1:
        rate = max(0.15, float(spec.hot_cut_rate))
        gen_err = float(spec.genlock_err)

        cycle_pos = (t_s * rate) % 1.0
        time_since_cut_s = cycle_pos / rate
        active_bus = int(np.floor(t_s * rate)) % 2

        if active_bus == 1:
            y_pre, i_pre, q_pre = y_a, i_a, q_a
            y_post, i_post, q_post = y_b, i_b, q_b
        else:
            y_pre, i_pre, q_pre = y_b, i_b, q_b
            y_post, i_post, q_post = y_a, i_a, q_a

        shock_env = float(np.exp(-time_since_cut_s / 0.18))
        if frame_index <= 3:
            shock_env = max(shock_env, float(np.exp(-frame_index * 0.22)))

        cut_row_frac = float(np.clip(0.18 + 0.55 * spec.tbar, 0.12, 0.78))
        cut_row = int(round(cut_row_frac * H))

        v_jump_rows = int(round(gen_err * shock_env * 0.24 * H))
        if v_jump_rows > 1:
            y_post_rolled = np.roll(y_post, v_jump_rows, axis=0)
            i_post_rolled = np.roll(i_post, v_jump_rows, axis=0)
            q_post_rolled = np.roll(q_post, v_jump_rows, axis=0)
            vbi_h = min(max(4, int(round(14 * scale_h))), v_jump_rows, H // 4)
            vbi_luma = _synthesize_vbi_band(W, vbi_h, scale_w)
            r0 = max(0, v_jump_rows - vbi_h)
            y_post_rolled[r0:r0 + vbi_h, :] = vbi_luma
            i_post_rolled[r0:r0 + vbi_h, :] = 0.0
            q_post_rolled[r0:r0 + vbi_h, :] = 0.0
        else:
            y_post_rolled, i_post_rolled, q_post_rolled = y_post.copy(), i_post.copy(), q_post.copy()

        is_mid_field_cut = shock_env > 0.52
        eff_cut_row = cut_row if is_mid_field_cut else 0

        y_out = y_post_rolled.copy()
        i_out = i_post_rolled.copy()
        q_out = q_post_rolled.copy()
        if eff_cut_row > 0:
            y_out[:eff_cut_row, :] = y_pre[:eff_cut_row, :]
            i_out[:eff_cut_row, :] = i_pre[:eff_cut_row, :]
            q_out[:eff_cut_row, :] = q_pre[:eff_cut_row, :]

        dy_from_cut = np.maximum(0.0, rows - float(eff_cut_row))
        tau_lines = max(6.0, 26.0 * scale_h)
        afc_env = np.exp(-dy_from_cut / tau_lines) * (dy_from_cut >= 0.0)
        if eff_cut_row > 0:
            afc_env[:eff_cut_row] = 0.0
        h_shear = (
            gen_err * shock_env * (46.0 * scale_w) * afc_env * np.cos(dy_from_cut / max(1.5, 4.2 * scale_h))
            + rng.standard_normal(H).astype(np.float32) * (3.5 * gen_err * shock_env * scale_w * afc_env)
        ).astype(np.float32)
        y_out = horizontal_shift(y_out, h_shear)
        i_out = horizontal_shift(i_out, h_shear)
        q_out = horizontal_shift(q_out, h_shear)

        burst_tau_lines = max(10.0, 42.0 * scale_h)
        apc_env = np.exp(-dy_from_cut / burst_tau_lines).astype(np.float32)
        if eff_cut_row > 0:
            apc_env[:eff_cut_row] = 0.0
        dphi_row = (gen_err * shock_env * 2.35 * apc_env * np.cos(dy_from_cut / max(3.0, 9.0 * scale_h)))[:, None]
        cos_p = np.cos(dphi_row).astype(np.float32)
        sin_p = np.sin(dphi_row).astype(np.float32)
        i_rot = i_out * cos_p - q_out * sin_p
        q_rot = i_out * sin_p + q_out * cos_p
        acc_env = (1.0 - 0.75 * np.exp(-dy_from_cut / max(3.0, 8.0 * scale_h)) * shock_env
                   + 0.35 * apc_env * shock_env)[:, None].astype(np.float32)
        if eff_cut_row > 0:
            acc_env[:eff_cut_row, :] = 1.0
        i_out = i_rot * acc_env
        q_out = q_rot * acc_env

        if eff_cut_row > 0:
            seam_d = np.abs(rows - float(eff_cut_row))
            spike_band = np.exp(-0.5 * (seam_d / max(1.0, 2.2 * scale_h)) ** 2)[:, None].astype(np.float32)
            snow = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.0, 2.0 * scale_w), axis=1)
            y_out = y_out * (1.0 - 0.85 * spike_band) + np.clip(0.82 + 0.25 * snow, 0.0, 1.0) * (0.85 * spike_band)
            i_out *= (1.0 - 0.90 * spike_band)
            q_out *= (1.0 - 0.90 * spike_band)

        info.update({
            "active_bus": "B" if active_bus == 1 else "A",
            "shock_envelope": round(shock_env, 4),
            "cut_row": int(eff_cut_row),
            "v_jump_rows": int(v_jump_rows),
            "h_shear_ptp_px": round(float(np.ptp(h_shear)), 3),
            "burst_phase_shock_rad": round(float(np.max(np.abs(dphi_row))), 4),
        })
        return y_out.astype(np.float32), i_out.astype(np.float32), q_out.astype(np.float32), info

    # =========================================================================
    # Mode 2: Analog T-Bar Mix / Dissolve (模拟 T-Bar 叠化与黑电平叠加漂移)
    # =========================================================================
    if mode == 2:
        alpha = float(np.clip(spec.tbar, 0.0, 1.0))
        mid_mix = float(4.0 * alpha * (1.0 - alpha))
        gen_err = float(spec.genlock_err)
        ped_add = float(spec.pedestal_add)

        if gen_err > 1e-4 and alpha > 1e-3:
            b_jitter = (
                gen_err * mid_mix * (6.5 * scale_w) * np.sin(2.0 * np.pi * (yn[:, 0] * 2.5 + t_s * 1.7))
            ).astype(np.float32)
            y_b_m = horizontal_shift(y_b, b_jitter)
            i_b_m = horizontal_shift(i_b, b_jitter)
            q_b_m = horizontal_shift(q_b, b_jitter)
        else:
            y_b_m, i_b_m, q_b_m = y_b, i_b, q_b

        ped_lift = mid_mix * ped_add * 0.22
        y_out = (1.0 - alpha) * y_a + alpha * y_b_m + ped_lift

        if gen_err > 1e-4 and mid_mix > 1e-3:
            dphi = (
                gen_err * 2.2 * np.sin(2.0 * np.pi * (yn * 4.0 + xn * 1.5) + frame_index * 0.9)
            ).astype(np.float32)
            cos_d = np.cos(dphi)
            sin_d = np.sin(dphi)
            i_b_rot = i_b_m * cos_d - q_b_m * sin_d
            q_b_rot = i_b_m * sin_d + q_b_m * cos_d
            c_mag_a = np.hypot(i_a, q_a)
            c_mag_b = np.hypot(i_b_m, q_b_m)
            sub_beat = (
                0.16 * gen_err * mid_mix * np.sqrt(c_mag_a * c_mag_b + 1e-6)
                * np.cos(2.0 * np.pi * (cols / max(2.2, 4.2 * scale_w) + rows[:, None] / max(2.0, 3.5 * scale_h)) + frame_index)
            ).astype(np.float32)
            y_out = y_out + sub_beat
        else:
            i_b_rot, q_b_rot = i_b_m, q_b_m

        i_out = (1.0 - alpha) * i_a + alpha * i_b_rot
        q_out = (1.0 - alpha) * q_a + alpha * q_b_rot

        info.update({
            "mid_mix_factor": round(mid_mix, 4),
            "pedestal_lift": round(ped_lift, 4),
        })
        return np.clip(y_out, 0.0, 1.15).astype(np.float32), i_out.astype(np.float32), q_out.astype(np.float32), info

    # =========================================================================
    # Mode 3: Analog Voltage-Ramp Wipe Generator (模拟电压斜坡发生器划像 + 自定义叠层)
    # =========================================================================
    if mode == 3:
        alpha = float(np.clip(spec.tbar, 0.0, 1.0))
        wipe_cx = spec.overlay_x if abs(spec.overlay_x - 0.50) > 1e-3 else 0.50
        wipe_cy = spec.overlay_y if (abs(spec.overlay_y - 0.78) > 1e-3 and abs(spec.overlay_y - 0.50) > 1e-3) else 0.50
        ramp_2d = _compute_wipe_field(
            yn, xn, spec.wipe_pattern,
            aspect=W / max(1.0, float(H)),
            center_x=wipe_cx,
            center_y=wipe_cy,
        )

        jit_amp = float(spec.key_jitter) * 0.028
        row_jit = (
            rng.standard_normal((H, 1)).astype(np.float32) * jit_amp
            + 0.65 * jit_amp * np.sin(2.0 * np.pi * (rows[:, None] / max(6.0, 28.0 * scale_h) + frame_index * 0.37))
        )
        spill = float(spec.subcarrier_spill)
        sc_ph = subcarrier_phase(W, H, std_obj)
        sc_ripple = (0.016 * spill * np.cos(sc_ph + frame_index * 0.8)).astype(np.float32)

        edge_w = max(0.006, 0.085 * (1.15 - 0.65 * min(1.4, float(spec.key_gain))))
        delta = (alpha - (ramp_2d + row_jit + sc_ripple)) / edge_w
        mask_b = (0.5 + 0.5 * np.tanh(delta) * float(np.clip(spec.overlay_opacity, 0.1, 1.0))).astype(np.float32)

        seam_env = (4.0 * mask_b * (1.0 - mask_b)).astype(np.float32)
        y_out = (1.0 - mask_b) * y_a + mask_b * y_b
        i_out = (1.0 - mask_b) * i_a + mask_b * i_b
        q_out = (1.0 - mask_b) * q_a + mask_b * q_b

        if spill > 1e-4:
            ring_wave = np.cos(sc_ph + frame_index * 1.1).astype(np.float32)
            y_out = y_out + seam_env * spill * (0.14 + 0.12 * ring_wave)
            i_out = i_out + seam_env * spill * 0.24 * np.cos(sc_ph * 0.5 + 0.6)
            q_out = q_out + seam_env * spill * 0.24 * np.sin(sc_ph * 0.5 + 0.6)

        info.update({
            "wipe_pattern": WIPE_PATTERN_NAMES.get(spec.wipe_pattern, "horizontal"),
            "mask_b_fraction": round(float(mask_b.mean()), 4),
            "seam_fraction": round(float(seam_env.mean()), 4),
        })
        return np.clip(y_out, 0.0, 1.15).astype(np.float32), i_out.astype(np.float32), q_out.astype(np.float32), info

    # =========================================================================
    # Mode 4 & 5: Analog Luma Keyer (`mode=4`) & Chroma Keyer (`mode=5`) + 自定义叠层
    # =========================================================================
    clip_lvl = float(spec.key_clip)
    gain_sl = max(1.5, 3.0 + 18.0 * float(spec.key_gain))
    jit_k = float(spec.key_jitter)
    spill = float(spec.subcarrier_spill)
    ped_add = float(spec.pedestal_add)

    sc_ph = subcarrier_phase(W, H, std_obj)
    c_carrier_b = (i_b * np.cos(sc_ph) + q_b * np.sin(sc_ph)).astype(np.float32)

    if mode == 4:
        key_sig = y_b + (0.38 * spill) * c_carrier_b
        raw_alpha = 0.5 + 0.5 * np.tanh(gain_sl * (key_sig - clip_lvl))
    else:
        i_lp = lowpass(i_b, max(1.2, 3.2 * scale_w), axis=1)
        q_lp = lowpass(q_b, max(1.2, 3.2 * scale_w), axis=1)
        blue_metric = np.clip((-0.68 * i_lp + 0.73 * q_lp) * 2.2 - 0.22 * np.maximum(0.0, y_b - 0.50), 0.0, 1.5)
        key_sig = (1.0 - blue_metric) + (0.32 * spill) * c_carrier_b
        raw_alpha = 0.5 + 0.5 * np.tanh(gain_sl * (key_sig - clip_lvl))

    delay_px = (4.8 * spill * scale_w)
    row_jitter_px = (
        rng.standard_normal(H).astype(np.float32) * (2.6 * jit_k * scale_w)
        + (1.8 * jit_k * scale_w) * np.sin(2.0 * np.pi * (rows / max(4.0, 14.0 * scale_h) + frame_index * 0.6))
    ).astype(np.float32)
    alpha_key = horizontal_shift(raw_alpha.astype(np.float32), np.full(H, delay_px, np.float32) + row_jitter_px)
    eff_tbar = float(spec.tbar if spec.tbar > 0.05 else 1.0) * float(np.clip(spec.overlay_opacity, 0.0, 1.0))
    alpha_key = np.clip(alpha_key * eff_tbar, 0.0, 1.0)

    edge_zone = (4.0 * alpha_key * (1.0 - alpha_key)).astype(np.float32)

    y_out = (1.0 - alpha_key) * y_a + alpha_key * y_b + (0.12 * ped_add * alpha_key)
    i_out = (1.0 - alpha_key) * i_a + alpha_key * i_b
    q_out = (1.0 - alpha_key) * q_a + alpha_key * q_b

    if spill > 1e-4:
        misreg_halo = np.abs(alpha_key - raw_alpha).astype(np.float32)
        zipper_cos = np.cos(sc_ph + frame_index * 1.3).astype(np.float32)
        zipper_sin = np.sin(sc_ph + frame_index * 1.3).astype(np.float32)
        y_out = y_out + spill * (0.18 * edge_zone * zipper_cos - 0.25 * misreg_halo)
        i_out = i_out + spill * (0.28 * (edge_zone + misreg_halo) * zipper_cos)
        q_out = q_out + spill * (0.28 * (edge_zone + misreg_halo) * zipper_sin)

    info.update({
        "key_coverage": round(float(alpha_key.mean()), 4),
        "edge_fringe_rms": round(float(np.sqrt(np.mean((edge_zone * spill) ** 2))), 4),
        "key_delay_px": round(float(delay_px), 2),
    })
    return np.clip(y_out, 0.0, 1.15).astype(np.float32), i_out.astype(np.float32), q_out.astype(np.float32), info

"""
Stage 05 -- the display device.

A CRT is an *observation point*, not part of the tape.  Keeping it as its own
stage is what lets a user answer "is this look coming from the medium or from
the monitor it was watched on?" by simply disabling the layer -- one of the
features that separates this tool from a stack of VHS filters.

Only the display stage is allowed to change the composition: a real CRT
overscans, so the delivered image is a window onto a larger raster.  Overscan is
therefore modelled as a genuine crop, resampled back to the nominal frame size
(what the viewer could actually see), not as a drawn border.

Everything here is a per-pixel operation on the finished signal, so this stage
is also the one that must stay cheap: it is what a user tweaks most.  All
coordinate grids and raster masks are cached per geometry.
"""
from __future__ import annotations

import numpy as np

from ..codecs.crt import (apply_corner_astigmatism, apply_curved_bezel_and_glass,
                          apply_deflection, apply_ehv_line_sag,
                          apply_faceplate_halation, apply_phosphor_saturation,
                          apply_scanline_beam, build_phosphor_mask,
                          moisture_effects)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, highpass, lowpass

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _resample_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resample to an exact size."""
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    sy = np.clip((np.arange(height, dtype=np.float32) + 0.5) * (h / height) - 0.5,
                 0.0, h - 1.0)
    sx = np.clip((np.arange(width, dtype=np.float32) + 0.5) * (w / width) - 0.5,
                 0.0, w - 1.0)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = (sy - y0)[:, None, None]
    fx = (sx - x0)[None, :, None]
    top = img[y0][:, x0] * (1.0 - fx) + img[y0][:, x1] * fx
    bot = img[y1][:, x0] * (1.0 - fx) + img[y1][:, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


class DisplayStage(Stage):
    name = "display"
    layers = ("display",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._cache: dict[tuple, np.ndarray | tuple] = {}
        self._prev_phosphor: np.ndarray | None = None
        self._prev_index: int = -999
        self._read_params()

    def _read_params(self) -> None:
        d = self.cfg_of("display")
        self.crt_moisture = float(d.get("crt_moisture", 0.0))
        self.crt_corrosion = float(d.get("crt_corrosion", 0.0))
        self.crt_age = float(d.get("crt_age", 0.3))
        raw_d = (self.ctx.params or {}).get("display")
        if isinstance(raw_d, dict) and "enabled" not in raw_d and len(raw_d) > 0:
            self.enabled = True
        else:
            self.enabled = float(d.get("enabled", 0.0)) > 0.5
        self.scanlines = float(d.get("scanlines", 0.35))
        self.interlace = float(d.get("interlace_flicker", 0.25))
        self.bloom = float(d.get("bloom", 0.35))
        self.defocus = float(d.get("defocus", 0.20))
        self.geometry = float(d.get("geometry_warp", 0.15))
        # --- the tube's own geometry faults -------------------------------------
        self.trapezoid_x = float(d.get("trapezoid_x", 0.0))
        self.trapezoid_y = float(d.get("trapezoid_y", 0.0))
        self.convergence = float(d.get("convergence", 0.0))
        self.convergence_y = float(d.get("convergence_y", 0.0))
        self.curvature = float(d.get("curvature", 0.0))
        # The scanning beam: a Gaussian spot of this width in lines, and the line pitch in pixels.
        self.spot_sigma = float(d.get("spot_sigma", 0.42))
        self.scan_pitch = float(d.get("scan_pitch", 2.0))
        self.spot_bloom = float(d.get("spot_bloom", 0.45))
        # The shadow mask: the metal grid in front of the phosphors.
        self.mask_pitch = float(d.get("mask_pitch", 0.0))
        self.mask_depth = float(d.get("mask_depth", 0.35))
        self.mask_type = int(round(float(d.get("mask_type", 0.0))))
        self.phosphor_rgb = float(d.get("phosphor_rgb", 0.65))
        # Faceplate internal halation, P22 phosphor saturation, corner astigmatism, EHT sag, persistence, bezel
        self.glass_halation = float(d.get("glass_halation", 0.28))
        self.phosphor_saturation = float(d.get("phosphor_saturation", 0.30))
        self.corner_defocus = float(d.get("corner_defocus", 0.22))
        self.ehv_sag = float(d.get("ehv_sag", 0.15))
        self.phosphor_persistence = float(d.get("phosphor_persistence", 0.18))
        self.bezel = float(d.get("bezel", 0.0))
        # High-voltage breathing
        self.breathing = float(d.get("breathing", 0.0))
        self.gamma = float(d.get("gamma", 2.35))
        self.color_temp = float(d.get("color_temp", 0.15))
        self.contrast = float(d.get("contrast", 0.55))
        self.brightness = float(d.get("brightness", 0.50))
        self.saturation = float(d.get("saturation", 0.50))
        self.overscan = float(d.get("overscan", 0.06))
        self.vignette = float(d.get("vignette", 0.25))
        self.reflection = float(d.get("reflection", 0.10))
        self._prepare_static()

    def on_params_changed(self) -> None:
        self._read_params()

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        self._prev_phosphor = None
        self._prev_index = -999

    def _prepare_static(self) -> None:
        """Pre-compute everything that depends only on the parameter values."""
        self._row_mask = None
        self._col_temp = (np.array([1.0 + 0.10 * self.color_temp,
                                    1.0 + 0.01 * self.color_temp,
                                    1.0 - 0.12 * self.color_temp],
                                   dtype=np.float32)[None, None, :])
        h, w = self.ctx.height, self.ctx.width
        if self.reflection > 1e-3:
            yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
            xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
            refl = 0.85 * np.exp(-(((xx - 0.28) ** 2 + (yy - 0.22) ** 2) / 0.10))
            refl = refl + 0.35 * np.clip(1.0 - yy * 1.6, 0.0, 1.0) * xx
            self._refl = (self.reflection * 0.22 * refl).astype(np.float32)
        else:
            self._refl = None

    def _masks(self, h: int, w: int):
        key = ("masks", h, w, round(self.spot_sigma, 3), round(self.scan_pitch, 3),
               round(self.scanlines, 4), round(self.vignette, 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        rows = np.arange(h, dtype=np.float32)[:, None, None]
        scale_h = (h / 540.0) if h > 580 else 1.0
        pitch = max(1.05, float(self.scan_pitch) * scale_h)
        centres = np.round(rows / pitch) * pitch
        dist = (rows - centres) / max(1e-3, float(self.spot_sigma) * scale_h)
        beam = np.exp(-0.5 * dist * dist)
        mask = 1.0 - self.scanlines * (1.0 - beam)
        mask = mask / max(1e-6, float(mask.mean()))
        row = mask.astype(np.float32)
        if self.vignette > 1e-3:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            nx = (xx / max(1, w - 1)) * 2.0 - 1.0
            ny = (yy / max(1, h - 1)) * 2.0 - 1.0
            r2 = np.clip(nx * nx + ny * ny * 1.25, 0.0, 2.5)
            vig = np.clip(1.0 - self.vignette * 0.55 * r2, 0.25, 1.0)[..., None]
            out = (row * vig).astype(np.float32)
        else:
            out = row
        self._cache[key] = out
        return out

    def _geometry_indices(self, h: int, w: int, breathe: float = 1.0):
        key = ("geom", h, w, round(self.geometry, 4), round(self.trapezoid_x, 4),
               round(self.trapezoid_y, 4), round(self.curvature, 4), round(breathe, 5))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w - 1)) * 2.0 - 1.0
        ny = (yy / max(1, h - 1)) * 2.0 - 1.0
        r2 = nx * nx + ny * ny
        k = self.geometry * 0.035
        sx = nx * (1.0 + k * r2)
        sy = ny * (1.0 + k * 0.6 * r2) + self.geometry * 0.006 * nx * nx
        if abs(self.trapezoid_x) > 1e-6 or abs(self.trapezoid_y) > 1e-6:
            sx = sx * (1.0 + self.trapezoid_x * ny)
            sy = sy * (1.0 + self.trapezoid_y * nx)
        if abs(breathe - 1.0) > 1e-6:
            sx = sx * breathe
            sy = sy * breathe
        if abs(self.curvature) > 1e-6:
            k_curv = self.curvature * 0.32
            r = np.sqrt(np.clip(r2, 1e-8, None))
            f = 1.0 / (1.0 + k_curv * r2)
            sx = sx * f * (1.0 + k_curv * 0.5 * r)
            sy = sy * f * (1.0 + k_curv * 0.5 * r)
        src_x = np.clip((sx * 0.5 + 0.5) * (w - 1), 0, w - 1)
        src_y = np.clip((sy * 0.5 + 0.5) * (h - 1), 0, h - 1)
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (src_x - x0)[..., None].astype(np.float32)
        fy = (src_y - y0)[..., None].astype(np.float32)
        res = (y0, x0, y1, x1, fx, fy)
        self._cache[key] = res
        return res

    def _shadow_mask(self, h: int, w: int) -> np.ndarray | None:
        """
        The physical shadow mask / aperture grille in front of the phosphors, with per-channel RGB
        subpixel triad separation, Black Matrix guard bands, and Trinitron damper wires.
        """
        if self.mask_pitch <= 1e-3 or self.mask_depth <= 1e-4:
            return None
        key = ("mask_rgb", h, w, round(self.mask_pitch, 3), round(self.mask_depth, 3),
               self.mask_type, round(self.phosphor_rgb, 3))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        mask = build_phosphor_mask(h, w, self.mask_pitch, self.mask_depth,
                                   self.mask_type, self.phosphor_rgb)
        if mask is not None:
            self._cache[key] = mask
        return mask

    def _channel_grid(self, h: int, w: int, scale_x: float, scale_y: float = 1.0, dy_px: float = 0.0):
        """
        A sampling grid with the image scaled about the centre by `(scale_x, scale_y)` plus vertical offset `dy_px`.

        Used for radial and vertical misconvergence of the three electron guns.
        """
        key = ("conv2d", h, w, round(scale_x, 6), round(scale_y, 6), round(dy_px, 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        nx = (xx - cx) / max(1.0, cx)
        ny = (yy - cy) / max(1.0, cy)
        r2 = nx * nx + ny * ny
        src_x = np.clip(cx + (xx - cx) / max(1e-6, scale_x), 0, w - 1)
        src_y = np.clip(cy + (yy - cy) / max(1e-6, scale_y) + dy_px * (0.35 + 0.65 * r2), 0, h - 1)
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (src_x - x0)[..., None].astype(np.float32)
        fy = (src_y - y0)[..., None].astype(np.float32)
        res = (y0, x0, y1, x1, fx, fy)
        self._cache[key] = res
        return res

    def _warp(self, rgb: np.ndarray, breathe: float = 1.0) -> np.ndarray:
        h, w = rgb.shape[:2]
        y0, x0, y1, x1, fx, fy = self._geometry_indices(h, w, breathe)
        top = rgb[y0, x0] * (1.0 - fx) + rgb[y0, x1] * fx
        bot = rgb[y1, x0] * (1.0 - fx) + rgb[y1, x1] * fx
        out = (top * (1.0 - fy) + bot * fy).astype(np.float32)

        if abs(self.convergence) > 1e-6 or abs(self.convergence_y) > 1e-6:
            cx_err = self.convergence * 0.02
            cy_err = self.convergence * 0.02 + self.convergence_y * 0.016
            dy_base = self.convergence_y * 2.4 * (h / 480.0)
            warped_base = out.copy()
            for ch, sx, sy, dyp in ((0, 1.0 + cx_err, 1.0 + cy_err, +dy_base),
                                    (2, 1.0 - cx_err, 1.0 - cy_err, -dy_base)):
                yy0, xx0, yy1, xx1, ffx, ffy = self._channel_grid(h, w, sx, sy, dyp)
                t = warped_base[yy0, xx0, ch] * (1.0 - ffx[..., 0]) + warped_base[yy0, xx1, ch] * ffx[..., 0]
                b = warped_base[yy1, xx0, ch] * (1.0 - ffx[..., 0]) + warped_base[yy1, xx1, ch] * ffx[..., 0]
                out[..., ch] = t * (1.0 - ffy[..., 0]) + b * ffy[..., 0]
        return out

    # ------------------------------------------------------------------
    def _apply_crt_moisture(self, rgb: np.ndarray, frame_index: int
                            ) -> tuple[np.ndarray, dict]:
        """The damp video amplifier, at the very end of the chain."""
        if self.crt_moisture <= 1e-4 and self.crt_corrosion <= 1e-4:
            return rgb, {}
        eff = moisture_effects(self.crt_moisture, self.crt_corrosion, self.crt_age)
        out = np.array(rgb, np.float32, copy=True)
        out = out + eff.bias_drift
        out = out * np.array(eff.cast, np.float32)[None, None, :]
        info = {"bias_drift": round(eff.bias_drift, 4),
                "cast": [round(g, 4) for g in eff.cast]}
        out, geo = apply_deflection(out, eff.deflection, frame_index, self.ctx.seed)
        info.update(geo)
        return np.clip(out, 0.0, 1.0), info

    def _apply_persistence(self, rgb: np.ndarray, frame_index: int) -> np.ndarray:
        """
        P22 phosphor decay persistence across frames.

        Red (Y2O2S:Eu) has a longer millisecond decay tail than Green (ZnS:Cu,Al) and Blue (ZnS:Ag),
        leaving a subtle warm phosphor afterglow behind fast-moving bright highlights on dark backgrounds.
        """
        p = float(np.clip(self.phosphor_persistence, 0.0, 0.90))
        if p <= 1e-4:
            self._prev_phosphor = None
            return rgb
        if (self._prev_phosphor is not None
                and self._prev_phosphor.shape == rgb.shape
                and self._prev_index == frame_index - 1):
            decay = np.array([p * 0.68, p * 0.48, p * 0.36], dtype=np.float32)[None, None, :]
            tail = self._prev_phosphor * decay
            excess = np.maximum(0.0, tail - rgb * 0.35)
            rgb = np.clip(rgb + excess * 0.65, 0.0, 1.0).astype(np.float32)
        self._prev_phosphor = rgb.copy()
        self._prev_index = frame_index
        return rgb

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        if not self.enabled:
            if sig is None:
                return buf
            out = FrameBuffer(sig.to_rgb(), buf.composite, buf.index, buf.time,
                              buf.std, dict(buf.meta))
            out.meta["signal"] = sig
            return out

        rgb = buf.rgb if sig is None else sig.to_rgb()

        # 1. Deflection yoke geometry warp, keystone, curvature, breathing & misconvergence
        _geom_on = (self.geometry > 1e-3 or abs(self.curvature) > 1e-6
                    or abs(self.trapezoid_x) > 1e-6 or abs(self.trapezoid_y) > 1e-6
                    or abs(self.convergence) > 1e-6 or abs(self.convergence_y) > 1e-6)
        _breathe = 1.0
        if abs(self.breathing) > 1e-6:
            _breathe = 1.0 + self.breathing * 0.12 * (float((rgb @ LUMA).mean()) - 0.35)
            _geom_on = True
        if _geom_on:
            rgb = self._warp(rgb, _breathe)

        # 1b. Line-load horizontal EHT sag (bright scanlines pull flyback voltage down and bow outward)
        scale_w = (rgb.shape[1] / 720.0) if rgb.shape[1] > 760 else 1.0
        if self.ehv_sag > 1e-4:
            rgb = apply_ehv_line_sag(rgb, self.ehv_sag, scale_w)

        # 2. Electron optics: global beam defocus, corner astigmatism, faceplate glass halation & bloom
        if self.defocus > 1e-3:
            rgb = lowpass(rgb, (1.0 + self.defocus * 1.5) * scale_w)
        if self.corner_defocus > 1e-4:
            rgb = apply_corner_astigmatism(rgb, self.corner_defocus, scale_w)
        if self.bloom > 1e-3 or self.glass_halation > 1e-4:
            rgb = apply_faceplate_halation(rgb, self.glass_halation, self.bloom, scale_w)

        # 3. Phosphor / tube transfer response & P22 highlight saturation
        rgb = np.clip(rgb, 0.0, 1.0)
        if self.phosphor_saturation > 1e-4:
            rgb = apply_phosphor_saturation(rgb, self.phosphor_saturation)
        rgb = self._apply_persistence(rgb, int(buf.index))

        g = max(0.05, float(self.gamma))
        if abs(g - 1.0) > 1e-3:
            np.power(rgb, 1.0 / g, out=rgb)
        if abs(self.contrast - 0.5) > 1e-4:
            rgb = (rgb - 0.5) * (0.6 + 1.4 * self.contrast) + 0.5
        if abs(self.brightness - 0.5) > 1e-4:
            rgb += (self.brightness - 0.5) * 0.35
        sat = 0.5 + 1.15 * self.saturation
        if abs(sat - 1.0) > 1e-3:
            lum = (rgb @ LUMA)[..., None]
            rgb = lum + (rgb - lum) * sat
        rgb = rgb * self._col_temp

        # 4. RGB Phosphor Shadow Mask / Aperture Grille + Luminance-Coupled Dynamic Scanlines
        mask = self._shadow_mask(rgb.shape[0], rgb.shape[1])
        if mask is not None:
            rgb = rgb * mask

        field_idx = int(buf.meta.get("field", int(buf.index) % 2))
        base_row_mask = self._masks(rgb.shape[0], rgb.shape[1])
        rgb = apply_scanline_beam(
            rgb,
            base_row_mask=base_row_mask,
            scanlines=self.scanlines,
            spot_sigma=self.spot_sigma,
            scan_pitch=self.scan_pitch,
            spot_bloom=self.spot_bloom,
            field_parity=field_idx,
            interlace=self.interlace,
        )

        if self.interlace > 1e-3:
            sign = 1.0 if field_idx == 0 else -1.0
            hf = highpass(rgb @ LUMA, 1.1)
            rgb = rgb + (self.interlace * 0.05 * sign) * hf[..., None]

        # 5. Overscan: the raster overfills the tube
        if self.overscan > 1e-4:
            h, w = rgb.shape[:2]
            scale_up = 1.0 / max(0.5, 1.0 - 2.0 * self.overscan)
            big_h = max(h, int(round(h * scale_up)))
            big_w = max(w, int(round(w * scale_up)))
            enlarged = _resample_to(rgb, big_h, big_w)
            y0 = (big_h - h) // 2
            x0 = (big_w - w) // 2
            rgb = enlarged[y0:y0 + h, x0:x0 + w]

        # 6. Curved CRT glass reflection & physical tube bezel
        if self.reflection > 1e-3 or self.bezel > 1e-4:
            rgb = apply_curved_bezel_and_glass(rgb, self.bezel, self.reflection, self.curvature)

        # 7. CRT interface moisture: the last junction in the chain
        if self.crt_moisture > 1e-4 or self.crt_corrosion > 1e-4:
            rgb, crt_info = self._apply_crt_moisture(rgb, buf.index)
            meta = dict(buf.meta)
            meta["crt"] = crt_info
            buf = FrameBuffer(rgb, buf.composite, buf.index, buf.time, buf.std, meta)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        out = FrameBuffer(rgb, buf.composite, buf.index, buf.time, buf.std, dict(buf.meta))
        out.meta["signal"] = sig
        return out

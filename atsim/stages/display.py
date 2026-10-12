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
        d = self.cfg_of("display")
        # CRT interface moisture.  Read here, after super().__init__, because `cfg_of` reads
        # `self.cfg` and that is what super() establishes.  The first insertion put these
        # lines *above* the super call, so the stage could not be constructed at all --
        # `AttributeError: 'DisplayStage' object has no attribute 'cfg'`.
        self.crt_moisture = float(d.get("crt_moisture", 0.0))
        self.crt_corrosion = float(d.get("crt_corrosion", 0.0))
        self.crt_age = float(d.get("crt_age", 0.3))
        self.enabled = float(d.get("enabled", 1.0)) > 0.5
        self.scanlines = float(d.get("scanlines", 0.35))
        self.interlace = float(d.get("interlace_flicker", 0.25))
        self.bloom = float(d.get("bloom", 0.35))
        self.defocus = float(d.get("defocus", 0.20))
        self.geometry = float(d.get("geometry_warp", 0.15))
        # --- the tube's own geometry faults -------------------------------------
        # A trapezoid raster: the yoke's scan is not rectangular to the tube face, so the top
        # and bottom edges can differ in width and the left and right edges in height.
        self.trapezoid_x = float(d.get("trapezoid_x", 0.0))
        self.trapezoid_y = float(d.get("trapezoid_y", 0.0))
        # Misconvergence: three beams through a shadow mask, so red and blue land slightly
        # scaled about the centre.  Zero at the centre and growing outwards, which is why a real
        # tube fringes at the edges and not in the middle.
        self.convergence = float(d.get("convergence", 0.0))
        self.gamma = float(d.get("gamma", 2.35))
        self.color_temp = float(d.get("color_temp", 0.15))
        self.contrast = float(d.get("contrast", 0.55))
        self.brightness = float(d.get("brightness", 0.50))
        self.saturation = float(d.get("saturation", 0.50))
        self.overscan = float(d.get("overscan", 0.06))
        self.vignette = float(d.get("vignette", 0.25))
        self.reflection = float(d.get("reflection", 0.10))
        self._cache: dict[tuple[str, int, int], np.ndarray | tuple] = {}
        self._prepare_static()

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        pass

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
        key = ("masks", h, w)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        rows = np.arange(h, dtype=np.float32)[:, None, None]
        # The delivered image is presented on a finer display raster; a
        # scanline period of ~1.4 output lines keeps the classic fine line
        # structure visible after any resampling.
        mod = 0.5 + 0.5 * np.cos(2.0 * np.pi * rows / 1.4)
        row = (1.0 - self.scanlines * 0.45 * mod).astype(np.float32)
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

    def _geometry_indices(self, h: int, w: int):
        key = ("geom", h, w)
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
        # a keystone: the raster's width tapers from the top to the bottom, and its height from the left
        # to the right.  Two independent faults, as on a real set with a knocked yoke.
        if abs(self.trapezoid_x) > 1e-6 or abs(self.trapezoid_y) > 1e-6:
            sx = sx * (1.0 + self.trapezoid_x * ny)
            sy = sy * (1.0 + self.trapezoid_y * nx)
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

    def _channel_grid(self, h: int, w: int, scale: float):
        """
        A sampling grid with the image scaled about the centre by `scale`.

        Used for misconvergence: scaling each channel about the centre makes the error zero in the middle and
        worse at the edges, which is the shape a real tube has.  A uniform shift would instead fringe evenly
        across the whole picture, which is the approximation this avoids.
        """
        key = ("conv", h, w, round(scale, 6))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        src_x = np.clip(cx + (xx - cx) / max(1e-6, scale), 0, w - 1)
        src_y = np.clip(cy + (yy - cy) / max(1e-6, scale), 0, h - 1)
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (src_x - x0)[..., None].astype(np.float32)
        fy = (src_y - y0)[..., None].astype(np.float32)
        res = (y0, x0, y1, x1, fx, fy)
        self._cache[key] = res
        return res

    def _warp(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        y0, x0, y1, x1, fx, fy = self._geometry_indices(h, w)
        top = rgb[y0, x0] * (1.0 - fx) + rgb[y0, x1] * fx
        bot = rgb[y1, x0] * (1.0 - fx) + rgb[y1, x1] * fx
        out = (top * (1.0 - fy) + bot * fy).astype(np.float32)

        if abs(self.convergence) > 1e-6:
            # red is drawn slightly larger than green and blue slightly smaller, both about the centre, so
            # the fringing is zero in the middle and grows towards the edges
            c = self.convergence * 0.02
            for ch, scale in ((0, 1.0 + c), (2, 1.0 - c)):
                yy0, xx0, yy1, xx1, ffx, ffy = self._channel_grid(h, w, scale)
                t = rgb[yy0, xx0] * (1.0 - ffx) + rgb[yy0, xx1] * ffx
                b = rgb[yy1, xx0] * (1.0 - ffx) + rgb[yy1, xx1] * ffx
                plane = (t * (1.0 - ffy) + b * ffy)
                out[..., ch] = plane[..., ch]
        return out

    # ------------------------------------------------------------------
    def _apply_crt_moisture(self, rgb: np.ndarray, frame_index: int
                            ) -> tuple[np.ndarray, dict]:
        """
        The damp video amplifier, at the very end of the chain.

        Three effects with three separate causes, applied in the order the signal meets them:
        the amplifier's bias shifts the level, the three channels' independent wear produce a
        cast, and the sagging supply shrinks and breathes the raster.

        The geometry is the one that matters diagnostically: it is the *only* fault in the
        model that moves the edges of the raster itself. No upstream fault can do that -- a
        tape cannot bend the picture -- so if the frame edges are moving, the fault is here
        and nowhere else.
        """
        if self.crt_moisture <= 1e-4 and self.crt_corrosion <= 1e-4:
            return rgb, {}
        from ..codecs.crt import apply_deflection, moisture_effects
        eff = moisture_effects(self.crt_moisture, self.crt_corrosion, self.crt_age)
        out = np.array(rgb, np.float32, copy=True)

        # the level: a bias shift moves black across the whole picture, so a damp set looks
        # washed out rather than grainy
        out = out + eff.bias_drift

        # the cast: separate amplifiers per cathode, so the channels gain differently.  The
        # spread grows with the set's age, which is why a damp room washes out a new set and
        # tints an old one.
        out = out * np.array(eff.cast, np.float32)[None, None, :]

        info = {"bias_drift": round(eff.bias_drift, 4),
                "cast": [round(g, 4) for g in eff.cast]}

        # the geometry last, because it resamples the picture the first two produced
        out, geo = apply_deflection(out, eff.deflection, frame_index, self.ctx.seed)
        info.update(geo)
        return np.clip(out, 0.0, 1.0), info

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        # Even with the display stage disabled, the frame that leaves the chain
        # must be the *reconstructed picture*.  Returning the pass-through RGB
        # buffer here would silently discard every bandwidth, crosstalk and
        # time-base change upstream -- which is exactly what made this stage's
        # "enabled" switch look like a no-op for the whole format layer.
        if not self.enabled:
            if sig is None:
                return buf
            out = FrameBuffer(sig.to_rgb(), buf.composite, buf.index, buf.time,
                              buf.std, dict(buf.meta))
            out.meta["signal"] = sig
            return out

        rgb = buf.rgb if sig is None else sig.to_rgb()

        # The tracking-error shift that used to live here has been removed: the
        # parameter belongs to the tape layer, where it is implemented against the
        # head-switch band it actually displaces.  Keeping a second copy in the
        # display stage duplicated the defect and, worse, had the display reading
        # the medium's parameters -- which is what happens when a parameter moves
        # between layers and the old reader is forgotten.

        if self.geometry > 1e-3:
            rgb = self._warp(rgb)

        # --- electron optics: beam defocus, then highlight bloom -----------
        if self.defocus > 1e-3:
            rgb = lowpass(rgb, 1.0 + self.defocus * 1.5)
        if self.bloom > 1e-3:
            lum = rgb @ LUMA
            hot = np.clip(lum - 0.72, 0.0, None)
            halo = lowpass(hot, 3.5 + 8.0 * self.bloom)
            peak = float(halo.max())
            if peak > 1e-6:
                rgb = rgb + (self.bloom * 0.55 / peak) * halo[..., None]

        # --- phosphor / tube response --------------------------------------
        rgb = np.clip(rgb, 0.0, 1.0)
        g = max(1.0, self.gamma)
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

        # --- raster mask, vignette, glass ----------------------------------
        rgb = rgb * self._masks(rgb.shape[0], rgb.shape[1])
        if self._refl is not None:
            rgb = rgb + self._refl
        if self.interlace > 1e-3:
            # thin bright lines flicker because each field carries only
            # alternate lines; visible on detail, invisible on flat areas
            sign = 1.0 if int(buf.meta.get("field", 0)) == 0 else -1.0
            hf = highpass(rgb @ LUMA, 1.1)
            rgb = rgb + (self.interlace * 0.05 * sign) * hf[..., None]

        # --- overscan: the raster overfills the tube ----------------------
        # Geometry matters here, and the obvious implementations are both wrong.
        #
        # Cropping the edges and stretching what remains magnifies the picture by
        # 1/(1-2*overscan) and shifts the outermost rows off the frame -- which
        # silently erased the head-switching band, because that band lives in
        # exactly those bottom rows.  Cropping and padding with black instead
        # buried the band under a 6% black border.
        #
        # What a set actually does is adjust the raster so it *overfills* the
        # screen: the whole picture is enlarged and its outer edges fall off the
        # tube.  So the picture is scaled up and the frame takes the centre --
        # the edges are genuinely cut, nothing is stretched back, and no border
        # appears.
        if self.overscan > 1e-4:
            h, w = rgb.shape[:2]
            scale_up = 1.0 / max(0.5, 1.0 - 2.0 * self.overscan)
            big_h = max(h, int(round(h * scale_up)))
            big_w = max(w, int(round(w * scale_up)))
            enlarged = _resample_to(rgb, big_h, big_w)
            y0 = (big_h - h) // 2
            x0 = (big_w - w) // 2
            rgb = enlarged[y0:y0 + h, x0:x0 + w]

        # CRT interface moisture: the last junction in the chain, and the only fault
        # that can move the edges of the raster.
        if self.crt_moisture > 1e-4 or self.crt_corrosion > 1e-4:
            rgb, crt_info = self._apply_crt_moisture(rgb, buf.index)
            meta = dict(buf.meta)
            meta["crt"] = crt_info
            buf = FrameBuffer(rgb, buf.composite, buf.index, buf.time, buf.std, meta)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        out = FrameBuffer(rgb, buf.composite, buf.index, buf.time, buf.std, dict(buf.meta))
        out.meta["signal"] = sig
        return out

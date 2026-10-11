"""
Stage 01 -- the imaging system.

This layer answers "what shot it", before any video signal exists.  Everything
here happens in RGB (light), which is why it cannot be folded into the format
layer: a lens/CCD artefact has a different spatial and chromatic signature than
a bandwidth limit applied to an already-encoded signal.

Motion blur is the one genuinely temporal operation: a 1/60 s shutter integrates
two fields, so it is implemented as a blend with the previous frame and needs
the previous input frame.  `set_previous` is called by the renderer.
"""
from __future__ import annotations

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import ValueNoise3D, bandlimited_noise, rng_for
from ..core.signal import gaussian_kernel, lowpass


class CameraStage(Stage):
    name = "camera"
    layers = ("camera",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()
        # state that must survive a parameter change: the auto-control loops take time to
        # settle, so re-creating them on every edit would defeat the time constant they exist
        # to express
        self._prev: np.ndarray | None = None
        self._fpn: np.ndarray | None = None
        self._noise: ValueNoise3D | None = None
        self._ae_state: float | None = None
        self._awb_state: np.ndarray | None = None
        self._af_state: float | None = None

    def _read_params(self) -> None:
        cfg = self.cfg_of("camera")
        self.sensor_noise = float(cfg.get("sensor_noise", 0.0))
        self.grain = float(cfg.get("noise_grain", 1.4))
        self.fixed_pattern = float(cfg.get("fixed_pattern", 0.0))
        self.dynamic_range = float(cfg.get("dynamic_range", 8.0))
        self.exposure = float(cfg.get("exposure", 0.0))
        self.wb_temp = float(cfg.get("white_balance_temp", 0.0))
        self.sharpness = float(cfg.get("sharpness", 1.0))
        self.aberration = float(cfg.get("chroma_aberration", 0.0))
        self.motion_blur = float(cfg.get("motion_blur", 0.0))
        self.smear = float(cfg.get("ccd_smear", 0.0))
        self.halation = float(cfg.get("halation", 0.0))
        self.gain = float(cfg.get("ccd_gain", 0.0))
        self.color_cast = float(cfg.get("color_cast", 0.0))
        self.highlight_tint = float(cfg.get("highlight_tint", 0.0))
        # --- lens and sensor type -------------------------------------
        self.lens_wear = float(cfg.get("lens_wear", 0.0))
        self.lens_flare = float(cfg.get("lens_flare", 0.0))
        self.flare_blades = float(cfg.get("flare_blades", 6.0))
        self.sensor_type = str(cfg.get("sensor_type", 0.0))
        self.is_3ccd = self.sensor_type in ("1", "1.0", "3ccd", "3CCD")


    # ------------------------------------------------------------------

    def on_params_changed(self) -> None:
        """Re-read every parameter this stage cached, so a per-frame change reaches it."""
        self._read_params()

    def _noise_field(self, shape: tuple[int, int]) -> ValueNoise3D:
        """
        The sensor noise field, built for the frame shape actually arriving.

        `ctx.height/width` is the renderer's *nominal* size; a source can deliver
        frames of another size (a proxy render, for instance), and pre-building
        the field from the context then fails to broadcast against the real frame.
        """
        if self._noise is None or self._noise_shape != (int(shape[0]), int(shape[1])):
            h, w = int(shape[0]), int(shape[1])
            self._noise = ValueNoise3D(
                seed=self.ctx.seed * 31 + 7, height=h, width=w,
                spatial_scale=max(2.0, self.grain * 2.0),
                time_scale=max(2.0, 6.0 / max(1e-3, self.ctx.fps) * 4.0),
            )
            self._noise_shape = (h, w)
            self._fpn = None
        return self._noise

    # ------------------------------------------------------------------
    def set_previous(self, rgb: np.ndarray | None) -> None:
        self._prev = None if rgb is None else rgb

    def _fixed_pattern(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        if self._fpn is None or self._fpn.shape[:2] != (h, w):
            rng = rng_for(self.ctx.seed, "camera", "fpn")
            # column-oriented pattern (CCD column amplifiers) plus a little grain
            col = rng.standard_normal((1, w, 3)).astype(np.float32)
            col = lowpass(col, 1.2, axis=1)
            grain = bandlimited_noise(rng, (h, w, 3), 1.6, 1.6)
            self._fpn = (0.65 * col + 0.35 * grain).astype(np.float32)
        return self._fpn

    def _white_balance(self, rgb: np.ndarray) -> np.ndarray:
        if abs(self.wb_temp) < 1e-4:
            return rgb
        t = float(self.wb_temp)
        gains = np.array([1.0 + 0.18 * t, 1.0 + 0.02 * t, 1.0 - 0.20 * t], dtype=np.float32)
        return rgb * gains[None, None, :]

    def _expose(self, rgb: np.ndarray) -> np.ndarray:
        if abs(self.exposure) < 1e-4:
            return rgb
        return rgb * float(2.0 ** self.exposure)

    def _roll_off(self, rgb: np.ndarray) -> np.ndarray:
        """
        Highlight compression: a sensor with `dynamic_range` stops clips
        everything above its saturation point.  Low-range CCDs go to flat white
        in windows and skies, which is a strong era cue.
        """
        dr = float(np.clip(self.dynamic_range, 3.0, 16.0))
        knee = float(np.clip(0.5 ** (dr / 8.0), 0.05, 0.95))
        lin = np.clip(rgb, 0.0, None)
        # soft knee: x / (1 + x/limit) style compression, normalised to unity at knee
        comp = np.where(
            lin <= knee,
            lin,
            knee + (1.0 - knee) * (1.0 - np.exp(-(lin - knee) / max(1e-4, 1.0 - knee))),
        )
        return comp.astype(np.float32)

    def _flare(self, rgb: np.ndarray) -> np.ndarray:
        """
        CCD vertical smear: bright highlights bleed along the readout column.

        This is the single most recognisable low-light signature of an interlaced
        CCD camcorder -- a street lamp drags a vertical stripe down the frame.  It
        is an *imager* artefact, not a display or tape one, so it belongs here and
        must not be confusable with the display stage's bloom.
        """
        if self.smear < 1e-4:
            return rgb
        peak = rgb.max(axis=2, keepdims=True)
        hot = np.clip(peak - 0.82, 0.0, None)
        if not np.any(hot > 0.0):
            return rgb
        col = hot.mean(axis=0, keepdims=True)
        # smear is a charge-transfer effect: it is sharper than a lens flare and
        # carries a slight colour tint from the filter array
        col_s = lowpass(col, 3.0, axis=1)
        tint = np.array([1.0, 0.92, 1.12], dtype=np.float32)
        return (rgb + (0.85 * self.smear) * col_s * tint[None, None, :]).astype(np.float32)

    def _halation(self, rgb: np.ndarray) -> np.ndarray:
        """
        Highlight bloom at the sensor, with the magenta fringe of an overexposed
        CCD.  Applied before the tape, so it sits *under* any tape artefacts --
        which is what the reference material shows: the glow is part of the
        original exposure, the noise sits on top of it.
        """
        if self.halation < 1e-4:
            return rgb
        a = float(self.halation)
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        hot = np.clip((lum - 0.72) / 0.28, 0.0, None)
        # a tight core dominates; the wide veil is deliberately much weaker and
        # *scaled by how much of the frame is actually blown out*, so a mostly
        # mid-tone scene does not get a global pink wash (an earlier version used
        # a flat veil and tinted the whole sky)
        core = lowpass(hot, 2.5 + 3.5 * a)
        veil = lowpass(hot, 12.0 + 20.0 * a)
        peak = float(core.max())
        if peak > 1e-6:
            core = core / peak
        coverage = float(np.clip(hot.mean() * 14.0, 0.0, 1.0))
        halo = np.clip(0.80 * core + 0.32 * coverage * veil, 0.0, 1.6)
        tint = np.array([1.05, 0.88, 1.05], dtype=np.float32)
        return (rgb + (a * 0.52) * halo[..., None] * tint[None, None, :]).astype(np.float32)

    def _color_science(self, rgb: np.ndarray) -> np.ndarray:
        """
        The imager's colour cast and its highlight tint.

        Consumer Sony CCDs of this era render shadows distinctly cyan-blue, which
        is why night footage reads as cold even under sodium light.  This is a
        *camera* characteristic: a different camcorder pointed at the same scene
        would not have it.
        """
        out = rgb
        if abs(self.color_cast) > 1e-4:
            c = float(self.color_cast)
            # +1 warm (R up, B down), -1 cool/cyan (R down, B up, G slightly up)
            gains = np.array([1.0 - 0.16 * c, 1.0 + 0.03 * c, 1.0 + 0.20 * c],
                             dtype=np.float32)
            out = out * gains[None, None, :]
        if self.highlight_tint > 1e-4:
            t = float(self.highlight_tint)
            lum = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            over = np.clip((lum - 0.84) / 0.16, 0.0, 1.0)[..., None]
            tint = np.array([1.0 + 0.07 * t, 1.0 - 0.12 * t, 1.0 + 0.10 * t],
                            dtype=np.float32)
            out = out * (1.0 - over) + out * tint[None, None, :] * over
        return out.astype(np.float32)

    def _color_cleanup(self, rgb: np.ndarray) -> np.ndarray:
        """
        3CCD colour purity.

        A three-chip camera splits the image with a prism, so every sensor sees
        the full resolution of its own primary and there is no colour-filter
        array to interpolate.  Two consequences are visible in the reference
        footage: colour transitions are clean (no false colour on fine detail),
        and chroma noise is far lower.
        """
        y = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # isolate the chroma difference signals and smooth them slightly: enough
        # to suppress single-pixel false colour, not enough to blur real colour
        cb = rgb[..., 2] - y
        cr = rgb[..., 0] - y
        cb = lowpass(cb, 0.85)
        cr = lowpass(cr, 0.85)
        out = np.empty_like(rgb)
        out[..., 0] = y + cr
        out[..., 2] = y + cb
        # green carries the difference so that luma is preserved exactly
        out[..., 1] = (y - 0.299 * out[..., 0] - 0.114 * out[..., 2]) / 0.587
        return out.astype(np.float32)

    def _gain_noise(self, rgb: np.ndarray, frame_index: int) -> np.ndarray:
        """
        AGC gain: raising the gain amplifies the sensor's read noise with it, and
        on these camcorders that noise is strongly *chromatic*.  Modelled as a
        single luminance-ish grain plus a colour-difference noise term, which is
        what makes night footage speckle in red/green/blue rather than grey.
        """
        if self.gain <= 1e-4:
            return rgb
        g = float(self.gain)
        rng = np.random.default_rng(
            (int(self.ctx.seed) * 2654435761 + frame_index * 40503) & 0xFFFFFFFF)
        shape = rgb.shape[:2]
        lum_grain = self._coarse_noise(rng, shape, 1.0)
        chroma_grain = self._coarse_noise(rng, shape, 2.2)
        amt = 0.055 * g
        out = rgb + (amt * lum_grain)[..., None]
        # push the colour noise along R/B and G in opposition (chroma speckle)
        out[..., 0] += amt * 0.85 * chroma_grain
        out[..., 2] += amt * 0.85 * chroma_grain
        out[..., 1] -= amt * 0.55 * chroma_grain
        return out.astype(np.float32)

    def _coarse_noise(self, rng: np.random.Generator,
                      shape: tuple[int, int], grain: float) -> np.ndarray:
        """Spatially correlated noise: real sensor grain is not per-pixel white."""
        from ..core.rng import bandlimited_noise
        return bandlimited_noise(rng, shape, grain, grain)

    def _lens_wear(self, rgb: np.ndarray) -> np.ndarray:
        """
        Worn lens coatings.

        Coating damage is not "blur": it scatters light *away* from where it
        should go, so the effect is a loss of contrast plus a veiling haze that
        lifts the blacks -- and, because the anti-reflective layers are what
        suppress them, it simultaneously makes internal reflections *worse*.  The
        ghosting therefore rises automatically with wear rather than being set
        independently, which is how a worn lens actually behaves.
        """
        if self.lens_wear < 1e-4:
            return rgb
        a = float(self.lens_wear)
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # veil: a wide, energy-weighted glare that lifts the shadows
        veil = lowpass(lum, 18.0 + 30.0 * a)
        veil = veil / max(1e-6, float(veil.max()))
        # contrast loss: pull toward a hazy mid-tone
        rgb = rgb * (1.0 - 0.34 * a) + (veil[..., None] * 0.42 * a)
        # slight desaturation, as scattered light is largely achromatic
        lm = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        return (lm + (rgb - lm) * (1.0 - 0.18 * a)).astype(np.float32)

    def _lens_flare(self, rgb: np.ndarray) -> np.ndarray:
        """
        Lens flare: aperture-shaped ghosts and a geometric starburst.

        Distinguish this from `halation`.  Halation is a *sensor* effect -- an
        overexposed CCD blooming into a round veil.  Flare is an *optical* one:
        internal reflections image the aperture, so the ghosts carry the iris
        shape (a polygon) and a starburst whose ray count follows the number of
        diaphragm blades.  Both are modelled here as separable 1-D kernels so a
        six-blade iris produces a recognisable hexagonal starburst.
        """
        if self.lens_flare < 1e-4:
            return rgb
        a = float(self.lens_flare)
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        src = np.clip(lum - 0.80, 0.0, None)
        if not np.any(src > 0.0):
            return rgb
        peak = float(src.max())
        if peak <= 1e-6:
            return rgb
        src = src / peak
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = float(np.argmax(src) // w), float(np.argmax(src) % w)

        # --- iris-shaped ghost -------------------------------------------
        # build a regular polygon at a modest radius from the flare source
        blades = int(np.clip(round(self.flare_blades), 3, 10))
        radius = 0.14 * max(h, w) * (0.6 + 0.8 * a)
        ang = np.arctan2(yy - cy, xx - cx)
        # polygon radius as a function of angle (apothem/cos of wrapped angle)
        sector = 2.0 * np.pi / blades
        rel = np.mod(ang + sector * 0.5, sector) - sector * 0.5
        poly_r = (radius * np.cos(sector * 0.5)) / np.maximum(np.cos(rel), 1e-3)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ghost = np.clip(1.0 - np.abs(dist - poly_r) / max(2.0, 0.02 * max(h, w)), 0.0, 1.0)
        ghost = lowpass(ghost, 2.0 + 6.0 * a)

        # --- starburst: one ray per blade -------------------------------
        rays = np.zeros((h, w), dtype=np.float32)
        ray_len = 0.30 * max(h, w) * (0.5 + 1.2 * a)
        for k in range(blades):
            theta = 2.0 * np.pi * k / blades + 0.15
            dy, dx = np.sin(theta), np.cos(theta)
            proj = (yy - cy) * dy + (xx - cx) * dx
            perp = np.abs(-(yy - cy) * dx + (xx - cx) * dy)
            rays = np.maximum(rays, np.clip(1.0 - perp / (1.5 + 4.0 * a), 0.0, 1.0)
                              * np.clip(1.0 - proj / ray_len, 0.0, 1.0) * (proj > 0))
        rays = lowpass(rays, 1.2)

        # flare is tinted by the coating stack: a cool blue-violet cast
        tint = np.array([0.85, 0.90, 1.25], dtype=np.float32)
        flare = (0.55 * ghost + 0.65 * rays) * a
        return (rgb + flare[..., None] * tint[None, None, :] * 0.55).astype(np.float32)

    def _aberration(self, rgb: np.ndarray) -> np.ndarray:
        """Lateral chromatic aberration: channels focus at slightly different planes."""
        if self.aberration < 1e-4:
            return rgb
        a = float(self.aberration)
        r = lowpass(rgb[..., 0], 1.0 + 2.2 * a)
        b = lowpass(rgb[..., 2], 1.0 + 2.6 * a)
        g = lowpass(rgb[..., 1], 1.0 + 0.5 * a)
        return np.stack([r, g, b], axis=-1).astype(np.float32)

    def _sharpen(self, rgb: np.ndarray) -> np.ndarray:
        """
        Lens + camera detail.  sharpness=1.0 is neutral; below 1 the lens is
        soft, above 1 the camera's aperture correction adds a halo.
        """
        s = float(self.sharpness)
        if abs(s - 1.0) < 1e-3:
            return rgb
        if s < 1.0:
            return lowpass(rgb, 1.0 + (1.0 - s) * 1.6)
        amount = (s - 1.0) * 0.9
        blurred = lowpass(rgb, 0.9)
        return (rgb + amount * (rgb - blurred)).astype(np.float32)

    # ------------------------------------------------------------------
    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        rgb = buf.rgb.astype(np.float32, copy=False)
        rgb = self._white_balance(rgb)
        rgb = self._expose(rgb)

        if self.motion_blur > 1e-4 and self._prev is not None and self._prev.shape == rgb.shape:
            k = float(np.clip(self.motion_blur, 0.0, 0.95))
            rgb = (rgb * (1.0 - k) + self._prev * k).astype(np.float32)

        if self.sensor_noise > 1e-4:
            n = self._noise_field(rgb.shape[:2]).sample(buf.index)
            rgb = rgb + (self.sensor_noise * 0.35) * n[..., None]

        if self.fixed_pattern > 1e-4:
            rgb = rgb + (self.fixed_pattern * 0.06) * self._fixed_pattern(rgb.shape[:2])

        rgb = self._gain_noise(rgb, buf.index)

        # 3CCD: each sensor carries one primary, so read noise enters as
        # *luminance* grain rather than as colour speckle, and there is no
        # colour-filter mosaic -- hence the lower noise and cleaner colour
        # separation that distinguish a 3-chip camera from a single-chip one.
        # This is a different sensor, not a better setting on the same sensor.
        if self.is_3ccd:
            rgb = self._color_cleanup(rgb)

        rgb = self._roll_off(rgb)
        rgb = self._halation(rgb)
        rgb = self._lens_flare(rgb)
        rgb = self._flare(rgb)
        rgb = self._aberration(rgb)
        rgb = self._sharpen(rgb)
        rgb = self._lens_wear(rgb)
        rgb = self._color_science(rgb)

        return FrameBuffer(np.clip(rgb, 0.0, 2.0).astype(np.float32), None,
                           buf.index, buf.time, buf.std, dict(buf.meta))

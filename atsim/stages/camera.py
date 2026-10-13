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

from ..codecs.operator import HAND
from ..codecs.autocontrol import (ExposureLoop, FocusLoop, WhiteBalanceLoop,
                                  centre_weights)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import ValueNoise3D, bandlimited_noise, derive_seed, rng_for
from ..core.signal import gaussian_kernel, lowpass
from ..codecs.lens import LensSpec, field_curvature, longitudinal_ca, chromatic_aberration, coma, astigmatism, spherical, distortion


def _subpixel_shift(rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Move the frame by a fractional number of pixels.

    A hand does not move in whole pixels, and the difference is visible: an integer shift gives a
    stepped, mechanical shake, while a fractional one gives a smooth movement.  Bilinear sampling is
    enough here because the shift is small -- at the amplitudes a hand produces, the interpolation
    error is far below the sensor noise the frame is about to receive.
    """
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return rgb
    h, w = rgb.shape[:2]
    ys = np.arange(h, dtype=np.float32)[:, None] - dy
    xs = np.arange(w, dtype=np.float32)[None, :] - dx
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - y0, 0.0, 1.0)[:, :, None]
    fx = np.clip(xs - x0, 0.0, 1.0)[:, :, None]
    a = rgb[y0, x0]
    b = rgb[y0, x1]
    c = rgb[y1, x0]
    d = rgb[y1, x1]
    top = a * (1.0 - fx) + b * fx
    bot = c * (1.0 - fx) + d * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


class CameraStage(Stage):
    name = "camera"
    layers = ("camera",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()
        # the loops themselves, created once and kept: their whole purpose is to take time, so
        # re-creating them on a parameter change would defeat the time constant they express
        self._ae = ExposureLoop(target=self.ae_target, tau_s=self.ae_tau_s,
                                limit_stops=self.ae_limit_stops,
                                presettle_s=self.autopresettle_s)
        self._awb = WhiteBalanceLoop(tau_s=self.awb_tau_s, limit=self.awb_limit,
                                     presettle_s=self.autopresettle_s)
        self._af = FocusLoop(tau_s=self.af_tau_s)
        self._weights = None
        self._bumps = None
        self._sites = None
        self._zoom_softness = 0.0
        self._defocus = 0.0
        self._af_offset = 0.0
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
        # --- automatic control loops -------------------------------------------
        # A handycam has no operator setting exposure; it has a loop that measures the
        # picture and moves a control, with a time constant, and gets it slightly wrong.
        # These are the only mechanisms in the model that carry state forward, because they
        # read the picture -- see the module docstring in codecs/autocontrol.py.
        self.ae_enabled = float(cfg.get("ae_enabled", 0.0)) > 0.5
        self.ae_target = float(cfg.get("ae_target", 0.46))
        self.ae_tau_s = float(cfg.get("ae_tau_s", 1.1))
        self.ae_limit_stops = float(cfg.get("ae_limit_stops", 2.0))
        self.awb_enabled = float(cfg.get("awb_enabled", 0.0)) > 0.5
        self.awb_tau_s = float(cfg.get("awb_tau_s", 1.6))
        self.awb_limit = float(cfg.get("awb_limit", 0.42))
        self.af_enabled = float(cfg.get("af_enabled", 0.0)) > 0.5
        self.af_tau_s = float(cfg.get("af_tau_s", 0.9))
        self.autopresettle_s = float(cfg.get("autopresettle_s", 0.0))
        self.wb_temp = float(cfg.get("white_balance_temp", 0.0))
        self.sharpness = float(cfg.get("sharpness", 1.0))
        # **All the lens's parameters in one object**, so eight floats cannot be passed in the wrong order.
        self.lens = LensSpec.from_params(cfg)
        # **Field curvature**: a simple lens images a flat subject onto a curved surface, so the plane in
        # focus at the centre is not the plane in focus at the corners.  The blur is a function of the
        # *field radius* -- zero at the centre and rising outward -- because the point-spread function
        # grows with the square of the field angle.
        self.field_curvature = float(cfg.get("field_curvature", 0.0))
        # **Longitudinal chromatic aberration**: the wavelengths focus at different distances along the
        # axis, so at the sensor one is in focus and the others are not.  Unlike the lateral kind this is
        # per-channel *defocus* with no displacement, so it softens an edge in place rather than
        # separating the channels -- a different signature, which is why the two are asserted differently.
        self.longitudinal_ca = float(cfg.get("longitudinal_ca", 0.0))
        self.motion_blur = float(cfg.get("motion_blur", 0.0))
        # --- the operator's hand ------------------------------------------------
        # Shake is the camera body moving, so it is applied before the sensor sees anything.
        self.shake_intensity = float(cfg.get("shake_intensity", 0.0))
        self.bump_rate_hz = float(cfg.get("bump_rate_hz", 0.0))
        self.bump_amount = float(cfg.get("bump_amount", 0.0))
        self.zoom_travel = float(cfg.get("zoom_travel", 0.0))
        self.zoom_kind = str(cfg.get("zoom_kind", 1.0))
        self.smear = float(cfg.get("ccd_smear", 0.0))
        self.halation = float(cfg.get("halation", 0.0))
        # --- the note's optical and sensor mechanisms ---------------------------
        self.lens_fog = float(cfg.get("lens_fog", 0.0))
        # **The overflow drain's dark rim.**  A photosite whose well overflows has its excess charge
        # removed by the anti-blooming structure, which pulls the neighbouring sites *below* their own
        # exposure -- so the ring just outside a blown highlight is darker than the background.  Measured
        # without it: the ring sat 0.012 *above* the ceiling and the profile fell monotonically from the
        # core, so nothing in the model could produce a rim at all.
        self.overflow_drain = float(cfg.get("overflow_drain", 0.0))
        self.overflow_width = float(cfg.get("overflow_width", 2.0))
        self.bloom_amount = float(cfg.get("bloom_amount", 0.0))
        self.bloom_threshold = float(cfg.get("bloom_threshold", 0.82))
        self.dead_sites_rate = float(cfg.get("dead_sites", 0.0))
        self.stuck_fraction = float(cfg.get("stuck_fraction", 0.45))
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
        """
        The static colour cast, and the automatic loop's gains when it is running.

        The two multiply rather than replace each other: the static value is the camera's *null*
        error, which is there whether or not the loop is engaged, and the loop adds its own
        correction on top. With the loop off this is exactly the old behaviour.
        """
        gains = np.ones(3, np.float32)
        if abs(self.wb_temp) > 1e-4:
            t = float(self.wb_temp)
            gains = gains * np.array([1.0 + 0.18 * t, 1.0 + 0.02 * t, 1.0 - 0.20 * t],
                                     dtype=np.float32)
        loop_gains = getattr(self, "_awb_gains", None)
        if self.awb_enabled and loop_gains is not None:
            gains = gains * np.asarray(loop_gains, np.float32)
        if np.allclose(gains, 1.0):
            return rgb
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
        # **The sensor's saturation point, and the curve must reach it exactly.**  The curve that was here was
        # asymptotic -- it approaches 1.0 and never arrives -- so a saturated highlight left this step at 0.991:
        # measured, 504 pixels at or above 1.0 went in and zero came out.  No pixel in any picture could be pure
        # white, which is the one thing the reference shows and this model did not.
        #
        # The white point follows the range, which is what `dynamic_range` means: 2 ** ((dr - 8) / 8) is twice the
        # reference at 8 stops, 1.32x at 5 and 2.83x at 12.  A rational shoulder normalised so that
        # `comp(knee) == knee` and `comp(white) == 1.0` exactly, so both points are exact rather than fitted.
        #
        # **Nothing is clamped after this.**  The gain stages -- the AE and AWB loops -- run before it, so they
        # already act on a value that may exceed white; a hold added afterwards is what broke the colour-cast
        # check on an earlier attempt, and it was solving a problem that did not exist.
        white = float(2.0 ** ((dr - 8.0) / 8.0))
        lin = np.clip(rgb, 0.0, None)
        span = max(1e-4, white - knee)
        t = np.clip((lin - knee) / span, 0.0, 1.0)
        rise = (2.0 * t) / (1.0 + t)
        comp = np.where(lin <= knee, lin, knee + (1.0 - knee) * rise)
        comp = np.where(lin >= white, 1.0, comp)
        return comp.astype(np.float32)

    def _overflow_drain(self, rgb: np.ndarray) -> np.ndarray:
        """
        The dark ring outside a saturated highlight, from the anti-blooming structure.

        When a photosite's well fills, the excess charge is drained away -- and the drain takes it from the
        neighbours, pulling them **below** their own exposure. The ring just outside a blown highlight is
        therefore darker than the background, which is the opposite direction from a bloom and cannot be produced
        by one. Measured before this existed: the ring sat 0.012 *above* the ceiling and the profile fell
        monotonically from the core, so the model had only the outward direction.

        Three properties make it a mechanism rather than a decoration:

          * **the width is in photosites**, because the drain acts on neighbouring sites and not on a distance in
            the picture;
          * **the depth is a fraction of each site's own value**, so a bright background gets a proportionally
            deeper rim and a black one gets none -- the drain removes charge that is there, and there is none to
            remove from nothing;
          * **it cannot go below zero**, for the same reason.

        The mask is computed here from `rgb` rather than read from an attribute another method wrote, which is a
        coupling this file has already turned into three faults.
        """
        if self.overflow_drain < 1e-4:
            return rgb
        dr = float(np.clip(self.dynamic_range, 3.0, 16.0))
        white = float(2.0 ** ((dr - 8.0) / 8.0))
        # **A photosite overflows as it approaches full, not only at exactly full.**  The drain structure is
        # specified with a barrier below the well's capacity, and the practical consequence is the same as the
        # measured one: with a mask of `>= white` the ring lands on the lens's own taper, which is already
        # falling, so the depression is absorbed and nothing shows -- `drain 0.25` produced no rim at all.  A
        # trigger at 0.9 of the white point puts the ring where the taper is gentle, which is also where the
        # reference's rim sits: a separate dark band rather than a shoulder on a gradient.
        m = np.asarray(rgb).max(axis=2) >= white * 0.9
        if not m.any():
            return rgb
        # dilate by re-indexing: a square of radius w, which is one photosite's neighbourhood
        w = max(1, int(round(self.overflow_width)))
        ring = np.zeros_like(m)
        for dy in range(-w, w + 1):
            for dx in range(-w, w + 1):
                if dy == 0 and dx == 0:
                    continue
                sh = np.zeros_like(m)
                ys = slice(max(0, dy), m.shape[0] + min(0, dy))
                yd = slice(max(0, -dy), m.shape[0] + min(0, -dy))
                xs = slice(max(0, dx), m.shape[1] + min(0, dx))
                xd = slice(max(0, -dx), m.shape[1] + min(0, -dx))
                sh[yd, xd] = m[ys, xs]
                ring |= sh
        ring &= ~m
        if not ring.any():
            return rgb
        out = rgb * (1.0 - float(self.overflow_drain) * ring[..., None])
        return np.clip(out, 0.0, None).astype(np.float32)

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
        #
        # **Two faults were measured in the version this replaces.**  The ray's width was `1.5 + 4.0 * a`, where
        # `a` is the *flare strength* rather than anything to do with the ray -- so at a = 0.6 each ray was about
        # 8 px across, and a six-blade iris draws twelve of them.  Twelve 8-px-wide lines radiating from one point
        # **overlap into a disc near the centre**, which is why all 24 angular sectors carried ink at both 6 and 8
        # blades and why the blade count moved the total ink by only 8%: it was changing something invisible.
        #
        # So: the width is in *pixels* and does not scale with the flare's brightness, because a star's rays are
        # thin along their whole length; and the blade count sets the angular spacing, which is the one thing the
        # parameter is supposed to control.  The ray count is `blades` rather than `2 * blades` -- an even-bladed
        # iris gives rays along each blade and the opposite blade's ray is the same line, which is why a six-blade
        # iris shows six points and not twelve.
        rays = np.zeros((h, w), dtype=np.float32)
        ray_len = max(6.0, 0.34 * max(h, w) * (0.45 + 1.35 * a))
        dy_ax = yy - cy
        dx_ax = xx - cx
        rad = np.sqrt(dy_ax * dy_ax + dx_ax * dx_ax)
        # a thin ray, with a very slight widening outward so it reads as a streak rather than a hairline
        for k in range(blades):
            theta = 2.0 * np.pi * k / blades + 0.15
            uy, ux = float(np.sin(theta)), float(np.cos(theta))
            proj = np.abs(dy_ax * uy + dx_ax * ux)
            perp = np.abs(-dy_ax * ux + dx_ax * uy)
            width = 0.9 + 0.010 * proj
            along = np.clip(1.0 - proj / ray_len, 0.0, 1.0) ** 1.6
            across = np.clip(1.0 - perp / np.maximum(width, 1e-3), 0.0, 1.0)
            rays = np.maximum(rays, along * across)
        rays = lowpass(rays, 0.8)

        # flare is tinted by the coating stack: a cool blue-violet cast
        tint = np.array([0.85, 0.90, 1.25], dtype=np.float32)
        flare = (0.55 * ghost + 0.65 * rays) * a
        return (rgb + flare[..., None] * tint[None, None, :] * 0.55).astype(np.float32)

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
        dt = 1.0 / max(1e-6, self.ctx.fps)

        # --- the hand, before the sensor sees anything -------------------------
        # Applying the shake later would give a sharp picture with an offset, which is not the same
        # thing as a picture taken by a moving camera -- and it would leave the sensor's own motion
        # blur unrelated to the movement that caused it.
        # The zoom's own consequence: at long travel a consumer lens softens, because the elements are
        # furthest from the spacing they were designed around.  Applied through the operator's motor
        # curve rather than linearly, because that is how the control moves.  Without this the parameter
        # is read and never used -- which is the "reachable but inert" fault this project keeps finding.
        if self.zoom_travel > 1e-6:
            from ..codecs.operator import zoom_curve
            # at long travel a consumer lens softens: the elements are furthest from the spacing they
            # were designed around, and the motor's own curve is how the control gets there
            self._zoom_softness = 0.22 * zoom_curve(self.zoom_travel, "motor")
        else:
            self._zoom_softness = 0.0
        # the total defocus this frame: the loop's motor position plus the zoom's softness.  Applied as
        # a blur, not as the coating loss `_lens_wear` models -- a misfocused lens convolves the image,
        # while a worn coating scatters light and lifts the blacks, and they are different faults.
        self._defocus = float(np.clip(getattr(self, "_af_offset", 0.0) + self._zoom_softness,
                                      0.0, 1.6))

        if self.shake_intensity > 1e-6 or self.bump_amount > 1e-6:
            from ..codecs.operator import plan_bumps, shake_offset
            if self._bumps is None:
                self._bumps = plan_bumps(self.bump_rate_hz, self.bump_amount,
                                         int(getattr(self.ctx, "frame_count", 0) or 600),
                                         self.ctx.fps,
                                         derive_seed(self.ctx.seed, "camera", "bumps"))
            dx, dy = shake_offset(HAND, self.shake_intensity, buf.index, self.ctx.fps,
                                  derive_seed(self.ctx.seed, "camera", "hand"), self._bumps,
                                  width=rgb.shape[1])
            rgb = _subpixel_shift(rgb, dx, dy)

        # --- automatic control, in the order the camera makes its decisions ----
        # focus first: it is the lens, before anything else in the chain
        if self.af_enabled:
            pos = self._af.update(rgb, dt)
            # Kept separately from the lens parameters, which `_read_params` re-reads whenever a
            # keyframe
            # or the environment fires -- so writing the loop's result there would let the static
            # parameter overwrite it on those frames and the loop would stutter.
            # The loop reports a motor position; how much blur that is, is the lens's business.
            self._af_offset = 0.55 * pos

        # then the loops that move analogue gain, so that a loop which opens up also brings up
        # the sensor's noise rather than cleaning the picture afterwards
        if self.ae_enabled:
            if self._weights is None or self._weights.shape != rgb.shape[:2]:
                self._weights = centre_weights(rgb.shape[0], rgb.shape[1])
            # The loop's return value *is* the correction, so this assigns it -- and it takes it
            # from `update`, not from `state`, because the step and the read have to be one
            # expression.  Both other forms are wrong in ways that look right:
            #   `self.exposure += update(...)` uses the return but adds it to the current value,
            #     so the correction compounds: measured, +1.5 stops of need went 0.06, 0.85, 2.43
            #     and then sat at the +3.0 limit;
            #   `self.exposure = self._ae.state` reads the state from *before* the step, so the loop
            #     runs correctly and its result is never used -- measured, exactly 0.0000 on a scene
            #     needing +1.5 stops.
            self.exposure = float(np.clip(self._ae.update(rgb, dt, self._weights), -3.0, 3.0))
        if self.awb_enabled:
            self._awb_gains = self._awb.update(rgb, dt)

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
        # fog is on the front element, so light passes through it before anything else
        if self.lens_fog > 1e-4:
            from ..codecs.optics import lens_fog
            rgb = lens_fog(rgb, self.lens_fog)
        rgb = self._halation(rgb)
        rgb = self._overflow_drain(rgb)
        # charge, at the sensor: vertical because the transfer channel is vertical
        if self.bloom_amount > 1e-4:
            from ..codecs.optics import blooming
            rgb = blooming(rgb, self.bloom_threshold, self.bloom_amount)
        rgb = self._lens_flare(rgb)
        rgb = self._flare(rgb)
        # **Lens aberrations, before the camera's own sharpening.**  Aperture correction is applied by
        # the camera *after* the lens has done its work, so sharpening first would let it undo the
        # corner softening -- which is the one thing field curvature is supposed to leave visible.
        # **The lens, in the order the light meets it.**  The geometry of the glass first, then the
        # field-dependent blurs, then the wavelength errors -- and the camera's own sharpening last,
        # because aperture correction is applied *after* the lens and sharpening first would undo the
        # corner softening that field curvature exists to leave visible.
        # **The mechanisms that existed before keep their original relative order.**  `chromatic_aberration` is
        # the old `_aberration` and `longitudinal_ca` was already here, so `field_curvature` stays between them
        # exactly where it was.  The new mechanisms are placed around that sequence rather than in front of it:
        # `distortion` first because it is the geometry of the glass and everything else is defined in the frame it
        # produces, and `coma`, `astigmatism` and `spherical` after the field-dependent blur they belong with.
        #
        # Measured, putting `distortion` ahead of `field_curvature` cost 2.96e-04 in the corners even with
        # `distortion` at zero -- it returns `rgb` through a copy, so the array `field_curvature` receives has a
        # different float32 history.  A move has to preserve the *sequence*, not just the set.
        rgb = distortion(rgb, self.lens)
        rgb = chromatic_aberration(rgb, self.lens)
        rgb = field_curvature(rgb, self.lens)
        rgb = longitudinal_ca(rgb, self.lens)
        rgb = coma(rgb, self.lens)
        rgb = astigmatism(rgb, self.lens)
        rgb = spherical(rgb, self.lens)
        rgb = self._sharpen(rgb)
        if getattr(self, "_defocus", 0.0) > 1e-4:
            # a Gaussian of radius growing with defocus.  The factor is chosen so that full defocus
            # (1.0) is a clearly soft picture and the loop's typical steps are visible but gentle.
            rgb = lowpass(rgb, 0.6 + 4.5 * self._defocus, axis=None)
        rgb = self._lens_wear(rgb)
        rgb = self._color_science(rgb)
        # The sensor's own readout, at the very end.  A stuck site is a *readout* defect, so it happens
        # after the charge has accumulated, after the analogue gain and after the camera's own signal
        # processing -- which means after `_sharpen` in particular.  Writing it earlier looked right and
        # did nothing: an unsharp mask computes `0 * (1 + k) - k * neighbour_mean` at a dead site and pulls
        # it straight back to the neighbourhood average.  Measured before this move: the site was set to
        # 0.0000 and the stage's cached output read 0.5199 at the same position.
        if self.dead_sites_rate > 1e-6:
            from ..codecs.optics import apply_defect_sites, plan_defect_sites
            if self._sites is None:
                self._sites = plan_defect_sites(
                    self.dead_sites_rate, self.stuck_fraction,
                    rgb.shape[0], rgb.shape[1],
                    derive_seed(self.ctx.seed, "camera", "dead_sites"))
            rgb = apply_defect_sites(rgb, self._sites)

        return FrameBuffer(np.clip(rgb, 0.0, 2.0).astype(np.float32), None,
                           buf.index, buf.time, buf.std, dict(buf.meta))

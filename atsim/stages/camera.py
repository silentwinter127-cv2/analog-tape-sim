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

import os

import numpy as np

from ..codecs.operator import HAND
from ..codecs.autocontrol import (ExposureLoop, FocusLoop, WhiteBalanceLoop,
                                  centre_weights)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import ValueNoise3D, bandlimited_noise, derive_seed, rng_for
from ..core.signal import gaussian_kernel, lowpass

#: The luma vector, used by the tone mechanisms.  Rec.601 weights, as elsewhere in the model.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
from ..codecs.lens import (LensSpec, field_curvature, longitudinal_ca, chromatic_aberration,
                           coma, astigmatism, spherical, distortion, source_glow)


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
        # --- the night-vision modes ------------------------------------------------
        # Physical reconstruction rather than a colour grade: the illuminator, the removal of the IR-cut
        # filter, the monochrome green display, and frame accumulation.  Built against the reference clip
        # `索尼F717夜视测试`, segmented into its four modes and measured per segment.
        from ..codecs.nightvision import NightVisionSpec
        self.nv = NightVisionSpec.from_params(cfg)
        #: **The sensor's saturation point, in stops below the range's own white.**  Zero leaves `_roll_off` as it was.
        #: See the comment there: the curve clips correctly, but at the default range its white point is 1.0 while the
        #: picture's maximum is 0.87, so nothing ever reaches it and the model can never show a blown highlight.
        #
        #: **And it applies only when the night-vision mode is on**, which is a measurement rather than a convenience.
        #: Lowering the white point is a statement about *this* sensor's saturation, and the mode it was fitted to is
        #: NightShot; with it applying always, the suite caught that **the scanline structure began costing light**
        #: (the frame's mean varied by 0.00275 across the four spot sizes) -- a property verified earlier in this work
        #: and measured on a configuration with no night vision in it at all.  So the white point moves where the
        #: measurement says it should and nowhere else.
        sat = float(cfg.get("saturation_stops", 0.0))
        self.saturation_stops = sat if float(cfg.get("nv_mode", 0.0)) >= 0.5 else 0.0
        #: **The night-vision mode's own exposure offset, and it must NOT be the global `exposure`.**  Measured, the
        #: highlight calibration wants a **-0.10 stop** trim to bring the mean onto the footage's.  **Setting the global
        #: `exposure` default to -0.10 broke a check that has nothing to do with night vision** -- "a cold machine renders
        #: differently from a warm one", whose largest mean difference fell to 0.0008 -- because `exposure` is the camera's
        #: own control and the AE loop, the spot sizes and the thermal drift are all measured through it.  **A trim
        #: belonging to one mode belongs to that mode**, exactly as the saturation point does.
        #:
        #: **Read from the config rather than hard-coded**, so it can be tuned: the first version of this line was a
        #: literal -0.10 and two subsequent attempts to sweep it changed nothing, which is this work's own "declared,
        #: reachable, inert" fault class -- a parameter that exists and cannot be reached.
        trim = float(cfg.get("nv_exposure_trim", -0.10))
        self.nv_exposure_trim = (trim if float(cfg.get("nv_mode", 0.0)) >= 0.5 else 0.0)
        # **The near-IR material labelling, and whether this machine can do it at all.**  `note.txt`'s second mechanism
        # asks for near-IR albedo per material rather than an RGB-to-grey conversion, and a vision-language model can
        # supply it; where none is installed the estimate is used, **which is not a degraded mode but the documented
        # fallback**.  The labels are computed once per clip and cached, because the source is a still or a clip whose
        # materials do not change frame to frame.
        self.nir_material = bool(round(float(cfg.get("nv_nir_material", 1.0))))
        #: the checkpoint, if one is named.  **An environment variable rather than a schema parameter**, because every
        #: parameter in this schema is numeric and a path is not -- and because naming a model is a property of the
        #: machine, not of the look being simulated.
        self.nir_model_dir = os.environ.get("ATSIM_NIR_MODEL") or None
        self.nir_margin_floor = float(cfg.get("nv_nir_margin_floor", 0.02))
        #: **How fine the material labelling is, in PIXELS rather than in cells, and that distinction is the fix for
        #: a block artefact the user reported.**
        #:
        #: It used to be a cell count (`nv_nir_grid`, default 8), and a cell count is not a physical quantity: at the
        #: 533x400 the figures were calibrated on, 8x8 means **66x50 px per cell**, while on the clip's own
        #: **1440x1080** the same 8 means **180x135 px** -- a 24000-pixel block, 7.5x the area, and the user's
        #: photograph showed exactly that: large light and dark rectangles over the case.  **A material's spatial
        #: scale is a property of the scene, so the label grid has to be tied to the picture's pixels or the same
        #: parameter means a different physical size on every input.**
        #:
        #: The default cell is **66 px**, which reproduces the physical scale the material table was calibrated at:
        #: `533/8 = 66` and `400/8 = 50`, so 66 on the shorter edge.  Measured on 720x540, a 66 px cell gives 11x8
        #: cells and takes the labelling to about the same order as the 16x16 measurement (16.5 s), and the analysis
        #: runs **once per clip**, so that is the whole cost.
        self.nir_cell_px = float(np.clip(float(cfg.get("nv_nir_cell_px", 66.0)), 16.0, 512.0))
        self.nir_grid_max = int(np.clip(round(float(cfg.get("nv_nir_grid_max", 32.0))), 2, 64))
        #: the lattice actually used, `(grid_h, grid_w)`, filled by `_grid_for` once the frame's size is known
        self.nir_grid: tuple[int, int] = (8, 8)
        self.nir_label_source: str | None = None
        self.nir_depth_source: str | None = None
        #: whether to infer a depth map, so the illuminator's falloff can be about distance rather than a radius
        self.nir_depth = bool(round(float(cfg.get("nv_nir_depth", 1.0))))
        #: **The DSP's output level.**  Gain and black level, in the order the camcorder applies them -- see
        #: `_dsp_level` for why two quantities are needed rather than one brightness, and why both default neutral.
        #: **These are re-read from the spec in `_night_vision`**, because the MODE is what sets them: the values that
        #: put the NightShot material's percentiles on target belong to that mode, not to the camera, and using them
        #: as the schema default darkened every ordinary visible-light render.
        self.dsp_gain = float(np.clip(cfg.get("dsp_gain", 1.0), 0.1, 4.0))
        self.dsp_black = float(np.clip(cfg.get("dsp_black", 0.0), -0.5, 0.5))
        self._nir_albedo: np.ndarray | None = None
        self._nir_depth: np.ndarray | None = None
        self._nir_failed = False
        #: The recent post-exposure frames, so a mode that accumulates has frames to sum.  The module
        #: returns the mean, so the picture stays at the same scale as a single frame -- which is what makes
        #: the noise *fall* rather than the brightness rise.
        self._nv_history: list = []
        self.exposure = float(cfg.get("exposure", 0.0))
        # **The night-vision exposure bias is added here, not applied as a gain at the end.**  The stage's own
        # exposure feeds the auto-exposure loop, the roll-off, the gain stages and the white balance in a fixed
        # order; a multiplier bolted on afterwards would lighten the picture without it ever having been exposed
        # differently -- no shift in the highlight roll-off and nothing for the AE loop to react to.  Super
        # NightShot's character *is* a different exposure, so the bias belongs in the value.
        self._exposure_base = self.exposure
        # the mode's own bias, added to whatever the user set -- through the same value, so it is exposed
        # rather than painted on.  Zero when the mode is off.
        self.exposure = self._exposure_base + self._nv_exposure_bias()
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
        #: How much the halation's tint follows the **blown region's own colour** instead of the fixed
        #: lens tint.  A scattering halo keeps the source's spectrum, so this is the physical behaviour;
        #: it is a blend because the fixed tint is also a real lens property.
        self.halation_source_colour = float(cfg.get("halation_source_colour", 0.0))
        #: **Midtone and shadow compression that leaves highlights at unity gain.**  Raises the contrast of a
        #: bright thin line against its surroundings without adding light to the line, which is the only way
        #: to do it without diluting the line's colour -- adding light raises the line and its surroundings
        #: together and so reduces their difference.
        self.shadow_depth = float(cfg.get("shadow_depth", 0.0))
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
        self.spike_spread = float(cfg.get("spike_spread", 0.6))
        # per-channel focus error of the *flare*: the out-of-focus aperture image is larger in
        # blue than in red, so a bright source wears a coloured halo rather than a coloured core
        self.flare_defocus = float(cfg.get("flare_defocus", 0.5))
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

    def _nv_source_frames(self, n: int, index: int | None) -> list:
        """
        The last `n` source frames, for a mode that accumulates.

        **Read from the source rather than kept on the stage**, because each frame is rendered by its own call and a
        list on the stage would only ever hold the current one. The source keeps its own small LRU, so asking for a few
        frames costs little, and the camera already reads `index - 1` for its shutter for the same reason.

        **`index` is required rather than defaulted.**  The first version read `getattr(self.ctx, "index", 0)`, and
        `RenderContext` has no `index` -- so it silently used 0, the window clipped to a single frame, and accumulation
        did nothing while looking correctly wired. Measured, the noise was identical at 1, 4 and 8 frames. A missing
        attribute that yields a plausible value instead of raising is how a mechanism ends up inert.

        The window is clipped to the frames that exist -- an accumulation of 16 on frame 2 returns 2 frames, not an
        error -- and `len` is reported by the caller so a request larger than the sequence is visible rather than
        silently truncated.
        """
        src = getattr(self.ctx, "source", None)
        if src is None or index is None:
            return []
        idx = int(index)
        out = []
        for k in range(n - 1, -1, -1):
            j = idx - k
            if j < 0:
                continue
            try:
                out.append(np.asarray(src.frame(j), np.float32))
            except Exception:                                                   # noqa: BLE001
                continue
        return out

    def _night_vision(self, rgb: np.ndarray, stage: str, index: int | None = None) -> np.ndarray:
        """
        The night-vision terms, called twice per frame because they belong in two different places.

        `stage="scene"` runs **before the sensor**: the illuminator's light reaches the scene and comes back, and the
        cut filter's removal lets the sensor admit near-infrared. `stage="display"` runs **after every sensor term**:
        the monochrome green view, which is the viewfinder's output and therefore the last thing to happen.

        Kept as one method with a stage argument rather than two, so the physical ordering is visible in one place
        instead of being implied by two call sites far apart in `apply`.
        """
        from ..codecs.nightvision import (accumulate, add_illuminator, ir_highlights,
                                           mode_of, monochrome_green, read_noise,
                                           shutter_integrate)
        if mode_of(self.nv) == "off":
            return rgb
        # **The mode's own settings.**  The stage drives the module's pieces directly rather than through
        # `apply_mode`, so it has to resolve the mode's values too -- and that is the right place for it, because the
        # stage is what sequences the pieces.  Without this, selecting Color Slow Shutter returned a green picture:
        # measured through the API, ratio 0.330:0.383:0.287, because `nv_green_mix` defaults to 1 and that mode's
        # requirement is 0.
        from ..codecs.nightvision import mode_spec
        self.nv = mode_spec(self.nv)
        # **The mode's DSP level, adopted onto the stage.**  `mode_spec` resolves what the mode settles on -- the
        # green display, the frame count, and now the gain and black level, which come from NightShot material and
        # must not be applied to a mode that is not NightShot.  `_dsp_level` runs at the very end of the chain, after
        # every call to `_night_vision`, so reading them here is reading them once per frame in time.
        self.dsp_gain = float(np.clip(float(getattr(self.nv, "dsp_gain", 1.0)), 0.1, 4.0))
        self.dsp_black = float(np.clip(float(getattr(self.nv, "dsp_black", 0.0)), -0.5, 0.5))
        if stage == "scene":
            # **The analysis and the attachment are two steps, and an earlier version of this folded them into one and
            # made the distance term inert.**  The analysis is expensive -- a vision model, once per clip -- while the
            # attachment is two assignments that must happen **after** `mode_spec`, which runs every frame and rebuilds
            # the spec.  With one combined method that returned early once the analysis was cached, the assignments were
            # skipped on every frame including the first, because the cache was filled *by* that same call: measured,
            # `nv_nir_depth` on and off gave bit-identical output (luma 0.2390, contrast 0.2004) and the correlation
            # between rendered luminance and inferred depth was **+0.148 either way**, where a working term drives it
            # negative.
            self._analyse_scene_nir(rgb)
            self.nv.ir_albedo = self._nir_albedo
            self.nv.ir_depth = self._nir_depth
            # **Accumulation comes first, while the picture is still at the scene's own level.**  It ran in the
            # display stage initially -- the last thing to execute -- where it replaced a *processed* picture (mean
            # 0.4886) with the mean of several *source* frames (mean 0.1768), a factor of 3.66.  Then it ran after
            # `add_illuminator`, which left the two at different levels again and a scalar level match was needed to
            # reconcile them -- and that match scaled the whole frame and clipped it (0.409 where a mean must give
            # 0.5).  **Both faults were the same mistake: summing frames at one level and using the result at
            # another.**  Summing them here removes the mismatch instead of correcting for it.
            #
            # It is also the physically right place.  **The sensor integrates the light it receives**, and a steady
            # illuminator's contribution is simply part of that light -- so integrating the scene and then adding the
            # lamp's field is equivalent to integrating the lamp along with everything else.  What matters for the
            # mechanism is that the frames **differ**, so the uncorrelated noise falls as sqrt(N) while the signal
            # holds.
            n = int(round(float(np.clip(self.nv.accumulate, 1.0, 64.0))))
            # **The slow shutter is its own mechanism, not a longer accumulation.**  `accumulate` sums separate
            # exposures, which makes the picture *cleaner*; a shutter held open smears a moving subject and does not
            # bring it back.  `shutter_frames` therefore drives a separate path.
            #
            # **One window serves both, and an earlier version of this branched on `accumulate` alone** -- so with
            # `shutter_frames` added and defaulting to 1, the `else` branch was taken and **`accumulate` never ran**:
            # measured, temporal noise at `accumulate = 8` was **0.07968 against a single frame's 0.07968, i.e. no
            # reduction at all**, where 1/sqrt(8) = 0.354 is what Super NightShot exists to produce.  **A new mechanism
            # introduced a regression in the old one**, which is the fourth time in this work that adding a term has
            # made an existing one inert.  So the window length is the larger of the two and both mechanisms are
            # applied to it, in the physical order: the sensor integrates the light first, the read-out sums exposures.
            shut = int(round(float(np.clip(self.nv.shutter_frames, 1.0, 64.0))))
            want = max(n, shut)
            out = rgb
            if want > 1:
                stack = self._nv_source_frames(want, index)
                if len(stack) > 1:
                    # **One exposure held open, then the separate exposures summed.**  The integration is applied to
                    # the frames because a long exposure's smear is in the light the sensor collects, before anything
                    # is read out of it.
                    if shut > 1:
                        stack = [shutter_integrate(stack, self.nv)]
                    out = accumulate(stack, self.nv) if n > 1 else stack[0]
            out = add_illuminator(out, self.nv)
            out = ir_highlights(out, self.nv)
            return out
        #: **The display's own chroma noise, and it is not the sensor's.**  The chain above ends with
        #: `monochrome_green`, which maps every pixel onto the display's primary ratios and therefore **removes chroma
        #: noise by construction** -- measured on the camcorder's own frame, the dark band's hue spread arrives at
        #: 0.212, reaches 0.951 after the sensor's read noise, and then comes out of the green matrix as **0.000**.
        #: So a chroma term added on the sensor side is deleted by the next step; that is the fifth time in this work a
        #: value has been produced and then annihilated before use.
        #:
        #: **And it belongs here physically.**  A camcorder's read noise is one number per photosite, sensor-side.
        #: What makes a *near-black* pixel's hue wander is the chroma channel's signal-to-noise ratio, and in a
        #: NightShot camera the chroma path is downstream of the mono conversion -- the display's own channels are
        #: where colour stops being defined.  **It is weighted by darkness rather than by brightness**, which is the
        #: opposite of shot noise: the first version reused `read_noise`, whose shot term grows as `sqrt(lum)`, and
        #: measured, the bright band's hue spread went to **1.314 against the footage's 0.211** -- noise where the
        #: picture has none.  A display's chroma S/N is worst where the signal is smallest.
        #:
        #: **The magnitude is `nv_chroma_noise` alone, and decoupling it from `nv_read_noise` is the point.**  The
        #: sensor-side term scales its chroma component by `read_noise` (`chan = normal(0, a * cn)` inside
        #: `read_noise()`), which is self-consistent for a sensor but wrong here: **a display's chroma channel does not
        #: have a noise floor of zero because the sensor's read noise is zero.**  They are two amplifiers.  The first
        #: version kept the coupling and the term was therefore skipped entirely at the default -- measured, the dark
        #: band stayed at 0.037 with the term "on".
        out = monochrome_green(rgb, self.nv)
        cn = float(np.clip(getattr(self.nv, "chroma_noise", 0.0), 0.0, 1.0))
        if cn > 1e-6:
            lum = np.clip(out @ np.array([0.299, 0.587, 0.114], np.float32), 0.0, 1.0)
            # darkness weight: one at black, falling to zero by mid-grey
            wdark = np.clip(1.0 - lum / 0.35, 0.0, 1.0)[..., None]
            rng = np.random.default_rng(0x9E3779B9 ^ ((int(index) if index is not None else 0) * 2654435761 & 0xFFFFFFFF))
            chan = rng.normal(0.0, 1.0, out.shape).astype(np.float32)
            # zero mean across the channels, so it is chroma and not brightness -- the same construction the
            # sensor-side term uses, and the reason it can be added without moving the level
            chan = chan - chan.mean(axis=-1, keepdims=True)
            out = np.clip(out + chan * (0.35 * cn) * wdark, 0.0, 1.0)
        return out

    def _grid_for(self, h: int, w: int) -> tuple[int, int]:
        """
        The material-label lattice whose cells are **as close to `nir_cell_px` square as the frame allows**.

        **Why the cell size is the parameter and the cell count is not.**  A 7.5x step in `IR_ALBEDO` at a cell
        boundary is stamped onto the picture as a band, so the size of a cell is how wide that band's plateau is --
        a physical quantity, tied to how large a patch of one material is in the scene.  A cell *count* is not: the
        same 8 means 66x50 px at 533x400 and **180x135 px at 1440x1080**, which is why the user's photograph of a case
        showed rectangles a sixth of the frame across while the figures that "verified" the mechanism were taken at a
        quarter of that area.

        **Both directions are derived, because a cell has to be square.**  A single count makes the cell `w/n` by
        `h/n`, so a 4:3 frame gets cells a third taller than they are wide -- and then the *shape* of the quantisation
        depends on the aspect ratio, which is not a physical statement about the scene.  Deriving each direction from
        the same pixel size gives square cells at every aspect ratio, and the lattice simply has more cells along the
        longer edge.

        The count is clamped by `nir_grid_max` -- 32 is where the measured labelling cost reaches about a minute and
        the classifier's unsure fraction has stopped improving (9% at 8x8, 11% at 16x16, 11.5% at 32x32).
        """
        cell = max(16.0, float(self.nir_cell_px))
        gw = int(round(w / cell))
        gh = int(round(h / cell))
        top = max(2, self.nir_grid_max)
        return int(np.clip(gh, 2, top)), int(np.clip(gw, 2, top))

    def _analyse_scene_nir(self, rgb: np.ndarray) -> None:
        """
        Label the scene's materials once per clip and put the resulting per-pixel near-IR albedo on the spec.

        **This is the model's fallback boundary, and it is explicit rather than implicit.**  Three things can make the
        labelling unavailable, and **all three leave the picture exactly as it was** -- the estimate from brightness,
        which is the documented behaviour rather than a failure:

          * the switch is off;
          * no vision model is installed, which `material.available()` reports without raising, **because a user without
            one is a supported configuration** -- the note's own wording is that the material table is what a filter
            cannot fake, not that a model is required;
          * an analysis that raises for any reason, after which it is not retried (the flag is set), so one bad frame
            cannot cost a render per frame.

        **And the provenance is recorded** (`nir_label_source`), so any figure can say whether the albedo came from a
        model or an estimate -- the same discipline the chain's stage traces follow.
        """
        if self._nir_failed:
            return
        # **The guard watches the right cache.**  It watched `_nir_albedo`, which the *labelling* fills -- so with
        # `nv_nir_material = 0` that never filled and the analysis re-ran every frame, and with it on the first frame
        # filled it and **the depth was never analysed at all**.  Measured, that made `nv_nir_depth` completely inert:
        # the field came out identical in both settings (min 0.05160, max 0.06731) and the correlation between
        # luminance and inferred depth was **+0.148 either way**, where a working term must drive it negative.
        # **So each cache is guarded by its own flag**, and the two analyses are independent, which is what they are.
        need_albedo = self.nir_material and self._nir_albedo is None
        need_depth = self.nir_depth and self._nir_depth is None
        if not (need_albedo or need_depth):
            return
        # **The lattice is resolved from THIS frame's size, because the cell size is what is physical.**  Doing it here
        # rather than in `_read_params` means a preview at 0.35 scale and a full render get the same *physical* cell
        # size instead of the same cell count -- which is the whole point of the change, and the same class of fault as
        # a bandwidth expressed in pixels instead of MHz.
        grid_h, grid_w = self._grid_for(rgb.shape[0], rgb.shape[1])
        self.nir_grid = (grid_h, grid_w)
        try:
            from ..codecs import material
            from ..codecs.nightvision import albedo_map
        except Exception:                                                       # noqa: BLE001
            self._nir_failed = True
            return
        if not material.available(self.nir_model_dir):
            # **the fallback, taken deliberately**: no model here, so the brightness estimate stands
            self._nir_failed = True
            if self.nir_label_source is None:
                self.nir_label_source = "estimate (no vision model found)"
            return
        if need_albedo:
            try:
                labelled = material.analyse(rgb, model_dir=self.nir_model_dir,
                                            margin_floor=self.nir_margin_floor,
                                            grid_h=grid_h, grid_w=grid_w)
            except Exception:                                                   # noqa: BLE001
                labelled = None
            if labelled is None:
                self.nir_label_source = "estimate (labelling failed)"
            else:
                h, w = rgb.shape[:2]
                lum = np.clip(np.asarray(rgb, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), 0.0, 1.0)
                # the fallback the unlabelled cells take: the estimate's own shape, normalised the same way
                est = np.power(lum, 0.55)
                # **The guide is the picture's own luminance**, so the albedo's boundaries land where the image has
                # boundaries rather than at the cell edges the lattice happens to have.  See `_resample`.
                self._nir_albedo = albedo_map(h, w, labelled, fallback=est, guide=lum)
                classes = labelled["classes"]
                counts = {}
                for v in np.asarray(labelled["labels"], np.int32).ravel():
                    key = classes[v] if v >= 0 else "unsure"
                    counts[key] = counts.get(key, 0) + 1
                self.nir_label_source = "%s: %s" % (
                    labelled["source"],
                    ", ".join("%s %d" % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1])))
        if need_depth:
            try:
                dep = material.analyse_depth(rgb, grid=min(grid_h, grid_w), model_dir=self.nir_model_dir)
            except Exception:                                                   # noqa: BLE001
                dep = None
            if dep is not None:
                self._nir_depth = dep["depth"]
                self.nir_depth_source = ("%s: relative depth %.2f..%.2f (mean %.2f), confidence %.2f..%.2f"
                                         % (dep["source"], float(dep["depth"].min()), float(dep["depth"].max()),
                                            float(dep["depth"].mean()), float(dep["confidence"].min()),
                                            float(dep["confidence"].max())))

    def _nv_exposure_bias(self) -> float:
        """
        The extra exposure a night-vision mode asks for, in the model's exposure units.

        **This is how Super NightShot and Color Slow Shutter get brighter** -- through the same exposure path as every
        other setting, so the picture meets the same roll-off, gain and white-balance machinery. A mode off returns
        nothing, so the parameter is inert unless a mode is selected.
        """
        from ..codecs.nightvision import mode_of
        m = mode_of(self.nv)
        if m == "off":
            return 0.0
        # **The calibration trim, and it has to be added even when the mode's own steps are zero.**  Measured, the
        # highlight calibration on the new photograph wants a **-0.10 stop** trim: with it the mean reads **0.2480**
        # against the footage's 0.2346, without it 0.2657.  **Setting the GLOBAL `exposure` default to -0.10 broke a
        # check with nothing to do with night vision** -- "a cold machine renders differently from a warm one", whose
        # largest mean difference fell to 0.0008 -- because `exposure` is the camera's own control and the AE loop, the
        # spot sizes and the thermal drift are all measured through it.  **A trim belonging to one mode belongs to that
        # mode**, exactly as the saturation point does, so it lives here and nowhere else.
        return float(self.nv.exposure_steps) + float(getattr(self, "nv_exposure_trim", 0.0))

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
        # **The white point can also be brought DOWN, which is what makes the sensor saturate at all.**  Measured: the
        # model's brightest pixel is **0.8686 and nothing anywhere in the frame is clipped**, where the footage reaches
        # **1.0 with 0.66% of the frame at or above 0.98**.  And this curve is not the cause of that -- it clips
        # correctly, at `white`.  **The cause is that nothing reaches `white`**: at the default `dynamic_range` of 8 the
        # white point is exactly 1.0 while the picture's own maximum after exposure and white balance is 0.87, so the
        # `lin >= white` branch never fires and the shoulder sits above the content it is meant to compress.
        #
        # **So the saturation point is exposed directly, in stops below the range's own white.**  `nv_saturation_stops`
        # of 0 leaves the curve exactly as it was; more than 0 lowers the white point by that many stops, so a highlight
        # that would have ended at 0.95 is pinned at 1.0 instead and the sensor blows out where a real one does.  **A
        # sensor's saturation point is a property of the sensor and not of its dynamic range**, which is why it is a
        # separate control rather than a different `dynamic_range`.
        sat_stops = float(np.clip(getattr(self, "saturation_stops", 0.0), 0.0, 3.0))
        if sat_stops > 1e-4:
            white = white * float(2.0 ** (-sat_stops))
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
        # **Per channel, because the excess charge is a property of each well.**  The red channel's well fills
        # first in a tungsten-lit highlight, then green's, then blue's -- so the charge that spills down the
        # vertical transfer channel carries a per-channel excess, and because blue has the most left to give the
        # streak reads **cyan-blue**.  The version this replaces reduced the highlight to `hot.mean(axis=0)`, one
        # scalar, and added it back with a single tint: all three channels got the same vertical profile, so the
        # smear was **achromatic at its root** and a blown white lamp produced a neutral grey stripe.  No parameter
        # could make it blue in the direction the physics requires, because the blue end has to come from blue
        # having more excess rather than from a tint that happens to be blue.
        #
        # The thresholds are the same shape as the axial-CA discussion elsewhere: the *order* is the physics (red
        # clips first), and the three numbers are the filter transmissions.
        clips = np.array([0.78, 0.86, 0.94], dtype=np.float32)[None, None, :]
        hot = np.clip(rgb - clips, 0.0, None)
        if not np.any(hot > 0.0):
            return rgb
        # **Each channel keeps its own column profile.**  The line here used to be `hot.mean(axis=0)`, which
        # reduced the three channels back to one -- so the comment above claimed per-channel behaviour that the
        # code did not have, which is the fault this project keeps finding.  Sharper than a lens flare, because
        # this is charge transfer rather than scattering.
        cols = hot.mean(axis=0, keepdims=True)
        cols = lowpass(cols, 3.0, axis=1)
        # a wavelength-dependent transfer efficiency, because blue is transferred less efficiently along the
        # readout and lingers: the reason a night smear reads cyan rather than the grey a pure threshold gives
        lead = np.array([0.80, 1.00, 1.30], dtype=np.float32)[None, None, :]
        return (rgb + (0.85 * self.smear) * cols * lead).astype(np.float32)

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
        # **The halo's colour is the source's, not a constant.**  Scattered light keeps the spectrum of what it came
        # from, so a halo around a blue-white line should be blue-white.  The fixed tint below is a real lens
        # property, but as a *constant* it dilutes a small chroma difference -- measured, a thin blue line under a
        # halation came back at B-R +0.000 against its own +0.024, and no value of the existing parameters recovered
        # it, because raising saturation cannot restore a difference that neutral light has divided away.
        #
        # So the tint is taken from the blown region itself, normalised so its **luminance is unchanged**: a scene
        # with no blown source, or a neutral one, therefore behaves exactly as before, and only a coloured source
        # moves.  Built from the same `hot` mask, per channel, so it is the colour of what is actually blowing out.
        blown = rgb * hot[..., None]
        mass = float(hot.sum())
        if mass > 1e-6:
            mean_rgb = blown.reshape(-1, 3).sum(axis=0) / mass
            g = float(mean_rgb[1])
            if g > 1e-4:
                src_tint = np.clip(mean_rgb / g, 0.55, 1.75).astype(np.float32)
            else:
                src_tint = np.ones(3, np.float32)
        else:
            src_tint = np.ones(3, np.float32)
        lens_tint = np.array([1.05, 0.88, 1.05], dtype=np.float32)
        mix = float(np.clip(self.halation_source_colour, 0.0, 1.0))
        tint = (1.0 - mix) * lens_tint + mix * src_tint
        return (rgb + (a * 0.52) * halo[..., None] * tint[None, None, :]).astype(np.float32)

    def _shadow_depth(self, rgb: np.ndarray) -> np.ndarray:
        """
        Compress the midtones and shadows while leaving the highlights at unity gain.

        **Why this exists.** The pole's blue in the reference scene is diluted by the *total* neutral light added
        around it, so the glow terms are the wrong lever: turning them down costs the line's contrast, because the
        contrast is the pole's brightness **minus its surroundings'** and a glow raises both. Measured, reducing the
        glow moved the contrast from 0.2185 to 0.2069 -- the wrong way.

        **So the contrast is raised from the other side of the subtraction**, by darkening what is around the line
        instead of brightening the line. That adds no light to the pole at all, so nothing dilutes its colour.

        The gain is one-sided and **pinned to unity at white**:

            gain(L) = 1 - a * (1 - L)**2

        which is exactly **1 at L = 0 and at L = 1**. That shape is the whole design constraint: if the curve touched
        the highlights it would alter the pole's own colour and reintroduce the problem it is meant to solve, and if it
        touched black it would lift the shadows rather than deepen them.

        The first attempt carried a trailing `* L` and dipped only to **0.859 at a = 1.0** -- a maximum of 14% of
        compression, too small to move a contrast measurement, which would have made the mechanism look inert for the
        same reason a mechanism that never fires does. Dropping it keeps both endpoints exact and deepens the dip to
        **0.4375 in the shadows and 0.75 in the midtones**.
        """
        if self.shadow_depth < 1e-4:
            return rgb
        a = float(self.shadow_depth)
        lum = np.clip(rgb @ LUMA, 0.0, 1.0)
        gain = 1.0 - a * (1.0 - lum) ** 2
        return (rgb * gain[..., None]).astype(np.float32)

    def _dsp_level(self, rgb: np.ndarray) -> np.ndarray:
        """
        The DSP's output level: **gain, and the black level it decides on.**

        **Why these two and not one "brightness".**  Measured against the clip's own frames at native resolution, the
        model's picture is wrong in a specific way, and it is not a brightness error:

            percentile     the model    the real frames    ratio
            p1  (black)      0.0162        0.0368          0.44x   <- the model's black is TOO LOW
            p50 (mid)        0.2572        0.1619          1.59x   <- and its mid is too HIGH
            p99 (white)      0.9067        0.8451          1.07x

        **A single gain cannot do both, and neither can a gamma** -- `x ** p` sets one ratio and leaves the others to
        follow, so matching the mid-tone would push the black further down while matching the black pushes the mid up.
        Two independent quantities are needed because two independent things are happening, and both are real
        operations in a camcorder's DSP:

          * **`dsp_black`** is the *setup* / pedestal level.  A camera does not record its darkest scene pixel at code
            zero -- it puts it above blanking, so that the noise floor is not clipped away and so that a picture has a
            reference black.  **This is why the footage's 1st percentile is 0.037 and not 0.000**: the measurement is
            of a camcorder's output, and a camcorder's output has a black level.  Sony's own DV black is code 16
            (0.063 in full-range terms), a little above what this material carries.
          * **`dsp_gain`** is the gain the DSP settles on.  With `ae_enabled` off -- which is the default, so the loop
            is *not* the mechanism here -- this is what a fixed-gain camcorder applies, and the model needs it above
            or below unity to sit where the target does.

        **The order is gain then black, and that order is the physical one**: the pedestal is added to an already
        amplified signal, on the digitised side of the chain.  Doing it the other way scales the pedestal too, which
        would make `dsp_black` mean something different at every gain and the two controls inseparable.

        **Both default to neutral** (`gain = 1.0`, `black = 0.0`), so this changes nothing until someone calibrates
        it -- deliberately, because a change to a level reaches every figure in this repository and should not arrive
        as a side effect of adding the mechanism.  The values that put the two percentiles on the target are measured
        and recorded in the schema; promoting them to the defaults is a separate, visible decision.
        """
        g = float(getattr(self, "dsp_gain", 1.0))
        b = float(getattr(self, "dsp_black", 0.0))
        if abs(g - 1.0) < 1e-4 and abs(b) < 1e-5:
            return rgb
        # the pedestal is held below the white point so it can only lift, never invert, the black end
        b = float(np.clip(b, -0.5, 0.5))
        return np.clip(rgb * g + b, 0.0, 2.0).astype(np.float32)

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
        # **One ray field per channel, because diffraction separates by wavelength.**  Each channel gets its own
        # angular offset and its own length, so the spikes of the three channels do not coincide and the result is
        # coloured streaks rather than one grey pattern with a colour on top.  `spread` is the parameter; the signs
        # are fixed and the red and blue are placed on opposite sides of green, which is the first-order form of
        # `fringe spacing ~ lambda * f / D`.
        #
        # The achromatic version could not do this at any strength, because a single geometry multiplied by a
        # colour gives all three channels the same rays.  Measured against the user's reference photographs, which
        # show separate red, green and blue streaks out of a blown street lamp.
        spread = float(getattr(self, "spike_spread", 0.0))
        base_theta = 0.15
        ray_len = max(6.0, 0.34 * max(h, w) * (0.45 + 1.35 * a))
        dy_ax = yy - cy
        dx_ax = xx - cx
        rad = np.sqrt(dy_ax * dy_ax + dx_ax * dx_ax)
        # per channel: (angular offset, length factor).  Green is the reference the eye focuses on.
        # The offsets are in **pixels of the spike's own width**, not degrees: two degrees against the 60 degrees
        # between blades moved a spike by a fraction of its width, which measured as almost no separation at all
        # (R-G 0.228 -> 0.259 over the whole range).  A shear of `spread * 0.9` px is comparable to the spike's own
        # width, so the channels' spikes land in different places.
        channel_offsets = ((-1.0, 1.0 + 0.22 * spread),
                           (0.0, 1.0),
                           (1.0, 1.0 - 0.22 * spread))
        ray_stack = []
        for ch_sign, len_factor in channel_offsets:
            rays_ch = np.zeros((h, w), dtype=np.float32)
            # a **lateral** shift of the spike rather than a rotation about the source, which is what a
            # wavelength-dependent fringe spacing does to a long thin spike
            shear_px = ch_sign * spread * 0.9
            for k in range(blades):
                theta = 2.0 * np.pi * k / blades + base_theta
                uy, ux = float(np.sin(theta)), float(np.cos(theta))
                proj = np.abs(dy_ax * uy + dx_ax * ux)
                # the chromatic shift is across the spike, so each channel's spike is displaced sideways
                perp = np.abs(-dy_ax * ux + dx_ax * uy - shear_px)
                width = 0.9 + 0.010 * proj
                along = np.clip(1.0 - proj / max(2.0, ray_len * len_factor), 0.0, 1.0) ** 1.6
                across = np.clip(1.0 - perp / np.maximum(width, 1e-3), 0.0, 1.0)
                rays_ch = np.maximum(rays_ch, along * across)
            ray_stack.append(lowpass(rays_ch, 0.8))
        rays_rgb = np.stack(ray_stack, axis=-1)

        # the coating's cool blue-violet cast still applies, but the **rays carry their own per-channel
        # geometry** -- so the spikes separate into red, green and blue instead of being one tinted pattern.
        tint = np.array([0.85, 0.90, 1.25], dtype=np.float32)
        ghost_part = (0.55 * ghost * a)[..., None] * tint[None, None, :]
        ray_part = (0.65 * a) * rays_rgb * tint[None, None, :]
        flare = (ghost_part + ray_part) * 0.55
        # **Per-channel defocus, which is what makes a halo coloured.**  The channels focus at different distances,
        # so the out-of-focus image of the aperture is larger in blue than in red: the halo is a blue ring *around
        # and below* the source rather than a colour painted on it.  This is the same physics as the lens module's
        # `longitudinal_ca`, applied to the flare instead of the picture -- and it needs its own parameter because
        # the flare is light that has already left the aperture's focus.
        defocus = float(getattr(self, "flare_defocus", 0.5))
        if defocus > 1e-4:
            # **The defocus belongs on the compact core, not on the whole field.**  Defocus acts on the image of
            # the aperture -- a small disc around the source -- and it does not act on a long spike or on the ghost
            # polygon, which is drawn at a fixed radius.  Measured with it applied to everything, the blue/red
            # width ratio was **1.00** across the whole range, because the flare's second moment is about 61 px and
            # dominated by the polygon at 56 px, so a 5-17 px kernel moved it by well under a percent.
            #
            # So the field is split: the core is what a narrow blur keeps, and only that is defocused per channel.
            # That is also the only place a coloured halo can form -- around the source, where the aperture's own
            # disc is imaged.
            core = lowpass(flare, 3.5)
            rest = flare - core
            radii = (1.0 + 4.0 * defocus, 1.0 + 9.0 * defocus, 1.0 + 16.0 * defocus)
            planes = [lowpass(core[..., c], radii[c]) for c in range(3)]
            flare = np.stack(planes, axis=-1) + rest
        return (rgb + flare).astype(np.float32)

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
            # **The mode's own trim is added AFTER the loop, because the loop REPLACES this value.**  Measured: setting
            # `nv_exposure_trim` to -0.10, -0.6, -1.0, -1.4 and -1.8 all gave bit-identical output, and the reason is
            # this assignment -- `self.exposure = ...update(...)` overwrites whatever the constructor put there, so the
            # trim was computed, stored, read by `_nv_exposure_bias` and then discarded one line later.  **`self.exposure`
            # is an ATTRIBUTE and this is the last statement that writes it**, so the trim belongs here, where the value
            # is finally settled, rather than in the constructor where the loop can still take it away.
            self.exposure = float(np.clip(self.exposure + float(getattr(self, "nv_exposure_trim", 0.0)), -3.0, 3.0))
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

        # **The night-vision scene terms, before the sensor.**  The illuminator is outside the camera, so its
        # light is part of what the camera is looking at -- added here rather than at the end, where it would
        # be light added to an already-exposed picture.
        rgb = self._night_vision(rgb, "scene", buf.index)
        rgb = self._roll_off(rgb)
        # fog is on the front element, so light passes through it before anything else
        if self.lens_fog > 1e-4:
            from ..codecs.optics import lens_fog
            rgb = lens_fog(rgb, self.lens_fog)
        # **After the light mechanisms, so it deepens what they left around a source.**  It adds no light
        # of its own, which is why it can raise a line's contrast without touching the line's colour.
        rgb = self._shadow_depth(rgb)
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
        # **Scattered light first, because it is the widest and softest of the light mechanisms.**  A halo
        # that carries the source's own colour, which is what the reference's near field shows: 60 and 100
        # rows below a blown core it runs to B-R -0.170 and -0.510, where the model reached -0.014 and
        # -0.034.  Nothing else in the module follows the source's colour.
        rgb = source_glow(rgb, self.lens)
        rgb = spherical(rgb, self.lens)
        rgb = self._sharpen(rgb)
        # **The monochrome green view, after every sensor term.**  It is the viewfinder's output, so nothing at the
        # sensor can follow it.
        #
        # **Accumulation is NOT here.**  It was, and it ran a second time -- overwriting the fully-processed picture
        # with the mean of several *source* frames, 0.1768 replacing 0.45429.  That is the 2.25x loss the stage trace
        # showed, and it survived because the scene-stage version was added without this one being removed.  A
        # mechanism wired in two places is worse than one wired in none: the second silently wins, and the first's
        # measurements describe a picture that never reaches the output.  Accumulation now runs once, in
        # `_night_vision`'s scene stage, before the illuminator -- see the comment there.
        rgb = self._night_vision(rgb, "display")
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

        # --- the DSP's output level ---------------------------------------
        # Last, and after the readout defects, because it is the stage that sets what the recorded signal *is*: the
        # gain the camera settles on and the black level it puts under it.  Everything before this is the signal.
        rgb = self._dsp_level(rgb)

        return FrameBuffer(np.clip(rgb, 0.0, 2.0).astype(np.float32), None,
                           buf.index, buf.time, buf.std, dict(buf.meta))

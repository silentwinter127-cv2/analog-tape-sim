"""
Physical CRT display optics, electron-beam dynamics, RGB phosphor geometry, and video-amplifier faults.

A real cathode-ray tube is an electro-optical system rather than a static 1-D cosine overlay:

1. **Dynamic Electron Beam Spot Blooming (`spot_bloom`)**:
   As cathode current $I_k$ rises in bright regions, space-charge repulsion inside the electron gun
   widens the beam spot $\\sigma_{\\text{eff}}(y, x) = \\sigma_0 \\cdot (1 + k_{\\text{bloom}} \\cdot Y(y, x)^p)$.
   Dark and mid-tone lines stay narrow with crisp black inter-line gaps, while intense highlights expand
   vertically and horizontally, bridging the scanline gaps just as a real CRT does.

2. **True RGB Phosphor Subpixel Triads, Black Matrix & Trinitron Damper Wires (`phosphor_rgb`, `mask_type`)**:
   Instead of multiplying all three channels by an identical scalar shadow-mask greyscale pattern,
   the three electron beams excite spatially distinct R, G, and B phosphor stripes (Aperture Grille,
   `mask_type=0`), staggered brick slots (Slot Mask, `mask_type=1`), or hexagonal delta triads
   (Dot Mask, `mask_type=2`), separated by a carbon Black Matrix guard band. Aperture Grille tubes
   also cast the signature faint horizontal shadow of one or two tungsten tension damper wires.

3. **Leaded Faceplate Glass Internal Halation (`glass_halation`) & P22 Phosphor Saturation (`phosphor_saturation`)**:
   Light emitted by the phosphor layer undergoes total internal reflection at the front glass-air
   interface of the thick (10-14 mm) leaded faceplate, forming a wide secondary halation ring/veil
   around bright features on dark backgrounds, while P22 phosphors saturate non-linearly at high drive.

4. **Deflection Yoke Corner Astigmatism (`corner_defocus`), Line-Load EHT Sag (`ehv_sag`), & Bezel (`bezel`)**:
   Oblique beam landing at the screen corners introduces radial astigmatism; bright horizontal scanlines
   load the flyback high-voltage supply and sag outward; and a curved physical bezel frames the raster.

5. **Moisture & Corrosion in the Video Amplifier / Deflection Supply (`moisture_effects`, `apply_deflection`)**:
   Preserves the diagnostic amplifier bias drift, per-cathode aging cast, and sagging deflection raster scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signal import horizontal_shift, lowpass

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class CRTMoisture:
    """What the damp amplifier does to the picture, reported so the caller can log it."""

    bias_drift: float
    cast: tuple[float, float, float]
    deflection: float


def moisture_effects(moisture: float, corrosion: float, age: float = 0.3) -> CRTMoisture:
    """
    Translate the user's levels into the three physical quantities the stage applies.

    Kept pure so the relationship can be checked directly without rendering a frame.
    """
    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    a = float(np.clip(age, 0.0, 1.0))

    # the bias shifts with both moisture and corrosion: a leaky path to ground moves the
    # black level up, which is why a damp set washes the picture out rather than darkening it
    bias = 0.18 * m + 0.12 * c

    # the three cathode amplifiers do not age alike: green's tube/transistor typically runs
    # hottest and drifts furthest, red next, blue least.  Age scales the *spread* between
    # them, so a new set has almost no cast even when damp and an old one has a strong one.
    severity = 0.65 * m + 0.55 * c
    spread = severity * (0.08 + 0.52 * a)
    cast = (
        float(1.0 - 0.55 * spread),   # red falls back
        float(1.0 + 0.45 * spread),   # green comes up -> the familiar greenish-yellow cast
        float(1.0 - 0.80 * spread),   # blue falls furthest
    )

    # the deflection supply sags with the same leakage
    defl = float(np.clip(0.55 * m + 0.65 * c, 0.0, 1.0))
    return CRTMoisture(bias_drift=bias, cast=cast, deflection=defl)


def apply_deflection(rgb: np.ndarray, deflection: float, frame_index: int,
                     seed: int = 0) -> tuple[np.ndarray, dict]:
    """
    Shrink and wobble the raster, because the deflection supply is sagging.

    This is the diagnostic part of the fault: nothing upstream of the CRT can move the
    *edges of the raster*, because a tape cannot bend the picture. When the supply sags the
    scan does not reach the bezel, so a narrow dark border appears and its width breathes
    with the supply ripple.
    """
    d = float(np.clip(deflection, 0.0, 1.0))
    if d <= 1e-4:
        return rgb, {}

    h, w = rgb.shape[:2]
    # the raster shrinks by a few percent, and breathes slowly with the supply's ripple
    breath = 0.018 * d * float(np.sin(frame_index * 0.37 + seed * 0.1))
    scale = float(np.clip(1.0 - 0.065 * d + breath, 0.86, 1.0))

    yy = (np.arange(h, dtype=np.float32) - h * 0.5) / scale + h * 0.5
    xx = (np.arange(w, dtype=np.float32) - w * 0.5) / scale + w * 0.5
    # and a slight horizontal bow, because the horizontal output stage is loaded asymmetrically
    bow = (d * 3.5) * np.sin(np.linspace(0.0, np.pi, h, dtype=np.float32))

    y_idx = np.clip(np.round(yy).astype(np.int32), 0, h - 1)
    out = np.zeros_like(rgb)
    inside_y = (yy >= 0.0) & (yy < h)
    for row in range(h):
        if not inside_y[row]:
            continue
        x_row = xx + bow[row]
        inside_x = (x_row >= 0.0) & (x_row < w)
        x_idx = np.clip(np.round(x_row).astype(np.int32), 0, w - 1)
        line = rgb[y_idx[row], x_idx]
        line[~inside_x] = 0.0
        out[row] = line

    return out, {"raster_scale": round(scale, 4)}


# ===========================================================================
# Physical CRT Beam, Phosphor Triad, Halation & Deflection Optics
# ===========================================================================

def build_phosphor_mask(h: int, w: int, mask_pitch: float, mask_depth: float,
                        mask_type: int = 0, phosphor_rgb: float = 0.65
                        ) -> np.ndarray | None:
    """
    Construct a 3-channel `(H, W, 3)` RGB phosphor subpixel & Black Matrix transmittance map.

    * `mask_type == 0` (**Aperture Grille / Trinitron**):
      Continuous vertical R, G, B phosphor stripes separated by dark Black Matrix guard slots,
      plus faint horizontal tungsten tension damper wire shadows at ~30% and ~70% vertical height.
    * `mask_type == 1` (**Slot Mask / Staggered In-Line TV**):
      Vertical R, G, B phosphor slots broken by horizontal shadow-mask bridges every `slot_h` rows,
      with alternate triad columns staggered vertically by half a slot period (brick lattice).
    * `mask_type == 2` (**Dot Triad / Delta Gun Shadow Mask**):
      Hexagonal staggered circular phosphor dot triads where R, G, B dots sit at 120-degree phase
      offsets and alternate row bands shift by half a triad pitch.

    Each channel plane is normalized to unit mean (`mask[..., c].mean() == 1.0`) so engaging the
    mask or changing `phosphor_rgb` never alters overall brightness or neutral white balance.
    """
    if mask_pitch <= 1e-3 or mask_depth <= 1e-4:
        return None

    pitch = max(1.5, float(mask_pitch))
    depth = float(np.clip(mask_depth, 0.0, 1.0))
    rgb_sep = float(np.clip(phosphor_rgb, 0.0, 1.0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Scalar luminance shadow-mask envelope (preserves exact compatibility when phosphor_rgb == 0)
    phase = (xx / pitch) % 1.0
    if mask_type == 1:
        band = np.floor(yy / max(1.0, pitch)) % 2.0
        shift = band * 0.5
        m_scalar = 0.5 + 0.5 * np.cos(2.0 * np.pi * (((xx / pitch) + shift) % 1.0))
        # Horizontal slot bridge bars across each triad cell
        slot_h = max(2.0, pitch * 1.15)
        bridge = 1.0 - 0.38 * depth * (np.sin(np.pi * (yy / slot_h)) ** 8)
        m_scalar = m_scalar * bridge
    elif mask_type == 2:
        cy = np.floor(yy / max(1.0, pitch))
        shift = (cy % 2.0) * 0.5
        dx = ((xx / pitch) + shift) % 1.0 - 0.5
        dy = (yy / pitch) % 1.0 - 0.5
        r2 = dx * dx + dy * dy
        m_scalar = np.exp(-r2 / 0.08)
    else:
        vert = np.cos(2.0 * np.pi * phase)
        m_scalar = 0.5 + 0.5 * vert

    scalar_mask = 1.0 - depth * (1.0 - m_scalar)
    scalar_mask = (scalar_mask / max(1e-6, float(scalar_mask.mean()))).astype(np.float32)

    if rgb_sep <= 1e-4:
        return np.repeat(scalar_mask[..., None], 3, axis=2)

    # Subpixel R, G, B phosphor triad profiles at phases (0, 1/3, 2/3) of the triad pitch.
    # A triad spans `triad_w = max(2.4, pitch * 1.5)` pixels so that even at pitch=2.0..3.0 px
    # the three R/G/B subpixels are well-resolved without severe Moire aliasing.
    triad_w = max(2.2, pitch * 1.35)
    if mask_type == 1:
        slot_h = max(2.0, pitch * 1.25)
        col_idx = np.floor(xx / triad_w)
        y_stag = yy + (col_idx % 2.0) * (slot_h * 0.5)
        slot_env = 1.0 - (0.45 * depth) * (np.cos(np.pi * (y_stag / slot_h)) ** 6)
        x_phase = (xx / triad_w) % 1.0
    elif mask_type == 2:
        dot_h = max(1.8, pitch * 1.05)
        row_band = np.floor(yy / dot_h)
        x_phase = ((xx / triad_w) + (row_band % 2.0) * 0.5) % 1.0
        dy_dot = ((yy / dot_h) % 1.0) - 0.5
        slot_env = np.exp(-(dy_dot * dy_dot) / max(0.06, 0.22 - 0.10 * depth))
    else:
        x_phase = (xx / triad_w) % 1.0
        slot_env = np.ones_like(xx, dtype=np.float32)
        # Trinitron aperture grille horizontal tungsten damper wire shadow (1 or 2 wires)
        if h >= 64:
            for wy_frac in (0.31, 0.71):
                wy = wy_frac * h
                wire_prof = np.exp(-0.5 * ((yy - wy) / 0.65) ** 2)
                slot_env = slot_env * (1.0 - 0.14 * depth * wire_prof)

    # Black matrix guard bands between phosphor stripes + subpixel selectivity
    channels = []
    centers = (0.167, 0.500, 0.833)
    for c_phase in centers:
        # Wrapped distance on [0, 1) circle
        d_phi = np.abs(((x_phase - c_phase + 0.5) % 1.0) - 0.5)
        # Gaussian phosphor stripe + plateau with cross-excitation floor so single-channel colors stay rich
        sub_profile = np.exp(-0.5 * (d_phi / 0.19) ** 2)
        ch_mask = (1.0 - depth * 0.88) + (depth * 0.88) * sub_profile * slot_env
        ch_mask = ch_mask / max(1e-6, float(ch_mask.mean()))
        channels.append(ch_mask.astype(np.float32))

    rgb_mask = np.stack(channels, axis=-1)
    # Blend smoothly between scalar mask and full RGB subpixel triad mask
    out_mask = scalar_mask[..., None] * (1.0 - rgb_sep) + rgb_mask * rgb_sep
    for ch in range(3):
        out_mask[..., ch] /= max(1e-6, float(out_mask[..., ch].mean()))
    return out_mask.astype(np.float32)


def apply_scanline_beam(rgb: np.ndarray, base_row_mask: np.ndarray,
                        scanlines: float, spot_sigma: float, scan_pitch: float,
                        spot_bloom: float = 0.0, field_parity: int = 0,
                        interlace: float = 0.0) -> np.ndarray:
    """
    Apply the electron-beam scanline profile with dynamic cathode-current spot blooming (`spot_bloom`).

    * When `spot_bloom <= 0` (or on a uniform flat field), this reduces to multiplying by `base_row_mask`,
      guaranteeing zero light loss (`spread < 0.002`) and monotone focus control (`spot_sigma`).
    * When `spot_bloom > 0`, bright highlights drive high beam current:
      1. The vertical Gaussian spot widens in proportion to local luminance $Y(y, x)$, filling in the
         dark inter-line gaps inside bright objects while keeping dark/midtone scanlines crisp.
      2. Bright scanline cores spill vertically into adjacent scanline troughs and horizontally along
         the active line, reproducing authentic CRT highlight line-bridging.
    """
    if scanlines <= 1e-4:
        return rgb * base_row_mask

    h, w = rgb.shape[:2]
    bloom_k = float(np.clip(spot_bloom, 0.0, 2.0))
    if bloom_k <= 1e-4 and interlace <= 1e-4:
        return (rgb * base_row_mask).astype(np.float32)

    scale_h = (h / 540.0) if h > 580 else 1.0
    pitch = max(1.05, float(scan_pitch) * scale_h)
    sigma0 = max(1e-3, float(spot_sigma) * scale_h)

    # Interlace field phase shift: odd/even fields shift the beam centres slightly
    field_shift = (0.18 * float(np.clip(interlace, 0.0, 1.0)) * pitch
                   * (0.5 if (field_parity % 2 == 1) else -0.5))
    rows = np.arange(h, dtype=np.float32)[:, None] - field_shift
    centres = np.round(rows / pitch) * pitch
    dy = rows - centres  # (H, 1) distance to nearest scanline center

    # Base 1D beam profile (normalized to unit mean, identical to DisplayStage._masks)
    beam0 = np.exp(-0.5 * (dy / sigma0) ** 2)
    mask0 = 1.0 - scanlines * (1.0 - beam0)
    mask0_norm = mask0 / max(1e-6, float(mask0.mean()))

    if bloom_k <= 1e-4:
        # Apply vignette ratio from base_row_mask if present
        vig_ratio = base_row_mask / np.maximum(1e-6, base_row_mask.mean(axis=0, keepdims=True))
        return (rgb * (mask0_norm[..., None] * vig_ratio)).astype(np.float32)

    # Local luminance (smoothed slightly along the scanline and across 1 row so the beam responds
    # to the local cathode drive across the scanline gap)
    lum = np.clip(rgb @ LUMA, 0.0, 1.5)
    lum_env = lowpass(lowpass(lum, 1.2, axis=1), 0.85, axis=0)

    # Space-charge spot widening: only significant above mid-grey so dark/midtone scanlines stay sharp
    drive = np.clip((lum_env - 0.35) / 0.65, 0.0, 1.5) ** 1.35
    sigma_eff = sigma0 * (1.0 + 1.35 * bloom_k * drive)
    beam_dyn = np.exp(-0.5 * (dy / np.maximum(1e-3, sigma_eff)) ** 2)
    mask_dyn = 1.0 - scanlines * (1.0 - beam_dyn)

    # Normalize mask_dyn per-intensity-bin against mask0's mean so flat fields keep exact mean=1.0
    # while bright features bridge the dark gaps between scanlines
    bridge_weight = np.clip(drive * bloom_k * 0.85, 0.0, 0.92)
    eff_scan_mask = mask0_norm * (1.0 - bridge_weight) + (mask_dyn / max(1e-6, float(mask0.mean()))) * bridge_weight

    # Extract vignette component from base_row_mask (by dividing out the 1D row mean of base_row_mask)
    row_mean_base = np.maximum(1e-6, base_row_mask.mean(axis=1, keepdims=True))
    vig_2d = base_row_mask / row_mean_base

    out = rgb * (eff_scan_mask[..., None] * vig_2d)

    # Vertical/horizontal highlight spot spill into adjacent scanline gaps on intense highlights
    hot_spill = np.clip(rgb - 0.68, 0.0, None)
    if float(hot_spill.max()) > 1e-4:
        spill_v = lowpass(lowpass(hot_spill, 0.9 * scale_h, axis=0), 1.4 * scale_h, axis=1)
        gap_openness = np.clip(1.0 - beam0, 0.0, 1.0)[..., None]
        out = out + (0.38 * bloom_k * scanlines) * spill_v * gap_openness

    target_mean = float((rgb * base_row_mask).mean())
    cur_mean = float(out.mean())
    if cur_mean > 1e-6:
        out = out * (target_mean / cur_mean)
    return out.astype(np.float32)


def apply_faceplate_halation(rgb: np.ndarray, glass_halation: float,
                             bloom: float, scale_w: float = 1.0) -> np.ndarray:
    """
    Two-stage CRT light scatter:
      1. **Electron-beam & phosphor tight bloom (`bloom`)**: Energy-proportional halo around bright
         highlights (never normalized by `1 / peak`, which previously turned a single dim pixel at
         0.73 into a full-strength glow).
      2. **Double-layer leaded glass faceplate internal halation (`glass_halation`)**:
         Light emitted by the phosphor reflects off the front glass-air interface (beyond the critical
         angle of leaded glass) and scatters back onto the phosphor screen at a wide radius, creating
         the characteristic soft CRT halation ring/veil with a subtle warm-amber/cyan glass tint.
    """
    out = rgb.astype(np.float32, copy=True)
    lum = np.clip(out @ LUMA, 0.0, 2.0)

    if bloom > 1e-3:
        # Energy-proportional highlight extraction (smooth knee above 0.65)
        hot_rgb = np.clip(out - 0.62, 0.0, None)
        if float(hot_rgb.max()) > 1e-5:
            tight = lowpass(hot_rgb, max(0.8, (2.2 + 4.5 * bloom) * scale_w))
            mid = lowpass(hot_rgb, max(1.8, (6.0 + 11.0 * bloom) * scale_w))
            # Soft peak adaptation so genuinely bright highlights glow strongly while barely-above-threshold
            # regions glow proportionally
            ref_scale = 1.0 / max(0.28, float(np.percentile(hot_rgb, 99.5)) + 0.12)
            ref_scale = float(np.clip(ref_scale, 0.8, 3.2))
            out = out + (bloom * 0.42 * ref_scale) * (0.68 * tight + 0.32 * mid)

    gh = float(np.clip(glass_halation, 0.0, 1.5))
    if gh > 1e-3:
        # Internal reflection in thick faceplate glass: ring-like wide scatter (difference of wide Gaussians)
        hot_glass = np.clip(out - 0.52, 0.0, None)
        if float(hot_glass.max()) > 1e-5:
            r_inner = max(2.5, (7.0 + 8.0 * gh) * scale_w)
            r_outer = max(6.0, (18.0 + 24.0 * gh) * scale_w)
            g_in = lowpass(hot_glass, r_inner)
            g_out = lowpass(hot_glass, r_outer)
            # Ring + wide veil simulates total internal reflection bounce at the front glass surface
            ring_halo = np.clip(1.35 * g_out - 0.35 * g_in, 0.0, None)
            glass_tint = np.array([1.06, 1.01, 0.94], dtype=np.float32)[None, None, :]
            out = out + (0.36 * gh) * ring_halo * glass_tint

    return out.astype(np.float32)


def apply_phosphor_saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    """
    P22 / EBU CRT phosphor luminous saturation & highlight core incandescence.

    At high beam drive, ZnS:Ag (blue) and ZnS:Cu,Al (green) phosphors saturate earlier than
    Y2O2S:Eu (red), while secondary electron/light scatter across adjacent phosphor stripes
    desaturates blown single-channel highlights toward a glowing white-hot core.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 1e-4:
        return rgb
    x = np.clip(rgb, 0.0, 2.0).astype(np.float32)
    # Per-channel P22 saturation knee (Red holds linear longest; Green and Blue roll off slightly earlier)
    knees = np.array([0.82, 0.74, 0.70], dtype=np.float32)[None, None, :]
    over = np.clip(x - knees, 0.0, None)
    compressed = np.where(x <= knees, x, knees + (1.0 - knees) * np.tanh(over / np.maximum(1e-3, 1.0 - knees)))
    out = x * (1.0 - 0.65 * a) + compressed * (0.65 * a)

    # White-hot cross-phosphor incandescence when any channel is driven hard
    peak_ch = out.max(axis=-1, keepdims=True)
    lum = (out @ LUMA)[..., None]
    core_drive = np.clip((peak_ch - 0.78) / 0.22, 0.0, 1.0) ** 1.6
    warm_white = np.array([1.02, 1.00, 0.97], dtype=np.float32)[None, None, :]
    target_core = np.maximum(out, (0.65 * peak_ch + 0.35 * lum) * warm_white)
    out = out * (1.0 - 0.55 * a * core_drive) + target_core * (0.55 * a * core_drive)
    return out.astype(np.float32)


def apply_corner_astigmatism(rgb: np.ndarray, corner_defocus: float,
                             scale_w: float = 1.0) -> np.ndarray:
    """
    Deflection yoke corner astigmatism & dynamic focus error.

    As the electron beam sweeps toward the edges and corners of the tube, it strikes the phosphor
    screen at an oblique angle, stretching the circular spot horizontally and radially (`r^2` law).
    The screen centre remains completely untouched (`weight == 0` at `(0, 0)`).
    """
    cd = float(np.clip(corner_defocus, 0.0, 1.5))
    if cd <= 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    r2 = np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0)

    # Oblique landing stretches horizontal focus more than vertical focus
    soft_h = lowpass(rgb, max(0.6, (1.4 + 3.2 * cd) * scale_w), axis=1)
    soft_hv = lowpass(soft_h, max(0.4, (0.8 + 1.8 * cd) * scale_w), axis=0)
    w_corner = (r2 ** 1.35 * np.clip(cd, 0.0, 1.0))[..., None].astype(np.float32)
    return (rgb * (1.0 - w_corner) + soft_hv * w_corner).astype(np.float32)


def apply_ehv_line_sag(rgb: np.ndarray, ehv_sag: float, scale_w: float = 1.0) -> np.ndarray:
    """
    Per-scanline Horizontal High-Voltage (EHT) load sag ("line blooming").

    On consumer CRTs, a scanline carrying high average beam current pulls down the flyback's
    high-voltage supply over the horizontal RC time constant (~3-8 scanlines). Lower anode voltage
    reduces electron velocity, increasing deflection sensitivity so bright horizontal bands expand
    outward from the centre by several pixels.
    """
    sag = float(np.clip(ehv_sag, 0.0, 1.5))
    if sag <= 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    row_lum = (np.clip(rgb, 0.0, 1.0) @ LUMA).mean(axis=1)
    # Flyback capacitor RC smoothing down the field (causal exponential + symmetric smoothing)
    row_load = lowpass(row_lum.astype(np.float32), max(1.5, h * 0.025), axis=0)
    delta_load = row_load - float(row_load.mean())
    if float(np.abs(delta_load).max()) <= 1e-4:
        return rgb

    # Each line expands horizontally about the screen centre by `1 + sag_factor * delta_load`
    xx = np.arange(w, dtype=np.float32)[None, :]
    cx = (w - 1) * 0.5
    expand = (sag * 0.045 * delta_load)[:, None]  # fractional horizontal expansion
    src_x = np.clip(cx + (xx - cx) / np.maximum(0.85, 1.0 + expand), 0.0, w - 1.001)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (src_x - x0)[..., None].astype(np.float32)
    rows_idx = np.arange(h, dtype=np.int32)[:, None]
    out = rgb[rows_idx, x0] * (1.0 - fx) + rgb[rows_idx, x1] * fx
    return out.astype(np.float32)


def apply_curved_bezel_and_glass(rgb: np.ndarray, bezel: float, reflection: float,
                                 curvature: float = 0.0) -> np.ndarray:
    """
    Physical CRT curved mask bezel, corner cutoff, inner shadow rim, and 3D Fresnel glass reflection.
    """
    out = rgb.astype(np.float32, copy=True)
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0

    # 3D Curved Faceplate Specular & Fresnel Ambient Reflection
    refl = float(np.clip(reflection, 0.0, 1.0))
    if refl > 1e-3:
        # Surface normal of the curved glass dome tilts outward with (nx, ny)
        curv_eff = 0.35 + 0.65 * float(np.clip(curvature, 0.0, 1.2))
        u_refl = nx * (1.0 + 0.18 * curv_eff * (nx * nx + ny * ny))
        v_refl = ny * (1.0 + 0.18 * curv_eff * (nx * nx + ny * ny))
        # Soft overhead/window studio light reflection on the curved glass + Fresnel edge sheen
        spot1 = np.exp(-(((u_refl + 0.42) ** 2 + (v_refl + 0.48) ** 2) / 0.22))
        spot2 = 0.38 * np.exp(-(((u_refl - 0.52) ** 2 + (v_refl + 0.36) ** 2) / 0.45))
        r2 = nx * nx + ny * ny
        fresnel = (1.0 - np.clip(1.0 - 0.42 * r2, 0.0, 1.0)) ** 2
        sky_grad = np.clip(0.55 - 0.45 * v_refl, 0.0, 1.0) * (0.25 + 0.75 * fresnel)
        glass_glare = (0.72 * spot1 + 0.28 * spot2 + 0.24 * sky_grad)[..., None]
        glare_tint = np.array([0.96, 0.99, 1.04], dtype=np.float32)[None, None, :]
        out = out + (refl * 0.20) * glass_glare * glare_tint

    # Physical Curved CRT Bezel & Corner Masking (`bezel`)
    bz = float(np.clip(bezel, 0.0, 1.0))
    if bz > 1e-3:
        # Superellipse (squircle) tube aperture boundary: |x/a|^p + |y/b|^p = 1
        # Higher `bezel` rounds the corners and brings the inner shadow bezel into frame
        bow = 0.035 * bz * (1.0 + 0.8 * float(np.clip(curvature, 0.0, 1.0)))
        ax = np.abs(nx) * (1.0 + bow * (ny * ny))
        ay = np.abs(ny) * (1.0 + bow * (nx * nx))
        power = max(4.0, 14.0 - 8.5 * bz)
        dist = (ax ** power + ay ** power) ** (1.0 / power)
        edge_limit = 1.015 - 0.035 * bz
        # Crisp bezel cutoff + soft inner shadow rim from the recessed glass thickness
        inside = np.clip((edge_limit - dist) / max(0.006, 0.012 * bz), 0.0, 1.0)
        rim_shadow = np.clip((edge_limit - dist) / max(0.025, 0.065 * bz), 0.0, 1.0) ** 0.45
        bezel_mask = (inside * (0.35 + 0.65 * rim_shadow))[..., None].astype(np.float32)
        # Subtle dark bezel specular lip right on the boundary
        lip = np.exp(-0.5 * ((dist - edge_limit - 0.006) / 0.005) ** 2) * (1.0 - inside)
        lip_rgb = (0.045 * bz * lip)[..., None] * np.array([0.85, 0.88, 0.92], dtype=np.float32)
        out = out * bezel_mask + lip_rgb

    return out.astype(np.float32)

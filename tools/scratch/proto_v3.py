import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from PIL import Image
from atsim.core.signal import lowpass

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def bilinear_sample(plane: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    h, w = plane.shape[:2]
    sy_c = np.clip(sy, 0.0, h - 1.001)
    sx_c = np.clip(sx, 0.0, w - 1.001)
    y0 = np.floor(sy_c).astype(np.int32)
    x0 = np.floor(sx_c).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = sy_c - y0
    fx = sx_c - x0
    if plane.ndim == 3:
        fy = fy[..., None]
        fx = fx[..., None]
    top = plane[y0, x0] * (1.0 - fx) + plane[y0, x1] * fx
    bot = plane[y1, x0] * (1.0 - fx) + plane[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def coma_v3(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    if coma_amt < 1e-4:
        return rgb
    a = float(coma_amt)
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0

    # Tilted C-mount lens coma center at (-6.4*a, -3.2*a):
    # Gives ~18 deg at top-left, ~32 deg at bottom-left, ~12 deg at top-right, ~21 deg at bottom-right!
    dx_ax = -float(axis_offset) * 6.4 if abs(axis_offset) > 1e-4 else -0.01
    dy_ax = -float(axis_offset) * 3.2 if abs(axis_offset) > 1e-4 else -0.01
    rx = nx - dx_ax
    ry = ny - dy_ax
    unorm = np.maximum(np.hypot(rx, ry), 1e-6)
    ux, uy = rx / unorm, ry / unorm
    vx, vy = -uy, ux

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 15.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    emitter = hot_m ** 1.15
    reach = (2.5 + 11.5 * a) * scale_w

    # 3 comatic rays: Central ray (psi = 0) is longest (1.22x), flanked by two shorter side rays (0.76x at -28 deg and +28 deg)!
    # When source disk is large (upper lamps), this forms the exact tapered D-wedge / guitar-pick nose!
    # When source disk is small (lower lamps), the 3 rays separate at their tips into 3 distinct fingers!
    lobe_specs = (
        (-0.50, 0.78, 0.88),
        ( 0.00, 1.22, 1.00),
        ( 0.50, 0.76, 0.86),
    )

    bundle_w = emitter.copy()
    tail_b = np.zeros((h, w), np.float32)

    for ang, r_sc, lw in lobe_specs:
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        dx_l = ca * ux + sa * vx
        dy_l = ca * uy + sa * vy
        for frac in (0.16, 0.32, 0.48, 0.64, 0.80, 0.96, 1.14):
            dist = reach * r_sc * frac
            val = bilinear_sample(emitter, yy - dy_l * dist, xx - dx_l * dist)
            # Saturated white core reaches up to frac ~ 0.82
            w_core = lw * np.clip(1.28 - 1.18 * frac, 0.0, 1.0) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 1.38)
            # Sub-saturated cyan-blue dispersion skirt extends beyond the white core (frac 0.65..1.14)
            w_blue = lw * np.exp(-0.5 * ((frac - 0.88) / 0.32) ** 2) * a
            tail_b += val * w_blue * 0.48

    # Smooth the combined white bundle slightly so the wedge boundary is continuous
    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.25), max(0.45, 0.60 * scale_w))
    core_blend = np.clip((bundle_w - 0.25) / 0.65, 0.0, 1.0)[..., None]

    # Inside core_blend, the highlight is PURE NEUTRAL WHITE (1.0, 1.0, 1.0)!
    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    # Outside the saturated white core (where 1 - core_blend > 0), add the soft slate-cyan/blue trailing skirt
    skirt_w = (1.0 - core_blend[..., 0] * 0.92) * np.clip(tail_b, 0.0, 0.85)
    out[..., 2] = np.clip(out[..., 2] + skirt_w * 0.52, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + skirt_w * 0.24, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] - skirt_w * 0.28, 0.0, 1.15)
    return out.astype(np.float32)


def overflow_drain_v3(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
    if drain_amt < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 15.0 * scale_w))

    core_mask = (lum >= 0.76) & ((lum - bg_lum) >= 0.14)
    if not core_mask.any():
        return rgb

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    dx_ax = -float(axis_offset) * 6.4 if abs(axis_offset) > 1e-4 else -0.01
    dy_ax = -float(axis_offset) * 3.2 if abs(axis_offset) > 1e-4 else -0.01
    rx = nx - dx_ax
    ry = ny - dy_ax
    unorm = np.maximum(np.hypot(rx, ry), 1e-6)
    ux, uy = rx / unorm, ry / unorm
    vx, vy = -uy, ux

    # Local centroid of each highlight
    w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
    sig_c = max(2.5, 9.5 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)
    r_loc = np.hypot(dx_loc, dy_loc)

    s_par = dx_loc * ux + dy_loc * uy
    s_perp = dx_loc * vx + dy_loc * vy
    psi = np.arctan2(s_perp, np.maximum(s_par + 1.8, 1e-3))

    # Inter-lobe notches between the central ray and the two side rays (at psi = -0.25 and +0.26)
    # Because smaller/lower lamps have narrower cores, the notches cut deeper into them to separate the 3 fingers,
    # whereas on larger upper lamps they form the subtle bottom-edge scallop/dimple!
    # We measure local lamp size directly from `w_sum` (larger lamp -> higher w_sum at centroid)!
    size_norm = np.clip(lowpass(w_sum, max(1.5, 4.0 * scale_w)) / 0.28, 0.25, 1.2)
    small_factor = np.clip(1.35 - 0.85 * size_norm, 0.35, 1.15)

    notch1 = np.exp(-0.5 * ((psi + 0.25) / 0.115) ** 2) * small_factor
    notch2 = np.exp(-0.5 * ((psi - 0.26) / 0.125) ** 2) * np.maximum(small_factor, 0.65)
    radial_gate = np.clip((s_par - 0.2) / 2.5, 0.0, 1.0) * np.clip((13.5 - r_loc) / 4.0, 0.0, 1.0)
    notch_carve = np.clip((notch1 + notch2) * radial_gate * float(np.clip(coma_amt * 1.35, 0.0, 1.0)), 0.0, 1.0)

    eff_core = core_mask & (notch_carve < 0.44)

    # Outer overflow-drain crescent on the leading (left & top) arc of eff_core
    r_px = max(1, int(round(2.0 * scale_w)))
    dil1 = binary_dilation(eff_core, iterations=r_px)
    dil2 = binary_dilation(eff_core, iterations=r_px + 1)
    dil3 = binary_dilation(eff_core, iterations=r_px + 2)
    ring_inner = dil1 & (~eff_core)
    ring_outer = dil2 & (~dil1)
    ring_far   = dil3 & (~dil2)

    # Leading U-shaped black rim: wraps around the left (s_par < 0.5) and top (dy_loc < -1.0),
    # plus a small dark notch at the bottom-left corner of the core!
    lead_u = np.clip((1.2 - s_par) / 2.5, 0.0, 1.0)
    top_u = np.clip((-dy_loc - 0.2) / 2.2, 0.0, 1.0) * np.clip((4.5 - s_par) / 3.5, 0.0, 1.0)
    rim_dir = np.clip(np.maximum(lead_u, 0.88 * top_u), 0.0, 1.0)

    drain_strength = float(np.clip(drain_amt * 1.50, 0.0, 0.96))
    notch_mask = notch_carve * (core_mask | ring_inner).astype(np.float32) * (0.82 * drain_strength)
    rim_atten = (
        ring_inner.astype(np.float32) * rim_dir * drain_strength
        + ring_outer.astype(np.float32) * rim_dir * (0.38 * drain_strength)
        + notch_mask
    )
    rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

    out = rgb.copy()
    # Apply dark rim: in the inter-lobe notches, preserve blue/cyan so the grooves between fingers are deep cyan-blue!
    out[..., 0] *= (1.0 - rim_atten)
    out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.18 * notch_carve))
    out[..., 2] *= (1.0 - rim_atten * (1.0 - 0.42 * notch_carve))

    # Warm orange-red / terracotta fringe ON and JUST ABOVE the top/upper-left dark rim (in ring_outer & ring_far where dy_loc < -1.5)
    warm_arc = (ring_outer | ring_far | ring_inner).astype(np.float32) * np.clip((-dy_loc - 1.0) / 3.0, 0.0, 1.0) * np.clip((4.5 - dx_loc) / 6.0, 0.2, 1.0)
    warm_arc = lowpass(warm_arc, max(0.8, 1.2 * scale_w)) * float(np.clip(coma_amt * 1.3, 0.0, 1.0))
    out[..., 0] = np.clip(out[..., 0] + 0.22 * warm_arc, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.04 * warm_arc, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] - 0.16 * warm_arc, 0.0, 1.0)

    # Inside eff_core: 100 IRE composite white-clip plateau (~0.87) + 110 IRE inner-edge peaking (~0.975)
    # Keep the interior PURE NEUTRAL WHITE, with only a subtle cyan tint on the outermost 2px of the trailing right/bottom edge!
    eroded = binary_erosion(eff_core, iterations=max(1, int(round(1.6 * scale_w))))
    inner_edge = eff_core & (~eroded)
    plateau_mask = lowpass(eroded.astype(np.float32), max(0.5, 0.8 * scale_w))
    peaking_mask = lowpass(inner_edge.astype(np.float32), max(0.4, 0.6 * scale_w))

    out_lum = np.maximum(out @ LUMA, 1e-4)
    target_lum = out_lum * (1.0 - plateau_mask) + 0.870 * plateau_mask
    target_lum = np.clip(target_lum + 0.105 * peaking_mask, 0.0, 0.99)
    # Neutralize interior core chroma so the core is pure white like the real capture!
    core_neutral = lowpass(eff_core.astype(np.float32), max(0.4, 0.6 * scale_w))[..., None]
    out_scaled = out * (target_lum / out_lum)[..., None]
    out = out_scaled * (1.0 - 0.85 * core_neutral) + target_lum[..., None] * (0.85 * core_neutral)

    # Add the subtle trailing-edge cyan-blue fringe on the lower-right perimeter of the core & inner_edge
    trail_edge = lowpass((inner_edge | ring_inner).astype(np.float32), max(0.8, 1.2 * scale_w))
    trail_zone = trail_edge * np.clip((s_par - 0.5) / 4.5, 0.0, 1.0) * float(np.clip(coma_amt * 1.2, 0.0, 1.0))
    out[..., 0] = np.clip(out[..., 0] - 0.14 * trail_zone, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.02 * trail_zone, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.14 * trail_zone, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = coma_v3(clean_arr, coma_amt=0.75, axis_offset=0.50)
step2 = overflow_drain_v3(step1, drain_amt=0.65, coma_amt=0.75, axis_offset=0.50)

sim_im = Image.fromarray((np.clip(step2, 0, 1) * 255).astype(np.uint8))

c_left = clean_im.crop((0, 0, 245, 265)).resize((245*2, 265*2), Image.NEAREST)
s_left = sim_im.crop((0, 0, 245, 265)).resize((245*2, 265*2), Image.NEAREST)
r_left = im_real.crop((0, 0, 245, 265)).resize((245*2, 265*2), Image.NEAREST)

s_right = sim_im.crop((585, 0, 720, 265)).resize((135*2, 265*2), Image.NEAREST)
r_right = im_real.crop((585, 0, 720, 265)).resize((135*2, 265*2), Image.NEAREST)

canvas = Image.new("RGB", (c_left.width * 3 + s_right.width * 2 + 32, c_left.height), (24, 24, 24))
x_cur = 0
for p in (c_left, s_left, r_left, s_right, r_right):
    canvas.paste(p, (x_cur, 0))
    x_cur += p.width + 8

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom_v3.png")
print("Saved test_sim_zoom_v3.png")

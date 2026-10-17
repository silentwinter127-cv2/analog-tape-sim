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


def coma_v4(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    """
    Seidel Coma + Tangential/Sagittal Caustic Wedge (Born & Wolf §5.3.2) for a tilted C-mount lens.
    - The coma axis theta_u sits at ~36..42 deg (down-right) on the left field and ~16..24 deg on the right field.
    - The three pupil-zone lobes sit at:
        1. Upper-right tangential lobe (~theta_u - 22 deg),
        2. Central sagittal/principal lobe (~theta_u, longest reach),
        3. Lower-left caustic lobe (~theta_u + 26 deg, which builds the tall ~74 deg left wall and lower-left tooth!).
    """
    if coma_amt < 1e-4:
        return rgb
    a = float(coma_amt)
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0

    # Coma field angle theta_u: ~38 deg on left (nx < 0), ~18 deg on right (nx > 0), steepening slightly with ny
    right_t = np.clip((nx + 0.25) / 0.85, 0.0, 1.0)
    theta_u = np.deg2rad((37.0 + 9.0 * ny) * (1.0 - right_t) + (17.0 + 8.0 * ny) * right_t)
    ux = np.cos(theta_u).astype(np.float32)
    uy = np.sin(theta_u).astype(np.float32)
    vx, vy = -uy, ux

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 15.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    emitter = hot_m ** 1.10
    # Slightly wider source head to match the full-bodied Seidel coma envelope
    emitter_head = lowpass(emitter, max(0.6, 1.05 * scale_w)) * 1.18
    reach = (3.0 + 12.5 * a) * scale_w

    # On the right side (right_t > 0.5), the central/upper-right nose is dominant (forming the D-wedge);
    # on the left side, all 3 lobes (including the +26 deg lower-left caustic tooth) are strong!
    lobe_specs = (
        (-0.40, 0.86, 0.90),  # Upper-right lobe (~theta_u - 23 deg)
        ( 0.00, 1.16, 1.00),  # Central lobe (~theta_u)
        ( 0.46, 0.84, 0.92),  # Lower-left caustic lobe (~theta_u + 26 deg, forms the ~74 deg left wall!)
    )

    bundle_w = np.clip(emitter_head * 1.35, 0.0, 1.35)
    tail_b = np.zeros((h, w), np.float32)

    for idx_l, (ang, r_sc, lw) in enumerate(lobe_specs):
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        dx_l = ca * ux + sa * vx
        dy_l = ca * uy + sa * vy
        for frac in (0.14, 0.28, 0.42, 0.56, 0.72, 0.88, 1.05):
            dist = reach * r_sc * frac
            src_plane = emitter_head if frac <= 0.45 else emitter
            val = bilinear_sample(src_plane, yy - dy_l * dist, xx - dx_l * dist)
            w_core = lw * np.clip(1.32 - 1.18 * frac, 0.0, 1.0) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 1.42)
            # Cobalt/slate-blue chromatic dispersion tail on the trailing half of the rays
            w_blue = lw * np.exp(-0.5 * ((frac - 0.82) / 0.30) ** 2) * a
            tail_b += val * w_blue * (0.42 if idx_l < 2 else 0.28)

    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.30), max(0.45, 0.60 * scale_w))
    core_blend = np.clip((bundle_w - 0.24) / 0.62, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    # Outside the clipped white core, add the cobalt/slate-blue chromatic dispersion skirt (B > G > R, muted G!)
    skirt_w = (1.0 - core_blend[..., 0] * 0.94) * np.clip(tail_b, 0.0, 0.75) * (1.0 - 0.35 * right_t)
    out[..., 2] = np.clip(out[..., 2] + skirt_w * 0.42, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + skirt_w * 0.08, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] - skirt_w * 0.32, 0.0, 1.15)
    return out.astype(np.float32)


def overflow_drain_v4(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
    if drain_amt < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 15.0 * scale_w))

    core_mask = (lum >= 0.75) & ((lum - bg_lum) >= 0.14)
    if not core_mask.any():
        return rgb

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    right_t = np.clip((nx + 0.25) / 0.85, 0.0, 1.0)
    theta_u = np.deg2rad((37.0 + 9.0 * ny) * (1.0 - right_t) + (17.0 + 8.0 * ny) * right_t)
    ux = np.cos(theta_u).astype(np.float32)
    uy = np.sin(theta_u).astype(np.float32)
    vx, vy = -uy, ux

    # Local centroid of each highlight
    w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
    sig_c = max(2.5, 10.0 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)
    r_loc = np.hypot(dx_loc, dy_loc)

    s_par = dx_loc * ux + dy_loc * uy
    s_perp = dx_loc * vx + dy_loc * vy
    psi = np.arctan2(s_perp, np.maximum(s_par + 2.2, 1e-3))

    # Measure local lamp size from w_sum: larger upper lamps keep a broader unified cap with a bottom notch,
    # while smaller lower lamps have deeper inter-lobe notches that split the trailing fan into 3 fingers!
    size_norm = np.clip(lowpass(w_sum, max(1.5, 4.5 * scale_w)) / 0.34, 0.25, 1.25)
    small_factor = np.clip(1.42 - 0.92 * size_norm, 0.28, 1.15)

    notch1 = np.exp(-0.5 * ((psi + 0.22) / 0.105) ** 2) * small_factor * (1.0 - 0.45 * right_t)
    notch2 = np.exp(-0.5 * ((psi - 0.24) / 0.115) ** 2) * np.maximum(small_factor, 0.72)
    radial_gate = np.clip((s_par - 0.4) / 2.4, 0.0, 1.0) * np.clip((14.5 - r_loc) / 4.0, 0.0, 1.0)
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

    # Leading U-shaped black rim: wraps around the left wall (dx_loc < -1.0 or s_par < 0.8) and top roof (dy_loc < -1.0),
    # plus a dark dimple at the bottom-left/bottom notch of the right-side lamps!
    lead_left = np.clip((-dx_loc + 0.5) / 2.5, 0.0, 1.0)
    top_roof  = np.clip((-dy_loc - 0.5) / 2.5, 0.0, 1.0) * np.clip((4.0 - dx_loc) / 5.5, 0.0, 1.0)
    rim_dir = np.clip(np.maximum(lead_left, 0.85 * top_roof), 0.0, 1.0)

    drain_strength = float(np.clip(drain_amt * 1.52, 0.0, 0.96))
    notch_mask = notch_carve * (core_mask | ring_inner).astype(np.float32) * (0.84 * drain_strength)
    rim_atten = (
        ring_inner.astype(np.float32) * rim_dir * drain_strength
        + ring_outer.astype(np.float32) * rim_dir * (0.40 * drain_strength)
        + notch_mask
    )
    rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

    out = rgb.copy()
    # Apply dark rim; in the trailing notches, keep cobalt-blue so the inter-finger grooves read as deep blue-black!
    out[..., 0] *= (1.0 - rim_atten)
    out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.14 * notch_carve))
    out[..., 2] *= (1.0 - rim_atten * (1.0 - 0.38 * notch_carve))

    # Warm orange-red / terracotta fringe ON and JUST ABOVE the upper roof (dy_loc < -1.0):
    # On left lamps it sits on the top/top-left roof; on right lamps it extends along the top-right sloping roof!
    warm_x_gate = np.where(right_t > 0.5, np.clip((dx_loc + 5.0) / 4.0, 0.2, 1.0), np.clip((5.5 - dx_loc) / 6.0, 0.2, 1.0))
    warm_arc = (ring_outer | ring_far | ring_inner).astype(np.float32) * np.clip((-dy_loc - 0.8) / 2.8, 0.0, 1.0) * warm_x_gate
    warm_arc = lowpass(warm_arc, max(0.9, 1.4 * scale_w)) * float(np.clip(coma_amt * 1.35, 0.0, 1.0))
    out[..., 0] = np.clip(out[..., 0] + 0.24 * warm_arc, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.03 * warm_arc, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] - 0.18 * warm_arc, 0.0, 1.0)

    # Inside eff_core: 100 IRE composite white-clip plateau (~0.870) + 110 IRE inner-edge peaking (~0.975)
    eroded = binary_erosion(eff_core, iterations=max(1, int(round(1.6 * scale_w))))
    inner_edge = eff_core & (~eroded)
    plateau_mask = lowpass(eroded.astype(np.float32), max(0.5, 0.8 * scale_w))
    peaking_mask = lowpass(inner_edge.astype(np.float32), max(0.4, 0.6 * scale_w))

    out_lum = np.maximum(out @ LUMA, 1e-4)
    target_lum = out_lum * (1.0 - plateau_mask) + 0.870 * plateau_mask
    target_lum = np.clip(target_lum + 0.105 * peaking_mask, 0.0, 0.99)
    core_neutral = lowpass(eff_core.astype(np.float32), max(0.4, 0.6 * scale_w))[..., None]
    out_scaled = out * (target_lum / out_lum)[..., None]
    out = out_scaled * (1.0 - 0.88 * core_neutral) + target_lum[..., None] * (0.88 * core_neutral)

    # Subtle cobalt-blue fringe along the trailing bottom-right perimeter of the core
    trail_edge = lowpass((inner_edge | ring_inner).astype(np.float32), max(0.8, 1.3 * scale_w))
    trail_zone = trail_edge * np.clip((s_par - 0.8) / 4.5, 0.0, 1.0) * float(np.clip(coma_amt * 1.15, 0.0, 1.0)) * (1.0 - 0.30 * right_t)
    out[..., 0] = np.clip(out[..., 0] - 0.15 * trail_zone, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.01 * trail_zone, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.15 * trail_zone, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = coma_v4(clean_arr, coma_amt=0.75, axis_offset=0.50)
step2 = overflow_drain_v4(step1, drain_amt=0.65, coma_amt=0.75, axis_offset=0.50)

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

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom_v4.png")
print("Saved test_sim_zoom_v4.png")

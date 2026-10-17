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


def coma_v5(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    """
    Physical Seidel Coma + Tangential/Sagittal Caustic Wedge (Born & Wolf §5.3.2):
      - Lobe 1 (Top-Right Tangential Caustic): ~10..14 deg (forms the wide arched top roof & right nose)
      - Lobe 2 (Central Principal Ray):        ~38 deg on left, ~22 deg on right (forms the central finger / nose)
      - Lobe 3 (Left-Wall Sagittal Caustic):   ~73 deg on left, ~64 deg on right (forms the tall vertical-ish left wall & bottom-left tooth!)
    """
    if coma_amt < 1e-4:
        return rgb
    a = float(coma_amt)
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    right_t = np.clip((nx + 0.15) / 0.75, 0.0, 1.0)
    # Perspective depth factor: 0 near top of ceiling (y=0, close to camera), 1 further down ceiling (y=260, far from camera)
    depth_t = np.clip((ny + 0.85) / 0.85, 0.0, 1.0)

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    emitter = hot_m ** 1.05
    # Near lamps (small depth_t) have a broader head; far lamps have a tighter head that splits into 3 fingers
    emitter_wide = lowpass(emitter, max(0.8, 1.55 * scale_w)) * 1.25
    reach = (3.5 + 11.5 * a) * scale_w * (1.05 - 0.18 * depth_t)

    # Three caustic lobe angle fields (in radians):
    ang1 = np.deg2rad(13.0 * (1.0 - right_t) + 7.0 * right_t + 4.0 * depth_t)
    ang2 = np.deg2rad(39.0 * (1.0 - right_t) + 21.0 * right_t + 5.0 * depth_t)
    ang3 = np.deg2rad(74.0 * (1.0 - right_t) + 58.0 * right_t - 2.0 * depth_t)

    # Also include 2 intermediate web angles (ang12, ang23) that fill the upper caustic wedge on close/large lamps
    # and fade out at the outer tips (and on far/small lamps) so the trailing edge splits into 3 fingers!
    ang12 = 0.5 * (ang1 + ang2)
    ang23 = 0.5 * (ang2 + ang3)

    bundle_w = np.clip(emitter_wide * 1.35, 0.0, 1.35)
    tail_b = np.zeros((h, w), np.float32)
    tail_g = np.zeros((h, w), np.float32)

    # 1. Trace the 3 main caustic lobes
    lobe_defs = (
        (ang1, 0.92 * (1.0 - right_t) + 1.12 * right_t, 0.94, 0.42),
        (ang2, 1.08 * (1.0 - right_t) + 1.18 * right_t, 1.00, 0.50),
        (ang3, 0.88 * (1.0 - right_t) + 0.78 * right_t, 0.94, 0.26),
    )
    for ang_f, r_sc, lw, blue_w in lobe_defs:
        dx_l = np.cos(ang_f).astype(np.float32)
        dy_l = np.sin(ang_f).astype(np.float32)
        for frac in (0.14, 0.28, 0.42, 0.56, 0.70, 0.85, 1.02):
            dist = reach * r_sc * frac
            # Use wider source near the apex (frac <= 0.42), tighter source toward the finger tips
            src_p = emitter_wide if frac <= 0.42 else emitter
            val = bilinear_sample(src_p, yy - dy_l * dist, xx - dx_l * dist)
            w_core = lw * np.clip(1.34 - 1.16 * frac, 0.0, 1.0) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 1.45)
            # Soft slate-blue / cyan dispersion on the trailing half of each finger
            w_blue = np.exp(-0.5 * ((frac - 0.80) / 0.28) ** 2) * a
            tail_b += val * w_blue * blue_w
            tail_g += val * w_blue * (blue_w * 0.35)

    # 2. Fill the inner caustic web between the lobes up to frac ~ 0.68 on upper/larger lamps (and on the right D-wedge)
    web_reach_gate = (1.0 - 0.52 * depth_t) + 0.35 * right_t
    for ang_w in (ang12, ang23):
        dx_w = np.cos(ang_w).astype(np.float32)
        dy_w = np.sin(ang_w).astype(np.float32)
        for frac in (0.18, 0.36, 0.54, 0.70):
            dist = reach * web_reach_gate * frac
            val = bilinear_sample(emitter_wide, yy - dy_w * dist, xx - dx_w * dist)
            w_web = np.clip(1.25 - 1.35 * frac, 0.0, 1.0) * a
            bundle_w = np.maximum(bundle_w, val * w_web * 1.35)

    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.30), max(0.45, 0.58 * scale_w))
    core_blend = np.clip((bundle_w - 0.24) / 0.58, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    # Outside the clipped white core, add the soft slate-blue / cobalt-cyan dispersion skirt
    skirt_w = (1.0 - core_blend[..., 0] * 0.92) * (1.0 - 0.28 * right_t)
    out[..., 2] = np.clip(out[..., 2] + np.clip(tail_b, 0.0, 0.70) * skirt_w * 0.40, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + np.clip(tail_g, 0.0, 0.70) * skirt_w * 0.16, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] - np.clip(tail_b, 0.0, 0.70) * skirt_w * 0.30, 0.0, 1.15)
    return out.astype(np.float32)


def overflow_drain_v5(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
    if drain_amt < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))

    core_mask = (lum >= 0.75) & ((lum - bg_lum) >= 0.14)
    if not core_mask.any():
        return rgb

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    right_t = np.clip((nx + 0.15) / 0.75, 0.0, 1.0)
    depth_t = np.clip((ny + 0.85) / 0.85, 0.0, 1.0)

    # Local centroid of each highlight
    w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
    sig_c = max(2.5, 10.5 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)
    r_loc = np.hypot(dx_loc, dy_loc)

    # Angle of pixel relative to the lamp's TOP-LEFT APEX (cx_loc - 3.0, cy_loc - 3.0):
    # Remember our 3 lobes radiate from the top-left apex at ~13 deg, ~39 deg, and ~74 deg!
    ax_dx = dx_loc + 3.2
    ax_dy = dy_loc + 3.2
    ax_r = np.hypot(ax_dx, ax_dy)
    theta_px = np.arctan2(ax_dy, np.maximum(ax_dx, 1e-3))  # in radians, 0 = right, pi/2 = down

    # The 2 inter-lobe notch angles (between Lobe 1 & 2 at ~26 deg, and between Lobe 2 & 3 at ~56 deg):
    n_ang1 = np.deg2rad(26.0 * (1.0 - right_t) + 14.0 * right_t + 4.0 * depth_t)
    n_ang2 = np.deg2rad(56.0 * (1.0 - right_t) + 40.0 * right_t + 2.0 * depth_t)

    # On upper lamps (depth_t small), only n_ang2 carves the single bottom-edge dimple/notch at r_ax > 8.5 px;
    # On lower lamps (depth_t large), BOTH n_ang1 and n_ang2 carve deeper inward (from r_ax > 5.5 px) to split the 3 fingers!
    r_start_1 = 9.2 - 3.8 * depth_t
    r_start_2 = 8.0 - 2.6 * depth_t
    gate1 = np.clip((ax_r - r_start_1) / 2.2, 0.0, 1.0) * np.clip((16.5 - ax_r) / 3.5, 0.0, 1.0) * (0.25 + 0.85 * depth_t) * (1.0 - 0.55 * right_t)
    gate2 = np.clip((ax_r - r_start_2) / 2.2, 0.0, 1.0) * np.clip((16.0 - ax_r) / 3.5, 0.0, 1.0) * (0.75 + 0.35 * depth_t)

    notch1 = np.exp(-0.5 * ((theta_px - n_ang1) / 0.11) ** 2) * gate1
    notch2 = np.exp(-0.5 * ((theta_px - n_ang2) / 0.12) ** 2) * gate2
    notch_carve = np.clip((notch1 + notch2) * float(np.clip(coma_amt * 1.35, 0.0, 1.0)), 0.0, 1.0)

    eff_core = core_mask & (notch_carve < 0.44)

    # Outer overflow-drain crescent on the leading (left wall & top roof) of eff_core
    r_px = max(1, int(round(2.0 * scale_w)))
    dil1 = binary_dilation(eff_core, iterations=r_px)
    dil2 = binary_dilation(eff_core, iterations=r_px + 1)
    dil3 = binary_dilation(eff_core, iterations=r_px + 2)
    ring_inner = dil1 & (~eff_core)
    ring_outer = dil2 & (~dil1)
    ring_far   = dil3 & (~dil2)

    # Leading L/U-shaped black rim:
    # 1. Left vertical-ish wall: dx_loc < -1.2 (all the way down the left wall!)
    # 2. Top arched roof: dy_loc < -1.2 and dx_loc < +4.5
    left_wall = np.clip((-dx_loc - 0.2) / 2.4, 0.0, 1.0)
    top_roof  = np.clip((-dy_loc - 0.8) / 2.4, 0.0, 1.0) * np.clip((5.0 - dx_loc) / 5.5, 0.0, 1.0)
    rim_dir = np.clip(np.maximum(left_wall, 0.85 * top_roof), 0.0, 1.0)

    drain_strength = float(np.clip(drain_amt * 1.52, 0.0, 0.96))
    notch_mask = notch_carve * (core_mask | ring_inner).astype(np.float32) * (0.85 * drain_strength)
    rim_atten = (
        ring_inner.astype(np.float32) * rim_dir * drain_strength
        + ring_outer.astype(np.float32) * rim_dir * (0.42 * drain_strength)
        + notch_mask
    )
    rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

    out = rgb.copy()
    # In the trailing notches, keep cobalt-blue so the inter-finger grooves read as deep blue-black!
    out[..., 0] *= (1.0 - rim_atten)
    out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.16 * notch_carve))
    out[..., 2] *= (1.0 - rim_atten * (1.0 - 0.42 * notch_carve))

    # Warm orange-red / terracotta fringe ON and JUST ABOVE the upper roof (dy_loc < -1.0):
    warm_x_gate = np.where(right_t > 0.5, np.clip((dx_loc + 5.0) / 4.0, 0.25, 1.0), np.clip((6.0 - dx_loc) / 6.0, 0.25, 1.0))
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
    out = out_scaled * (1.0 - 0.90 * core_neutral) + target_lum[..., None] * (0.90 * core_neutral)

    # Subtle cobalt-blue fringe along the trailing bottom-right perimeter of the core
    trail_edge = lowpass((inner_edge | ring_inner).astype(np.float32), max(0.8, 1.3 * scale_w))
    trail_zone = trail_edge * np.clip((dx_loc + dy_loc - 1.2) / 5.0, 0.0, 1.0) * float(np.clip(coma_amt * 1.10, 0.0, 1.0)) * (1.0 - 0.30 * right_t)
    out[..., 0] = np.clip(out[..., 0] - 0.14 * trail_zone, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.01 * trail_zone, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.14 * trail_zone, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = coma_v5(clean_arr, coma_amt=0.75, axis_offset=0.50)
step2 = overflow_drain_v5(step1, drain_amt=0.65, coma_amt=0.75, axis_offset=0.50)

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

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom_v5.png")
print("Saved test_sim_zoom_v5.png")

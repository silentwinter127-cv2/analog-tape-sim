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


def coma_v8(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    if coma_amt < 1e-4:
        return rgb
    a = float(coma_amt)
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    right_t = np.clip((nx + 0.15) / 0.75, 0.0, 1.0)
    depth_t = np.clip((ny + 0.88) / 0.88, 0.0, 1.0)

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    emitter_narrow = lowpass(hot_m ** 1.15, max(0.4, 0.65 * scale_w))
    emitter_base   = lowpass(hot_m ** 0.85, max(0.8, 1.45 * scale_w))
    reach = (6.0 + 14.5 * a) * scale_w * (1.08 - 0.16 * depth_t)

    # Three Seidel caustic lobe angles:
    # Lobe 1: top roof (~11 deg on left, ~13 deg on right)
    # Lobe 2: middle principal ray (~35 deg on left, ~22 deg on right)
    # Lobe 3: left vertical-ish wall (~74 deg on left, ~66 deg on right)
    ang1 = np.deg2rad(11.0 * (1.0 - right_t) + 13.0 * right_t + 3.0 * depth_t)
    ang2 = np.deg2rad(35.0 * (1.0 - right_t) + 22.0 * right_t + 4.0 * depth_t)
    ang3 = np.deg2rad(74.0 * (1.0 - right_t) + 66.0 * right_t - 2.0 * depth_t)

    bundle_w = np.clip(emitter_base * 1.30, 0.0, 1.35)
    tail_feather = np.zeros((h, w), np.float32)
    tail_b = np.zeros((h, w), np.float32)
    tail_g = np.zeros((h, w), np.float32)

    lobe_defs = (
        (ang1, 0.94 * (1.0 - right_t) + 1.12 * right_t, 0.96, 0.55),
        (ang2, 1.08 * (1.0 - right_t) + 1.18 * right_t, 1.00, 0.65),
        (ang3, 0.72 * (1.0 - right_t) + 0.64 * right_t, 0.95, 0.04),
    )
    for ang_f, r_sc, lw, blue_w in lobe_defs:
        dx_l = np.cos(ang_f).astype(np.float32)
        dy_l = np.sin(ang_f).astype(np.float32)
        for frac in (0.10, 0.20, 0.30, 0.42, 0.54, 0.66, 0.80, 0.94, 1.10):
            dist = reach * r_sc * frac
            src_p = emitter_base if frac <= 0.36 else emitter_narrow
            val = bilinear_sample(src_p, yy - dy_l * dist, xx - dx_l * dist)
            w_core = lw * np.clip(1.68 - 1.58 * frac, 0.0, 1.25) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 2.20)
            # Soft additive luminous + cyan-blue comet tail feathering out from frac 0.50 to 1.10!
            w_tail = np.exp(-0.5 * ((frac - 0.76) / 0.28) ** 2) * a
            tail_feather += val * w_tail * (blue_w * 0.45)
            tail_b += val * w_tail * blue_w
            tail_g += val * w_tail * (blue_w * 0.48)

    # Inner caustic web between the 3 rays
    web_reach_scale = (0.76 - 0.38 * depth_t) * (1.0 - right_t) + (0.88 - 0.24 * depth_t) * right_t
    for w_mix in (0.25, 0.50, 0.75):
        for ang_a, ang_b in ((ang1, ang2), (ang2, ang3)):
            ang_w = (1.0 - w_mix) * ang_a + w_mix * ang_b
            dx_w = np.cos(ang_w).astype(np.float32)
            dy_w = np.sin(ang_w).astype(np.float32)
            for frac in (0.15, 0.30, 0.45, 0.60, 0.75):
                dist = reach * web_reach_scale * frac
                val = bilinear_sample(emitter_base, yy - dy_w * dist, xx - dx_w * dist)
                w_web = np.clip(1.58 - 1.58 * frac, 0.0, 1.15) * a
                bundle_w = np.maximum(bundle_w, val * w_web * 2.00)

    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.35), max(0.45, 0.58 * scale_w))
    core_blend = np.clip((bundle_w - 0.28) / 0.52, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    # Smoothly feather the trailing luminous cyan-blue comet tail into the ceiling (NEVER darker than the ceiling!)
    tf_smooth = lowpass(np.clip(tail_feather, 0.0, 0.85), max(0.9, 1.5 * scale_w))
    tb_smooth = lowpass(np.clip(tail_b, 0.0, 0.85), max(1.0, 1.6 * scale_w))
    tg_smooth = lowpass(np.clip(tail_g, 0.0, 0.85), max(1.0, 1.6 * scale_w))
    skirt_w = (1.0 - core_blend[..., 0] * 0.88) * (1.0 - 0.25 * right_t)
    # Note: Even R gets a slight positive lift from tf_smooth so the tail is a BRIGHT luminous sky-blue/cyan feather,
    # never a dark ring!
    out[..., 2] = np.clip(out[..., 2] + (0.22 * tf_smooth + 0.36 * tb_smooth) * skirt_w, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + (0.20 * tf_smooth + 0.16 * tg_smooth) * skirt_w, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] + (0.04 * tf_smooth - 0.16 * tb_smooth) * skirt_w, 0.0, 1.15)
    return out.astype(np.float32)


def overflow_drain_v8(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
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
    depth_t = np.clip((ny + 0.88) / 0.88, 0.0, 1.0)

    # Local centroid of each highlight using compact sigma (6.5 * scale_w) so nearby lamps never pull each other!
    w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
    sig_c = max(2.0, 6.5 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)

    # Coordinates relative to the lamp's TOP-LEFT APEX (cx_loc - 4.5, cy_loc - 4.2)
    ax_dx = dx_loc + 4.5
    ax_dy = dy_loc + 4.2
    ax_r = np.hypot(ax_dx, ax_dy)
    theta_px = np.arctan2(ax_dy, np.maximum(ax_dx, 1e-3))

    # Inter-lobe notch angles:
    n_ang1 = np.deg2rad(23.0 * (1.0 - right_t) + 17.5 * right_t + 3.5 * depth_t)
    n_ang2 = np.deg2rad(54.5 * (1.0 - right_t) + 44.0 * right_t + 1.0 * depth_t)

    r_start_1 = 9.0 - 3.5 * depth_t
    r_start_2 = 7.8 - 2.4 * depth_t
    # Stop the dark notch carving BEFORE the outer tip of the fingers (at ax_r ~ 14.5 px) so the tips stay open & feathered!
    gate1 = np.clip((ax_r - r_start_1) / 2.0, 0.0, 1.0) * np.clip((14.5 - ax_r) / 2.5, 0.0, 1.0) * (0.18 + 0.92 * depth_t) * (1.0 - 0.50 * right_t)
    gate2 = np.clip((ax_r - r_start_2) / 2.0, 0.0, 1.0) * np.clip((14.0 - ax_r) / 2.5, 0.0, 1.0) * (0.82 + 0.28 * depth_t)

    notch1 = np.exp(-0.5 * ((theta_px - n_ang1) / 0.105) ** 2) * gate1
    notch2 = np.exp(-0.5 * ((theta_px - n_ang2) / 0.115) ** 2) * gate2
    notch_carve = np.clip((notch1 + notch2) * float(np.clip(coma_amt * 1.35, 0.0, 1.0)), 0.0, 1.0)

    eff_core = core_mask & (notch_carve < 0.44)

    # Outer overflow-drain crescent ONLY on the leading (left wall & top roof) of eff_core!
    r_px = max(1, int(round(2.0 * scale_w)))
    dil1 = binary_dilation(eff_core, iterations=r_px)
    dil2 = binary_dilation(eff_core, iterations=r_px + 1)
    dil3 = binary_dilation(eff_core, iterations=r_px + 2)
    ring_inner = dil1 & (~eff_core)
    ring_outer = dil2 & (~dil1)
    ring_far   = dil3 & (~dil2)

    # Leading L/U-shaped black rim:
    # Strictly on the left wall (dx_loc < -1.2) and top roof (dy_loc < -1.2 and dx_loc < +5.5),
    # ZERO on the trailing right/bottom-right so the fingers have NO dark outline around their tips!
    left_wall = np.clip((-dx_loc - 0.8) / 2.2, 0.0, 1.0) * np.clip((4.5 - dy_loc) / 3.0, 0.0, 1.0)
    top_roof  = np.clip((-dy_loc - 1.0) / 2.2, 0.0, 1.0) * np.clip((5.5 - dx_loc) / 5.0, 0.0, 1.0)
    rim_dir = np.clip(np.maximum(left_wall, 0.85 * top_roof), 0.0, 1.0)

    drain_strength = float(np.clip(drain_amt * 1.52, 0.0, 0.96))
    # Only carve notches INSIDE core_mask (between fingers), NOT in ring_inner outside the tips!
    notch_mask = notch_carve * core_mask.astype(np.float32) * (0.82 * drain_strength)
    rim_atten = (
        ring_inner.astype(np.float32) * rim_dir * drain_strength
        + ring_outer.astype(np.float32) * rim_dir * (0.40 * drain_strength)
        + notch_mask
    )
    rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

    out = rgb.copy()
    out[..., 0] *= (1.0 - rim_atten)
    out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.18 * notch_carve))
    out[..., 2] *= (1.0 - rim_atten * (1.0 - 0.42 * notch_carve))

    # Warm orange-red / terracotta fringe ON and JUST ABOVE the upper roof (dy_loc < -1.0):
    warm_x_gate = np.where(right_t > 0.5, np.clip((dx_loc + 6.0) / 4.0, 0.25, 1.0), np.clip((6.5 - dx_loc) / 6.0, 0.25, 1.0))
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
    out = out_scaled * (1.0 - 0.92 * core_neutral) + target_lum[..., None] * (0.92 * core_neutral)

    # Soft cyan-blue dispersion on the trailing perimeter of eff_core (strictly luminance-preserving: +B, +G, -R)
    trail_edge = lowpass(inner_edge.astype(np.float32), max(0.9, 1.4 * scale_w))
    right_tail_gate = np.clip((dx_loc - 0.2) / 3.0, 0.0, 1.0) * np.clip((dy_loc + 2.0) / 3.0, 0.0, 1.0)
    trail_zone = trail_edge * right_tail_gate * float(np.clip(coma_amt * 1.15, 0.0, 1.0)) * (1.0 - 0.30 * right_t)
    out[..., 0] = np.clip(out[..., 0] - 0.12 * trail_zone, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.03 * trail_zone, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.16 * trail_zone, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = coma_v8(clean_arr, coma_amt=0.75, axis_offset=0.50)
step2 = overflow_drain_v8(step1, drain_amt=0.65, coma_amt=0.75, axis_offset=0.50)

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

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom_v8.png")
print("Saved test_sim_zoom_v8.png")

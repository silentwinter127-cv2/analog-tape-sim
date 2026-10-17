import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing, label
from PIL import Image
from atsim.core.signal import lowpass

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
real = np.asarray(im_real, dtype=np.float32) / 255.0
H, W, _ = real.shape
lum_real = real @ LUMA

# 1. Reconstruct clean_ceil.png where each lamp is placed at its TOP-LEFT APEX (before coma stretches it right and down!)
ceil_zone = np.zeros((H, W), dtype=bool)
ceil_zone[0:268, 0:245] = True
ceil_zone[0:265, 588:720] = True

raw_cores = binary_closing((lum_real > 0.76) & ceil_zone, structure=np.ones((3, 3), dtype=bool))
lbl, n_lamps = label(raw_cores)
raw_list = []
for idx in range(1, n_lamps + 1):
    ys, xs = np.nonzero(lbl == idx)
    if len(ys) < 5:
        continue
    # Place the clean round source near the upper-left apex (28th percentile of y and x) so coma expands it down & right!
    cy = float(np.percentile(ys, 28))
    cx = float(np.percentile(xs, 26))
    raw_list.append((len(ys), cy, cx, ys, xs))

raw_list.sort(key=lambda t: -t[0])
lamps = []
lamp_footprint = np.zeros((H, W), dtype=bool)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

for area, cy, cx, ys, xs in raw_list:
    if any(np.hypot(cy - ly, cx - lx) < 16.5 for ly, lx, _ in lamps):
        continue
    equiv_r = np.sqrt(area / np.pi)
    r0 = float(np.clip(equiv_r * 0.46, 2.8, 4.8))
    lamps.append((cy, cx, r0))
    cy_mid, cx_mid = float(ys.mean()), float(xs.mean())
    h_span = float(ys.max() - ys.min() + 1)
    w_span = float(xs.max() - xs.min() + 1)
    lamp_footprint |= ((yy - cy_mid)**2 + (xx - cx_mid)**2) <= (max(h_span, w_span) * 0.90 + 10.0)**2

clean = real.copy()
clean[:5, :] = clean[5:10, :].mean(axis=0, keepdims=True)
clean[-5:, :] = clean[-10:-5, :].mean(axis=0, keepdims=True)
clean[:, :10] = clean[:, 10:15].mean(axis=1, keepdims=True)
clean[:, -9:] = clean[:, -14:-9].mean(axis=1, keepdims=True)

exit_mask = np.zeros((H, W), dtype=np.float32)
exit_mask[300:340, 635:675] = 1.0
exit_mask = gaussian_filter(exit_mask, 5.0)
clean_lum = clean @ LUMA

valid = ((~lamp_footprint) & ceil_zone & (lum_real > 0.40) & (lum_real < 0.52)).astype(np.float32)
ceil_bg = gaussian_filter(clean_lum * valid, 24.0) / np.maximum(gaussian_filter(valid, 24.0), 1e-4)
blend_w = gaussian_filter((lamp_footprint & ceil_zone).astype(np.float32), 4.0)
blend_w = np.clip(blend_w * 2.2, 0.0, 1.0) * ceil_zone.astype(np.float32)
clean_lum = clean_lum * (1.0 - blend_w) + ceil_bg * blend_w
col_edge_template = np.median(clean_lum[275:320, 236:250], axis=0, keepdims=True)
clean_lum[0:268, 238:250] = col_edge_template[:, 2:]
clean_lum = gaussian_filter(clean_lum, 0.55)
clean_rgb = np.stack([clean_lum, clean_lum, clean_lum], axis=-1)
clean_rgb = clean_rgb * (1.0 - exit_mask[..., None]) + clean * exit_mask[..., None]

for cy, cx, r0 in lamps:
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    disk = np.clip((r0 + 0.5 - dist) / 0.9, 0.0, 1.0)
    clean_rgb = clean_rgb * (1.0 - disk[..., None]) + 1.0 * disk[..., None]

clean_u8 = (np.clip(clean_rgb, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(clean_u8).save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png")


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


def coma_v6(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    """
    Seidel Coma + Tangential/Sagittal Caustic Wedge (Born & Wolf §5.3.2).
    Because an overexposed emitter has ~5x full-well radiance, the 3 caustic lobes stay ABOVE the saturation
    threshold all the way out to `frac = 0.85` (10..14 px from the top-left apex!), forming:
      - A tall ~76 deg left wall (Lobe 3),
      - A wide ~10 deg arched top roof (Lobe 1),
      - A central ~38 deg (left) / ~20 deg (right) principal finger (Lobe 2),
      - An inner caustic web that merges the upper half into a broad bell / D-wedge on close lamps and
        leaves the outer half split into 3 distinct fingers on smaller/far lamps!
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
    depth_t = np.clip((ny + 0.88) / 0.88, 0.0, 1.0)

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    # Narrow finger source vs wider base source
    emitter_narrow = lowpass(hot_m ** 1.2, max(0.4, 0.55 * scale_w))
    emitter_base   = lowpass(hot_m ** 0.9, max(0.8, 1.35 * scale_w))
    reach = (4.5 + 11.5 * a) * scale_w * (1.10 - 0.22 * depth_t)

    # Three caustic lobe angles (in radians, 0 = right, pi/2 = down):
    # Lobe 1: top roof (~10 deg on left, ~12 deg on right)
    # Lobe 2: middle principal ray (~39 deg on left, ~23 deg on right)
    # Lobe 3: left vertical-ish wall (~75 deg on left, ~68 deg on right)
    ang1 = np.deg2rad(10.0 * (1.0 - right_t) + 12.0 * right_t + 3.0 * depth_t)
    ang2 = np.deg2rad(39.0 * (1.0 - right_t) + 23.0 * right_t + 4.0 * depth_t)
    ang3 = np.deg2rad(75.0 * (1.0 - right_t) + 68.0 * right_t - 2.0 * depth_t)

    bundle_w = np.clip(emitter_base * 1.25, 0.0, 1.35)
    tail_b = np.zeros((h, w), np.float32)
    tail_g = np.zeros((h, w), np.float32)

    # 1. Trace the 3 main caustic fingers with high overexposure gain (2.6x) so they stay > 0.75 out to frac=0.85!
    lobe_defs = (
        (ang1, 0.88 * (1.0 - right_t) + 1.08 * right_t, 0.96, 0.44),
        (ang2, 1.04 * (1.0 - right_t) + 1.15 * right_t, 1.00, 0.52),
        (ang3, 0.76 * (1.0 - right_t) + 0.68 * right_t, 0.95, 0.28),
    )
    for ang_f, r_sc, lw, blue_w in lobe_defs:
        dx_l = np.cos(ang_f).astype(np.float32)
        dy_l = np.sin(ang_f).astype(np.float32)
        for frac in (0.12, 0.24, 0.36, 0.48, 0.62, 0.76, 0.90, 1.05):
            dist = reach * r_sc * frac
            src_p = emitter_base if frac <= 0.36 else emitter_narrow
            val = bilinear_sample(src_p, yy - dy_l * dist, xx - dx_l * dist)
            # High overexposure gain: stays > 0.80 up to frac = 0.82!
            w_core = lw * np.clip(1.65 - 1.55 * frac, 0.0, 1.2) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 2.15)
            # Cobalt/slate-blue chromatic dispersion skirt at the outer tips & flanks (frac 0.55..1.05)
            w_blue = np.exp(-0.5 * ((frac - 0.78) / 0.28) ** 2) * a
            tail_b += val * w_blue * blue_w
            tail_g += val * w_blue * (blue_w * 0.38)

    # 2. Inner caustic web between the 3 rays:
    #    On upper/larger lamps (depth_t small) and on the right D-wedge (right_t large), the web extends further
    #    so the upper body is a solid broad bell/D-wedge! On lower/smaller lamps (depth_t large), the web is shorter
    #    so the 3 fingers separate cleanly!
    web_reach_scale = (0.72 - 0.36 * depth_t) * (1.0 - right_t) + (0.85 - 0.25 * depth_t) * right_t
    for w_mix in (0.28, 0.50, 0.72):
        for ang_a, ang_b in ((ang1, ang2), (ang2, ang3)):
            ang_w = (1.0 - w_mix) * ang_a + w_mix * ang_b
            dx_w = np.cos(ang_w).astype(np.float32)
            dy_w = np.sin(ang_w).astype(np.float32)
            for frac in (0.16, 0.32, 0.48, 0.64, 0.80):
                dist = reach * web_reach_scale * frac
                val = bilinear_sample(emitter_base, yy - dy_w * dist, xx - dx_w * dist)
                w_web = np.clip(1.55 - 1.60 * frac, 0.0, 1.1) * a
                bundle_w = np.maximum(bundle_w, val * w_web * 1.95)

    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.35), max(0.42, 0.52 * scale_w))
    core_blend = np.clip((bundle_w - 0.32) / 0.48, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    # Outside the saturated white core, add the soft slate-blue / cobalt-cyan dispersion skirt
    skirt_w = (1.0 - core_blend[..., 0] * 0.90) * (1.0 - 0.22 * right_t)
    out[..., 2] = np.clip(out[..., 2] + np.clip(tail_b, 0.0, 0.75) * skirt_w * 0.42, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + np.clip(tail_g, 0.0, 0.75) * skirt_w * 0.16, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] - np.clip(tail_b, 0.0, 0.75) * skirt_w * 0.32, 0.0, 1.15)
    return out.astype(np.float32)


def overflow_drain_v6(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
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

    # Local centroid of each highlight
    w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
    sig_c = max(2.5, 11.0 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)

    # Coordinates relative to the lamp's TOP-LEFT APEX (cx_loc - 4.2, cy_loc - 4.2)
    ax_dx = dx_loc + 4.2
    ax_dy = dy_loc + 4.2
    ax_r = np.hypot(ax_dx, ax_dy)
    theta_px = np.arctan2(ax_dy, np.maximum(ax_dx, 1e-3))

    # Inter-lobe notch angles (between Lobe 1 & 2, and between Lobe 2 & 3):
    n_ang1 = np.deg2rad(24.5 * (1.0 - right_t) + 17.5 * right_t + 3.5 * depth_t)
    n_ang2 = np.deg2rad(57.0 * (1.0 - right_t) + 45.0 * right_t + 1.0 * depth_t)

    # On upper lamps (depth_t small), n_ang2 forms the bottom-edge dark dimple/notch while n_ang1 is shallow;
    # On lower lamps (depth_t large), BOTH n_ang1 and n_ang2 cut inward to separate the 3 long fingers!
    r_start_1 = 8.5 - 3.5 * depth_t
    r_start_2 = 7.2 - 2.4 * depth_t
    gate1 = np.clip((ax_r - r_start_1) / 2.0, 0.0, 1.0) * np.clip((18.0 - ax_r) / 3.5, 0.0, 1.0) * (0.20 + 0.90 * depth_t) * (1.0 - 0.45 * right_t)
    gate2 = np.clip((ax_r - r_start_2) / 2.0, 0.0, 1.0) * np.clip((17.0 - ax_r) / 3.5, 0.0, 1.0) * (0.80 + 0.30 * depth_t)

    notch1 = np.exp(-0.5 * ((theta_px - n_ang1) / 0.115) ** 2) * gate1
    notch2 = np.exp(-0.5 * ((theta_px - n_ang2) / 0.125) ** 2) * gate2
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
    # 1. Left vertical-ish wall (dx_loc < -1.0)
    # 2. Top arched roof (dy_loc < -1.0 and dx_loc < +5.5)
    left_wall = np.clip((-dx_loc - 0.2) / 2.4, 0.0, 1.0)
    top_roof  = np.clip((-dy_loc - 0.8) / 2.4, 0.0, 1.0) * np.clip((6.0 - dx_loc) / 5.5, 0.0, 1.0)
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
    out[..., 0] *= (1.0 - rim_atten)
    out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.16 * notch_carve))
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
    out = out_scaled * (1.0 - 0.90 * core_neutral) + target_lum[..., None] * (0.90 * core_neutral)

    # Subtle cobalt-blue fringe along the trailing bottom-right perimeter of the core
    trail_edge = lowpass((inner_edge | ring_inner).astype(np.float32), max(0.8, 1.3 * scale_w))
    trail_zone = trail_edge * np.clip((dx_loc + dy_loc - 1.2) / 5.0, 0.0, 1.0) * float(np.clip(coma_amt * 1.10, 0.0, 1.0)) * (1.0 - 0.30 * right_t)
    out[..., 0] = np.clip(out[..., 0] - 0.14 * trail_zone, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.01 * trail_zone, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.14 * trail_zone, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = coma_v6(clean_arr, coma_amt=0.75, axis_offset=0.50)
step2 = overflow_drain_v6(step1, drain_amt=0.65, coma_amt=0.75, axis_offset=0.50)

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

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom_v6.png")
print("Saved test_sim_zoom_v6.png")

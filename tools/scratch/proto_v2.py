import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing, label
from PIL import Image
from atsim.core.signal import lowpass

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
real = np.asarray(im_real, dtype=np.float32) / 255.0
H, W, _ = real.shape
lum_real = real @ LUMA

# 1. Reconstruct clean_ceil.png with accurate perspective lamp radii (larger near top of ceiling, smaller further down)
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
    # Leading-edge center (upstream of the down-right comatic extension)
    cy = float(np.percentile(ys, 38))
    cx = float(np.percentile(xs, 34))
    raw_list.append((len(ys), cy, cx, ys, xs))

raw_list.sort(key=lambda t: -t[0])
lamps = []
lamp_footprint = np.zeros((H, W), dtype=bool)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

for area, cy, cx, ys, xs in raw_list:
    if any(np.hypot(cy - ly, cx - lx) < 16.5 for ly, lx, _, _ in lamps):
        continue
    # Perspective size: lamps higher up (smaller cy) are closer and wider; recessed fixtures are slightly elliptical
    equiv_r = np.sqrt(area / np.pi)
    ry = float(np.clip(equiv_r * 0.62, 3.6, 7.2))
    rx = float(np.clip(equiv_r * 0.72, 4.2, 8.4))
    lamps.append((cy, cx, ry, rx))
    h_span = float(ys.max() - ys.min() + 1)
    w_span = float(xs.max() - xs.min() + 1)
    lamp_footprint |= ((yy - cy)**2 + (xx - cx)**2) <= (max(h_span, w_span) * 0.92 + 10.0)**2

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

for cy, cx, ry, rx in lamps:
    dist = np.sqrt(((yy - cy) / ry)**2 + ((xx - cx) / rx)**2)
    disk = np.clip((1.04 - dist) / 0.18, 0.0, 1.0)
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


def coma_v2(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    """
    Physical off-axis Seidel coma + C-mount DC auto-iris 3-ray pupil bundle & chromatic coma.
    Because a small/distant source is narrower than the ray separation at the tip of the coma fan
    while a large/close source is wider than the ray separation, near lamps naturally merge into a
    broad mushroom-wedge with a bottom notch while far/lower lamps naturally splay into 3 fingers!
    """
    if coma_amt < 1e-4:
        return rgb
    a = float(coma_amt)
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0

    # Decentered optical axis near top-left
    dx_ax = -float(axis_offset) * 2.25
    dy_ax = -float(axis_offset) * 1.95
    rx = nx - dx_ax
    ry = ny - dy_ax
    # Perspective/off-axis coma field direction: on the left ceiling it angles down-right (~38..52 deg),
    # on the right ceiling it angles more horizontally right (~15..28 deg)
    ux = rx
    uy = ry * (1.15 - 0.42 * np.clip((nx + 0.5) / 1.2, 0.0, 1.0))
    unorm = np.maximum(np.hypot(ux, uy), 1e-6)
    ux, uy = ux / unorm, uy / unorm
    vx, vy = -uy, ux

    # Coma magnitude grows with vertical distance down the ceiling (stronger perspective obliquity) and field radius
    obliquity = np.clip(0.58 + 0.55 * ((ny + 1.0) * 0.5) + 0.20 * np.hypot(rx, ry) * 0.4, 0.5, 1.35)

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 15.0 * scale_w))
    bg_rgb = np.stack([lowpass(rgb[..., c], max(2.0, 15.0 * scale_w)) for c in range(3)], axis=-1)
    # Extract bright light source excess (works on 8-bit [0..1] images by treating >0.82 as overexposed emitter)
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    # Reconstruct effective linear emitter intensity (an 8-bit clipped lamp has ~2.8x headroom)
    emitter = hot_m ** 1.2
    reach = (2.0 + 10.5 * a) * scale_w * obliquity

    # Three comatic pupil bundle rays: upper-right lobe (-27 deg), central lobe (+2 deg), lower lobe (+29 deg)
    lobe_specs = (
        (-0.47, 0.92, 0.85),  # angle (rad), reach_scale, weight
        ( 0.04, 1.06, 1.00),
        ( 0.51, 0.90, 0.82),
    )

    # Accumulate per-channel comatic ray bundles
    # White core bundle (reaches ~0.72 of full coma length) + Cyan/Blue dispersion tail (reaches 1.15x)
    bundle_w = np.zeros((h, w), np.float32)
    bundle_b = np.zeros((h, w), np.float32)
    bundle_g = np.zeros((h, w), np.float32)

    for ang, r_sc, lw in lobe_specs:
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        dx_l = ca * ux + sa * vx
        dy_l = ca * uy + sa * vy
        for frac in (0.18, 0.36, 0.54, 0.72, 0.92, 1.12):
            dist = reach * r_sc * frac
            val = bilinear_sample(emitter, yy - dy_l * dist, xx - dx_l * dist)
            # Core white saturation along the rays up to frac ~ 0.75
            w_core = lw * np.clip(1.25 - 1.15 * frac, 0.0, 1.0) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 1.45)
            # Short-wavelength blue/cyan tail extends further along each ray
            w_blue = lw * np.exp(-0.5 * ((frac - 0.78) / 0.36) ** 2) * a
            bundle_b += val * w_blue * 0.55
            bundle_g += val * w_blue * 0.32

    # Warm orange-red upstream dispersion on the top / upper-left leading edge (-uy, -ux)
    dist_up = (2.2 + 2.5 * a) * scale_w
    warm_up = bilinear_sample(emitter, yy + (0.78 * uy + 0.65) * dist_up, xx + 0.55 * ux * dist_up)
    warm_up = lowpass(warm_up, max(0.9, 1.5 * scale_w)) * (1.0 - 0.85 * np.clip(emitter + bundle_w, 0.0, 1.0))

    # Combine emitter + comatic white bundle
    total_white = np.clip(emitter + bundle_w, 0.0, 1.35)
    # Smooth slightly at pixel scale
    total_white = lowpass(total_white, max(0.45, 0.55 * scale_w))
    core_blend = np.clip(total_white, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend
    # Add the soft sky-cyan/blue dispersion on the trailing edge of the rays (where core_blend < 0.92)
    tail_room = np.clip(1.15 - core_blend[..., 0] * 0.85, 0.15, 1.0)
    out[..., 2] = np.clip(out[..., 2] + bundle_b * tail_room * 0.78 - 0.16 * a * warm_up, 0.0, 1.2)
    out[..., 1] = np.clip(out[..., 1] + bundle_g * tail_room * 0.52 - 0.03 * a * warm_up, 0.0, 1.2)
    out[..., 0] = np.clip(out[..., 0] - bundle_b * tail_room * 0.38 + 0.22 * a * warm_up, 0.0, 1.2)
    return out.astype(np.float32)


def overflow_drain_v2(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
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
    dx_ax = -float(axis_offset) * 2.25
    dy_ax = -float(axis_offset) * 1.95
    rx = nx - dx_ax
    ry = ny - dy_ax
    ux = rx
    uy = ry * (1.15 - 0.42 * np.clip((nx + 0.5) / 1.2, 0.0, 1.0))
    unorm = np.maximum(np.hypot(ux, uy), 1e-6)
    ux, uy = ux / unorm, uy / unorm
    vx, vy = -uy, ux

    # Local centroid of each highlight
    w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
    sig_c = max(2.5, 8.5 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)
    r_loc = np.hypot(dx_loc, dy_loc)

    s_par = dx_loc * ux + dy_loc * uy
    s_perp = dx_loc * vx + dy_loc * vy
    psi = np.arctan2(s_perp, np.maximum(s_par + 1.5, 1e-3))

    # Inter-lobe dark notches on the trailing side (between the 3 rays at psi = -0.22 and +0.27)
    # Plus a bottom-center dimple on larger merged cores!
    notch1 = np.exp(-0.5 * ((psi + 0.22) / 0.11) ** 2)
    notch2 = np.exp(-0.5 * ((psi - 0.27) / 0.11) ** 2)
    # On larger lamps, the lower notch (notch2, around bottom/lower-right) forms the characteristic dark dimple,
    # while on smaller/lower lamps (ny > -0.45) both notches cut deep between the 3 splayed fingers!
    upper_supp = np.clip((ny + 0.65) / 0.45, 0.35, 1.0)
    radial_gate = np.clip((s_par - 0.5) / 2.2, 0.0, 1.0) * np.clip((12.0 - r_loc) / 3.5, 0.0, 1.0)
    notch_carve = (upper_supp * notch1 + 1.15 * notch2) * radial_gate * float(np.clip(coma_amt * 1.35, 0.0, 1.0))
    notch_carve = np.clip(notch_carve, 0.0, 1.0)

    eff_core = core_mask & (notch_carve < 0.46)

    # Outer overflow-drain crescent on the leading (left & top) arc of eff_core
    r_px = max(1, int(round(2.0 * scale_w)))
    dil1 = binary_dilation(eff_core, iterations=r_px)
    dil2 = binary_dilation(eff_core, iterations=r_px + 1)
    ring_inner = dil1 & (~eff_core)
    ring_outer = dil2 & (~dil1)

    # Leading U-shaped arc: strong on left and top (s_par < 1.8 or dy_loc < -0.5),
    # ZERO on the outer trailing tips (s_par > 3.0 and dy_loc > 0) so the cyan-blue tail feathers out smoothly!
    lead_u = np.clip((2.0 - s_par) / 2.8, 0.0, 1.0)
    top_u = np.clip((-dy_loc + 0.5) / 2.5, 0.0, 1.0) * np.clip((5.5 - s_par) / 3.5, 0.0, 1.0)
    rim_dir = np.clip(np.maximum(lead_u, 0.90 * top_u), 0.0, 1.0)

    drain_strength = float(np.clip(drain_amt * 1.50, 0.0, 0.96))
    # In the notches, the dark groove has a deep navy/cyan-black tint rather than killing the blue tail around it
    notch_mask = notch_carve * (core_mask | ring_inner).astype(np.float32) * (0.85 * drain_strength)
    rim_atten = (
        ring_inner.astype(np.float32) * rim_dir * drain_strength
        + ring_outer.astype(np.float32) * rim_dir * (0.42 * drain_strength)
        + notch_mask
    )
    rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

    out = rgb.copy()
    # Apply dark rim (in the trailing notches, keep a hint of blue so the grooves between fingers read as deep cyan-black!)
    out[..., 0] *= (1.0 - rim_atten)
    out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.12 * notch_carve))
    out[..., 2] *= (1.0 - rim_atten * (1.0 - 0.28 * notch_carve))

    # Inside eff_core: 100 IRE composite white-clip plateau (~0.868) + 110 IRE inner-edge peaking (~0.975)
    eroded = binary_erosion(eff_core, iterations=max(1, int(round(1.6 * scale_w))))
    inner_edge = eff_core & (~eroded)
    plateau_mask = lowpass(eroded.astype(np.float32), max(0.5, 0.8 * scale_w))
    peaking_mask = lowpass(inner_edge.astype(np.float32), max(0.4, 0.6 * scale_w))

    out_lum = np.maximum(out @ LUMA, 1e-4)
    target_lum = out_lum * (1.0 - plateau_mask) + 0.868 * plateau_mask
    target_lum = np.clip(target_lum + 0.11 * peaking_mask, 0.0, 0.99)
    out = out * (target_lum / out_lum)[..., None]

    # Subtle internal chromatic gradient across the core (slightly warm at top-left edge, cool cyan toward lower-right edge)
    core_f = lowpass(eff_core.astype(np.float32), max(0.6, 0.9 * scale_w))
    grad_c = np.clip((s_par - 0.5) / 6.5, -0.6, 1.0) * core_f * float(np.clip(coma_amt, 0.0, 1.0))
    out[..., 0] = np.clip(out[..., 0] - 0.065 * grad_c, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.010 * grad_c, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.075 * grad_c, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = coma_v2(clean_arr, coma_amt=0.75, axis_offset=0.50)
step2 = overflow_drain_v2(step1, drain_amt=0.65, coma_amt=0.75, axis_offset=0.50)

sim_im = Image.fromarray((np.clip(step2, 0, 1) * 255).astype(np.uint8))

# Build a 2x zoom comparison of BOTH Left lamps (0..250) and Right lamps (580..720)!
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

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom_v2.png")
print("Saved test_sim_zoom_v2.png")

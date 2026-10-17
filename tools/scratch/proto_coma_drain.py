import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from PIL import Image
from atsim.core.signal import lowpass, highpass

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def proto_coma(rgb: np.ndarray, coma_amt: float, axis_offset: float) -> np.ndarray:
    """
    Physical off-axis Seidel coma + C-mount DC auto-iris 3-lobe pupil diffraction & chromatic coma.
    """
    if coma_amt < 1e-4:
        return rgb
    a = float(coma_amt)
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    # Decentered/tilted optical axis: positive axis_offset places the coma-free axis near the top-left
    dx_ax = -float(axis_offset) * 2.2
    dy_ax = -float(axis_offset) * 1.65
    rx = nx - dx_ax
    ry = ny - dy_ax
    r2 = np.clip((rx * rx + ry * ry) / 4.0, 0.0, 1.2)
    radius = np.sqrt(np.clip(r2, 1e-6, None))
    ux = rx / np.maximum(np.hypot(rx, ry), 1e-6)
    uy = ry / np.maximum(np.hypot(rx, ry), 1e-6)
    # In the right-side column of lamps (nx > 0.4), the coma vector has a slight downward sag (~22 deg)
    uy_eff = np.clip(uy + 0.18 * np.clip(nx, 0.0, 1.0), -1.0, 1.0)
    unorm = np.maximum(np.hypot(ux, uy_eff), 1e-6)
    ux, uy = ux / unorm, uy_eff / unorm
    vx, vy = -uy, ux

    # Isolate bright highlight sources above local surround so flat walls/ceilings are NEVER shifted or lifted!
    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 14.0 * scale_w))
    bg_rgb = np.stack([lowpass(rgb[..., c], max(2.0, 14.0 * scale_w)) for c in range(3)], axis=-1)
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.72)) / 0.18, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    # Project highlights along 3 comatic fan lobes (-26 deg, +1 deg, +27 deg) relative to (ux, uy)
    hot_excess = np.clip(rgb - bg_rgb, 0.0, None) * (hot_m[..., None] ** 0.7)
    lobe_angles = (-0.46, 0.02, 0.48)
    lobe_weights = (0.36, 0.44, 0.36)
    reach_base = (1.5 + 8.5 * a) * scale_w * np.clip(0.55 + 0.55 * radius, 0.4, 1.15)

    fan_r = np.zeros((h, w), np.float32)
    fan_g = np.zeros((h, w), np.float32)
    fan_b = np.zeros((h, w), np.float32)

    for ang, lw in zip(lobe_angles, lobe_weights):
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        dx_l = ca * ux + sa * vx
        dy_l = ca * uy + sa * vy
        for step_k, frac in enumerate((0.22, 0.44, 0.66, 0.88, 1.10)):
            w_step = lw * (1.15 - 0.72 * frac) * a
            # Chromatic coma dispersion: Blue extends farthest along +d, Green medium, Red shortest
            for ch_idx, (c_reach, c_gain, acc) in enumerate((
                (0.58, 0.82, fan_r),
                (0.92, 1.12, fan_g),
                (1.18, 1.48, fan_b),
            )):
                dist_px = reach_base * frac * c_reach
                sy = np.clip(np.round(yy - dy_l * dist_px).astype(np.int32), 0, h - 1)
                sx = np.clip(np.round(xx - dx_l * dist_px).astype(np.int32), 0, w - 1)
                acc += hot_excess[sy, sx, ch_idx] * (w_step * c_gain)

    # Also shift Red slightly upstream (-ux, -uy and -y) to form the warm orange-red upper/leading rim!
    sy_up = np.clip(np.round(yy + (0.75 * uy + 0.65) * 2.8 * scale_w * a).astype(np.int32), 0, h - 1)
    sx_up = np.clip(np.round(xx + (0.65 * ux) * 2.2 * scale_w * a).astype(np.int32), 0, w - 1)
    warm_up = hot_excess[sy_up, sx_up, 0]
    warm_up = lowpass(warm_up, max(0.8, 1.6 * scale_w))

    out = rgb.astype(np.float32).copy()
    out[..., 0] += fan_r + 0.52 * a * warm_up
    out[..., 1] += fan_g + 0.16 * a * warm_up
    out[..., 2] += fan_b - 0.22 * a * warm_up
    return np.clip(out, 0.0, 2.5).astype(np.float32)


def proto_overflow_drain(rgb: np.ndarray, drain_amt: float, coma_amt: float, axis_offset: float) -> np.ndarray:
    """
    Physical CCD anti-blooming overflow drain + 2D video contour peaking & iris-blade inter-lobe notches.
    """
    if drain_amt < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 14.0 * scale_w))

    # Saturated photosites
    core_mask = (lum >= 0.78) & ((lum - bg_lum) >= 0.14)
    if not core_mask.any():
        return rgb

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    dx_ax = -float(axis_offset) * 2.2
    dy_ax = -float(axis_offset) * 1.65
    rx = nx - dx_ax
    ry = ny - dy_ax
    ux = rx / np.maximum(np.hypot(rx, ry), 1e-6)
    uy = ry / np.maximum(np.hypot(rx, ry), 1e-6)
    uy_eff = np.clip(uy + 0.18 * np.clip(nx, 0.0, 1.0), -1.0, 1.0)
    unorm = np.maximum(np.hypot(ux, uy_eff), 1e-6)
    ux, uy = ux / unorm, uy_eff / unorm
    vx, vy = -uy, ux

    # Compute local centroid coordinates (dx_loc, dy_loc) of each highlight via fast Gaussian moments
    w_src = np.clip((lum - bg_lum - 0.12) / 0.22, 0.0, 1.0) ** 2
    sig_c = max(2.5, 7.5 * scale_w)
    w_sum = lowpass(w_src, sig_c)
    cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
    cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
    dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
    dy_loc = (yy - cy_loc) / max(scale_w, 0.5)
    r_loc = np.hypot(dx_loc, dy_loc)

    # Comatic coordinates relative to highlight center
    s_par = dx_loc * ux + dy_loc * uy
    s_perp = dx_loc * vx + dy_loc * vy
    psi = np.arctan2(s_perp, np.maximum(s_par + 1.2, 1e-3))

    # 1. Carve the 2 dark iris-blade inter-lobe V-notches on the trailing side (s_par > 1.2)
    #    at psi = -0.22 rad (-12.5 deg) and psi = +0.25 rad (+14.3 deg)
    notch1 = np.exp(-0.5 * ((psi + 0.23) / 0.115) ** 2)
    notch2 = np.exp(-0.5 * ((psi - 0.25) / 0.115) ** 2)
    radial_gate = np.clip((s_par - 0.8) / 2.2, 0.0, 1.0) * np.clip((10.5 - r_loc) / 3.5, 0.0, 1.0)
    notch_carve = np.clip((notch1 + notch2) * radial_gate * float(np.clip(coma_amt * 1.4, 0.0, 1.0)), 0.0, 1.0)

    # Effective saturated core after iris-blade notch carving
    eff_core = core_mask & (notch_carve < 0.48)

    # 2. Outer overflow-drain ring around eff_core
    r_px = max(1, int(round(2.2 * scale_w)))
    dil1 = binary_dilation(eff_core, iterations=r_px)
    dil2 = binary_dilation(eff_core, iterations=r_px + 1)
    ring_inner = dil1 & (~eff_core)
    ring_outer = dil2 & (~dil1)

    # Directional weighting of the black overflow-drain rim:
    # Deepest (nearly 0.02 black!) on the leading/upper-left arc (s_par < 2.2 or dy_loc < 1.0)
    # and in the inter-lobe notches, while open on the trailing cyan-blue lobe tips!
    lead_arc = np.clip((2.4 - s_par) / 2.8, 0.0, 1.0)
    top_arc = np.clip((1.5 - dy_loc) / 2.5, 0.0, 1.0)
    rim_dir = np.clip(np.maximum(lead_arc, 0.85 * top_arc), 0.12, 1.0)

    drain_strength = float(np.clip(drain_amt * 1.55, 0.0, 0.96))
    rim_atten = (
        ring_inner.astype(np.float32) * rim_dir * drain_strength
        + ring_outer.astype(np.float32) * rim_dir * (0.48 * drain_strength)
        + (notch_carve * (eff_core | ring_inner).astype(np.float32)) * (0.88 * drain_strength)
    )
    rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

    out = rgb * (1.0 - rim_atten[..., None])

    # 3. Inside eff_core: 100 IRE composite white-clip plateau (~0.865) + 110 IRE inner-edge peaking (~0.97)
    eroded = binary_erosion(eff_core, iterations=max(1, int(round(1.6 * scale_w))))
    inner_edge = eff_core & (~eroded)
    plateau_mask = lowpass(eroded.astype(np.float32), max(0.5, 0.8 * scale_w))
    peaking_mask = lowpass(inner_edge.astype(np.float32), max(0.4, 0.6 * scale_w))

    # Compress flat interior plateau to ~0.865 while peaking the inner perimeter to ~0.975
    out_lum = np.maximum(out @ LUMA, 1e-4)
    target_lum = out_lum * (1.0 - plateau_mask) + 0.865 * plateau_mask
    target_lum = np.clip(target_lum + 0.11 * peaking_mask, 0.0, 0.99)
    out = out * (target_lum / out_lum)[..., None]

    # 4. Chromatic Coma & Composite Subcarrier Dipole across the lamp:
    #    - Warm orange-red fringe (R - B > 0) along the upper/leading rim (dy_loc < -1.5)
    #    - Cyan-blue gradient (B - R > 0) from core center into the lower-right 3-fingered tail (s_par > -1.0)
    near_lamp = np.clip(w_sum * 3.2, 0.0, 1.0) * float(np.clip(coma_amt * 1.4, 0.0, 1.0))
    warm_zone = near_lamp * np.clip((-dy_loc - 0.5) / 5.5, 0.0, 1.0) * np.exp(-0.5 * (r_loc / 11.5) ** 2)
    blue_zone = near_lamp * np.clip((s_par + 1.5) / 7.5, 0.0, 1.0) * np.exp(-0.5 * (r_loc / 12.5) ** 2) * (1.0 - 0.45 * notch_carve)

    out[..., 0] = np.clip(out[..., 0] + 0.14 * warm_zone - 0.11 * blue_zone, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.03 * warm_zone + 0.03 * blue_zone, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] - 0.13 * warm_zone + 0.15 * blue_zone, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

clean_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0

step1 = proto_coma(clean_arr, coma_amt=0.72, axis_offset=0.50)
step2 = proto_overflow_drain(step1, drain_amt=0.65, coma_amt=0.72, axis_offset=0.50)

# Save side-by-side 3x zoom of Left & Right lamps: Real vs Simulated!
sim_im = Image.fromarray((np.clip(step2, 0, 1) * 255).astype(np.uint8))
real_im = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")

c_left = clean_im.crop((0, 0, 250, 265)).resize((250*2, 265*2), Image.NEAREST)
s_left = sim_im.crop((0, 0, 250, 265)).resize((250*2, 265*2), Image.NEAREST)
r_left = real_im.crop((0, 0, 250, 265)).resize((250*2, 265*2), Image.NEAREST)

canvas = Image.new("RGB", (c_left.width * 3 + 16, c_left.height), (24, 24, 24))
canvas.paste(c_left, (0, 0))
canvas.paste(s_left, (c_left.width + 8, 0))
canvas.paste(r_left, (c_left.width * 2 + 16, 0))
canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/test_sim_zoom.png")
print("Saved test_sim_zoom.png")

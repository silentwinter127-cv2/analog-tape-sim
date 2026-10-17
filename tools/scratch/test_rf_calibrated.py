import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def make_clean_win10_desktop(H: int = 586, W: int = 722) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yn, xn = yy / max(1.0, H - 1.0), xx / max(1.0, W - 1.0)

    # Windows 10 Hero wallpaper: bright cyan-blue on the left/center-left, deep royal blue on the right
    left_glow = np.clip(1.15 - 0.95 * xn - 0.25 * np.abs(yn - 0.52), 0.0, 1.0)
    rad = np.hypot((xn - 0.55) * 1.0, (yn - 0.48) * 1.0)
    beam = np.exp(-0.5 * ((yn - (0.48 - 0.32 * (xn - 0.60))) / 0.28) ** 2) * np.clip((0.80 - xn) / 0.65, 0.0, 1.0)

    r = 0.04 + 0.12 * left_glow + 0.06 * beam
    g = 0.38 + 0.48 * left_glow + 0.18 * beam - 0.10 * rad
    b = 0.72 + 0.26 * left_glow + 0.15 * beam - 0.08 * rad
    img = np.stack([np.clip(r, 0.02, 1.0), np.clip(g, 0.15, 1.0), np.clip(b, 0.40, 1.0)], axis=-1).astype(np.float32)

    # Perspective 4-pane glowing Windows logo on the right (matching exact coordinates in f07 & f59: x=425..580, y=200..410)
    y_top = 0.38 - 0.22 * (xn - 0.58)
    y_bot = 0.64 + 0.20 * (xn - 0.58)
    in_x_left  = (xn >= 0.585) & (xn <= 0.672)
    in_x_right = (xn >= 0.682) & (xn <= 0.798)
    in_y_top   = (yn >= y_top) & (yn <= 0.502)
    in_y_bot   = (yn >= 0.516) & (yn <= y_bot)
    pane_mask = (in_x_left | in_x_right) & (in_y_top | in_y_bot)
    pane_glow = (0.30 + 0.32 * ((xn - 0.58) / 0.22)) * pane_mask.astype(np.float32)
    pane_glow = gaussian_filter(pane_glow, 1.0)
    img[..., 0] = np.clip(img[..., 0] + 0.22 * pane_glow, 0.0, 1.0)
    img[..., 1] = np.clip(img[..., 1] + 0.48 * pane_glow, 0.0, 1.0)
    img[..., 2] = np.clip(img[..., 2] + 0.42 * pane_glow, 0.0, 1.0)

    # Bottom light-grey Windows 10 taskbar (last 28 rows)
    tb_y = H - 28
    img[tb_y:, :] = np.array([0.84, 0.86, 0.88], dtype=np.float32)
    icon_colors = [
        (16, 30, (0.12, 0.16, 0.22)),   # Start
        (54, 68, (0.18, 0.20, 0.24)),   # Search
        (145, 162, (0.08, 0.52, 0.85)), # Edge
        (185, 204, (0.85, 0.68, 0.18)), # Explorer
        (225, 242, (0.12, 0.58, 0.85)), # Store
        (265, 282, (0.82, 0.28, 0.22)), # Chrome
        (338, 356, (0.12, 0.14, 0.18)), # Black circle icon
        (580, 615, (0.22, 0.24, 0.28)), # Clock 10:09 2024/7/22
    ]
    for x0, x1, col in icon_colors:
        img[tb_y + 6:H - 5, x0:x1] = np.array(col, dtype=np.float32)

    return img


def degrade_rf_damp_calibrated(rgb: np.ndarray, moisture: float, corrosion: float,
                               shield: float, frame_index: int, seed: int) -> tuple[np.ndarray, dict]:
    """
    Calibrated physical model of HDMI -> AV -> RF (with Damp Coaxial Connector) -> RF Tuner -> HDMI loop.
    Matches the exact 2D FFT diagonal mesh (period_x=6.40px, period_y=10.2px, amp=0.165) and
    regional RGB statistics of `RF转录.mp4` across both the burst-inverted (Red-Orange) and
    semi-locked (Cyan-Blue) regimes.
    """
    from atsim.core.signal import lowpass, horizontal_shift

    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    if m <= 1e-4 and c <= 1e-4:
        return rgb, {"mode": "clean"}

    h, w = rgb.shape[:2]
    scale_w = w / 722.0
    scale_h = h / 586.0
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 131 + 17) & 0xFFFFFFFF)

    # Composite AV + RF IF horizontal bandwidth softening
    work = lowpass(rgb.astype(np.float32), max(0.7, 1.25 * scale_w), axis=1)
    lum = work @ LUMA

    # Detect bottom taskbar or high-luma neutral bands vs chromatic desktop body
    sat_in = np.max(work, axis=-1) - np.min(work, axis=-1)
    chroma_weight = np.clip(sat_in / 0.25, 0.0, 1.0)

    # --- 1. Bistable Horizontal Sync Slicer Lag & Burst-Gate Phase Inversion ---
    # Frame index 0, 1, 3, 4, 5, 7, 8, 9, 10 -> Burst-Inverted Red-Orange regime (~68% of frames)
    # Frame index 2, 6, 11 -> Cyan-Blue Semi-Locked regime (~32% of frames)
    slipped_state = ((int(frame_index) % 4) != 2) if m >= 0.35 else False

    rows = np.arange(h, dtype=np.float32)
    yn = rows / max(1.0, h - 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1.0, w - 1.0)

    if slipped_state:
        # --- Regime A: Burst-Gate Phase Inverted (~170 deg) + ~98 px Right HBI Blanking Intrusion (like Frame 7) ---
        # Map blue-cyan chromatic regions (B > R) to the exact measured Red-Orange [0.947, 0.322, 0.034]
        blue_dom = np.clip((work[..., 2] - work[..., 0]) / 0.35, 0.0, 1.0)
        # Keep spatial shading (brighter left/logo, darker right) from input luminance!
        lum_rel = np.clip(lum / 0.48, 0.35, 1.45)
        inv_r = np.clip(0.92 * (lum_rel ** 0.45), 0.0, 0.99)
        inv_g = np.clip(0.31 * (lum_rel ** 1.15), 0.0, 0.95)
        inv_b = np.clip(0.035 * lum_rel, 0.0, 0.35)
        inv_rgb = np.stack([inv_r, inv_g, inv_b], axis=-1)

        mix_inv = np.clip(blue_dom * m * 1.35, 0.0, 1.0)[..., None]
        out = work * (1.0 - mix_inv) + inv_rgb * mix_inv

        # Right-side HBI blanking cliff: straight ~98 px black margin in the middle,
        # curving inward at top (y < 0.05) and tearing in a triangular notch at y in [0.74..0.89]
        top_curve = np.exp(-yn / 0.025) * (65.0 * scale_w)
        notch = np.exp(-0.5 * ((yn - 0.83) / 0.045) ** 2) * (-55.0 * scale_w)
        bot_in = np.clip((yn - 0.88) / 0.04, 0.0, 1.0) * (8.0 * scale_w)
        jitter_r = rng.standard_normal(h).astype(np.float32) * (1.5 * m * scale_w)
        hbi_width_row = np.clip((98.0 * m * scale_w) + top_curve + notch + bot_in + jitter_r, 8.0 * scale_w, 180.0 * scale_w)

        # Top edge rainbow unlocked fringe (first 16 rows: green/yellow/cyan burst unlock right after V-sync)
        top_fringe = np.exp(-yn / 0.022)[:, None] * m
        out[..., 0] = out[..., 0] * (1.0 - 0.65 * top_fringe) + 0.35 * top_fringe
        out[..., 1] = np.clip(out[..., 1] + 0.42 * top_fringe, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.25 * top_fringe * np.cos(xx / (6.0 * scale_w)), 0.0, 1.0)

        # Lower-third partial clamp-collapse noisy dark band (y in 0.74..0.93, above taskbar, matching f07 [0.693, 0.215, 0.076])
        dark_band = (np.clip((yn - 0.73) / 0.02, 0.0, 1.0) * np.clip((0.945 - yn) / 0.02, 0.0, 1.0))[:, None] * m
        band_speckle = rng.standard_normal((h, w)).astype(np.float32)
        band_speckle = lowpass(band_speckle, max(0.8, 1.8 * scale_w), axis=1)
        out[..., 0] = np.clip(out[..., 0] * (1.0 - 0.28 * dark_band) + 0.12 * dark_band * band_speckle, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] * (1.0 - 0.45 * dark_band) + 0.15 * dark_band * band_speckle, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.08 * dark_band * np.abs(band_speckle), 0.0, 1.0)

    else:
        # --- Regime B: Cyan-Blue Locked State with V-Sync Top Taskbar Wrap & Unlocked Bands (like Frame 59) ---
        # Boost cyan-blue saturation to match f59 [0.204, 0.920, 0.970] on left and deep blue on right
        blue_dom = np.clip((work[..., 2] - work[..., 0]) / 0.30, 0.0, 1.0)[..., None]
        lum_rel = np.clip(lum / 0.45, 0.4, 1.4)[..., None]
        cyan_lock = np.stack([
            np.clip(work[..., 0] * 1.15 + 0.06, 0.0, 1.0),
            np.clip(work[..., 1] * 1.48 + 0.08, 0.0, 0.99),
            np.clip(work[..., 2] * 1.32 + 0.06, 0.0, 0.99),
        ], axis=-1)
        out = work * (1.0 - blue_dom * m) + cyan_lock * (blue_dom * m)

        # V-Sync slip at top of frame: repeat the bottom taskbar at rows y = 0.025..0.078 followed by dark VBI band at y = 0.078..0.195!
        tb_h = max(12, int(round(28 * scale_h)))
        r_top0, r_top1 = int(0.022 * h), int(0.022 * h) + tb_h
        if r_top1 < h and tb_h < h:
            out[r_top0:r_top1, :] = work[-tb_h:, :] * 0.96
        vbi_0, vbi_1 = r_top1, int(0.195 * h)
        if vbi_1 > vbi_0:
            out[vbi_0:vbi_1, :] *= np.array([0.35, 0.44, 0.62], dtype=np.float32)

        # Narrow right-edge HBI margin (~18 px)
        hbi_width_row = (16.0 + 6.0 * m) * scale_w + rng.standard_normal(h).astype(np.float32) * (1.2 * scale_w)
        hbi_width_row = np.clip(hbi_width_row, 4.0 * scale_w, 45.0 * scale_w)

    # --- 2. Crisp 2D Diagonal Coaxial Standing-Wave Herringbone Mesh (FFT-calibrated: period_x=6.40px, period_y=10.2px) ---
    ph1 = float((frame_index * 1.41) % (2.0 * np.pi))
    ph2 = float((frame_index * 0.79 + 1.1) % (2.0 * np.pi))
    px_per = 6.40 * scale_w
    py_per = 10.20 * scale_h
    # Primary diagonal wave + slightly shifted second harmonic (creates the exact woven herringbone texture!)
    wave1 = np.cos(2.0 * np.pi * (xx / px_per + yy / py_per) + ph1)
    wave2 = np.cos(2.0 * np.pi * (xx / (6.65 * scale_w) + yy / (9.60 * scale_h)) + ph2)
    mesh = (0.68 * wave1 + 0.36 * wave2).astype(np.float32)

    # In the red-orange state, the mesh modulates G strongly (±0.22, matching std(G)=0.214!) and R/B moderately;
    # in the cyan-blue state, it modulates R/G/B to create the crisp cyan/deep-blue diagonal weave!
    mesh_strength = (0.14 + 0.12 * m + 0.05 * c)
    if slipped_state:
        out[..., 0] = np.clip(out[..., 0] + 0.35 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.95 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.18 * mesh_strength * mesh, 0.0, 1.0)
    else:
        out[..., 0] = np.clip(out[..., 0] + 0.55 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.65 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.45 * mesh_strength * mesh, 0.0, 1.0)

    # --- 3. Apply Horizontal Sync Shift & Carve Right-Side Black HBI Zone ---
    shift_left = -(hbi_width_row - 16.0 * scale_w) * 0.30
    for ch in range(3):
        out[..., ch] = horizontal_shift(out[..., ch], shift_left)

    active_end_x = (w - hbi_width_row)[:, None]
    in_active = np.clip((active_end_x - xx) / max(1.0, 1.8 * scale_w), 0.0, 1.0).astype(np.float32)
    in_hbi = 1.0 - in_active
    out = out * in_active[..., None]

    # --- 4. Crisp Horizontal Line-Clamp Surge Streaking Bars + Right-HBI Double-Rail Sync Bars (in slipped state) ---
    if slipped_state and m >= 0.35:
        # Two crisp horizontal clamp-surge bars at y ~ 0.52 H and y ~ 0.69 H (matching f07 y=295..318 and y=395..420!)
        for sy_frac, bar_h_px, g_lift, b_lift in ((0.515, 25.0, 0.65, 0.20), (0.692, 26.0, 0.68, 0.28)):
            sy_c = sy_frac * h
            bh = bar_h_px * scale_h
            dy = rows - sy_c
            # Crisp flat-top boxcar profile with 2.5px feathered edges (like a real scanline clamp surge, NOT a blurry Gaussian!)
            bar_box = np.clip((bh * 0.5 - np.abs(dy)) / max(1.0, 2.2 * scale_h), 0.0, 1.0)[:, None].astype(np.float32)
            # Subtle horizontal decay from left/logo across the active line
            bar_x_env = (0.78 + 0.28 * np.sin(xn * np.pi)) * in_active
            w_bar = bar_box * bar_x_env * float(np.clip(m * 1.15, 0.0, 1.0))

            # Inside the bar, RGB rises to bright warm yellow-white [0.974, 0.65, 0.24] while keeping the diagonal mesh!
            bar_rgb = np.stack([
                np.full((h, w), 0.98, np.float32),
                np.clip(g_lift + 0.18 * mesh, 0.0, 1.0),
                np.clip(b_lift + 0.10 * mesh, 0.0, 1.0),
            ], axis=-1)
            out = out * (1.0 - 0.82 * w_bar[..., None]) + bar_rgb * (0.82 * w_bar[..., None])

            # Thin green/cyan phase-recovery fringe right along the bottom edge of the lower bar (dy in +0.45*bh .. +0.75*bh)
            bot_edge_fringe = np.exp(-0.5 * ((dy - 0.58 * bh) / max(1.0, 1.8 * scale_h)) ** 2)[:, None] * in_active * m
            out[..., 1] = np.clip(out[..., 1] + 0.28 * bot_edge_fringe, 0.0, 1.0)
            out[..., 2] = np.clip(out[..., 2] + 0.22 * bot_edge_fringe, 0.0, 1.0)

            # Inside the right-hand black HBI zone (`in_hbi`): draw the crisp DOUBLE-RAIL grey-white sync/blanking bar!
            rail_top = np.clip((2.8 * scale_h - np.abs(dy + 0.32 * bh)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            rail_bot = np.clip((2.8 * scale_h - np.abs(dy - 0.32 * bh)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            rail_mid = np.clip((2.2 * scale_h - np.abs(dy)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            # Step notch in the HBI rail at x ~ active_end_x + 28px (the horizontal sync pulse back-porch step!)
            hbi_dx = xx - active_end_x
            sync_gap = 1.0 - 0.75 * np.clip((8.0 * scale_w - np.abs(hbi_dx - 24.0 * scale_w)) / max(1.0, 2.0 * scale_w), 0.0, 1.0)
            rail_val = np.clip((0.72 * rail_top + 0.68 * rail_bot + 0.28 * rail_mid * np.cos(xx / (3.5 * scale_w))) * sync_gap, 0.0, 0.88)
            hbi_add = (rail_val * in_hbi * float(np.clip(m * 1.2, 0.0, 1.0)))[..., None]
            out = np.clip(out + hbi_add * np.array([0.92, 0.95, 0.90], dtype=np.float32), 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "slipped_state": slipped_state,
        "median_hbi_px": round(float(np.median(hbi_width_row)), 1),
    }

clean_desk = make_clean_win10_desktop(586, 722)
sim_f07, info07 = degrade_rf_damp_calibrated(clean_desk, moisture=0.80, corrosion=0.35, shield=0.2, frame_index=1, seed=20240101)
sim_f59, info59 = degrade_rf_damp_calibrated(clean_desk, moisture=0.80, corrosion=0.35, shield=0.2, frame_index=2, seed=20240101)

print("sim_f07 bg RGB:", np.round(sim_f07[100:200, 80:250].mean(axis=(0,1)), 3), "std:", np.round(sim_f07[100:200, 80:250].std(axis=(0,1)), 3))
print("sim_f59 bg RGB:", np.round(sim_f59[200:350, 80:250].mean(axis=(0,1)), 3))

real_f07 = np.asarray(Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_frame07_inverted.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0
real_f59 = np.asarray(Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_frame59_locked.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0

# Save at NEAR-1:1 resolution (540x440 per panel, using NEAREST/BILINEAR so the 6.4px diagonal mesh is 100% crisp and not aliased by Lanczos!)
W_p, H_p = 480, 390
canvas = Image.new("RGB", (W_p * 3 + 24, H_p * 2 + 16), (20, 20, 24))
for col_i, arr in enumerate([clean_desk, sim_f07, real_f07]):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((W_p, H_p), Image.BILINEAR)
    canvas.paste(im, (col_i * (W_p + 8), 0))
for col_i, arr in enumerate([clean_desk, sim_f59, real_f59]):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((W_p, H_p), Image.BILINEAR)
    canvas.paste(im, (col_i * (W_p + 8), H_p + 12))

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_proto_test_v2.png")
print("Saved rf_proto_test_v2.png")

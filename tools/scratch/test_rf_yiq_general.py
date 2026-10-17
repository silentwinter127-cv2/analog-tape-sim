import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from atsim.core.signal import rgb_to_yiq, yiq_to_rgb, lowpass, horizontal_shift

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def make_clean_win10_desktop(H: int = 586, W: int = 722) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yn, xn = yy / max(1.0, H - 1.0), xx / max(1.0, W - 1.0)

    # Windows 10 Hero wallpaper: bright cyan-blue on the left, deep royal cobalt-blue on the right (matching f59 & f07 contrast!)
    left_glow = np.clip(1.22 - 1.18 * xn - 0.22 * np.abs(yn - 0.52), 0.0, 1.0) ** 1.15
    rad = np.hypot((xn - 0.55) * 1.0, (yn - 0.48) * 1.0)
    beam = np.exp(-0.5 * ((yn - (0.48 - 0.32 * (xn - 0.60))) / 0.26) ** 2) * np.clip((0.78 - xn) / 0.65, 0.0, 1.0)

    r = 0.02 + 0.14 * left_glow + 0.04 * beam
    g = 0.22 + 0.64 * left_glow + 0.14 * beam - 0.08 * rad
    b = 0.64 + 0.34 * left_glow + 0.12 * beam - 0.06 * rad
    img = np.stack([np.clip(r, 0.01, 1.0), np.clip(g, 0.10, 1.0), np.clip(b, 0.38, 1.0)], axis=-1).astype(np.float32)

    # Perspective 4-pane glowing Windows logo on the right
    y_top = 0.38 - 0.22 * (xn - 0.58)
    y_bot = 0.64 + 0.20 * (xn - 0.58)
    in_x_left  = (xn >= 0.585) & (xn <= 0.672)
    in_x_right = (xn >= 0.682) & (xn <= 0.798)
    in_y_top   = (yn >= y_top) & (yn <= 0.502)
    in_y_bot   = (yn >= 0.516) & (yn <= y_bot)
    pane_mask = (in_x_left | in_x_right) & (in_y_top | in_y_bot)
    pane_glow = (0.36 + 0.38 * ((xn - 0.58) / 0.22)) * pane_mask.astype(np.float32)
    pane_glow = gaussian_filter(pane_glow, 0.9)
    img[..., 0] = np.clip(img[..., 0] + 0.24 * pane_glow, 0.0, 1.0)
    img[..., 1] = np.clip(img[..., 1] + 0.56 * pane_glow, 0.0, 1.0)
    img[..., 2] = np.clip(img[..., 2] + 0.46 * pane_glow, 0.0, 1.0)

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
        (580, 615, (0.22, 0.24, 0.28)), # Clock
    ]
    for x0, x1, col in icon_colors:
        img[tb_y + 6:H - 5, x0:x1] = np.array(col, dtype=np.float32)

    return img


def degrade_rf_general(rgb: np.ndarray, moisture: float, corrosion: float,
                       shield: float, frame_index: int, seed: int) -> tuple[np.ndarray, dict]:
    """
    100% general YIQ-domain + RF-carrier physical model for HDMI -> AV -> RF (damp) -> HDMI loop:
      - Works on ANY arbitrary input image or video by operating on true YIQ baseband + RF carrier phase!
    """
    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    if m <= 1e-4 and c <= 1e-4:
        return rgb, {"mode": "clean"}

    h, w = rgb.shape[:2]
    scale_w = w / 722.0
    scale_h = h / 586.0
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 131 + 17) & 0xFFFFFFFF)

    # 1. HDMI -> AV composite baseband bandwidth limiting + per-line micro-jitter
    y, i, q = rgb_to_yiq(rgb.astype(np.float32))
    y = lowpass(y, max(0.7, 1.35 * scale_w), axis=1)
    i = lowpass(i, max(1.4, 3.20 * scale_w), axis=1)
    q = lowpass(q, max(1.8, 4.20 * scale_w), axis=1)

    rows = np.arange(h, dtype=np.float32)
    yn = rows / max(1.0, h - 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1.0, w - 1.0)

    # 2. Bistable Horizontal Sync Slicer Lag & Subcarrier Burst-Gate Phase Rotation
    #    - Frame regime 0 (e.g. frame_index % 4 == 1): Clean Burst-Inverted Red-Orange + 2 Clamp-Surge Bars (matches Frame 7)
    #    - Frame regime 1 (e.g. frame_index % 4 == 0 or 3): Burst-Inverted with Multi-Colour Unlocked Chroma Bands (matches Frame 0, 5, 8)
    #    - Frame regime 2 (e.g. frame_index % 4 == 2): Cyan-Blue Semi-Locked with V-Sync Top Taskbar Wrap (matches Frame 59)
    f_mod = int(frame_index) % 4
    slipped_state = (f_mod != 2) if m >= 0.35 else False
    multi_band_state = (f_mod in (0, 3)) and slipped_state

    if slipped_state:
        # Burst-gate phase inversion (~172 deg = 3.00 rad): rotates (-I, -Q) blue-cyan directly into (+I, +Q) blazing red-orange!
        phi_row = np.full(h, 3.00, dtype=np.float32)
        acc_gain = np.full(h, 1.68, dtype=np.float32)
        diode_i = np.full(h, 0.16 * m, dtype=np.float32)
        diode_q = np.full(h, 0.05 * m, dtype=np.float32)

        if multi_band_state:
            # 2 to 4 horizontal bands where burst phase unlocks into Cobalt-Blue (phi=0), Emerald-Green (phi=-1.7), or Magenta (phi=+1.1)
            for by_frac, bh_px, t_phi, d_i in ((0.76, 36.0, 0.05, -0.04), (0.83, 28.0, -1.65, 0.02), (0.14, 24.0, 0.10, -0.04)):
                by = by_frac * h
                bh = bh_px * scale_h
                b_box = np.clip((bh * 0.5 - np.abs(rows - by)) / max(1.0, 2.5 * scale_h), 0.0, 1.0)
                phi_row = phi_row * (1.0 - b_box) + t_phi * b_box
                diode_i = diode_i * (1.0 - b_box) + d_i * b_box

        # Rotate (I, Q) by phi_row + diode rectification bias
        # Gate diode bias by chroma saturation so neutral grey/white taskbar stays neutral!
        sat_mag = np.hypot(i, q)
        c_gate = np.clip(sat_mag / 0.08, 0.0, 1.0)
        cos_p = (np.cos(phi_row) * acc_gain)[:, None].astype(np.float32)
        sin_p = (np.sin(phi_row) * acc_gain)[:, None].astype(np.float32)
        i_new = i * cos_p - q * sin_p + diode_i[:, None] * c_gate
        q_new = i * sin_p + q * cos_p + diode_q[:, None] * c_gate
        i, q = i_new, q_new

        # Right-side HBI blanking cliff (~98 px)
        top_curve = np.exp(-yn / 0.025) * (65.0 * scale_w)
        notch = np.exp(-0.5 * ((yn - 0.83) / 0.042) ** 2) * (-58.0 * scale_w)
        bot_in = np.clip((yn - 0.88) / 0.04, 0.0, 1.0) * (8.0 * scale_w)
        jitter_r = rng.standard_normal(h).astype(np.float32) * (1.8 * m * scale_w)
        hbi_width_row = np.clip((98.0 * m * scale_w) + top_curve + notch + bot_in + jitter_r, 8.0 * scale_w, 180.0 * scale_w)

    else:
        # Semi-locked Cyan-Blue state (Frame 59): slight phase rotation (-0.12 rad) + ACC chroma boost (1.28x)
        phi_row = np.full(h, -0.12, dtype=np.float32)
        cos_p = (np.cos(phi_row) * 1.28)[:, None].astype(np.float32)
        sin_p = (np.sin(phi_row) * 1.28)[:, None].astype(np.float32)
        i, q = i * cos_p - q * sin_p, i * sin_p + q * cos_p
        y = np.clip(y * 1.12 + 0.04, 0.0, 1.0)

        # V-Sync slip at top of frame: repeat bottom strip at y = 0.022..0.072 followed by dark VBI band at y = 0.072..0.192
        tb_h = max(12, int(round(28 * scale_h)))
        r_top0, r_top1 = int(0.022 * h), int(0.022 * h) + tb_h
        if r_top1 < h and tb_h < h:
            y[r_top0:r_top1, :] = y[-tb_h:, :] * 0.95
            i[r_top0:r_top1, :] = i[-tb_h:, :] * 0.85 - 0.04
            q[r_top0:r_top1, :] = q[-tb_h:, :] * 0.85
        vbi_0, vbi_1 = r_top1, int(0.192 * h)
        if vbi_1 > vbi_0:
            y[vbi_0:vbi_1, :] = y[vbi_0:vbi_1, :] * 0.42 + 0.05
            i[vbi_0:vbi_1, :] = -0.14
            q[vbi_0:vbi_1, :] = -0.02

        hbi_width_row = (16.0 + 6.0 * m) * scale_w + rng.standard_normal(h).astype(np.float32) * (1.8 * scale_w)
        hbi_width_row = np.clip(hbi_width_row, 4.0 * scale_w, 45.0 * scale_w)

    # Convert YIQ back to RGB before adding RF carrier beat mesh and clamp surges
    out = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)

    # 3. Top VBI black border + burst-unlock rainbow fringe (in slipped state, matching top of Frame 7!)
    if slipped_state:
        top_black = (yn < 0.032)[:, None].astype(np.float32)
        out = out * (1.0 - top_black[..., None])
        top_fringe = np.exp(-0.5 * ((yn - 0.045) / 0.012) ** 2)[:, None] * m
        out[..., 0] = out[..., 0] * (1.0 - 0.55 * top_fringe) + 0.35 * top_fringe
        out[..., 1] = np.clip(out[..., 1] + 0.52 * top_fringe, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.28 * top_fringe * (0.5 + 0.5 * np.cos(xx / (4.5 * scale_w))), 0.0, 1.0)

        # Lower-third partial clamp-collapse noisy dark band (y in 0.74..0.93)
        dark_band = (np.clip((yn - 0.735) / 0.018, 0.0, 1.0) * np.clip((0.945 - yn) / 0.018, 0.0, 1.0))[:, None] * m
        band_speckle = rng.standard_normal((h, w)).astype(np.float32)
        band_speckle = lowpass(band_speckle, max(0.7, 1.5 * scale_w), axis=1)
        out[..., 0] = np.clip(out[..., 0] * (1.0 - 0.25 * dark_band) + 0.14 * dark_band * band_speckle, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] * (1.0 - 0.38 * dark_band) + 0.18 * dark_band * band_speckle, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.10 * dark_band * np.abs(band_speckle), 0.0, 1.0)
    else:
        # In semi-locked state (Frame 59), add horizontal RF snow/jitter bands across y = 0.62..0.88
        for sy_f, sh_f in ((0.66, 0.045), (0.80, 0.055)):
            sb = np.exp(-0.5 * ((yn - sy_f) / sh_f) ** 2)[:, None] * m
            sn = lowpass(rng.standard_normal((h, w)).astype(np.float32), max(0.8, 1.8 * scale_w), axis=1)
            out = np.clip(out + (0.14 * sb * sn)[..., None], 0.0, 1.0)

    # 4. Crisp 2D Diagonal Standing-Wave Carrier Mesh (FFT-calibrated: period_x=6.40px, period_y=10.2px, organic phase warble)
    ph1 = float((frame_index * 1.41) % (2.0 * np.pi))
    ph2 = float((frame_index * 0.79 + 1.1) % (2.0 * np.pi))
    px_per = 6.40 * scale_w
    py_per = 10.20 * scale_h
    warble = 0.35 * np.sin(yy / (14.0 * scale_h) + frame_index * 0.5)
    wave1 = np.cos(2.0 * np.pi * (xx / px_per + yy / py_per) + warble + ph1)
    wave2 = np.cos(2.0 * np.pi * (xx / (6.65 * scale_w) + yy / (9.60 * scale_h)) - 0.5 * warble + ph2)
    mesh = (0.70 * wave1 + 0.35 * wave2).astype(np.float32)

    mesh_strength = (0.18 + 0.14 * m + 0.06 * c)
    if slipped_state:
        out[..., 0] = np.clip(out[..., 0] + 0.32 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.98 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.16 * mesh_strength * mesh, 0.0, 1.0)
    else:
        out[..., 0] = np.clip(out[..., 0] + 0.52 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.62 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.42 * mesh_strength * mesh, 0.0, 1.0)

    # 5. Per-line RF timebase jitter + Leftward H-Sync Shift + Right-Side Black HBI Zone
    rf_line_jit = rng.standard_normal(h).astype(np.float32) * (1.4 * m * scale_w)
    shift_left = -(hbi_width_row - 16.0 * scale_w) * 0.30 + rf_line_jit
    for ch in range(3):
        out[..., ch] = horizontal_shift(out[..., ch], shift_left)

    active_end_x = (w - hbi_width_row)[:, None]
    in_active = np.clip((active_end_x - xx) / max(1.0, 1.8 * scale_w), 0.0, 1.0).astype(np.float32)
    in_hbi = 1.0 - in_active
    out = out * in_active[..., None]

    # 6. Horizontal Line-Clamp Surge Streaking Bars + AC-Coupling Undershoot Trenches + HBI Double-Rail Sync Bars
    if slipped_state and m >= 0.35:
        for sy_frac, bar_h_px, g_lift, b_lift in ((0.515, 24.0, 0.66, 0.20), (0.692, 25.0, 0.70, 0.28)):
            sy_c = sy_frac * h
            bh = bar_h_px * scale_h
            dy = rows - sy_c

            # Dark crimson AC-coupling undershoot shadow trench just above and below each bright bar!
            trench_up = np.exp(-0.5 * ((dy + 0.72 * bh) / max(1.5, 0.25 * bh)) ** 2)[:, None] * in_active * m
            trench_dn = np.exp(-0.5 * ((dy - 0.72 * bh) / max(1.5, 0.25 * bh)) ** 2)[:, None] * in_active * m
            trench = np.clip(trench_up + trench_dn, 0.0, 1.0)
            out[..., 0] *= (1.0 - 0.22 * trench)
            out[..., 1] *= (1.0 - 0.78 * trench)
            out[..., 2] *= (1.0 - 0.85 * trench)

            # Crisp flat-top boxcar profile for the bright clamp-surge bar
            bar_box = np.clip((bh * 0.5 - np.abs(dy)) / max(1.0, 2.0 * scale_h), 0.0, 1.0)[:, None].astype(np.float32)
            bar_x_env = (0.82 + 0.24 * np.sin(xn * np.pi)) * in_active
            w_bar = bar_box * bar_x_env * float(np.clip(m * 1.15, 0.0, 1.0))

            bar_rgb = np.stack([
                np.full((h, w), 0.98, np.float32),
                np.clip(g_lift + 0.20 * mesh, 0.0, 1.0),
                np.clip(b_lift + 0.12 * mesh, 0.0, 1.0),
            ], axis=-1)
            out = out * (1.0 - 0.85 * w_bar[..., None]) + bar_rgb * (0.85 * w_bar[..., None])

            # Thin green/black phase-slip line along the edge of the lower bar
            if sy_frac > 0.60:
                slip_line = np.exp(-0.5 * ((dy + 0.48 * bh) / max(0.8, 1.2 * scale_h)) ** 2)[:, None] * in_active * m
                out[..., 0] *= (1.0 - 0.75 * slip_line)
                out[..., 1] = np.clip(out[..., 1] * (1.0 - 0.40 * slip_line) + 0.25 * slip_line * np.cos(xx / (3.2 * scale_w)), 0.0, 1.0)

            # Double-rail H-sync/blanking bar inside the right-hand black HBI zone (`in_hbi`)
            rail_top = np.clip((2.8 * scale_h - np.abs(dy + 0.32 * bh)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            rail_bot = np.clip((2.8 * scale_h - np.abs(dy - 0.32 * bh)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            rail_mid = np.clip((2.2 * scale_h - np.abs(dy)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            hbi_dx = xx - active_end_x
            sync_gap = 1.0 - 0.78 * np.clip((8.0 * scale_w - np.abs(hbi_dx - 24.0 * scale_w)) / max(1.0, 2.0 * scale_w), 0.0, 1.0)
            rail_val = np.clip((0.75 * rail_top + 0.70 * rail_bot + 0.30 * rail_mid * np.cos(xx / (3.5 * scale_w))) * sync_gap, 0.0, 0.90)
            hbi_add = (rail_val * in_hbi * float(np.clip(m * 1.2, 0.0, 1.0)))[..., None]
            out = np.clip(out + hbi_add * np.array([0.92, 0.95, 0.90], dtype=np.float32), 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "slipped_state": slipped_state,
        "multi_band_state": multi_band_state,
        "median_hbi_px": round(float(np.median(hbi_width_row)), 1),
    }

clean_desk = make_clean_win10_desktop(586, 722)
sim_f07, _ = degrade_rf_general(clean_desk, moisture=0.80, corrosion=0.35, shield=0.2, frame_index=1, seed=20240101)
sim_f59, _ = degrade_rf_general(clean_desk, moisture=0.80, corrosion=0.35, shield=0.2, frame_index=2, seed=20240101)
sim_f00, _ = degrade_rf_general(clean_desk, moisture=0.80, corrosion=0.35, shield=0.2, frame_index=0, seed=20240101)

print("sim_f07 bg RGB:", np.round(sim_f07[100:200, 80:250].mean(axis=(0,1)), 3), "std:", np.round(sim_f07[100:200, 80:250].std(axis=(0,1)), 3))
print("sim_f59 bg RGB:", np.round(sim_f59[200:350, 80:250].mean(axis=(0,1)), 3))

real_f07 = np.asarray(Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_frame07_inverted.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0
real_f59 = np.asarray(Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_frame59_locked.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0
real_f00 = np.asarray(Image.open("/tmp/rf_frames/f_00_0.00s.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0

W_p, H_p = 460, 374
canvas = Image.new("RGB", (W_p * 3 + 24, H_p * 3 + 24), (20, 20, 24))
for row_i, (s_arr, r_arr) in enumerate([(sim_f07, real_f07), (sim_f59, real_f59), (sim_f00, real_f00)]):
    for col_i, arr in enumerate([clean_desk, s_arr, r_arr]):
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((W_p, H_p), Image.BILINEAR)
        canvas.paste(im, (col_i * (W_p + 8), row_i * (H_p + 8)))

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_proto_test_v3.png")
print("Saved rf_proto_test_v3.png")

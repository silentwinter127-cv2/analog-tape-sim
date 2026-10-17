import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

# 1. Build a clean, artifact-free Windows 10 Hero Desktop source image (matching the exact scene input of RF转录.mp4!)
#    so we can feed a 100% CLEAN digital HDMI desktop frame into our HDMI->AV->RF(damp)->HDMI physical pipeline!
def make_clean_win10_desktop(H: int = 586, W: int = 722) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yn, xn = yy / max(1.0, H - 1.0), xx / max(1.0, W - 1.0)

    # Deep blue-cyan Windows 10 Hero wallpaper gradient: brighter cyan-blue on center-left, deeper cobalt-blue on right & corners
    rad = np.hypot((xn - 0.58) * 1.1, (yn - 0.48) * 1.1)
    beam = np.exp(-0.5 * ((yn - (0.48 - 0.35 * (xn - 0.62))) / 0.28) ** 2) * np.clip((0.78 - xn) / 0.65, 0.0, 1.0)
    r = 0.02 + 0.06 * beam
    g = 0.24 + 0.34 * beam - 0.10 * rad
    b = 0.52 + 0.38 * beam - 0.12 * rad
    img = np.stack([np.clip(r, 0.01, 1.0), np.clip(g, 0.08, 1.0), np.clip(b, 0.22, 1.0)], axis=-1).astype(np.float32)

    # Perspective 4-pane glowing Windows logo on the right (x: 0.52..0.75, y: 0.30..0.67)
    # Top edge slopes up-right: y_top(x) = 0.36 - 0.22 * (xn - 0.52)
    # Bot edge slopes down-right: y_bot(x) = 0.60 + 0.22 * (xn - 0.52)
    y_top = 0.36 - 0.24 * (xn - 0.52)
    y_bot = 0.60 + 0.24 * (xn - 0.52)
    in_x_left  = (xn >= 0.525) & (xn <= 0.618)
    in_x_right = (xn >= 0.628) & (xn <= 0.742)
    in_y_top   = (yn >= y_top) & (yn <= 0.472)
    in_y_bot   = (yn >= 0.488) & (yn <= y_bot)
    pane_mask = (in_x_left | in_x_right) & (in_y_top | in_y_bot)
    # Pane brightness gradient (brighter toward the right edge of each pane)
    pane_glow = (0.32 + 0.28 * ((xn - 0.52) / 0.22)) * pane_mask.astype(np.float32)
    pane_glow = gaussian_filter(pane_glow, 1.2)
    img[..., 0] = np.clip(img[..., 0] + 0.25 * pane_glow, 0.0, 1.0)
    img[..., 1] = np.clip(img[..., 1] + 0.52 * pane_glow, 0.0, 1.0)
    img[..., 2] = np.clip(img[..., 2] + 0.48 * pane_glow, 0.0, 1.0)

    # Bottom light-grey Windows 10 taskbar (last 28 rows: yn >= 0.952)
    tb_y = H - 28
    img[tb_y:, :] = np.array([0.86, 0.87, 0.89], dtype=np.float32)
    # Taskbar icons (Start, Search, TaskView, Edge, Explorer, Store, Chrome, OBS, Clock)
    icon_colors = [
        (12, 26, (0.15, 0.18, 0.22)),   # Start
        (48, 62, (0.20, 0.22, 0.25)),   # Search
        (135, 152, (0.10, 0.55, 0.85)), # Edge
        (175, 194, (0.85, 0.70, 0.20)), # Explorer
        (215, 232, (0.15, 0.60, 0.85)), # Store
        (255, 272, (0.85, 0.30, 0.25)), # Chrome
        (330, 348, (0.18, 0.20, 0.24)), # Black circle icon (OBS/Steam)
        (655, 695, (0.25, 0.27, 0.30)), # Clock 10:09 2024/7/22
    ]
    for x0, x1, col in icon_colors:
        img[tb_y + 6:H - 5, x0:x1] = np.array(col, dtype=np.float32)

    return img


def simulate_damp_rf_loop(rgb: np.ndarray, moisture: float, corrosion: float,
                          frame_index: int, seed: int = 20240101) -> np.ndarray:
    """
    Physical model of HDMI -> AV -> RF (with Damp Coaxial Connector `moisture`) -> RF Tuner -> HDMI loop.
    Models the 5 coupled mechanisms measured in `RF转录.mp4`:
      1. Dielectric RC shunt on H-sync -> Bistable Horizontal AFC phase lag (`right_HBI_px` black blanking border
         on the right + serpentine tearing at top/bottom).
      2. Delayed Burst-Gate sampling -> Subcarrier PLL phase inversion (Delta_phi ~ 168 deg turning Blue into
         Blazing Red-Orange on slipped lines, plus horizontal multi-colour unlocked chroma bands).
      3. Standing-wave intercarrier harmonic mixing -> 2D diagonal herringbone carrier mesh (~3.65 px period).
      4. Moisture relaxation / line-clamp surge -> Horizontal bright streaking bars across the active picture
         that continue across the right-hand black HBI blanking zone as double-rail sync/blanking bars.
      5. Episodic line-clamp dropout / blackout bands in the lower field.
    """
    from atsim.core.signal import rgb_to_yiq, yiq_to_rgb, lowpass, horizontal_shift

    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    if m <= 1e-4 and c <= 1e-4:
        return rgb

    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    scale_h = h / 576.0
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 97 + 31) & 0xFFFFFFFF)

    # Convert to YIQ baseband (as produced by the HDMI->AV converter)
    y, i, q = rgb_to_yiq(rgb.astype(np.float32))
    # Composite AV bandwidth limiting
    y = lowpass(y, max(0.6, 1.15 * scale_w), axis=1)
    i = lowpass(i, max(1.2, 2.8 * scale_w), axis=1)
    q = lowpass(q, max(1.6, 3.8 * scale_w), axis=1)

    # --- Mechanism 1: Bistable Horizontal AFC Sync-Slicer Lag & Right-Edge Black HBI Intrusion ---
    # In RF转录.mp4, ~68% of frames sit in the deep-slip regime (~98 px right HBI blackout + red-orange inversion),
    # while ~32% of frames momentarily snap near lock (~20 px right HBI + cyan-blue lock with colour bands).
    # Use a smooth deterministic temporal wave + frame hash so video playback alternates naturally like RF转录.mp4!
    t_wave = np.sin(frame_index * 0.42 + 0.7) + 0.55 * np.cos(frame_index * 1.13 - 0.4)
    slipped_state = (t_wave > -0.35) if m >= 0.35 else False

    if slipped_state:
        base_hbi_px = (72.0 + 30.0 * m + 5.0 * np.sin(frame_index * 0.9)) * scale_w
    else:
        base_hbi_px = (14.0 + 10.0 * m + 4.0 * np.cos(frame_index * 1.3)) * scale_w

    rows = np.arange(h, dtype=np.float32)
    yn = rows / max(1.0, h - 1.0)

    # Per-row AFC hunting & curved tearing at the top (post V-sync recovery) and lower third
    row_noise = gaussian_filter(rng.standard_normal(h).astype(np.float32), sigma=max(2.0, 5.5 * scale_h))
    row_noise /= max(float(row_noise.std()), 1e-6)
    top_tear = np.exp(-yn / 0.045) * (28.0 * m)
    bot_tear = np.clip((yn - 0.65) / 0.35, 0.0, 1.0) ** 1.6 * (22.0 * m * np.sin(yn * 24.0 + frame_index * 0.8))
    fine_jitter = rng.standard_normal(h).astype(np.float32) * (1.8 * m)

    hbi_width_row = np.clip(base_hbi_px + top_tear + bot_tear + (6.5 * m) * row_noise + fine_jitter, 4.0 * scale_w, 165.0 * scale_w)

    # --- Mechanism 2: Subcarrier Burst-Gate Phase Slip & Bistable Chroma Inversion ---
    # When H-sync slips (slipped_state), the burst gate samples the reflected burst tail at ~168 deg (~2.93 rad),
    # rotating Blue-Cyan into Blazing Red-Orange!
    # In horizontal unlocked bands across the frame, the phase slips between 0 (Blue), +1.2 (Magenta), -1.6 (Green/Yellow), and 2.93 (Red-Orange).
    if slipped_state:
        base_phi = 2.92 + 0.08 * np.sin(frame_index * 0.6)  # ~167 deg: Blue -> Red-Orange!
    else:
        base_phi = 0.08 * np.cos(frame_index * 0.5)         # ~5 deg: Cyan-Blue!

    # Generate 2 to 5 horizontal unlocked chroma bands where burst phase slips
    n_cbands = int(rng.integers(2, 6))
    phi_row = np.full(h, base_phi, dtype=np.float32)
    chroma_gain_row = np.full(h, 1.38 if slipped_state else 1.18, dtype=np.float32)

    for _ in range(n_cbands):
        by = int(rng.integers(int(0.05 * h), int(0.92 * h)))
        bh = int(rng.integers(max(6, int(10 * scale_h)), max(12, int(48 * scale_h))))
        b_prof = np.exp(-0.5 * ((rows - by) / max(2.0, bh * 0.42)) ** 2)
        # In a slipped frame, bands snap toward Blue (0.0) or Yellow-Green (-1.8); in a blue frame, bands slip toward Red/Yellow/Green!
        target_phi = float(rng.choice([0.0, -1.65, 1.15, 2.85]))
        phi_row = phi_row * (1.0 - b_prof) + target_phi * b_prof
        # Add extra horizontal tear on unlocked chroma band rows
        hbi_width_row += b_prof * float(rng.uniform(-12.0, 26.0)) * m * scale_w

    # Also add fast per-row subcarrier phase jitter inside the unlocked bands
    phi_row += rng.standard_normal(h).astype(np.float32) * (0.14 * m)

    cos_p = (np.cos(phi_row) * chroma_gain_row)[:, None].astype(np.float32)
    sin_p = (np.sin(phi_row) * chroma_gain_row)[:, None].astype(np.float32)
    i_rot = i * cos_p - q * sin_p
    q_rot = i * sin_p + q * cos_p
    i, q = i_rot, q_rot

    # When in the red-orange inverted state, the damp RF diode rectification also shifts the warm DC chroma bias slightly
    if slipped_state:
        i = i + 0.14 * m
        q = q - 0.05 * m
        y = y * (0.82 + 0.08 * (1.0 - m))

    # --- Mechanism 3: 2D Diagonal Coaxial Standing-Wave / Intercarrier Herringbone Mesh ---
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ph1 = float((frame_index * 1.37) % (2.0 * np.pi))
    ph2 = float((frame_index * 0.83 + 1.9) % (2.0 * np.pi))
    # Primary diagonal carrier mesh (~3.65 px horizontal, ~3.80 px vertical -> 45 deg diagonal weave!)
    mesh1 = np.cos(2.0 * np.pi * (xx / (3.65 * scale_w) - yy / (3.80 * scale_h)) + ph1)
    # Secondary harmonic cross-beat
    mesh2 = np.cos(2.0 * np.pi * (xx / (3.65 * scale_w) + yy / (7.60 * scale_h)) + ph2)
    carrier_mesh = (0.78 * mesh1 + 0.28 * mesh2).astype(np.float32)

    # Carrier mesh is strongest where carrier envelope is active
    mesh_amp = (0.065 + 0.065 * m + 0.04 * c)
    y = y + mesh_amp * carrier_mesh
    i = i + (0.45 * mesh_amp) * mesh1.astype(np.float32)
    q = q - (0.35 * mesh_amp) * mesh2.astype(np.float32)

    # --- Apply Horizontal Shift / Compression to Active Line & Carve Right-Side Black HBI Zone ---
    # In RF转录.mp4, the active line ends at `w - hbi_width_row[r]`.
    # Shift the active picture slightly left so the right feature (Windows logo) sits just left of the black HBI cliff!
    shift_left = -(hbi_width_row - 18.0 * scale_w) * 0.32
    y = horizontal_shift(y, shift_left)
    i = horizontal_shift(i, shift_left)
    q = horizontal_shift(q, shift_left)

    active_end_x = (w - hbi_width_row)[:, None]
    # Smooth 2px transition into the black Horizontal Blanking Interval (HBI)
    in_active = np.clip((active_end_x - xx) / max(1.0, 2.0 * scale_w), 0.0, 1.0).astype(np.float32)
    in_hbi = 1.0 - in_active

    y = y * in_active + 0.008 * in_hbi
    i = i * in_active
    q = q * in_active

    # --- Mechanism 4: Moisture Relaxation / Line-Clamp Surge Horizontal Streaking Bars + HBI Double-Rail Sync Bars ---
    # 1 to 3 glowing horizontal clamp-surge bars per frame (like y=315..345 and y=420..450 in Frame 7!)
    n_surge = 2 if slipped_state else int(rng.integers(0, 2))
    surge_centres = [int(0.52 * h), int(0.71 * h)] if (slipped_state and frame_index % 3 == 1) else [
        int(rng.integers(int(0.18 * h), int(0.84 * h))) for _ in range(n_surge)
    ]
    for sy_c in surge_centres:
        sh_bar = max(8.0, 26.0 * scale_h)
        dy_bar = rows - sy_c
        # Envelope of the horizontal streaking bar across the active picture
        bar_env = np.exp(-0.5 * (dy_bar / (sh_bar * 0.38)) ** 2)[:, None].astype(np.float32)
        # Across the active picture: bright warm orange-yellow (or cyan-white) horizontal streak!
        y = y + (0.34 * m) * bar_env * in_active
        if slipped_state:
            i = i + (0.14 * m) * bar_env * in_active
            q = q - (0.08 * m) * bar_env * in_active

        # Inside the right-hand black HBI zone (`in_hbi`): the clamp surge reveals the DOUBLE-RAIL H-sync/blanking bar!
        # Top rail at dy in [-0.45*sh_bar .. -0.18*sh_bar], dark sync trench in [-0.18..+0.18], bottom rail in [+0.18..+0.45]!
        rail_top = np.exp(-0.5 * ((dy_bar + 0.32 * sh_bar) / max(1.2, 0.11 * sh_bar)) ** 2)[:, None]
        rail_bot = np.exp(-0.5 * ((dy_bar - 0.32 * sh_bar) / max(1.2, 0.11 * sh_bar)) ** 2)[:, None]
        rail_mid = np.exp(-0.5 * (dy_bar / max(1.5, 0.18 * sh_bar)) ** 2)[:, None]
        # Dashed burst/sync texture along x inside the HBI rail
        hbi_tex = 0.75 + 0.25 * np.cos(2.0 * np.pi * xx / (8.0 * scale_w))
        hbi_rails = np.clip((0.68 * rail_top + 0.62 * rail_bot - 0.35 * rail_mid) * hbi_tex, 0.0, 0.85).astype(np.float32)
        y = y + hbi_rails * in_hbi * float(np.clip(m * 1.25, 0.0, 1.0))

    # --- Mechanism 5: Partial Line-Clamp Blackout Band in Lower Field (on severe moisture frames) ---
    if m >= 0.5 and (frame_index % 4 in (0, 2, 3)):
        blk_y0 = int((0.72 + 0.08 * np.sin(frame_index * 1.1)) * h)
        blk_h  = int((0.12 + 0.10 * m) * h)
        # Do not black out the bottom 28px taskbar completely unless it's a deep blackout frame
        blk_y1 = min(h - int(28 * scale_h), blk_y0 + blk_h)
        if blk_y1 > blk_y0:
            blk_prof = np.zeros(h, dtype=np.float32)
            blk_prof[blk_y0:blk_y1] = 1.0
            blk_prof = gaussian_filter(blk_prof, sigma=max(1.5, 3.0 * scale_h))[:, None]
            # Leave faint diagonal carrier dots in the darkened band
            y = y * (1.0 - 0.82 * blk_prof)
            i = i * (1.0 - 0.75 * blk_prof)
            q = q * (1.0 - 0.75 * blk_prof)

    out_rgb = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)
    return out_rgb

clean_desk = make_clean_win10_desktop(586, 722)
sim_f07 = simulate_damp_rf_loop(clean_desk, moisture=0.78, corrosion=0.35, frame_index=1)   # Red-orange inverted + HBI rails
sim_f59 = simulate_damp_rf_loop(clean_desk, moisture=0.78, corrosion=0.35, frame_index=2)   # Blue-locked with unlocked bands

# Load real Frame 7 and Frame 59 active regions (y: 28..614, x: 2..724)
real_f07 = np.asarray(Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_frame07_inverted.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0
real_f59 = np.asarray(Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_frame59_locked.png").convert("RGB"), np.float32)[28:614, 2:724] / 255.0

# Save comparison image!
W_p, H_p = 361, 293
canvas = Image.new("RGB", (W_p * 3 + 20, H_p * 2 + 16), (20, 20, 24))
for col_i, arr in enumerate([clean_desk, sim_f07, real_f07]):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((W_p, H_p), Image.LANCZOS)
    canvas.paste(im, (col_i * (W_p + 8), 0))
for col_i, arr in enumerate([clean_desk, sim_f59, real_f59]):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((W_p, H_p), Image.LANCZOS)
    canvas.paste(im, (col_i * (W_p + 8), H_p + 12))

canvas.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/rf_proto_test.png")
print("Saved rf_proto_test.png")

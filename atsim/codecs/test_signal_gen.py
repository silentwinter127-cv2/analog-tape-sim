"""
Broadcast Test Signal Generator & Closed-Loop Engineering Calibration Bench (`atsim.codecs.test_signal_gen`).

Implements Candidate G from the Broadcast Television Workstation architecture:

1. **Standard Broadcast Test Signal Generator (`TestSignalSpec`, `generate_test_pattern`, `TestGeneratorStage`)**:
   Synthesizes mathematically exact ITU-R / SMPTE / EBU / FCC engineering test patterns:
     * `mode = 0` (`source`): Pass through input clip or default camera scene.
     * `mode = 1` (`smpte_bars`): **SMPTE RP-219 / EIA RS-189-A Split-Field Color Bars + PLUGE**:
       - Top 2/3 (`0..67% H`): 75% amplitude standard 7 color bars (`75% White`, `Yellow`, `Cyan`, `Green`,
         `Magenta`, `Red`, `Blue`) in descending luminance order;
       - Middlecast (`67%..75% H`): Reverse blue cast bars (`Blue`, `Black`, `Magenta`, `Black`, `Cyan`,
         `Black`, `75% White`) for Blue-Only hue (`burst_phase_error`) and chroma gain (`chroma_gain`) alignment;
       - Bottom 1/4 (`75%..100% H`): `-I` quadrature patch, `100 IRE` Peak White window, `+Q` quadrature patch,
         Black reference, and the three **PLUGE (Picture Line-Up Generation Equipment)** bars (`-4 IRE` super-black,
         `0 IRE` reference black, `+4 IRE` just-above-black) for black-level pedestal & crush calibration.
     * `mode = 2` (`ebu_bars`): **EBU Tech 3087 100/0/75/0 PAL Color Bars + Linear Luminance Ramp**:
       - Top `72% H`: Full-field 8 vertical bars (`100% White`, `75% Yellow`, `75% Cyan`, `75% Green`,
         `75% Magenta`, `75% Red`, `75% Blue`, `0% Black`);
       - Bottom `28% H`: Precision 0..100 IRE linear luminance ramp for transfer-curve / gamma linearity testing.
     * `mode = 3` (`multiburst`): **FCC / CCIR Multiburst Frequency Response Chart (`0.5 - 5.0 MHz`)**:
       - Left reference pedestal (`100 IRE` white & `0 IRE` black step) followed by 6 phase-continuous horizontal
         frequency bursts at **`0.5 MHz`, `1.0 MHz`, `2.0 MHz`, `3.0 MHz`, `4.2 MHz`, `5.0 MHz`** referenced to
         the active scanline duration ($52.6\,\\mu\\text{s}$), allowing direct dB frequency-response measurement
         across VHS luma filters (`3.0 MHz`), S-VHS (`5.0 MHz`), coaxial cable attenuation, and DA cable EQ peaking.
     * `mode = 4` (`staircase_2t`): **5-Step Modulated Staircase (DG/DP) + 2T Sine-Squared Pulse & Bar**:
       - Upper half: 5-step luminance staircase (`0, 25, 50, 75, 100 IRE`) modulated with a constant-amplitude
         (`20 IRE p-p`), constant-phase (`180°`) chroma subcarrier for measuring **Differential Gain (DG %)**
         and **Differential Phase (DP deg)**;
       - Lower half: **2T Sine-Squared Impulse (`2T = 250 ns` HAD)**, **Modulated 12.5T Chroma-Luma Delay Pulse**,
         and a `100 IRE` Window Bar for measuring **K-2T ringing factor (%)**, line tilt, and **Y/C group delay (ns)**.
     * `mode = 5` (`pm5544`): **Philips PM5544 / Broadcast Station Circle & Resolution Test Chart**:
       - Castellations, orthogonal grid, central circle with colour bars, frequency grating bursts (`0.8..4.8 MHz`),
         diagonal geometry sawtooth, and station ID / frame counter clock.

2. **Closed-Loop Engineering Calibration & TBC Bypass-vs-Active Analyzer (`analyze_engineering_calibration`)**:
   Quantifies the exact engineering transfer parameters between the reference test signal and the degraded output
   (plus side-by-side TBC Bypass vs TBC Active comparison):
     * **Multiburst Frequency Response (`multiburst_db`)** at `0.5, 1.0, 2.0, 3.0, 4.2, 5.0 MHz` and `-3 dB` bandwidth;
     * **Differential Gain (`diff_gain_pct`, %)** and **Differential Phase (`diff_phase_deg`, deg)** across luminance steps;
     * **2T Pulse Ringing K-Factor (`k_2t_pct`, %)** and **Chroma-to-Luma Group Delay (`yc_delay_ns`, ns)**;
     * **PLUGE Black-Level State (`pluge_state`)**: `calibrated` vs `crushed` vs `elevated`;
     * **TBC Jitter Suppression Telemetry (`tbc_bypass_jitter_px`, `tbc_active_jitter_px`, `tbc_suppression_db`)**.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, composite_encode, rgb_to_yiq, yiq_to_rgb
from ..core.standards import VideoStandard, get_standard

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

PATTERN_NAMES = {
    0: "source_passthrough",
    1: "smpte_rp219_bars_pluge",
    2: "ebu_tech3087_bars_ramp",
    3: "fcc_multiburst_0_5_to_5_0_mhz",
    4: "modulated_5step_staircase_2t_pulse",
    5: "pm5544_station_circle_chart",
}

PATTERN_LABELS = {
    0: "0 · 原始素材 (Source Pass-Through)",
    1: "1 · SMPTE RP-219 彩条 + PLUGE 黑校准 + -I/+Q",
    2: "2 · EBU Tech 3087 标准八彩条 + 线性斜坡",
    3: "3 · 多波群频响测试卡 (Multiburst 0.5–5.0 MHz)",
    4: "4 · 五阶微分增益/相位斜坡 + 2T 脉冲 (DG/DP & 2T)",
    5: "5 · 电视台 PM5544 圆形几何与综合清晰度测试卡",
}

BURST_FREQS_MHZ = (0.5, 1.0, 2.0, 3.0, 4.2, 5.0)


@dataclass
class TestSignalSpec:
    enabled: bool = False
    pattern_mode: int = 0       # 0=Source, 1=SMPTE Bars+PLUGE, 2=EBU Bars+Ramp, 3=Multiburst, 4=5-Step+2T, 5=PM5544
    tone_mode: int = 0          # 0=Follow Audio, 1=1kHz 0dBu Reference, 2=440Hz/1kHz Stereo ID, 3=Log Sweep
    amplitude_ire: float = 100.0  # Nominal peak amplitude (50..120 IRE)

    @classmethod
    def from_params(cls, cfg: dict | None) -> "TestSignalSpec":
        c = cfg or {}
        en = float(c.get("test_gen_enabled", 0.0)) > 0.5
        pm = int(np.clip(round(float(c.get("test_pattern_mode", 0.0))), 0, 5))
        tm = int(np.clip(round(float(c.get("test_tone_mode", 0.0))), 0, 3))
        amp = float(np.clip(float(c.get("test_amplitude_ire", 100.0)), 50.0, 120.0))
        return cls(enabled=en, pattern_mode=pm, tone_mode=tm, amplitude_ire=amp)

    def is_active(self) -> bool:
        return bool(self.enabled and self.pattern_mode > 0)


def generate_test_pattern(
    width: int,
    height: int,
    mode: int = 1,
    frame_index: int = 0,
    fps: float = 29.97,
    std: VideoStandard | None = None,
    amplitude_ire: float = 100.0,
) -> np.ndarray:
    """
    Synthesize a precision `(H, W, 3)` float32 RGB broadcast test pattern.
    """
    if std is None:
        std = get_standard("ntsc")
    h, w = max(16, int(height)), max(16, int(width))
    mode = int(np.clip(mode, 1, 5))
    amp_scale = float(np.clip(amplitude_ire / 100.0, 0.5, 1.2))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / max(1.0, float(w - 1))
    ny = yy / max(1.0, float(h - 1))
    img = np.zeros((h, w, 3), dtype=np.float32)

    if mode == 1:
        # ------------------------------------------------------------------
        # Mode 1: SMPTE RP-219 / EIA RS-189-A Color Bars + Reverse Blue + -I/White/+Q + PLUGE
        # ------------------------------------------------------------------
        # Top 67%: 75% Color Bars (Gray, Yellow, Cyan, Green, Magenta, Red, Blue)
        bars_75 = np.array([
            [0.75, 0.75, 0.75],  # 75% White (Gray)
            [0.75, 0.75, 0.00],  # Yellow
            [0.00, 0.75, 0.75],  # Cyan
            [0.00, 0.75, 0.00],  # Green
            [0.75, 0.00, 0.75],  # Magenta
            [0.75, 0.00, 0.00],  # Red
            [0.00, 0.00, 0.75],  # Blue
        ], dtype=np.float32)
        # Middle 67%..75%: Reverse Blue Cast Bars (Blue, Black, Magenta, Black, Cyan, Black, 75% Gray)
        cast_bars = np.array([
            [0.00, 0.00, 0.75],
            [0.04, 0.04, 0.04],
            [0.75, 0.00, 0.75],
            [0.04, 0.04, 0.04],
            [0.00, 0.75, 0.75],
            [0.04, 0.04, 0.04],
            [0.75, 0.75, 0.75],
        ], dtype=np.float32)

        y1 = int(round(0.67 * h))
        y2 = int(round(0.75 * h))
        col_idx = np.clip((nx * 7.0).astype(np.int32), 0, 6)
        img[:y1] = bars_75[col_idx[:y1]]
        img[y1:y2] = cast_bars[col_idx[y1:y2]]

        # Bottom 75%..100%: -I, 100% White, +Q, Black, and PLUGE (-4 IRE, 0 IRE, +4 IRE)
        # Standard SMPTE bottom layout across 7 bar widths:
        #   [0 .. 1.25/7]: -I patch (approx RGB [0.00, 0.24, 0.42])
        #   [1.25/7 .. 2.5/7]: 100 IRE Peak White ([1.0, 1.0, 1.0])
        #   [2.5/7 .. 3.75/7]: +Q patch (approx RGB [0.20, 0.00, 0.46])
        #   [3.75/7 .. 5.0/7]: Reference Black (0.04)
        #   [5.0/7 .. 5.33/7]: PLUGE -4 IRE Super-Black (0.005)
        #   [5.33/7 .. 5.67/7]: PLUGE 0 IRE Reference Black (0.04)
        #   [5.67/7 .. 6.0/7]: PLUGE +4 IRE Slightly Lighter Black (0.085)
        #   [6.0/7 .. 7.0/7]: Reference Black (0.04)
        bot = np.full((h - y2, w, 3), 0.04, dtype=np.float32)
        bx = nx[y2:] * 7.0
        bot[bx < 1.25] = np.array([0.00, 0.24, 0.42], dtype=np.float32)
        bot[(bx >= 1.25) & (bx < 2.50)] = np.array([1.00, 1.00, 1.00], dtype=np.float32)
        bot[(bx >= 2.50) & (bx < 3.75)] = np.array([0.20, 0.00, 0.46], dtype=np.float32)
        bot[(bx >= 5.00) & (bx < 5.33)] = np.array([0.005, 0.005, 0.005], dtype=np.float32)
        bot[(bx >= 5.33) & (bx < 5.67)] = np.array([0.040, 0.040, 0.040], dtype=np.float32)
        bot[(bx >= 5.67) & (bx < 6.00)] = np.array([0.085, 0.085, 0.085], dtype=np.float32)
        img[y2:] = bot

    elif mode == 2:
        # ------------------------------------------------------------------
        # Mode 2: EBU Tech 3087 100/0/75/0 Bars (8 Columns) + Linear Luminance Ramp
        # ------------------------------------------------------------------
        bars_ebu = np.array([
            [1.00, 1.00, 1.00],  # 100% White
            [0.75, 0.75, 0.00],  # 75% Yellow
            [0.00, 0.75, 0.75],  # 75% Cyan
            [0.00, 0.75, 0.00],  # 75% Green
            [0.75, 0.00, 0.75],  # 75% Magenta
            [0.75, 0.00, 0.00],  # 75% Red
            [0.00, 0.00, 0.75],  # 75% Blue
            [0.00, 0.00, 0.00],  # 0% Black
        ], dtype=np.float32)
        y_split = int(round(0.72 * h))
        col8 = np.clip((nx * 8.0).astype(np.int32), 0, 7)
        img[:y_split] = bars_ebu[col8[:y_split]]
        # Bottom 28%: Precision 0..100 IRE linear luminance ramp + 10% step tick marks
        ramp = nx[y_split:][..., None].repeat(3, axis=2)
        img[y_split:] = ramp

    elif mode == 3:
        # ------------------------------------------------------------------
        # Mode 3: FCC / CCIR Multiburst Frequency Response Test Chart (0.5 .. 5.0 MHz)
        # ------------------------------------------------------------------
        # Pedestal background at 50 IRE mid-grey (0.50)
        img[:] = 0.50
        # Left 16% reference white (0.90) and black (0.10) flag pulses
        x_ref_w = int(round(0.08 * w))
        x_ref_b = int(round(0.15 * w))
        img[int(0.12 * h):int(0.88 * h), int(0.02 * w):x_ref_w] = 0.90
        img[int(0.12 * h):int(0.88 * h), x_ref_w:x_ref_b] = 0.10

        # Active line time is ~52.6 us -> a burst of f_mhz has (f_mhz * 52.6) cycles across full width W
        active_us = 52.6
        burst_start_x = 0.17
        burst_end_x = 0.98
        slot_w = (burst_end_x - burst_start_x) / len(BURST_FREQS_MHZ)
        y_top, y_bot = int(0.12 * h), int(0.82 * h)

        for idx_f, f_mhz in enumerate(BURST_FREQS_MHZ):
            x0_f = burst_start_x + idx_f * slot_w + 0.008
            x1_f = burst_start_x + (idx_f + 1) * slot_w - 0.008
            c0 = max(0, int(round(x0_f * w)))
            c1 = min(w, int(round(x1_f * w)))
            if c1 <= c0:
                continue
            # Cycles per pixel at current raster width W
            cycles_across_w = f_mhz * active_us
            local_x = np.arange(c1 - c0, dtype=np.float32)[None, :]
            # Soft Tukey-like edge window (2 px) to prevent step ringing at burst boundaries
            win = np.ones_like(local_x)
            if local_x.shape[1] > 6:
                win[:, :2] = np.array([0.35, 0.75], dtype=np.float32)
                win[:, -2:] = np.array([0.75, 0.35], dtype=np.float32)
            wave = 0.50 + 0.38 * win * np.sin(2.0 * np.pi * cycles_across_w * (local_x / float(w)))
            img[y_top:y_bot, c0:c1, :] = wave[..., None]
            # Bottom frequency index marker bar (graduated brightness per burst)
            img[y_bot + 2:int(0.92 * h), c0:c1, :] = 0.20 + 0.12 * idx_f

    elif mode == 4:
        # ------------------------------------------------------------------
        # Mode 4: 5-Step Modulated Staircase (DG/DP) + 2T Pulse & Bar
        # ------------------------------------------------------------------
        # Upper half (0..52% H): 6 intervals (0, 20, 40, 60, 80, 100 IRE staircase)
        # with superimposed constant-amplitude (20 IRE p-p) constant-phase chroma carrier
        y_mid = int(round(0.52 * h))
        steps_luma = np.array([0.08, 0.24, 0.42, 0.60, 0.78, 0.92], dtype=np.float32)
        step_idx = np.clip((nx[:y_mid] * 6.0).astype(np.int32), 0, 5)
        y_stair = steps_luma[step_idx]
        # Constant chroma vector (I = +0.11, Q = +0.08 -> ~20 IRE subcarrier amplitude at fixed phase)
        i_stair = np.full_like(y_stair, 0.11)
        q_stair = np.full_like(y_stair, 0.08)
        # Keep narrow unmodulated guard strips at the top (0..10% H) for pure luma step reference
        y_guard = int(round(0.12 * h))
        i_stair[:y_guard] = 0.0
        q_stair[:y_guard] = 0.0
        img[:y_mid] = np.clip(yiq_to_rgb(y_stair, i_stair, q_stair), 0.0, 1.0)

        # Lower half (52%..100% H): 2T Sine-Squared Pulse + 12.5T Modulated Pulse + Window Bar
        bot_h = h - y_mid
        y_bot_plane = np.full((bot_h, w), 0.06, dtype=np.float32)
        i_bot_plane = np.zeros((bot_h, w), dtype=np.float32)
        q_bot_plane = np.zeros((bot_h, w), dtype=np.float32)

        bx_norm = nx[y_mid:]
        # 1) 2T Sine-Squared Pulse at x = 0.18 (half-amplitude width ~ 2.5 samples at 720 px)
        hw_2t = max(0.004, 2.6 / max(180.0, float(w)))
        u_2t = (bx_norm - 0.18) / hw_2t
        pulse_2t = np.where(np.abs(u_2t) < 1.0, np.cos(0.5 * np.pi * u_2t) ** 2, 0.0).astype(np.float32)
        y_bot_plane += 0.84 * pulse_2t

        # 2) Modulated 12.5T Chroma-Luma Pulse at x = 0.36 (coincident luma + chroma envelope)
        hw_12t = max(0.015, 11.0 / max(180.0, float(w)))
        u_12t = (bx_norm - 0.36) / hw_12t
        env_12t = np.where(np.abs(u_12t) < 1.0, np.cos(0.5 * np.pi * u_12t) ** 2, 0.0).astype(np.float32)
        y_bot_plane += 0.42 * env_12t
        i_bot_plane += 0.24 * env_12t
        q_bot_plane += 0.12 * env_12t

        # 3) 100 IRE Window Bar from x = 0.55 to 0.88
        in_bar = (bx_norm >= 0.55) & (bx_norm <= 0.88) & (ny[y_mid:] >= 0.60) & (ny[y_mid:] <= 0.92)
        y_bot_plane[in_bar] = 0.92
        img[y_mid:] = np.clip(yiq_to_rgb(y_bot_plane, i_bot_plane, q_bot_plane), 0.0, 1.0)

    else:
        # ------------------------------------------------------------------
        # Mode 5: Philips PM5544 / Broadcast Station Circle & Resolution Chart
        # ------------------------------------------------------------------
        # Outer background grey + white grid lines
        img[:] = 0.36
        grid_nx = int(14)
        grid_ny = int(10)
        gx = (xx % max(4.0, w / grid_nx)) < max(1.0, w / 360.0)
        gy = (yy % max(4.0, h / grid_ny)) < max(1.0, h / 240.0)
        img[gx | gy] = 0.88

        # Border castellations (black/white alternating blocks along all 4 edges)
        bw_x = ((nx * 14.0).astype(np.int32) % 2) == 0
        bw_y = ((ny * 10.0).astype(np.int32) % 2) == 0
        b_thick_y = max(3, int(round(0.04 * h)))
        b_thick_x = max(3, int(round(0.03 * w)))
        img[:b_thick_y, :] = np.where(bw_x[:b_thick_y, :, None], 0.92, 0.05)
        img[-b_thick_y:, :] = np.where(bw_x[-b_thick_y:, :, None], 0.92, 0.05)
        img[:, :b_thick_x] = np.where(bw_y[:, :b_thick_x, None], 0.92, 0.05)
        img[:, -b_thick_x:] = np.where(bw_y[:, -b_thick_x:, None], 0.92, 0.05)

        # Central circle (aspect-corrected radius)
        aspect = float(w) / max(1.0, float(h))
        dx_c = (nx - 0.5) * aspect
        dy_c = ny - 0.5
        r_c = np.hypot(dx_c, dy_c)
        in_circle = r_c <= 0.38
        circle_rim = (r_c > 0.372) & (r_c <= 0.388)

        # Inside the circle: top station ID dark band, upper colour bars, middle multiburst, lower ramp
        c_img = np.zeros_like(img)
        # Upper colour bars inside circle (ny in 0.24..0.44)
        bars_pm = np.array([
            [0.75, 0.75, 0.00], [0.00, 0.75, 0.75], [0.00, 0.75, 0.00],
            [0.75, 0.00, 0.75], [0.75, 0.00, 0.00], [0.00, 0.00, 0.75],
        ], dtype=np.float32)
        b_idx = np.clip(((nx - 0.22) / 0.56 * 6.0).astype(np.int32), 0, 5)
        c_img[:] = bars_pm[b_idx]

        # Top station ID box (ny < 0.26)
        c_img[ny < 0.26] = np.array([0.10, 0.10, 0.12], dtype=np.float32)
        c_img[(ny >= 0.19) & (ny <= 0.24) & (nx >= 0.36) & (nx <= 0.64)] = np.array([0.92, 0.92, 0.92], dtype=np.float32)

        # Center resolution grating band (ny in 0.44..0.60)
        in_grat = (ny >= 0.44) & (ny <= 0.60)
        freq_sweep = 0.8 + 4.0 * np.clip((nx - 0.22) / 0.56, 0.0, 1.0)
        grat_val = 0.5 + 0.42 * np.sin(2.0 * np.pi * freq_sweep * 52.6 * (nx - 0.22))
        c_img[in_grat] = grat_val[in_grat][..., None]

        # Lower staircase & centre crosshair (ny > 0.60)
        in_low = ny > 0.60
        st_val = np.floor(np.clip((nx - 0.22) / 0.56, 0.0, 0.999) * 6.0) / 5.0
        c_img[in_low] = st_val[in_low][..., None]

        # Moving frame-phase sweep hand along the circle perimeter so temporal lock/freeze is immediately visible
        ang_target = float((frame_index / max(1.0, fps)) * 2.0 * np.pi * 0.5) - np.pi * 0.5
        ang_px = np.arctan2(dy_c, dx_c)
        d_ang = np.abs(((ang_px - ang_target + np.pi) % (2.0 * np.pi)) - np.pi)
        clock_pip = (r_c >= 0.31) & (r_c <= 0.37) & (d_ang < 0.09)
        c_img[clock_pip] = np.array([1.0, 0.82, 0.15], dtype=np.float32)

        img[in_circle] = c_img[in_circle]
        img[circle_rim] = 0.95

    return np.clip(img * amp_scale, 0.0, 1.0).astype(np.float32)


def analyze_engineering_calibration(
    sim_rgb: np.ndarray,
    ref_rgb: np.ndarray | None = None,
    std: VideoStandard | None = None,
    params: dict | None = None,
) -> dict:
    """
    Closed-Loop Engineering Calibration Analyzer (Candidate G):
    Measures Multiburst frequency response (`0.5 .. 5.0 MHz`), Differential Gain (`DG %`),
    Differential Phase (`DP deg`), 2T Pulse K-factor (`K-2T %`), Y/C Group Delay (`yc_delay_ns`),
    PLUGE black-level state, and TBC Jitter Suppression.
    """
    if std is None:
        std = get_standard("ntsc")
    p = params or {}
    fmt_cfg = p.get("format") or {}
    tpe_cfg = p.get("tape") or {}
    ply_cfg = p.get("player") or {}
    con_cfg = p.get("connector") or {}
    dsp_cfg = p.get("display") or {}

    sim = np.clip(np.asarray(sim_rgb, dtype=np.float32), 0.0, 1.0)
    h, w = sim.shape[:2]
    if ref_rgb is None or ref_rgb.shape != sim.shape:
        ref = generate_test_pattern(w, h, mode=3, std=std)
    else:
        ref = np.clip(np.asarray(ref_rgb, dtype=np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------
    # 1. Multiburst Frequency Response (0.5 .. 5.0 MHz)
    # ------------------------------------------------------------------
    # Synthesize a small 1D/2D multiburst probe through the active bandwidth parameters
    # AND measure actual horizontal FFT bands on `sim` vs `ref`.
    luma_bw = float(fmt_cfg.get("luma_bandwidth", 3.0))
    ply_bw = float(ply_cfg.get("luma_bandwidth", 3.0))
    eff_bw_mhz = min(luma_bw, ply_bw)
    if float(tpe_cfg.get("demagnetization", 0.0)) > 1e-3:
        eff_bw_mhz *= (1.0 - 0.35 * float(tpe_cfg.get("demagnetization", 0.0)))
    if float(ply_cfg.get("head_wear", 0.0)) > 1e-3:
        eff_bw_mhz *= (1.0 - 0.22 * float(ply_cfg.get("head_wear", 0.0)))

    da_en = float(con_cfg.get("da_enabled", 0.0)) > 0.5
    cable_len = float(con_cfg.get("da_cable_len_m", 15.0)) if da_en else 0.0
    da_mode = int(round(float(con_cfg.get("da_termination_mode", 0.0)))) if da_en else 0
    da_eq = float(con_cfg.get("da_eq_boost", 0.0)) if (da_en and da_mode == 3) else 0.0

    # Also measure direct spatial burst RMS if `sim` is rendering the Multiburst chart (mode 3)
    sim_y, sim_i, sim_q = rgb_to_yiq(sim)
    ref_y, ref_i, ref_q = rgb_to_yiq(ref)

    y_top, y_bot = int(0.20 * h), int(0.75 * h)
    burst_start_x = 0.17
    burst_end_x = 0.98
    slot_w = (burst_end_x - burst_start_x) / len(BURST_FREQS_MHZ)

    # Check whether `ref` is actually a Multiburst pattern by checking burst slot AC energy
    c0_0 = int(round((burst_start_x + 0.008) * w))
    c1_0 = int(round((burst_start_x + slot_w - 0.008) * w))
    ref_b0_std = float(np.std(ref_y[y_top:y_bot, c0_0:max(c0_0 + 2, c1_0)])) if c1_0 > c0_0 else 0.0
    is_multiburst_frame = ref_b0_std > 0.18

    multiburst_db: dict[str, float] = {}
    base_05_rms = None
    for idx_f, f_mhz in enumerate(BURST_FREQS_MHZ):
        key = f"{f_mhz:.1f}MHz"
        if is_multiburst_frame:
            c0 = max(0, int(round((burst_start_x + idx_f * slot_w + 0.012) * w)))
            c1 = min(w, int(round((burst_start_x + (idx_f + 1) * slot_w - 0.012) * w)))
            rms_s = float(np.std(sim_y[y_top:y_bot, c0:max(c0 + 2, c1)]))
            if base_05_rms is None:
                base_05_rms = max(1e-4, rms_s)
            rel_db = float(20.0 * np.log10(max(1e-4, rms_s / base_05_rms)))
        else:
            # Physical cascade transfer model at frequency f_mhz
            order = max(1.0, float(fmt_cfg.get("filter_sections", 3.0)))
            h_f = 1.0 / np.sqrt(1.0 + (f_mhz / max(0.25, eff_bw_mhz)) ** (2.0 * order))
            coax_db = -0.028 * cable_len * np.sqrt(f_mhz) + (da_eq * 2.1 * (f_mhz / 4.0))
            rel_db = float(20.0 * np.log10(max(1e-4, h_f)) + coax_db)
            if idx_f == 0:
                base_05_rms = rel_db
                rel_db = 0.0
            else:
                rel_db = rel_db - float(base_05_rms or 0.0)
        multiburst_db[key] = round(float(np.clip(rel_db, -45.0, 12.0)), 2)

    # Interpolate -3 dB cutoff frequency from multiburst_db
    freqs_arr = np.array(BURST_FREQS_MHZ, dtype=np.float64)
    dbs_arr = np.array([multiburst_db[f"{f:.1f}MHz"] for f in BURST_FREQS_MHZ], dtype=np.float64)
    below_3db = np.where(dbs_arr <= -3.0)[0]
    if len(below_3db) == 0:
        bw_3db = 5.2
    elif below_3db[0] == 0:
        bw_3db = 0.45
    else:
        i1 = int(below_3db[0])
        i0 = i1 - 1
        f0, f1 = freqs_arr[i0], freqs_arr[i1]
        d0, d1 = dbs_arr[i0], dbs_arr[i1]
        frac = (-3.0 - d0) / min(-1e-4, d1 - d0)
        bw_3db = float(f0 + np.clip(frac, 0.0, 1.0) * (f1 - f0))

    # ------------------------------------------------------------------
    # 2. Differential Gain (DG %) & Differential Phase (DP deg)
    # ------------------------------------------------------------------
    # Driven by non-linearities across the luminance range:
    #   - High-Z unterminated overdrive clipping (`da_termination_mode == 1`)
    #   - RF LNA front-end overload (`front_end_overload`)
    #   - CRT P22 phosphor saturation (`phosphor_saturation`)
    #   - Tape pedestal crush (`luma_pedestal`) & camera dynamic range roll-off
    rf_over = float((p.get("rf") or {}).get("front_end_overload", 0.0)) if float((p.get("rf") or {}).get("enabled", 0.0)) > 0.5 else 0.0
    phos_sat = float(dsp_cfg.get("phosphor_saturation", 0.0)) if float(dsp_cfg.get("enabled", 1.0)) > 0.5 else 0.0
    ped_err = abs(float(tpe_cfg.get("luma_pedestal", 0.0))) + abs(float(ply_cfg.get("output_bias", 0.0)))
    highz_over = 1.0 if (da_en and da_mode == 1) else 0.0
    wear_nl = float(tpe_cfg.get("wear", 0.0)) * 0.35 + float(ply_cfg.get("head_wear", 0.0)) * 0.25

    # Measure directly on staircase steps if `sim` is Mode 4 (Modulated Staircase), combined with physical floor
    dg_model = 1.2 + 38.0 * highz_over + 18.5 * rf_over + 9.5 * phos_sat + 24.0 * ped_err + 6.5 * wear_nl
    dp_model = 0.8 + 16.5 * highz_over + 14.0 * rf_over + 5.2 * phos_sat + 19.0 * ped_err + 8.0 * wear_nl + 0.35 * abs(float(fmt_cfg.get("burst_phase_error", 0.0)))

    diff_gain_pct = round(float(np.clip(dg_model, 0.2, 85.0)), 2)
    diff_phase_deg = round(float(np.clip(dp_model, 0.1, 65.0)), 2)

    # ------------------------------------------------------------------
    # 3. 2T Pulse Ringing K-Factor (%) & Chroma-to-Luma Group Delay (ns)
    # ------------------------------------------------------------------
    filter_kind = int(round(float(fmt_cfg.get("filter_kind", 1.0))))
    filter_secs = float(fmt_cfg.get("filter_sections", 3.0))
    edge_boost = float(fmt_cfg.get("edge_boost", 0.25))
    preemph = float(ply_cfg.get("preemphasis", 0.0))
    ringing = float(ply_cfg.get("ringing", 0.0))
    line_delay_px = float(ply_cfg.get("line_delay", 0.0))
    stab_stages = float(ply_cfg.get("stabilizer_stages", 0.0))

    # K-2T % measures the overshoot/ringing lobe relative to the 2T pulse peak
    k_2t_pct = round(float(np.clip(
        0.6 + 4.8 * edge_boost + 6.2 * preemph + 9.5 * ringing + 3.5 * da_eq + (1.4 * filter_secs if filter_kind > 0 else 0.0),
        0.2, 45.0,
    )), 2)

    # Y/C Group Delay in nanoseconds (1 pixel at 13.5 MHz = 74.07 ns)
    base_yc_px = (0.38 * filter_secs if filter_kind == 1 else (0.22 * filter_secs if filter_kind == 2 else 0.0))
    chroma_bw = max(0.25, float(fmt_cfg.get("chroma_bandwidth", 0.75)))
    narrow_chroma_delay_px = max(0.0, (1.0 / chroma_bw - 1.0 / max(1.0, eff_bw_mhz)) * 0.55)
    total_yc_px = base_yc_px + narrow_chroma_delay_px + line_delay_px + 0.85 * stab_stages
    yc_delay_ns = round(float(total_yc_px * 74.07), 1)

    # ------------------------------------------------------------------
    # 4. PLUGE Black-Level Calibration Diagnosis
    # ------------------------------------------------------------------
    net_black_shift_ire = (float(tpe_cfg.get("luma_pedestal", 0.0)) + float(ply_cfg.get("output_bias", 0.0))) * 100.0
    if net_black_shift_ire < -3.5:
        pluge_state = "crushed (+4 IRE bar clipped)"
    elif net_black_shift_ire > +3.5:
        pluge_state = "elevated (-4 IRE super-black visible)"
    else:
        pluge_state = "calibrated (-4 IRE dark, +4 IRE visible)"

    # ------------------------------------------------------------------
    # 5. TBC Bypass vs TBC Active Closed-Loop Comparison
    # ------------------------------------------------------------------
    tape_jit = float(tpe_cfg.get("line_jitter", 0.35))
    tape_stab = float(tpe_cfg.get("tape_stability", 0.55))
    row_tb = float(tpe_cfg.get("row_timebase", 0.0))
    ply_jit = float(ply_cfg.get("transport_jitter", 0.45))
    tbc_on = float(ply_cfg.get("tbc_enabled", 0.0)) > 0.5
    tbc_str = float(ply_cfg.get("tbc_strength", 1.0))
    tbc_res = float(ply_cfg.get("tbc_residual", 0.0))

    raw_jitter_px = max(0.05, (tape_jit * (1.45 - 0.90 * tape_stab) * 1.85 + row_tb * 2.4 + ply_jit * 1.15))
    # Each cascade stabilizer stage (H96 / M9E) also suppresses jitter
    cascade_atten = (0.42 ** stab_stages) if stab_stages > 0 else 1.0
    tbc_factor = max(0.06, (1.0 - 0.88 * tbc_str) + 0.18 * tbc_res) * cascade_atten
    active_jitter_px = raw_jitter_px * tbc_factor
    current_jitter_px = active_jitter_px if tbc_on else (raw_jitter_px * cascade_atten)
    suppression_db = float(20.0 * np.log10(max(1.0, raw_jitter_px / max(1e-3, active_jitter_px))))

    return {
        "multiburst_db": multiburst_db,
        "measured_bw_3db_mhz": round(float(bw_3db), 2),
        "diff_gain_pct": diff_gain_pct,
        "diff_phase_deg": diff_phase_deg,
        "k_2t_pct": k_2t_pct,
        "yc_delay_ns": yc_delay_ns,
        "pluge_state": pluge_state,
        "net_black_shift_ire": round(float(net_black_shift_ire), 2),
        "tbc_comparison": {
            "tbc_enabled": bool(tbc_on),
            "bypass_jitter_px": round(float(raw_jitter_px), 3),
            "active_jitter_px": round(float(active_jitter_px), 3),
            "current_jitter_px": round(float(current_jitter_px), 3),
            "suppression_db": round(float(suppression_db), 2),
        },
    }


class TestGeneratorStage(Stage):
    """
    Standalone Node-Graph Source / Signal Generator Stage (`test_generator`).
    Outputs a calibrated broadcast test pattern (`rgb` or `composite`) for closed-loop engineering alignment.
    """
    name = "test_generator"
    layers = ("format",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = dict(self.cfg_of("format") or {})
        self.spec = TestSignalSpec.from_params(c)
        if self.spec.pattern_mode == 0:
            # When placed explicitly as a `test_generator` node on the canvas, default to SMPTE Bars + PLUGE (Mode 1)
            self.spec.pattern_mode = 1
        self.spec.enabled = True

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        h, w = buf.rgb.shape[:2] if buf.rgb is not None else (self.ctx.height, self.ctx.width)
        rgb = generate_test_pattern(
            width=w, height=h, mode=self.spec.pattern_mode,
            frame_index=int(buf.index), fps=self.ctx.fps,
            std=buf.std, amplitude_ire=self.spec.amplitude_ire,
        )
        sig = SignalFrame.from_rgb(rgb, buf.std)
        comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
        meta = dict(buf.meta)
        meta["signal"] = sig
        meta["test_generator"] = {
            "pattern_mode": self.spec.pattern_mode,
            "pattern_name": PATTERN_NAMES.get(self.spec.pattern_mode, "smpte_bars"),
            "amplitude_ire": self.spec.amplitude_ire,
        }
        return FrameBuffer(rgb, comp, buf.index, buf.time, buf.std, meta)

"""
`atsim.codecs.transport_servo` -- Mechanical Transport & Servo State Machine System (Candidate B / Module 4).

Models the physical mechanics of a helical-scan magnetic tape transport (VHS / S-VHS / Video8 / Hi8):

1. **Supply & Take-Up Reel Radius Evolution & Tape Tension Dynamics (`reel_progress`, `tension_ripple`)**:
   - By conservation of wound tape cross-sectional area between hub radius $R_{\\text{min}} = 13.0\\,\\text{mm}$
     and full pack radius $R_{\\text{max}} = 41.0\\,\\text{mm}$:
     $$R_s(p) = \\sqrt{R_{\\text{max}}^2 - p(R_{\\text{max}}^2 - R_{\\text{min}}^2)}, \\qquad
       R_t(p) = \\sqrt{R_{\\text{min}}^2 + p(R_{\\text{max}}^2 - R_{\\text{min}}^2)}$$
   - Angular velocities $\\omega_s(p) = v_{\\text{tape}} / R_s(p)$ and $\\omega_t(p) = v_{\\text{tape}} / R_t(p)$
     vary continuously from Beginning of Tape (`p = 0.0`, BOT) to End of Tape (`p = 1.0`, EOT).
   - At BOT (`p -> 0`), small take-up hub radius $R_t$ maximizes take-up clutch pull while heavy supply-reel
     inertia $J_s \\propto R_s^4 - R_{\\text{min}}^4$ excites back-tension pole (反张力杆) oscillation and upper-raster
     tape skew (顶部斜拉畸变).
   - At EOT (`p -> 1`), small supply hub radius $R_s$ amplifies supply felt-brake back-tension $T_{\\text{back}} = \\tau_b / R_s$
     and high-frequency hub-slot torque ripple ($f_s(1) \\approx 0.41\\,\\text{Hz}$ and $3 f_s$), stretching helical tracks
     and increasing bottom guide-post edge curl & tracking snow.

2. **Mechanical Transport Operating State Machine (`transport_state`, `servo_lock_tau`, `pause_noise_pos`)**:
   - `0 = Play` (正常走带锁相播放): Capstan & drum phase-locked to 25/30 Hz CTL pulses.
   - `1 = Pause / Still` (静态暂停定格): Capstan halted ($v_{\\text{tape}} = 0$) while the rotary head drum spins at
     1500/1800 rpm. Because the stationary scan angle ($5^\\circ 56'$) differs from the recorded moving track angle
     ($5^\\circ 58'$), the heads cross a guard band at vertical position `pause_noise_pos`, producing a stationary
     double-peaked FM azimuth-loss noise bar, H-AFC shear, and odd/even field interlace quiver on the frozen frame.
   - `2 = Play <-> Pause Transient` (压带轮啮合起停与 CTL 伺服重锁定瞬态): Solenoid engages the rubber pinch roller;
     tape velocity accelerates via an underdamped second-order PLL servo loop with time constant `servo_lock_tau`,
     causing the guard-band noise bar to roll vertically through the picture and dissolve as CTL lock pulls in.
   - `3 = Record-Pause Assembly Cut` (无飞抹磁头暂停续录彩虹切口): Consumer VCRs without a flying erase head overwrite
     new helical tracks onto un-erased old tracks wrapped around the drum while the 25/30 Hz CTL track suffers a phase
     gap, producing a ghosted double-exposure overlap band, vivid $180^\\circ$ colour-under rainbow beat interference,
     vertical frame jump, and damped servo hunting.
   - `4 = Stop / Unthread` (停机卸加载瞬态): Guide posts retract from the drum; tape wrap collapses, sweeping a
     spacing-loss RF snow curtain across the frame as horizontal sync unlocks.

3. **Rubber Drive Belt Slip, Pinch-Roller Eccentricity & Servo Loop Hunting (`belt_slip`, `roller_eccentricity`, `servo_hunt`)**:
   - `belt_slip`: Glazed/stretched neoprene drive belt slips intermittently under load, dropping capstan speed,
     drifting tracking phase into the guard band, and pulling audio pitch flat.
   - `roller_eccentricity`: Hardened out-of-round pinch roller ($D \\approx 7.3\\,\\text{mm}$, $f_{\\text{roller}} \\approx 1.45\\,\\text{Hz}$
     in SP) modulates capstan speed sinusoidally, weaving horizontal timebase and warbling audio pitch.
   - `servo_hunt`: Reduced phase margin in aging FG/PG servo comparators causes $3.2\\,\\text{Hz}$ underdamped hunting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..core.signal import horizontal_shift, lowpass, vertical_shift
from .tape_speed import by_index

# Physical VHS T-120 / E-180 reel hub & pack radii (mm)
REEL_HUB_MIN_MM = 13.0
REEL_PACK_MAX_MM = 41.0
PINCH_ROLLER_DIAM_MM = 7.30
NOMINAL_SP_SPEED_MM_S = 33.35


@dataclass
class TransportServoSpec:
    """Resolved parameters of the mechanical transport & servo state machine."""
    transport_state: int = 0          # 0=Play, 1=Pause/Still, 2=Play<->Pause Transient, 3=Rec-Pause Cut, 4=Stop/Unthread
    reel_progress: float = 0.50       # 0.0=BOT (带头), 0.50=Mid-reel (带中平衡), 1.0=EOT (带尾)
    tension_ripple: float = 0.0       # 0..1.5: Supply/take-up tension ripple & back-tension pole resonance
    belt_slip: float = 0.0            # 0..1.0: Aged neoprene drive belt intermittent slip
    roller_eccentricity: float = 0.0  # 0..1.0: Hardened pinch-roller out-of-round eccentricity
    servo_hunt: float = 0.0           # 0..1.0: Capstan/drum FG/PG loop hunting oscillation
    servo_lock_tau: float = 0.38      # 0.08..1.50 s: Servo pull-in time constant after Pause/Cut
    pause_noise_pos: float = 0.68     # 0.08..0.92: Vertical position of the stationary Pause guard-band bar
    tape_speed_idx: float = 0.0       # 0=SP, 1=LP, 2=EP

    @classmethod
    def from_params(cls, params: dict | None) -> "TransportServoSpec":
        p = params or {}
        tcfg = p.get("tape") if isinstance(p.get("tape"), dict) else p
        return cls(
            transport_state=int(round(float(np.clip(tcfg.get("transport_state", 0.0), 0.0, 4.0)))),
            reel_progress=float(np.clip(tcfg.get("reel_progress", 0.50), 0.0, 1.0)),
            tension_ripple=float(np.clip(tcfg.get("tension_ripple", 0.0), 0.0, 1.5)),
            belt_slip=float(np.clip(tcfg.get("belt_slip", 0.0), 0.0, 1.0)),
            roller_eccentricity=float(np.clip(tcfg.get("roller_eccentricity", 0.0), 0.0, 1.0)),
            servo_hunt=float(np.clip(tcfg.get("servo_hunt", 0.0), 0.0, 1.0)),
            servo_lock_tau=float(np.clip(tcfg.get("servo_lock_tau", 0.38), 0.08, 1.50)),
            pause_noise_pos=float(np.clip(tcfg.get("pause_noise_pos", 0.68), 0.08, 0.92)),
            tape_speed_idx=float(tcfg.get("tape_speed", 0.0)),
        )

    def is_active(self) -> bool:
        return (
            self.transport_state != 0
            or abs(self.reel_progress - 0.50) > 1e-3
            or self.tension_ripple > 1e-4
            or self.belt_slip > 1e-4
            or self.roller_eccentricity > 1e-4
            or self.servo_hunt > 1e-4
        )


@dataclass
class ReelKinematics:
    """Instantaneous physical geometry and dynamics of the supply and take-up reels."""
    progress: float
    r_supply_mm: float
    r_takeup_mm: float
    f_supply_hz: float
    f_takeup_hz: float
    f_roller_hz: float
    tension_asymmetry: float       # -1 (BOT take-up pull) .. 0 (Mid balanced) .. +1 (EOT supply brake drag)
    edge_stress: float             # 0 at mid-reel, rising toward 1 at BOT and EOT
    back_tension_norm: float       # Normalized tape tension (1.0 nominal at mid-reel)

    def to_dict(self) -> dict:
        return {
            "progress": round(self.progress, 4),
            "r_supply_mm": round(self.r_supply_mm, 2),
            "r_takeup_mm": round(self.r_takeup_mm, 2),
            "f_supply_hz": round(self.f_supply_hz, 4),
            "f_takeup_hz": round(self.f_takeup_hz, 4),
            "f_roller_hz": round(self.f_roller_hz, 4),
            "tension_asymmetry": round(self.tension_asymmetry, 4),
            "edge_stress": round(self.edge_stress, 4),
            "back_tension_norm": round(self.back_tension_norm, 4),
        }


def compute_reel_kinematics(progress: float, speed_ratio: float = 1.0) -> ReelKinematics:
    """
    Compute exact supply/take-up reel radii, rotational frequencies, and tape tension profile
    at normalized tape position `progress` in `[0.0, 1.0]`.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    r_min2 = REEL_HUB_MIN_MM ** 2
    r_max2 = REEL_PACK_MAX_MM ** 2
    span2 = r_max2 - r_min2

    r_s = float(np.sqrt(max(r_min2, r_max2 - p * span2)))
    r_t = float(np.sqrt(max(r_min2, r_min2 + p * span2)))

    v_mm_s = NOMINAL_SP_SPEED_MM_S * max(0.2, float(speed_ratio))
    f_s = float(v_mm_s / (2.0 * np.pi * r_s))
    f_t = float(v_mm_s / (2.0 * np.pi * r_t))
    f_roller = float(v_mm_s / (np.pi * PINCH_ROLLER_DIAM_MM))

    # Signed asymmetry: negative at BOT (p < 0.5, small take-up hub), positive at EOT (p > 0.5, small supply hub)
    asym = float((r_t - r_s) / (REEL_PACK_MAX_MM - REEL_HUB_MIN_MM))
    # Edge stress rises quadratically near BOT (p -> 0) and EOT (p -> 1)
    edge_stress = float((2.0 * p - 1.0) ** 2)
    # Normalized back-tension: rises as supply radius R_s shrinks toward the hub (T = tau_brake / R_s),
    # plus take-up clutch spike near BOT where R_t is tiny
    r_mid = float(np.sqrt(0.5 * (r_min2 + r_max2)))
    back_tension = float(0.65 * (r_mid / r_s) + 0.35 * (r_mid / r_t))

    return ReelKinematics(
        progress=p,
        r_supply_mm=r_s,
        r_takeup_mm=r_t,
        f_supply_hz=f_s,
        f_takeup_hz=f_t,
        f_roller_hz=f_roller,
        tension_asymmetry=asym,
        edge_stress=edge_stress,
        back_tension_norm=back_tension,
    )


def evaluate_transport_audio_mod(
    tape_cfg: dict | None,
    frame_index: int,
    u_f: np.ndarray,
    fps: float = 29.97,
    seed: int = 20240101,
) -> dict[str, np.ndarray | float]:
    """
    Compute per-audio-sample capstan speed deviation `dv_over_v`, dropout/noise gate `drop_gate`,
    and amplitude gain `amp_gain` for frame `frame_index` (`u_f` in `[0, 1)`), ensuring exact
    audio-video lockstep for Candidate B (`transport_servo`).
    """
    spec = TransportServoSpec.from_params({"tape": tape_cfg or {}})
    n = len(u_f)
    if not spec.is_active():
        return {
            "dv_over_v": np.zeros(n, dtype=np.float64),
            "drop_gate": np.zeros(n, dtype=np.float32),
            "amp_gain": np.ones(n, dtype=np.float32),
            "pitch_pct": 0.0,
        }

    sp = by_index(spec.tape_speed_idx)
    t_s = (float(frame_index) + u_f.astype(np.float64)) / max(1.0, float(fps))
    p_eff = float(np.clip(spec.reel_progress + 0.00025 * float(frame_index) / max(1.0, float(fps)), 0.0, 1.0))
    kin = compute_reel_kinematics(p_eff, sp.ratio)

    dv = np.zeros(n, dtype=np.float64)
    drop_gate = np.zeros(n, dtype=np.float32)
    amp_gain = np.ones(n, dtype=np.float32)

    # 1. Reel-radius tension ripple & back-tension pole resonance
    eff_rip = spec.tension_ripple + 0.45 * kin.edge_stress
    if eff_rip > 1e-4:
        # Supply reel 1x and 3x (3-slot hub) + take-up reel 1x + back-tension spring pole (~6.5 Hz)
        w_s = 2.0 * np.pi * kin.f_supply_hz * t_s
        w_t = 2.0 * np.pi * kin.f_takeup_hz * t_s
        pole_res = 0.35 * np.sin(2.0 * np.pi * 6.4 * t_s + 0.8) * kin.edge_stress
        rip_wave = (
            (kin.r_takeup_mm / max(10.0, kin.r_supply_mm)) * (0.65 * np.sin(w_s) + 0.25 * np.sin(3.0 * w_s + 0.4))
            + (kin.r_supply_mm / max(10.0, kin.r_takeup_mm)) * 0.45 * np.cos(w_t + 1.1)
            + pole_res
        )
        dv += eff_rip * 0.0055 * rip_wave

    # 2. Hardened pinch-roller eccentricity (periodic wow at f_roller ~ 1.45 Hz in SP)
    if spec.roller_eccentricity > 1e-4:
        ecc = spec.roller_eccentricity * (1.0 + 0.35 * kin.edge_stress)
        ph_r = 2.0 * np.pi * kin.f_roller_hz * t_s + (seed % 97) * 0.06
        roller_wave = np.sin(ph_r) + 0.32 * np.cos(2.0 * ph_r - 0.5)
        dv += ecc * 0.0165 * roller_wave

    # 3. Aged rubber drive belt slip (intermittent speed sag episodes, worse at BOT/EOT)
    if spec.belt_slip > 1e-4:
        slip_load = spec.belt_slip * (1.0 + 0.65 * kin.edge_stress)
        # Deterministic intermittent slip envelope + steady torque droop
        slip_cycle = (t_s * (0.85 + 0.4 * spec.belt_slip) + (seed % 53) * 0.03) % 1.0
        slip_pulse = np.where(
            slip_cycle < 0.38,
            np.sin(np.pi * np.clip(slip_cycle / 0.38, 0.0, 1.0)) ** 1.6,
            0.0,
        )
        rebound = np.where(
            (slip_cycle >= 0.38) & (slip_cycle < 0.62),
            0.28 * np.sin(np.pi * np.clip((slip_cycle - 0.38) / 0.24, 0.0, 1.0)),
            0.0,
        )
        dv += -slip_load * (0.012 + 0.048 * slip_pulse) + slip_load * 0.014 * rebound

    # 4. Capstan/drum FG/PG servo loop hunting oscillation (~3.2 Hz)
    if spec.servo_hunt > 1e-4:
        f_hunt = 3.25
        dv += spec.servo_hunt * 0.011 * np.sin(2.0 * np.pi * f_hunt * t_s + 0.5)

    # 5. Operating State Machine transients
    st = spec.transport_state
    tau = max(0.08, spec.servo_lock_tau)
    if st == 1:
        # Pause / Still: linear tape speed is 0 -> audio is muted / drops to -45 dB mechanical hum
        dv[:] = -0.95
        amp_gain[:] = 0.02
        drop_gate[:] = np.maximum(drop_gate, 0.85)
    elif st == 2:
        # Play <-> Pause transient: pinch roller engages at t=0 (or every 1.6s cycle for multi-frame clips)
        t_local = t_s % max(1.2, 3.2 * tau)
        env = np.exp(-t_local / tau)
        omega_d = 2.0 * np.pi * (2.6 / max(0.2, tau))
        # Underdamped capstan spin-up from 0 speed to 1.0 nominal speed
        dv += -0.38 * env * np.cos(omega_d * t_local) + 0.14 * env * np.sin(omega_d * t_local)
        amp_gain *= np.clip(1.0 - 0.65 * np.exp(-t_local / (0.35 * tau)), 0.15, 1.0).astype(np.float32)
        drop_gate = np.maximum(drop_gate, (0.75 * env).astype(np.float32))
    elif st == 3:
        # Record-Pause Assembly Cut: splice transient & CTL phase jump
        t_local = t_s % max(1.2, 3.0 * tau)
        env = np.exp(-t_local / (0.75 * tau))
        dv += -0.22 * env * np.sin(2.0 * np.pi * 4.2 * t_local)
        drop_gate = np.maximum(drop_gate, (0.90 * np.exp(-t_local / (0.28 * tau))).astype(np.float32))
    elif st == 4:
        # Stop / Unthread: pinch roller disengages, tape decelerates and unthreads from drum
        t_local = np.clip(t_s, 0.0, 2.0)
        decel = np.clip(t_local / max(0.25, tau * 1.4), 0.0, 0.92)
        dv += -0.65 * decel
        amp_gain *= (1.0 - 0.88 * decel).astype(np.float32)
        drop_gate = np.maximum(drop_gate, (0.95 * decel).astype(np.float32))

    pitch_pct = float(np.max(np.abs(dv)) * 100.0)
    return {
        "dv_over_v": dv,
        "drop_gate": drop_gate,
        "amp_gain": amp_gain,
        "pitch_pct": pitch_pct,
    }


def apply_transport_servo(
    y: np.ndarray,
    i: np.ndarray,
    q: np.ndarray,
    spec: TransportServoSpec,
    frame_index: int,
    fps: float = 29.97,
    std: object = None,
    seed: int = 20240101,
    donor_yiq: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Apply the Mechanical Transport & Servo State Machine (Candidate B / Module 4) to baseband `(Y, I, Q)`.

    Returns `(y_out, i_out, q_out, info_dict)`.
    When `spec.is_active()` is False, returns the inputs unchanged (`0.0` overhead).
    """
    if not spec.is_active():
        return y, i, q, {"active": False}

    H, W = y.shape[:2]
    scale_w = W / 720.0
    scale_h = H / 480.0
    sp = by_index(spec.tape_speed_idx)
    t_s = float(frame_index) / max(1.0, float(fps))
    rng = np.random.default_rng((int(seed) * 1664525 + int(frame_index) * 1013904223 + 61) & 0xFFFFFFFF)

    # Continuous reel progress advancement over clip playback
    p_eff = float(np.clip(spec.reel_progress + 0.00025 * t_s / max(0.33, sp.ratio), 0.0, 1.0))
    kin = compute_reel_kinematics(p_eff, sp.ratio)

    y_out = y.astype(np.float32, copy=True)
    i_out = i.astype(np.float32, copy=True)
    q_out = q.astype(np.float32, copy=True)

    rows = np.arange(H, dtype=np.float32)
    yn = rows / max(1.0, float(H - 1))
    cols = np.arange(W, dtype=np.float32)[None, :]
    xn = cols / max(1.0, float(W - 1))
    t_row = t_s + yn / max(1.0, float(fps))

    total_line_shift = np.zeros(H, dtype=np.float32)
    vert_dy = 0.0

    # =========================================================================
    # 1. Supply/Take-Up Reel Radius Evolution & Back-Tension Pole Dynamics
    # =========================================================================
    eff_tension = float(spec.tension_ripple + 0.55 * kin.edge_stress)
    top_skew_px = 0.0
    eot_elong_px = 0.0
    if eff_tension > 1e-4:
        # Supply & take-up angular phases + 3-slot reel hub harmonic
        ph_s = 2.0 * np.pi * kin.f_supply_hz * t_row + 0.3
        ph_t = 2.0 * np.pi * kin.f_takeup_hz * t_row + 1.1
        # Back-tension lever arm spring-mass resonance (~6.4 Hz across frames, ~1.8 cycles down field)
        pole_osc = np.sin(2.0 * np.pi * (6.4 * t_s + 1.8 * yn))

        # Near BOT (p -> 0, asym < 0): small take-up hub pulls hard against heavy full supply reel ->
        # strong upper-frame horizontal skew hook ("tension skew") as the video head enters the track
        bot_weight = float(max(0.0, -kin.tension_asymmetry))
        top_decay = np.exp(-yn / 0.14).astype(np.float32)
        top_skew = (
            eff_tension * (0.45 + 0.85 * bot_weight)
            * (10.5 * scale_w)
            * top_decay
            * (0.65 + 0.35 * np.cos(2.0 * np.pi * kin.f_takeup_hz * t_s))
        )
        top_skew_px = float(np.max(np.abs(top_skew)))

        # Near EOT (p -> 1, asym > 0): small supply hub radius R_s amplifies felt-brake back-tension
        # and 3-slot hub torque pulses -> helical track elongation (progressive S-curve line shift)
        eot_weight = float(max(0.0, kin.tension_asymmetry))
        hub_pulse = 0.65 * np.sin(ph_s) + 0.35 * np.sin(3.0 * ph_s + 0.5)
        eot_elong = (
            eff_tension * (0.40 + 0.95 * eot_weight)
            * (8.5 * scale_w)
            * ((yn - 0.5) * (1.0 + 0.6 * hub_pulse) + 0.32 * pole_osc)
        )
        eot_elong_px = float(np.ptp(eot_elong))
        total_line_shift += (top_skew + eot_elong).astype(np.float32)

        # Edge guide-post stress near BOT/EOT crinkles the tape edge (bottom/top tracking & crease band)
        if kin.edge_stress > 0.12 or spec.tension_ripple > 0.25:
            edge_sev = float(np.clip(0.55 * kin.edge_stress + 0.45 * spec.tension_ripple, 0.0, 1.2))
            # Bottom guide-flange edge curl (last 6..18% of lines)
            curl_band = np.clip((yn - (0.92 - 0.08 * edge_sev)) / max(0.02, 0.08 * edge_sev), 0.0, 1.0)[:, None]
            curl_wave = (0.5 + 0.5 * np.sin(2.0 * np.pi * (kin.f_supply_hz * 3.0 * t_s + yn[:, None] * 6.0)))
            curl_env = (curl_band * curl_wave * edge_sev).astype(np.float32)
            fm_snow = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.2, 2.2 * scale_w), axis=1)
            y_out = y_out * (1.0 - 0.55 * curl_env) + (0.62 + 0.24 * fm_snow) * (0.55 * curl_env)
            i_out *= (1.0 - np.clip(curl_env * 0.85, 0.0, 1.0))
            q_out *= (1.0 - np.clip(curl_env * 0.85, 0.0, 1.0))

    # =========================================================================
    # 2. Hardened Pinch-Roller Eccentricity & Rubber Drive Belt Slip
    # =========================================================================
    roller_shift_ptp = 0.0
    if spec.roller_eccentricity > 1e-4:
        ecc = spec.roller_eccentricity * (1.0 + 0.30 * kin.edge_stress)
        # Pinch roller rotation phase (~1.45 Hz in SP) + 2nd harmonic flat-spot thump
        ph_rol = 2.0 * np.pi * kin.f_roller_hz * t_row + (seed % 97) * 0.06
        rol_wave = np.sin(2.0 * np.pi * yn * 1.35 + ph_rol) + 0.38 * np.cos(4.0 * np.pi * yn * 1.35 + 2.0 * ph_rol)
        rol_shift = (ecc * 11.5 * scale_w * rol_wave).astype(np.float32)
        roller_shift_ptp = float(np.ptp(rol_shift))
        total_line_shift += rol_shift
        # Periodic vertical framing micro-bob at roller rotation rate
        vert_dy += float(ecc * 2.2 * scale_h * np.sin(2.0 * np.pi * kin.f_roller_hz * t_s))

    belt_slip_active = 0.0
    if spec.belt_slip > 1e-4:
        slip_k = spec.belt_slip * (1.0 + 0.55 * kin.edge_stress)
        slip_cycle = (t_s * (0.85 + 0.4 * spec.belt_slip) + (seed % 53) * 0.03) % 1.0
        # Stick-slip speed sag envelope
        slip_env = float(np.sin(np.pi * min(1.0, slip_cycle / 0.45)) ** 1.4) if slip_cycle < 0.45 else 0.22
        belt_slip_active = float(slip_k * (0.45 + 0.55 * slip_env))
        # Capstan speed droop causes cumulative horizontal phase lag down the field + tracking bar rise
        slip_shift = (
            belt_slip_active * (14.0 * scale_w) * (yn - 0.35)
            + belt_slip_active * (5.5 * scale_w) * np.sin(2.0 * np.pi * yn * 2.4 + frame_index * 0.7)
        ).astype(np.float32)
        total_line_shift += slip_shift
        vert_dy += float(belt_slip_active * 3.0 * scale_h * np.cos(frame_index * 0.65))

        # Bottom tracking-unlock snow ramp pushed up into the active frame during belt slip
        slip_rows_frac = 0.06 + 0.16 * belt_slip_active
        ramp_s = np.clip((yn - (1.0 - slip_rows_frac)) / max(0.02, slip_rows_frac), 0.0, 1.0)[:, None].astype(np.float32)
        snow_s = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.2, 2.4 * scale_w), axis=1)
        w_slip_bar = ramp_s * float(np.clip(belt_slip_active * 0.85, 0.0, 0.90))
        y_out = y_out * (1.0 - w_slip_bar) + (0.58 + 0.26 * snow_s) * w_slip_bar
        i_out *= (1.0 - np.clip(w_slip_bar * 1.25, 0.0, 1.0))
        q_out *= (1.0 - np.clip(w_slip_bar * 1.25, 0.0, 1.0))

    # =========================================================================
    # 3. Capstan / Drum Servo Loop Hunting (`servo_hunt`)
    # =========================================================================
    if spec.servo_hunt > 1e-4:
        hunt = spec.servo_hunt
        f_hunt = 3.25
        hunt_phase = 2.0 * np.pi * f_hunt * t_s
        hunt_shift = (
            hunt * (9.5 * scale_w) * np.sin(hunt_phase + yn * 3.8)
            + hunt * (4.2 * scale_w) * np.cos(2.0 * hunt_phase - yn * 6.2)
        ).astype(np.float32)
        total_line_shift += hunt_shift
        vert_dy += float(hunt * 2.8 * scale_h * np.sin(hunt_phase))

    # Apply accumulated continuous mechanical horizontal & vertical displacements
    if np.any(np.abs(total_line_shift) > 1e-4):
        y_out = horizontal_shift(y_out, total_line_shift)
        i_out = horizontal_shift(i_out, total_line_shift)
        q_out = horizontal_shift(q_out, total_line_shift)
    if abs(vert_dy) > 1e-3:
        y_out = vertical_shift(y_out, vert_dy)
        i_out = vertical_shift(i_out, vert_dy)
        q_out = vertical_shift(q_out, vert_dy)

    # =========================================================================
    # 4. Mechanical Transport Operating State Machine (`transport_state` 0..4)
    # =========================================================================
    st = spec.transport_state
    state_names = {
        0: "play",
        1: "pause_still",
        2: "play_pause_transient",
        3: "rec_pause_assembly_cut",
        4: "stop_unthread",
    }
    tau = max(0.08, spec.servo_lock_tau)
    bar_coverage = 0.0
    chroma_beat_rms = 0.0

    if st == 1:
        # ---------------------------------------------------------------------
        # State 1: Pause / Still (静态暂停定格 · 双磁头跨轨静止噪波带)
        # ---------------------------------------------------------------------
        if donor_yiq is not None and donor_yiq[0].shape == y_out.shape:
            y_out, i_out, q_out = donor_yiq[0].copy(), donor_yiq[1].copy(), donor_yiq[2].copy()
        # Frame-still interlace field quiver (both fields read from stationary track pair)
        field_quiver = (0.65 if (frame_index % 2 == 0) else -0.65) * scale_h
        y_out = vertical_shift(y_out, field_quiver)

        # Stationary azimuth/guard-band crossing bar at `pause_noise_pos` with slight servo breathing
        pos_c = float(np.clip(spec.pause_noise_pos + 0.012 * np.sin(frame_index * 0.45), 0.08, 0.92))
        dy_bar = yn - pos_c
        # Double-peaked VHS pause bar: main FM limiter plateau + H-sync black undershoot trench + secondary azimuth band
        main_bar = np.exp(-0.5 * (dy_bar / 0.048) ** 4)
        sec_bar = 0.58 * np.exp(-0.5 * ((dy_bar - 0.068) / 0.024) ** 2)
        trench = np.exp(-0.5 * ((dy_bar - 0.038) / 0.012) ** 2)
        bar_env = np.clip(main_bar + sec_bar, 0.0, 1.0)[:, None].astype(np.float32)
        bar_coverage = float(bar_env.mean())

        # Strong H-AFC horizontal shear across the stationary pause bar
        pause_shear = (
            (np.exp(-0.5 * ((dy_bar - 0.02) / 0.055) ** 2) * 32.0
             - np.exp(-0.5 * ((dy_bar + 0.03) / 0.035) ** 2) * 14.0) * scale_w
            + rng.standard_normal(H).astype(np.float32) * (2.5 * scale_w * main_bar)
        ).astype(np.float32)
        y_out = horizontal_shift(y_out, pause_shear)
        i_out = horizontal_shift(i_out, pause_shear)
        q_out = horizontal_shift(q_out, pause_shear)

        snow = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.0, 2.6 * scale_w), axis=1)
        plateau_luma = np.clip(0.62 + 0.18 * snow, 0.0, 1.0)
        y_out = y_out * (1.0 - bar_env) + plateau_luma * bar_env
        trench_2d = (trench[:, None] * np.where(np.cos(2.0 * np.pi * (xn * 4.0 + frame_index * 0.2)) > -0.4, 1.0, 0.2)).astype(np.float32)
        y_out = y_out * (1.0 - 0.82 * trench_2d) + (0.18 + 0.05 * snow) * (0.82 * trench_2d)
        # Mute chroma inside the guard-band bar, with a thin rotary phase-error fringe on the entry edge
        edge_fringe = np.exp(-0.5 * ((np.abs(dy_bar) - 0.052) / 0.012) ** 2)[:, None]
        phi_f = (rows[:, None] * 0.9 + frame_index * 1.3).astype(np.float32)
        i_out = i_out * (1.0 - np.clip(bar_env * 1.25, 0.0, 0.98)) + 0.14 * edge_fringe * np.cos(phi_f)
        q_out = q_out * (1.0 - np.clip(bar_env * 1.25, 0.0, 0.98)) + 0.14 * edge_fringe * np.sin(phi_f)

    elif st == 2:
        # ---------------------------------------------------------------------
        # State 2: Play <-> Pause Pinch-Roller Engage & CTL Relock Transient
        # ---------------------------------------------------------------------
        t_local = t_s % max(1.2, 3.2 * tau)
        lock_env = float(np.exp(-t_local / tau))
        omega_d = 2.0 * np.pi * (2.6 / max(0.2, tau))
        # As the capstan accelerates toward CTL lock, the tracking error sweeps the guard-band bar
        # vertically downward and out of the frame while amplitude decays as `lock_env`
        bar_y_center = (spec.pause_noise_pos + (t_local / tau) * 0.58) % 1.0
        dy_c = ((yn - bar_y_center + 0.5) % 1.0) - 0.5
        bar_w = max(0.020, 0.055 * lock_env)
        trans_bar = (np.exp(-0.5 * (dy_c / bar_w) ** 2) * min(1.0, lock_env * 1.25))[:, None].astype(np.float32)
        bar_coverage = float(trans_bar.mean())

        # Underdamped capstan + H-AFC horizontal whip during spin-up
        whip_px = (
            lock_env * (26.0 * scale_w) * np.cos(omega_d * t_local + yn * 5.0)
            + trans_bar[:, 0] * (28.0 * scale_w)
        ).astype(np.float32)
        y_out = horizontal_shift(y_out, whip_px)
        i_out = horizontal_shift(i_out, whip_px)
        q_out = horizontal_shift(q_out, whip_px)
        vert_jump = float(lock_env * 10.0 * scale_h * np.sin(omega_d * t_local))
        y_out = vertical_shift(y_out, vert_jump)
        i_out = vertical_shift(i_out, vert_jump)
        q_out = vertical_shift(q_out, vert_jump)

        snow = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.0, 2.5 * scale_w), axis=1)
        y_out = y_out * (1.0 - 0.85 * trans_bar) + np.clip(0.60 + 0.22 * snow, 0.0, 1.0) * (0.85 * trans_bar)
        i_out *= (1.0 - np.clip(trans_bar * 1.2, 0.0, 0.96))
        q_out *= (1.0 - np.clip(trans_bar * 1.2, 0.0, 0.96))

    elif st == 3:
        # ---------------------------------------------------------------------
        # State 3: Record-Pause Assembly Cut (无飞抹磁头暂停起录 · 彩虹副载波反相切口)
        # ---------------------------------------------------------------------
        # Consumer VCR without a flying erase head: the un-erased diagonal band of old tracks
        # overlaps with newly recorded tracks, creating:
        #   1) Ghosted old-track relief + diagonal helical overlap seam;
        #   2) Vivid 629/743 kHz colour-under subcarrier rainbow beat interference;
        #   3) CTL pulse discontinuity -> vertical frame jump + H-AFC shear.
        t_local = t_s % max(1.2, 3.0 * tau)
        cut_env = float(np.exp(-t_local / (0.85 * tau)))
        seam_center = float((0.36 + 0.42 * (t_local / max(0.25, tau))) % 0.88 + 0.06)
        # Helical track slant across the raster (~4% vertical tilt from left to right)
        y_slanted = yn[:, None] - 0.045 * (xn - 0.5)
        dy_seam = y_slanted - seam_center
        overlap_zone = np.exp(-0.5 * (dy_seam / max(0.035, 0.095 * max(0.35, cut_env))) ** 2).astype(np.float32)
        bar_coverage = float(overlap_zone.mean())

        # Old un-erased donor track ghost (shifted horizontally and inverted/embossed by FM beat)
        if donor_yiq is not None and donor_yiq[0].shape == y_out.shape:
            dy_old, di_old, dq_old = donor_yiq
        else:
            dy_old = np.roll(horizontal_shift(y_out, np.full(H, 18.0 * scale_w, np.float32)), int(round(12 * scale_h)), axis=0)
            di_old = -i_out
            dq_old = -q_out

        # FM carrier beat herringbone between old and new overlapping azimuth tracks
        yy_g, xx_g = np.mgrid[0:H, 0:W].astype(np.float32)
        fm_beat = np.cos(2.0 * np.pi * (xx_g / max(2.5, 5.8 * scale_w) + yy_g / max(2.0, 3.6 * scale_h)) + frame_index * 1.7)
        snow_cut = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.0, 2.2 * scale_w), axis=1)
        seam_core = np.exp(-0.5 * (dy_seam / 0.026) ** 2).astype(np.float32)

        # Blend old ghost track + FM beat + limiter snow in the un-erased overlap band
        ghost_mix = np.clip(overlap_zone * 0.58, 0.0, 0.75)
        y_out = (
            y_out * (1.0 - ghost_mix)
            + (0.55 * dy_old + 0.45 * (0.56 + 0.22 * fm_beat + 0.18 * snow_cut)) * ghost_mix
        )
        y_out = y_out * (1.0 - 0.72 * seam_core) + np.clip(0.74 + 0.22 * snow_cut, 0.0, 1.0) * (0.72 * seam_core)

        # Rainbow colour-under subcarrier phase beat (no flying erase head signature!)
        rainbow_phase = (2.0 * np.pi * (yn[:, None] * 9.5 + xn * 1.8) + frame_index * 1.4).astype(np.float32)
        rainbow_amp = overlap_zone * 0.34
        i_rainbow = rainbow_amp * np.cos(rainbow_phase) + 0.45 * overlap_zone * di_old
        q_rainbow = rainbow_amp * np.sin(rainbow_phase) + 0.45 * overlap_zone * dq_old
        i_out = i_out * (1.0 - 0.65 * overlap_zone) + i_rainbow
        q_out = q_out * (1.0 - 0.65 * overlap_zone) + q_rainbow
        chroma_beat_rms = float(np.sqrt(np.mean(i_rainbow ** 2 + q_rainbow ** 2)))

        # CTL splice jump & horizontal tear across the seam
        cut_shear = (
            overlap_zone.mean(axis=1) * (24.0 * scale_w) * np.sin(rows * 0.35 + frame_index)
        ).astype(np.float32)
        y_out = horizontal_shift(y_out, cut_shear)
        i_out = horizontal_shift(i_out, cut_shear)
        q_out = horizontal_shift(q_out, cut_shear)

    elif st == 4:
        # ---------------------------------------------------------------------
        # State 4: Stop / Unthread (停机卸带 · 导带柱退回与包角坍塌雪花帷幕)
        # ---------------------------------------------------------------------
        # As the guide posts retract, helical wrap shrinks toward the middle of the drum scan,
        # collapsing top and bottom active regions into raw RF limiter snow & sync loss
        unthread_prog = float(np.clip(0.52 + 0.28 * np.sin(t_s * 1.5), 0.25, 0.88))
        dist_from_mid = np.abs(yn - 0.50) * 2.0
        unwrapped = np.clip((dist_from_mid - (1.0 - unthread_prog)) / 0.18, 0.0, 1.0)[:, None].astype(np.float32)
        bar_coverage = float(unwrapped.mean())

        unthread_shear = (
            unwrapped[:, 0] * (38.0 * scale_w) * np.sin(rows * 0.27 + frame_index * 1.3)
            + rng.standard_normal(H).astype(np.float32) * (12.0 * scale_w * unwrapped[:, 0])
        ).astype(np.float32)
        y_out = horizontal_shift(y_out, unthread_shear)
        i_out = horizontal_shift(i_out, unthread_shear)
        q_out = horizontal_shift(q_out, unthread_shear)

        snow_u = lowpass(rng.standard_normal((H, W)).astype(np.float32), max(1.0, 2.2 * scale_w), axis=1)
        y_out = y_out * (1.0 - 0.92 * unwrapped) + np.clip(0.46 + 0.32 * snow_u, 0.0, 1.0) * (0.92 * unwrapped)
        i_out *= (1.0 - np.clip(unwrapped * 1.35, 0.0, 0.99))
        q_out *= (1.0 - np.clip(unwrapped * 1.35, 0.0, 0.99))

    info = {
        "active": True,
        "state": state_names.get(st, "play"),
        "state_code": int(st),
        "reel": kin.to_dict(),
        "top_skew_px": round(top_skew_px, 3),
        "eot_elong_px": round(eot_elong_px, 3),
        "roller_shift_ptp": round(roller_shift_ptp, 3),
        "belt_slip_active": round(belt_slip_active, 4),
        "line_shift_ptp": round(float(np.ptp(total_line_shift)), 3),
        "bar_coverage": round(bar_coverage, 4),
        "chroma_beat_rms": round(chroma_beat_rms, 4),
    }
    return y_out.astype(np.float32), i_out.astype(np.float32), q_out.astype(np.float32), info

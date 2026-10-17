"""
Connector degradation: moisture and corrosion at the physical interface.

A connector is not a pipe. It is a node in the signal chain with an impedance, a contact
resistance and a shield, and when those change the *kind* of damage depends entirely on
what the interface was carrying. That is why this is one module with four very different
behaviours rather than one "damage" control:

  * **HDMI** carries a high-speed differential digital stream. Corrosion raises the contact
    impedance, which for a differential pair means the eye closes and bits flip. Digital
    video has no "slight snow" state -- a bit is right or wrong -- so the artefacts are
    macroblocks, frozen regions and dropouts, and below a threshold the link fails to
    handshake at all and reports no signal.
  * **AV (composite)** carries baseband video with the ground on the shell. A bad shell
    contact is a ground-loop path, which puts mains hum on the signal as a slowly rolling
    bright bar; a raised centre-pin impedance reflects, which puts a delayed copy of the
    picture on top of itself as a ghost.
  * **S-Video** splits luma and chroma onto separate pins. Moisture between pins is a
    leakage path, so chroma bleeds into luma and back: coloured fringes on edges, which
    looks convincingly like a bad tape but originates in the connector.
  * **RF** carries a modulated carrier at tens of MHz. Moisture changes the dielectric
    constant (water is about 80 against air's 1), which shifts the impedance off 75 ohms
    and reflects energy back; the reflection interferes with the forward wave, so the
    degradation depends on electrical length and therefore on frequency. Corrosion adds a
    nonlinear contact that rectifies, generating harmonics which appear as interference on
    *other* channels.
  * **VGA** carries wideband analogue RGBHV and is where high-frequency loss shows first,
    as smeared edges, plus sync degradation because the sync pulses are the lowest-amplitude
    part of the signal.

Deliberately not attributed to specific chemistry. "Rust" and "moisture" are the levels the
user can set; what the model commits to is the *electrical* consequence (impedance shift,
contact resistance, shield loss) and the artefact that follows. Naming an oxide would be a
guess dressed as a finding, and the same picture can come from several causes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConnectorSpec:
    """The electrical envelope of one connector type."""

    kind: str
    nominal_z: float              # characteristic impedance, ohms
    signal: str                   # "digital" | "composite" | "svideo" | "rf" | "rgbhv"
    max_mhz: float                # highest frequency the interface is expected to carry
    label: str = ""
    # how much the impedance shifts at full moisture, in ohms (signed, either way)
    moisture_dz: float = 0.0
    # how steeply attenuation rises with frequency at full corrosion
    corrosion_loss: float = 0.0
    # how nonlinear the contact becomes at full corrosion
    corrosion_nonlinearity: float = 0.0


CONNECTORS: dict[str, ConnectorSpec] = {
    "hdmi": ConnectorSpec(
        kind="hdmi", nominal_z=100.0, signal="digital", max_mhz=3400.0,
        moisture_dz=-25.0, corrosion_loss=0.55, corrosion_nonlinearity=0.0,
        label="HDMI（高速差分数字）"),
    "av": ConnectorSpec(
        kind="av", nominal_z=75.0, signal="composite", max_mhz=6.0,
        moisture_dz=-20.0, corrosion_loss=0.35, corrosion_nonlinearity=0.25,
        label="AV / RCA 复合视频"),
    "svideo": ConnectorSpec(
        kind="svideo", nominal_z=75.0, signal="svideo", max_mhz=8.0,
        moisture_dz=-18.0, corrosion_loss=0.30, corrosion_nonlinearity=0.15,
        label="S-Video（Y/C 分离）"),
    "vga": ConnectorSpec(
        kind="vga", nominal_z=75.0, signal="rgbhv", max_mhz=150.0,
        moisture_dz=-15.0, corrosion_loss=0.60, corrosion_nonlinearity=0.10,
        label="VGA（宽带 RGBHV）"),
    "rf": ConnectorSpec(
        kind="rf", nominal_z=75.0, signal="rf", max_mhz=900.0,
        moisture_dz=45.0, corrosion_loss=0.50, corrosion_nonlinearity=0.60,
        label="RF / 天线同轴"),
}


def reflection_coefficient(z_line: float, z_load: float) -> float:
    """
    Gamma = (Z_L - Z_0) / (Z_L + Z_0), the fraction of the wave sent back.

    The physical quantity behind every ghost, standing wave and frequency-dependent notch
    in this module. Returned as a signed fraction; the energy reflected is its square, and
    that distinction matters when setting levels -- a 0.2 coefficient is only 4% of the
    power, but it is a visible ghost.
    """
    return float((z_load - z_line) / (z_load + z_line))


def impedance(spec: ConnectorSpec, moisture: float, corrosion: float) -> float:
    """
    Effective impedance of the degraded interface.

    Water's dielectric constant is about 80 against air's 1, so its presence in the gap
    raises the capacitance per unit length and pulls the impedance down. Corrosion works the
    other way at a contact -- oxide is a poor conductor, so the effective path resistance
    rises and the junction looks more like an open than a matched load.

    The result is returned in ohms and is *not* clamped to a plausible-looking range: a
    badly degraded 75-ohm junction reaching 40 or 120 ohms is the point, and a mismatch that
    large is what produces the visible reflections.
    """
    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    # Corrosion raises the *junction* resistance.  The coefficient is a fraction of the
    # nominal impedance rather than a multiple of it: at 0.55 the added term was 55 ohms on
    # an HDMI pair, which swamped the moisture term (-25) and made a soaked connector come
    # out at 130 ohms against a 100-ohm nominal -- i.e. the model said moisture *raised* the
    # impedance of a differential pair, which is backwards.  Measured with the corrected
    # coefficient: HDMI at full moisture and no corrosion gives 75 ohms, and full corrosion
    # gives 135, so the two mechanisms now act in the directions their physics says.
    return float(spec.nominal_z + spec.moisture_dz * m
                 + 0.35 * spec.nominal_z * c)


def attenuation_per_mhz(spec: ConnectorSpec, moisture: float, corrosion: float) -> float:
    """
    Fraction of amplitude lost per MHz of bandwidth, at this level of degradation.

    Surface roughness and oxide both make the loss rise with frequency, which is why the
    *first* thing to go is fine detail rather than overall brightness. A small baseline is
    kept even when clean, because a real interface is not lossless.
    """
    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    return float(0.004 + 0.02 * m + spec.corrosion_loss * 0.05 * c)


def nonlinearity(spec: ConnectorSpec, corrosion: float) -> float:
    """
    Strength of the rectifying contact, which is what generates harmonics.

    An oxide layer with moisture in it can behave as a crude diode: it passes current more
    easily one way than the other, so a sine comes out with one side flattened. Flattening a
    sine adds harmonics of it, which is why a corroded RF joint interferes with channels
    other than the one being watched -- the interference is generated at the joint rather
    than picked up from outside.
    """
    c = float(np.clip(corrosion, 0.0, 1.0))
    return float(spec.corrosion_nonlinearity * c)


# ---------------------------------------------------------------------------
# per-signal-domain degradations
# ---------------------------------------------------------------------------
def degrade_digital(rgb: np.ndarray, spec: ConnectorSpec, moisture: float,
                    corrosion: float, frame_index: int, seed: int
                    ) -> tuple[np.ndarray, dict]:
    """
    HDMI: bit errors, and link failure below a threshold.

    Digital video cannot degrade smoothly. A bit is right or it is wrong, so a marginal
    link does not produce a faint copy of the picture -- it produces blocks of wrong data
    until correction fails, and then nothing at all. Modelled as:

      * a *bit error rate* rising with the contact impedance, which corrupts blocks rather
        than pixels, because a serial stream's errors land in a run;
      * whole-frame loss once the rate passes a threshold, because that is what a failed
        handshake looks like -- NO SIGNAL, not a bad picture.
    """
    z = impedance(spec, moisture, corrosion)
    gamma = abs(reflection_coefficient(spec.nominal_z, z))
    # The sensitivity is calibrated so that the ends of the ranges give the ends of the
    # behaviour.  At the first attempt it was 1.6x the reflection coefficient plus 0.25x
    # corrosion, which peaked at a bit error rate of 0.23 -- below the link-failure
    # threshold, so the "HDMI fails outright" behaviour was unreachable no matter what the
    # user set.  A control whose extreme does not reach the state it exists to show is the
    # same defect as a control whose whole range is invisible.
    #
    # Note the two mechanisms can partly *cancel*: moisture pulls a 100-ohm pair down and
    # corrosion pushes it up, so a wet-and-rusted joint can land nearer 100 ohms than a
    # merely wet one.  That is real, and it is why the maximum error appears at one extreme
    # rather than in the middle.
    ber = float(np.clip(gamma * 3.4 + 0.42 * corrosion, 0.0, 1.0))
    info = {"impedance_ohm": round(z, 1), "reflection": round(gamma, 4),
            "bit_error_rate": round(ber, 4), "link": "ok"}

    if ber > 0.55:
        # handshake failure: the link drops rather than degrading
        info["link"] = "no signal"
        return np.zeros_like(rgb), info

    out = np.array(rgb, np.float32, copy=True)
    if ber > 1e-4:
        rng = np.random.default_rng(seed)
        h, w = out.shape[:2]
        # errors arrive in runs, because they come from a serial stream: pick run starts and
        # corrupt a horizontal span from each, in block units like the transport
        block = 16
        n_runs = max(1, int(round(ber * 40)))
        for _ in range(n_runs):
            y = int(rng.integers(0, max(1, h)))
            x = int(rng.integers(0, max(1, w)))
            bh = int(rng.integers(1, 4)) * block
            bw = int(rng.integers(1, 5)) * block
            y1, x1 = min(h, y + bh), min(w, x + bw)
            # a corrupted macroblock: value replaced by a wrong one, which is what an
            # unrecoverable transform coefficient looks like
            out[y:y1, x:x1] = rng.random((y1 - y, x1 - x, out.shape[2])).astype(np.float32)
    return out, info


def degrade_composite(comp: np.ndarray, spec: ConnectorSpec, moisture: float,
                      corrosion: float, ground_loop: float, mains_hz: float,
                      std, frame_index: int) -> tuple[np.ndarray, dict]:
    """
    AV: mains hum from a bad ground, and a ghost from the impedance step.

    Two mechanisms, and they are independent because they come from different parts of the
    connector:

      * the **shell is the ground**. A high-resistance shell contact puts the cable's
        reference at a different potential from the source's, and the difference is
        whatever hum is on the mains -- which appears as a bright bar that rolls up or down
        the picture at the difference between mains and field rate. The roll is the whole
        character of the artefact; a static bar would be a different fault;
      * the **centre pin** carries the signal. A raised impedance at the junction reflects,
        and a reflection is a delayed copy of the picture laid over itself. The delay is
        twice the electrical length to the mismatch, so it lands a short distance to the
        right of every edge -- a ghost, not a blur.
    """
    out = np.array(comp, np.float32, copy=True)
    z = impedance(spec, moisture, corrosion)
    gamma = reflection_coefficient(spec.nominal_z, z)
    info = {"impedance_ohm": round(z, 1), "reflection": round(gamma, 4)}

    # --- ground loop hum: a rolling bar at the mains/field beat
    gnd = float(np.clip(ground_loop + 0.35 * moisture + 0.45 * corrosion, 0.0, 1.0))
    if gnd > 1e-3:
        h, w = out.shape[:2]
        field_hz = float(getattr(std, "field_hz",
                                 getattr(std, "frames_per_second", 50.0)))
        beat = abs(mains_hz - field_hz)
        # the bar's position advances by the beat each frame, which is what makes it roll
        y = np.arange(h, dtype=np.float32)[:, None]
        phase = 2.0 * np.pi * ((y / max(1.0, h)) + (frame_index * beat / max(1e-6, field_hz)))
        bar = np.sin(phase)
        out = out + (0.10 * gnd) * bar.astype(np.float32)
        info["ground_loop"] = round(gnd, 3)
        info["hum_beat_hz"] = round(beat, 2)

    # --- reflection ghost: the picture laid over itself, delayed
    if abs(gamma) > 1e-3:
        delay = int(round(6 + 26 * min(1.0, abs(gamma) * 2.0)))
        if delay > 0:
            ghost = np.roll(out, delay, axis=1)
            ghost[:, :delay] = out[:, :delay]
            out = out + (-gamma * 0.55) * ghost
            info["ghost_delay_px"] = delay

    return out.astype(np.float32), info


def degrade_svideo(y: np.ndarray, c_i: np.ndarray, c_q: np.ndarray,
                   spec: ConnectorSpec, moisture: float, corrosion: float,
                   leakage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    S-Video: chroma leaking into luma and back, between adjacent pins.

    Y and C travel on separate pins a few millimetres apart. Moisture between them is a
    leakage path, so each carries a little of the other. The result is *coloured fringes on
    edges* rather than a colour cast: the luma's sharp transitions appear in the chroma
    channel, and chroma's lower-frequency content appears in luma, so the fringes sit
    exactly on the edges -- which is why this reads as a bad tape when it is a wet connector.
    """
    z = impedance(spec, moisture, corrosion)
    gamma = abs(reflection_coefficient(spec.nominal_z, z))
    leak = float(np.clip(leakage + 0.6 * moisture, 0.0, 1.0))
    info = {"impedance_ohm": round(z, 1), "reflection": round(gamma, 4),
            "leakage": round(leak, 4)}
    if leak <= 1e-4:
        return y, c_i, c_q, info

    from ..core.signal import lowpass
    # luma's edges into chroma: high-pass the luma by subtracting its low-passed self
    lp = lowpass(y, max(0.3, 0.06 * y.shape[1]), axis=1)
    edges = y - lp
    # chroma's content into luma: its own low frequencies
    c_lp_i = lowpass(c_i, max(0.3, 0.02 * y.shape[1]), axis=1)
    c_lp_q = lowpass(c_q, max(0.3, 0.02 * y.shape[1]), axis=1)
    y_out = y + leak * 0.9 * (c_lp_i + c_lp_q)
    i_out = c_i + leak * 0.75 * edges
    q_out = c_q - leak * 0.75 * edges
    return (y_out.astype(np.float32), i_out.astype(np.float32),
            q_out.astype(np.float32), info)


def degrade_rf(comp: np.ndarray, spec: ConnectorSpec, moisture: float, corrosion: float,
               shield: float, frame_index: int, seed: int
               ) -> tuple[np.ndarray, dict]:
    """
    RF: reflection notches, frequency-dependent loss, and rectified interference.

    The interface carries a modulated carrier, so damage does not alter the picture
    directly -- it alters the *channel* the picture travels in, and the tuner then
    demodulates whatever arrives.

      * moisture shifts the impedance, and the reflection interferes with the forward wave.
        Whether that interference adds or cancels depends on the electrical length, so the
        damage is *frequency selective*: it bites at some frequencies and not others rather
        than fading everything;
      * corrosion loss also rises with frequency, so the chroma subcarrier (higher up the
        band than most of the luma energy) loses first -- colour goes before detail;
      * the rectifying contact generates harmonics, which appear as interference. That
        interference is *generated at the joint*, so its amplitude does not depend on what
        the antenna is receiving, and it drifts with temperature and humidity rather than
        frame to frame.
    """
    out = np.array(comp, np.float32, copy=True)
    z = impedance(spec, moisture, corrosion)
    gamma = reflection_coefficient(spec.nominal_z, z)
    info = {"impedance_ohm": round(z, 1), "reflection": round(gamma, 4)}

    h, w = out.shape[:2]
    x = np.arange(w, dtype=np.float32)[None, :]

    # --- frequency-selective ripple: the standing wave across the channel
    if abs(gamma) > 1e-3:
        # ripple period in pixels set by the electrical length; a longer mismatch gives a
        # finer ripple, which is why the visible banding changes as the connector degrades
        period = max(4.0, 90.0 / max(1e-3, abs(gamma) * 6.0))
        ripple = np.cos(2.0 * np.pi * x / period)
        out = out + (0.22 * abs(gamma)) * ripple.astype(np.float32)
        info["ripple_period_px"] = round(float(period), 1)

    # --- frequency-dependent loss: high frequencies first
    loss = attenuation_per_mhz(spec, moisture, corrosion)
    if loss > 1e-4:
        from ..core.signal import lowpass
        # translate the per-MHz figure into a pixel-domain cutoff for the current raster
        bw_mhz = max(0.4, spec.max_mhz * (1.0 - 40.0 * loss))
        px = std_pixels_for_bandwidth_local(bw_mhz, w)
        out = lowpass(out, max(0.35, px), axis=1)
        info["bandwidth_mhz"] = round(float(bw_mhz), 2)

    # --- rectified interference from the nonlinear contact
    nl = nonlinearity(spec, corrosion)
    if nl > 1e-4 and shield < 0.95:
        rng = np.random.default_rng(seed + frame_index // 4)   # drifts slowly, not per frame
        # harmonics land as a coarse pattern that moves with the beat against the line rate
        k = np.sin(2.0 * np.pi * (x / max(3.0, 28.0 / max(0.2, nl)))
                   + frame_index * 0.21)
        out = out + (0.16 * nl * (1.0 - shield)) * k.astype(np.float32)
        out = out + (0.05 * nl) * rng.standard_normal(out.shape).astype(np.float32)
        info["interference"] = round(float(nl * (1.0 - shield)), 4)

    return out.astype(np.float32), info


def std_pixels_for_bandwidth_local(mhz: float, width: int) -> float:
    """
    Pixels per cycle-ish for a bandwidth in MHz at this raster width.

    Kept local rather than imported so this module has no dependency on the standard object
    for its arithmetic; the caller passes whatever width it has.
    """
    # a 6 MHz channel across a 720-wide active line is the reference point
    return max(0.35, (width / 720.0) * (6.0 / max(0.2, mhz)) * 0.5)


def degrade_rf_signal(rgb: np.ndarray, spec: ConnectorSpec, moisture: float, corrosion: float,
                      shield: float, frame_index: int, seed: int
                      ) -> tuple[np.ndarray, dict]:
    """
    Full physical model of an HDMI -> AV -> RF -> HDMI loop with a damp/corroded RF coaxial interface
    (calibrated against `RF转录.mp4`):
      1. Dielectric RC shunt on horizontal sync pulses -> Bistable Horizontal AFC phase lag (`right_HBI_px`
         black blanking border on the right + serpentine tearing at top/bottom).
      2. Delayed Burst-Gate sampling -> Subcarrier PLL phase inversion (Delta_phi ~ 172 deg rotating (-I,-Q)
         into (+I,+Q), turning blue-cyan into blazing red-orange on slipped lines, plus horizontal multi-colour
         unlocked chroma bands that jut into the right HBI margin).
      3. Standing-wave intercarrier harmonic mixing -> 2D diagonal herringbone carrier mesh (period_x=6.40px,
         period_y=10.20px at 722x586).
      4. Moisture relaxation / line-clamp surge -> Crisp horizontal bright streaking bars with AC-coupling
         undershoot trenches that continue across the right-hand black HBI zone as double-rail sync bars.
      5. V-Sync slip top taskbar wrap-around & lower-field partial clamp-collapse noisy dark bands.
    """
    from scipy.ndimage import gaussian_filter
    from ..core.signal import rgb_to_yiq, yiq_to_rgb, lowpass, horizontal_shift

    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    z = impedance(spec, m, c)
    gamma = reflection_coefficient(spec.nominal_z, z)
    info: dict = {"impedance_ohm": round(z, 1), "reflection": round(gamma, 4)}
    if m <= 1e-4 and c <= 1e-4 and shield <= 1e-4:
        return rgb.astype(np.float32), info
    # Clean-connector closed-circuit CATV VHF branch (when moisture <= 0.08 and corrosion <= 0.08, e.g. VHF Ch 6
    # 171.25 MHz coaxial feed into Panasonic NV-J27 tuner):
    # Instead of damp-connector 172 deg colour inversion, apply PAL-D VSB channel shaping, subtle 171.25 MHz
    # closed-circuit harmonic herringbone ripple (`shield`), and slight coaxial impedance reflection.
    if m <= 0.08 and c <= 0.08:
        from ..core.signal import rgb_to_yiq, yiq_to_rgb, lowpass, highpass, horizontal_shift
        h, w = rgb.shape[:2]
        scale_w = w / 720.0
        scale_h = h / 576.0
        rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 131 + 17) & 0xFFFFFFFF)
        y, i, q = rgb_to_yiq(np.clip(rgb, 0.0, 1.0).astype(np.float32))
        # PAL-D VHF Ch 6 (168..176 MHz, f_v = 171.25 MHz, f_a = 177.75 MHz) VSB IF filter & SAW group-delay ringing
        y_lp = lowpass(y, max(0.65, 1.05 * scale_w), axis=1)
        y_hp = highpass(y_lp, max(0.9, 1.65 * scale_w), axis=1)
        y = y_lp + 0.22 * y_hp
        # 75-ohm coaxial reflection micro-ghost (~5 px rightward)
        refl_amt = float(np.clip(0.035 + 0.25 * abs(gamma) + 0.04 * shield, 0.0, 0.18))
        delay_px = max(2, int(round(5.0 * scale_w)))
        ghost = np.roll(y, delay_px, axis=1)
        ghost[:, :delay_px] = y[:, :delay_px]
        y = y + refl_amt * ghost
        # PAL-D colour-under / IF chroma bandwidth & slight rightward group delay (+2 px)
        i = lowpass(i, max(1.1, 2.25 * scale_w), axis=1)
        q = lowpass(q, max(1.3, 2.65 * scale_w), axis=1)
        sh_c = np.full(h, 1.8 * scale_w, dtype=np.float32)
        i = horizontal_shift(i, sh_c)
        q = horizontal_shift(q, sh_c)
        # Subtle VHF 171.25 MHz closed-circuit carrier / 6.5 MHz sound-subcarrier intercarrier beat mesh
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ph = float((frame_index * 1.17) % (2.0 * np.pi))
        px_per = max(2.0, 5.60 * scale_w)
        py_per = max(2.5, 9.40 * scale_h)
        beat = np.cos(2.0 * np.pi * (xx / px_per + yy / py_per) + ph).astype(np.float32)
        beat_amp = float(np.clip(0.012 + 0.045 * shield + 0.08 * m, 0.0, 0.08))
        y = y + beat_amp * beat
        # Subtle tuner IF thermal snow in shadows
        if_noise = lowpass(rng.standard_normal((h, w)).astype(np.float32), max(0.7, 1.25 * scale_w), axis=1)
        sh_w = np.clip(1.15 - 0.65 * y, 0.35, 1.15)
        y = y + (0.008 + 0.025 * m + 0.015 * shield) * if_noise * sh_w
        out = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)
        info.update({"vhf_catv_mode": True, "vision_carrier_mhz": 171.25, "ghost_delay_px": delay_px})
        return out, info

    h, w = rgb.shape[:2]
    scale_w = w / 722.0
    scale_h = h / 586.0
    rng = np.random.default_rng((int(seed) * 2654435761 + int(frame_index) * 131 + 17) & 0xFFFFFFFF)

    # 1. HDMI -> AV composite baseband bandwidth limiting
    y, i, q = rgb_to_yiq(np.clip(rgb, 0.0, 1.0).astype(np.float32))
    y = lowpass(y, max(0.7, 1.35 * scale_w), axis=1)
    i = lowpass(i, max(1.4, 3.20 * scale_w), axis=1)
    q = lowpass(q, max(1.8, 4.20 * scale_w), axis=1)

    rows = np.arange(h, dtype=np.float32)
    yn = rows / max(1.0, h - 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1.0, w - 1.0)

    # 2. Bistable Horizontal Sync Slicer Lag & Subcarrier Burst-Gate Phase Rotation
    eff_m = float(np.clip(m + 0.45 * c, 0.0, 1.0))
    f_mod = int(frame_index) % 4
    slipped_state = (f_mod != 2) if eff_m >= 0.30 else False
    multi_band_state = (f_mod in (0, 3)) and slipped_state

    if slipped_state:
        phi_row = np.full(h, 3.00, dtype=np.float32)
        acc_gain = np.full(h, 1.68, dtype=np.float32)
        diode_i = np.full(h, 0.16 * eff_m, dtype=np.float32)
        diode_q = np.full(h, 0.05 * eff_m, dtype=np.float32)

        top_curve = np.exp(-yn / 0.025) * (65.0 * scale_w)
        notch = np.exp(-0.5 * ((yn - 0.83) / 0.042) ** 2) * (-58.0 * scale_w)
        bot_in = np.clip((yn - 0.88) / 0.04, 0.0, 1.0) * (8.0 * scale_w)
        jitter_r = rng.standard_normal(h).astype(np.float32) * (1.8 * eff_m * scale_w)
        hbi_width_row = (98.0 * eff_m * scale_w) + top_curve + notch + bot_in + jitter_r

        if multi_band_state:
            # Top & bottom multi-colour unlocked chroma bands that jut rightward into the black HBI zone
            band_specs = (
                (0.055, 16.0, 0.02, -0.05, -42.0),
                (0.088, 14.0, -1.55, 0.02, -38.0),
                (0.125, 18.0, 0.15, -0.04, -35.0),
                (0.755, 22.0, 0.02, -0.05, -48.0),
                (0.835, 24.0, 0.05, -0.05, -52.0),
                (0.905, 18.0, 0.08, -0.04, -40.0),
            )
            for by_frac, bh_px, t_phi, d_i, hbi_delta in band_specs:
                by = by_frac * h
                bh = bh_px * scale_h
                b_box = np.clip((bh * 0.5 - np.abs(rows - by)) / max(1.0, 1.8 * scale_h), 0.0, 1.0)
                phi_row = phi_row * (1.0 - b_box) + t_phi * b_box
                diode_i = diode_i * (1.0 - b_box) + d_i * b_box
                hbi_width_row = hbi_width_row + b_box * (hbi_delta * scale_w * eff_m)

        hbi_width_row = np.clip(hbi_width_row, 8.0 * scale_w, 180.0 * scale_w)

        sat_mag = np.hypot(i, q)
        c_gate = np.clip(sat_mag / 0.08, 0.0, 1.0)
        cos_p = (np.cos(phi_row) * acc_gain)[:, None].astype(np.float32)
        sin_p = (np.sin(phi_row) * acc_gain)[:, None].astype(np.float32)
        i_new = i * cos_p - q * sin_p + diode_i[:, None] * c_gate
        q_new = i * sin_p + q * cos_p + diode_q[:, None] * c_gate
        i, q = i_new, q_new
    else:
        phi_row = np.full(h, -0.12, dtype=np.float32)
        cos_p = (np.cos(phi_row) * 1.28)[:, None].astype(np.float32)
        sin_p = (np.sin(phi_row) * 1.28)[:, None].astype(np.float32)
        i, q = i * cos_p - q * sin_p, i * sin_p + q * cos_p
        y = np.clip(y * 1.12 + 0.04, 0.0, 1.0)

        tb_h = max(6, int(round(28 * scale_h)))
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

        hbi_width_row = (16.0 + 6.0 * eff_m) * scale_w + rng.standard_normal(h).astype(np.float32) * (1.8 * scale_w)
        hbi_width_row = np.clip(hbi_width_row, 4.0 * scale_w, 45.0 * scale_w)

    out = np.clip(yiq_to_rgb(y, i, q), 0.0, 1.0).astype(np.float32)

    # 3. Top VBI black border + burst-unlock rainbow fringe & lower clamp-collapse band
    if slipped_state:
        top_black = (yn < 0.032)[:, None].astype(np.float32)
        out = out * (1.0 - top_black[..., None])
        top_fringe = np.exp(-0.5 * ((yn - 0.045) / 0.012) ** 2)[:, None] * eff_m
        out[..., 0] = out[..., 0] * (1.0 - 0.55 * top_fringe) + 0.35 * top_fringe
        out[..., 1] = np.clip(out[..., 1] + 0.52 * top_fringe, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.28 * top_fringe * (0.5 + 0.5 * np.cos(xx / max(1.0, 4.5 * scale_w))), 0.0, 1.0)

        if not multi_band_state:
            dark_band = (np.clip((yn - 0.735) / 0.018, 0.0, 1.0) * np.clip((0.945 - yn) / 0.018, 0.0, 1.0))[:, None] * eff_m
            band_speckle = rng.standard_normal((h, w)).astype(np.float32)
            band_speckle = lowpass(band_speckle, max(0.7, 1.5 * scale_w), axis=1)
            out[..., 0] = np.clip(out[..., 0] * (1.0 - 0.25 * dark_band) + 0.14 * dark_band * band_speckle, 0.0, 1.0)
            out[..., 1] = np.clip(out[..., 1] * (1.0 - 0.38 * dark_band) + 0.18 * dark_band * band_speckle, 0.0, 1.0)
            out[..., 2] = np.clip(out[..., 2] + 0.10 * dark_band * np.abs(band_speckle), 0.0, 1.0)
    else:
        for sy_f, sh_f in ((0.66, 0.045), (0.80, 0.055)):
            sb = np.exp(-0.5 * ((yn - sy_f) / sh_f) ** 2)[:, None] * eff_m
            sn = lowpass(rng.standard_normal((h, w)).astype(np.float32), max(0.8, 1.8 * scale_w), axis=1)
            out = np.clip(out + (0.14 * sb * sn)[..., None], 0.0, 1.0)

    # 4. Crisp 2D Diagonal Standing-Wave Carrier Mesh (FFT-calibrated: period_x=6.40px, period_y=10.20px)
    ph1 = float((frame_index * 1.41) % (2.0 * np.pi))
    ph2 = float((frame_index * 0.79 + 1.1) % (2.0 * np.pi))
    px_per = max(2.0, 6.40 * scale_w)
    py_per = max(2.5, 10.20 * scale_h)
    warble = 0.35 * np.sin(yy / max(2.0, 14.0 * scale_h) + frame_index * 0.5)
    wave1 = np.cos(2.0 * np.pi * (xx / px_per + yy / py_per) + warble + ph1)
    wave2 = np.cos(2.0 * np.pi * (xx / max(2.0, 6.65 * scale_w) + yy / max(2.5, 9.60 * scale_h)) - 0.5 * warble + ph2)
    mesh = (0.70 * wave1 + 0.35 * wave2).astype(np.float32)

    mesh_strength = (0.18 + 0.14 * eff_m + 0.06 * c) * (1.0 - 0.35 * float(np.clip(shield, 0.0, 1.0)))
    if slipped_state:
        out[..., 0] = np.clip(out[..., 0] + 0.32 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.98 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.32 * mesh_strength * mesh, 0.0, 1.0)
    else:
        out[..., 0] = np.clip(out[..., 0] + 0.52 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.62 * mesh_strength * mesh, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.42 * mesh_strength * mesh, 0.0, 1.0)

    # 5. Per-line RF timebase jitter + Leftward H-Sync Shift + Right-Side Black HBI Zone
    rf_line_jit = rng.standard_normal(h).astype(np.float32) * (1.4 * eff_m * scale_w)
    shift_left = -(hbi_width_row - 16.0 * scale_w) * 0.30 + rf_line_jit
    for ch in range(3):
        out[..., ch] = horizontal_shift(out[..., ch], shift_left)

    active_end_x = (w - hbi_width_row)[:, None]
    in_active = np.clip((active_end_x - xx) / max(1.0, 1.8 * scale_w), 0.0, 1.0).astype(np.float32)
    in_hbi = 1.0 - in_active
    out = out * in_active[..., None]

    # 6. Horizontal Line-Clamp Surge Streaking Bars + AC-Coupling Undershoot Trenches + HBI Double-Rail Sync Bars
    if slipped_state and (not multi_band_state) and eff_m >= 0.30:
        for sy_frac, bar_h_px, g_lift, b_lift in ((0.515, 24.0, 0.66, 0.20), (0.692, 25.0, 0.70, 0.28)):
            sy_c = sy_frac * h
            bh = max(4.0, bar_h_px * scale_h)
            dy = rows - sy_c

            trench_up = np.exp(-0.5 * ((dy + 0.72 * bh) / max(1.2, 0.25 * bh)) ** 2)[:, None] * in_active * eff_m
            trench_dn = np.exp(-0.5 * ((dy - 0.72 * bh) / max(1.2, 0.25 * bh)) ** 2)[:, None] * in_active * eff_m
            trench = np.clip(trench_up + trench_dn, 0.0, 1.0)
            out[..., 0] *= (1.0 - 0.22 * trench)
            out[..., 1] *= (1.0 - 0.78 * trench)
            out[..., 2] *= (1.0 - 0.85 * trench)

            bar_box = np.clip((bh * 0.5 - np.abs(dy)) / max(1.0, 2.0 * scale_h), 0.0, 1.0)[:, None].astype(np.float32)
            bar_x_env = (0.82 + 0.24 * np.sin(xn * np.pi)) * in_active
            w_bar = bar_box * bar_x_env * float(np.clip(eff_m * 1.15, 0.0, 1.0))

            bar_rgb = np.stack([
                np.full((h, w), 0.98, np.float32),
                np.clip(g_lift + 0.20 * mesh, 0.0, 1.0),
                np.clip(b_lift + 0.12 * mesh, 0.0, 1.0),
            ], axis=-1)
            out = out * (1.0 - 0.85 * w_bar[..., None]) + bar_rgb * (0.85 * w_bar[..., None])

            if sy_frac > 0.60:
                slip_line = np.exp(-0.5 * ((dy + 0.48 * bh) / max(0.8, 1.2 * scale_h)) ** 2)[:, None] * in_active * eff_m
                out[..., 0] *= (1.0 - 0.75 * slip_line)
                out[..., 1] = np.clip(out[..., 1] * (1.0 - 0.40 * slip_line) + 0.25 * slip_line * np.cos(xx / max(1.0, 3.2 * scale_w)), 0.0, 1.0)

            rail_top = np.clip((2.8 * scale_h - np.abs(dy + 0.32 * bh)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            rail_bot = np.clip((2.8 * scale_h - np.abs(dy - 0.32 * bh)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            rail_mid = np.clip((2.2 * scale_h - np.abs(dy)) / max(0.8, 1.2 * scale_h), 0.0, 1.0)[:, None]
            hbi_dx = xx - active_end_x
            sync_gap = 1.0 - 0.78 * np.clip((8.0 * scale_w - np.abs(hbi_dx - 24.0 * scale_w)) / max(1.0, 2.0 * scale_w), 0.0, 1.0)
            rail_val = np.clip((0.75 * rail_top + 0.70 * rail_bot + 0.30 * rail_mid * np.cos(xx / max(1.0, 3.5 * scale_w))) * sync_gap, 0.0, 0.90)
            hbi_add = (rail_val * in_hbi * float(np.clip(eff_m * 1.2, 0.0, 1.0)))[..., None]
            out = np.clip(out + hbi_add * np.array([0.92, 0.95, 0.90], dtype=np.float32), 0.0, 1.0)

    info.update({
        "slipped_state": slipped_state,
        "multi_band_state": multi_band_state,
        "right_hbi_px": round(float(np.median(hbi_width_row)), 1),
        "ripple_period_px": round(float(px_per), 2),
    })
    return np.clip(out, 0.0, 1.0).astype(np.float32), info

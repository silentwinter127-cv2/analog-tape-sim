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

"""
The operator: the hands holding the camera, the zoom motor, and the way a person uses a camcorder.

The note's seventh dimension, and it calls it the most easily overlooked and the most real: "设备不是自动
运行的" -- the equipment does not run itself. Everything else in this model is a property of a device; this
is a property of the person holding it, and it is the layer that makes footage read as *a recording of
something* rather than as a picture with faults applied.

**Handheld shake is not noise.** The obvious implementation is a random offset per frame, and it is wrong:
human tremor is narrowband and rhythmic, and the rhythm is what makes a shake read as a hand rather than
as a glitch. A hand moves as a sum of a few quasi-periodic components, each from a different part of the
body:

    0.4-1.5 Hz   postural sway, from the whole body's slow balance corrections -- the largest
    1.5-3 Hz     walking and breathing, when the operator is moving
    4-8 Hz       arm and shoulder muscle activity
    8-12 Hz      physiological tremor, a property of the motor system and always present

Their amplitudes fall with frequency -- tremor is roughly a hundredth of sway -- and that *ratio* is
what the eye reads. Getting the rhythm right matters more than the amplitude, which is why the components
are oscillators with independent phases rather than a filtered random walk.

A real hand also does something a sum of sines cannot: it is occasionally jostled. A step, a door, a
person bumping an elbow. Those are episodes, so they are events, like the tape's dropouts and the deck's
stiction -- the model already has the pattern for a fault that is discrete rather than continuous.

**The zoom motor is not linear.** A consumer camcorder's zoom is a small DC motor driving a gear train:
it takes time to overcome stiction, then sweeps, then the operator releases the rocker and the mechanism
stops abruptly rather than decelerating. The note names exactly that shape. It is worth representing
because a linear zoom is a thing no camcorder does and the eye notices immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShakeComponent:
    """One quasi-periodic component of a hand's movement."""

    name: str
    hz: float
    #: how far this component moves the framing, as a fraction of the frame width
    amplitude: float
    #: how the component's phase drifts over time, as a fraction of its own frequency.
    #: A pure sine is unnatural; a small drift is what makes two seconds apart look different.
    drift: float = 0.0


#: A hand, as the four things a hand actually does.  The amplitudes fall with frequency, and the
#: ratio between them is what the eye reads as a human hand rather than as a fault.
#:
#: **The rates are calibrated against footage, not chosen.**  A 103 s handheld clip's own offset autocorrelates at
#: **48 frames (1.60 s) at r = 0.671**; at the original rates (0.9 / 2.1 / 5.5 / 9.5 Hz) this profile autocorrelated at
#: **25 frames (0.83 s) at r = 0.15**, i.e. at the footage's *second harmonic* rather than its fundamental.  Scaling
#: every rate by **0.60** brings the composite to **42-45 frames (1.40-1.50 s)**.
#:
#: **Its periodicity strength is a known limit and is deliberately not forced.**  Across fifteen combinations of rate
#: and drift the autocorrelation peak stays between **0.12 and 0.22** against the footage's **0.671**, and lowering the
#: drift does not raise it -- it makes a drift component dominant and pushes the period out to **4-9 s**.  So this
#: represents an *unsteady hand*, whose motion is random at a characteristic rate, and the footage shows an *operator
#: sweeping deliberately*, which is a different kind of motion.  Adding a second sway term to reach r = 0.67 would give
#: the model two sources for one behaviour, so the rates are the part that is corrected and the strength is recorded.
HAND: tuple[ShakeComponent, ...] = (
    ShakeComponent("postural sway", 0.54, 0.0100, 0.20),
    ShakeComponent("walking / breathing", 1.26, 0.0042, 0.25),
    ShakeComponent("arm and shoulder", 3.30, 0.0016, 0.30),
    ShakeComponent("physiological tremor", 5.70, 0.0008, 0.35),
)


@dataclass
class BumpEvent:
    """A jostle: a step, a door, an elbow. Discrete, like a dropout."""

    start_frame: int
    frames: int
    amplitude: float
    field: int = 0

    def to_dict(self) -> dict:
        return {"kind": "bump", "start_frame": self.start_frame, "frames": self.frames,
                "amplitude_px": round(float(self.amplitude), 2)}


def plan_bumps(rate_hz: float, amount: float, frames: int, fps: float,
               seed: int) -> list[BumpEvent]:
    """
    Jostles as a Poisson process, the same shape as the tape's dropouts and the deck's stiction.

    A hand is not jostled on a schedule, so the arrivals are random with a rate the user sets. The
    duration is short -- a jostle is a jerk and a recovery, not a sustained movement -- and the
    amplitude is drawn from a distribution with a tail, because most bumps are small and a few are not.
    """
    if rate_hz <= 1e-6 or amount <= 1e-6 or frames <= 1:
        return []
    rng = np.random.default_rng(seed)
    out: list[BumpEvent] = []
    t, total = 0.0, frames / max(1e-6, fps)
    guard = 0
    while t < total and guard < 4096:
        guard += 1
        t += float(rng.exponential(1.0 / max(1e-6, rate_hz)))
        if t >= total:
            break
        # long enough to contain both the jerk and its recovery.  At the old floor of two frames the
        # whole event was the initial step: the decay that makes it read as a jostle never appeared, and
        # a check measuring the largest excursion found it at a frame where no bump was planned.
        dur = float(np.clip(rng.exponential(0.26), max(4.0 / fps, 0.24), 0.9))
        # a tail: most jostles are small, a few are large
        amp = float(amount * 0.02 * (0.35 + rng.exponential(1.0)))
        out.append(BumpEvent(int(round(t * fps)), max(1, int(round(dur * fps))),
                             amp * 100.0))
    return out


def shake_offset(components: tuple[ShakeComponent, ...], intensity: float,
                 frame_index: int, fps: float, seed: int,
                 bumps: list[BumpEvent] | None = None,
                 width: int = 720) -> tuple[float, float]:
    """
    Where the hand has moved the camera on this frame, in pixels, as (dx, dy).

    Deterministic in the frame number: the phases come from the seed, so the same clip has the same
    hand every time, and the offset can be computed for any frame without rendering the ones before it.
    That property is the same one the environment and the keyframes have, and it is what allows the
    renderer to skip cached frames and still produce the same picture.

    The two axes are given different phase sets, because a hand does not move diagonally: the vertical
    component is dominated by breathing and the horizontal by weight shift, and making them share phases
    produces a shake along one line.
    """
    if intensity <= 1e-6:
        return 0.0, 0.0
    t = frame_index / max(1e-6, fps)
    dx = dy = 0.0
    for k, c in enumerate(components):
        # independent phase per component and per axis, drawn once from the seed
        rng_x = np.random.default_rng(seed * 131 + k * 17 + 1)
        rng_y = np.random.default_rng(seed * 131 + k * 17 + 2)
        px, py = float(rng_x.random()) * 2 * np.pi, float(rng_y.random()) * 2 * np.pi
        # the drift makes the phase wander slowly, so the pattern is quasi-periodic rather than
        # exactly periodic -- two seconds apart do not look identical
        drift_x = 1.0 + c.drift * np.sin(2 * np.pi * 0.07 * t + px)
        drift_y = 1.0 + c.drift * np.sin(2 * np.pi * 0.061 * t + py)
        amp = c.amplitude * intensity * width
        dx += amp * np.sin(2 * np.pi * c.hz * drift_x * t + px)
        dy += amp * np.cos(2 * np.pi * c.hz * drift_y * t + py)

    if bumps:
        for ev in bumps:
            if ev.start_frame <= frame_index < ev.start_frame + ev.frames:
                # a jerk and a recovery: fast in, exponential out, which is what a bump feels like
                u = (frame_index - ev.start_frame) / max(1, ev.frames)
                shape = (1.0 - np.exp(-14.0 * u)) * np.exp(-3.2 * u)
                rng_b = np.random.default_rng(seed * 977 + ev.start_frame)
                ang = float(rng_b.random()) * 2 * np.pi
                dx += ev.amplitude * shape * np.cos(ang)
                dy += ev.amplitude * shape * np.sin(ang)
    return float(dx), float(dy)


def zoom_curve(u: float, kind: str = "motor") -> float:
    """
    The zoom position for a rocker held from 0 to 1 of its travel.

    A camcorder's zoom is a small motor and a gear train, and the note names the shape: slow at first
    while it overcomes stiction, fast through the middle, and an abrupt stop when the rocker is
    released. That is not an ease-in-out; the end is a stop, not a deceleration, because the motor is
    simply switched off and the mechanism has enough friction to hold position.
    """
    u = float(np.clip(u, 0.0, 1.0))
    if kind == "linear":
        return u
    if kind == "cinema":            # a servo zoom on a professional lens: smooth both ends
        return 0.5 - 0.5 * np.cos(np.pi * u)
    # motor: stiction to overcome, then a sweep, then switched off.  The cube gives the slow start and
    # the fast middle; the linear term keeps it from being flat at the beginning.
    return float(np.clip(0.35 * u ** 3 + 0.65 * u, 0.0, 1.0))

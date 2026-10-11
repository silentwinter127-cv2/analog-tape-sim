"""
The camera's automatic control loops: auto-exposure, auto white balance, and auto focus.

This is the note's sixth dimension, and it names the reason it matters: "摄影机本身一直在'自动犯错'"
-- the camera is always making small automatic mistakes, and that is where a consumer camcorder's
character comes from. A handycam has no operator setting exposure; it has a loop that measures the
picture and moves a control, with a time constant, and gets it slightly wrong.

**These are the only mechanisms in the model that need genuine state.** Everything added so far is a
function of the frame number, which is what makes it safe to render frames in any order. A control loop
cannot be, because it reads the picture: exposure at frame N depends on the brightness the loop has
seen, which depends on frames 0..N-1. The renderer pumps frames from zero and caches each stage
independently, so the loop is walked once per frame in order and its value is cached with the frame --
correct by construction, given that discipline. The invariant is stated here because it is the one thing
that could break it.

**Each loop is an exponential approach with a time constant, and the correction must be immediate.**
`state += (target - state) * alpha` is the obvious form and is wrong in a way that looks right: the
state approaches the target but never reaches it, so a loop that has had ten seconds to settle is still
carrying 1/e of its initial error. A camera's loop settles. The form used here is
`state = target + (state - target) * exp(-dt / tau)`, which puts the state *at* the target when the
measurement stops changing and decays the previous error rather than accumulating a new one.

Where each loop's character comes from:

  * **exposure** reacts to mean scene luminance, with a dead band so it does not hunt on grain, and it is
    weighted towards the centre of the frame because a real metering pattern is not an average;
  * **white balance** reacts to the scene's average chroma, and it is *the* source of the colour drift
    that reads as a 1990s home recording: the loop nulls the lamp, so a room lit by a warm bulb goes
    neutral over a second or two and swings back when the camera turns to a window;
  * **focus** hunts. It cannot measure focus directly -- it can only measure *contrast* and try a
    direction -- so it steps, watches whether contrast improved, and reverses when it did not. That is
    why a handycam hunts on a low-contrast scene and locks on a detailed one, and it is the behaviour
    the note describes as "清晰 → 模糊 → 寻找焦点 → 清晰".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Rec.601 luma coefficients, in one place.  Three copies of these existed in this project before,
#: which is how "luminance" comes to mean different things in different files.
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an RGB frame.  Accepts a 2-D plane unchanged, so callers may pass either."""
    a = np.asarray(rgb, np.float32)
    return a if a.ndim == 2 else a @ LUMA


#: the metering pattern: centre-weighted, as a real camcorder's average metering is.
def centre_weights(height: int, width: int, falloff: float = 0.45) -> np.ndarray:
    """
    A centre-weighted metering map normalised to a mean of one.

    A real average-metering pattern is not a uniform average; it weights the middle of the frame
    because that is usually the subject. The consequence is visible and wanted: a bright sky at the
    top of the frame does not drag the exposure down as far as a flat average would, which is why
    camcorder footage so often has a blown sky.
    """
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    r = np.sqrt((xx * 1.0) ** 2 + (yy * 1.0) ** 2)
    w = np.exp(-falloff * r * r).astype(np.float32)
    return w / max(1e-6, float(w.mean()))


@dataclass
class ExposureLoop:
    """
    Auto-exposure: track mean scene luminance towards a target, with a dead band.

    The dead band exists because a real loop does not chase grain. Without it the exposure would
    oscillate on noise, which reads as flicker rather than as a camera settling.
    """

    target: float = 0.46
    tau_s: float = 1.1
    dead_band: float = 0.012
    #: how far the loop may push the exposure control, in stops
    limit_stops: float = 2.0
    state: float = 0.0
    #: seconds this camera has already been adapting to this scene before the clip starts.  A clip
    #: that begins in a lit room should not open with the exposure swinging in, and a camera carried
    #: there has been adapting for longer than the clip.
    presettle_s: float = 0.0
    _presettled: bool = False

    def update(self, rgb: np.ndarray, dt: float, weights: np.ndarray) -> float:
        """
        Measure the frame, move one step, return the correction in stops.

        Takes the frame rather than a luma plane: a parameter that means something narrower than its
        name invites the caller to pass what it has, and the first call did exactly that.
        """
        if rgb.size == 0:
            return self.state
        luma = luminance(rgb)
        if luma.shape != weights.shape:
            # a caller may hand a differently sized frame than the weights were built for
            weights = centre_weights(luma.shape[0], luma.shape[1])
        measure = float((luma * weights).sum() / max(1e-6, weights.sum()))
        # the error this loop wants to remove, in stops
        target_stops = float(np.clip(np.log2(max(1e-4, self.target) / max(1e-4, measure)),
                                     -self.limit_stops, self.limit_stops))
        if abs(target_stops) < self.dead_band:
            target_stops = 0.0
        if self.tau_s <= 1e-6:
            self.state = target_stops
            return self.state
        if not self._presettled:
            # apply the adaptation that happened before the clip as an analytic decay of the
            # initial error, which is what the loop would have done had it been running.  There is
            # no first-frame special case: the loop always takes tau.
            k0 = float(np.exp(-max(0.0, self.presettle_s) / self.tau_s))
            self.state = target_stops + (self.state - target_stops) * k0
            self._presettled = True
        # decay the *previous* error and take up the new target, so the loop settles rather than
        # asymptotically approaching and never arriving
        k = float(np.exp(-dt / self.tau_s))
        self.state = target_stops + (self.state - target_stops) * k
        return float(np.clip(self.state, -self.limit_stops, self.limit_stops))


@dataclass
class WhiteBalanceLoop:
    """
    Auto white balance: null the scene's average chroma, which is what makes indoor footage drift.

    The loop sees the average colour of the frame and gains the channels to cancel it. That is correct
    behaviour and it is also why a home recording looks the way it does: a room lit by a warm bulb goes
    neutral over a second or two, and swings back when the camera turns towards a window. A static
    colour cast cannot express that, which is the argument for a loop rather than a slider.

    The gains are bounded, because a real AWB has a limited correction range -- beyond it the camera
    gives up and keeps the cast, which is why footage under strongly coloured light stays coloured.
    """

    tau_s: float = 1.6
    #: correction range per channel, as a fraction
    limit: float = 0.42
    #: how much of the frame's colour the loop believes; a low value makes it lazy
    gain: float = 0.85
    state: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    #: seconds already spent adapting, as for the exposure loop
    presettle_s: float = 0.0
    _presettled: bool = False

    def update(self, rgb: np.ndarray, dt: float) -> np.ndarray:
        """Measure the frame's average colour, move one step, return per-channel gains."""
        if rgb.size == 0:
            return self.state
        mean = rgb.reshape(-1, 3).mean(axis=0)
        grey = float(mean.mean())
        if grey < 1e-4:
            return self.state
        # the correction that would make this frame neutral, bounded by the loop's range
        want = np.clip(grey / np.maximum(1e-4, mean), 1.0 - self.limit, 1.0 + self.limit)
        want = 1.0 + self.gain * (want - 1.0)
        if self.tau_s <= 1e-6:
            self.state = want.astype(np.float32)
            return self.state
        if not self._presettled:
            k0 = float(np.exp(-max(0.0, self.presettle_s) / self.tau_s))
            self.state = (want + (self.state - want) * k0).astype(np.float32)
            self._presettled = True
        k = float(np.exp(-dt / self.tau_s))
        # same form as the exposure loop: the state sits at the target and the previous error decays
        self.state = (want + (self.state - want) * k).astype(np.float32)
        return self.state


def contrast_measure(rgb: np.ndarray) -> float:
    """
    How much detail the frame has, as a focus loop would measure it.

    A camera cannot measure focus, only contrast, and it measures it on the *luminance* after the
    lens -- so a horizontal gradient energy is the right quantity and it is also what a real
    focus-detection circuit computes.
    """
    if rgb.size == 0:
        return 0.0
    y = luminance(rgb)
    gx = np.abs(np.diff(y, axis=1)).mean()
    gy = np.abs(np.diff(y, axis=0)).mean()
    return float(gx + gy)


@dataclass
class FocusLoop:
    """
    Auto focus: step the focus position, watch contrast, reverse when it gets worse.

    A hill-climb with hysteresis, which is what a contrast-detection AF actually is. The reason this
    matters for a 1990s look is its *failures*: on a low-contrast or moving scene the loop never finds
    a peak and hunts, and a camcorder hunting is one of the most recognisable things in home footage.
    A model that only applies a fixed defocus cannot produce that at all.
    """

    tau_s: float = 0.9
    #: how far each step moves the focus position
    step: float = 0.055
    #: how much better a new position must be before the direction is kept
    hysteresis: float = 0.06
    #: the scene must have at least this much contrast for the loop to trust a measurement
    min_contrast: float = 1.5e-3
    #: below this, the loop gives up and drifts, which is what a handycam does in the dark
    give_up_contrast: float = 5.0e-4
    position: float = 0.0
    direction: float = 1.0
    best: float = 0.0
    _last_contrast: float = 0.0
    _settle: float = 0.0

    def update(self, rgb: np.ndarray, dt: float) -> float:
        """
        Move one step and return the focus position, 0 sharp to 1 fully defocused.

        The position is what the camera's focus motor is at; how much blur that produces is the lens's
        business, so the caller maps it. Returning the position rather than a blur amount keeps the
        loop independent of the optics.
        """
        c = contrast_measure(rgb)
        if c < self.give_up_contrast:
            # nowhere to focus: a real loop drifts and hunts rather than locking, which is exactly
            # the behaviour worth reproducing
            self.position = float(np.clip(self.position + self.direction * self.step * 0.35, 0.0, 1.0))
            if self.position <= 0.0 or self.position >= 1.0:
                self.direction = -self.direction
            return self.position

        self._settle += dt
        if self._settle < self.tau_s:
            return self.position
        self._settle = 0.0

        if c < self.min_contrast:
            # low contrast: keep hunting at a reduced step rather than trusting the measurement
            self.position = float(np.clip(self.position + self.direction * self.step * 0.5, 0.0, 1.0))
            if self.position <= 0.0 or self.position >= 1.0:
                self.direction = -self.direction
            return self.position

        if c > self.best * (1.0 + self.hysteresis):
            # an improvement: keep going this way, and remember the new best
            self.best = c
            self.position = float(np.clip(self.position + self.direction * self.step, 0.0, 1.0))
            if self.position <= 0.0 or self.position >= 1.0:
                self.direction = -self.direction
        else:
            # no improvement: reverse and take a smaller step, as a real AF narrows its search
            self.direction = -self.direction
            self.step = float(max(0.012, self.step * 0.72))
            self.best = max(self.best * 0.85, c)
            self.position = float(np.clip(self.position + self.direction * self.step, 0.0, 1.0))
        return self.position

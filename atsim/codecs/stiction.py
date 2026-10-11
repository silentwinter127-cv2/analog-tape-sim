"""
Tape-deck moisture: stiction, which is an intermittent fault rather than a smooth one.

Measured on a real capture, the per-row displacement spread flips between roughly 0.1 and 3
pixels from one sample to the next -- a factor of fourteen within seconds, and uncorrelated
with picture content. That is the shape of an *intermittent* fault, and it is the one kind
the model has no representation for: every existing mechanism scales smoothly with a
parameter, so none of them can produce it.

The physics is stiction. A tape's binder is hygroscopic, and a tape stored damp carries a
thin water film. Against the capstan and the pinch roller that film is adhesive: the tape
sticks, the transport keeps pulling, and the tension rises until the adhesion gives way and
the tape jerks forward. Then it sticks again. The cycle repeats at whatever rate the local
humidity and the tape's condition set, which is why:

  * the fault is **episodic**. Between events the tape behaves normally; the transport is
    not uniformly worse, it is mostly fine and occasionally not;
  * each event is a **grab and a release**, so the displacement ramps during the grab and
    snaps back on release -- a sawtooth, not a step, and not a sinusoid;
  * it depends on **position along the tape**, because the water film is not uniform and
    the binder's condition varies down the length;
  * it is **temperature- and humidity-dependent** in the long run, which is why the same
    cassette can be fine in the morning and unusable in the evening. That is modelled by the
    rate parameter rather than by anything per-frame.

Represented as a list of events on the timeline, like dropouts and creases, rather than as a
per-frame field. An event-driven fault cannot be produced by a per-frame field: smoothing a
field removes exactly the discontinuities that make it look like a fault.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StictionEvent:
    """
    One grab-and-release cycle.

    `frames` is how long the tape is stuck, `magnitude` how far the misread rows are
    displaced at peak, and `severity` how much signal is lost along with the displacement
    (a tape held against a stalled capstan also loses contact).
    """

    start_frame: int
    frames: int
    position_s: float
    duration_s: float
    magnitude: float
    severity: float
    rise: float          # fraction of the event spent building up before release
    field: int

    def to_dict(self) -> dict:
        return {
            "kind": "stiction",
            "start_frame": self.start_frame,
            "frames": self.frames,
            "start_s": round(self.position_s, 4),
            "duration_s": round(self.duration_s, 4),
            "magnitude_px": round(float(self.magnitude), 2),
            "severity": round(float(self.severity), 3),
            "rise": round(float(self.rise), 3),
            "field": self.field,
        }


def plan_stiction(rate_hz: float, amount: float, frames: int, fps: float,
                  seed: int, stretch: float = 0.0
                  ) -> list[StictionEvent]:
    """
    Generate the events for `frames` frames of tape.

    A Poisson arrival process, as with dropouts: stiction events are independent of each
    other and their rate is set by how damp the tape is, which is what `rate_hz` means.
    Durations are drawn from an exponential distribution, because most grabs are brief and
    a few are long -- the same long-tailed shape dropouts have.

    `stretch` biases the rate along the tape, so a cassette that is worse near the end (damp
    at the hub, or simply more worn where it has been played most) can be expressed. It is a
    linear factor on the rate across the length, not a per-frame amplitude, because the fault
    is in the events and not in the signal.
    """
    if rate_hz <= 1e-6 or amount <= 1e-6 or frames <= 1:
        return []
    rng = np.random.default_rng(seed)
    mean_gap_s = 1.0 / max(1e-6, rate_hz)
    events: list[StictionEvent] = []
    t = 0.0
    total_s = frames / max(1e-6, fps)
    guard = 0
    while t < total_s and guard < 4096:
        guard += 1
        # position-dependent rate: the stretch factor applies across the tape's length
        frac = min(1.0, t / max(1e-6, total_s))
        local = rate_hz * (1.0 + stretch * (2.0 * frac - 1.0))
        gap = float(rng.exponential(1.0 / max(1e-6, local)))
        t += gap
        if t >= total_s:
            break
        dur_s = float(rng.exponential(mean_gap_s * 0.35))
        dur_s = float(np.clip(dur_s, 1.0 / fps, max(1.5 / fps, 0.35)))
        n_frames = max(1, int(round(dur_s * fps)))
        # severity rises with how much stiction there is, and each event varies around it
        sev = float(np.clip(amount * rng.uniform(0.55, 1.25), 0.0, 1.5))
        events.append(StictionEvent(
            start_frame=int(round(t * fps)),
            frames=n_frames,
            position_s=t,
            duration_s=dur_s,
            magnitude=float(2.0 + 26.0 * sev),   # pixels at peak displacement
            severity=sev,
            rise=float(rng.uniform(0.45, 0.8)),
            field=int(rng.integers(0, 2)),
        ))
    return events


def stiction_displacement(events: list[StictionEvent], frame_index: int, height: int
                          ) -> tuple[np.ndarray, float]:
    """
    The per-row displacement and the signal-loss weight for one frame.

    The shape is the point. During a grab the tape is held and the transport pulls, so the
    displacement *ramps*; on release it snaps back. That gives a sawtooth in time and a
    smooth ramp in space, which is what a grab looks like -- not a step, which would be a
    splice, and not a sine, which would be a flutter.

    Returns (displacement in pixels per row, loss weight 0..1).
    """
    disp = np.zeros(height, np.float32)
    loss = 0.0
    for ev in events:
        if not (ev.start_frame <= frame_index < ev.start_frame + ev.frames):
            continue
        k = (frame_index - ev.start_frame) / max(1, ev.frames - 1)   # 0..1 through the event
        # ramp up over `rise` of the event, then release: the sawtooth
        if k <= ev.rise:
            frac = k / max(1e-6, ev.rise)
        else:
            frac = 1.0 - (k - ev.rise) / max(1e-6, 1.0 - ev.rise)
        amp = ev.magnitude * frac
        # in space: the grab affects the field it lands on, strongest where the head is
        rows = np.arange(height, dtype=np.float32)
        if ev.field == 0:
            lo = height * 0.45
            w = np.clip((rows - lo) / max(1.0, height * 0.55), 0.0, 1.0)
        else:
            lo = height * 0.05
            w = np.clip((rows - lo) / max(1.0, height * 0.5), 0.0, 1.0)
        disp = disp + amp * w.astype(np.float32)
        loss = max(loss, ev.severity * frac)
    return disp, float(np.clip(loss, 0.0, 1.0))

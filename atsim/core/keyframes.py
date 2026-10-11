"""
Keyframes: the machine's life as a timeline rather than as a single set of numbers.

A real recording is a slice of a machine's history. The camcorder was cold when the tape started,
warmed over twenty minutes, was carried indoors where the air was damp, and had its tape transport
knocked while walking. None of that is one number, and representing it as a per-clip constant loses
exactly the thing the note calls the difference between a fault and a life cycle.

A keyframe track is a list of (time, value) points for one parameter, evaluated at any frame. Three
properties follow from that shape:

**Deterministic and order-independent.** The value at a frame is a function of the track and the frame
number, not of which frames were rendered before it. The renderer pumps frames from zero but also skips
cached ones and can be asked for frame 300 first, so anything that accumulated state would make the
same project render differently depending on the request order -- and this project's premise is that
it does not.

**Editable and legible.** A recording session is a short list of moments, and stating them is clearer
than any single "age" or "temperature" slider could be. It is also the only way to express a *change*
mid-clip at all.

**Interpolation is part of the physics, not a detail.** A power supply warming up approaches equilibrium
exponentially; a tape being rewound moves linearly; a mode switch holds until the next moment. The
curve is therefore a property of the point that *starts* the segment, because it describes how the
quantity gets from there to the next one.

Curves are chosen from what the quantity does rather than for looks:

    linear  -- a mechanism driven at a constant rate (winding, a zoom motor, tape position)
    exp     -- an approach to equilibrium (temperature, a servo's settling, a capacitor charging)
    hold    -- a discrete state that does not change until told (a mode switch, a power cycle)
    smooth  -- no physical rate is known; a cosine ease, used only where nothing better applies

Nothing here is on by default. A project with no keyframes renders exactly as it did before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: the interpolation curves, in the order the UI should offer them
CURVES = ("linear", "exp", "hold", "smooth")


@dataclass
class Keyframe:
    """One moment: a time in seconds, a value, and how to leave this point."""

    t: float
    value: float
    curve: str = "linear"

    def to_dict(self) -> dict:
        return {"t": round(float(self.t), 4), "value": float(self.value), "curve": self.curve}


@dataclass
class Track:
    """
    One parameter over time.

    `layer` and `key` name what is being driven. `absolute` decides whether the values are the
    parameter's own units or a relative offset from whatever the project already had, because both
    readings are wanted: "the head is at 0.6 by now" and "the temperature is 5 degrees above where it
    started" are different statements and neither should be expressed as the other.
    """

    layer: str
    key: str
    points: list[Keyframe] = field(default_factory=list)
    absolute: bool = False

    def value_at(self, t: float) -> float | None:
        """
        The track's value at time `t`, or None if the track is empty.

        Before the first point the value is held at the first one, and after the last at the last,
        which is what a parameter does when nothing is driving it. That choice matters for a track
        that starts at 10 s: a clip's first ten seconds are the machine as it already was, not the
        value of the first keyframe applied retroactively.
        """
        pts = sorted(self.points, key=lambda p: p.t)
        if not pts:
            return None
        if t <= pts[0].t:
            return float(pts[0].value)
        if t >= pts[-1].t:
            return float(pts[-1].value)
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a.t <= t <= b.t:
                span = max(1e-9, b.t - a.t)
                u = (t - a.t) / span
                return _interp(a.value, b.value, u, a.curve)
        return float(pts[-1].value)

    def to_dict(self) -> dict:
        return {"layer": self.layer, "key": self.key, "absolute": bool(self.absolute),
                "points": [p.to_dict() for p in sorted(self.points, key=lambda p: p.t)]}

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        pts = [Keyframe(float(p.get("t", 0.0)), float(p.get("value", 0.0)),
                        str(p.get("curve", "linear"))) for p in (d.get("points") or [])]
        return cls(layer=str(d.get("layer", "")), key=str(d.get("key", "")),
                   points=pts, absolute=bool(d.get("absolute", False)))


def _interp(a: float, b: float, u: float, curve: str) -> float:
    """Interpolate one segment. `u` is the fraction of the way from a to b."""
    u = float(np.clip(u, 0.0, 1.0))
    if curve == "hold":
        # a discrete state does not change until the next moment
        return float(a)
    if curve == "smooth":
        # cosine ease: used only where no physical rate is known
        u = 0.5 - 0.5 * np.cos(np.pi * u)
        return float(a + (b - a) * u)
    if curve == "exp":
        # An approach to equilibrium *across the segment the user drew*: the segment's length is
        # what the user is specifying, so the time constant is a fraction of it.  tau = 0.30 puts
        # the far end at 1 - exp(-1/0.30) ~= 96% of the way, which is what "it has warmed up by
        # then" means, and leaves the last few per cent to the following segment -- which is what
        # happens physically, and is why a warm-up is not expressible as a single moment.
        #
        # An earlier value of 0.217 was chosen so the curve *arrived* at u = 1, which made a
        # 420-second warm-up complete in about 90 and then hold: plausible-looking output from a
        # wrong constant, measured as 0.937 of the way after 95% of the nominal time.
        tau = max(1e-9, 0.30)
        return float(b + (a - b) * np.exp(-u / tau))
    # linear: a mechanism driven at a constant rate
    return float(a + (b - a) * u)


# ---------------------------------------------------------------------------
# machine-level helpers: the tracks a recording session actually wants
# ---------------------------------------------------------------------------
def warmup_track(tau_s: float = 420.0, span: float = 1.0,
                 switch_on_t: float = 0.0) -> Track:
    """
    One warm-up, as an absolute track on the environment's temperature factor.

    Provided as a constructor because it is the most common thing a session wants and because
    writing it by hand as keyframes is easy to get wrong: the curve belongs to the point that
    *starts* the segment, so a two-point track needs `exp` on the first point, not the second.
    """
    return Track(layer="environment", key="warmth", absolute=True,
                 points=[Keyframe(switch_on_t, 0.0, "exp"),
                         Keyframe(switch_on_t + 4.6 * tau_s, span, "hold")])


def session(
    switch_on_t: float = 0.0,
    warmup_s: float = 420.0,
    power_cycle_t: float | None = None,
    heater_failure_t: float | None = None,
) -> list[Track]:
    """
    A plausible recording session, as tracks.

    Named moments rather than a single age: switched on, warmed, power-cycled (which puts the machine
    back to cold), and a failure of the heater or the supply partway through. Each is optional, and the
    result is a list of tracks that can be edited like any other.
    """
    tracks = [warmup_track(tau_s=warmup_s, span=1.0, switch_on_t=switch_on_t)]
    warmth = tracks[0]
    if power_cycle_t is not None:
        # down for a moment, so the machine is cold again on restart and warms from there
        warmth.points = [p for p in warmth.points if p.t < power_cycle_t]
        warmth.points.append(Keyframe(power_cycle_t, 1.0, "hold"))
        warmth.points.append(Keyframe(power_cycle_t + 2.0, 0.0, "exp"))
        warmth.points.append(Keyframe(power_cycle_t + 2.0 + 4.6 * warmup_s, 1.0, "hold"))
    if heater_failure_t is not None:
        # the machine stops reaching temperature, so it drifts cold and stays there
        warmth.points.append(Keyframe(heater_failure_t, 0.55, "exp"))
        warmth.points.append(Keyframe(heater_failure_t + 4.6 * 600.0, 0.92, "hold"))
    return tracks


@dataclass
class KeyframeSet:
    """A project's tracks, and the only thing the renderer needs to know about them."""

    tracks: list[Track] = field(default_factory=list)

    def value_at(self, layer: str, key: str, t: float) -> float | None:
        for tr in self.tracks:
            if tr.layer == layer and tr.key == key:
                return tr.value_at(t)
        return None

    def apply(self, params: dict[str, dict[str, float]], t: float,
              ranges: dict[str, tuple[float, float]] | None = None
              ) -> tuple[dict[str, dict[str, float]], list[dict]]:
        """
        Evaluate every track at time `t` and write the results into a copy of `params`.

        `absolute` tracks replace the value; relative ones add to it.  For a relative track on a
        parameter whose nominal is zero, the value is read as a fraction of the parameter's range --
        the same rule the per-unit spread and the environment shifts use, and for the same reason: a
        parameter that ships at zero would otherwise be undrivable by a track.
        """
        if not self.tracks:
            return params, []
        out = {layer: dict(vals) for layer, vals in params.items()}
        record: list[dict] = []
        for tr in self.tracks:
            v = tr.value_at(t)
            if v is None or tr.layer not in out or tr.key not in out[tr.layer]:
                continue
            before = float(out[tr.layer][tr.key])
            if tr.absolute:
                after = float(v)
            elif abs(before) > 1e-9:
                after = before * (1.0 + float(v))
            else:
                lo, hi = (ranges or {}).get(tr.key, (0.0, 1.0))
                after = (hi - lo) * float(v)
            out[tr.layer][tr.key] = after
            record.append({"layer": tr.layer, "key": tr.key, "t": round(float(t), 3),
                           "before": before, "after": after,
                           "absolute": bool(tr.absolute), "curve": (sorted(
                               tr.points, key=lambda p: p.t)[0].curve if tr.points else "-")})
        return out, record

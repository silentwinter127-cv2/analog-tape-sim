"""
Make the environment keyframe-driven: warmth and ageing become editable timelines.

The first version computed warmth from a closed-form exponential of elapsed time. That was
deterministic and cache-safe -- which is why it was chosen -- but it could only express *one* warm-up
from switch-on. A real recording session has a life: switched on cold, warmed, carried indoors, had its
tape transport knocked, power-cycled in the middle. None of that is one exponential.

So warmth becomes a value the keyframe set supplies, and this module turns it into parameter shifts. The
properties that made the closed-form version good are kept, because a keyframe track is also a pure
function of the frame number: no accumulated state, nothing to reset when the chain is rebuilt, and the
same project renders the same picture whatever order frames are asked for.

**Why the effects are listed here rather than as keyframe tracks.** A keyframe drives a scalar; the
environment's effect is a *vector* of parameter shifts whose direction comes from physics -- a cold
crystal pulls the colour decoder, a sagging rail loses gain and headroom. Writing each of those as its
own track would let a user produce a cold machine with warm colour, which is not a machine. So the
timeline drives one number, `warmth`, and the physics decides what warmth does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Environment:
    """
    The machine's condition, as levels the timeline supplies.

    `warmth` is 1.0 when the machine is at operating temperature and 0.0 when it is cold; `age` is
    capacitor ageing, 0 as new and 1 at end of life; `humidity` is the damp it sits in, 0 dry and 1
    saturated. All three are normally driven by keyframe tracks, and the defaults correspond to a
    machine that is already warm, not aged, and dry -- so an environment that is not asked for moves
    nothing.
    """

    warmth: float = 1.0
    age: float = 0.0
    humidity: float = 0.0
    #: how far the cold end of the range reaches, as a multiplier on the effects below.  1.0 means
    #: a fully cold machine shows the full listed shift; lower values are a machine that barely
    #: notices, which is a real difference between one model and another.
    cold_sensitivity: float = 1.0
    #: the same for ageing
    age_sensitivity: float = 1.0


#: what a fully cold machine does, as (parameter, relative shift at warmth = 0, why).
#:
#: The signs come from the physics rather than from taste.  A crystal oscillator's frequency falls
#: slightly as it warms toward its turnover point and the decoder's phase trim is uncompensated at
#: switch-on; a cold CCD has more dark current; a cold belt and grease make the servo work harder.
TEMPERATURE_EFFECTS: tuple[tuple[str, float, str], ...] = (
    ("color_cast", -0.055,
     "a cold crystal and uncompensated resistors pull the colour decoder; it nulls as the machine warms"),
    ("white_balance_temp", 0.045, "the same drift reaches the white-balance reference"),
    ("exposure", -0.035, "the AE reference drifts cold"),
    ("sensor_noise", 0.22, "a cold CCD has higher dark current, so it is noisiest when first switched on"),
    ("frequency_response", -0.06, "the replay equaliser drifts until the amplifier reaches temperature"),
    ("transport_jitter", 0.18, "servo gain, belt stiffness and grease viscosity all change with temperature"),
    ("defocus", 0.06, "lens and tube geometry change slightly as they warm"),
)

#: what a sagging rail does, as (parameter, relative shift at age = 1, why).
#:
#: Electrolytic capacitance falls with age, so the rail droops under load.  A low rail loses gain and
#: headroom and moves the clamp reference.
SUPPLY_EFFECTS: tuple[tuple[str, float, str], ...] = (
    ("output_bias", -0.30, "a low rail moves the clamp reference, so the black level drops"),
    ("output_gain", -0.22, "less headroom, so less gain before clipping"),
    ("ccd_gain", -0.18, "sensor drive falls with the rail"),
    ("brightness", -0.10, "CRT cut-off follows the rail"),
    ("contrast", -0.08, "CRT drive follows the rail"),
    ("dynamic_range", -0.15, "a low rail clips earlier"),
)


def environment_shifts(env: Environment) -> dict[str, tuple[float, str]]:
    """
    Every parameter this machine's condition moves, as key -> (relative shift, why).

    Returned rather than applied, so a caller can attribute a change to the environment and the shifts
    can be inspected without rendering -- the same reason the per-unit record is returned.
    """
    cold = (1.0 - float(np.clip(env.warmth, 0.0, 1.0))) * float(env.cold_sensitivity)
    aged = float(np.clip(env.age, 0.0, 1.0)) * float(env.age_sensitivity)
    out: dict[str, tuple[float, str]] = {}
    for key, amount, why in TEMPERATURE_EFFECTS:
        v = amount * cold
        if abs(v) > 1e-9:
            out[key] = (v, why)
    for key, amount, why in SUPPLY_EFFECTS:
        v = amount * aged
        if abs(v) > 1e-9:
            prev = out.get(key)
            if prev is not None:
                v += prev[0]
                why = prev[1] + "; " + why
            out[key] = (v, why)
    return out


def apply_environment(params: dict[str, dict[str, float]], env: Environment,
                      ranges: dict[str, tuple[float, float]],
                      layer_of) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """
    Apply the environment's shifts to resolved parameters, and return what was moved.

    Multiplicative where the nominal is non-zero, and a fraction of the parameter's *range* where it is
    zero -- the same rule the per-unit spread and the keyframe tracks use, and for the same reason: a
    parameter that ships at zero would otherwise be immune to a cold machine, and those are exactly the
    drift parameters where the warm-up shows.
    """
    out = {layer: dict(vals) for layer, vals in params.items()}
    record: list[dict] = []
    for key, (shift, why) in sorted(environment_shifts(env).items()):
        layer = layer_of(key)
        if layer is None or layer not in out or key not in out[layer]:
            continue
        nominal = float(out[layer][key])
        if abs(nominal) > 1e-9:
            new = nominal * (1.0 + shift)
        else:
            lo, hi = ranges.get(key, (0.0, 1.0))
            new = (hi - lo) * shift
        out[layer][key] = new
        record.append({"layer": layer, "key": key, "nominal": nominal, "value": new,
                       "shift": round(shift, 5), "why": why})
    return out, record

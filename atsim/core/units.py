"""
Device DNA: one machine, not a model number.

The note's first and most important dimension. Today this model has one "VHS" and one "camcorder" that
every unit shares, so a Panasonic from a dry cupboard and one that spent ten years in a car boot
produce the same picture. Real machines of one model differ because components have tolerances: two
CCDs from the same batch differ in dark current, two lens assemblies differ in centring, two servo
boards differ in gain.

Two properties make this representable without touching the architecture:

**It is deterministic per unit.** The same identity always yields the same machine, so a fault stays
reproducible and a tape can be revisited. This is the same requirement as the seeded defect timeline,
and it is what makes the dimension useful rather than decorative.

**It is a property of the parameter.** Each parameter carries a `unit_spread`, the fraction by which a
particular unit may sit from nominal. Where a service tolerance is known the spread comes from it;
where none is known the spread stays zero, because an invented spread is worse than none -- it looks
like provenance and stops the next person looking.

A machine is described by a short identity string, and the spread is drawn from a generator seeded
with it. Each device class has its own list of what actually varies with unit, taken from what a
service manual would call an alignment tolerance rather than from what looks good.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def unit_seed(identity: str) -> int:
    """A stable seed from an identity string, so one machine is one machine forever."""
    h = hashlib.blake2s(str(identity).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


@dataclass
class DeviceUnit:
    """
    One physical machine.

    `identity` is the whole specification: a serial number, or a model plus a serial, or any string
    the user wants. Nothing else is needed to reproduce it.
    """

    identity: str = ""
    label: str = ""
    #: per-device-class deviation of one unit from nominal, in standard deviations of the
    #: declared spread.  Filled by `draw()`.
    deviations: dict[str, float] = field(default_factory=dict)

    @property
    def serial(self) -> str:
        return self.identity or "(nominal unit)"

    def draw(self, spreads: dict[str, tuple[str, float]]) -> "DeviceUnit":
        """
        Draw this unit's deviations from the declared spreads.

        A standard normal per parameter, clipped at 2 sigma: an alignment tolerance is a bound, and a
        tail beyond twice it is a broken machine rather than a unit of the model. Re-drawing for the
        same identity gives the same numbers because the generator is seeded by it.

        **An empty identity draws nothing.** A serial number is a statement that this is a particular
        machine; with no such statement the model must be the nominal one it was before this dimension
        existed. Without this guard the anonymous unit is just another machine, and every existing
        preset, capture and calibration shifts by a few per cent -- which is the kind of change that
        invalidates a model quietly.
        """
        self.deviations = {}
        if not str(self.identity).strip():
            return self
        rng = np.random.default_rng(unit_seed(self.identity))
        for key, (_layer, spread) in sorted(spreads.items()):
            if spread <= 0.0:
                continue
            self.deviations[key] = float(np.clip(rng.standard_normal(), -2.0, 2.0))
        return self

    def describe(self, spreads: dict[str, tuple[str, float]]) -> list[tuple[str, str, float]]:
        """
        What this unit is, as (layer, parameter, deviation in the parameter's own units).

        Returned rather than printed so a caller can show it, log it, or put it in a defect report --
        the point of an individual machine is that its deviations can be named.
        """
        out = []
        for key, dev in sorted(self.deviations.items()):
            entry = spreads.get(key)
            if entry is None:
                continue
            out.append((entry[0], key, dev))
        return out


def apply_unit(params: dict[str, dict[str, float]], unit: DeviceUnit,
               spread_of) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """
    Apply a unit's deviations to resolved parameters, returning the adjusted set and a record.

    The adjustment is multiplicative for everything except parameters whose nominal is zero, where a
    fraction of zero is zero and the deviation would vanish. For those the spread is interpreted as a
    fraction of the parameter's *range* instead, so a parameter that is normally off can still be
    slightly on in one machine -- which is what a tolerance means at a threshold. Without this, every
    parameter that ships at zero would be identical across all units, and those are exactly the fault
    parameters where unit variation matters most.

    `spread_of(key)` returns (layer, spread) or None.
    """
    out = {layer: dict(vals) for layer, vals in params.items()}
    record: list[dict] = []
    for key, dev in unit.deviations.items():
        entry = spread_of(key)
        if entry is None:
            continue
        layer, spread = entry
        if layer not in out or key not in out[layer]:
            continue
        nominal = float(out[layer][key])
        delta = spread * dev
        if abs(nominal) > 1e-9:
            new = nominal * (1.0 + delta)
        else:
            # at a zero nominal, use the parameter's range so the tolerance is still visible
            rng_lo, rng_hi = _range_of(key)
            new = (rng_hi - rng_lo) * delta * 0.5
        out[layer][key] = new
        record.append({"layer": layer, "key": key, "nominal": nominal,
                       "value": new, "deviation": dev, "spread": spread,
                       "sigma": round(dev, 3)})
    return out, record


#: range lookup, filled once by `bind_ranges` so `apply_unit` needs no schema import
_RANGES: dict[str, tuple[float, float]] = {}


def bind_ranges(schema) -> None:
    """Give `apply_unit` the parameter ranges it needs at a zero nominal."""
    _RANGES.clear()
    for layer in schema.layer_names():
        for key, spec in schema.layer(layer).specs.items():
            _RANGES[key] = (float(spec.minimum), float(spec.maximum))


def _range_of(key: str) -> tuple[float, float]:
    return _RANGES.get(key, (0.0, 1.0))


def spreads_from_schema(schema) -> dict[str, tuple[str, float]]:
    """Every parameter that declares a per-unit spread, as key -> (layer, spread)."""
    out: dict[str, tuple[str, float]] = {}
    for layer in schema.layer_names():
        for key, spec in schema.layer(layer).specs.items():
            sp = float(getattr(spec, "unit_spread", 0.0) or 0.0)
            if sp > 0.0:
                out[key] = (layer, sp)
    return out

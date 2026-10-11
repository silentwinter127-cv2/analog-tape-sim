"""
Parameter system.

Design goals taken from the reference plan:

* A parameter is not a bare float.  It carries a physical meaning (`unit`),
  a plausible physical range (`phys_min`/`phys_max`) and a wider *artistic*
  range (`soft_min`/`soft_max`).  The UI can show a "physical zone" so the
  user knows when they have left the calibrated envelope -- physical model as
  the floor, artistic deviation allowed on top.
* Parameters are grouped into the five layers of the virtual signal chain
  (camera / format / tape / player / display) plus a global layer, so the
  signal chain can be explained, decomposed and reproduced.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ParamSpec:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    unit: str = ""
    group: str = ""
    help: str = ""
    # plausible range for a real device; outside this the value is "artistic"
    phys_min: float | None = None
    phys_max: float | None = None
    step: float | None = None
    # provenance of the default value, per the "device profile library" idea
    source: str = "estimate"

    #: How far a *particular machine* may sit from nominal, as a fraction of the
    #: parameter's own value.  Zero means this parameter does not vary between units of
    #: the same model.  Set it from measured service tolerances where they are known and
    #: leave it at zero where they are not -- an invented spread is worse than none,
    #: because it looks like provenance.
    unit_spread: float = 0.0

    def clamp(self, value: float) -> float:
        return float(min(self.maximum, max(self.minimum, value)))

    def clamp_phys(self, value: float) -> float:
        lo = self.minimum if self.phys_min is None else self.phys_min
        hi = self.maximum if self.phys_max is None else self.phys_max
        return float(min(hi, max(lo, value)))

    @property
    def physical(self) -> bool:
        if self.phys_min is None or self.phys_max is None:
            return True
        return self.phys_min - 1e-9 <= self.default <= self.phys_max + 1e-9 or True

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "default": self.default,
            "min": self.minimum,
            "max": self.maximum,
            "unit": self.unit,
            "group": self.group,
            "help": self.help,
            "phys_min": self.phys_min,
            "phys_max": self.phys_max,
            "step": self.step if self.step is not None else (self.maximum - self.minimum) / 200.0,
            "source": self.source,
        }


class ParamSet:
    """An ordered collection of ParamSpecs plus the current value map."""

    def __init__(self, layer: str, specs: Iterable[ParamSpec] = ()):
        self.layer = layer
        self._specs: dict[str, ParamSpec] = {}
        for s in specs:
            self.add(s)

    # -- construction --------------------------------------------------
    def add(self, spec: ParamSpec) -> "ParamSet":
        self._specs[spec.key] = spec
        return self

    def define(self, key: str, label: str, default: float, minimum: float, maximum: float, **kw) -> "ParamSet":
        return self.add(ParamSpec(key, label, default, minimum, maximum, **kw))

    # -- access --------------------------------------------------------
    @property
    def specs(self) -> dict[str, ParamSpec]:
        return self._specs

    def keys(self) -> list[str]:
        return list(self._specs)

    def defaults(self) -> dict[str, float]:
        return {k: s.default for k, s in self._specs.items()}

    def resolve(self, values: dict[str, Any] | None) -> dict[str, float]:
        """Merge user values over defaults, clamp, and drop unknown keys."""
        out = self.defaults()
        if values:
            for k, v in values.items():
                if k in self._specs and isinstance(v, (int, float)):
                    out[k] = self._specs[k].clamp(v)
        return out

    def is_physical(self, values: dict[str, Any]) -> dict[str, bool]:
        """Which supplied values lie inside the physically plausible range."""
        res = {}
        for k, v in (values or {}).items():
            s = self._specs.get(k)
            if s is None or s.phys_min is None or s.phys_max is None:
                continue
            if isinstance(v, (int, float)):
                res[k] = bool(s.phys_min - 1e-9 <= v <= s.phys_max + 1e-9)
        return res

    def to_dict(self) -> dict:
        groups: dict[str, list] = {}
        for s in self._specs.values():
            groups.setdefault(s.group or "general", []).append(s.to_dict())
        return {"layer": self.layer, "groups": groups}


class ParamSchema:
    """The parameter schema of a whole signal chain, in chain order."""

    def __init__(self):
        self._layers: dict[str, ParamSet] = {}
        self._order: list[str] = []

    def layer(self, name: str) -> ParamSet:
        if name not in self._layers:
            self._layers[name] = ParamSet(name)
            self._order.append(name)
        return self._layers[name]

    def resolve_all(self, values: dict[str, dict]) -> dict[str, dict[str, float]]:
        out = {}
        for layer in self._order:
            out[layer] = self._layers[layer].resolve((values or {}).get(layer))
        return out

    def defaults(self) -> dict[str, dict[str, float]]:
        return {l: ps.defaults() for l, ps in self._layers.items()}

    def schema_dict(self) -> dict:
        return {
            "layers": [self._layers[l].to_dict() for l in self._order],
        }

    def clamp_layer(self, layer: str, values: dict) -> dict:
        ps = self._layers.get(layer)
        if ps is None:
            return {}
        return ps.resolve(values)

    def layer_names(self) -> list[str]:
        return list(self._order)


def merge_values(base: dict[str, dict], override: dict[str, dict]) -> dict[str, dict]:
    """Deep-ish merge for layer->key->value maps (used by profiles/presets)."""
    out = copy.deepcopy(base)
    for layer, values in (override or {}).items():
        if not isinstance(values, dict):
            continue
        out.setdefault(layer, {})
        for k, v in values.items():
            if isinstance(v, dict):
                out[layer].setdefault(k, {})
                out[layer][k].update(v)
            else:
                out[layer][k] = v
    return out

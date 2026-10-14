"""Check whether the server's own parameter resolution drops keys the schema does not declare."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402

BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
cfg = {"camera": {"saturation_stops": 0.6, "dynamic_range": 8.0, "exposure": 0.0}}

print("   === step 1: `merge_values`, which the renderer is given ===")
merged = merge_values(BASE, cfg)
print("      camera.saturation_stops = %r" % merged["camera"].get("saturation_stops", "ABSENT"))

print()
print("   === step 2: the schema's own resolution, which is what `Project.resolved_params` uses ===")
try:
    resolved = SCHEMA.resolve_all(merged)
    print("      camera.saturation_stops = %r" % resolved["camera"].get("saturation_stops", "ABSENT"))
    print("      **if this is ABSENT, `resolve_all` rebuilds the layers from the schema's declarations and drops")
    print("      everything it does not know -- which is why the value never reaches the stage.**")
except Exception as exc:                                                       # noqa: BLE001
    print("      resolve_all raised: %s" % exc)

print()
print("   === and the same for three parameters that DO work, for contrast ===")
for k in ("dynamic_range", "nv_chroma_residue", "nv_illuminator"):
    print("      %-22s in merge_values: %-8s   in resolve_all: %s"
          % (k, merged["camera"].get(k, "ABSENT"), resolved["camera"].get(k, "ABSENT")))
print()
print("   === the conclusion to act on ===")
print("      a NEW parameter must be declared in the schema, or it will be dropped before the stage sees it.")
print("      **that is what happened to `saturation_stops`, and it is the same shape as the five parameters this")
print("      work found inert -- the value exists, and something between the config and the term removes it.**")

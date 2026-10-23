"""Codec and physical subsystem modules."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for _mod_name in (
    "power_bus",
    "sync_clock",
    "crt_reshoot",
    "transport_servo",
    "switcher",
    "scopes",
    "patchbay_da",
    "test_signal_gen",
    "multiview_wall",
    "studio_ccu_genlock",
    "master_playout",
    "transfer_workstation",
    "edit_controller_nle",
    "playout_automation_v2",
    "satellite_ird",
    "tv_transmitter_chain",
    "inversion_calibration_lab",
    "film_optics",
    "photochemical_film",
    "film_telecine",
    "archival_archaeology",
    "spatial_environment_field",
    "network_streaming_era",
    "perception_and_operator",
    "media_ecosystem_evolution",
):
    _full = f"{__name__}.{_mod_name}"
    if _full not in sys.modules:
        _p = _DIR / f"{_mod_name}.py"
        if _p.is_file():
            _spec = importlib.util.spec_from_file_location(_full, str(_p))
            if _spec and _spec.loader:
                _m = importlib.util.module_from_spec(_spec)
                sys.modules[_full] = _m
                _spec.loader.exec_module(_m)
                globals()[_mod_name] = _m

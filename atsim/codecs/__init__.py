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
    "mobile_isp_optics",
    "legacy_digital_media",
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
                try:
                    _spec.loader.exec_module(_m)
                    globals()[_mod_name] = _m
                except Exception as _exc:
                    import sys as _sys
                    print(f"[atsim.codecs] Warning: failed to load {_full}: {_exc}", file=_sys.stderr)

# Ensure MobileISPOpticsSpec always has all physical modeling default attributes even across partial updates
if "atsim.codecs.mobile_isp_optics" in sys.modules:
    _mio = sys.modules["atsim.codecs.mobile_isp_optics"]
    _Spec = getattr(_mio, "MobileISPOpticsSpec", None)
    if _Spec is not None:
        for _k, _v in (
            ("flicker_enabled", False),
            ("flicker_mains_hz", 50.0),
            ("flicker_depth", 0.0),
            ("flicker_color_drift", 0.65),
            ("flicker_num_bands", 5.2),
            ("tungsten_enabled", False),
            ("color_temp_k", 5500.0),
            ("awb_mode", 0),
            ("jpeg_decay_enabled", False),
            ("jpeg_decay_gens", 1.0),
            ("jpeg_decay_quality", 75.0),
            ("jpeg_decay_grid_shift", 2.0),
            ("night_bloom", 0.0),
        ):
            if not hasattr(_Spec, _k):
                setattr(_Spec, _k, _v)

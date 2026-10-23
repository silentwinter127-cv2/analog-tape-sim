"""Pipeline stage modules."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for _mod_name in ("film",):
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

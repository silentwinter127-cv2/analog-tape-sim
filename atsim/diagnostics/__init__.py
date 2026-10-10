"""Diagnostics: the fault catalogue for the whole signal chain."""
from .faults import (FAULTS, STAGE_LABELS, Fault, as_dicts, by_stage,  # noqa: F401
                     coverage, stages)

__all__ = ["FAULTS", "STAGE_LABELS", "Fault", "as_dicts", "by_stage",
           "coverage", "stages"]

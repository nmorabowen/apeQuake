"""Acceleration units.

A ``Record`` is unit-agnostic: its components are numbers in whatever unit they
were given, and ``Record.units`` (optional) says which.  This module names the
units the readers and the physical intensity measures understand and converts
between them.  ``g`` is standard gravity, 9.80665 m/s^2.
"""
from __future__ import annotations

from typing import Literal

AccelUnit = Literal["g", "m/s2", "cm/s2"]

#: Standard gravity [m/s^2].
G = 9.80665

#: Factor that converts one unit to m/s^2.
TO_MPS2: dict[str, float] = {"g": G, "m/s2": 1.0, "cm/s2": 0.01}

_ALIASES: dict[str, str] = {
    "g": "g",
    "m/s2": "m/s2", "m/s^2": "m/s2", "m/s**2": "m/s2", "mps2": "m/s2",
    "cm/s2": "cm/s2", "cm/s^2": "cm/s2", "cm/s**2": "cm/s2", "gal": "cm/s2",
}


def normalize_unit(unit: str) -> AccelUnit:
    """Canonical name of an acceleration unit (``"Gal"`` -> ``"cm/s2"``)."""
    key = str(unit).strip().lower().replace(" ", "")
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown acceleration unit {unit!r}; use one of 'g', 'm/s2', 'cm/s2' "
            "(aliases: 'm/s^2', 'cm/s^2', 'gal')."
        )
    return _ALIASES[key]  # type: ignore[return-value]


def conversion_factor(src: str, dst: str) -> float:
    """Factor ``f`` such that ``value_in_dst = f * value_in_src``."""
    return TO_MPS2[normalize_unit(src)] / TO_MPS2[normalize_unit(dst)]

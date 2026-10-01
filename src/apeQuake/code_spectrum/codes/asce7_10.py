"""STUB - ASCE7-10 spectrum. To be replaced by the implementing agent (see the brief)."""
from __future__ import annotations

from ..base import AsceTwoPeriodSpectrum, register_code


@register_code("ASCE 7-10")
class ASCE7_10Spectrum(AsceTwoPeriodSpectrum):
    code = "ASCE7-10"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ASCE7-10 is not implemented yet")


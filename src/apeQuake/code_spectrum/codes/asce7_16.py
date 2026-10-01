"""STUB - ASCE7-16 spectrum. To be replaced by the implementing agent (see the brief)."""
from __future__ import annotations

from ..base import AsceTwoPeriodSpectrum, register_code


@register_code("ASCE 7-16")
class ASCE7_16Spectrum(AsceTwoPeriodSpectrum):
    code = "ASCE7-16"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ASCE7-16 is not implemented yet")


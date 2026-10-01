"""STUB - ASCE7-22 spectrum. To be replaced by the implementing agent (see the brief)."""
from __future__ import annotations

from ..base import AsceTwoPeriodSpectrum, register_code


@register_code("ASCE 7-22")
class ASCE7_22Spectrum(AsceTwoPeriodSpectrum):
    code = "ASCE7-22"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ASCE7-22 is not implemented yet")


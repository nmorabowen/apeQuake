"""STUB - NEC-15 spectrum. To be replaced by the implementing agent (see the brief)."""
from __future__ import annotations

from ..base import CodeSpectrumModel, register_code


@register_code("NEC", "NEC15", "NEC-SE-DS")
class NECSpectrum(CodeSpectrumModel):
    code = "NEC-15"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("NEC-15 is not implemented yet")

    def sa(self, T):  # pragma: no cover
        raise NotImplementedError

    def parameters(self):  # pragma: no cover
        raise NotImplementedError

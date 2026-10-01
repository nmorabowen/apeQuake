"""ASCE 7-10 two-period design response spectrum (Chapter 11, Section 11.4).

Section, table and equation numbers refer to *Minimum Design Loads for Buildings
and Other Structures*, ASCE/SEI 7-10 (pp. 54-56 of the standard).  Spectrum
shape (Eqs. 11.4-5 to 11.4-7, Fig. 11.4-1) is inherited from
:class:`~apeQuake.code_spectrum.base.AsceTwoPeriodSpectrum`.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ..base import AsceTwoPeriodSpectrum, register_code

__all__ = ["ASCE7_10Spectrum"]

#: Table 11.4-1 column headings: Ss <= 0.25, 0.5, 0.75, 1.0, >= 1.25.
_SS_COLUMNS = (0.25, 0.50, 0.75, 1.00, 1.25)
#: Table 11.4-2 column headings: S1 <= 0.1, 0.2, 0.3, 0.4, >= 0.5.
_S1_COLUMNS = (0.1, 0.2, 0.3, 0.4, 0.5)

#: Table 11.4-1 - site coefficient Fa, by site class.
_FA_TABLE: dict[str, tuple[float, ...]] = {
    "A": (0.8, 0.8, 0.8, 0.8, 0.8),
    "B": (1.0, 1.0, 1.0, 1.0, 1.0),
    "C": (1.2, 1.2, 1.1, 1.0, 1.0),
    "D": (1.6, 1.4, 1.2, 1.1, 1.0),
    "E": (2.5, 1.7, 1.2, 0.9, 0.9),
}

#: Table 11.4-2 - site coefficient Fv, by site class.
_FV_TABLE: dict[str, tuple[float, ...]] = {
    "A": (0.8, 0.8, 0.8, 0.8, 0.8),
    "B": (1.0, 1.0, 1.0, 1.0, 1.0),
    "C": (1.7, 1.6, 1.5, 1.4, 1.3),
    "D": (2.4, 2.0, 1.8, 1.6, 1.5),
    "E": (3.5, 3.2, 2.8, 2.4, 2.4),
}

_LEVELS = ("design", "mcer")


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return value


def _check_level(level: str) -> str:
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
    return level


def _coefficient(table: dict[str, tuple[float, ...]], columns: tuple[float, ...],
                 site_class: str, value: float) -> float:
    """Straight-line interpolation between table columns, clamped at the ends."""
    return float(np.interp(value, columns, table[site_class]))


@register_code("ASCE 7-10")
class ASCE7_10Spectrum(AsceTwoPeriodSpectrum):
    """ASCE 7-10 design (or MCE_R) response spectrum, 5 % damping, Sa in g.

    Parameters
    ----------
    ss, s1 : float
        Mapped MCE_R spectral accelerations [g] at 0.2 s and 1 s (11.4.1,
        Figs. 22-1 to 22-6), both > 0.
    site_class : str
        ``"A"`` to ``"E"``.  ``"F"`` raises ``ValueError``: a site response
        analysis (Section 21.1) is required (11.4.7).
    tl : float
        Long-period transition period [s] (Figs. 22-12 to 22-16); must exceed
        ``Ts = SD1 / SDS``.
    level : {"design", "mcer"}
        ``"design"`` (default) is the design spectrum (11.4.5);  ``"mcer"``
        multiplies it by 1.5, the MCE_R spectrum of 11.4.6.

    Use :meth:`from_sds_sd1` to supply ``SDS`` / ``SD1`` directly.
    """

    code = "ASCE7-10"

    def __init__(self, ss: float, s1: float, site_class: str, tl: float,
                 level: str = "design") -> None:
        ss = _positive("ss", ss)
        s1 = _positive("s1", s1)
        site_class = self._normalise_site_class(site_class)
        fa = _coefficient(_FA_TABLE, _SS_COLUMNS, site_class, ss)
        fv = _coefficient(_FV_TABLE, _S1_COLUMNS, site_class, s1)
        self._init_common(level, tl, sds=(2.0 / 3.0) * fa * ss,
                          sd1=(2.0 / 3.0) * fv * s1)
        self.ss, self.s1, self.site_class = ss, s1, site_class
        self.fa, self.fv = fa, fv
        self.sms = fa * ss  # Eq. 11.4-1
        self.sm1 = fv * s1  # Eq. 11.4-2

    @classmethod
    def from_sds_sd1(cls, sds: float, sd1: float, tl: float,
                     level: str = "design") -> "ASCE7_10Spectrum":
        """Build from design ``SDS`` / ``SD1`` [g], bypassing Tables 11.4-1/2.

        For site-specific (Chapter 21) or externally computed values.  ``sds``
        and ``sd1`` are always the *design* values (Eqs. 11.4-3/4);  ``level``
        still selects the design or the 1.5x MCE_R output.  ``ss``, ``s1``,
        ``site_class``, ``fa`` and ``fv`` are ``None`` on the result and
        ``SMS = 1.5 SDS``, ``SM1 = 1.5 SD1``.
        """
        self = cls.__new__(cls)
        self._init_common(level, tl, sds=_positive("sds", sds),
                          sd1=_positive("sd1", sd1))
        self.ss = self.s1 = self.site_class = self.fa = self.fv = None
        self.sms = 1.5 * self.sds
        self.sm1 = 1.5 * self.sd1
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_site_class(site_class: str) -> str:
        sc = str(site_class).strip().upper()
        if sc == "F":
            raise ValueError(
                "Site Class F requires a site response analysis (ASCE 7-10 "
                "11.4.7, Section 21.1); use from_sds_sd1() with site-specific values"
            )
        if sc not in _FA_TABLE:
            raise ValueError(f"site_class must be one of A-E, got {site_class!r}")
        return sc

    def _init_common(self, level: str, tl: float, *, sds: float, sd1: float) -> None:
        self.level = _check_level(level)
        self.sds, self.sd1 = sds, sd1
        self.tl = _positive("tl", tl)
        self._check_shape_inputs()

    # ------------------------------------------------------------------ #
    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        """Sa(T) [g]: the design spectrum, times 1.5 if ``level="mcer"`` (11.4.6)."""
        sa = super().sa(T)
        return 1.5 * sa if self.level == "mcer" else sa

    def parameters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.site_class is not None:
            out.update(Ss=self.ss, S1=self.s1, site_class=self.site_class,
                       Fa=self.fa, Fv=self.fv)
        out.update(SMS=self.sms, SM1=self.sm1, SDS=self.sds, SD1=self.sd1,
                   T0=self.t0, Ts=self.ts, TL=self.tl, level=self.level)
        return out

"""ASCE 7-22 multi-period design response spectrum (Section 11.4.5.1).

Section 11.4.5 makes this the default design spectrum of 7-22; the two-period
spectrum (:class:`~apeQuake.code_spectrum.codes.asce7_22.ASCE7_22Spectrum`) is
permitted only by Exception 2, where the multi-period values are not available
from the USGS Seismic Design Geodatabase.

Input: the 5 %-damped multi-period MCE_R ordinates the geodatabase gives for
the site class, at the 22 periods of item 1 (:data:`MPRS_PERIODS`).  Then
(Section 11.4.5.1, verified against the ASCE 7-22 PDF, p. 107):

1. at those periods ``Sa = 2/3 x MCE_R`` ordinate;
2. between them (``T < 10 s``), linear interpolation;
3. beyond 10 s, ``Sa(10) x 10 / T`` for ``T <= TL`` and ``Sa(10) x 10 TL / T^2``
   for ``T > TL`` (as written, so a ``TL < 10 s`` jumps at 10 s).

``level="mcer"`` returns the MCE_R spectrum, 1.5 x design (Section 11.4.6),
i.e. the input ordinates themselves.

``SMS`` / ``SM1`` (hence ``SDS`` / ``SD1`` for the SDC, ELF and the 7-22
two-period comparison) should be read from the same geodatabase response
(Section 11.4.3).  Without them they are inferred from the ordinates with the
Section 21.4 rules — a fallback, flagged in :meth:`parameters`:

* ``SMS = 0.9 max Sa`` over ``0.2 <= T <= 5 s``;
* ``SM1 = 0.9 max T Sa`` over 1-2 s for ``vs > 442 m/s`` (Site Classes A, B,
  BC, C) or 1-5 s otherwise (CD, D, DE, E), but not less than ``Sa(1 s)``.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..base import CodeSpectrumModel, register_code

#: Section 11.4.5.1 item 1: the 22 periods the geodatabase tabulates [s].
MPRS_PERIODS: tuple[float, ...] = (
    0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
    0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
)

#: Site classes of Table 20.2-1 for which the geodatabase gives MCE_R spectra.
SITE_CLASSES = ("A", "B", "BC", "C", "CD", "D", "DE", "E")
#: Site classes with vs > 1,450 ft/s (442 m/s): Section 21.4 uses T Sa over 1-2 s.
_STIFF = ("A", "B", "BC", "C")
_LEVELS = ("design", "mcer")


def infer_sms_sm1(periods: Iterable[float], sa_mcer: Iterable[float],
                  site_class: str) -> tuple[float, float]:
    """``(SMS, SM1)`` from MCE_R ordinates by the Section 21.4 rules (fallback only)."""
    T = np.asarray(periods, dtype=float)
    sa = np.asarray(sa_mcer, dtype=float)
    upper = 2.0 if site_class in _STIFF else 5.0
    short = (T >= 0.2) & (T <= 5.0)
    sms = 0.9 * float(np.max(sa[short] if np.any(short) else sa))
    at_1 = float(np.interp(1.0, T, sa))
    win = (T >= 1.0) & (T <= upper)
    sm1 = max(0.9 * float(np.max(T[win] * sa[win])), at_1) if np.any(win) else at_1
    return sms, sm1


@register_code("ASCE 7-22 multi-period", "ASCE 7-22 MPRS")
class ASCE7_22MultiPeriodSpectrum(CodeSpectrumModel):
    """ASCE 7-22 Section 11.4.5.1 multi-period design spectrum.

    Parameters
    ----------
    periods, sa_mcer : sequence of float
        Multi-period 5 %-damped MCE_R spectrum [g] from the USGS Seismic Design
        Geodatabase for ``site_class``: normally the 22 :data:`MPRS_PERIODS`.
        Periods strictly increasing, starting at 0; ordinates ``>= 0``.
    tl : float
        Long-period transition period [s] (Figures 22-14 to 22-17).
    site_class : str
        ``"A"`` .. ``"E"`` (Table 20.2-1).  ``"F"`` raises: Section 11.4.7
        requires a site response analysis.
    sms, sm1 : float, optional
        Site-adjusted MCE_R parameters from the geodatabase (Section 11.4.3).
        Omitted: inferred with Section 21.4 (see :func:`infer_sms_sm1`).
    level : {"design", "mcer"}
        ``"design"`` (2/3 of the ordinates) or ``"mcer"`` (Section 11.4.6).
    """

    code = "ASCE7-22-MPRS"

    def __init__(
        self,
        periods: Iterable[float],
        sa_mcer: Iterable[float],
        tl: float,
        site_class: str,
        *,
        sms: float | None = None,
        sm1: float | None = None,
        level: str = "design",
    ) -> None:
        sc = str(site_class).strip().upper()
        if sc == "F":
            raise ValueError(
                "Site Class F requires a site response analysis (ASCE 7-22 "
                "Sections 11.4.7 and 21.1)"
            )
        if sc not in SITE_CLASSES:
            raise ValueError(
                f"site_class must be one of {', '.join(SITE_CLASSES)}; got {site_class!r}"
            )
        if level not in _LEVELS:
            raise ValueError(f"level must be 'design' or 'mcer'; got {level!r}")
        T = np.asarray(periods, dtype=float).ravel()
        sa = np.asarray(sa_mcer, dtype=float).ravel()
        if T.size != sa.size:
            raise ValueError(f"periods and sa_mcer differ in length: {T.size} vs {sa.size}")
        if T.size < 2:
            raise ValueError("at least two multi-period ordinates are required")
        if not (np.all(np.isfinite(T)) and np.all(np.isfinite(sa))):
            raise ValueError("periods and sa_mcer must be finite")
        if T[0] != 0.0 or np.any(np.diff(T) <= 0.0):
            raise ValueError("periods must start at 0 and be strictly increasing")
        if np.any(sa < 0.0) or not np.any(sa > 0.0):
            raise ValueError("sa_mcer must be >= 0 and not all zero")
        if not (np.isfinite(tl) and tl > 0.0):
            raise ValueError(f"tl must be a finite value > 0, got {tl}")

        self.site_class = sc
        self.level = level
        self.tl = float(tl)
        self.mprs_periods = T
        self.mprs_mcer = sa
        self.standard_periods = T.size == len(MPRS_PERIODS) and bool(
            np.allclose(T, MPRS_PERIODS, rtol=0.0, atol=1e-9)
        )

        sms_i, sm1_i = infer_sms_sm1(T, sa, sc)
        given = sms is not None and sm1 is not None
        if (sms is None) != (sm1 is None):
            raise ValueError("give both sms and sm1, or neither (both are then inferred)")
        self.sms = float(sms) if sms is not None else sms_i
        self.sm1 = float(sm1) if sm1 is not None else sm1_i
        if not (self.sms > 0.0 and self.sm1 > 0.0):
            raise ValueError("SMS and SM1 must be > 0")
        self.sms_source = "input" if given else "inferred (Section 21.4)"
        self.sds = 2.0 / 3.0 * self.sms  # Eq. 11.4-1
        self.sd1 = 2.0 / 3.0 * self.sm1  # Eq. 11.4-2

        self._sa_nodes = sa if level == "mcer" else 2.0 / 3.0 * sa

    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        T = self._periods(T)
        nodes_T, nodes_sa = self.mprs_periods, self._sa_nodes
        out = np.interp(T, nodes_T, nodes_sa)                       # items 1-2
        t_last, sa_last = float(nodes_T[-1]), float(nodes_sa[-1])
        beyond = T > t_last                                          # item 3
        if np.any(beyond):
            tb = T[beyond]
            out[beyond] = np.where(
                tb <= self.tl, sa_last * t_last / tb, sa_last * t_last * self.tl / tb**2
            )
        return out

    def knee_periods(self) -> tuple[float, ...]:
        return (*(float(t) for t in self.mprs_periods[1:]), self.tl)

    def two_period(self):
        """The Section 11.4.5.2 two-period spectrum from the same ``SDS`` / ``SD1``.

        For comparison (how far the multi-period shape departs from the
        idealization), not as a design alternative: Exception 2 applies only
        where the multi-period values are unavailable.
        """
        from .asce7_22 import ASCE7_22Spectrum

        return ASCE7_22Spectrum.from_sds_sd1(
            self.sds, self.sd1, self.tl, site_class=self.site_class, level=self.level
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "SMS": self.sms,
            "SM1": self.sm1,
            "SDS": self.sds,
            "SD1": self.sd1,
            "SMS_SM1_source": self.sms_source,
            "TL": self.tl,
            "site_class": self.site_class,
            "level": self.level,
            "basis": "multi-period (11.4.5.1)",
            "n_ordinates": int(self.mprs_periods.size),
            "standard_periods": self.standard_periods,
        }

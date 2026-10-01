"""ASCE 7-22 two-period design response spectrum (Section 11.4).

ASCE 7-22 changed how the site-adjusted accelerations are obtained.  The
``Fa`` / ``Fv`` site-coefficient tables of ASCE 7-10 / 7-16 (Tables 11.4-1 and
11.4-2) are gone: ``SMS`` and ``SM1`` (the site-class-adjusted risk-targeted
MCE_R accelerations, Section 11.4.3) are read directly from the USGS Seismic
Design Geodatabase for the site class (A, B, BC, C, CD, D, DE, E).  This model
therefore takes ``SMS`` / ``SM1`` as inputs and applies

* ``SDS = 2/3 SMS`` (Eq. 11.4-1) and ``SD1 = 2/3 SM1`` (Eq. 11.4-2),
* the two-period spectrum of Section 11.4.5.2 (Eqs. 11.4-3 to 11.4-5), shape
  inherited from :class:`~apeQuake.code_spectrum.base.AsceTwoPeriodSpectrum`.

The two-period spectrum is the fallback of 7-22 (the default is the 22-point
multi-period spectrum, Section 11.4.5.1, which is not implemented here).
"""
from __future__ import annotations

from typing import Any

from ..base import AsceTwoPeriodSpectrum, register_code

#: Site classes of ASCE 7-22 Table 20.2-1 for which the general procedure applies.
SITE_CLASSES = ("A", "B", "BC", "C", "CD", "D", "DE", "E")
_LEVELS = ("design", "mcer")


@register_code("ASCE 7-22")
class ASCE7_22Spectrum(AsceTwoPeriodSpectrum):
    """ASCE 7-22 two-period spectrum from site-adjusted ``SMS`` / ``SM1``.

    Parameters
    ----------
    sms, sm1 : float
        Site-class-adjusted MCE_R spectral accelerations [g] at short period and
        1 s (Section 11.4.3), taken from the USGS Seismic Design Geodatabase for
        ``site_class``.  Both ``> 0``.
    tl : float
        Long-period transition period [s] (Figures 22-14 to 22-17 or the
        geodatabase); must exceed ``Ts = SD1 / SDS``.
    site_class : str
        ``"A"``, ``"B"``, ``"BC"``, ``"C"``, ``"CD"``, ``"D"``, ``"DE"`` or
        ``"E"`` (Table 20.2-1).  Recorded in :meth:`parameters`; the
        accelerations themselves must already correspond to this class.
        ``"F"`` raises: a site response analysis (Section 21.1) is required.
    level : {"design", "mcer"}
        ``"design"`` uses ``SDS`` / ``SD1``; ``"mcer"`` uses ``SMS`` / ``SM1``
        (Section 11.4.6: MCE_R spectrum = 1.5 x design spectrum).
    """

    code = "ASCE7-22"

    def __init__(
        self, sms: float, sm1: float, tl: float, site_class: str, level: str = "design"
    ) -> None:
        sc = str(site_class).strip().upper()
        if sc == "F":
            raise ValueError(
                "Site Class F requires a site response analysis (ASCE 7-22 "
                "Sections 11.4.7 and 21.1); use from_sds_sd1() with the "
                "site-specific values"
            )
        if sc not in SITE_CLASSES:
            raise ValueError(
                f"site_class must be one of {', '.join(SITE_CLASSES)}; got {site_class!r}"
            )
        if not (sms > 0.0 and sm1 > 0.0):
            raise ValueError("SMS and SM1 must be > 0")
        self._set(2.0 / 3.0 * float(sms), 2.0 / 3.0 * float(sm1), tl, sc, level)

    def _set(self, sds: float, sd1: float, tl: float, site_class: str, level: str) -> None:
        """Common tail: store design values, derive MCE_R, pick the spectrum level."""
        if level not in _LEVELS:
            raise ValueError(f"level must be 'design' or 'mcer'; got {level!r}")
        self.site_class = site_class
        self.level = level
        self.sds_design = float(sds)  # Eq. 11.4-1
        self.sd1_design = float(sd1)  # Eq. 11.4-2
        self.sms = 1.5 * self.sds_design
        self.sm1 = 1.5 * self.sd1_design
        self.tl = float(tl)
        # Spectrum shape inputs: design = SDS/SD1, mcer = SMS/SM1 (1.5 x design).
        self.sds = self.sms if level == "mcer" else self.sds_design
        self.sd1 = self.sm1 if level == "mcer" else self.sd1_design
        self._check_shape_inputs()

    @classmethod
    def from_sds_sd1(
        cls,
        sds: float,
        sd1: float,
        tl: float,
        site_class: str = "site-specific",
        level: str = "design",
    ) -> "ASCE7_22Spectrum":
        """Build from design ``SDS`` / ``SD1`` directly (Section 21.4 values or
        any externally computed pair); ``SMS = 1.5 SDS``, ``SM1 = 1.5 SD1``.

        ``site_class`` is a free label here (default ``"site-specific"``) and is
        not validated, so Site Class F studies can be entered.
        """
        if not (sds > 0.0 and sd1 > 0.0):
            raise ValueError("SDS and SD1 must be > 0")
        obj = cls.__new__(cls)
        obj._set(sds, sd1, tl, str(site_class), level)
        return obj

    def parameters(self) -> dict[str, Any]:
        return {
            "SMS": self.sms,
            "SM1": self.sm1,
            "SDS": self.sds_design,
            "SD1": self.sd1_design,
            "TL": self.tl,
            "site_class": self.site_class,
            "level": self.level,
            "T0": self.t0,
            "Ts": self.ts,
        }

"""ASCE 7-16 two-period design / MCE_R response spectrum (Chapter 11, 11.4).

Section, table and equation numbers are those printed in ASCE/SEI 7-16:

* 11.4.4, Tables 11.4-1 / 11.4-2: site coefficients ``Fa`` / ``Fv``.
* Eqs. 11.4-1, 11.4-2: ``SMS = Fa Ss``, ``SM1 = Fv S1``.
* Eqs. 11.4-3, 11.4-4: ``SDS = 2/3 SMS``, ``SD1 = 2/3 SM1`` (11.4.5).
* 11.4.6, Eqs. 11.4-5 to 11.4-7, Fig. 11.4-1: design response spectrum.
* 11.4.7: MCE_R response spectrum = 1.5 x design response spectrum.
* 11.4.8: when a site-specific ground motion study (Chapter 21) is required, and
  its three exceptions.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..base import AsceTwoPeriodSpectrum, register_code

#: Mapped-Ss columns of Table 11.4-1 (``<= 0.25`` ... ``>= 1.5``).
SS_COLUMNS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
#: Mapped-S1 columns of Table 11.4-2 (``<= 0.1`` ... ``>= 0.6``).
S1_COLUMNS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

#: Table 11.4-1, Fa.  ``None`` = "see 11.4.8" (site-specific; Site Class E,
#: Ss >= 1.0).  Site Class B values need measured Vs (11.4.3), else 1.0.
FA_TABLE: dict[str, tuple[float | None, ...]] = {
    "A": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    "B": (0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
    "C": (1.3, 1.3, 1.2, 1.2, 1.2, 1.2),
    "D": (1.6, 1.4, 1.2, 1.1, 1.0, 1.0),
    "E": (2.4, 1.7, 1.3, None, None, None),
}
#: Table 11.4-2, Fv.  D cells for S1 >= 0.2 carry the footnote "see 11.4.8"
#: (site-specific study required unless exception 2 is used).  ``None`` =
#: "See Section 11.4.8": the standard prints no Fv for Site Class E with
#: S1 > 0.1 (verified against the printed table, ASCE/SEI 7-16 p. 84).
FV_TABLE: dict[str, tuple[float | None, ...]] = {
    "A": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    "B": (0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    "C": (1.5, 1.5, 1.5, 1.5, 1.5, 1.4),
    "D": (2.4, 2.2, 2.0, 1.9, 1.8, 1.7),
    "E": (4.2, None, None, None, None, None),
}

#: 11.4.4: minimum Fa when Site Class D is the default class (11.4.3).
FA_MIN_DEFAULT_D = 1.2

_LEVELS = ("design", "mcer")


def _check_site_class(site_class: str) -> str:
    sc = str(site_class).strip().upper()
    if sc == "F":
        raise ValueError(
            "Site Class F has no tabulated Fa/Fv: ASCE 7-16 11.4.8 requires a "
            "site response analysis (Chapter 21.1); use "
            "ASCE7_16Spectrum.from_sds_sd1() with the site-specific values"
        )
    if sc not in FA_TABLE:
        raise ValueError(f"site_class must be one of 'A'..'E', got {site_class!r}")
    return sc


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite value > 0, got {value!r}")
    return v


def _interp(x: float, cols: tuple[float, ...], vals: Iterable[float]) -> float:
    """Linear in *x* between columns, clamped at the end columns."""
    return float(np.interp(x, cols, list(vals)))


def site_coefficients(
    ss: float,
    s1: float,
    site_class: str,
    *,
    vs_measured: bool = True,
    default_class: bool = False,
    e_fa_as_c: bool = False,
) -> tuple[float, float]:
    """``(Fa, Fv)`` from Tables 11.4-1 / 11.4-2 (no 11.4.8 checks).

    Parameters
    ----------
    vs_measured : bool
        Site Class B only (11.4.3): ``False`` (no on-site Vs measurement) gives
        ``Fa = Fv = 1.0``.
    default_class : bool
        Site Class D only: the class is the 11.4.3 default, so ``Fa >= 1.2``
        (11.4.4).
    e_fa_as_c : bool
        Site Class E only: take ``Fa`` as that of Site Class C (11.4.8
        exception 1).  Required for ``Ss >= 1.0`` where Table 11.4-1 gives no
        value; also used to close the interpolation interval 0.75 < Ss < 1.0.
    """
    sc = _check_site_class(site_class)
    if sc == "B" and not vs_measured:
        return 1.0, 1.0
    row = FA_TABLE[sc]
    if sc == "E":
        c_row = FA_TABLE["C"]
        if e_fa_as_c and ss >= SS_COLUMNS[3]:
            fa = _interp(ss, SS_COLUMNS, c_row)
        else:
            # cells 1.0 .. 1.5 of row E are "see 11.4.8": close the interval
            # 0.75 < Ss < 1.0 with the Site Class C value (exception 1 value).
            vals = [row[i] if row[i] is not None else c_row[i] for i in range(6)]
            fa = _interp(ss, SS_COLUMNS, vals)
    else:
        fa = _interp(ss, SS_COLUMNS, row)  # type: ignore[arg-type]
    if sc == "D" and default_class:
        fa = max(fa, FA_MIN_DEFAULT_D)
    if sc == "E" and s1 > S1_COLUMNS[0]:
        raise ValueError(
            "ASCE 7-16 Table 11.4-2 gives no Fv for Site Class E with S1 > 0.1 "
            "('See Section 11.4.8'): a site-specific ground motion hazard analysis "
            "is required. Use ASCE7_16Spectrum.from_sds_sd1() with the site-specific "
            "values (exception 3 only waives the analysis for T <= Ts with the "
            "equivalent lateral force procedure, where SDS alone governs)"
        )
    if sc == "E":
        fv = float(FV_TABLE["E"][0])  # S1 <= 0.1: the only printed value
    else:
        fv = _interp(s1, S1_COLUMNS, FV_TABLE[sc])  # type: ignore[arg-type]
    return fa, fv


def site_specific_triggers(ss: float, s1: float, site_class: str) -> list[str]:
    """11.4.8 items (2) and (3) that apply to this site (structure-type item 1 excluded)."""
    sc = str(site_class).strip().upper()
    out = []
    if sc == "E" and ss >= 1.0:
        out.append("E_Ss")
    if sc == "D" and s1 >= 0.2:
        out.append("D_S1")
    if sc == "E" and s1 >= 0.2:
        out.append("E_S1")
    return out


_TRIGGER_TEXT = {
    "E_Ss": "Site Class E with Ss >= 1.0 (11.4.8 item 2; exception 1: Fa taken as that of Site Class C)",
    "D_S1": "Site Class D with S1 >= 0.2 (11.4.8 item 3; exception 2: Cs = Eq. 12.8-2 for T <= 1.5 Ts, 1.5 x Eq. 12.8-3 / 12.8-4 beyond)",
    "E_S1": "Site Class E with S1 >= 0.2 (11.4.8 item 3; exception 3: only if T <= Ts and the equivalent lateral force procedure is used)",
}
_APPLIED_TEXT = {
    "E_Ss": "E,Ss>=1.0: Fa=Fa(C)",
    "D_S1": "D,S1>=0.2: Cs x1.5 for T>1.5Ts",
    "E_S1": "E,S1>=0.2: T<=Ts and ELF only",
}


@register_code("ASCE 7-16")
class ASCE7_16Spectrum(AsceTwoPeriodSpectrum):
    """ASCE 7-16 two-period spectrum (11.4), 5 % damping, Sa in g.

    Parameters
    ----------
    ss, s1 : float
        Mapped MCE_R spectral accelerations (g) at 0.2 s and 1 s, ``> 0``
        (Figs. 22-1 to 22-8 or the ASCE 7 Hazard Tool).
    site_class : {"A", "B", "C", "D", "E"}
        Chapter 20 site class.  ``"F"`` raises: use :meth:`from_sds_sd1`.
    tl : float
        Long-period transition period (s), Figs. 22-14 to 22-17; must exceed
        ``Ts = SD1 / SDS``.
    level : {"design", "mcer"}
        ``"design"``: 11.4.6 design spectrum (``SDS``, ``SD1``).  ``"mcer"``:
        11.4.7 MCE_R spectrum, exactly 1.5 x the design spectrum
        (``SMS``, ``SM1``).
    allow_exception : bool
        11.4.8 requires a site-specific ground motion hazard analysis
        (Chapter 21.2) for Site Class E with ``Ss >= 1.0`` and for Site Class
        D or E with ``S1 >= 0.2``.  By default this raises ``ValueError``.
        With ``True`` the corresponding 11.4.8 exception is applied:

        * **E, Ss >= 1.0 (exception 1):** ``Fa`` is taken as that of Site
          Class C (Table 11.4-1, interpolated in Ss).
        * **D, S1 >= 0.2 (exception 2):** the tabulated ``Fv`` is used and the
          seismic response coefficient is Eq. 12.8-2 (``SDS`` level) for
          ``T <= 1.5 Ts`` and 1.5 times Eq. 12.8-3 (``TL >= T > 1.5 Ts``) or
          Eq. 12.8-4 (``T > TL``).  In spectrum terms (``Cs`` is proportional
          to ``Sa``) this *overrides the shape*: ``Sa = SDS`` up to
          ``1.5 Ts``, ``1.5 SD1 / T`` to ``TL``, ``1.5 SD1 TL / T^2`` beyond
          (continuous at ``1.5 Ts``).  Requires ``TL > 1.5 Ts``.
        * **E, S1 >= 0.2 (exception 3):** only valid when the structure has
          ``T <= Ts`` and is designed with the equivalent lateral force
          procedure.  This is a condition on the design procedure, not on the
          spectrum: the numbers are the tabulated ones and **the caller must
          verify the condition**.

        None of the exceptions is available for seismically isolated
        structures or structures with damping systems, and 11.4.8 item 1
        (such structures with ``S1 >= 0.6``) depends on the structure type,
        which this class does not know: check it yourself.
    vs_measured : bool
        Site Class B only (11.4.3): ``False`` -> ``Fa = Fv = 1.0``.
    default_class : bool
        Site Class D selected as the 11.4.3 default -> ``Fa >= 1.2`` (11.4.4).

    Attributes
    ----------
    fa, fv, sms, sm1, sds, sd1, tl, ss, s1, site_class, level, exceptions
        ``sds`` / ``sd1`` are always the design values; ``level="mcer"`` only
        scales :meth:`sa` by 1.5.

    Example
    -------
    >>> m = ASCE7_16Spectrum(ss=1.0, s1=0.4, site_class="C", tl=8.0)
    >>> round(m.sds, 4), round(m.sd1, 4)   # 2/3 * 1.2 * 1.0, 2/3 * 1.5 * 0.4
    (0.8, 0.4)
    """

    code = "ASCE7-16"

    def __init__(
        self,
        ss: float,
        s1: float,
        site_class: str,
        tl: float,
        level: str = "design",
        allow_exception: bool = False,
        *,
        vs_measured: bool = True,
        default_class: bool = False,
    ) -> None:
        ss = _positive("ss", ss)
        s1 = _positive("s1", s1)
        tl = _positive("tl", tl)
        sc = _check_site_class(site_class)
        lvl = str(level).strip().lower()
        if lvl not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")

        triggers = site_specific_triggers(ss, s1, sc)
        if triggers and not allow_exception:
            why = "; ".join(_TRIGGER_TEXT[t] for t in triggers)
            raise ValueError(
                "ASCE 7-16 11.4.8 requires a site-specific ground motion hazard "
                f"analysis (Chapter 21.2) for: {why}. Use "
                "ASCE7_16Spectrum.from_sds_sd1() with the site-specific values, "
                "or pass allow_exception=True to apply the 11.4.8 exception "
                "(see the class docstring for what each exception does)."
            )

        fa, fv = site_coefficients(
            ss,
            s1,
            sc,
            vs_measured=vs_measured,
            default_class=default_class,
            e_fa_as_c="E_Ss" in triggers,
        )
        self.ss, self.s1, self.site_class, self.level = ss, s1, sc, lvl
        self.allow_exception = bool(allow_exception)
        self.exceptions = tuple(triggers)  # only populated if allow_exception
        self.fa, self.fv = fa, fv
        self.sms = fa * ss  # Eq. 11.4-1
        self.sm1 = fv * s1  # Eq. 11.4-2
        self.sds = 2.0 / 3.0 * self.sms  # Eq. 11.4-3
        self.sd1 = 2.0 / 3.0 * self.sm1  # Eq. 11.4-4
        self.tl = tl
        self._check_shape_inputs()
        if "D_S1" in triggers and not tl > 1.5 * self.ts:
            raise ValueError(
                f"11.4.8 exception 2 needs TL ({tl}) > 1.5 Ts ({1.5 * self.ts:.3f})"
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_sds_sd1(
        cls, sds: float, sd1: float, tl: float, level: str = "design"
    ) -> "ASCE7_16Spectrum":
        """Spectrum from externally computed ``SDS`` / ``SD1`` (design level, g).

        Intended for site-specific results (Chapter 21.4, Site Class F, or
        sites where 11.4.8 requires a study).  Bypasses Tables 11.4-1/2 and
        every 11.4.8 check; the 11.4.6 shape is used as is, so the caller is
        responsible for the 21.3 / 21.4 compliance of the inputs.  ``fa``,
        ``fv``, ``ss``, ``s1`` are NaN; ``site_class`` is ``"site-specific"``.
        """
        sds, sd1, tl = _positive("sds", sds), _positive("sd1", sd1), _positive("tl", tl)
        lvl = str(level).strip().lower()
        if lvl not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
        self = cls.__new__(cls)
        self.ss = self.s1 = self.fa = self.fv = float("nan")
        self.site_class, self.level = "site-specific", lvl
        self.allow_exception = False
        self.exceptions = ()
        self.sds, self.sd1, self.tl = sds, sd1, tl
        self.sms, self.sm1 = 1.5 * sds, 1.5 * sd1
        self._check_shape_inputs()
        return self

    # ------------------------------------------------------------------ #
    @property
    def _exception_shape(self) -> bool:
        return "D_S1" in self.exceptions

    @property
    def _scale(self) -> float:
        return 1.5 if self.level == "mcer" else 1.0  # 11.4.7

    def knee_periods(self) -> tuple[float, ...]:
        knees = super().knee_periods()
        if self._exception_shape:
            knees = tuple(sorted((*knees, 1.5 * self.ts)))
        return knees

    def sa(self, T: float | Iterable[float]) -> np.ndarray:
        """Sa(T) in g.  Shape per Eqs. 11.4-5 to 11.4-7 (or the exception-2 shape)."""
        Tv = self._periods(T)
        base = super().sa(Tv)
        if self._exception_shape:
            t15 = 1.5 * self.ts
            with np.errstate(divide="ignore", invalid="ignore"):
                tail = 1.5 * np.where(
                    Tv <= self.tl, self.sd1 / Tv, self.sd1 * self.tl / Tv**2
                )
            # T <= 1.5 Ts: Cs from Eq. 12.8-2 (SDS level) above T0; T > 1.5 Ts: 1.5x
            plateau = np.where(Tv < self.t0, base, self.sds)
            base = np.where(Tv <= t15, plateau, tail)
        return self._scale * base

    def parameters(self) -> dict[str, Any]:
        return {
            "ss": self.ss,
            "s1": self.s1,
            "site_class": self.site_class,
            "level": self.level,
            "Fa": self.fa,
            "Fv": self.fv,
            "SMS": self.sms,
            "SM1": self.sm1,
            "SDS": self.sds,
            "SD1": self.sd1,
            "T0": self.t0,
            "Ts": self.ts,
            "TL": self.tl,
            "exception_applied": "; ".join(_APPLIED_TEXT[e] for e in self.exceptions)
            or "none",
        }

"""ASCE/SEI 7 design parameters: risk category, Ie, SDC and the ELF procedure.

Ported from codeSpectra and checked against the ASCE 7-10, 7-16 and 7-22 PDFs:

* Table 1.5-2: seismic importance factor ``Ie`` (I, II: 1.00; III: 1.25; IV: 1.50).
* Section 11.6, Tables 11.6-1 / 11.6-2: seismic design category.
* Section 12.8 (equivalent lateral force): ``Ta = Ct hn^x`` (Table 12.8-2), the
  upper limit ``Cu Ta`` (Table 12.8-1), ``Cs`` with its caps and floors,
  ``V = Cs W`` and the vertical distribution ``Fx = Cvx V``.

ASCE 7-22 changed Section 12.8.1.1: **Method 1** takes ``Cs = Sa(T) / (R / Ie)``
from the multi-period spectrum of Section 11.4.5.1 (or Chapter 21), using the
maximum ``Sa`` where ``T`` is below the period of that maximum; **Method 2** is
the classical ``SDS / (R / Ie)`` with the ``SD1`` caps.  Method 1 is not allowed
where Exception 2 of Section 11.4.5 applies (two-period spectrum).  The floors
apply to both.  7-10 and 7-16 have only the classical form.  Equation numbers
in the results follow the edition.

Section 12.8.2.1 also permits ``Ta = 0.1 N`` for moment frames of up to 12
stories (:func:`approximate_period_stories`) and ``Ta = Cq hn / sqrt(Cw)`` for
masonry or concrete shear-wall structures up to 120 ft (:func:`shear_wall_period`);
pass either result to :func:`base_shear` as ``ta``.  ``max_sds_12_8_1_3`` applies
the optional ``SDS = 1.0`` (not below 70 % of ``SDS``) of Section 12.8.1.3 in
7-16 / 7-22, and ``table_11_6_1_only`` the Section 11.6 exception that assigns
the SDC from Table 11.6-1 alone.  The conditions those provisions place on
irregularity, redundancy and diaphragms cannot be checked from these inputs:
passing the flag asserts them.

``Cq``: 7-16 and 7-22 print ``0.0019 ft (0.00058 m)``.  ``0.0019 s/ft`` is
``0.0062 s/m`` (the NEC-SE-DS value), so the printed SI figure is not used:
heights are converted to ft and the ft coefficient applied.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from ..code_spectrum.base import CodeSpectrumModel

__all__ = [
    "STRUCTURE_TYPES",
    "ELFResult",
    "SDCResult",
    "approximate_period",
    "approximate_period_stories",
    "base_shear",
    "importance_factor",
    "seismic_design_category",
    "seismic_response_coefficient",
    "shear_wall_period",
    "upper_limit_coefficient",
    "vertical_distribution",
    "vertical_distribution_exponent",
]

Edition = Literal["7-10", "7-16", "7-22"]
_EDITIONS = ("7-10", "7-16", "7-22")

#: Table 1.5-2, seismic importance factor Ie by risk category.
_IE = {"I": 1.00, "II": 1.00, "III": 1.25, "IV": 1.50}

#: Table 12.8-2: (Ct with hn in m, Ct with hn in ft, x).  Same in 7-10, 7-16, 7-22.
STRUCTURE_TYPES: dict[str, tuple[float, float, float]] = {
    "steel_moment_frame": (0.0724, 0.028, 0.8),
    "concrete_moment_frame": (0.0466, 0.016, 0.9),
    "steel_eccentrically_braced_frame": (0.0731, 0.03, 0.75),
    "steel_buckling_restrained_braced_frame": (0.0731, 0.03, 0.75),
    "other": (0.0488, 0.02, 0.75),
}

#: Table 12.8-1: SD1 breakpoints and Cu (clamped below 0.1 and above 0.4).
_CU_SD1 = (0.1, 0.15, 0.2, 0.3, 0.4)
_CU = (1.7, 1.6, 1.5, 1.4, 1.4)

#: Tables 11.6-1 / 11.6-2: upper bounds and SDC for (RC I-III, RC IV).
_Bands = tuple[tuple[float, str, str], ...]
_SDS_BANDS: _Bands = ((0.167, "A", "A"), (0.33, "B", "C"), (0.50, "C", "D"), (np.inf, "D", "D"))
_SD1_BANDS: _Bands = ((0.067, "A", "A"), (0.133, "B", "C"), (0.20, "C", "D"), (np.inf, "D", "D"))

#: Equation numbers per edition (7-22 renumbered them when it added Method 1).
_EQ = {
    "7-10": {"V": "12.8-1", "method1": None, "method2": "12.8-2", "cap": "12.8-3",
             "cap_TL": "12.8-4", "min": "12.8-5", "min_S1": "12.8-6", "Ta": "12.8-7",
             "Ta_N": "12.8-8", "Ta_wall": "12.8-9", "Cw": "12.8-10"},
    "7-16": {"V": "12.8-1", "method1": None, "method2": "12.8-2", "cap": "12.8-3",
             "cap_TL": "12.8-4", "min": "12.8-5", "min_S1": "12.8-6", "Ta": "12.8-7",
             "Ta_N": "12.8-8", "Ta_wall": "12.8-9", "Cw": "12.8-10"},
    "7-22": {"V": "12.8-1", "method1": "12.8-2", "method2": "12.8-3", "cap": "12.8-4",
             "cap_TL": "12.8-5", "min": "12.8-6", "min_S1": "12.8-7", "Ta": "12.8-8",
             "Ta_N": "12.8-9", "Ta_wall": "12.8-10", "Cw": "12.8-11"},
}

#: Section 12.8.2.1: ``Ta = 0.1 N`` is for frames that are entirely moment frames.
_MOMENT_FRAMES = ("steel_moment_frame", "concrete_moment_frame")
#: Section 12.8.2.1: ``Cq`` [s/ft] of the shear-wall formula; limit hn <= 120 ft.
_CQ_FT = 0.0019
_FT = 0.3048
_WALL_HN_MAX_FT = 120.0


def _edition(edition: str) -> Edition:
    e = str(edition).replace("ASCE", "").replace(" ", "").lstrip("7").lstrip("-")
    e = f"7-{e}"
    if e not in _EDITIONS:
        raise ValueError(f"edition must be one of {', '.join(_EDITIONS)}; got {edition!r}")
    return e  # type: ignore[return-value]


def _risk(risk_category: str) -> str:
    rc = str(risk_category).strip().upper()
    if rc not in _IE:
        raise ValueError(f"risk_category must be 'I', 'II', 'III' or 'IV'; got {risk_category!r}")
    return rc


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not (np.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a finite value > 0, got {value!r}")
    return v


def importance_factor(risk_category: str) -> float:
    """Seismic importance factor ``Ie`` (Table 1.5-2)."""
    return _IE[_risk(risk_category)]


@dataclass(frozen=True)
class SDCResult:
    """Seismic design category (Section 11.6) and how it was reached."""

    sdc: str
    from_sds: str
    from_sd1: str
    risk_category: str
    s1_override: bool | None
    """True if ``S1 >= 0.75`` set E/F; None if ``S1`` was not given (not checked)."""
    table_11_6_1_only: bool = False
    """True if the Section 11.6 exception set the SDC from Table 11.6-1 alone."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seismic_design_category(
    sds: float,
    sd1: float,
    *,
    s1: float | None,
    risk_category: str = "II",
    table_11_6_1_only: bool = False,
    ta: float | None = None,
    t_drift: float | None = None,
    ts: float | None = None,
) -> SDCResult:
    """Seismic design category per Section 11.6 (7-10, 7-16 and 7-22 alike).

    Where ``S1 >= 0.75``: E for Risk Category I-III, F for IV.  Otherwise the
    more severe of Table 11.6-1 (``SDS``) and Table 11.6-2 (``SD1``).  ``s1`` is
    keyword-only and required: pass ``None`` only where it is truly unknown,
    and the result records that the ``S1`` override was not checked.

    ``table_11_6_1_only=True`` applies the Section 11.6 exception (``S1 < 0.75``
    only): the SDC comes from Table 11.6-1 alone.  ``ta`` (Section 12.8.2.1),
    ``t_drift`` (the period used for story drift) and ``ts`` are then required
    and checked against items 1 and 2 (``ta < 0.8 Ts``, ``t_drift < Ts``, the
    larger of the two orthogonal directions).  Items 3 (``Cs`` by Eq. 12.8-2 of
    the edition) and 4 (rigid diaphragms, or vertical elements at most 40 ft /
    12.192 m apart) are asserted by passing the flag.
    """
    rc = _risk(risk_category)
    if not (sds >= 0.0 and sd1 >= 0.0):
        raise ValueError("SDS and SD1 must be >= 0")
    iv = rc == "IV"
    a = next(row[2] if iv else row[1] for row in _SDS_BANDS if sds < row[0])
    b = next(row[2] if iv else row[1] for row in _SD1_BANDS if sd1 < row[0])
    worst = max(a, b)  # "A" < "B" < ... alphabetically
    if s1 is None:
        return SDCResult(worst, a, b, rc, None)
    if s1 >= 0.75:
        return SDCResult("F" if rc == "IV" else "E", a, b, rc, True)
    if table_11_6_1_only:
        _check_table_11_6_1_only(ta, t_drift, ts)
        return SDCResult(a, a, b, rc, False, True)
    return SDCResult(worst, a, b, rc, False)


def _check_table_11_6_1_only(ta: float | None, t_drift: float | None,
                             ts: float | None) -> None:
    if ta is None or t_drift is None or ts is None:
        raise ValueError("table_11_6_1_only needs ta, t_drift and ts (Section 11.6 items 1-2)")
    ta, t_drift, ts = _positive("ta", ta), _positive("t_drift", t_drift), _positive("ts", ts)
    if not ta < 0.8 * ts:
        raise ValueError(f"Section 11.6 item 1: Ta = {ta:g} s is not below 0.8 Ts = "
                         f"{0.8 * ts:g} s")
    if not t_drift < ts:
        raise ValueError(f"Section 11.6 item 2: the drift period {t_drift:g} s is not "
                         f"below Ts = {ts:g} s")


def approximate_period(hn: float, structure_type: str = "other", *, metric: bool = True) -> float:
    """``Ta = Ct hn^x`` (Section 12.8.2.1), ``hn`` in m (``metric``) or ft."""
    hn = _positive("hn", hn)
    if structure_type not in STRUCTURE_TYPES:
        raise ValueError(f"structure_type must be one of {sorted(STRUCTURE_TYPES)}; "
                         f"got {structure_type!r}")
    ct_m, ct_ft, x = STRUCTURE_TYPES[structure_type]
    return float((ct_m if metric else ct_ft) * hn**x)


def approximate_period_stories(
    n_stories: int,
    structure_type: str = "concrete_moment_frame",
    *,
    average_story_height: float,
    metric: bool = True,
) -> float:
    """``Ta = 0.1 N`` (Section 12.8.2.1; Eq. 12.8-8 in 7-10 / 7-16, 12.8-9 in 7-22).

    Permitted only where the seismic force-resisting system is entirely concrete
    or steel moment frames, ``N <= 12`` stories above the base and the average
    story height is at least 3 m (10 ft); each condition is checked.
    """
    if structure_type not in _MOMENT_FRAMES:
        raise ValueError("Ta = 0.1 N needs a system entirely of moment frames "
                         f"({', '.join(_MOMENT_FRAMES)}); got {structure_type!r}")
    if int(n_stories) != n_stories or not 1 <= n_stories <= 12:
        raise ValueError(f"Ta = 0.1 N is for 1 to 12 stories above the base; got {n_stories!r}")
    h = _positive("average_story_height", average_story_height)
    if h < (3.0 if metric else 10.0):
        raise ValueError(f"Ta = 0.1 N needs an average story height of at least "
                         f"{'3 m' if metric else '10 ft'}; got {h:g}")
    return 0.1 * int(n_stories)


def shear_wall_period(
    hn: float,
    base_area: float,
    walls: Iterable[tuple[float, ...]],
    *,
    edition: str = "7-16",
    metric: bool = True,
) -> float:
    """``Ta = Cq hn / sqrt(Cw)`` for masonry or concrete shear-wall structures.

    Section 12.8.2.1 (Eqs. 12.8-9 / 12.8-10 in 7-10 and 7-16, 12.8-10 / 12.8-11
    in 7-22), for ``hn <= 120 ft (36.6 m)``.  ``walls`` holds, per shear wall
    effective in the direction considered, ``(Ai, Di)``: web area and length.
    ``base_area`` is ``AB``.  Lengths in m and areas in m^2 (``metric``), or ft
    and ft^2.

    ``Cw = 100 / AB * sum(Ai / (1 + 0.83 (h / Di)^2))`` with ``h = hn`` in 7-16 and
    7-22.  7-10 prints ``h_i`` there without defining it; give each wall as
    ``(Ai, Di, hi)`` to use its own height, otherwise ``hn`` is taken.
    ``Cq = 0.0019 s/ft`` (see the module notes on the printed SI value).
    """
    ed = _edition(edition)
    hn = _positive("hn", hn)
    ab = _positive("base_area", base_area)
    hn_ft = hn / _FT if metric else hn
    if hn_ft > _WALL_HN_MAX_FT + 1e-9:
        raise ValueError("the shear-wall period is for hn <= 120 ft (36.6 m); "
                         f"got {hn:g} {'m' if metric else 'ft'}")
    total, count = 0.0, 0
    for i, wall in enumerate(walls):
        if len(wall) not in (2, 3) or (len(wall) == 3 and ed != "7-10"):
            raise ValueError(f"walls[{i}]: expected (Ai, Di)"
                             + (" or (Ai, Di, hi)" if ed == "7-10" else ""))
        ai, di = _positive(f"walls[{i}] Ai", wall[0]), _positive(f"walls[{i}] Di", wall[1])
        h = _positive(f"walls[{i}] hi", wall[2]) if len(wall) == 3 else hn
        total += ai / (1.0 + 0.83 * (h / di) ** 2)
        count += 1
    if count == 0:
        raise ValueError("walls: give at least one shear wall")
    cw = 100.0 / ab * total            # dimensionless: Ai / AB, h / Di
    return float(_CQ_FT * hn_ft / np.sqrt(cw))


def upper_limit_coefficient(sd1: float) -> float:
    """``Cu`` of Table 12.8-1, linear between the tabulated ``SD1`` values."""
    return float(np.interp(sd1, _CU_SD1, _CU))


def seismic_response_coefficient(
    *,
    sds: float,
    sd1: float,
    s1: float,
    T: float,
    tl: float,
    R: float,
    Ie: float = 1.0,
    edition: str = "7-16",
    spectrum: CodeSpectrumModel | None = None,
    t_peak_max: float = 5.0,
) -> tuple[float, dict[str, Any]]:
    """``Cs`` per Section 12.8.1.1 with every limit; returns ``(Cs, breakdown)``.

    ``spectrum`` (7-22 only) selects Method 1: ``Cs = Sa(T) / (R / Ie)`` from that
    design spectrum (the Section 11.4.5.1 multi-period model or a Chapter 21
    site-specific one), with ``Sa = max Sa`` where ``T`` is below the period of
    the maximum (searched on ``0..t_peak_max``).  Without it, Method 2.
    ``breakdown`` maps each equation (edition numbering) to its value and names
    the governing one.
    """
    ed = _edition(edition)
    eq = _EQ[ed]
    T, tl, R, Ie = (_positive(n, v) for n, v in (("T", T), ("tl", tl), ("R", R), ("Ie", Ie)))
    ratio = R / Ie
    out: dict[str, Any] = {"edition": ed}
    if spectrum is not None:
        if ed != "7-22":
            raise ValueError("Method 1 (Cs from the spectrum) exists only in ASCE 7-22")
        grid = np.unique(np.concatenate([np.linspace(0.0, t_peak_max, 2001),
                                         spectrum.default_periods(t_peak_max, 50)]))
        sa_grid = spectrum.sa(grid)
        i_max = int(np.nanargmax(sa_grid))
        t_at_max = float(grid[i_max])
        sa = float(sa_grid[i_max]) if T < t_at_max else float(spectrum.sa(T)[0])
        cs = sa / ratio
        out.update({"method": 1, f"Cs_eq_{eq['method1']}": cs, "Sa(T)": sa,
                    "T_at_max_Sa": t_at_max})
    else:
        cs_base = sds / ratio
        cap_eq = eq["cap"] if T <= tl else eq["cap_TL"]
        cap = sd1 / (T * ratio) if T <= tl else sd1 * tl / (T**2 * ratio)
        cs = min(cs_base, cap)
        out.update({"method": 2 if ed == "7-22" else None,
                    f"Cs_eq_{eq['method2']}": cs_base, f"Cs_cap_eq_{cap_eq}": cap})
    floor = max(0.044 * sds * Ie, 0.01)
    floor_s1 = 0.5 * s1 / ratio if s1 >= 0.6 else 0.0
    out[f"Cs_min_eq_{eq['min']}"] = floor
    out[f"Cs_min_eq_{eq['min_S1']}"] = floor_s1
    governing = min(out[k] for k in out if k.startswith("Cs_") and "min" not in k)
    Cs = max(governing, floor, floor_s1)
    gov = next(k for k in out if k.startswith("Cs_") and np.isclose(out[k], Cs, rtol=0, atol=1e-15))
    out["governing"] = gov.replace("Cs_", "").replace("_", " ")
    out["Cs"] = Cs
    return Cs, out


def vertical_distribution_exponent(T: float, *, interpolate: bool = True) -> float:
    """``k`` (Section 12.8.3): 1 for ``T <= 0.5 s``, 2 for ``T >= 2.5 s``; between,
    linear (``interpolate``, default) or 2 (the other option the code allows)."""
    if T <= 0.5:
        return 1.0
    if T >= 2.5 or not interpolate:
        return 2.0
    return 1.0 + (T - 0.5) / 2.0


def vertical_distribution(
    V: float, weights: Iterable[float], heights: Iterable[float], k: float
) -> tuple[np.ndarray, np.ndarray]:
    """``Fx = Cvx V`` with ``Cvx = wx hx^k / sum(wi hi^k)``, and story shears ``Vx``.

    Returns ``(Fx, Vx)`` in the input order; ``Vx`` sums the forces at and above
    each level (by height).  Shared by the NEC procedure (same expression).
    """
    w = np.asarray(weights, dtype=float).ravel()
    h = np.asarray(heights, dtype=float).ravel()
    if w.shape != h.shape or w.size == 0:
        raise ValueError(f"weights and heights must be non-empty and equal length; "
                         f"got {w.size} and {h.size}")
    if np.any(w < 0.0) or np.any(h <= 0.0):
        raise ValueError("weights must be >= 0 and heights > 0 (above the base)")
    num = w * h**k
    if not num.sum() > 0.0:
        raise ValueError("sum of w h^k is zero")
    Fx = V * num / num.sum()
    order = np.argsort(-h, kind="stable")
    Vx = np.empty_like(Fx)
    Vx[order] = np.cumsum(Fx[order])
    return Fx, Vx


@dataclass(frozen=True)
class ELFResult:
    """Outcome of an ASCE 7 Section 12.8 ELF calculation (forces in W's units)."""

    edition: str
    V: float
    Cs: float
    T: float
    Ta: float | None
    Cu: float | None
    k: float
    W: float
    R: float
    Ie: float
    breakdown: dict[str, Any]
    Fx: np.ndarray | None = field(default=None, repr=False)
    Vx: np.ndarray | None = field(default=None, repr=False)

    @property
    def governing(self) -> str:
        return self.breakdown["governing"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("Fx", "Vx"):
            if d[key] is not None:
                d[key] = d[key].tolist()
        return d


def base_shear(
    *,
    W: float,
    R: float,
    sds: float,
    sd1: float,
    s1: float,
    tl: float,
    risk_category: str = "II",
    Ie: float | None = None,
    T: float | None = None,
    hn: float | None = None,
    structure_type: str = "other",
    metric: bool = True,
    apply_upper_limit: bool = True,
    edition: str = "7-16",
    spectrum: CodeSpectrumModel | None = None,
    weights: Iterable[float] | None = None,
    heights: Iterable[float] | None = None,
    k_interpolate: bool = True,
    ta: float | None = None,
    max_sds_12_8_1_3: bool = False,
    n_stories: int | None = None,
    site_class: str | None = None,
) -> ELFResult:
    """Seismic base shear ``V = Cs W`` (Section 12.8.1) and, optionally, ``Fx`` / ``Vx``.

    Parameters
    ----------
    W, R
        Effective seismic weight (Section 12.7.2) and Table 12.2-1 ``R``.
    sds, sd1, s1, tl
        Design parameters and mapped ``S1`` (for the ``S1 >= 0.6`` floor).
    risk_category, Ie
        ``Ie`` defaults to Table 1.5-2 for ``risk_category``.
    T, hn
        Analysis period and/or structural height.  With ``hn``, ``Ta`` is
        computed and ``T`` (if given) is capped at ``Cu Ta`` (Section 12.8.2)
        unless ``apply_upper_limit=False``; without ``T``, ``Ta`` is used.
    edition
        ``"7-10"``, ``"7-16"`` or ``"7-22"`` (equation numbering; Method 1).
    spectrum
        7-22 Method 1: the design spectrum ``Sa(T)`` comes from (normally an
        :class:`~apeQuake.code_spectrum.ASCE7_22MultiPeriodSpectrum`).
    weights, heights
        Story weights and heights above the base, for ``Fx`` and ``Vx``.
    ta
        An approximate period from another Section 12.8.2.1 formula
        (:func:`approximate_period_stories`, :func:`shear_wall_period`), used
        instead of ``Ct hn^x``; give ``ta`` or ``hn``, not both.
    max_sds_12_8_1_3, n_stories, site_class
        Section 12.8.1.3 (7-16, 7-22): ``Cs`` computed with ``SDS = 1.0`` but not
        less than 0.7 ``SDS``.  Checked here: ``T <= 0.5 s``, ``n_stories <= 5``,
        ``site_class`` not E or F, Risk Category I or II and, in 7-22,
        ``R >= 3``.  The flag asserts the rest (no Section 12.3.2
        irregularities; ``rho = 1.0`` permitted by Section 12.3.4.2).
    """
    W = _positive("W", W)
    Ie = importance_factor(risk_category) if Ie is None else _positive("Ie", Ie)
    if ta is not None and hn is not None:
        raise ValueError("give hn (Ta = Ct hn^x) or ta (another Section 12.8.2.1 period), "
                         "not both")
    if T is None and hn is None and ta is None:
        raise ValueError("give the period T, the height hn (or ta), or both")
    if ta is not None:
        Ta: float | None = _positive("ta", ta)
    else:
        Ta = approximate_period(hn, structure_type, metric=metric) if hn is not None else None
    Cu = upper_limit_coefficient(sd1) if Ta is not None else None
    if T is None:
        T_used = float(Ta)  # type: ignore[arg-type]
    else:
        T_used = _positive("T", T)
        if Ta is not None and apply_upper_limit:
            T_used = min(T_used, Cu * Ta)  # type: ignore[operator]
    sds_cs = sds
    if max_sds_12_8_1_3:
        sds_cs = _sds_12_8_1_3(sds, T=T_used, R=R, edition=edition, spectrum=spectrum,
                               n_stories=n_stories, site_class=site_class,
                               risk_category=risk_category)
    Cs, bd = seismic_response_coefficient(sds=sds_cs, sd1=sd1, s1=s1, T=T_used, tl=tl, R=R,
                                          Ie=Ie, edition=edition, spectrum=spectrum)
    if max_sds_12_8_1_3:
        bd["SDS_12.8.1.3"] = sds_cs
    V = Cs * W
    k = vertical_distribution_exponent(T_used, interpolate=k_interpolate)
    Fx = Vx = None
    if (weights is None) != (heights is None):
        raise ValueError("give both weights and heights, or neither")
    if weights is not None:
        Fx, Vx = vertical_distribution(V, weights, heights, k)  # type: ignore[arg-type]
    return ELFResult(bd["edition"], V, Cs, T_used, Ta, Cu, k, W, float(R), Ie, bd, Fx, Vx)


def _sds_12_8_1_3(sds: float, *, T: float, R: float, edition: str,
                  spectrum: CodeSpectrumModel | None, n_stories: int | None,
                  site_class: str | None, risk_category: str) -> float:
    """``SDS`` for ``Cs`` under Section 12.8.1.3, after checking its criteria."""
    ed = _edition(edition)
    if ed == "7-10":
        raise ValueError("ASCE 7-10 Section 12.8.1.3 caps Ss at 1.5 (it changes SDS through "
                         "Fa), not SDS; build the spectrum with Ss = 1.5 instead")
    if spectrum is not None:
        raise ValueError("Section 12.8.1.3 caps SDS in Method 2; it does not apply to Cs "
                         "from a spectrum (Method 1)")
    if n_stories is None or site_class is None:
        raise ValueError("max_sds_12_8_1_3 needs n_stories and site_class")
    checks = (
        (T <= 0.5, f"item 3: T = {T:g} s exceeds 0.5 s"),
        (int(n_stories) == n_stories and 1 <= n_stories <= 5,
         f"item 2: {n_stories!r} stories (at most 5)"),
        (str(site_class).strip().upper() not in ("E", "F"),
         f"item 5: Site Class {site_class!r}"),
        (_risk(risk_category) in ("I", "II"), f"item 6: Risk Category {risk_category!r}"),
        (ed != "7-22" or R >= 3.0, f"item 7 (7-22): R = {R:g} is below 3"),
    )
    for ok, why in checks:
        if not ok:
            raise ValueError(f"Section 12.8.1.3 does not apply, {why}")
    return min(sds, max(1.0, 0.7 * sds))

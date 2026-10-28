"""NEC-SE-DS 2015 force-based design (DBF): importance, period, base shear, Fx.

Ported from codeSpectra and checked against NEC-SE-DS (peligro sismico):

* Tabla 6 (4.1): importance coefficient ``I`` (esencial 1.5, especial 1.3,
  otra 1.0).
* 6.3.3 a: Metodo 1 ``Ta = Ct hn^alpha`` (``hn`` in m); 6.3.3 b: a Metodo 2
  (Rayleigh) period may not exceed 1.30 x the Metodo 1 value.
* 6.3.2: ``V = I Sa(Ta) W / (R phiP phiE)``, with ``Sa`` the 3.3.1 spectrum and
  the plateau extended to T = 0 (:meth:`NECSpectrum.reduced_sa`).
* 6.3.5: ``Fx = wx hx^k V / sum(wi hi^k)`` with ``k = 1`` (``T <= 0.5``),
  ``0.75 + 0.50 T`` (``0.5 < T <= 2.5``), ``2`` (``T > 2.5``).

* 6.3.3 a, alternatively for reinforced-concrete or masonry wall structures
  (``alpha = 1``): ``Ct = 0.0062 / sqrt(Cw)``,
  ``Cw = 100 / AB * sum((hn / hwi)^2 Awi / (1 + 0.83 (hwi / lwi)^2))``
  (:func:`wall_period`; NEC-SE-DS p. 62).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .asce7 import _positive, vertical_distribution

if TYPE_CHECKING:
    from ..code_spectrum.codes.nec import NECSpectrum

__all__ = [
    "IMPORTANCE",
    "METHOD_2_MAX_RATIO",
    "PERIOD_COEFFICIENTS",
    "NECELFResult",
    "approximate_period",
    "base_shear",
    "importance_factor",
    "limit_period_method_2",
    "vertical_distribution_exponent",
    "wall_period",
]

#: Tabla 6: importance coefficient I by category.
IMPORTANCE: dict[str, float] = {"esencial": 1.5, "especial": 1.3, "otra": 1.0}

#: 6.3.3 a: (Ct, alpha) by structure type, hn in metres.
PERIOD_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "acero_sin_arriostramientos": (0.072, 0.80),
    "acero_con_arriostramientos": (0.073, 0.75),
    "hormigon_porticos": (0.055, 0.90),   # sin muros estructurales ni diagonales
    "hormigon_con_muros": (0.055, 0.75),  # muros, diagonales, mamposteria estructural
}

#: 6.3.3 b: Metodo 2 may exceed Metodo 1 by at most 30 %.
METHOD_2_MAX_RATIO = 1.30


def importance_factor(category: str) -> float:
    """``I`` from Tabla 6: ``"esencial"``, ``"especial"`` or ``"otra"``."""
    key = str(category).strip().lower()
    if key not in IMPORTANCE:
        raise ValueError(f"category must be one of {sorted(IMPORTANCE)}; got {category!r}")
    return IMPORTANCE[key]


def approximate_period(hn: float, structure_type: str = "hormigon_porticos") -> float:
    """Metodo 1 ``Ta = Ct hn^alpha`` (6.3.3 a), ``hn`` in metres."""
    hn = _positive("hn", hn)
    if structure_type not in PERIOD_COEFFICIENTS:
        raise ValueError(f"structure_type must be one of {sorted(PERIOD_COEFFICIENTS)}; "
                         f"got {structure_type!r}")
    ct, alpha = PERIOD_COEFFICIENTS[structure_type]
    return float(ct * hn**alpha)


def wall_period(hn: float, base_area: float, walls: Iterable[tuple[float, float, float]]) -> float:
    """Metodo 1 alternative for wall structures: ``T = Ct hn`` with ``Ct = 0.0062 / sqrt(Cw)``.

    ``walls`` holds ``(Awi, lwi, hwi)`` per wall resisting the seismic forces in
    the direction studied: minimum shear area at the first level [m^2], length
    [m] and height from the base [m].  ``base_area`` is ``AB`` [m^2], ``hn`` [m].
    """
    hn = _positive("hn", hn)
    ab = _positive("base_area", base_area)
    total, count = 0.0, 0
    for i, wall in enumerate(walls):
        if len(wall) != 3:
            raise ValueError(f"walls[{i}]: expected (Awi, lwi, hwi)")
        aw, lw, hw = (_positive(f"walls[{i}] {n}", v) for n, v in zip(("Awi", "lwi", "hwi"), wall))
        total += (hn / hw) ** 2 * aw / (1.0 + 0.83 * (hw / lw) ** 2)
        count += 1
    if count == 0:
        raise ValueError("walls: give at least one wall")
    cw = 100.0 / ab * total
    return float(0.0062 / np.sqrt(cw) * hn)


def limit_period_method_2(T_method_2: float, Ta_method_1: float) -> float:
    """Cap a Metodo 2 period at 1.30 x the Metodo 1 value (6.3.3 b)."""
    return min(_positive("T_method_2", T_method_2),
               METHOD_2_MAX_RATIO * _positive("Ta_method_1", Ta_method_1))


def vertical_distribution_exponent(T: float) -> float:
    """``k`` of 6.3.5."""
    if T <= 0.5:
        return 1.0
    if T > 2.5:
        return 2.0
    return 0.75 + 0.50 * T


@dataclass(frozen=True)
class NECELFResult:
    """Outcome of a NEC-SE-DS 6.3 base shear (forces in W's units)."""

    V: float
    Sa_Ta: float
    Ta: float
    Ta_method_1: float | None
    k: float
    W: float
    R: float
    phi_p: float
    phi_e: float
    I: float
    Fx: np.ndarray | None = field(default=None, repr=False)
    Vx: np.ndarray | None = field(default=None, repr=False)

    @property
    def shear_coefficient(self) -> float:
        """``V / W = I Sa(Ta) / (R phiP phiE)``."""
        return self.V / self.W

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("Fx", "Vx"):
            if d[key] is not None:
                d[key] = d[key].tolist()
        d["V/W"] = self.shear_coefficient
        return d


def base_shear(
    spectrum: NECSpectrum,
    *,
    W: float,
    R: float,
    I: float = 1.0,
    phi_p: float = 1.0,
    phi_e: float = 1.0,
    T: float | None = None,
    hn: float | None = None,
    structure_type: str = "hormigon_porticos",
    weights: Iterable[float] | None = None,
    heights: Iterable[float] | None = None,
    ta: float | None = None,
) -> NECELFResult:
    """Design base shear ``V = I Sa(Ta) W / (R phiP phiE)`` (6.3.2).

    ``spectrum`` is the **elastic** horizontal :class:`NECSpectrum`; the
    reduction is applied here.  Give ``T`` (Metodo 2), ``hn`` (Metodo 1) or
    both (``T`` then capped at 1.30 x Metodo 1).  ``I`` is Tabla 6
    (:func:`importance_factor`).  ``ta`` is a Metodo 1 period from the wall
    formula (:func:`wall_period`), used instead of ``Ct hn^alpha``.
    """
    if getattr(spectrum, "component", "horizontal") != "horizontal":
        raise ValueError("base_shear needs the horizontal NEC spectrum")
    W = _positive("W", W)
    if ta is not None and hn is not None:
        raise ValueError("give hn (Ct hn^alpha) or ta (the wall formula), not both")
    if T is None and hn is None and ta is None:
        raise ValueError("give the period T, the height hn (or ta), or both")
    if ta is not None:
        Ta1: float | None = _positive("ta", ta)
    else:
        Ta1 = approximate_period(hn, structure_type) if hn is not None else None
    if T is None:
        Ta = float(Ta1)  # type: ignore[arg-type]
    else:
        Ta = limit_period_method_2(T, Ta1) if Ta1 is not None else _positive("T", T)
    cs = float(spectrum.reduced_sa(Ta, r_factor=R, phi_p=phi_p, phi_e=phi_e, importance=I)[0])
    sa = cs * R * phi_p * phi_e / I
    V = cs * W
    k = vertical_distribution_exponent(Ta)
    Fx = Vx = None
    if (weights is None) != (heights is None):
        raise ValueError("give both weights and heights, or neither")
    if weights is not None:
        Fx, Vx = vertical_distribution(V, weights, heights, k)  # type: ignore[arg-type]
    return NECELFResult(V, sa, Ta, Ta1, k, W, float(R), float(phi_p), float(phi_e), float(I),
                        Fx, Vx)

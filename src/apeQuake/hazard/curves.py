"""Hazard-curve math: return period, exceedance probability and power-law fits.

Conventions: ``rate`` is the mean annual rate of exceedance lambda [1/yr], ``tr`` the
return period 1/lambda [yr], ``poe`` the probability of exceedance in ``t`` years under a
Poisson model, poe = 1 - exp(-lambda t).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def tr_from_poe(poe: ArrayLike, t: float = 50.0) -> np.ndarray:
    """Return period [yr] for a probability of exceedance ``poe`` in ``t`` years.

    >>> round(float(tr_from_poe(0.10, 50)))   # 10 % in 50 yr
    475
    """
    poe = np.asarray(poe, float)
    return -t / np.log1p(-poe)


def poe_from_tr(tr: ArrayLike, t: float = 50.0) -> np.ndarray:
    """Probability of exceedance in ``t`` years for a return period ``tr`` [yr]."""
    return -np.expm1(-t / np.asarray(tr, float))


def powerlaw_fit(sa1: ArrayLike, tr1: float, sa2: ArrayLike, tr2: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Fit lambda(Sa) = k0 * Sa**(-k) through two points (Cornell 1968).

    The two points are (sa1, 1/tr1) and (sa2, 1/tr2); arrays are fitted element-wise
    (e.g. one fit per spectral period). Returns (k0, k). The fit is exact at both points
    and is a straight line in log-log space, so it is the natural interpolant between two
    published return periods. Away from them it is an extrapolation: real hazard curves
    bend downward (k grows with Sa), so the fit tends to overestimate Sa at long TR.
    """
    sa1, sa2 = np.asarray(sa1, float), np.asarray(sa2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log(tr2 / tr1) / np.log(sa2 / sa1)
        k0 = (1.0 / tr1) * sa1 ** k
    return k0, k


def powerlaw_sa(tr: ArrayLike, k0: ArrayLike, k: ArrayLike) -> np.ndarray:
    """Spectral acceleration with return period ``tr`` on lambda = k0 Sa^-k."""
    return (np.asarray(k0, float) * np.asarray(tr, float)) ** (1.0 / np.asarray(k, float))


def powerlaw_rate(sa: ArrayLike, k0: ArrayLike, k: ArrayLike) -> np.ndarray:
    """Annual rate of exceedance of ``sa`` on lambda = k0 Sa^-k."""
    return np.asarray(k0, float) * np.asarray(sa, float) ** (-np.asarray(k, float))


def loglog_interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> np.ndarray:
    """Interpolate linearly in log-log space; NaN outside [min(xp), max(xp)].

    ``xp`` may be decreasing (hazard curves are tabulated as rate vs. Sa, and inverting
    them gives a decreasing abscissa).
    """
    x, xp, fp = (np.asarray(v, float) for v in (x, xp, fp))
    order = np.argsort(xp)
    lx, lxp, lfp = np.log(x), np.log(xp[order]), np.log(fp[order])
    out = np.exp(np.interp(lx, lxp, lfp))
    out = np.where((lx < lxp[0]) | (lx > lxp[-1]), np.nan, out)
    return out

"""Time integration of acceleration and least-squares baseline correction.

``integrate`` is exact for an acceleration that varies linearly between samples:

    v[i+1] = v[i] + dt (a[i] + a[i+1]) / 2
    d[i+1] = d[i] + v[i] dt + dt^2 (2 a[i] + a[i+1]) / 6

``baseline_correct`` removes a polynomial drift fitted to the velocity by least
squares.  The velocity drift is ``sum_k c_k t^(k+1)`` (k = 0..order; no constant
term, since v(0) = 0), i.e. an acceleration offset ``c_0 + 2 c_1 t + ...`` of
degree ``order``.  Its derivative is subtracted from the acceleration and its
integral from the displacement, so the three stay consistent.  ``order=2`` is
EarthquakeSignal's ``BaselineCorrection`` (there solved with the continuous
least-squares integrals; here on the samples, which agrees to well under 1 %).
"""
from __future__ import annotations

import numpy as np


def integrate(acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and displacement from acceleration, zero initial conditions.

    Units follow the input: ``acc`` in u/s^2 gives velocity in u/s and
    displacement in u.
    """
    a = np.asarray(acc, dtype=float)
    if a.ndim != 1:
        raise ValueError("acc must be 1D.")
    if not dt > 0.0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    if a.size < 2:
        return np.zeros_like(a), np.zeros_like(a)
    vel = np.concatenate(([0.0], np.cumsum(0.5 * dt * (a[:-1] + a[1:]))))
    dd = vel[:-1] * dt + dt * dt * (2.0 * a[:-1] + a[1:]) / 6.0
    disp = np.concatenate(([0.0], np.cumsum(dd)))
    return vel, disp


def baseline_correct(
    acc: np.ndarray, dt: float, *, order: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares polynomial baseline correction (velocity fit).

    Parameters
    ----------
    acc : ndarray
        Acceleration samples (any unit).
    dt : float
        Time step [s].
    order : int
        Degree of the acceleration drift polynomial (0, 1, 2, ...).  Default 2.

    Returns
    -------
    acc, vel, disp : ndarray
        Corrected acceleration, velocity and displacement (input units, x s, x s^2).
    """
    if int(order) != order or order < 0:
        raise ValueError(f"order must be an integer >= 0, got {order}.")
    order = int(order)
    a = np.asarray(acc, dtype=float)
    vel, disp = integrate(a, dt)
    n = a.size
    if n < order + 2:
        raise ValueError(f"Need at least {order + 2} samples for an order-{order} correction.")

    t = np.arange(n, dtype=float) * dt
    T = t[-1]
    powers = np.arange(order + 1) + 1
    basis = (t / T)[:, None] ** powers[None, :]  # fit on [0, 1] for conditioning
    b, *_ = np.linalg.lstsq(basis, vel, rcond=None)
    c = b / T ** powers  # coefficients of t^(k+1)

    v_fit = np.zeros(n)
    a_fit = np.zeros(n)
    d_fit = np.zeros(n)
    for ck, p in zip(c, powers):
        v_fit += ck * t ** p
        a_fit += ck * p * t ** (p - 1)
        d_fit += ck * t ** (p + 1) / (p + 1)
    return a - a_fit, vel - v_fit, disp - d_fit

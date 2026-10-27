"""Acceleration integration, baseline correction, PGA/PGV/PGD and Arias intensity.

References:
  * closed-form velocity / displacement / Arias integral of a sine burst,
  * an exactly removable polynomial drift (velocity t, t^2, t^3),
  * EarthquakeSignal's BaselineCorrection (continuous least-squares closed form,
    re-implemented below independently of apeQuake.core.integration).
"""
import numpy as np
import pytest

from apeQuake import Record
from apeQuake.core.integration import baseline_correct, integrate
from apeQuake.core.units import G

DT = 0.005
A = 0.3            # m/s^2
F = 1.25           # Hz, 10 full cycles over 8 s
T_END = 8.0
t = np.arange(0.0, T_END + 0.5 * DT, DT)
W = 2.0 * np.pi * F
SINE = A * np.sin(W * t)


def test_integrate_matches_closed_form():
    vel, disp = integrate(SINE, DT)
    v_ref = A / W * (1.0 - np.cos(W * t))
    d_ref = A / W * t - A / W**2 * np.sin(W * t)
    # Trapezoid error on a sine is O((w dt)^2 / 12) ~ 1e-4 here.
    assert np.max(np.abs(vel - v_ref)) < 1e-3 * np.max(np.abs(v_ref))
    assert np.max(np.abs(disp - d_ref)) < 1e-3 * np.max(np.abs(d_ref))


def test_integrate_is_exact_for_linear_acceleration():
    a = 2.0 + 3.0 * t
    vel, disp = integrate(a, DT)
    np.testing.assert_allclose(vel, 2.0 * t + 1.5 * t**2, atol=1e-10)
    np.testing.assert_allclose(disp, t**2 + 0.5 * t**3, atol=1e-9)


def test_baseline_removes_polynomial_drift_exactly():
    drift = 0.02 - 0.01 * t + 0.003 * t**2
    acc, vel, disp = baseline_correct(drift, DT, order=2)
    assert np.max(np.abs(acc)) < 1e-6
    assert np.max(np.abs(vel)) < 1e-6
    assert np.max(np.abs(disp)) < 1e-6


def test_baseline_leaves_drift_free_part_and_bounds_displacement():
    # A ground motion that returns to rest, plus an acceleration offset.
    env = np.exp(-((t - 4.0) ** 2) / 0.5)
    clean = 0.5 * env * np.sin(2.0 * np.pi * 3.0 * t)
    acc, _, disp = baseline_correct(clean + 0.01, DT, order=2)
    _, d_raw = integrate(clean + 0.01, DT)
    assert np.max(np.abs(disp)) < 0.05 * np.max(np.abs(d_raw))
    assert np.max(np.abs(acc - clean)) < 0.02 * np.max(np.abs(clean))


def _earthquakesignal_baseline(a, dt):
    """EarthquakeSignal.core.base_line.BaselineCorrection, vectorized (units-free)."""
    n = len(a)
    time = np.arange(n) * dt
    vel = np.concatenate(([0.0], np.cumsum((a[:-1] + a[1:]) * dt / 2)))
    ti, ti1, vi, ai, ai1 = time[:-1], time[1:], vel[:-1], a[:-1], a[1:]
    d = ti1 - ti
    A1 = np.sum(0.5 * vi * d * (ti + ti1) + d**2 / 24 * (ai * (3 * ti + 5 * ti1) + ai1 * (ti + 3 * ti1)))
    A2 = np.sum(vi * d * (ti**2 + ti * ti1 + ti1**2) / 3 + d**2 / 60 * (
        ai * (4 * ti**2 + 7 * ti * ti1 + 9 * ti1**2) + ai1 * (ti**2 + 3 * ti * ti1 + 6 * ti1**2)))
    A3 = np.sum(vi * d * (ti**3 + ti**2 * ti1 + ti * ti1**2 + ti1**3) / 4 + d**2 / 120 * (
        ai * (5 * ti**3 + 9 * ti**2 * ti1 + 12 * ti * ti1**2 + 14 * ti1**3)
        + ai1 * (ti**3 + 3 * ti**2 * ti1 + 6 * ti * ti1**2 + 10 * ti1**3)))
    tT = time[-1]
    C0 = 300 * A1 / tT**3 - 900 * A2 / tT**4 + 630 * A3 / tT**5
    C1 = -900 * A1 / tT**4 + 2880 * A2 / tT**5 - 2100 * A3 / tT**6
    C2 = 630 * A1 / tT**5 - 2100 * A2 / tT**6 + 1575 * A3 / tT**7
    return a - (C0 + 2 * C1 * time + 3 * C2 * time**2)


def test_baseline_agrees_with_earthquakesignal_closed_form():
    rng = np.random.default_rng(7)
    env = np.exp(-((t - 3.0) ** 2) / 2.0)
    a = env * rng.standard_normal(t.size) + 0.02 + 0.004 * t
    ours, _, _ = baseline_correct(a, DT, order=2)
    ref = _earthquakesignal_baseline(a, DT)
    # Discrete vs continuous least squares: the fitted drifts agree to < 1 %.
    assert np.max(np.abs((a - ours) - (a - ref))) < 1e-2 * np.max(np.abs(a - ref))


def test_filter_baseline_correction_does_not_mutate_record():
    rec = Record(x=SINE + 0.05, y=SINE, dt=DT)
    before = rec.df.copy()
    out = rec.filter.baseline_correction()
    assert list(out.columns) == ["time", "X", "Y"]
    assert rec.df.equals(before)
    rec.intensity_measures.apply_baseline_correction(order=1)
    assert rec.intensity_measures.applied_filters == ["baseline_correction(order=1)"]
    assert rec.df.equals(before)


def test_pga_pgv_pgd():
    rec = Record(x=SINE, dt=DT)
    im = rec.intensity_measures
    assert np.isclose(im.pga("X"), A, rtol=1e-4)
    assert np.isclose(im.pgv("X"), 2.0 * A / W, rtol=1e-3)
    assert np.isclose(im.pgd("X"), A / W * T_END, rtol=1e-3)  # drift peak at the end
    t_v, v = im.velocity("X")
    assert t_v.shape == v.shape == rec.time.shape


def test_arias_intensity_closed_form_and_units():
    ia_ref = np.pi / (2.0 * G) * A**2 * T_END / 2.0
    rec = Record(x=SINE, dt=DT, units="m/s2")
    assert np.isclose(rec.intensity_measures.arias_intensity("X"), ia_ref, rtol=1e-6)
    # Same motion in g and in cm/s^2 gives the same Ia.
    in_g = Record(x=SINE / G, dt=DT, units="g").intensity_measures.arias_intensity("X")
    in_gal = Record(x=SINE * 100.0, dt=DT).intensity_measures.arias_intensity("X", units="gal")
    assert np.isclose(in_g, ia_ref, rtol=1e-6) and np.isclose(in_gal, ia_ref, rtol=1e-6)


def test_arias_needs_units():
    with pytest.raises(ValueError, match="physical units"):
        Record(x=SINE, dt=DT).intensity_measures.arias_intensity("X")


def test_summary_table():
    rec = Record(x=SINE, y=0.5 * SINE, dt=DT)
    s = rec.intensity_measures.summary()
    assert list(s.index) == ["X", "Y"]
    assert list(s.columns) == ["PGA", "PGV", "PGD", "D5-75", "D5-95"]
    assert "Ia" in rec.intensity_measures.summary(units="m/s2").columns
    assert np.isclose(s.loc["Y", "PGA"], 0.5 * s.loc["X", "PGA"])

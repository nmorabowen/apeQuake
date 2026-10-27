"""RotDnn spectra: geometric identities, brute-force rotation of the input,
Numba/Python agreement and the filter pipeline."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apeQuake import Record

DT = 0.01
t = np.arange(0.0, 20.0, DT)
ENV = np.exp(-((t - 6.0) ** 2) / 8.0)
X = ENV * (np.sin(2 * np.pi * 1.1 * t) + 0.6 * np.sin(2 * np.pi * 3.7 * t))
Y = ENV * (0.8 * np.cos(2 * np.pi * 0.7 * t) + 0.4 * np.sin(2 * np.pi * 2.3 * t + 0.4))
T = np.array([0.1, 0.3, 0.7, 1.5, 3.0])


def test_single_component_identities():
    rec = Record(x=X, y=np.zeros_like(X), dt=DT)
    r = rec.response_spectra.rotd(T)
    rec.response_spectra.compute(T, components=["X"])
    sa_x = rec.response_spectra.Sa["X"]
    np.testing.assert_allclose(r["RotD100"], sa_x, rtol=1e-10)
    np.testing.assert_allclose(r["RotD00"], 0.0, atol=1e-12 * sa_x.max())
    # max_t |cos(theta) u| = |cos(theta)| Sd: the median over theta is cos(45 deg).
    np.testing.assert_allclose(r["RotD50"], np.cos(np.pi / 4) * sa_x, rtol=0.02)
    assert np.all(r["angle_RotD100"] == 0.0)


def test_identical_components_peak_at_45_deg():
    rec = Record(x=X, y=X, dt=DT)
    r = rec.response_spectra.rotd(T)
    rec.response_spectra.compute(T, components=["X"])
    np.testing.assert_allclose(r["RotD100"], np.sqrt(2.0) * rec.response_spectra.Sa["X"], rtol=1e-10)
    assert np.all(r["angle_RotD100"] == 45.0)


def test_matches_brute_force_rotation_of_the_input():
    rec = Record(x=X, y=Y, dt=DT)
    r = rec.response_spectra.rotd(T)
    for deg in (0, 37, 90, 151):
        th = np.radians(deg)
        rot = Record(x=np.cos(th) * X + np.sin(th) * Y, dt=DT)
        rot.response_spectra.compute(T)
        np.testing.assert_allclose(r["Sa_angles"][:, deg], rot.response_spectra.Sa["X"], rtol=1e-9)


def test_rotd_bounds_and_ordering():
    rec = Record(x=X, y=Y, dt=DT)
    r = rec.response_spectra.rotd(T, percentiles=(0, 50, 84, 100))
    rec.response_spectra.compute(T)
    sa = rec.response_spectra.Sa
    assert {k for k in r if k.startswith("RotD")} == {"RotD00", "RotD50", "RotD84", "RotD100"}
    assert np.all(r["RotD00"] <= r["RotD50"]) and np.all(r["RotD50"] <= r["RotD84"])
    assert np.all(r["RotD100"] >= np.maximum(sa["X"], sa["Y"]) - 1e-12)
    assert np.all(r["RotD00"] <= np.minimum(sa["X"], sa["Y"]) + 1e-12)


def test_numba_and_python_paths_agree():
    rec = Record(x=X, y=Y, dt=DT)
    fast = rec.response_spectra.rotd(T, parallel=True, chunk=2)
    slow = rec.response_spectra.rotd(T, parallel=False)
    np.testing.assert_allclose(fast["Sa_angles"], slow["Sa_angles"], rtol=1e-9)


def test_filter_pipeline_applies_like_compute():
    rec = Record(x=X, y=Y, dt=DT)
    rs = rec.response_spectra
    rs.add_filter(rec.filter.band_pass, Tc_low=0.2, Tc_high=5.0)
    with_pipeline = rs.rotd(T)["RotD50"]
    filtered = rec.filter.band_pass(Tc_low=0.2, Tc_high=5.0)
    np.testing.assert_allclose(rs.rotd(T, df=filtered)["RotD50"], with_pipeline, rtol=1e-12)
    raw = rs.rotd(T, use_filters=False)["RotD50"]
    assert not np.allclose(raw, with_pipeline)


def test_validation_and_plot():
    rec = Record(x=X, y=Y, dt=DT)
    rs = rec.response_spectra
    with pytest.raises(RuntimeError):
        rs.plot_rotd()
    with pytest.raises(ValueError, match="divide 180"):
        rs.rotd(T, angle_step=7.0)
    with pytest.raises(ValueError, match="two different"):
        rs.rotd(T, components=("X", "X"))
    with pytest.raises(ValueError, match="not found"):
        rs.rotd(T, components=("X", "Z"))
    r = rs.rotd(T, angle_step=5.0)
    assert r["angles"].size == 36
    _fig, ax = rs.plot_rotd()
    assert len(ax.get_lines()) == 2 + 3

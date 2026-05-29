"""Filter toolbox tests (ObsPy-backed)."""
import numpy as np
import pytest

from apeQuake import Record


@pytest.fixture
def rec():
    dt = 0.01
    t = np.arange(0.0, 20.0, dt)
    x = np.sin(2.0 * np.pi * 1.5 * t)
    return Record(x=x, y=0.5 * x, dt=dt)


def test_detrend_demean_removes_mean(rec):
    # add an offset via a copy then demean it back out
    df = rec.df.copy()
    df["X"] = df["X"] + 50.0
    out = rec.filter.detrend(df=df, type="demean")
    assert abs(out["X"].mean()) < 1e-9


def test_filter_does_not_mutate_baseline(rec):
    baseline = rec.df["X"].to_numpy().copy()
    _ = rec.filter.band_pass(Tc_low=0.1, Tc_high=5.0)
    assert np.array_equal(rec.df["X"].to_numpy(), baseline)


def test_filter_preserves_shape_and_columns(rec):
    out = rec.filter.band_pass(Tc_low=0.1, Tc_high=5.0)
    assert list(out.columns) == ["time", "X", "Y"]
    assert len(out) == len(rec.df)


def test_band_pass_attenuates_out_of_band():
    dt = 0.005
    t = np.arange(0.0, 40.0, dt)
    # 0.5 Hz (in band) + 20 Hz (out of band)
    x = np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 20.0 * t)
    rec = Record(x=x, dt=dt)
    # keep periods 0.5..5 s -> 0.2..2 Hz, so 20 Hz should be removed, 0.5 Hz kept
    out = rec.filter.band_pass(Tc_low=0.5, Tc_high=5.0)
    # energy should drop because the 20 Hz component is filtered out
    assert np.var(out["X"]) < np.var(rec.df["X"])


def test_detrend_invalid_type_raises(rec):
    with pytest.raises(ValueError, match="demean.*linear"):
        rec.filter.detrend(type="bogus")

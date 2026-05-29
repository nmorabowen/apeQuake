"""Intensity-measure tests: Husid curve and significant duration."""
import numpy as np
import pytest

from apeQuake import Record


@pytest.fixture
def rec():
    dt = 0.01
    t = np.arange(0.0, 30.0, dt)
    # amplitude-modulated burst so the Husid curve is non-trivial
    env = np.exp(-((t - 10.0) ** 2) / (2.0 * 3.0**2))
    x = env * np.sin(2.0 * np.pi * 2.0 * t)
    return Record(x=x, dt=dt)


def test_husid_monotone_and_normalized(rec):
    t, hus = rec.intensity_measures.husid("X")
    assert hus[0] >= 0.0
    assert np.isclose(hus[-1], 1.0)
    assert np.all(np.diff(hus) >= -1e-12)  # non-decreasing


def test_significant_duration_positive_and_bounded(rec):
    d = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
    assert 0.0 < d < rec.time[-1]


def test_d5_75_shorter_than_d5_95(rec):
    d_5_75 = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.75)
    d_5_95 = rec.intensity_measures.significant_duration("X", p1=0.05, p2=0.95)
    assert d_5_75 < d_5_95


def test_zero_signal_duration_zero():
    rec = Record(x=np.zeros(1000), dt=0.01)
    assert rec.intensity_measures.significant_duration("X") == 0.0


def test_invalid_percentile_bounds(rec):
    with pytest.raises(ValueError):
        rec.intensity_measures.significant_duration("X", p1=0.5, p2=0.2)

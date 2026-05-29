"""Tests for the core Record container: construction, validation, properties."""
import numpy as np
import pytest

from apeQuake import Record


@pytest.fixture
def sine():
    dt = 0.01
    t = np.arange(0, 10, dt)
    return dt, t, np.sin(2 * np.pi * 1.5 * t)


def test_construct_from_dt(sine):
    dt, t, x = sine
    rec = Record(x=x, dt=dt)
    assert rec.components == ("X",)
    assert rec.dt == dt
    assert len(rec.df) == len(x)
    # generated time array starts at 0 with spacing dt
    assert np.isclose(rec.time[0], 0.0)
    assert np.allclose(np.diff(rec.time), dt)
    assert rec._interpolation_flag is False


def test_multicomponent(sine):
    dt, t, x = sine
    rec = Record(x=x, y=0.5 * x, z=-x, dt=dt)
    assert rec.components == ("X", "Y", "Z")
    assert np.allclose(rec.y, 0.5 * x)
    assert np.allclose(rec.z, -x)
    assert rec.data.shape == (len(x), 3)


def test_uniform_time_array_no_interpolation(sine):
    dt, t, x = sine
    rec = Record(x=x, time_array=t)
    assert rec._interpolation_flag is False
    assert np.isclose(rec.dt, dt)
    assert np.allclose(rec.time, t)


def test_nonuniform_time_array_triggers_interpolation():
    # strictly increasing but non-uniform spacing
    t = np.array([0.0, 0.01, 0.025, 0.03, 0.05])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rec = Record(x=x, time_array=t)
    assert rec._interpolation_flag is True
    # resampled onto a uniform grid
    assert np.allclose(np.diff(rec.time), rec.dt)


def test_dt_mismatch_triggers_interpolation(sine):
    dt, t, x = sine
    rec = Record(x=x, time_array=t, dt=0.02)  # differs from the 0.01 grid
    assert rec._interpolation_flag is True
    assert np.isclose(rec.dt, 0.02)


def test_missing_component_raises(sine):
    dt, t, x = sine
    rec = Record(x=x, dt=dt)
    with pytest.raises(AttributeError):
        _ = rec.y


def test_requires_at_least_one_component():
    with pytest.raises(ValueError, match="At least one component"):
        Record(dt=0.01)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        Record(x=np.zeros(10), y=np.zeros(11), dt=0.01)


def test_dt_required_without_time():
    with pytest.raises(ValueError, match="dt must be provided"):
        Record(x=np.zeros(10))


def test_nonpositive_dt_raises():
    with pytest.raises(ValueError, match="dt must be > 0"):
        Record(x=np.zeros(10), dt=0.0)


def test_non_increasing_time_raises():
    t = np.array([0.0, 0.01, 0.01, 0.02])  # repeated value
    with pytest.raises(ValueError, match="strictly increasing"):
        Record(x=np.zeros(4), time_array=t)


def test_repr_does_not_crash(sine):
    # Regression: __repr__ previously referenced IntensityMeasures attributes.
    dt, t, x = sine
    rec = Record(x=x, y=x, dt=dt, name="demo")
    text = repr(rec)
    assert "Record" in text
    assert "name='demo'" in text
    assert "components=[X,Y]" in text


def test_composites_attached(sine):
    dt, t, x = sine
    rec = Record(x=x, dt=dt)
    for attr in ("filter", "spectrum", "spectrogram",
                 "response_spectra", "intensity_measures", "plot_record"):
        assert hasattr(rec, attr)

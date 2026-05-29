"""Spectrogram tests: cached compute() feeds plot_spectrogram (no recompute,
no second round of filtering)."""
import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pytest

from apeQuake import Record


@pytest.fixture
def rec():
    dt = 0.01
    t = np.arange(0.0, 20.0, dt)
    x = np.sin(2.0 * np.pi * 2.0 * t) + 0.4 * np.sin(2.0 * np.pi * 6.0 * t)
    return Record(x=x, y=0.5 * x, dt=dt, name="sg")


def test_compute_populates_cache(rec):
    rec.spectrogram.compute(NFFT=128, noverlap=64)
    sg = rec.spectrogram
    assert sg.freqs is not None and sg.bins is not None
    assert set(sg.power.keys()) == {"X", "Y"}
    assert sg._nfft == 128 and sg._noverlap == 64
    # power arrays are (n_freqs, n_time_bins)
    assert sg.power["X"].shape == (len(sg.freqs), len(sg.bins))


def test_plot_reuses_cached_compute(rec):
    sg = rec.spectrogram
    sg.compute(NFFT=128, noverlap=64)
    cached = sg.power            # same object should survive a matching plot
    fig, axes = sg.plot_spectrogram(NFFT=128, noverlap=64, show=False)
    assert sg.power is cached    # not recomputed
    assert len(axes) == 2        # one row per component


def test_plot_recomputes_on_param_change(rec):
    sg = rec.spectrogram
    sg.compute(NFFT=128, noverlap=64)
    cached = sg.power
    sg.plot_spectrogram(NFFT=256, noverlap=128, show=False)
    assert sg.power is not cached      # stale cache was refreshed
    assert sg._nfft == 256


def test_apply_filter_invalidates_cache(rec):
    sg = rec.spectrogram
    sg.compute(NFFT=128, noverlap=64)
    assert sg.power
    sg.apply_detrend(type="demean")
    assert sg.power == {} and sg.freqs is None and sg._nfft is None


def test_plot_auto_computes_without_explicit_compute(rec):
    # plot_spectrogram should compute on demand if nothing is cached yet
    fig, axes = rec.spectrogram.plot_spectrogram(show=False)
    assert rec.spectrogram.power           # populated by the plot call
    assert len(axes) == 2

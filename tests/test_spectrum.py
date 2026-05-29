"""Amplitude-spectrum tests: peak location and scaling."""
import numpy as np

from apeQuake import Record


def test_spectrum_peak_at_drive_frequency():
    dt = 0.005
    t = np.arange(0.0, 60.0, dt)
    f0 = 3.0
    amp = 2.0
    x = amp * np.sin(2.0 * np.pi * f0 * t)
    rec = Record(x=x, dt=dt)

    rec.spectrum.compute(scale="amplitude")
    freqs = rec.spectrum.freqs
    A = rec.spectrum.amplitudes["X"]

    peak_freq = freqs[np.argmax(A)]
    assert abs(peak_freq - f0) < 0.05  # within one bin or so

    # single-sided amplitude scaling should recover the physical amplitude
    assert np.isclose(np.max(A), amp, rtol=0.02)


def test_spectrum_invalidated_when_df_set():
    dt = 0.01
    t = np.arange(0.0, 10.0, dt)
    rec = Record(x=np.sin(2 * np.pi * t), dt=dt)
    rec.spectrum.compute()
    assert rec.spectrum.freqs is not None
    rec.spectrum.set_df(rec.df.copy())
    assert rec.spectrum.freqs is None  # cache invalidated

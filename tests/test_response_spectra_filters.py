"""Regression tests for the response-spectra filter-pipeline semantics.

Previously `compute_response_spectrum` pre-copied `record.df` and passed it as a
user-supplied df, which made `compute()` treat the pipeline as opt-in and silently
DROP registered filters. These tests pin the corrected behaviour: the single- and
multi-component entry points apply the pipeline identically.
"""
import numpy as np
import pytest

from apeQuake import Record


@pytest.fixture
def dc_offset_record():
    dt = 0.01
    t = np.arange(0.0, 20.0, dt)
    x = np.sin(2.0 * np.pi * 1.5 * t) + 100.0  # large DC offset
    return Record(x=x, dt=dt), np.linspace(0.1, 3.0, 25)


def test_compute_response_spectrum_applies_registered_filters(dc_offset_record):
    rec, T = dc_offset_record
    rec.response_spectra.add_filter(rec.filter.detrend, type="demean")

    rs_filtered = rec.response_spectra.compute_response_spectrum(periods=T, component="X")

    rec.response_spectra.clear_filters()
    rs_raw = rec.response_spectra.compute_response_spectrum(periods=T, component="X")

    # The DC offset hugely inflates Sd when unfiltered; filtering must change the result.
    assert not np.allclose(rs_filtered["Sd"], rs_raw["Sd"])
    assert np.max(rs_filtered["Sd"]) < np.max(rs_raw["Sd"])


def test_single_and_multi_component_paths_agree(dc_offset_record):
    rec, T = dc_offset_record
    rec.response_spectra.add_filter(rec.filter.detrend, type="demean")

    # multi-component path (df=None -> filters on by default)
    rec.response_spectra.compute(periods=T, sa_mode="absolute")
    sd_multi = rec.response_spectra.Sd["X"].copy()

    # single-component convenience path must match
    rs = rec.response_spectra.compute_response_spectrum(
        periods=T, component="X", sa_mode="absolute"
    )
    assert np.allclose(sd_multi, rs["Sd"])


def test_use_filters_false_keeps_raw(dc_offset_record):
    rec, T = dc_offset_record
    rec.response_spectra.add_filter(rec.filter.detrend, type="demean")

    rs_off = rec.response_spectra.compute_response_spectrum(
        periods=T, component="X", use_filters=False
    )
    rec.response_spectra.clear_filters()
    rs_raw = rec.response_spectra.compute_response_spectrum(periods=T, component="X")

    assert np.allclose(rs_off["Sd"], rs_raw["Sd"])

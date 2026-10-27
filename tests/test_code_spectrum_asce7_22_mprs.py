"""ASCE 7-22 Section 11.4.5.1 multi-period design spectrum.

Rules checked against the ASCE 7-22 PDF (p. 107 for 11.4.5.1 / 11.4.6, p. 229
for 21.4).  Agreement with codeSpectra's ``ASCE7_22.from_mprs`` was checked
outside the suite (identical Sa on 0-16 s); the one intended difference is the
21.4 SM1 window for Site Class C, pinned below.
"""
import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import (
    MPRS_PERIODS,
    ASCE7_22MultiPeriodSpectrum,
    get_model_class,
)
from apeQuake.code_spectrum.codes.asce7_22 import ASCE7_22Spectrum
from apeQuake.code_spectrum.codes.asce7_22_mprs import infer_sms_sm1

PER = np.array(MPRS_PERIODS)
# Soft-site-like MCE_R ordinates [g]: T * Sa keeps rising past 2 s.
SA = np.array([0.60, 0.61, 0.63, 0.68, 0.80, 0.95, 1.10, 1.30, 1.42, 1.45, 1.44,
               1.38, 1.30, 1.10, 0.95, 0.75, 0.62, 0.48, 0.38, 0.30, 0.18, 0.11])


def model(**kw):
    kw = {"tl": 12.0, "site_class": "CD", **kw}
    return ASCE7_22MultiPeriodSpectrum(PER, SA, **kw)


def test_mprs_periods_are_the_22_of_item_1():
    assert len(MPRS_PERIODS) == 22
    assert MPRS_PERIODS[0] == 0.0 and MPRS_PERIODS[-1] == 10.0
    assert model().standard_periods is True


def test_nodes_are_two_thirds_and_mcer_is_one_and_a_half():
    d, m = model(), model(level="mcer")
    np.testing.assert_allclose(d.sa(PER), 2.0 / 3.0 * SA, rtol=1e-12)
    np.testing.assert_allclose(m.sa(PER), SA, rtol=1e-12)
    T = np.linspace(0.0, 15.0, 301)
    np.testing.assert_allclose(m.sa(T), 1.5 * d.sa(T), rtol=1e-12)    # 11.4.6


def test_linear_interpolation_between_nodes():
    d = model()
    mid = 0.5 * (PER[:-1] + PER[1:])
    np.testing.assert_allclose(d.sa(mid), 2.0 / 3.0 * 0.5 * (SA[:-1] + SA[1:]), rtol=1e-12)


def test_beyond_10_s_item_3():
    d = model(tl=12.0)
    sa10 = 2.0 / 3.0 * SA[-1]
    np.testing.assert_allclose(d.sa([11.0, 12.0]), sa10 * 10.0 / np.array([11.0, 12.0]))
    np.testing.assert_allclose(d.sa(16.0), sa10 * 10.0 * 12.0 / 16.0**2)
    # TL < 10 s: as written, T > 10 s > TL takes the 10 TL / T^2 branch.
    np.testing.assert_allclose(model(tl=8.0).sa(11.0), sa10 * 10.0 * 8.0 / 11.0**2)


def test_section_21_4_inference_and_site_class_c_window():
    # 0.9 max Sa over 0.2-5 s
    sms_cd, sm1_cd = infer_sms_sm1(PER, SA, "CD")
    assert np.isclose(sms_cd, 0.9 * 1.45)
    # vs <= 442 m/s: 0.9 max T Sa over 1-5 s (here 4 s x 0.38)
    assert np.isclose(sm1_cd, 0.9 * 4.0 * 0.38)
    # Site Class C has vs > 442 m/s (Table 20.2-1): window 1-2 s (2 s x 0.62).
    # codeSpectra used 1-5 s here.
    _, sm1_c = infer_sms_sm1(PER, SA, "C")
    assert np.isclose(sm1_c, 0.9 * 2.0 * 0.62)
    # ... but never below Sa(1 s)
    flat = np.where(PER >= 1.0, 1.0 / np.maximum(PER, 1e-9) * 0.5, 0.5)
    assert np.isclose(infer_sms_sm1(PER, flat, "B")[1], 0.5)


def test_sms_sm1_input_or_inferred():
    inferred = model()
    assert inferred.sms_source.startswith("inferred")
    given = model(sms=1.5, sm1=1.2)
    assert given.sms_source == "input"
    assert np.isclose(given.sds, 1.0) and np.isclose(given.sd1, 0.8)
    with pytest.raises(ValueError, match="both"):
        model(sms=1.5)


def test_two_period_comparison():
    d = model(sms=1.5, sm1=1.2)
    tp = d.two_period()
    assert isinstance(tp, ASCE7_22Spectrum)
    assert np.isclose(tp.sds, d.sds) and np.isclose(tp.sd1, d.sd1) and tp.tl == d.tl


@pytest.mark.parametrize("kw, match", [
    ({"site_class": "F"}, "Site Class F"),
    ({"site_class": "X"}, "site_class"),
    ({"level": "service"}, "level"),
    ({"tl": -1.0}, "tl"),
])
def test_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        model(**kw)


def test_validation_of_ordinates():
    with pytest.raises(ValueError, match="length"):
        ASCE7_22MultiPeriodSpectrum(PER, SA[:-1], 8.0, "D")
    with pytest.raises(ValueError, match="start at 0"):
        ASCE7_22MultiPeriodSpectrum(PER[1:], SA[1:], 8.0, "D")
    with pytest.raises(ValueError, match="start at 0"):
        ASCE7_22MultiPeriodSpectrum(PER[::-1], SA, 8.0, "D")
    with pytest.raises(ValueError, match=">= 0"):
        ASCE7_22MultiPeriodSpectrum(PER, -SA, 8.0, "D")


def test_registered_and_usable_from_record():
    assert get_model_class("ASCE 7-22 multi-period") is ASCE7_22MultiPeriodSpectrum
    assert get_model_class("asce7-22-mprs") is ASCE7_22MultiPeriodSpectrum
    rec = Record(x=np.zeros(100), dt=0.01)
    rec.code_spectrum.add("ASCE 7-22 MPRS", periods=PER, sa_mcer=SA, tl=12.0, site_class="D")
    rec.code_spectrum.compute(t_max=4.0)
    T = rec.code_spectrum.periods
    for node in PER[PER <= 4.0]:
        assert np.any(np.isclose(T, node))          # nodes sampled exactly
    assert rec.code_spectrum.summary().loc["ASCE7-22-MPRS", "basis"].startswith("multi-period")

"""Spectrum operations (scaled, reduced, envelope, floored_by, capped_by, peak)
and the ASCE 7-16 Section 11.9 vertical spectrum."""
import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import DerivedSpectrum
from apeQuake.code_spectrum.codes import ASCE7_16Spectrum, ASCE7_16VerticalSpectrum, NECSpectrum
from apeQuake.code_spectrum.codes.asce7_16 import vertical_coefficient

A = ASCE7_16Spectrum(1.5, 0.6, "C", 8.0)
N = NECSpectrum(z=0.4, site_class="D", region="costa")
T = np.linspace(0.0, 6.0, 601)


def test_scaled_and_reduced():
    np.testing.assert_allclose(A.scaled(1.5).sa(T), 1.5 * A.sa(T))
    np.testing.assert_allclose(A.reduced(8.0, Ie=1.25).sa(T), A.sa(T) * 1.25 / 8.0)
    np.testing.assert_allclose(N.reduced(6.0, Ie=1.3, phi_p=0.9, phi_e=0.9).sa(T),
                               N.sa(T) * 1.3 / (6.0 * 0.81))
    with pytest.raises(ValueError):
        A.scaled(0.0)


def test_envelope_floor_cap():
    env = A.envelope(N, A.scaled(0.5))
    np.testing.assert_allclose(env.sa(T), np.maximum(A.sa(T), N.sa(T)))
    np.testing.assert_allclose(N.floored_by(A, 0.8).sa(T), np.maximum(N.sa(T), 0.8 * A.sa(T)))
    np.testing.assert_allclose(N.capped_by(A).sa(T), np.minimum(N.sa(T), A.sa(T)))
    with pytest.raises(ValueError):
        A.envelope()


def test_derived_keeps_knees_and_works_everywhere(tmp_path):
    f = N.floored_by(A, 0.8, label="NEC floored")
    assert isinstance(f, DerivedSpectrum) and f.code == "NEC floored"
    assert set(A.knee_periods()) | set(N.knee_periods()) <= set(f.knee_periods())
    assert f.parameters()["floor_ratio"] == 0.8 and "ASCE7-16" in f.parameters()["from"]
    rec = Record(x=np.zeros(10), dt=0.01)
    rec.code_spectrum.add(f)
    rec.code_spectrum.compute()
    assert "NEC floored" in rec.code_spectrum.Sa
    assert np.isclose(f.sd(1.0)[0], f.sa(1.0)[0] * 9.80665 * (1.0 / (2 * np.pi)) ** 2)


def test_peak():
    t, sa = A.peak()
    assert np.isclose(sa, A.sds) and A.t0 - 1e-9 <= t <= A.ts + 1e-9


# ---------------------------- vertical (11.9) ---------------------------- #

@pytest.mark.parametrize("ss, sc, cv", [
    (0.1, "A", 0.7), (2.5, "B", 0.9), (0.6, "C", 1.0), (1.0, "C", 1.1),
    (2.0, "D", 1.5), (0.3, "E", 0.9), (0.45, "D", 1.0), (1.5, "C", 1.2),
])
def test_cv_table_11_9_1(ss, sc, cv):
    assert np.isclose(vertical_coefficient(ss, sc), cv)


def test_vertical_equations_11_9_1_to_4():
    m = ASCE7_16Spectrum(0.8, 0.15, "D", 8.0, level="mcer")   # floor rarely governs here
    v = m.vertical()
    a = v.cv * m.sms
    Tv = np.array([0.0, 0.025, 0.04, 0.1, 0.15, 1.0])
    ref = np.array([0.3 * a, 0.3 * a, 20 * a * 0.015 + 0.3 * a, 0.8 * a, 0.8 * a,
                    0.8 * a * 0.15**0.75])
    np.testing.assert_allclose(v.sa(Tv), np.maximum(ref, 0.5 * m.sa(Tv)))
    assert np.isnan(v.sa(2.5)[0])


def test_vertical_design_is_two_thirds_and_floor_is_half_horizontal():
    d = ASCE7_16Spectrum(1.5, 0.6, "C", 8.0).vertical()
    mc = ASCE7_16Spectrum(1.5, 0.6, "C", 8.0, level="mcer").vertical()
    Tv = np.linspace(0.0, 2.0, 201)
    np.testing.assert_allclose(d.sa(Tv), 2.0 / 3.0 * mc.sa(Tv))
    # codeSpectra omitted the 11.9.2 floor; here it governs at 0.5 s
    assert np.all(d.sa(Tv) >= 0.5 * A.sa(Tv) - 1e-12)
    assert np.isclose(d.sa(0.5)[0], 0.5 * A.sa(0.5)[0])


def test_vertical_from_site_specific_needs_ss():
    s = ASCE7_16Spectrum.from_sds_sd1(1.0, 0.5, 8.0)
    with pytest.raises(ValueError, match="Ss"):
        s.vertical()
    v = s.vertical(ss=1.2, site_class="D")
    assert isinstance(v, ASCE7_16VerticalSpectrum) and np.isclose(v.cv, 1.34)
    assert v.default_periods(4.0).max() == 2.0

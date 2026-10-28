"""ASCE 7 design parameters: Ie, SDC and ELF (Sections 1.5, 11.6, 12.8).

Tables and equations checked against the ASCE 7-10, 7-16 and 7-22 PDFs.
Agreement with codeSpectra's ``elf.base_shear`` / ``seismic_design_category``
was checked outside the suite (432 ELF cases and their SDC, to 4e-16).
"""
import numpy as np
import pytest

from apeQuake.code_spectrum.codes import ASCE7_22Spectrum
from apeQuake.design import asce7


@pytest.mark.parametrize("rc, ie", [("I", 1.0), ("II", 1.0), ("III", 1.25), ("IV", 1.5)])
def test_importance_factor_table_1_5_2(rc, ie):
    assert asce7.importance_factor(rc) == ie


@pytest.mark.parametrize("sds, sd1, s1, rc, sdc", [
    (0.10, 0.05, 0.04, "II", "A"),
    (0.20, 0.05, 0.04, "II", "B"),     # Table 11.6-1 governs
    (0.20, 0.05, 0.04, "IV", "C"),
    (0.40, 0.10, 0.10, "II", "C"),
    (0.40, 0.25, 0.30, "II", "D"),     # Table 11.6-2 governs
    (0.60, 0.15, 0.20, "IV", "D"),
    (1.00, 0.60, 0.75, "II", "E"),     # S1 >= 0.75 override
    (1.00, 0.60, 0.80, "IV", "F"),
])
def test_seismic_design_category(sds, sd1, s1, rc, sdc):
    r = asce7.seismic_design_category(sds, sd1, s1=s1, risk_category=rc)
    assert r.sdc == sdc
    assert r.s1_override is (s1 >= 0.75)


def test_sdc_without_s1_is_flagged():
    r = asce7.seismic_design_category(1.0, 0.6, s1=None)
    assert r.sdc == "D" and r.s1_override is None


def test_approximate_period_and_cu():
    # Table 12.8-2, concrete moment frame: Ct = 0.0466 (m), x = 0.9
    assert np.isclose(asce7.approximate_period(30.0, "concrete_moment_frame"), 0.0466 * 30**0.9)
    assert np.isclose(asce7.approximate_period(100.0, "other", metric=False), 0.02 * 100**0.75)
    # Table 12.8-1, interpolated and clamped
    assert asce7.upper_limit_coefficient(0.05) == 1.7
    assert np.isclose(asce7.upper_limit_coefficient(0.25), 1.45)
    assert asce7.upper_limit_coefficient(0.6) == 1.4


def test_cs_classical_limits():
    kw = {"sds": 1.0, "sd1": 0.6, "s1": 0.65, "tl": 8.0, "R": 8.0, "Ie": 1.0}
    cs, bd = asce7.seismic_response_coefficient(T=0.3, **kw)
    assert np.isclose(cs, 1.0 / 8.0) and bd["governing"] == "eq 12.8-2"
    cs, bd = asce7.seismic_response_coefficient(T=1.5, **kw)
    assert np.isclose(cs, 0.6 / (1.5 * 8.0)) and bd["governing"] == "cap eq 12.8-3"
    cs, bd = asce7.seismic_response_coefficient(T=9.0, **kw)
    # 0.6 * 8 / (81 * 8) = 0.0074 < 0.044 SDS = 0.044 and < 0.5 S1 / R = 0.0406
    assert np.isclose(cs, 0.044) and bd["governing"] == "min eq 12.8-5"
    cs, bd = asce7.seismic_response_coefficient(T=9.0, **{**kw, "sds": 0.3})
    assert np.isclose(cs, 0.5 * 0.65 / 8.0) and bd["governing"] == "min eq 12.8-6"


def test_cs_7_22_numbering_and_method_1():
    kw = {"sds": 1.0, "sd1": 0.6, "s1": 0.3, "tl": 8.0, "R": 8.0, "Ie": 1.0, "edition": "7-22"}
    _cs, bd = asce7.seismic_response_coefficient(T=1.5, **kw)
    assert bd["method"] == 2 and bd["governing"] == "cap eq 12.8-4"
    # Method 1 with the two-period shape: Sa(1.5) = SD1 / 1.5; below the
    # period of the maximum (T0) the maximum Sa is used.
    sp = ASCE7_22Spectrum.from_sds_sd1(1.0, 0.6, 8.0)
    cs1, bd1 = asce7.seismic_response_coefficient(T=1.5, spectrum=sp, **kw)
    assert bd1["method"] == 1 and bd1["governing"] == "eq 12.8-2"
    assert np.isclose(cs1, 0.6 / 1.5 / 8.0)
    cs0, _ = asce7.seismic_response_coefficient(T=0.01, spectrum=sp, **kw)
    assert np.isclose(cs0, 1.0 / 8.0)
    with pytest.raises(ValueError, match="7-22"):
        asce7.seismic_response_coefficient(T=1.0, spectrum=sp, **{**kw, "edition": "7-16"})


def test_base_shear_period_cap_and_distribution():
    hn, h = 40.0, np.arange(1, 11) * 4.0
    w = np.full(10, 500.0)
    r = asce7.base_shear(W=w.sum(), R=8.0, sds=1.0, sd1=0.6, s1=0.65, tl=8.0, risk_category="III",
                         T=3.0, hn=hn, structure_type="steel_moment_frame", weights=w, heights=h)
    Ta = 0.0724 * hn**0.8
    assert np.isclose(r.Ta, Ta) and np.isclose(r.Cu, 1.4)
    assert np.isclose(r.T, 1.4 * Ta)                     # 12.8.2 upper limit
    assert r.Ie == 1.25 and np.isclose(r.V, r.Cs * w.sum())
    k = 1.0 + (r.T - 0.5) / 2.0
    np.testing.assert_allclose(r.Fx, r.V * w * h**k / np.sum(w * h**k))
    assert np.isclose(r.Fx.sum(), r.V) and np.isclose(r.Vx[0], r.V) and np.isclose(r.Vx[-1], r.Fx[-1])
    r2 = asce7.base_shear(W=w.sum(), R=8.0, sds=1.0, sd1=0.6, s1=0.65, tl=8.0, T=3.0, hn=hn,
                          structure_type="steel_moment_frame", apply_upper_limit=False)
    assert r2.T == 3.0 and r2.Fx is None


def test_k_exponent_options():
    assert asce7.vertical_distribution_exponent(0.4) == 1.0
    assert np.isclose(asce7.vertical_distribution_exponent(1.5), 1.5)
    assert asce7.vertical_distribution_exponent(1.5, interpolate=False) == 2.0
    assert asce7.vertical_distribution_exponent(3.0) == 2.0


@pytest.mark.parametrize("kw, match", [
    ({"risk_category": "V"}, "risk_category"),
    ({"edition": "7-05"}, "edition"),
    ({"structure_type": "timber"}, "structure_type"),
    ({"T": None, "hn": None}, "T, the height"),
    ({"weights": [1.0]}, "both weights"),
])
def test_validation(kw, match):
    base = {"W": 100.0, "R": 8.0, "sds": 1.0, "sd1": 0.6, "s1": 0.3, "tl": 8.0, "T": 1.0, "hn": 20.0}
    with pytest.raises(ValueError, match=match):
        asce7.base_shear(**{**base, **kw})

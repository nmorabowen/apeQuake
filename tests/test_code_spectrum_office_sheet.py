"""Regression against the office spreadsheet (independent implementation).

Values are the cached results of APE_Programas/Programas/NEC_SE_DS - ASCE 7-16
(2016 v.3).xlsm (sheets "Espectros NEC-SE-DS", "Espectros NMB" = ASCE 7-10,
"Espectros ASCE (2016)"), extracted once and pinned here.

Known, intentional differences from the sheet:

* ASCE 7-10 / 7-16: the sheet holds SDS from T = 0 to Ts; the standard (and
  this library) ramps 0.4 SDS -> SDS between T = 0 and T0, so only T >= T0 is compared.
* ASCE 7-16: the sheet applies Fv = 1.7 for Site Class D with S1 = 0.7 and ignores
  the 11.4.8 restriction; the library refuses unless allow_exception=True,
  so the plain spectrum shape is compared through from_sds_sd1.
* NEC Sd: the sheet uses g = 981 cm/s^2.
"""
import pytest

from apeQuake.code_spectrum import get_model_class

# (T [s], Sa [g], Sd [cm]) -- NEC, Z = 0.345, Costa, soil D
NEC_SHEET = [(0.0, 0.432975, 0.0), (0.121308, 0.779355, 0.284986), (0.166799, 0.779355, 0.538802), (0.530723, 0.779355, 5.454814), (1.636797, 0.317682, 21.149086), (2.848799, 0.182526, 36.809403), (4.487424, 0.115875, 39.941466), (6.0, 0.086664, 39.941466)]
# (T [s], Sa [g]) -- ASCE 7-10, Ss = 2.1, S1 = 0.7, site D, TL = 4 s
ASCE10_SHEET = [(0.5, 1.4), (0.541667, 1.292308), (0.875, 0.8), (2.2, 0.318182), (3.7, 0.189189), (4.96, 0.113814), (6.0, 0.077778)]
# (T [s], Sa [g]) -- ASCE 7-16, Ss = 2.1, S1 = 0.7, site D, TL = 4 s (SDS 1.4, SD1 0.7933)
ASCE16_SHEET = [(0.566667, 1.4), (0.602778, 1.316129), (0.891667, 0.88972), (2.2, 0.360606), (3.7, 0.214414), (4.96, 0.128989), (6.0, 0.088148)]


def test_nec_matches_sheet_spectrum_and_parameters():
    m = get_model_class("NEC")(z=0.345, site_class="D", region="costa")
    p = m.parameters()
    assert (p["Fa"], p["Fd"], p["Fs"]) == pytest.approx((1.255, 1.288, 1.182))
    assert (p["T0"], p["Tc"], p["TL"]) == pytest.approx((0.121308, 0.667194, 3.0912), abs=1e-5)
    for T, sa, sd_cm in NEC_SHEET:
        assert m.sa(T)[0] == pytest.approx(sa, abs=2e-6), T
        assert m.sd(T)[0] * 100.0 == pytest.approx(sd_cm, rel=5e-4, abs=1e-6), T  # g=981 vs 9.80665


def test_asce7_10_matches_sheet_for_T_at_or_above_T0():
    m = get_model_class("ASCE 7-10")(ss=2.1, s1=0.7, site_class="D", tl=4.0)
    assert (m.fa, m.fv, m.sds, m.sd1) == pytest.approx((1.0, 1.5, 1.4, 0.7))
    for T, sa in ASCE10_SHEET:
        assert m.sa(T)[0] == pytest.approx(sa, abs=2e-6), T


def test_asce7_16_parameters_and_plain_shape_match_sheet():
    cls = get_model_class("ASCE 7-16")
    # Site D with S1 >= 0.2 is a 11.4.8 case: refused by default (the sheet ignores it).
    with pytest.raises(ValueError, match="11.4.8"):
        cls(ss=2.1, s1=0.7, site_class="D", tl=4.0)
    m = cls(ss=2.1, s1=0.7, site_class="D", tl=4.0, allow_exception=True)
    assert (m.fa, m.fv, m.sds, m.sd1) == pytest.approx((1.0, 1.7, 1.4, 0.793333), abs=1e-5)
    plain = cls.from_sds_sd1(1.4, 0.7933333333333332, tl=4.0)
    for T, sa in ASCE16_SHEET:
        assert plain.sa(T)[0] == pytest.approx(sa, abs=2e-6), T

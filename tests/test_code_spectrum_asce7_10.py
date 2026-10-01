"""ASCE 7-10 two-period spectrum against the standard's tables and hand calculations.

Sources: ASCE/SEI 7-10, Sections 11.4.1-11.4.7 (printed pp. 54-56), Table 11.4-1
(Fa), Table 11.4-2 (Fv); and Charney, *Seismic Loads: Guide to the Seismic Load
Provisions of ASCE 7-10*, Appendix G-A, Tables GA-1/GA-2 (interpolation formulas)
and the Chapter 7 example (Ss=0.75, S1=0.22, Site Class C).
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import get_model_class
from apeQuake.code_spectrum.codes.asce7_10 import ASCE7_10Spectrum

SS_COLS = (0.25, 0.5, 0.75, 1.0, 1.25)
S1_COLS = (0.1, 0.2, 0.3, 0.4, 0.5)

# Table 11.4-1 (ASCE 7-10 p. 55), rows = site class, columns = SS_COLS.
FA = {
    "A": (0.8, 0.8, 0.8, 0.8, 0.8),
    "B": (1.0, 1.0, 1.0, 1.0, 1.0),
    "C": (1.2, 1.2, 1.1, 1.0, 1.0),
    "D": (1.6, 1.4, 1.2, 1.1, 1.0),
    "E": (2.5, 1.7, 1.2, 0.9, 0.9),
}
# Table 11.4-2 (ASCE 7-10 p. 55), rows = site class, columns = S1_COLS.
FV = {
    "A": (0.8, 0.8, 0.8, 0.8, 0.8),
    "B": (1.0, 1.0, 1.0, 1.0, 1.0),
    "C": (1.7, 1.6, 1.5, 1.4, 1.3),
    "D": (2.4, 2.0, 1.8, 1.6, 1.5),
    "E": (3.5, 3.2, 2.8, 2.4, 2.4),
}


def make(ss=1.0, s1=0.4, sc="D", tl=8.0, **kw):
    return ASCE7_10Spectrum(ss=ss, s1=s1, site_class=sc, tl=tl, **kw)


# ----------------------------------------------------------------- tables ---
@pytest.mark.parametrize("sc", list(FA))
@pytest.mark.parametrize("i", range(5))
def test_every_table_cell(sc, i):
    m = make(ss=SS_COLS[i], s1=S1_COLS[i], sc=sc)
    assert m.fa == pytest.approx(FA[sc][i])
    assert m.fv == pytest.approx(FV[sc][i])


def test_clamped_beyond_end_columns():
    lo = make(ss=0.05, s1=0.01, sc="E")
    assert (lo.fa, lo.fv) == (pytest.approx(2.5), pytest.approx(3.5))
    hi = make(ss=3.0, s1=2.0, sc="E")
    assert (hi.fa, hi.fv) == (pytest.approx(0.9), pytest.approx(2.4))


def test_linear_interpolation_matches_charney_formulas():
    """Independent check: Charney Tables GA-1 / GA-2 straight-line formulas."""
    for ss in np.linspace(0.5, 1.0, 11):
        assert make(ss=ss, sc="C").fa == pytest.approx(1.4 - 0.4 * ss)
    for ss in np.linspace(0.25, 0.75, 11):
        assert make(ss=ss, sc="D").fa == pytest.approx(1.8 - 0.8 * ss)
    for ss in np.linspace(0.75, 1.25, 11):
        assert make(ss=ss, sc="D").fa == pytest.approx(1.5 - 0.4 * ss)
    for ss, f in ((0.35, 3.3 - 3.2 * 0.35), (0.6, 2.7 - 2.0 * 0.6), (0.9, 2.1 - 1.2 * 0.9)):
        assert make(ss=ss, sc="E").fa == pytest.approx(f)
    for s1 in np.linspace(0.1, 0.5, 9):
        assert make(s1=s1, sc="C").fv == pytest.approx(1.8 - s1)
    for s1 in np.linspace(0.2, 0.4, 9):
        assert make(s1=s1, sc="D").fv == pytest.approx(2.4 - 2 * s1)
        assert make(s1=s1, sc="E").fv == pytest.approx(4.0 - 4 * s1)
    for s1 in np.linspace(0.1, 0.2, 5):
        assert make(s1=s1, sc="D").fv == pytest.approx(2.8 - 4 * s1)
        assert make(s1=s1, sc="E").fv == pytest.approx(3.8 - 3 * s1)


# ----------------------------------------------------------- hand examples ---
def test_hand_example_high_seismicity_site_d():
    # Ss=1.5 (>=1.25 -> Fa=1.0), S1=0.6 (>=0.5 -> Fv=1.5), Site D.
    m = make(ss=1.5, s1=0.6, sc="D", tl=8.0)
    assert (m.fa, m.fv) == (pytest.approx(1.0), pytest.approx(1.5))
    assert m.sms == pytest.approx(1.5)  # 1.0 * 1.5
    assert m.sm1 == pytest.approx(0.9)  # 1.5 * 0.6
    assert m.sds == pytest.approx(1.0)  # 2/3 * 1.5
    assert m.sd1 == pytest.approx(0.6)  # 2/3 * 0.9
    assert m.ts == pytest.approx(0.6) and m.t0 == pytest.approx(0.12)
    assert m.sa(0.0)[0] == pytest.approx(0.4)
    assert m.sa(0.06)[0] == pytest.approx(0.7)
    assert m.sa(0.3)[0] == pytest.approx(1.0)
    assert m.sa(2.0)[0] == pytest.approx(0.3)  # 0.6 / 2
    assert m.sa(10.0)[0] == pytest.approx(0.6 * 8.0 / 100.0)  # SD1 TL / T^2


def test_hand_example_interpolated_site_c():
    # Ss=0.6 between 0.5 (1.2) and 0.75 (1.1): Fa = 1.2 - 0.1*(0.1/0.25) = 1.16
    # S1=0.35 between 0.3 (1.5) and 0.4 (1.4): Fv = 1.45
    m = make(ss=0.6, s1=0.35, sc="C", tl=6.0)
    assert m.fa == pytest.approx(1.16) and m.fv == pytest.approx(1.45)
    assert m.sms == pytest.approx(1.16 * 0.6)  # 0.696
    assert m.sm1 == pytest.approx(1.45 * 0.35)  # 0.5075
    assert m.sds == pytest.approx(0.464)  # 2/3 * 0.696
    assert m.sd1 == pytest.approx(0.5075 * 2 / 3)  # 0.338333
    assert m.ts == pytest.approx(0.338333 / 0.464, rel=1e-5)  # 0.72917
    assert m.sa(1.0)[0] == pytest.approx(0.338333, rel=1e-5)


def test_charney_chapter7_example():
    # Ss=0.75, S1=0.22, Site C, TL=6 s: Fa=1.1, Fv=1.58, SDS=0.55 g, SD1=0.23 g.
    m = make(ss=0.75, s1=0.22, sc="C", tl=6.0)
    assert m.fa == pytest.approx(1.1) and m.fv == pytest.approx(1.58)
    assert m.sds == pytest.approx(0.55)
    assert m.sd1 == pytest.approx(2 / 3 * 1.58 * 0.22)
    assert m.sd1 == pytest.approx(0.23, abs=1e-2)


def test_fema451_example_site_b_gives_unit_coefficients():
    # Ss=0.438, S1=0.168, Site B -> Fa=Fv=1.0 (FEMA 451 example, TL=12 s).
    m = make(ss=0.438, s1=0.168, sc="B", tl=12.0)
    assert (m.fa, m.fv) == (1.0, 1.0)
    assert m.sms == pytest.approx(0.438) and m.sm1 == pytest.approx(0.168)


# ------------------------------------------------------------------ levels ---
def test_mcer_is_one_and_a_half_times_design():
    d, mc = make(), make(level="mcer")
    T = np.linspace(0.0, 12.0, 241)
    assert mc.sa(T) == pytest.approx(1.5 * d.sa(T))
    assert mc.sa(0.5)[0] == pytest.approx(mc.sms)  # plateau = SMS (T0 < 0.5 < Ts)
    assert mc.sa(2.0)[0] == pytest.approx(mc.sm1 / 2.0)  # SM1 / T
    assert mc.parameters()["Ts"] == pytest.approx(d.parameters()["Ts"])


def test_parameters_and_attributes():
    m = make(ss=1.5, s1=0.6, sc="d", tl=8.0)
    p = m.parameters()
    assert p["site_class"] == "D" and p["level"] == "design"
    expected = dict(Ss=1.5, S1=0.6, Fa=1.0, Fv=1.5, SMS=1.5, SM1=0.9, SDS=1.0,
                    SD1=0.6, T0=0.12, Ts=0.6, TL=8.0)
    for k, v in expected.items():
        assert p[k] == pytest.approx(v)
    assert m.knee_periods() == pytest.approx((0.12, 0.6, 8.0))


def test_from_sds_sd1_bypasses_tables():
    m = ASCE7_10Spectrum.from_sds_sd1(sds=1.0, sd1=0.6, tl=8.0)
    assert m.sa(2.0)[0] == pytest.approx(0.3)
    assert m.sms == pytest.approx(1.5) and m.sm1 == pytest.approx(0.9)
    assert m.fa is None and "Fa" not in m.parameters()
    mc = ASCE7_10Spectrum.from_sds_sd1(1.0, 0.6, 8.0, level="mcer")
    assert mc.sa(0.3)[0] == pytest.approx(1.5)


# ------------------------------------------------------------------ errors ---
def test_site_class_f_raises():
    with pytest.raises(ValueError, match="Site Class F"):
        make(sc="F")


@pytest.mark.parametrize("kw", [
    dict(sc="G"), dict(sc=""), dict(ss=0.0), dict(ss=-0.1), dict(s1=0.0),
    dict(ss=float("nan")), dict(s1=float("inf")), dict(tl=0.0),
    dict(level="maximum"),
])
def test_invalid_inputs(kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_tl_must_exceed_ts():
    # Ss=1.5, S1=0.6, D -> Ts = 0.6 s
    with pytest.raises(ValueError, match="TL"):
        make(ss=1.5, s1=0.6, sc="D", tl=0.6)
    with pytest.raises(ValueError, match="TL"):
        make(ss=1.5, s1=0.6, sc="D", tl=0.5)


def test_from_sds_sd1_errors():
    with pytest.raises(ValueError):
        ASCE7_10Spectrum.from_sds_sd1(0.0, 0.6, 8.0)
    with pytest.raises(ValueError):
        ASCE7_10Spectrum.from_sds_sd1(1.0, 0.6, 0.5)


# --------------------------------------------------------------- composite ---
def test_registered_and_roundtrip_through_composite():
    assert get_model_class("ASCE 7-10") is ASCE7_10Spectrum
    t = np.arange(0.0, 10.0, 0.01)
    rec = Record(x=0.1 * np.sin(2 * np.pi * t), dt=0.01)
    model = rec.code_spectrum.add("ASCE7-10", ss=1.5, s1=0.6, site_class="D", tl=8.0)
    assert isinstance(model, ASCE7_10Spectrum)
    rec.code_spectrum.compute()
    T, sa = rec.code_spectrum.periods, rec.code_spectrum.Sa["ASCE7-10"]
    assert sa[0] == pytest.approx(0.4)  # T = 0
    assert sa[np.argmin(abs(T - 2.0))] == pytest.approx(0.3, rel=1e-2)
    assert rec.code_spectrum.summary().loc["ASCE7-10", "SDS"] == pytest.approx(1.0)

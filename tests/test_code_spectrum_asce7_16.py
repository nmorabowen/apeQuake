"""ASCE 7-16 two-period spectrum against hand-worked references.

Reference values: ASCE/SEI 7-16 Tables 11.4-1 / 11.4-2 as printed in the Guide to
the Seismic Load Provisions of ASCE 7-16 (Table G3-1 / G3-2) and used in the worked
examples of Fanella, *Structural Load Determination 2018 IBC and ASCE 7-16*
(Fa = 1.0, Fv = 1.7 for D at Ss = 1.5, S1 = 0.6; Fa = 1.3 for C at Ss = 0.25-0.5;
Fv = 1.93 for D at S1 = 0.37; Fa = 1.01 for D at Ss = 1.23).
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import get_model_class
from apeQuake.code_spectrum.codes.asce7_16 import (
    FA_TABLE,
    FV_TABLE,
    ASCE7_16Spectrum,
    site_coefficients,
)

# Table 11.4-1 (Fa) and 11.4-2 (Fv), typed independently of the module.
FA_REF = {
    "A": [0.8] * 6,
    "B": [0.9] * 6,
    "C": [1.3, 1.3, 1.2, 1.2, 1.2, 1.2],
    "D": [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
    "E": [2.4, 1.7, 1.3],
}
FV_REF = {
    "A": [0.8] * 6,
    "B": [0.8] * 6,
    "C": [1.5, 1.5, 1.5, 1.5, 1.5, 1.4],
    "D": [2.4, 2.2, 2.0, 1.9, 1.8, 1.7],
    "E": [4.2],  # S1 > 0.1: "See Section 11.4.8", no printed value
}
SS_COLS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
S1_COLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# --------------------------------------------------------------- tables ---
def test_table_cells_match_reference():
    for sc, row in FA_REF.items():
        assert list(FA_TABLE[sc][: len(row)]) == row
    for sc in "ABCD":
        assert list(FA_TABLE[sc]) == FA_REF[sc]
    assert FA_TABLE["E"][3:] == (None, None, None)  # "see 11.4.8"
    for sc, row in FV_REF.items():
        assert list(FV_TABLE[sc][: len(row)]) == row
    assert FV_TABLE["E"][1:] == (None,) * 5


@pytest.mark.parametrize("sc", list("ABCD"))
def test_fa_at_columns(sc):
    for ss, ref in zip(SS_COLS, FA_REF[sc]):
        fa, _ = site_coefficients(ss, 0.1, sc)
        assert fa == pytest.approx(ref)


@pytest.mark.parametrize("sc", list("ABCDE"))
def test_fv_at_columns(sc):
    for s1, ref in zip(S1_COLS, FV_REF[sc]):
        _, fv = site_coefficients(0.25, s1, sc)
        assert fv == pytest.approx(ref)


def test_interpolation_and_clamping():
    # Fanella examples: D, Ss = 1.23 -> 1.1 - 0.1 * (0.23 / 0.25) = 1.008 (~1.01)
    assert site_coefficients(1.23, 0.37, "D")[0] == pytest.approx(1.008)
    # D, S1 = 0.37 -> 2.0 - 0.1 * 0.7 = 1.93 (Fanella: 1.93)
    assert site_coefficients(1.23, 0.37, "D")[1] == pytest.approx(1.93)
    # C, Ss = 0.375 between 1.3 and 1.3; S1 = 0.15 between 1.5 and 1.5
    assert site_coefficients(0.375, 0.15, "C") == pytest.approx((1.3, 1.5))
    # clamp below first / above last column
    assert site_coefficients(0.05, 0.01, "D") == pytest.approx((1.6, 2.4))
    assert site_coefficients(3.0, 2.0, "C") == pytest.approx((1.2, 1.4))
    # D, Ss = 0.625: midpoint of the 0.5 (1.4) and 0.75 (1.2) columns
    assert site_coefficients(0.625, 0.1, "D")[0] == pytest.approx(1.3)


def test_site_class_b_without_vs_measurement_and_default_d():
    assert site_coefficients(0.75, 0.3, "B") == pytest.approx((0.9, 0.8))
    assert site_coefficients(0.75, 0.3, "B", vs_measured=False) == (1.0, 1.0)
    # default D: Fa >= 1.2 (11.4.4); Ss = 1.23 -> table 1.008 -> 1.2 (Fanella)
    assert site_coefficients(1.23, 0.1, "D", default_class=True)[0] == pytest.approx(1.2)
    assert site_coefficients(0.25, 0.1, "D", default_class=True)[0] == pytest.approx(1.6)


# ------------------------------------------------- hand-worked examples ---
def test_hand_example_site_c_at_column_values():
    """Ss = 1.0, S1 = 0.4, C: Fa = 1.2, Fv = 1.5 (table cells)."""
    m = ASCE7_16Spectrum(ss=1.0, s1=0.4, site_class="C", tl=8.0)
    sms, sm1 = 1.2 * 1.0, 1.5 * 0.4  # Eq. 11.4-1, 11.4-2
    sds, sd1 = 2 / 3 * sms, 2 / 3 * sm1  # Eq. 11.4-3, 11.4-4 -> 0.8, 0.4
    assert (m.fa, m.fv) == pytest.approx((1.2, 1.5))
    assert (m.sms, m.sm1, m.sds, m.sd1) == pytest.approx((sms, sm1, sds, sd1))
    ts, t0 = sd1 / sds, 0.2 * sd1 / sds  # 0.5 s, 0.1 s
    assert (m.ts, m.t0) == pytest.approx((ts, t0))
    assert m.sa(0.0)[0] == pytest.approx(0.4 * sds)  # Eq. 11.4-5 at T = 0
    assert m.sa(0.05)[0] == pytest.approx(sds * (0.4 + 0.6 * 0.05 / t0))
    assert m.sa(0.3)[0] == pytest.approx(0.8)  # plateau
    assert m.sa(2.0)[0] == pytest.approx(0.4 / 2.0)  # Eq. 11.4-6
    assert m.sa(10.0)[0] == pytest.approx(0.4 * 8.0 / 100.0)  # Eq. 11.4-7


def test_hand_example_site_b_c_with_interpolation():
    """Ss = 0.62, S1 = 0.17, Site Class C; and B with/without Vs measurement."""
    # Fa(C): 0.5 -> 1.3, 0.75 -> 1.2: 1.3 - 0.1 * (0.12 / 0.25) = 1.252
    # Fv(C): 0.1 -> 1.5, 0.2 -> 1.5: 1.5
    m = ASCE7_16Spectrum(ss=0.62, s1=0.17, site_class="C", tl=6.0)
    fa, fv = 1.3 - 0.1 * 0.12 / 0.25, 1.5
    sds, sd1 = 2 / 3 * fa * 0.62, 2 / 3 * fv * 0.17  # 0.51750..., 0.17
    assert m.fa == pytest.approx(fa) and m.fv == pytest.approx(fv)
    assert m.sds == pytest.approx(sds) and m.sd1 == pytest.approx(sd1)
    T = 1.2  # > Ts = 0.17 / 0.5175 = 0.3285
    assert m.sa(T)[0] == pytest.approx(sd1 / T)
    # Site Class B, measured Vs: Fa = 0.9, Fv = 0.8, Ss = 0.75, S1 = 0.3
    # (Guide ex.: SDS = 2/3 * 0.9 * 0.75 = 0.45, SD1 = 2/3 * 0.8 * 0.3 = 0.16)
    b = ASCE7_16Spectrum(ss=0.75, s1=0.3, site_class="B", tl=8.0)
    assert (b.sds, b.sd1) == pytest.approx((0.45, 0.16))
    # not measured: Fa = Fv = 1.0
    b1 = ASCE7_16Spectrum(ss=0.75, s1=0.3, site_class="B", tl=8.0, vs_measured=False)
    assert (b1.sds, b1.sd1) == pytest.approx((0.5, 0.2))


def test_mcer_is_1p5_times_design():
    kw = dict(ss=0.9, s1=0.35, site_class="C", tl=8.0)
    d, mce = ASCE7_16Spectrum(**kw), ASCE7_16Spectrum(**kw, level="mcer")
    T = np.array([0.0, 0.05, 0.3, 1.0, 5.0, 12.0])
    np.testing.assert_allclose(mce.sa(T), 1.5 * d.sa(T))
    assert mce.sa(0.3)[0] == pytest.approx(d.sms)  # plateau = SMS
    assert mce.sds == d.sds  # attributes stay design-level


def test_shape_branches_and_continuity():
    m = ASCE7_16Spectrum(ss=1.5, s1=0.6, site_class="C", tl=4.0)
    for k in m.knee_periods():
        lo, hi = m.sa([k - 1e-9, k + 1e-9])
        assert lo == pytest.approx(hi, rel=1e-6)
    assert m.sa(6.0)[0] == pytest.approx(m.sd1 * 4.0 / 36.0)


def test_parameters_and_attributes():
    m = ASCE7_16Spectrum(ss=0.6, s1=0.2, site_class="C", tl=8.0)
    p = m.parameters()
    for key in ("ss", "s1", "site_class", "level", "Fa", "Fv", "SMS", "SM1", "SDS",
                "SD1", "T0", "Ts", "TL", "exception_applied"):
        assert key in p
    assert p["exception_applied"] == "none" and p["site_class"] == "C"
    assert m.knee_periods() == pytest.approx((m.t0, m.ts, m.tl))


# ------------------------------------------------------------ 11.4.8 ------
@pytest.mark.parametrize(
    "ss,s1,sc,match",
    [
        (1.0, 0.1, "E", "Site Class E with Ss >= 1.0"),
        (1.5, 0.1, "E", "11.4.8"),
        (0.5, 0.2, "D", "Site Class D with S1 >= 0.2"),
        (0.5, 0.2, "E", "Site Class E with S1 >= 0.2"),
    ],
)
def test_triggers_raise_by_default(ss, s1, sc, match):
    with pytest.raises(ValueError, match=match):
        ASCE7_16Spectrum(ss=ss, s1=s1, site_class=sc, tl=8.0)


@pytest.mark.parametrize(
    "ss,s1,sc",
    [(0.99, 0.1, "E"), (3.0, 0.19, "D"), (3.0, 3.0, "C"), (3.0, 3.0, "B"), (0.99, 0.19, "D")],
)
def test_untriggered_cases_do_not_raise(ss, s1, sc):
    ASCE7_16Spectrum(ss=ss, s1=s1, site_class=sc, tl=12.0)


def test_site_e_between_0p1_and_0p2_has_no_printed_fv():
    # Not a 11.4.8 trigger (S1 < 0.2), but Table 11.4-2 prints 4.2 only at S1 <= 0.1
    # and "See Section 11.4.8" at 0.2, so there is nothing to interpolate to.
    with pytest.raises(ValueError, match="no Fv for Site Class E"):
        ASCE7_16Spectrum(ss=0.5, s1=0.15, site_class="E", tl=12.0)


def test_exception_2_site_d_shape():
    """D, S1 >= 0.2: Cs from Eq. 12.8-2 up to 1.5 Ts, 1.5x Eq. 12.8-3/12.8-4 beyond."""
    m = ASCE7_16Spectrum(ss=1.5, s1=0.6, site_class="D", tl=8.0, allow_exception=True)
    # Fanella example: Fa = 1.0, Fv = 1.7 -> SDS = 1.00, SD1 = 0.68
    assert (m.sds, m.sd1) == pytest.approx((1.0, 0.68))
    ts = 0.68
    assert m.sa(0.0)[0] == pytest.approx(0.4)
    assert m.sa(ts)[0] == pytest.approx(1.0)
    assert m.sa(1.5 * ts)[0] == pytest.approx(1.0)  # still SDS (Eq. 12.8-2)
    assert m.sa(1.0)[0] == pytest.approx(1.0)  # 1.0 < 1.5 Ts = 1.02
    assert m.sa(2.0)[0] == pytest.approx(1.5 * 0.68 / 2.0)  # 1.5 SD1 / T
    assert m.sa(10.0)[0] == pytest.approx(1.5 * 0.68 * 8.0 / 100.0)  # 1.5 SD1 TL / T^2
    lo, hi = m.sa([1.5 * ts - 1e-9, 1.5 * ts + 1e-9])
    assert lo == pytest.approx(hi, rel=1e-6)  # continuous at 1.5 Ts
    assert 1.5 * ts in m.knee_periods()
    assert "D" in m.parameters()["exception_applied"]
    # MCER = 1.5x of the modified shape too
    mc = ASCE7_16Spectrum(ss=1.5, s1=0.6, site_class="D", tl=8.0, level="mcer",
                          allow_exception=True)
    np.testing.assert_allclose(mc.sa([0.5, 2.0, 10.0]), 1.5 * m.sa([0.5, 2.0, 10.0]))


def test_exception_2_needs_tl_above_1p5_ts():
    with pytest.raises(ValueError, match="1.5 Ts"):
        ASCE7_16Spectrum(ss=1.5, s1=0.6, site_class="D", tl=0.9, allow_exception=True)


def test_exception_1_site_e_fa_as_site_class_c():
    """E, Ss >= 1.0: Fa of Site Class C (Table 11.4-1), Fv tabulated (S1 < 0.2)."""
    m = ASCE7_16Spectrum(ss=1.25, s1=0.1, site_class="E", tl=8.0, allow_exception=True)
    assert m.fa == pytest.approx(1.2)  # C at Ss = 1.25
    assert m.fv == pytest.approx(4.2)
    assert m.sds == pytest.approx(2 / 3 * 1.2 * 1.25)
    assert m.sd1 == pytest.approx(2 / 3 * 4.2 * 0.1)
    assert "Fa=Fa(C)" in m.parameters()["exception_applied"]
    # spectrum shape is the unmodified one
    base = ASCE7_16Spectrum.from_sds_sd1(m.sds, m.sd1, 8.0)
    T = np.linspace(0, 12, 50)
    np.testing.assert_allclose(m.sa(T), base.sa(T))


def test_site_e_with_s1_above_0p1_has_no_printed_fv_even_with_exception():
    # Table 11.4-2 (p. 84): Site Class E, S1 > 0.1 -> "See Section 11.4.8", no value.
    # Exception 3 waives the hazard analysis only for T <= Ts with the ELF
    # procedure, where SDS alone governs, so no SD1 can be built from the table.
    with pytest.raises(ValueError, match="no Fv for Site Class E"):
        ASCE7_16Spectrum(ss=0.5, s1=0.3, site_class="E", tl=8.0, allow_exception=True)


def test_e_between_0p75_and_1_uses_c_value_at_the_open_cell():
    # Ss = 0.875: between 1.3 (E, 0.75) and the open cell closed with C (1.2)
    assert site_coefficients(0.875, 0.1, "E")[0] == pytest.approx(1.25)


def test_site_class_e_both_triggers_still_need_site_specific_sd1():
    with pytest.raises(ValueError, match="from_sds_sd1"):
        ASCE7_16Spectrum(ss=1.5, s1=0.5, site_class="E", tl=8.0, allow_exception=True)


# ----------------------------------------------------------- from_sds_sd1 ---
def test_from_sds_sd1_bypasses_tables_and_checks():
    m = ASCE7_16Spectrum.from_sds_sd1(1.1, 0.7, tl=6.0)
    assert (m.sds, m.sd1, m.sms, m.sm1) == pytest.approx((1.1, 0.7, 1.65, 1.05))
    assert m.parameters()["site_class"] == "site-specific"
    assert m.sa(2.0)[0] == pytest.approx(0.35)
    mc = ASCE7_16Spectrum.from_sds_sd1(1.1, 0.7, tl=6.0, level="mcer")
    assert mc.sa(2.0)[0] == pytest.approx(0.525)
    with pytest.raises(ValueError):
        ASCE7_16Spectrum.from_sds_sd1(1.0, 0.7, tl=0.5)  # TL <= Ts
    with pytest.raises(ValueError):
        ASCE7_16Spectrum.from_sds_sd1(-1.0, 0.7, tl=6.0)


# ------------------------------------------------------------ error paths ---
def test_error_paths():
    kw = dict(ss=0.5, s1=0.1, site_class="C", tl=8.0)
    with pytest.raises(ValueError, match="Site Class F"):
        ASCE7_16Spectrum(**{**kw, "site_class": "F"})
    with pytest.raises(ValueError, match="site_class"):
        ASCE7_16Spectrum(**{**kw, "site_class": "G"})
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ss"):
            ASCE7_16Spectrum(**{**kw, "ss": bad})
        with pytest.raises(ValueError, match="s1"):
            ASCE7_16Spectrum(**{**kw, "s1": bad})
    with pytest.raises(ValueError, match="level"):
        ASCE7_16Spectrum(**{**kw, "level": "x"})
    with pytest.raises(ValueError):
        ASCE7_16Spectrum(**kw).sa(-1.0)


def test_tl_not_exceeding_ts_raises():
    # B, Ss = 0.2, S1 = 0.6: Ts = (0.8*0.6) / (0.9*0.2) = 2.67 s
    with pytest.raises(ValueError, match="TL"):
        ASCE7_16Spectrum(ss=0.2, s1=0.6, site_class="B", tl=2.0)


# --------------------------------------------------------------- registry ---
def test_registry_and_composite_round_trip():
    assert get_model_class("ASCE 7-16") is ASCE7_16Spectrum
    t = np.arange(0.0, 20.0, 0.01)
    rec = Record(x=0.1 * np.sin(2 * np.pi * t), dt=0.01)
    cs = rec.code_spectrum
    cs.add("ASCE7-16", ss=1.0, s1=0.4, site_class="C", tl=8.0)
    sa = cs.compute()["ASCE7-16"]
    ref = ASCE7_16Spectrum(ss=1.0, s1=0.4, site_class="C", tl=8.0)
    np.testing.assert_allclose(sa, ref.sa(cs.periods))
    assert cs.summary().loc["ASCE7-16", "SDS"] == pytest.approx(0.8)

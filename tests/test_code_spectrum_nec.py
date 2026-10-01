"""NEC-15 (NEC-SE-DS) spectrum against the printed tables and hand calculations.

Table values are transcribed from the official PDF (NEC-SE-DS, printed pages
31-32), read from the rendered pages, and agree with the apeSoil and
nec-ds-sismico-skill tables.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import get_model_class
from apeQuake.code_spectrum.base import G
from apeQuake.code_spectrum.codes.nec import NECSpectrum, site_coefficients

Z_COLS = [0.15, 0.25, 0.30, 0.35, 0.40, 0.50]

# Printed tables (rows = site class, columns = Z_COLS).
TABLE3_FA = {  # p. 31
    "A": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    "B": [1, 1, 1, 1, 1, 1],
    "C": [1.4, 1.3, 1.25, 1.23, 1.2, 1.18],
    "D": [1.6, 1.4, 1.3, 1.25, 1.2, 1.12],
    "E": [1.8, 1.4, 1.25, 1.1, 1.0, 0.85],
}
TABLE4_FD = {  # p. 31
    "A": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    "B": [1, 1, 1, 1, 1, 1],
    "C": [1.36, 1.28, 1.19, 1.15, 1.11, 1.06],
    "D": [1.62, 1.45, 1.36, 1.28, 1.19, 1.11],
    "E": [2.1, 1.75, 1.7, 1.65, 1.6, 1.5],
}
TABLE5_FS = {  # p. 32
    "A": [0.75] * 6,
    "B": [0.75] * 6,
    "C": [0.85, 0.94, 1.02, 1.06, 1.11, 1.23],
    "D": [1.02, 1.06, 1.11, 1.19, 1.28, 1.40],
    "E": [1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
}


# ------------------------------------------------------------- tables ---
@pytest.mark.parametrize("sc", list("ABCDE"))
@pytest.mark.parametrize("col", range(6))
def test_site_coefficients_match_printed_tables(sc, col):
    fa, fd, fs = site_coefficients(Z_COLS[col], sc)
    assert fa == pytest.approx(TABLE3_FA[sc][col])
    assert fd == pytest.approx(TABLE4_FD[sc][col])
    assert fs == pytest.approx(TABLE5_FS[sc][col])


def test_interpolation_in_z_is_linear_between_columns():
    # Site C, Z = 0.275: midpoint of the 0.25 and 0.30 columns.
    fa, fd, fs = site_coefficients(0.275, "C")
    assert fa == pytest.approx((1.30 + 1.25) / 2)
    assert fd == pytest.approx((1.28 + 1.19) / 2)
    assert fs == pytest.approx((0.94 + 1.02) / 2)
    # Site E, Z = 0.45: midpoint of 0.40 and 0.50.
    fa, fd, fs = site_coefficients(0.45, "E")
    assert fa == pytest.approx((1.00 + 0.85) / 2)
    assert fd == pytest.approx((1.60 + 1.50) / 2)
    assert fs == pytest.approx((1.90 + 2.00) / 2)
    # Site D, Z = 0.20: 1/2 between 0.15 and 0.25.
    assert site_coefficients(0.20, "D")[0] == pytest.approx(1.50)


def test_z_outside_table_clamps_to_end_columns():
    assert site_coefficients(0.80, "D") == pytest.approx(site_coefficients(0.50, "D"))
    assert site_coefficients(0.05, "D") == pytest.approx(site_coefficients(0.15, "D"))


# --------------------------------------------------- worked examples ---
def test_quito_site_d_hand_calculation():
    # Z=0.40 (Pichincha), site D, Sierra. Table column Z=0.40: Fa=1.20, Fd=1.19, Fs=1.28.
    m = NECSpectrum(z=0.40, site_class="D", region="sierra")
    p = m.parameters()
    assert (p["Fa"], p["Fd"], p["Fs"]) == pytest.approx((1.20, 1.19, 1.28))
    assert p["eta"] == 2.48 and p["r"] == 1.0
    k = 1.28 * 1.19 / 1.20  # Fs*Fd/Fa
    assert p["T0"] == pytest.approx(0.10 * k) and p["T0"] == pytest.approx(0.1269, abs=1e-4)
    assert p["Tc"] == pytest.approx(0.55 * k) and p["Tc"] == pytest.approx(0.6981, abs=1e-4)
    assert p["TL"] == pytest.approx(2.4 * 1.19)  # 2.856 s < 4 s cap
    plateau = 2.48 * 0.40 * 1.20  # 1.1904 g
    assert m.sa(0.5)[0] == pytest.approx(plateau)
    assert m.sa(1.0)[0] == pytest.approx(plateau * 0.55 * k / 1.0)  # 0.8311
    assert m.sa(2.0)[0] == pytest.approx(plateau * 0.55 * k / 2.0)  # 0.4156
    assert m.sa(1.0)[0] == pytest.approx(0.8311, abs=1e-4)


def test_manta_site_e_hand_calculation():
    # Z=0.50 (Manabi), site E, Costa. Column Z>=0.50: Fa=0.85, Fd=1.50, Fs=2.00; r=1.5.
    m = NECSpectrum(z=0.50, site_class="E", region="costa")
    p = m.parameters()
    assert (p["Fa"], p["Fd"], p["Fs"]) == pytest.approx((0.85, 1.50, 2.00))
    assert p["eta"] == 1.80 and p["r"] == 1.5
    k = 2.00 * 1.50 / 0.85
    assert p["T0"] == pytest.approx(0.3529, abs=1e-4)
    assert p["Tc"] == pytest.approx(0.55 * k) and p["Tc"] == pytest.approx(1.9412, abs=1e-4)
    assert p["TL"] == pytest.approx(3.60)
    plateau = 1.80 * 0.50 * 0.85  # 0.765 g
    assert m.sa(1.0)[0] == pytest.approx(plateau)
    assert m.sa(3.0)[0] == pytest.approx(plateau * (0.55 * k / 3.0) ** 1.5)
    assert m.sa(3.0)[0] == pytest.approx(0.3982, abs=1e-4)


# ------------------------------------------------------------- shape ---
@pytest.mark.parametrize("asc", [False, True])
@pytest.mark.parametrize("sc,region", [("D", "sierra"), ("E", "costa"), ("C", "oriente")])
def test_continuity_plateau_and_decay(sc, region, asc):
    m = NECSpectrum(z=0.35, site_class=sc, region=region, ascending_branch=asc)
    p = m.parameters()
    for k in (p["T0"], p["Tc"]):
        lo, hi = m.sa([k - 1e-9, k + 1e-9])
        assert lo == pytest.approx(hi, rel=1e-6)
    plateau = p["eta"] * p["Z"] * p["Fa"]
    assert m.sa(0.5 * (p["T0"] + p["Tc"]))[0] == pytest.approx(plateau)
    tail = m.sa(np.linspace(p["Tc"], 6.0, 200))
    assert np.all(np.diff(tail) < 0)
    assert tail[0] == pytest.approx(plateau)


def test_default_is_full_fig3_curve_and_plateau_from_zero_is_opt_out():
    # Default: Fig. 3 with the left branch, Sa(0) = Z Fa (matches the office
    # spreadsheet "NEC_SE_DS - ASCE 7-16 (2016 v.3)", which draws the full curve).
    m = NECSpectrum(z=0.4, site_class="D", region="sierra")
    assert m.ascending_branch is True
    assert m.sa(0.0)[0] == pytest.approx(0.4 * 1.2)
    # NEC 3.3.1 / 10.1.2: fundamental mode & static analysis use the plateau at T=0.
    p = NECSpectrum(z=0.4, site_class="D", region="sierra", ascending_branch=False)
    assert p.sa(0.0)[0] == pytest.approx(2.48 * 0.4 * 1.2)
    h = m
    t0 = h.t0
    assert h.sa(0.5 * t0)[0] == pytest.approx(0.4 * 1.2 * (1 + 1.48 * 0.5))
    assert h.sa(t0)[0] == pytest.approx(2.48 * 0.4 * 1.2)


def test_td_cap_and_sd_branch():
    # Site E at Z=0.15: 2.4 * 2.10 = 5.04 s, capped to 4 s (3.3.1 note, p. 35).
    m = NECSpectrum(z=0.15, site_class="E", region="costa")
    assert m.tl == pytest.approx(4.0)
    # Site C is not capped by the NEC note: 2.4 * 1.36 = 3.264 s.
    assert NECSpectrum(z=0.15, site_class="C", region="costa").tl == pytest.approx(3.264)
    # 3.3.2: Sd = Sa g (T/2pi)^2 up to TL, constant (Fig. 4) after it.
    T = 2.0
    assert m.sd(T)[0] == pytest.approx(m.sa(T)[0] * G * (T / (2 * np.pi)) ** 2)
    assert m.sd(7.0)[0] == pytest.approx(m.sd(4.0)[0])
    assert m.sd(4.0)[0] == pytest.approx(m.sa(4.0)[0] * G * (4.0 / (2 * np.pi)) ** 2)


# ---------------------------------------------------------- regions ---
@pytest.mark.parametrize(
    "region,eta",
    [("costa", 1.80), ("Sierra", 2.48), ("oriente", 2.60), ("galapagos", 2.48),
     ("esmeraldas", 2.48)],
)
def test_eta_by_region_section_3_3_1(region, eta):
    assert NECSpectrum(z=0.3, site_class="C", region=region).parameters()["eta"] == eta


def test_overrides_skip_table_lookup():
    m = NECSpectrum(z=0.3, site_class="C", eta=2.0, fa=1.1, fd=1.2, fs=0.9)
    p = m.parameters()
    assert (p["eta"], p["Fa"], p["Fd"], p["Fs"]) == (2.0, 1.1, 1.2, 0.9)
    assert m.sa(0.3)[0] == pytest.approx(2.0 * 0.3 * 1.1)  # plateau (T0 = 0.098 < 0.3 < Tc = 0.54)
    assert p["T0"] == pytest.approx(0.1 * 0.9 * 1.2 / 1.1)


# ---------------------------------------------------------- vertical ---
def test_vertical_is_two_thirds_of_horizontal():
    h = NECSpectrum(z=0.4, site_class="D", region="sierra")
    v = NECSpectrum(z=0.4, site_class="D", region="sierra", component="vertical")
    T = np.linspace(0.0, 5.0, 51)
    assert v.sa(T) == pytest.approx(2.0 / 3.0 * h.sa(T))


# ------------------------------------------------------- reduced Sa ---
def test_reduced_sa_is_base_shear_coefficient_of_6_3_2():
    m = NECSpectrum(z=0.4, site_class="D", region="sierra")
    # V/W = I Sa / (R phiP phiE) = 1.3 * 1.1904 / (8 * 0.9 * 0.9)
    expected = 1.3 * 1.1904 / (8 * 0.9 * 0.9)
    got = m.reduced_sa(0.5, r_factor=8, phi_p=0.9, phi_e=0.9, importance=1.3)[0]
    assert got == pytest.approx(expected)
    # Never uses the ascending branch, even when the model has it enabled.
    h = NECSpectrum(z=0.4, site_class="D", region="sierra", ascending_branch=True)
    assert h.reduced_sa(0.0, r_factor=1.0)[0] == pytest.approx(2.48 * 0.4 * 1.2)
    # Beyond Tc it follows the elastic decay.
    assert m.reduced_sa(2.0, r_factor=4.0)[0] == pytest.approx(m.sa(2.0)[0] / 4.0)


# ------------------------------------------------------------ errors ---
def test_site_class_f_requires_site_specific_study():
    with pytest.raises(ValueError, match="site-specific"):
        NECSpectrum(z=0.4, site_class="F", region="sierra")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(z=0.4, site_class="G"),
        dict(z=0.4, site_class="D", region="mars"),
        dict(z=0.0, site_class="D"),
        dict(z=-0.1, site_class="D"),
        dict(z=float("nan"), site_class="D"),
        dict(z=0.4, site_class="D", eta=1.0),
        dict(z=0.4, site_class="D", fa=0.0),
        dict(z=0.4, site_class="D", component="lateral"),
    ],
)
def test_bad_inputs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        NECSpectrum(**kwargs)


def test_bad_periods_and_reduction_factors():
    m = NECSpectrum(z=0.4, site_class="D", region="sierra")
    with pytest.raises(ValueError):
        m.sa(-0.1)
    with pytest.raises(ValueError):
        m.reduced_sa(1.0, r_factor=0.0)
    with pytest.raises(ValueError):
        m.reduced_sa(1.0, r_factor=5.0, importance=-1.0)


def test_knee_periods_and_registry():
    m = NECSpectrum(z=0.4, site_class="D", region="sierra")
    assert m.knee_periods() == (m.t0, m.tc, m.tl)
    for alias in ("NEC", "NEC15", "NEC-SE-DS", "nec-15"):
        assert get_model_class(alias) is NECSpectrum


# --------------------------------------------------------- composite ---
def test_round_trip_through_record_composite():
    t = np.arange(0.0, 20.0, 0.01)
    rec = Record(x=0.05 * np.sin(2 * np.pi * t), dt=0.01)
    model = rec.code_spectrum.add("NEC", z=0.4, site_class="D", region="sierra")
    assert isinstance(model, NECSpectrum)
    rec.code_spectrum.compute()
    cs = rec.code_spectrum
    assert "NEC-15" in cs.Sa
    i = int(np.argmin(np.abs(cs.periods - 0.5)))
    assert cs.Sa["NEC-15"][i] == pytest.approx(model.sa(cs.periods[i])[0])
    assert cs.Sa["NEC-15"].max() == pytest.approx(2.48 * 0.4 * 1.2)

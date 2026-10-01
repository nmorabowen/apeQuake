"""ASCE 7-22 two-period spectrum: hand-worked references and error paths."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import get_model_class
from apeQuake.code_spectrum.codes.asce7_22 import ASCE7_22Spectrum


def test_registered_under_aliases():
    assert get_model_class("ASCE 7-22") is ASCE7_22Spectrum
    assert get_model_class("asce7_22") is ASCE7_22Spectrum


def test_hand_worked_design_example():
    # SMS = 1.50, SM1 = 0.90 -> SDS = 1.00, SD1 = 0.60 (Eqs. 11.4-1, 11.4-2)
    # Ts = 0.6, T0 = 0.12 (Section 11.4.5.2); TL = 8 s.
    m = ASCE7_22Spectrum(sms=1.5, sm1=0.9, tl=8.0, site_class="D")
    assert m.sds == pytest.approx(1.0) and m.sd1 == pytest.approx(0.6)
    assert m.ts == pytest.approx(0.6) and m.t0 == pytest.approx(0.12)
    assert m.sa(0.0)[0] == pytest.approx(0.4)  # 0.4 SDS (Eq. 11.4-3)
    assert m.sa(0.06)[0] == pytest.approx(0.7)  # 1.0 (0.4 + 0.6 * 0.06/0.12)
    assert m.sa(0.4)[0] == pytest.approx(1.0)  # plateau
    assert m.sa(2.0)[0] == pytest.approx(0.3)  # SD1/T (Eq. 11.4-4)
    assert m.sa(10.0)[0] == pytest.approx(0.6 * 8 / 100)  # SD1 TL/T^2 (Eq. 11.4-5)


def test_hand_worked_second_example_non_round():
    # SMS = 0.90, SM1 = 0.60 -> SDS = 0.60, SD1 = 0.40, Ts = 2/3, TL = 6.
    m = ASCE7_22Spectrum(0.9, 0.6, 6.0, "CD")
    assert m.ts == pytest.approx(2 / 3)
    assert m.sa(1.0)[0] == pytest.approx(0.4)  # SD1/T
    assert m.sa(3.0)[0] == pytest.approx(0.4 / 3)
    assert m.sa(8.0)[0] == pytest.approx(0.4 * 6 / 64)
    assert m.sa(1 / 15)[0] == pytest.approx(0.6 * (0.4 + 0.6 * 0.5))  # T = T0/2


def test_mcer_is_1p5_times_design():
    d = ASCE7_22Spectrum(1.5, 0.9, 8.0, "D")
    e = ASCE7_22Spectrum(1.5, 0.9, 8.0, "D", level="mcer")
    T = np.linspace(0, 12, 50)
    np.testing.assert_allclose(e.sa(T), 1.5 * d.sa(T))  # Section 11.4.6
    assert e.sa(0.4)[0] == pytest.approx(1.5)  # SMS plateau
    assert e.sa(1.0)[0] == pytest.approx(0.9)  # SM1 / T at T = 1
    assert e.parameters()["SDS"] == pytest.approx(1.0)  # design values reported


def test_from_sds_sd1():
    m = ASCE7_22Spectrum.from_sds_sd1(1.0, 0.6, 8.0)
    ref = ASCE7_22Spectrum(1.5, 0.9, 8.0, "D")
    T = np.linspace(0, 12, 40)
    np.testing.assert_allclose(m.sa(T), ref.sa(T))
    assert m.parameters()["SMS"] == pytest.approx(1.5)
    # Site-specific entry (e.g. a Class F study) is allowed here.
    assert ASCE7_22Spectrum.from_sds_sd1(0.8, 0.4, 6.0, site_class="F").site_class == "F"
    with pytest.raises(ValueError):
        ASCE7_22Spectrum.from_sds_sd1(0.0, 0.4, 6.0)
    with pytest.raises(ValueError):
        ASCE7_22Spectrum.from_sds_sd1(1.0, 0.9, 0.5)  # TL <= Ts


@pytest.mark.parametrize("sc", ["A", "B", "BC", "C", "CD", "D", "DE", "E", "bc"])
def test_site_classes_accepted(sc):
    assert ASCE7_22Spectrum(1.0, 0.5, 8.0, sc).parameters()["site_class"] == sc.upper()


def test_parameters_keys():
    p = ASCE7_22Spectrum(1.5, 0.9, 8.0, "D").parameters()
    assert set(p) == {"SMS", "SM1", "SDS", "SD1", "TL", "site_class", "level", "T0", "Ts"}


def test_error_paths():
    with pytest.raises(ValueError, match="Site Class F"):
        ASCE7_22Spectrum(1.0, 0.5, 8.0, "F")
    with pytest.raises(ValueError, match="site_class"):
        ASCE7_22Spectrum(1.0, 0.5, 8.0, "Z")
    with pytest.raises(ValueError, match="> 0"):
        ASCE7_22Spectrum(0.0, 0.5, 8.0, "D")
    with pytest.raises(ValueError, match="> 0"):
        ASCE7_22Spectrum(1.0, -0.1, 8.0, "D")
    with pytest.raises(ValueError, match="TL"):
        ASCE7_22Spectrum(1.0, 1.5, 0.5, "D")  # Ts = 1.0 > TL
    with pytest.raises(ValueError, match="level"):
        ASCE7_22Spectrum(1.0, 0.5, 8.0, "D", level="x")


def test_composite_round_trip():
    t = np.arange(0.0, 30.0, 0.01)
    rec = Record(x=0.1 * np.sin(2 * np.pi * t), dt=0.01)
    rec.code_spectrum.add("ASCE7-22", sms=1.5, sm1=0.9, site_class="D", tl=8.0)
    rec.code_spectrum.compute()
    assert "ASCE7-22" in rec.code_spectrum.summary().to_string()

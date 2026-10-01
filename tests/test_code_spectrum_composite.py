"""CodeSpectrum composite + shared ASCE shape, against closed-form references."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from apeQuake import Record
from apeQuake.code_spectrum import (
    AsceTwoPeriodSpectrum,
    CodeSpectrum,
    CodeSpectrumModel,
    available_codes,
    get_model_class,
)


class _Asce(AsceTwoPeriodSpectrum):
    """Minimal concrete model: SDS=1.0 g, SD1=0.6 g, TL=8 s."""

    code = "TEST-ASCE"

    def __init__(self, sds=1.0, sd1=0.6, tl=8.0):
        self.sds, self.sd1, self.tl = sds, sd1, tl
        self._check_shape_inputs()

    def parameters(self):
        return {"SDS": self.sds, "SD1": self.sd1, "TL": self.tl}


def _sine_record(amp_g=0.1, period=1.0, dur=60.0, dt=0.005, units_per_g=1.0):
    t = np.arange(0.0, dur, dt)
    return Record(x=amp_g * units_per_g * np.sin(2 * np.pi * t / period), dt=dt)


# ---------------------------------------------------------------- shape ---
def test_asce_shape_values_and_corners():
    m = _Asce()
    assert m.ts == pytest.approx(0.6)
    assert m.t0 == pytest.approx(0.12)
    assert m.sa(0.0)[0] == pytest.approx(0.4)  # 0.4 SDS
    assert m.sa(0.06)[0] == pytest.approx(0.7)  # 1.0 (0.4 + 0.6 * 0.5)
    assert m.sa(0.3)[0] == pytest.approx(1.0)  # plateau
    assert m.sa(2.0)[0] == pytest.approx(0.3)  # SD1 / T
    assert m.sa(10.0)[0] == pytest.approx(0.6 * 8.0 / 100.0)  # SD1 TL / T^2


def test_asce_shape_is_continuous_at_every_corner():
    m = _Asce()
    for k in m.knee_periods():
        lo, hi = m.sa([k - 1e-9, k + 1e-9])
        assert lo == pytest.approx(hi, rel=1e-6)


def test_sd_is_sa_times_g_times_omega_inverse_squared():
    m = _Asce()
    T = 1.5
    assert m.sd(T)[0] == pytest.approx(m.sa(T)[0] * 9.80665 * (T / (2 * np.pi)) ** 2)


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
def test_negative_or_nonfinite_periods_rejected(bad):
    with pytest.raises(ValueError):
        _Asce().sa(bad)


def test_tl_must_exceed_ts():
    with pytest.raises(ValueError):
        _Asce(tl=0.5)


def test_default_periods_contain_knees():
    m = _Asce()
    T = m.default_periods()
    for k in m.knee_periods():
        assert np.any(np.isclose(T, k, atol=1e-12)) or k > T.max()


# ------------------------------------------------------------ registry ---
def test_registry_aliases_and_unknown_code():
    assert get_model_class("asce 7-16") is get_model_class("ASCE7_16")
    with pytest.raises(KeyError, match="Unknown code"):
        get_model_class("EC8")
    assert {"NEC-15", "ASCE7-10", "ASCE7-16", "ASCE7-22"} <= set(available_codes())


# ----------------------------------------------------------- composite ---
def test_composite_attached_and_does_not_mutate_record_df():
    rec = _sine_record()
    before = rec.df.copy()
    rec.code_spectrum.add(_Asce())
    rec.code_spectrum.compute()
    rec.code_spectrum.compute_record()
    assert rec.df.equals(before)


def test_add_label_collisions_and_invalidation():
    cs = _sine_record().code_spectrum
    cs.add(_Asce())
    cs.compute()
    with pytest.raises(ValueError):
        cs.add(_Asce())
    cs.add(_Asce(sds=0.8), label="soft")
    assert cs.periods is None and cs.Sa == {}  # cache dropped on add
    cs.compute()
    assert set(cs.Sa) == {"TEST-ASCE", "soft"}
    cs.remove("soft")
    assert cs.Sa == {}


def test_compute_requires_a_model_and_plot_requires_compute():
    cs = _sine_record().code_spectrum
    with pytest.raises(RuntimeError):
        cs.compute()
    cs.add(_Asce())
    with pytest.raises(RuntimeError, match="compute"):
        cs.plot(show=False)


@pytest.mark.parametrize("units_per_g", [1.0, 9.80665, 981.0])
def test_record_spectrum_at_resonance_matches_closed_form(units_per_g):
    """Steady state at resonance: Sa = A / (2 xi) -> 0.1 g / 0.1 = 1.0 g."""
    rec = _sine_record(amp_g=0.1, period=1.0, units_per_g=units_per_g)
    cs = rec.code_spectrum
    cs.add(_Asce())
    cs.compute(periods=[0.5, 1.0, 2.0])
    cs.compute_record(units_per_g=units_per_g)
    sa = cs.record_Sa["X"]
    assert sa[1] == pytest.approx(1.0, rel=0.03)
    assert sa[0] < 0.5 * sa[1] and sa[2] < 0.5 * sa[1]  # off-resonance is far lower


def test_compute_record_skips_zero_period():
    cs = _sine_record().code_spectrum
    cs.add(_Asce())
    cs.compute()  # grid includes T = 0
    cs.compute_record()
    assert cs._record_periods.min() > 0.0
    cs.plot(show=False)


def test_summary_and_table():
    cs = _sine_record().code_spectrum
    cs.add(_Asce())
    assert cs.summary().loc["TEST-ASCE", "SDS"] == 1.0
    assert list(_Asce().table().columns) == ["T", "Sa", "Sd"]

"""NEC-SE-DS 6.3 force-based design: I, Ta, base shear and Fx.

Checked against NEC-SE-DS (Tabla 6, 6.3.2, 6.3.3, 6.3.5); agreement with
codeSpectra's ``nec.elf.base_shear`` was checked outside the suite (36 cases).
"""
import numpy as np
import pytest

from apeQuake.code_spectrum.codes import NECSpectrum
from apeQuake.design import nec

SP = NECSpectrum(z=0.4, site_class="D", region="costa")


def test_importance_tabla_6():
    assert [nec.importance_factor(c) for c in ("esencial", "especial", "otra")] == [1.5, 1.3, 1.0]
    with pytest.raises(ValueError):
        nec.importance_factor("vivienda")


def test_period_method_1_and_cap():
    assert np.isclose(nec.approximate_period(20.0, "hormigon_porticos"), 0.055 * 20**0.9)
    assert np.isclose(nec.approximate_period(20.0, "acero_sin_arriostramientos"), 0.072 * 20**0.8)
    assert np.isclose(nec.limit_period_method_2(2.0, 1.0), 1.3)
    assert nec.limit_period_method_2(1.1, 1.0) == 1.1


def test_k_exponent():
    assert nec.vertical_distribution_exponent(0.5) == 1.0
    assert np.isclose(nec.vertical_distribution_exponent(1.0), 1.25)
    assert nec.vertical_distribution_exponent(2.5) == 2.0
    assert nec.vertical_distribution_exponent(3.0) == 2.0


def test_base_shear_formula_and_plateau():
    h = np.arange(1, 6) * 3.0
    w = np.full(5, 200.0)
    r = nec.base_shear(SP, W=w.sum(), R=8.0, I=1.3, phi_p=0.9, phi_e=0.9, hn=15.0,
                       weights=w, heights=h)
    Ta = 0.055 * 15**0.9
    sa = SP.reduced_sa(Ta, r_factor=1.0)[0]           # elastic, plateau to T = 0
    assert np.isclose(r.Ta, Ta) and np.isclose(r.Sa_Ta, sa)
    assert np.isclose(r.V, 1.3 * sa * w.sum() / (8.0 * 0.9 * 0.9))
    assert np.isclose(r.shear_coefficient, r.V / w.sum())
    k = nec.vertical_distribution_exponent(Ta)
    np.testing.assert_allclose(r.Fx, r.V * w * h**k / np.sum(w * h**k))
    # A short period still uses the plateau (no ascending branch in static design).
    short = nec.base_shear(SP, W=1000.0, R=8.0, T=0.01)
    assert np.isclose(short.Sa_Ta, SP.reduced_sa(0.5, r_factor=1.0)[0])


def test_method_2_period_is_capped():
    r = nec.base_shear(SP, W=1000.0, R=6.0, T=3.0, hn=15.0)
    assert np.isclose(r.Ta, 1.3 * 0.055 * 15**0.9) and np.isclose(r.Ta_method_1, 0.055 * 15**0.9)


def test_validation():
    with pytest.raises(ValueError, match="horizontal"):
        nec.base_shear(NECSpectrum(z=0.4, site_class="D", component="vertical"), W=1.0, R=6.0, T=1.0)
    with pytest.raises(ValueError, match="T, the height"):
        nec.base_shear(SP, W=1.0, R=6.0)

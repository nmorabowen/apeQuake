"""Alternative periods, the Section 12.8.1.3 SDS limit and the Section 11.6 exception.

Hand calculations below follow the clauses as printed (ASCE 7-10 p. 72, 7-16
p. 101-102, 7-22 p. 124-125; NEC-SE-DS p. 62).
"""
import math

import numpy as np
import pytest

from apeQuake.code_spectrum.codes import NECSpectrum
from apeQuake.design import asce7, nec

WALLS = [(2.0, 5.0), (2.0, 5.0)]          # Ai [m2], Di [m]
HN, AB = 20.0, 400.0                      # m, m2


# ----------------------------------------------------------- Ta = 0.1 N
def test_ta_from_stories():
    assert asce7.approximate_period_stories(8, average_story_height=3.2) == pytest.approx(0.8)
    assert asce7.approximate_period_stories(
        12, "steel_moment_frame", average_story_height=10.0, metric=False) == pytest.approx(1.2)


@pytest.mark.parametrize("n, kind, h, match", [
    (8, "other", 3.2, "moment frames"),
    (13, "concrete_moment_frame", 3.2, "12 stories"),
    (2.5, "concrete_moment_frame", 3.2, "12 stories"),
    (8, "concrete_moment_frame", 2.8, "3 m"),
])
def test_ta_from_stories_conditions(n, kind, h, match):
    with pytest.raises(ValueError, match=match):
        asce7.approximate_period_stories(n, kind, average_story_height=h)


# -------------------------------------------------------- shear-wall period
def test_shear_wall_period_by_hand():
    cw = 100.0 / AB * sum(a / (1.0 + 0.83 * (HN / d) ** 2) for a, d in WALLS)
    ref = 0.0019 * (HN / 0.3048) / math.sqrt(cw)
    assert asce7.shear_wall_period(HN, AB, WALLS) == pytest.approx(ref, rel=1e-12)
    assert asce7.shear_wall_period(HN, AB, WALLS, edition="7-22") == pytest.approx(ref, rel=1e-12)
    # Cw is dimensionless, so the same building in ft gives the same period
    ft = 0.3048
    walls_ft = [(a / ft**2, d / ft) for a, d in WALLS]
    assert asce7.shear_wall_period(HN / ft, AB / ft**2, walls_ft, metric=False) == \
        pytest.approx(ref, rel=1e-12)


def test_shear_wall_period_is_not_the_printed_si_coefficient():
    """0.0019 s/ft is 0.00623 s/m (NEC rounds it to 0.0062); the printed 0.00058 m is ~1/10."""
    cw = 100.0 / AB * sum(a / (1.0 + 0.83 * (HN / d) ** 2) for a, d in WALLS)
    ta = asce7.shear_wall_period(HN, AB, WALLS)
    assert ta == pytest.approx(0.0062 * HN / math.sqrt(cw), rel=1e-2)
    assert ta > 5 * 0.00058 * HN / math.sqrt(cw)


def test_shear_wall_period_7_10_uses_wall_heights():
    short = [(2.0, 5.0, 10.0), (2.0, 5.0, 10.0)]
    cw = 100.0 / AB * sum(a / (1.0 + 0.83 * (h / d) ** 2) for a, d, h in short)
    ref = 0.0019 * (HN / 0.3048) / math.sqrt(cw)
    assert asce7.shear_wall_period(HN, AB, short, edition="7-10") == pytest.approx(ref)
    assert asce7.shear_wall_period(HN, AB, WALLS, edition="7-10") == \
        pytest.approx(asce7.shear_wall_period(HN, AB, WALLS))
    with pytest.raises(ValueError, match=r"\(Ai, Di\)"):
        asce7.shear_wall_period(HN, AB, short, edition="7-16")


def test_shear_wall_period_limits():
    with pytest.raises(ValueError, match="120 ft"):
        asce7.shear_wall_period(37.0, AB, WALLS)
    with pytest.raises(ValueError, match="at least one"):
        asce7.shear_wall_period(HN, AB, [])


def test_base_shear_with_an_alternative_ta():
    ta = asce7.approximate_period_stories(8, average_story_height=3.2)
    common = {"W": 1000.0, "R": 8.0, "sds": 1.0, "sd1": 0.6, "s1": 0.6, "tl": 8.0}
    r = asce7.base_shear(**common, ta=ta)
    assert (r.T, r.Ta, r.Cu) == (pytest.approx(0.8), pytest.approx(0.8), 1.4)
    r = asce7.base_shear(**common, ta=ta, T=2.0)                  # capped at Cu Ta
    assert r.T == pytest.approx(1.4 * 0.8)
    with pytest.raises(ValueError, match="not both"):
        asce7.base_shear(**common, ta=ta, hn=30.0)


# ------------------------------------------------------------- 12.8.1.3
SHORT = {"W": 1000.0, "R": 6.0, "sd1": 0.6, "s1": 0.6, "tl": 8.0, "T": 0.3, "n_stories": 3, "site_class": "D"}


@pytest.mark.parametrize("sds, used", [(1.5, 1.05), (1.2, 1.0), (0.8, 0.8)])
def test_sds_limit_12_8_1_3(sds, used):
    r = asce7.base_shear(**SHORT, sds=sds, max_sds_12_8_1_3=True)
    assert r.breakdown["SDS_12.8.1.3"] == pytest.approx(used)
    ref = asce7.base_shear(**{k: v for k, v in SHORT.items()
                              if k not in ("n_stories", "site_class")}, sds=used)
    assert r.Cs == pytest.approx(ref.Cs)


@pytest.mark.parametrize("change, match", [
    ({"edition": "7-10"}, "Ss at 1.5"),
    ({"T": 0.6}, "item 3"),
    ({"n_stories": 6}, "item 2"),
    ({"site_class": "E"}, "item 5"),
    ({"risk_category": "III"}, "item 6"),
    ({"edition": "7-22", "R": 2.5}, "item 7"),
    ({"n_stories": None}, "needs n_stories"),
])
def test_sds_limit_criteria(change, match):
    with pytest.raises(ValueError, match=match):
        asce7.base_shear(**{**SHORT, **change}, sds=1.5, max_sds_12_8_1_3=True)


# ----------------------------------------------------- 11.6 exception
def test_sdc_from_table_11_6_1_alone():
    base = asce7.seismic_design_category(0.4, 0.25, s1=0.3)
    assert (base.sdc, base.from_sds, base.from_sd1) == ("D", "C", "D")
    r = asce7.seismic_design_category(0.4, 0.25, s1=0.3, table_11_6_1_only=True,
                                      ta=0.3, t_drift=0.5, ts=0.6)
    assert (r.sdc, r.table_11_6_1_only) == ("C", True)
    e = asce7.seismic_design_category(0.4, 0.25, s1=0.8, table_11_6_1_only=True,
                                      ta=0.3, t_drift=0.5, ts=0.6)
    assert (e.sdc, e.s1_override, e.table_11_6_1_only) == ("E", True, False)


@pytest.mark.parametrize("ta, t_drift, match", [(0.5, 0.5, "item 1"), (0.3, 0.6, "item 2")])
def test_sdc_exception_conditions(ta, t_drift, match):
    with pytest.raises(ValueError, match=match):
        asce7.seismic_design_category(0.4, 0.25, s1=0.3, table_11_6_1_only=True,
                                      ta=ta, t_drift=t_drift, ts=0.6)
    with pytest.raises(ValueError, match="needs ta"):
        asce7.seismic_design_category(0.4, 0.25, s1=0.3, table_11_6_1_only=True)


# ------------------------------------------------------------ NEC walls
def test_nec_wall_period_by_hand():
    walls = [(2.0, 5.0, 20.0), (1.5, 4.0, 14.0)]                    # Awi, lwi, hwi
    cw = 100.0 / AB * sum((HN / h) ** 2 * a / (1.0 + 0.83 * (h / lw) ** 2)
                          for a, lw, h in walls)
    ref = 0.0062 / math.sqrt(cw) * HN
    assert nec.wall_period(HN, AB, walls) == pytest.approx(ref, rel=1e-12)
    with pytest.raises(ValueError, match="Awi, lwi, hwi"):
        nec.wall_period(HN, AB, [(2.0, 5.0)])


def test_nec_base_shear_with_wall_period():
    s = NECSpectrum(z=0.4, site_class="D", region="sierra")
    ta = nec.wall_period(HN, AB, [(2.0, 5.0, 20.0), (2.0, 5.0, 20.0)])
    r = nec.base_shear(s, W=1000.0, R=5.0, ta=ta)
    assert r.Ta == pytest.approx(ta) and r.Ta_method_1 == pytest.approx(ta)
    np.testing.assert_allclose(r.V, float(s.reduced_sa(ta, r_factor=5.0)[0]) * 1000.0)
    assert nec.base_shear(s, W=1000.0, R=5.0, ta=ta, T=2.0).Ta == pytest.approx(1.3 * ta)
    with pytest.raises(ValueError, match="not both"):
        nec.base_shear(s, W=1000.0, R=5.0, ta=ta, hn=20.0)

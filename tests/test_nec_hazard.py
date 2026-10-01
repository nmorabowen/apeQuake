"""NEC hazard curves: data integrity, interpolation, extrapolation, lookups."""
import numpy as np
import pytest

from apeQuake.nec import HazardCurve, load_hazard_database

db = load_hazard_database()
KEYS = ["PGA", "0.1", "0.2", "0.5", "1.0"]


def test_database_has_23_cities_with_five_curves():
    assert len(db) == 23
    for city in db:
        assert set(city.curves) == set(KEYS)


@pytest.mark.parametrize("city", list(db), ids=lambda c: c.name)
def test_curves_are_monotone_and_in_plot_bounds(city):
    for cur in city.curves.values():
        assert np.all(np.diff(cur.a) > 0)
        assert np.all(np.diff(cur.rate) < 0)
        assert 1e-5 <= cur.rate.min() and cur.rate.max() <= 1.0
        assert 0.0 <= cur.a[0] < 0.1 and cur.a[-1] <= city_xmax(city) + 1e-6


def city_xmax(city):
    return max(c.a[-1] for c in city.curves.values())


@pytest.mark.parametrize("city", list(db), ids=lambda c: c.name)
def test_period_ordering_at_475(city):
    """Short-period Sa exceeds PGA, and Sa(1 s) is the smallest, at Tr = 475 yr."""
    pga, s01, s02, s05, s10 = (city[k].a_at_return_period(475)[0] for k in KEYS)
    assert s10 < pga < max(s01, s02)
    assert s10 < s05


def test_rate_and_acceleration_round_trip():
    cur = db["Quito"]["PGA"]
    for tr in (72, 225, 475, 2500):
        a = cur.a_at_return_period(tr)[0]
        assert cur.rate_at(a)[0] == pytest.approx(1.0 / tr, rel=2e-3)


def test_quito_pga_475_close_to_nec_zone_factor():
    # NEC Table 19: Z(Quito) = 0.40 g.  Digitized PGA(475) must be nearby.
    assert db["Quito"]["PGA"].a_at_return_period(475)[0] == pytest.approx(0.40, abs=0.06)


def test_coastal_pga_475_exceeds_saturated_z():
    # Z saturates at 0.50 g; the Esmeraldas curve is well above it.
    assert db["Esmeraldas"]["PGA"].a_at_return_period(475)[0] > 0.5


def test_extrapolation_beyond_plotted_range_is_monotone():
    cur = db["Quito"]["1.0"]  # exits the plot through the 1e-5 floor at ~1.05 g
    rates = cur.rate_at(np.linspace(1.2, 3.0, 20))
    assert np.all(np.diff(rates) < 0)
    assert rates[0] < cur.rate[-1]
    assert np.isnan(cur.rate_at(2.0, extrapolate=False)[0])
    # a rate below the plotted floor is invertible and consistent
    a = cur.a_at_rate(1e-6)[0]
    assert a > cur.a[-1]
    assert cur.rate_at(a)[0] == pytest.approx(1e-6, rel=1e-2)


def test_extrapolation_back_test():
    """Hide the tail below 1e-3 and predict the acceleration at 1e-4 within 15 %."""
    errs = []
    for city in db:
        for k, cur in city.curves.items():
            if cur.aliased_from or cur.rate[-1] > 1e-4:
                continue
            m = cur.rate >= 1e-3
            if m.sum() < 30:
                continue
            truncated = HazardCurve(k, cur.a[m], cur.rate[m])
            pred = truncated.a_at_rate(1e-4)[0]
            true = float(np.exp(np.interp(np.log(1e-4), np.log(cur.rate[::-1]), np.log(cur.a[::-1]))))
            errs.append(pred / true - 1.0)
    errs = np.abs(errs)
    assert len(errs) > 50
    assert errs.mean() < 0.06
    assert np.percentile(errs, 90) < 0.15


def test_extrapolate_returns_extended_curve():
    cur = db["Tulcan"]["PGA"]
    ext = cur.extrapolate(2.5)
    assert ext.a[-1] == pytest.approx(2.5)
    assert len(ext.a) > len(cur.a)
    assert np.all(np.diff(ext.rate) < 0)


def test_probability_of_exceedance_matches_475_definition():
    cur = db["Quito"]["PGA"]
    a475 = cur.a_at_return_period(475)[0]
    assert cur.probability_of_exceedance(a475, 50.0)[0] == pytest.approx(0.10, abs=0.005)


def test_uhs_and_table():
    city = db["Guayaquil"]
    periods, sa = city.uhs(475)
    assert list(periods) == [0.0, 0.1, 0.2, 0.5, 1.0]
    assert sa.shape == (5,) and np.all(sa > 0)
    table = city.hazard_table()
    assert table.shape == (4, 5)
    assert table.loc["raro (Tr=475)", "PGA"] == pytest.approx(sa[0])
    # stronger shaking for rarer events
    assert table["PGA"].is_monotonic_increasing


def test_lookup_is_case_and_accent_insensitive():
    assert db["tulcán"].name == "Tulcan"
    assert db["SANTO DOMINGO"].name == "Santo Domingo"
    with pytest.raises(KeyError):
        db["Atlantis"]


def test_nearest_city():
    assert db.nearest(-0.18, -78.47).name == "Quito"
    assert db.nearest(-2.19, -79.89).name == "Guayaquil"


def test_getitem_by_period():
    city = db["Cuenca"]
    assert city[0] is city["PGA"]
    assert city[0.2] is city["0.2"]


def test_plot_smoke():
    import matplotlib

    matplotlib.use("Agg")
    ax = db["Quito"].plot(extrapolate_to=2.0)
    assert ax.get_yscale() == "log"

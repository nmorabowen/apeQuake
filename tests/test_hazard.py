"""IG-EPN hazard database: lookup, published values, TR interpolation, curves."""
import os
import warnings

import numpy as np
import pytest

from apeQuake.hazard import (
    PERIODS,
    EcuadorHazard,
    poe_from_tr,
    powerlaw_fit,
    tr_from_poe,
)
from apeQuake.hazard.curves import powerlaw_rate, powerlaw_sa


@pytest.fixture(scope="module")
def hz():
    return EcuadorHazard()


@pytest.fixture(scope="module")
def quito(hz):
    return hz.site("Quito")


# ---------------------------------------------------------------- curve math

def test_tr_poe_round_trip():
    assert tr_from_poe(0.10, 50) == pytest.approx(474.56, abs=0.01)
    assert tr_from_poe(0.02, 50) == pytest.approx(2474.9, abs=0.1)
    assert poe_from_tr(tr_from_poe(0.05, 50), 50) == pytest.approx(0.05)


def test_powerlaw_fit_is_exact_at_both_points():
    sa1, sa2 = np.array([0.4, 1.0]), np.array([0.8, 1.9])
    k0, k = powerlaw_fit(sa1, 475, sa2, 2475)
    np.testing.assert_allclose(powerlaw_rate(sa1, k0, k), 1 / 475)
    np.testing.assert_allclose(powerlaw_rate(sa2, k0, k), 1 / 2475)
    np.testing.assert_allclose(powerlaw_sa(475, k0, k), sa1)


# ---------------------------------------------------------------- database

def test_inventory_counts(hz):
    inv = hz.inventory().set_index("dataset")["rows"]
    assert inv["hazard cells"] == 3146
    assert inv["hazard grid (UHS)"] == 3146 * 2 * 4
    assert inv["hazard curves (digitized)"] == 365


def test_published_values_match_igepn_txt(hz):
    # Values of the IG-EPN UHS .txt attachment for cell S4.4480.44, TR 475
    s = hz.site(-4.44, -80.44)
    assert s.cell_id == "S4.4480.44"
    np.testing.assert_allclose(
        s.published(475, "mean"),
        [0.3547, 0.5765, 0.7432, 0.9155, 0.7817, 0.3700, 0.1847, 0.0813])
    np.testing.assert_allclose(s.published(475, "q84")[0], 0.4318)


def test_quantiles_are_ordered(hz):
    g = hz.hazard_map(475, 0.0, "q16").sa.to_numpy()
    m = hz.hazard_map(475, 0.0, "q50").sa.to_numpy()
    h = hz.hazard_map(475, 0.0, "q84").sa.to_numpy()
    assert np.all(g <= m) and np.all(m <= h)


def test_catalog_depths_positive_down(hz):
    for kind in ("historical", "deep"):
        assert hz.catalog(kind).depth_km.median() > 0


# ---------------------------------------------------------------- lookup

def test_merged_cell_lookup_uses_polygons(hz):
    # Guayaquil: its lattice position is merged into cell S2.2079.96; a nearest-centroid
    # lookup would wrongly call the point outside the grid.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = hz.site(-2.19, -79.89)
    assert s.cell_id == "S2.2079.96"


def test_point_on_cell_corner_is_inside(hz):
    # (-1.0, -78.0) is the corner shared by four cells
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = hz.site(-1.0, -78.0)
    assert s.distance_km < 7


def test_lookup_by_name(hz, quito):
    assert quito.cell_id == "S0.0478.52"
    assert hz.site("baños").label.startswith("BAÑOS")     # accent / case insensitive
    with pytest.raises(KeyError):
        hz.site("Atlantis")


def test_outside_grid_warns_or_raises(hz):
    with pytest.warns(UserWarning, match="outside"):
        hz.site(-5.5, -85.0)
    with pytest.raises(ValueError):
        hz.site(-5.5, -85.0, strict=True)


def test_bilinear_at_centroid_equals_cell(hz):
    a = hz.site(-1.48, -78.52)
    b = hz.site(-1.48, -78.52, interp="bilinear")
    assert b.interp == "bilinear"
    np.testing.assert_allclose(a.published(475), b.published(475), rtol=1e-9)
    assert not b.has_digitized_curves


def test_batch_query(hz):
    t = hz.uhs_at(["Manta", (-1.0, -78.0)], tr=475)
    assert len(t) == 2 and set(PERIODS) <= set(t.columns)


# ---------------------------------------------------------------- UHS / curves

def test_uhs_published_and_interpolated(quito):
    u475 = quito.uhs(475)
    assert (u475.source == "published").all()
    u975 = quito.uhs(975).Sa.to_numpy()
    assert np.all(u975 > quito.published(475)) and np.all(u975 < quito.published(2475))


def test_fit_method_reproduces_published(quito):
    np.testing.assert_allclose(quito.uhs(2475, method="fit").Sa, quito.published(2475))


def test_digitized_curve_agrees_with_published_uhs(quito):
    assert quito.has_digitized_curves
    for tr in (475, 2475):
        u = quito.uhs(tr, method="digitized")
        ok = u.source == "digitized"
        assert ok.sum() >= 6
        np.testing.assert_allclose(u.Sa[ok], quito.published(tr)[ok], rtol=0.06)


def test_digitized_curves_are_monotone(quito):
    for T in PERIODS:
        c = quito.hazard_curve(T, method="digitized")
        assert np.all(np.diff(c.rate) <= 1e-12)


def test_extrapolation_warns(quito):
    with pytest.warns(UserWarning, match="outside the published"):
        quito.uhs(10000, method="fit")


def test_hazard_curve_fit_hits_anchors(hz):
    s = hz.site(-1.48, -78.52)                    # not a capital cell: fit only
    assert not s.has_digitized_curves
    sa = [s.published(475)[0], s.published(2475)[0]]
    c = s.hazard_curve(0.0, sa=sa)
    np.testing.assert_allclose(c.tr, [475, 2475], rtol=1e-9)


def test_hazard_map_at_other_tr(hz):
    m = hz.hazard_map(975)
    lo, hi = hz.hazard_map(475).sa, hz.hazard_map(2475).sa
    assert len(m) == 3146 and np.all((m.sa > lo) & (m.sa < hi))


def test_plots_render(hz, quito):
    import matplotlib

    matplotlib.use("Agg")
    quito.plot_uhs([475, 975])
    quito.plot_hazard_curves([0.0, 1.0])
    hz.plot_map(475, faults=True)


@pytest.mark.skipif(not os.environ.get("APEQUAKE_NETWORK_TESTS"),
                    reason="set APEQUAKE_NETWORK_TESTS=1 to query IG-EPN online")
def test_fetch_recent_events():
    from apeQuake.hazard import fetch_recent_events

    ev = fetch_recent_events()
    assert len(ev) > 0 and {"time", "lat", "lon", "magnitude"} <= set(ev.columns)

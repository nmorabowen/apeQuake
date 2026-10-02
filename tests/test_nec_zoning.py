"""NEC-SE-DS zone factor Z / region lookup from the digitized Figura 1."""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeQuake.code_spectrum.codes.nec import NECSpectrum
from apeQuake.nec import ZONE_NAMES, NECZone, region_at, table19, zone_at
from apeQuake.nec.zoning import _zone_map

# (lat, lon, Z per Tabla 19, region) for cities where the figure and the table agree
CITIES = {
    "Quito": (-0.2200, -78.5125, 0.40, "sierra"),
    "Guayaquil": (-2.1894, -79.8891, 0.40, "costa"),
    "Cuenca": (-2.9001, -79.0059, 0.25, "sierra"),
    "Manta": (-0.9677, -80.7089, 0.50, "costa"),
    "Portoviejo": (-1.0546, -80.4545, 0.50, "costa"),
    "Esmeraldas": (0.9682, -79.6517, 0.50, "esmeraldas"),
    "Ibarra": (0.3517, -78.1223, 0.40, "sierra"),
    "Ambato": (-1.2491, -78.6168, 0.40, "sierra"),
    "Riobamba": (-1.6636, -78.6546, 0.40, "sierra"),
    "Loja": (-3.9931, -79.2042, 0.25, "sierra"),
    "Machala": (-3.2581, -79.9554, 0.40, "costa"),
    "Tena": (-0.9938, -77.8129, 0.35, "oriente"),
    "Puyo": (-1.4924, -78.0024, 0.30, "oriente"),
    "Nueva Loja": (0.0847, -76.8828, 0.15, "oriente"),
    "Santo Domingo": (-0.2530, -79.1754, 0.40, "costa"),
    "Babahoyo": (-1.8022, -79.5344, 0.30, "costa"),
    "Salinas": (-2.2145, -80.9585, 0.50, "costa"),
    "Guaranda": (-1.5926, -79.0019, 0.35, "sierra"),
}


@pytest.mark.parametrize("name", sorted(CITIES))
def test_city_zone_and_region(name):
    lat, lon, z, region = CITIES[name]
    r = zone_at(lat, lon)
    assert isinstance(r, NECZone)
    assert r.z == pytest.approx(z)
    assert r.zone == ZONE_NAMES[z]
    assert r.region == region
    assert r.source == "figura-1"


def test_eta_by_region():
    assert zone_at(-0.22, -78.51).eta == 2.48       # Sierra
    assert zone_at(-2.19, -79.89).eta == 1.80       # Costa
    assert zone_at(0.97, -79.65).eta == 2.48        # Esmeraldas
    assert zone_at(-0.99, -77.81).eta == 2.60       # Oriente


def test_galapagos_by_province():
    r = zone_at(-0.7432, -90.3135)                  # Puerto Ayora
    assert (r.z, r.region, r.eta, r.source) == (0.30, "galapagos", 2.48, "galapagos")
    assert r.nearest_listed is None


@pytest.mark.parametrize("lat, lon", [(-12.05, -77.04), (-1.0, -82.0), (4.71, -74.07)])
def test_outside_ecuador_raises(lat, lon):
    with pytest.raises(ValueError):
        zone_at(lat, lon)


def test_province_lookup():
    assert region_at(-0.22, -78.51) == ("PICHINCHA", "sierra")
    assert region_at(-2.19, -79.89)[0] == "GUAYAS"


def test_boundary_distance_and_warning():
    # Milagro sits just inside the 0.30 ellipse; Tabla 19 lists it at 0.35
    r = zone_at(-2.1347, -79.5872)
    assert r.boundary_km < 5.0
    assert r.z_across_boundary is not None and r.z_across_boundary != r.z
    assert any("km from the Z" in w for w in r.warnings)
    # far from any boundary: no warning, infinite or large distance
    m = zone_at(-0.9677, -80.7089)
    assert math.isinf(m.boundary_km) and not m.warnings


def test_listed_town_disagreement_is_flagged():
    r = zone_at(-1.0286, -79.4635)                  # Quevedo: map 0.30, Tabla 19 0.35
    assert r.nearest_listed is not None
    if r.nearest_listed.z != r.z and r.nearest_listed.distance_km <= 10:
        assert any("Tabla 19" in w for w in r.warnings)


def test_map_agrees_with_table19():
    """Regression guard on the digitization: >= 90 % of listed towns at the parish point."""
    t = table19().dropna(subset=["lat", "lon"])
    assert len(t) >= 540
    zmap = np.array([zone_at(a, o).z for a, o in zip(t.lat, t.lon)])
    assert np.mean(np.isclose(zmap, t.z.to_numpy())) >= 0.90


def test_zone_map_shape_and_values():
    zone, lon, lat, zv = _zone_map()
    assert zone.shape == (lat.size, lon.size)
    assert set(np.unique(zone)) <= set(range(len(zv))) | {255}
    assert np.allclose(zv, sorted(ZONE_NAMES))
    assert np.allclose(np.diff(lon), np.diff(lon)[0]) and np.allclose(np.diff(lat), np.diff(lat)[0])


def test_feeds_nec_spectrum():
    r = zone_at(-0.2200, -78.5125)
    spec = NECSpectrum(z=r.z, site_class="D", region=r.region)
    assert spec.sa(0.2) == pytest.approx(r.eta * r.z * spec.parameters()["Fa"], rel=1e-6)


def test_to_dict_is_json_ready():
    import json

    d = zone_at(-2.1347, -79.5872).to_dict()
    json.dumps(d)
    assert d["z"] == 0.30 and isinstance(d["warnings"], list)


def test_zone_notices_carry_params():
    r = zone_at(-2.1347, -79.5872)                               # Milagro
    assert [n.text for n in r.notices] == list(r.warnings)
    nb = next(n for n in r.notices if n.code == "near_boundary")
    assert nb.params["zAcross"] == r.z_across_boundary
    t19 = next(n for n in r.notices if n.code == "table19_differs")
    assert (t19.params["zTable"], t19.params["zMap"]) == (0.35, 0.30)
    assert r.to_dict()["notices"][0]["code"] in {"near_boundary", "table19_differs"}

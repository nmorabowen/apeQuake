"""NEC-SE-DS seismic zone factor Z and region (eta) at any point in Ecuador.

``zone_at(lat, lon)`` reads Z from the digitized NEC-SE-DS Figura 1 (section 3.1.1),
assigns the region that sets the spectral amplification eta (section 3.3.1: Costa
except Esmeraldas 1.80; Sierra, Esmeraldas and Galapagos 2.48; Oriente 2.60) from
the province containing the point, and cross-checks Tabla 19 (section 10.2):

* ``boundary_km``: distance to the nearest point of the map with a different Z.  The
  figure is a ~1.5 km/pixel raster, so values within a few km of a boundary are
  uncertain; the result carries a warning.
* ``nearest_listed``: the closest town of Tabla 19 and its Z.  NEC tells the designer
  to use the listed value for listed towns, and the nearest listed town when the map
  is hard to read; a warning is raised when it differs from the map.

Galapagos is not on the main map; the inset gives the whole province Z = 0.30 g.

How the map was digitized, and its agreement with Tabla 19, is documented in
``scripts/digitize_nec_zone_map.py``.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd

from ..code_spectrum.codes.nec import ETA_BY_REGION
from ..hazard import _data

__all__ = [
    "ZONE_NAMES",
    "ListedTown",
    "NECZone",
    "zone_at",
    "region_at",
    "table19",
]

Region = Literal["costa", "sierra", "oriente", "esmeraldas", "galapagos"]

#: Zone name by Z (NEC-SE-DS Tabla 1).
ZONE_NAMES: dict[float, str] = {0.15: "I", 0.25: "II", 0.30: "III", 0.35: "IV", 0.40: "V", 0.50: "VI"}

GALAPAGOS_Z = 0.30
BOUNDARY_WARN_KM = 5.0
LISTED_WARN_KM = 10.0
_SNAP_KM = 3.0          # off-mask tolerance (coastline / figure registration)
_SEARCH_PX = 15         # boundary search window half-width (~22 km)
_OUTSIDE = 255
_KM_PER_DEG = 111.195

_REGION_BY_PROVINCE: dict[str, Region] = {
    **dict.fromkeys(["MANABI", "SANTO DOMINGO DE LOS TSACHILAS", "LOS RIOS", "GUAYAS",
                     "SANTA ELENA", "EL ORO"], "costa"),
    "ESMERALDAS": "esmeraldas",
    **dict.fromkeys(["CARCHI", "IMBABURA", "PICHINCHA", "COTOPAXI", "TUNGURAHUA", "BOLIVAR",
                     "CHIMBORAZO", "CANAR", "AZUAY", "LOJA"], "sierra"),
    **dict.fromkeys(["SUCUMBIOS", "NAPO", "ORELLANA", "PASTAZA", "MORONA SANTIAGO",
                     "ZAMORA CHINCHIPE"], "oriente"),
    "GALAPAGOS": "galapagos",
}


@dataclass(frozen=True)
class ListedTown:
    """A Tabla 19 entry (located at its parish point) and its distance to the query."""

    town: str
    parish: str
    canton: str
    province: str
    z: float
    distance_km: float


@dataclass(frozen=True)
class NECZone:
    """NEC-SE-DS zoning of one point."""

    lat: float
    lon: float
    z: float
    zone: str
    province: str
    region: Region
    eta: float
    boundary_km: float          # inf when no other zone lies within ~20 km
    z_across_boundary: float | None
    nearest_listed: ListedTown | None
    source: Literal["figura-1", "galapagos"]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["nearest_listed"] = None if self.nearest_listed is None else dict(self.nearest_listed.__dict__)
        d["boundary_km"] = None if np.isinf(self.boundary_km) else self.boundary_km
        d["warnings"] = list(self.warnings)
        return d


@lru_cache(maxsize=1)
def _zone_map() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = resources.files("apeQuake.nec").joinpath("data", "nec_zone_map.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as f:
        return f["zone"], f["lon"], f["lat"], f["z_values"]


@lru_cache(maxsize=1)
def table19() -> pd.DataFrame:
    """Tabla 19 (NEC-SE-DS 10.2) with each row's parish point (lat/lon may be NaN)."""
    with resources.files("apeQuake.nec").joinpath("data", "nec_table19.csv").open("rb") as fh:
        return pd.read_csv(fh, dtype={"parish_code": str})


@lru_cache(maxsize=1)
def _province_paths():
    from matplotlib.path import Path

    out = []
    for f in _data.geojson("admin_provinces.geojson.gz")["features"]:
        name = _data.normalize(f["properties"]["DPA_DESPRO"]).replace("Ð", "N")
        g = f["geometry"]
        for poly in g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]:
            verts, codes = [], []
            for ring in poly:
                ring = np.asarray(ring, float)
                verts.append(ring)
                codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
            v = np.concatenate(verts)
            out.append((name, Path(v, codes), (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())))
    return out


def _province_at(lat: float, lon: float, tol_deg: float = _SNAP_KM / _KM_PER_DEG) -> str | None:
    best, best_d = None, np.inf
    for name, path, (x0, x1, y0, y1) in _province_paths():
        if not (x0 - tol_deg <= lon <= x1 + tol_deg and y0 - tol_deg <= lat <= y1 + tol_deg):
            continue
        if path.contains_point((lon, lat)):
            return name
        v = path.vertices
        d = float(np.min(np.hypot(v[:, 0] - lon, v[:, 1] - lat)))
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= tol_deg else None


def region_at(lat: float, lon: float) -> tuple[str, Region]:
    """Province containing the point and its NEC region (for eta).

    ``ZONA NO DELIMITADA`` (the three undelimited areas, all in the coastal lowlands)
    is assigned to ``costa``.
    """
    prov = _province_at(lat, lon)
    if prov is None:
        raise ValueError(f"({lat:.4f}, {lon:.4f}) is outside Ecuador")
    return prov, _REGION_BY_PROVINCE.get(prov, "costa")


def _km(dlat: np.ndarray, dlon: np.ndarray, lat: float) -> np.ndarray:
    return _KM_PER_DEG * np.hypot(dlat, dlon * np.cos(np.radians(lat)))


def _nearest_listed(lat: float, lon: float, max_km: float = 50.0) -> ListedTown | None:
    t = table19().dropna(subset=["lat", "lon"])
    d = _km(t.lat.to_numpy() - lat, t.lon.to_numpy() - lon, lat)
    if t.empty or d.min() > max_km:      # e.g. Galapagos: no listed towns
        return None
    r = t.iloc[int(d.argmin())]
    return ListedTown(r.poblacion, r.parroquia, r.canton, r.provincia, float(r.z), float(d.min()))


def zone_at(lat: float, lon: float) -> NECZone:
    """NEC zone factor Z, zone, region and eta at ``(lat, lon)`` [deg, WGS-84].

    Raises ``ValueError`` for points outside Ecuador.
    """
    lat, lon = float(lat), float(lon)
    province, region = region_at(lat, lon)
    eta = ETA_BY_REGION[region]
    listed = _nearest_listed(lat, lon)
    warnings: list[str] = []

    if region == "galapagos":
        z, src, bkm, zx = GALAPAGOS_Z, "galapagos", float("inf"), None
    else:
        zone, lons, lats, zv = _zone_map()
        dlon, dlat = lons[1] - lons[0], lats[1] - lats[0]
        i = int(round((lat - lats[0]) / dlat))
        j = int(round((lon - lons[0]) / dlon))
        i0, i1 = max(i - _SEARCH_PX, 0), min(i + _SEARCH_PX + 1, zone.shape[0])
        j0, j1 = max(j - _SEARCH_PX, 0), min(j + _SEARCH_PX + 1, zone.shape[1])
        win = zone[i0:i1, j0:j1]
        ii, jj = np.mgrid[i0:i1, j0:j1]
        dist = _km(lats[ii] - lat, lons[jj] - lon, lat)
        inside = win != _OUTSIDE
        if not inside.any() or dist[inside].min() > _SNAP_KM:
            raise ValueError(f"({lat:.4f}, {lon:.4f}) is not covered by the NEC zone map")
        k = int(win[inside][dist[inside].argmin()])
        z, src = float(zv[k]), "figura-1"
        other = inside & (win != k)
        if other.any():
            n = int(dist[other].argmin())
            bkm, zx = float(dist[other][n]), float(zv[int(win[other][n])])
        else:
            bkm, zx = float("inf"), None
        if bkm <= BOUNDARY_WARN_KM:
            warnings.append(
                f"Point is {bkm:.1f} km from the Z = {zx:.2f} zone; the digitized map is "
                f"~1.5 km/px, check Tabla 19 or the nearest listed town."
            )

    if listed is not None and listed.distance_km <= LISTED_WARN_KM and listed.z != z:
        warnings.append(
            f"Tabla 19 lists {listed.town} ({listed.canton}), {listed.distance_km:.1f} km away, "
            f"with Z = {listed.z:.2f}; the map gives Z = {z:.2f}."
        )
    if province == "ZONA NO DELIMITADA":
        warnings.append("Point is in a 'zona no delimitada'; region set to costa, confirm eta.")

    return NECZone(lat, lon, z, ZONE_NAMES[round(z, 2)], province, region, eta,
                   bkm, zx, listed, src, tuple(warnings))

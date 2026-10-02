"""Map layers for clients (``map.layer``): NEC zones, IG-EPN hazard and context layers.

Payloads only: geometry and values come from the bundled data; colouring and
legends are the client's.  Coordinates are rounded to 4 decimals (~11 m).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..hazard import EcuadorHazard, _data
from ..nec.zoning import ZONE_NAMES, zone_grid

LAYERS = ("necZones", "igepnHazard", "faults", "sourceZones", "capitals", "provinces")
SOURCE_TYPE = {"Crustal Sources": "crustal", "Interface Sources": "interface",
               "In-slab sources": "inslab", "Background sources": "background"}


def _round(coords: Any) -> Any:
    if isinstance(coords, (int, float)):
        return round(float(coords), 4)
    return [_round(c) for c in coords]


def _geometry(g: dict[str, Any]) -> dict[str, Any]:
    return {"type": g["type"], "coordinates": _round(g["coordinates"])}


def _fc(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


@lru_cache(maxsize=1)
def nec_zones() -> dict[str, Any]:
    """The digitized NEC-SE-DS Figura 1 as a run-length encoded grid (row 0 = north)."""
    zone, lon, lat, zv = zone_grid()
    rows = []
    for row in zone:
        runs: list[list[int]] = []
        for v in row.tolist():
            if runs and runs[-1][0] == v:
                runs[-1][1] += 1
            else:
                runs.append([int(v), 1])
        rows.append(runs)
    return {
        "layer": "necZones", "kind": "grid",
        "lon0": round(float(lon[0]), 6), "dlon": round(float(lon[1] - lon[0]), 8),
        "lat0": round(float(lat[0]), 6), "dlat": round(float(lat[1] - lat[0]), 8),
        "nx": int(zone.shape[1]), "ny": int(zone.shape[0]), "noData": 255,
        "classes": [{"code": i, "z": float(z), "zone": ZONE_NAMES[round(float(z), 2)]}
                    for i, z in enumerate(zv)],
        "rows": rows,
        "galapagosZ": 0.30,
    }


@lru_cache(maxsize=1)
def _cell_geometry() -> dict[str, dict[str, Any]]:
    return {f["properties"]["cell_id"]: _geometry(f["geometry"])
            for f in _data.geojson("hazard_cells.geojson.gz")["features"]}


def igepn_hazard(tr: int, period: float, stat: str) -> dict[str, Any]:
    df = EcuadorHazard.hazard_map(tr=tr, period=period, stat=stat)
    geo = _cell_geometry()
    feats = [{"type": "Feature", "geometry": geo[c], "properties": {"cellId": c, "sa": round(float(s), 4)}}
             for c, s in zip(df.cell_id, df.sa) if c in geo and np.isfinite(s)]
    sa = df.sa.to_numpy(float)
    return {"layer": "igepnHazard", "kind": "geojson", "tr": tr, "period": period, "stat": stat,
            "min": round(float(np.nanmin(sa)), 4), "max": round(float(np.nanmax(sa)), 4),
            "features": _fc(feats)}


@lru_cache(maxsize=1)
def faults() -> dict[str, Any]:
    feats = []
    for f in _data.geojson("faults.geojson")["features"]:
        p = f["properties"]
        feats.append({"type": "Feature", "geometry": _geometry(f["geometry"]),
                      "properties": {"name": p.get("Nombre"), "mmax": p.get("Mmax"),
                                     "dip": p.get("Dip"), "slipMmYr": p.get("Slip_Geod")}})
    return {"layer": "faults", "kind": "geojson", "features": _fc(feats)}


@lru_cache(maxsize=1)
def source_zones() -> dict[str, Any]:
    feats = []
    for f in _data.geojson("sources_beauval2018.geojson")["features"]:
        p = f["properties"]
        feats.append({"type": "Feature", "geometry": _geometry(f["geometry"]),
                      "properties": {"name": p.get("Nombre"),
                                     "type": SOURCE_TYPE.get(p.get("Tipo"), "other"),
                                     "mmax": p.get("MMax")}})
    return {"layer": "sourceZones", "kind": "geojson", "features": _fc(feats)}


@lru_cache(maxsize=1)
def capitals() -> dict[str, Any]:
    pts = [{"name": r.name, "canton": r.canton, "province": r.province, "type": r.type,
            "lat": round(float(r.lat), 4), "lon": round(float(r.lon), 4)}
           for r in _data.capitals().itertuples()]
    return {"layer": "capitals", "kind": "points", "points": pts}


@lru_cache(maxsize=1)
def provinces() -> dict[str, Any]:
    feats = []
    for f in _data.geojson("admin_provinces.geojson.gz")["features"]:
        name = _data.normalize(f["properties"]["DPA_DESPRO"]).replace("Ð", "Ñ")
        feats.append({"type": "Feature", "geometry": _geometry(f["geometry"]),
                      "properties": {"name": name}})
    return {"layer": "provinces", "kind": "geojson", "features": _fc(feats)}

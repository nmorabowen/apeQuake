"""Lazy, cached access to the bundled IG-EPN tables (see data/igepn/README.md)."""
from __future__ import annotations

import gzip
import json
import unicodedata
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

PERIODS: tuple[float, ...] = (0.0, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0)
"""Spectral periods of the IG-EPN model [s]; 0.0 is PGA."""

RETURN_PERIODS: tuple[int, ...] = (475, 2475)
"""Return periods published by IG-EPN [yr] (10 % and 2 % in 50 yr)."""

STATS: tuple[str, ...] = ("mean", "q16", "q50", "q84")
"""Statistics of the logic tree published by IG-EPN."""

GRID_STEP = 0.08
"""Grid spacing of the hazard cells [deg]."""

PERIOD_COLUMNS = tuple(f"T{T:.2f}" for T in PERIODS)


def _path(name: str):
    return files("apeQuake.hazard").joinpath("data", "igepn", name)


def _csv(name: str, **kw) -> pd.DataFrame:
    with _path(name).open("rb") as fh:
        return pd.read_csv(fh, compression="gzip" if name.endswith(".gz") else None, **kw)


@lru_cache(maxsize=None)
def manifest() -> dict[str, Any]:
    return json.loads(_path("manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def cells() -> pd.DataFrame:
    return _csv("hazard_cells.csv")


@lru_cache(maxsize=None)
def grid() -> pd.DataFrame:
    """UHS ordinates indexed by (cell_id, tr, stat); columns T0.00 ... T2.00."""
    return _csv("hazard_grid.csv.gz").set_index(["cell_id", "tr", "stat"]).sort_index()


@lru_cache(maxsize=None)
def capitals() -> pd.DataFrame:
    return _csv("capitals.csv")


@lru_cache(maxsize=None)
def curves() -> pd.DataFrame:
    """Digitized mean hazard curves of the capital cells: cell_id, period, sa_g, rate."""
    return _csv("hazard_curves_capitals.csv.gz")


@lru_cache(maxsize=None)
def curves_qc() -> pd.DataFrame:
    return _csv("hazard_curves_capitals_qc.csv")


@lru_cache(maxsize=None)
def population() -> pd.DataFrame:
    return _csv("cabeceras_population.csv")


@lru_cache(maxsize=None)
def catalog(kind: str) -> pd.DataFrame:
    df = _csv(f"catalog_{kind}.csv.gz")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="ISO8601")
    if kind == "historical":
        # the IG-EPN historical layer stores depth negative-down; make it positive-down
        # like the homogenized catalogs (the bundled file keeps the published sign)
        df["depth_km"] = -df["depth_km"]
    return df


@lru_cache(maxsize=None)
def cell_paths() -> tuple[list[str], list[Any], np.ndarray]:
    """Cell polygons as matplotlib Paths, with their bounding boxes (for lookup).

    The IG-EPN cells are not a pure lattice: some lattice positions (e.g. over the Guayas
    river) are merged into a neighbour, so point-in-polygon is the correct lookup.
    """
    from matplotlib.path import Path

    ids, paths, boxes = [], [], []
    for f in geojson("hazard_cells.geojson.gz")["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            verts, codes = [], []
            for ring in poly:                       # outer ring + holes as one compound path
                ring = np.asarray(ring, float)
                verts.append(ring)
                codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
            v = np.concatenate(verts)
            ids.append(f["properties"]["cell_id"])
            paths.append(Path(v, codes))
            boxes.append((v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()))
    return ids, paths, np.asarray(boxes)


@lru_cache(maxsize=None)
def admin(level: str) -> pd.DataFrame:
    return _csv(f"admin_{level}.csv")


@lru_cache(maxsize=None)
def geojson(name: str) -> dict[str, Any]:
    p = _path(name)
    raw = p.read_bytes()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def normalize(text: str) -> str:
    """Upper-case, accent-free, single-spaced text for name matching."""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.upper().replace("-", " ").split())

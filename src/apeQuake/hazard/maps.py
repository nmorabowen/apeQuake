"""Maps of the IG-EPN hazard with overlays (static, matplotlib)."""
from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Sequence, Union

import numpy as np
import pandas as pd

from . import _data
from .site import HazardSite

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from .ecuador import EcuadorHazard

PointsLike = Union[Iterable[Any], pd.DataFrame, None]
CatalogLike = Union[str, Sequence[str], pd.DataFrame, None]

SOURCE_TYPES: dict[str, str] = {
    "crustal": "Crustal Sources",
    "interface": "Interface Sources",
    "inslab": "In-slab sources",
    "background": "Background sources",
}
"""Short name -> ``Tipo`` value of the IG-EPN source model."""

SOURCE_COLORS = {"crustal": "tab:orange", "interface": "tab:red", "inslab": "tab:purple",
                 "background": "0.45"}
CATALOG_COLORS = {"shallow": "tab:blue", "deep": "tab:cyan", "historical": "tab:green",
                  "custom": "tab:purple"}


# ---------------------------------------------------------------------- inputs

def resolve_points(hz: "EcuadorHazard", points: PointsLike, **site_kw) -> list[HazardSite]:
    """Turn user points into HazardSites.

    Accepts place names, ``(lat, lon)`` or ``(lat, lon, label)`` tuples, HazardSite
    objects, or a DataFrame with ``lat`` / ``lon`` and optionally ``label`` / ``name``.
    """
    if points is None:
        return []
    if isinstance(points, (str, HazardSite)) or (
            isinstance(points, tuple) and len(points) in (2, 3)
            and all(isinstance(v, (int, float)) for v in points[:2])):
        points = [points]
    if isinstance(points, pd.DataFrame):
        lab = next((c for c in ("label", "name") if c in points), None)
        points = [(r.lat, r.lon, getattr(r, lab)) if lab else (r.lat, r.lon)
                  for r in points.itertuples()]
    out = []
    for p in points:
        if isinstance(p, HazardSite):
            out.append(p)
        elif isinstance(p, str):
            out.append(hz.site(p, **site_kw))
        else:
            s = hz.site(p[0], p[1], **site_kw)
            if len(p) > 2 and p[2] is not None:
                s.label = str(p[2])
            out.append(s)
    return out


def source_types(sources: bool | str | Sequence[str]) -> list[str]:
    """Normalize the ``sources`` argument to a list of short type names."""
    if sources is False or sources is None:
        return []
    if sources is True:
        return list(SOURCE_TYPES)
    if isinstance(sources, str):
        sources = [sources]
    bad = [s for s in sources if s not in SOURCE_TYPES]
    if bad:
        raise ValueError(f"unknown source type(s) {bad}; choose from {list(SOURCE_TYPES)}")
    return list(sources)


def catalog_frames(hz: "EcuadorHazard", catalog: CatalogLike, min_mw: float | None
                   ) -> list[tuple[str, pd.DataFrame]]:
    """Normalize the ``catalog`` argument to ``[(name, df with lat/lon/mw/...)]``."""
    if catalog is None:
        return []
    if isinstance(catalog, pd.DataFrame):
        items = [("custom", catalog.copy())]
    else:
        names = [catalog] if isinstance(catalog, str) else list(catalog)
        items = [(n, hz.catalog(n)) for n in names]
    out = []
    for name, df in items:
        if "mw" not in df and "magnitude" in df:
            df = df.rename(columns={"magnitude": "mw"})
        if not {"lat", "lon", "mw"} <= set(df.columns):
            raise ValueError("a catalog DataFrame needs lat, lon and mw (or magnitude)")
        if min_mw is not None:
            df = df[df.mw >= min_mw]
        out.append((name, df.dropna(subset=["lat", "lon", "mw"])))
    return out


def geom_lines(geom: dict | None) -> list[np.ndarray]:
    """Rings / lines of a GeoJSON geometry as (n, 2) lon/lat arrays."""
    if geom is None:
        return []
    t, c = geom["type"], geom["coordinates"]
    if t == "LineString":
        parts = [c]
    elif t in ("MultiLineString", "Polygon"):
        parts = c
    elif t == "MultiPolygon":
        parts = [ring for poly in c for ring in poly]
    else:
        return []
    return [np.asarray(p, float) for p in parts]


def _draw_geojson(ax: "Axes", name: str, **kw) -> None:
    for f in _data.geojson(name)["features"]:
        for xy in geom_lines(f["geometry"]):
            ax.plot(xy[:, 0], xy[:, 1], **kw)


def _mw_size(mw: np.ndarray) -> np.ndarray:
    """Marker area [pt^2] growing ~exponentially with magnitude."""
    return 3.0 * 2.2 ** (np.asarray(mw, float) - 4.0)


# ---------------------------------------------------------------------- static map

def plot_map(hz: "EcuadorHazard", tr: float = 475, period: float = 0.0, stat: str = "mean",
             ax: "Axes | None" = None, *, provinces: bool = True, faults: bool = False,
             sources: bool | str | Sequence[str] = False, catalog: CatalogLike = None,
             min_mw: float | None = None, capitals: bool = False, points: PointsLike = None,
             annotate: bool = True, extent: str | Sequence[float] | None = None,
             cmap: str = "magma_r", vmin: float | None = None, vmax: float | None = None,
             legend: bool = True) -> "Axes":
    """Static hazard map with overlays. See :meth:`EcuadorHazard.plot_map`."""
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = hz.hazard_map(tr, period, stat)
    step = _data.GRID_STEP
    lats = np.round(np.arange(m.lat.min(), m.lat.max() + step / 2, step), 2)
    lons = np.round(np.arange(m.lon.min(), m.lon.max() + step / 2, step), 2)
    Z = np.full((lats.size, lons.size), np.nan)
    ii = np.round((m.lat - lats[0]) / step).astype(int)
    jj = np.round((m.lon - lons[0]) / step).astype(int)
    Z[ii, jj] = m.sa
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7.5))
    pc = ax.pcolormesh(np.r_[lons - step / 2, lons[-1] + step / 2],
                       np.r_[lats - step / 2, lats[-1] + step / 2], Z, cmap=cmap,
                       vmin=vmin, vmax=vmax)
    lbl = "PGA" if period == 0 else f"Sa({period:g} s)"
    plt.colorbar(pc, ax=ax, label=f"{lbl} [g]", shrink=0.75)
    handles: list[Any] = []

    if provinces:
        _draw_geojson(ax, "admin_provinces.geojson.gz", color="0.35", lw=0.4)

    for t in source_types(sources):
        src = hz.sources()
        for g in src[src.Tipo == SOURCE_TYPES[t]].geometry:
            for xy in geom_lines(g):
                ax.plot(xy[:, 0], xy[:, 1], color=SOURCE_COLORS[t], lw=1.0, ls="--")
        handles.append(Line2D([], [], color=SOURCE_COLORS[t], ls="--",
                              label=f"{t} sources"))

    if faults:
        _draw_geojson(ax, "faults.geojson", color="tab:blue", lw=1.6)
        handles.append(Line2D([], [], color="tab:blue", lw=1.6, label="faults"))

    for name, df in catalog_frames(hz, catalog, min_mw):
        col = CATALOG_COLORS.get(name, CATALOG_COLORS["custom"])
        ax.scatter(df.lon, df.lat, s=_mw_size(df.mw), facecolors="none", edgecolors=col,
                   linewidths=0.6, alpha=0.8, zorder=3)
        handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=col,
                              label=f"{name} catalog ({len(df)})"))

    if capitals:
        c = _data.capitals()
        prov = c[c.type == "CAPITAL PROVINCIAL"].drop_duplicates("canton")
        ax.plot(prov.lon, prov.lat, "k.", ms=3, zorder=4)
        handles.append(Line2D([], [], marker=".", ls="", color="k",
                              label="provincial capitals"))

    sites = resolve_points(hz, points)
    if sites:
        idx = int(np.where(np.isclose(_data.PERIODS, period))[0][0])
        near_deg = 0.25 if extent is None else 0.05      # "close" relative to the window
        placed: list[tuple[float, float]] = []
        for s in sites:
            ax.plot(s.lon, s.lat, marker="*", ms=13, mfc="white", mec="k", mew=1.0, zorder=6)
            if annotate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = s.uhs(tr, stat).Sa.iloc[idx]
                # alternate the label corner when an already-labelled point is close by
                near = sum(np.hypot(s.lon - x, s.lat - y) < near_deg for x, y in placed)
                dx, dy = [(7, 7), (-7, -20), (7, -20), (-7, 7)][near % 4]
                placed.append((s.lon, s.lat))
                txt = ax.annotate(f"{s.label}\n{lbl} = {val:.2f} g", (s.lon, s.lat),
                                  xytext=(dx, dy), textcoords="offset points", fontsize=8,
                                  ha="left" if dx > 0 else "right", zorder=7)
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
        handles.append(Line2D([], [], marker="*", ls="", ms=11, mfc="white", mec="k",
                              label="sites"))

    ax.set_aspect("equal")
    if extent is None:
        ax.set_xlim(lons[0] - step, lons[-1] + step)
        ax.set_ylim(lats[0] - step, lats[-1] + step)
    elif extent == "points":
        if not sites:
            raise ValueError("extent='points' needs points")
        la = np.array([s.lat for s in sites])
        lo = np.array([s.lon for s in sites])
        pad = max(0.3, 0.15 * max(np.ptp(la), np.ptp(lo)))
        ax.set_xlim(lo.min() - pad, lo.max() + pad)
        ax.set_ylim(la.min() - pad, la.max() + pad)
    else:
        x0, x1, y0, y1 = extent
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"{lbl}, TR = {tr:g} yr ({stat}), rock - IG-EPN", fontsize=10)
    if legend and handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.85)
    return ax


# ---------------------------------------------------------------------- interactive map

def _round_coords(obj: Any, nd: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [_round_coords(v, nd) for v in obj]
    return obj


def _clean(v: Any) -> Any:
    """JSON-safe scalar (NaN -> None, numpy -> python)."""
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _feature_collection(df: pd.DataFrame) -> dict:
    feats = []
    for r in df.to_dict("records"):
        geom = r.pop("geometry")
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {k: _clean(v) for k, v in r.items()}})
    return {"type": "FeatureCollection", "features": feats}


def _events(df: pd.DataFrame) -> list[list]:
    """Compact [lat, lon, mw, depth_km, time] rows for the map."""
    if "mw" not in df and "magnitude" in df:
        df = df.rename(columns={"magnitude": "mw"})
    df = df.dropna(subset=["lat", "lon", "mw"])
    dep = df["depth_km"] if "depth_km" in df else pd.Series(np.nan, index=df.index)
    tim = df["time"] if "time" in df else pd.Series("", index=df.index)
    return [[round(float(a), 3), round(float(b), 3), round(float(m), 2), _clean(d),
             str(t)[:16] if pd.notna(t) else ""]
            for a, b, m, d, t in zip(df.lat, df.lon, df.mw, dep, tim)]


def explore_data(hz: "EcuadorHazard", tr: float = 475, period: float = 0.0,
                 stat: str = "mean", points: PointsLike = None,
                 catalogs: Sequence[str] = ("shallow", "deep", "historical"),
                 recent: bool | pd.DataFrame = False,
                 point_trs: Sequence[float] = (475, 975, 2475)) -> dict[str, Any]:
    """Everything the interactive map needs, as plain JSON-able structures."""
    if stat not in _data.STATS:
        raise ValueError(f"stat must be one of {_data.STATS}")
    pidx = np.where(np.isclose(_data.PERIODS, period))[0]
    if not pidx.size:
        raise ValueError(f"period must be one of {_data.PERIODS}")

    # one (cell, tr, stat, period) block: the grid index is sorted (cell, tr, stat), and
    # tr / stat sort in RETURN_PERIODS / STATS order, so a reshape is exact
    grid = _data.grid().reindex(pd.MultiIndex.from_product(
        [_data.grid().index.levels[0], _data.RETURN_PERIODS, _data.STATS]))
    block = grid[list(_data.PERIOD_COLUMNS)].to_numpy().round(4).reshape(
        -1, len(_data.RETURN_PERIODS) * len(_data.STATS) * len(_data.PERIODS))
    row = {cid: i for i, cid in enumerate(grid.index.levels[0])}
    cells = _data.cells().set_index("cell_id")
    digit = set(_data.curves_qc().cell_id)
    feats = []
    for f in _data.geojson("hazard_cells.geojson.gz")["features"]:
        cid = f["properties"]["cell_id"]
        v = block[row[cid]].tolist()          # flattened v[tr][stat][period], as the page reads it
        c = cells.loc[cid]
        feats.append({"type": "Feature",
                      "geometry": {"type": f["geometry"]["type"],
                                   "coordinates": _round_coords(f["geometry"]["coordinates"])},
                      "properties": {"id": cid, "lat": float(c.lat), "lon": float(c.lon),
                                     "v": v, "cap": cid in digit}})

    cats = {name: _events(hz.catalog(name)) for name in catalogs}
    if recent is not False:
        if recent is True:
            from .live import fetch_recent_events

            recent = fetch_recent_events()
        cats["recent"] = _events(recent)

    pop = _data.population().drop_duplicates("cell_id").set_index("cell_id")["pop_2010"]
    caps = [{"name": r.name, "canton": r.canton, "province": r.province, "lat": r.lat,
             "lon": r.lon, "pga475": r.pga_475, "pga2475": r.pga_2475,
             "prov": r.type == "CAPITAL PROVINCIAL",
             "pop": _clean(pop.get(r.cell_id)) if r.type == "CAPITAL PROVINCIAL" else None}
            for r in _data.capitals().itertuples()]

    pts = []
    for s in resolve_points(hz, points):
        uhs = []
        for t in point_trs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = s.uhs(t, "mean")
            uhs.append({"tr": _clean(t), "sa": [_clean(x) for x in u.Sa],
                        "source": "/".join(sorted(set(u.source)))})
        pts.append({"label": s.label, "lat": s.lat, "lon": s.lon, "cell_id": s.cell_id,
                    "uhs": uhs})

    m = _data.manifest()
    return {
        "periods": list(_data.PERIODS), "stats": list(_data.STATS),
        "trs": list(_data.RETURN_PERIODS),
        "init": {"tr": tr, "period": float(_data.PERIODS[int(pidx[0])]), "stat": stat},
        "cells": {"type": "FeatureCollection", "features": feats},
        "faults": _feature_collection(hz.faults()),
        "sources": _feature_collection(hz.sources()),
        "catalogs": cats, "capitals": caps, "points": pts,
        "meta": {"url": m["url"], "retrieved": m["retrieved_utc"][:10]},
    }


def explore(hz: "EcuadorHazard", path: "str | os.PathLike[str]" = "igepn_hazard_map.html",
            tr: float = 475, period: float = 0.0, stat: str = "mean",
            points: PointsLike = None,
            catalogs: Sequence[str] = ("shallow", "deep", "historical"),
            recent: bool | pd.DataFrame = False,
            point_trs: Sequence[float] = (475, 975, 2475),
            open_browser: bool = False) -> "Path":
    """Write the interactive map. See :meth:`EcuadorHazard.explore`."""
    import json
    import webbrowser
    from importlib.resources import files
    from pathlib import Path

    data = explore_data(hz, tr, period, stat, points, catalogs, recent, point_trs)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")          # data can never close the <script>
    html = files("apeQuake.hazard").joinpath("templates", "explore.html").read_text("utf-8")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/", payload, 1), encoding="utf-8")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out

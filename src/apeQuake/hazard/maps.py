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

# Catalogs differ by fill and outline, not by hue: a map carries at most three identity
# hues and those belong to the source-zone types. All epicenters are ink with a surface
# ring, so they read on every hazard color.
CATALOG_STYLE = {
    "shallow": {"fill": 0.0, "lw": 0.8, "ls": "-"},
    "deep": {"fill": 0.35, "lw": 0.8, "ls": "-"},
    "historical": {"fill": 0.0, "lw": 0.9, "ls": (0, (2, 1.5))},
    "recent": {"fill": 0.8, "lw": 0.6, "ls": "-"},
    "custom": {"fill": 0.8, "lw": 0.6, "ls": "-"},
}


def _catalog_style(name: str) -> dict:
    return CATALOG_STYLE.get(name, CATALOG_STYLE["custom"])


def _label_sites(ax, t, sites, tr, stat, period, near_deg: float) -> None:
    """Name + value beside each site, kept inside the map window.

    Labels go to the right unless the point sits in the right quarter of the window;
    a site close to an already-labelled one takes the label below instead.
    """
    from . import _style

    idx = int(np.where(np.isclose(_data.PERIODS, period))[0][0])
    x0, x1 = ax.get_xlim()
    placed: list[tuple[float, float]] = []
    for s in sites:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = s.uhs(tr, stat).Sa.iloc[idx]
        right = (s.lon - x0) / (x1 - x0) < 0.75
        near = sum(np.hypot(s.lon - x, s.lat - y) < near_deg for x, y in placed)
        dy = 6 if near % 2 == 0 else -20
        placed.append((s.lon, s.lat))
        txt = ax.annotate(f"{s.label}\n{val:.2f} g", (s.lon, s.lat),
                          xytext=(8 if right else -8, dy), textcoords="offset points",
                          fontsize=8, color=t.ink, ha="left" if right else "right", zorder=8)
        txt.set_path_effects(_style.halo(t, 3))


def plot_map(hz: "EcuadorHazard", tr: float = 475, period: float = 0.0, stat: str = "mean",
             ax: "Axes | None" = None, *, provinces: bool = True, faults: bool = False,
             sources: bool | str | Sequence[str] = False, catalog: CatalogLike = None,
             min_mw: float | None = None, capitals: bool = False, points: PointsLike = None,
             annotate: bool = True, extent: str | Sequence[float] | None = None,
             cmap: Any = None, vmin: float | None = None, vmax: float | None = None,
             legend: bool = True, theme: str = "light") -> "Axes":
    """Static hazard map with overlays. See :meth:`EcuadorHazard.plot_map`."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    from . import _style

    t = _style.theme(theme)
    m = hz.hazard_map(tr, period, stat)
    step = _data.GRID_STEP
    lats = np.round(np.arange(m.lat.min(), m.lat.max() + step / 2, step), 2)
    lons = np.round(np.arange(m.lon.min(), m.lon.max() + step / 2, step), 2)
    Z = np.full((lats.size, lons.size), np.nan)
    ii = np.round((m.lat - lats[0]) / step).astype(int)
    jj = np.round((m.lon - lons[0]) / step).astype(int)
    Z[ii, jj] = m.sa
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7.8))
    _style.style_axes(ax, t)
    ax.grid(False)                                   # the map is the data, not a grid
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(t.baseline)
        ax.spines[side].set_linewidth(0.8)
    pc = ax.pcolormesh(np.r_[lons - step / 2, lons[-1] + step / 2],
                       np.r_[lats - step / 2, lats[-1] + step / 2], Z,
                       cmap=cmap or _style.hazard_cmap(t.name), vmin=vmin, vmax=vmax,
                       rasterized=True)
    lbl = "PGA" if period == 0 else f"Sa({period:g} s)"
    cb = plt.colorbar(pc, ax=ax, shrink=0.72, fraction=0.035, pad=0.03)
    cb.set_label(f"{lbl} [g]", color=t.secondary, fontsize=8.5)
    cb.outline.set_edgecolor(t.baseline)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(colors=t.muted, labelcolor=t.secondary, labelsize=8, width=0.6)
    handles: list[Any] = []
    ring = _style.halo(t, 2.4)

    if provinces:
        _draw_geojson(ax, "admin_provinces.geojson.gz", color=t.secondary, lw=0.35,
                      alpha=0.55)

    for kind in source_types(sources):
        src = hz.sources()
        col = t.zones.get(kind, t.background_zone)
        for g in src[src.Tipo == SOURCE_TYPES[kind]].geometry:
            for xy in geom_lines(g):
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.3, path_effects=ring,
                        solid_joinstyle="round", zorder=3)
        handles.append(Line2D([], [], color=col, lw=1.6, label=f"{kind} source zones"))

    if faults:
        _draw_geojson(ax, "faults.geojson", color=t.ink, lw=1.8, path_effects=ring,
                      solid_capstyle="round", zorder=4)
        handles.append(Line2D([], [], color=t.ink, lw=1.8, label="faults"))

    for name, df in catalog_frames(hz, catalog, min_mw):
        st = _catalog_style(name)
        face = to_rgba(t.ink, st["fill"]) if st["fill"] else "none"
        sc = ax.scatter(df.lon, df.lat, s=_mw_size(df.mw), facecolors=face,
                        edgecolors=t.ink, linewidths=st["lw"], linestyles=[st["ls"]],
                        zorder=5)
        sc.set_path_effects(_style.halo(t, st["lw"] + 1.6))
        handles.append(Line2D([], [], marker="o", ls="", ms=7, mec=t.ink, mew=st["lw"],
                              mfc=face, label=f"{name} catalog ({len(df)})"))

    if capitals:
        c = _data.capitals()
        prov = c[c.type == "CAPITAL PROVINCIAL"].drop_duplicates("canton")
        ax.plot(prov.lon, prov.lat, "o", ms=4, mfc=t.surface, mec=t.ink, mew=0.8, zorder=6)
        handles.append(Line2D([], [], marker="o", ls="", ms=4, mfc=t.surface, mec=t.ink,
                              mew=0.8, label="provincial capitals"))

    sites = resolve_points(hz, points)
    if sites:
        for s in sites:
            ax.plot(s.lon, s.lat, "o", ms=9, mfc=t.ink, mec=t.surface, mew=2, zorder=7)
        handles.append(Line2D([], [], marker="o", ls="", ms=8, mfc=t.ink, mec=t.surface,
                              mew=1.6, label="sites"))

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
    if sites and annotate:
        _label_sites(ax, t, sites, tr, stat, period, near_deg=0.25 if extent is None else 0.05)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    _style.title(ax, t, f"{lbl}, TR = {tr:g} yr ({stat})",
                 "Rock, Vs30 = 760 m/s - IG-EPN (Beauval et al., 2018)")
    if legend and handles:
        _style.legend(ax, t, handles=handles, loc="lower left")
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


# Leaflet 1.9.4 (BSD-2-Clause, templates/vendor/leaflet/LICENSE). The vendored files match
# the Subresource Integrity hashes Leaflet publishes; the same hashes guard the CDN mode.
_LEAFLET_CDN = "https://unpkg.com/leaflet@1.9.4/dist/"
_LEAFLET_SRI = {"leaflet.js": "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=",
                "leaflet.css": "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="}


def _leaflet_tags(inline: bool) -> str:
    """``<link>``/``<script>`` for Leaflet: embedded (default) or from the CDN.

    Embedding makes the page work where external scripts are blocked (sandboxed file
    previews, mail / chat attachments, offline). The layer-control icons, which the
    stylesheet loads as separate files, become data URIs.
    """
    if not inline:
        return (f'<link rel="stylesheet" href="{_LEAFLET_CDN}leaflet.css" '
                f'integrity="{_LEAFLET_SRI["leaflet.css"]}" crossorigin="">\n'
                f'<script src="{_LEAFLET_CDN}leaflet.js" '
                f'integrity="{_LEAFLET_SRI["leaflet.js"]}" crossorigin=""></script>')
    import base64
    from importlib.resources import files

    v = files("apeQuake.hazard").joinpath("templates", "vendor", "leaflet")
    css = v.joinpath("leaflet.css").read_text("utf-8")
    for icon in ("layers.png", "layers-2x.png"):
        b64 = base64.b64encode(v.joinpath(icon).read_bytes()).decode("ascii")
        css = css.replace(f"url(images/{icon})", f"url(data:image/png;base64,{b64})")
    js = v.joinpath("leaflet.js").read_text("utf-8")
    return (f"<style>/* Leaflet 1.9.4, BSD-2-Clause */\n{css}</style>\n"
            f"<script>{js}</script>")


def explore(hz: "EcuadorHazard", path: "str | os.PathLike[str]" = "igepn_hazard_map.html",
            tr: float = 475, period: float = 0.0, stat: str = "mean",
            points: PointsLike = None,
            catalogs: Sequence[str] = ("shallow", "deep", "historical"),
            recent: bool | pd.DataFrame = False,
            point_trs: Sequence[float] = (475, 975, 2475),
            open_browser: bool = False, inline_leaflet: bool = True) -> "Path":
    """Write the interactive map. See :meth:`EcuadorHazard.explore`."""
    import json
    import webbrowser
    from importlib.resources import files
    from pathlib import Path

    data = explore_data(hz, tr, period, stat, points, catalogs, recent, point_trs)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")          # data can never close the <script>
    html = files("apeQuake.hazard").joinpath("templates", "explore.html").read_text("utf-8")
    html = html.replace("<!--__LEAFLET__-->", _leaflet_tags(inline_leaflet), 1)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/", payload, 1), encoding="utf-8")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out

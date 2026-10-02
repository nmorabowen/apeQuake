"""Isometric 3D relief of the IG-EPN hazard (static, matplotlib).

The static twin of the "3D view" of :meth:`EcuadorHazard.explore`: each cell is extruded
from its real outline, height and color both encode Sa, province outlines lie on the
ground and sites stand as needles with a red marker on top.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from . import _data
from .maps import PointsLike, _outlines, resolve_points

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .ecuador import EcuadorHazard

LAT0, LON0 = -1.6, -78.4                       # same local km frame as the HTML view
KX = 111.32 * np.cos(np.radians(LAT0))
KY = 110.57
SITE_RED = "#d62728"                           # the one identity mark: where the site is
LIGHT = np.array([-0.5, 0.6]) / np.hypot(-0.5, 0.6)     # light from the north-west
EXPORT_FORMATS = (".png", ".pdf", ".svg")
SUBTITLE = "Rock, Vs30 = 760 m/s - IG-EPN (Beauval et al., 2018)"


def _km(lon, lat):
    return (np.asarray(lon, float) - LON0) * KX, (np.asarray(lat, float) - LAT0) * KY


def _prisms() -> tuple[list[str], list[np.ndarray]]:
    """Cell ids and outer rings (counter-clockwise, open, km) of every cell polygon."""
    ids, rings = [], []
    for f in _data.geojson("hazard_cells.geojson.gz")["features"]:
        g = f["geometry"]
        for poly in g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]:
            x, y = _km(*np.asarray(poly[0], float).T)
            r = np.column_stack([x, y])
            if len(r) > 1 and np.allclose(r[0], r[-1]):
                r = r[:-1]
            if np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]) < 0:
                r = r[::-1]
            ids.append(f["properties"]["cell_id"])
            rings.append(r)
    return ids, rings


def _walls(rings, h, rgba, azim):
    """Visible side walls between neighbours (and the outer rim), shaded by orientation.

    A shared edge only needs a wall between the two heights; an unshared edge falls to
    the ground. Walls facing away from the camera are dropped.
    """
    edges: dict[tuple, list] = {}
    for k, r in enumerate(rings):
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            key = tuple(sorted([tuple(np.round(a, 3)), tuple(np.round(b, 3))]))
            edges.setdefault(key, []).append((h[k], k, a, b))
    cam = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    polys, cols, owner = [], [], []
    for ents in edges.values():
        ents.sort(key=lambda e: -e[0])
        hi, k, a, b = ents[0]
        lo = ents[1][0] if len(ents) > 1 else 0.0
        if hi - lo < 1e-9:
            continue
        n = np.array([b[1] - a[1], a[0] - b[0]])            # outward for a ccw ring
        n /= np.hypot(*n) or 1.0
        if n @ cam <= 0:
            continue
        s = float(np.clip(0.62 + 0.22 * (n @ LIGHT), 0, 1))
        polys.append([(*a, lo), (*b, lo), (*b, hi), (*a, hi)])
        cols.append((*(np.asarray(rgba[k][:3]) * s), 1.0))
        owner.append(k)
    return polys, cols, owner


def _need_3d(ax) -> None:
    if getattr(ax, "name", "") != "3d":
        raise ValueError("ax must be a 3D Axes (fig.add_subplot(projection='3d'))")


def plot_iso(hz: "EcuadorHazard", tr: float = 475, period: float = 0.0, stat: str = "mean",
             ax: "Axes | None" = None, *, points: PointsLike = None, provinces: bool = True,
             elev: float = 35, azim: float = -50, height_scale: float | None = None,
             cmap: Any = None, vmin: float | None = None, vmax: float | None = None,
             theme: str = "light", marker: str = "x", colorbar: bool = True,
             annotate: bool = True, title: bool = True, zoom: float = 1.15) -> "Axes":
    """Isometric relief of the hazard. See :meth:`EcuadorHazard.plot_iso`."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    from . import _style

    t = _style.theme(theme)
    m = hz.hazard_map(tr, period, stat)
    sa = m.set_index("cell_id").sa
    ids, rings = _prisms()
    v = sa.reindex(ids).to_numpy(float)
    ok = np.isfinite(v)
    ids, rings, v = [i for i, o in zip(ids, ok) if o], [r for r, o in zip(rings, ok) if o], v[ok]
    norm = Normalize(vmin=float(v.min()) if vmin is None else vmin,
                     vmax=float(v.max()) if vmax is None else vmax)
    cm = cmap or _style.hazard_cmap(t.name)
    rgba = cm(norm(v))
    scale = (140.0 / max(float(v.max()), 0.05)) if height_scale is None else height_scale
    h = v * scale                                           # km (vertical, exaggerated)

    if ax is None:
        fig = plt.figure(figsize=(8.5, 7.0))
        ax = fig.add_axes((0.0, -0.08, 0.92, 1.1), projection="3d")
    _need_3d(ax)
    fig = ax.figure
    ax.set_axis_off()
    ax.set_facecolor(t.surface)
    fig.set_facecolor(t.surface)

    if provinces:
        segs = []
        for f in _outlines()["features"]:
            for poly in f["geometry"]["coordinates"]:
                x, y = _km(*np.asarray(poly[0], float).T)
                segs.append(np.column_stack([x, y, np.zeros_like(x)]))
        ax.add_collection3d(Line3DCollection(segs, colors=t.secondary, linewidths=0.5,
                                             alpha=0.7), autolim=False)
    walls, wcols, owner = _walls(rings, h, rgba, azim)
    # painter's algorithm per prism: far cells first, each cell's walls before its top
    cam = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])
    depth = np.array([r.mean(axis=0) @ cam for r in rings])
    by_cell: dict[int, list[int]] = {}
    for j, k in enumerate(owner):
        by_cell.setdefault(k, []).append(j)
    polys, cols = [], []
    for k in np.argsort(depth):
        for j in by_cell.get(int(k), []):
            polys.append(walls[j])
            cols.append(wcols[j])
        polys.append([(*p, h[k]) for p in rings[k]])
        cols.append(tuple(rgba[k]))
    col = Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.25)
    col._zsortfunc = lambda z: 0.0        # keep the order above (stable sort, no re-sorting)
    ax.add_collection3d(col, autolim=False)

    allp = np.concatenate(rings)
    x0, x1, y0, y1 = allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()
    zmax = float(h.max())
    needle = 0.22 * zmax
    for s in resolve_points(hz, points):
        sx, sy = _km(s.lon, s.lat)
        z0 = float(sa.get(s.cell_id, np.nan) * scale)
        z0 = 0.0 if not np.isfinite(z0) else z0
        ax.plot([sx, sx], [sy, sy], [z0, z0 + needle], color=t.ink, lw=1.4, zorder=10)
        ax.plot([sx], [sy], [z0 + needle], ls="", marker=marker, ms=11, mew=4.4,
                color=t.surface, zorder=11)                  # halo
        ax.plot([sx], [sy], [z0 + needle], ls="", marker=marker, ms=11, mew=2.2,
                color=SITE_RED, zorder=12)
        if annotate and s.label:
            tx = ax.text(sx, sy, z0 + needle * 1.18, s.label, color=t.ink, fontsize=9,
                         ha="center", va="bottom", zorder=13)
            tx.set_path_effects(_style.halo(t, 3))
        zmax = max(zmax, z0 + needle * 1.3)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((x1 - x0, y1 - y0, zmax), zoom=zoom)
    ax.set_proj_type("ortho")                                 # isometric, no foreshortening
    ax.view_init(elev=elev, azim=azim)

    lbl = "PGA" if period == 0 else f"Sa({period:g} s)"
    if colorbar:
        cb = fig.colorbar(ScalarMappable(norm, cm), ax=ax, shrink=0.55, fraction=0.03,
                          pad=-0.02)
        cb.set_label(f"{lbl} [g]", color=t.secondary, fontsize=8.5)
        cb.outline.set_edgecolor(t.baseline)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(colors=t.muted, labelcolor=t.secondary, labelsize=8, width=0.6)
    if title:
        ax.text2D(0.02, 0.97, f"{lbl}, TR = {tr:g} yr ({stat})", transform=ax.transAxes,
                  fontsize=11, color=t.ink, fontweight="semibold", va="top")
        ax.text2D(0.02, 0.925, SUBTITLE, transform=ax.transAxes, fontsize=8.5,
                  color=t.secondary, va="top")
    return ax


def plot_iso_periods(hz: "EcuadorHazard", periods: Sequence[float] = (0.0, 0.2, 1.0, 2.0),
                     tr: float = 475, stat: str = "mean", *, ncols: int = 2,
                     colorbar: str = "panel", figsize: tuple[float, float] | None = None,
                     theme: str = "light", **kw) -> "Figure":
    """One isometric panel per period. See :meth:`EcuadorHazard.plot_iso_periods`."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from . import _style

    if colorbar not in ("panel", "shared"):
        raise ValueError("colorbar must be 'panel' or 'shared'")
    t = _style.theme(theme)
    periods = list(periods)
    ncols = max(1, min(ncols, len(periods)))
    nrows = -(-len(periods) // ncols)
    fig = plt.figure(figsize=figsize or (6.2 * ncols, 5.4 * nrows + 0.8))
    fig.set_facecolor(t.surface)
    if colorbar == "shared":
        allv = np.concatenate([hz.hazard_map(tr, p, stat).sa.dropna().to_numpy()
                               for p in periods])
        kw.setdefault("vmin", float(allv.min()))
        kw.setdefault("vmax", float(allv.max()))
        kw.setdefault("height_scale", 140.0 / float(allv.max()))
    fig.subplots_adjust(left=0.0, right=0.93 if colorbar == "shared" else 0.96, bottom=0.0,
                        top=0.93, wspace=0.08, hspace=-0.05)
    axes = []
    for i, p in enumerate(periods):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        plot_iso(hz, tr, p, stat, ax, theme=theme, colorbar=colorbar == "panel",
                 title=False, **{"zoom": 1.2, **kw})
        lbl = "PGA" if p == 0 else f"Sa({p:g} s)"
        ax.text2D(0.02, 0.97, lbl, transform=ax.transAxes, fontsize=10.5, color=t.ink,
                  fontweight="semibold", va="top")
        axes.append(ax)
    fig.suptitle(f"TR = {tr:g} yr ({stat})", x=0.02, y=0.985, ha="left", fontsize=12,
                 color=t.ink, fontweight="semibold")
    fig.text(0.02, 0.945, SUBTITLE, fontsize=8.5, color=t.secondary, va="top")
    if colorbar == "shared":
        cb = fig.colorbar(ScalarMappable(Normalize(kw["vmin"], kw["vmax"]),
                                         kw.get("cmap") or _style.hazard_cmap(t.name)),
                          ax=axes, shrink=0.5, pad=0.02)
        cb.set_label("Sa [g]", color=t.secondary, fontsize=8.5)
        cb.outline.set_edgecolor(t.baseline)
        cb.ax.tick_params(colors=t.muted, labelcolor=t.secondary, labelsize=8, width=0.6)
    return fig


def _save(fig: "Figure", path: "str | os.PathLike[str]", dpi: int) -> Path:
    import matplotlib.pyplot as plt

    out = Path(path)
    if out.suffix.lower() not in EXPORT_FORMATS:
        plt.close(fig)
        raise ValueError(f"unsupported extension {out.suffix!r}; use one of {EXPORT_FORMATS}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _check_ext(path) -> None:
    if Path(path).suffix.lower() not in EXPORT_FORMATS:
        raise ValueError(f"unsupported extension {Path(path).suffix!r}; "
                         f"use one of {EXPORT_FORMATS}")


def export_iso(hz: "EcuadorHazard", path: "str | os.PathLike[str]", tr: float = 475,
               period: float = 0.0, stat: str = "mean", points: PointsLike = None, *,
               dpi: int = 150, **kw) -> Path:
    """Write the isometric view to PNG / PDF / SVG. See :meth:`EcuadorHazard.export_iso`."""
    _check_ext(path)
    ax = plot_iso(hz, tr, period, stat, points=points, **kw)
    return _save(ax.figure, path, dpi)


def export_iso_periods(hz: "EcuadorHazard", path: "str | os.PathLike[str]",
                       periods: Sequence[float] = (0.0, 0.2, 1.0, 2.0), tr: float = 475,
                       stat: str = "mean", points: PointsLike = None, *, dpi: int = 150,
                       **kw) -> Path:
    """Write the multi-period figure to PNG / PDF / SVG."""
    _check_ext(path)
    return _save(plot_iso_periods(hz, periods, tr, stat, points=points, **kw), path, dpi)

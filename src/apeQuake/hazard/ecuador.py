"""Entry point to the IG-EPN seismic hazard database for Ecuador."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from . import _data
from .curves import powerlaw_fit, powerlaw_sa
from .site import HazardSite, Stat

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Point = tuple[float, float] | str

_EARTH_R_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


class EcuadorHazard:
    """Probabilistic seismic hazard of Ecuador, after IG-EPN (Beauval et al., 2018).

    A read-only view of the IG-EPN products bundled with apeQuake: uniform hazard
    spectra on a 0.08 deg grid (3146 cells, rock Vs30 = 760 m/s, TR = 475 and 2475 yr,
    mean / q16 / q50 / q84, PGA and 7 spectral periods), hazard curves digitized for the
    cantonal capitals, the seismic source model, earthquake catalogs and census data.
    Tables load lazily on first use.

    Examples
    --------
    >>> from apeQuake.hazard import EcuadorHazard
    >>> hz = EcuadorHazard()
    >>> quito = hz.site("Quito")                  # by name
    >>> site = hz.site(-2.19, -79.89)             # by coordinates (lat, lon)
    >>> quito.uhs(975)                            # UHS at any return period
    >>> quito.plot_hazard_curves()
    >>> hz.inventory()                            # everything that is available

    Always cite the source of the data, see :meth:`citation`.
    """

    # ------------------------------------------------------------------ info

    def __repr__(self) -> str:
        m = _data.manifest()
        return f"EcuadorHazard(IG-EPN snapshot retrieved {m['retrieved_utc'][:10]})"

    @staticmethod
    def citation() -> str:
        """References to cite when using this data."""
        m = _data.manifest()
        return (
            "Beauval C., Marinière J., Yepes H., Audin L., Nocquet J.M., Alvarado A., "
            "Baize S., Aguilar J., Singaucho J.C., Jomard H. (2018). A New Seismic Hazard "
            "Model for Ecuador. Bull. Seismol. Soc. Am. 108(3A), 1443-1464. "
            "doi:10.1785/0120170259\n"
            "Yepes H., Audin L., Alvarado A., Beauval C., Aguilar J., Font Y., Cotton F. "
            "(2016). A new view for the geodynamics of Ecuador: Implication in seismogenic "
            "source definition and seismic hazard assessment. Tectonics 35, 1249-1279. "
            "doi:10.1002/2015TC003941\n"
            "Instituto Geofísico - Escuela Politécnica Nacional (IG-EPN). Mapa Digital "
            "Interactivo del Peligro Sísmico Probabilístico para el Ecuador. Retrieved "
            f"{m['retrieved_utc'][:10]} from {m['url']}"
        )

    @staticmethod
    def manifest() -> dict[str, Any]:
        """Provenance of the snapshot: URLs, retrieval date, record counts."""
        return dict(_data.manifest())

    @staticmethod
    def inventory() -> pd.DataFrame:
        """Every dataset in the database, with row count and accessor."""
        c = _data.manifest()["counts"]
        rows = [
            ("hazard grid (UHS)", c["hazard_grid_rows"], "site(), uhs_at(), hazard_map()",
             "3146 cells x TR 475/2475 x mean/q16/q50/q84 x PGA + 7 periods [g]"),
            ("hazard cells", c["hazard_cells"], "cells()", "0.08 deg grid, centroids"),
            ("hazard curves (digitized)", int(_data.curves_qc().cell_id.nunique()),
             "site(...).hazard_curve()", "mean curves of the capital cells, 8 periods"),
            ("cantonal capitals", c["capitals"], "capitals()",
             "names, canton, province, cell, PGA 475/2475"),
            ("cabeceras population", c["cabeceras_population"], "population()",
             "census 2010, projection 2019, census 2022"),
            ("seismic sources", c["sources_beauval2018"], "sources()",
             "area sources: a, b, rate Mw>=4.5, Mmax, geometry"),
            ("faults", c["faults"], "faults()", "fault model: slip rates, Mmax, dip"),
            ("catalog shallow", c["catalog_shallow"], "catalog('shallow')",
             "homogenized PSHA catalog"),
            ("catalog deep", c["catalog_deep"], "catalog('deep')",
             "homogenized PSHA catalog, intermediate depth"),
            ("catalog historical", c["catalog_historical"], "catalog('historical')",
             "1587-2021"),
            ("provinces / cantons / parishes",
             c["admin_provinces"] + c["admin_cantons"] + c["admin_parishes"],
             "admin(level)", "INEC DPA names, codes, centroids, polygons"),
            ("recent events (live)", None, "apeQuake.hazard.fetch_recent_events()",
             "IG-EPN last 180 days, fetched online"),
        ]
        df = pd.DataFrame(rows, columns=["dataset", "rows", "accessor", "content"])
        df["rows"] = df["rows"].astype("Int64")
        return df

    # ------------------------------------------------------------------ tables

    @staticmethod
    def cells() -> pd.DataFrame:
        """Hazard grid cells: cell_id, lat, lon (centroid), area_km2."""
        return _data.cells().copy()

    @staticmethod
    def capitals() -> pd.DataFrame:
        """Cantonal and provincial capitals with their cell and PGA."""
        return _data.capitals().copy()

    @staticmethod
    def population() -> pd.DataFrame:
        """Population of the cantonal capitals (census 2010, 2019 projection, 2022)."""
        return _data.population().copy()

    @staticmethod
    def catalog(kind: Literal["shallow", "deep", "historical"] = "historical"
                ) -> pd.DataFrame:
        """Earthquake catalog: ``shallow`` / ``deep`` (homogenized PSHA catalog) or
        ``historical`` (1587-2021). ``depth_km`` is positive down in all three (the
        published historical layer is negative down; the sign is flipped here)."""
        if kind not in ("shallow", "deep", "historical"):
            raise ValueError("kind must be 'shallow', 'deep' or 'historical'")
        return _data.catalog(kind).copy()

    @staticmethod
    def admin(level: Literal["provinces", "cantons", "parishes"] = "cantons"
              ) -> pd.DataFrame:
        """Administrative units with names, codes and centroid coordinates."""
        return _data.admin(level).copy()

    @staticmethod
    def sources(model: Literal["beauval2018", "crust_interface"] = "beauval2018"
                ) -> pd.DataFrame:
        """Seismic area sources: one row per source, attributes plus ``geometry``
        (GeoJSON dict). ``beauval2018`` is the full set (incl. in-slab)."""
        return EcuadorHazard._features(f"sources_{model}.geojson")

    @staticmethod
    def faults() -> pd.DataFrame:
        """Fault model: attributes plus ``geometry`` (GeoJSON LineString)."""
        return EcuadorHazard._features("faults.geojson")

    @staticmethod
    def geojson(name: str) -> dict[str, Any]:
        """Raw GeoJSON of a bundled layer, e.g. ``"faults.geojson"``,
        ``"hazard_cells.geojson.gz"``, ``"admin_provinces.geojson.gz"``."""
        return _data.geojson(name)

    @staticmethod
    def _features(name: str) -> pd.DataFrame:
        fc = _data.geojson(name)
        rows = [{**f["properties"], "geometry": f["geometry"]} for f in fc["features"]]
        df = pd.DataFrame(rows).drop(columns=["OBJECTID", "OBJECTID_1"], errors="ignore")
        if "b" in df:
            df["b"] = pd.to_numeric(df["b"], errors="coerce")
        return df

    # ------------------------------------------------------------------ lookup

    @staticmethod
    def find(name: str) -> pd.DataFrame:
        """Places whose name matches ``name`` (accent / case insensitive).

        Searches capitals, parishes, cantons and provinces. ``score`` ranks exact
        matches first; capitals before parishes before cantons before provinces.
        """
        q = _data.normalize(name)
        out = []
        tables = [
            ("capital", _data.capitals(), "name", 0),
            ("parish", _data.admin("parishes"), "parish", 1),
            ("canton", _data.admin("cantons"), "canton", 2),
            ("province", _data.admin("provinces"), "province", 3),
        ]
        for kind, df, col, rank in tables:
            names = df[col].map(_data.normalize)
            exact = names == q
            starts = names.str.startswith(q) & ~exact
            contains = names.str.contains(q, regex=False) & ~exact & ~starts
            for mask, level in ((exact, 0), (starts, 10), (contains, 20)):
                for _, r in df[mask].iterrows():
                    out.append({
                        "kind": kind, "name": r[col],
                        "canton": r.get("canton"), "province": r.get("province"),
                        "lat": r["lat"], "lon": r["lon"], "score": level + rank,
                    })
        res = pd.DataFrame(out, columns=["kind", "name", "canton", "province", "lat", "lon",
                                         "score"])
        return res.sort_values(["score", "name"]).reset_index(drop=True)

    @staticmethod
    def _locate(lat: float, lon: float) -> tuple[pd.Series, float, bool]:
        """Cell whose polygon contains the point; else the nearest cell (inside=False)."""
        c = _data.cells().set_index("cell_id", drop=False)
        ids, paths, box = _data.cell_paths()
        cand = np.where((box[:, 0] <= lon) & (lon <= box[:, 1]) &
                        (box[:, 2] <= lat) & (lat <= box[:, 3]))[0]
        # a point exactly on a shared edge / corner belongs to no open polygon: retry with
        # a tiny tolerance (sign of radius depends on ring orientation, so try both)
        for radius in (0.0, 1e-7, -1e-7):
            for i in cand:
                if paths[i].contains_point((lon, lat), radius=radius):
                    row = c.loc[ids[i]]
                    return row, float(_haversine_km(lat, lon, row.lat, row.lon)), True
        d = _haversine_km(lat, lon, c.lat.to_numpy(), c.lon.to_numpy())
        i = int(np.argmin(d))
        return c.iloc[i], float(d[i]), False

    def site(self, where: float | str, lon: float | None = None, *,
             interp: Literal["cell", "bilinear"] = "cell", strict: bool = False
             ) -> HazardSite:
        """Hazard at a location given as ``(lat, lon)`` or as a place name.

        Parameters
        ----------
        where : float or str
            Latitude [deg, WGS84] (then ``lon`` is required) or a place name
            ("Quito", "Manta", "Baños"); see :meth:`find` for the candidates.
        lon : float, optional
            Longitude [deg].
        interp : {"cell", "bilinear"}
            ``cell`` returns the values of the IG-EPN cell containing the point (how
            IG-EPN publishes them; required for the digitized curves). ``bilinear``
            interpolates between the 4 surrounding cell centroids (falls back to
            ``cell`` at the border).
        strict : bool
            Raise instead of warning when the point is outside the hazard grid (offshore
            or abroad); otherwise the nearest cell is used.

        The containing cell is found by point-in-polygon on the published cell geometry:
        the cells are not a pure lattice (some positions are merged into a neighbour).
        """
        label = None
        if isinstance(where, str):
            cand = self.find(where)
            if cand.empty:
                raise KeyError(f"no place matches {where!r}")
            best = cand[cand.score == cand.score.iloc[0]]
            r = best.iloc[0]
            spread = _haversine_km(r.lat, r.lon, best.lat.to_numpy(), best.lon.to_numpy())
            if len(best) > 1 and spread.max() > 10:
                alts = ", ".join(f"{n} ({p})" for n, p in
                                 zip(best.name.iloc[1:6], best.province.iloc[1:6]))
                warnings.warn(f"{where!r} is ambiguous; using {r['name']} "
                              f"({r.province}). Other matches: {alts}. "
                              "Use find() and pass coordinates to choose.", stacklevel=2)
            lat, lon = float(r.lat), float(r.lon)
            kind = "" if r.kind == "capital" else f"{r.kind}, "
            label = f"{r['name']} ({kind}{r.province})"
        else:
            if lon is None:
                raise TypeError("site(lat, lon) needs both coordinates")
            lat = float(where)

        cell, dist, inside = self._locate(lat, float(lon))
        if not inside:
            msg = (f"({lat:.4f}, {lon:.4f}) is outside the IG-EPN hazard grid; nearest cell "
                   f"{cell.cell_id} is {dist:.1f} km away.")
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        g = _data.grid()
        values = g.loc[cell.cell_id]
        used = "cell"
        if interp == "bilinear":
            v = self._bilinear(lat, float(lon))
            if v is not None:
                values, used = v, "bilinear"
        elif interp != "cell":
            raise ValueError("interp must be 'cell' or 'bilinear'")
        return HazardSite(lat=lat, lon=float(lon), cell_id=cell.cell_id, cell_lat=cell.lat,
                          cell_lon=cell.lon, distance_km=dist, values=values.copy(),
                          interp=used, label=label)

    def _bilinear(self, lat: float, lon: float) -> pd.DataFrame | None:
        c = _data.cells()
        step = _data.GRID_STEP
        key = {(round(a, 2), round(b, 2)): cid for a, b, cid in zip(c.lat, c.lon, c.cell_id)}
        # cell centres sit on a regular lattice offset from the origin; anchor on any cell
        lat0, lon0 = float(c.lat.iloc[0]), float(c.lon.iloc[0])
        i = np.floor((lat - lat0) / step)
        j = np.floor((lon - lon0) / step)
        la, lo = lat0 + i * step, lon0 + j * step
        corners = [(la, lo), (la, lo + step), (la + step, lo), (la + step, lo + step)]
        ids = [key.get((round(a, 2), round(b, 2))) for a, b in corners]
        if any(x is None for x in ids):
            return None
        t, u = (lat - la) / step, (lon - lo) / step
        w = [(1 - t) * (1 - u), (1 - t) * u, t * (1 - u), t * u]
        g = _data.grid()
        return sum(wk * g.loc[cid] for wk, cid in zip(w, ids))

    def sites(self, places: Iterable[Point] | pd.DataFrame, **kw) -> list[HazardSite]:
        """Several sites at once: names, ``(lat, lon)`` tuples, or a DataFrame with
        ``lat`` / ``lon`` columns."""
        if isinstance(places, pd.DataFrame):
            places = list(zip(places["lat"], places["lon"]))
        return [self.site(p, **kw) if isinstance(p, str) else self.site(p[0], p[1], **kw)
                for p in places]

    def uhs_at(self, places: Iterable[Point] | pd.DataFrame, tr: float = 475,
               stat: Stat = "mean", **kw) -> pd.DataFrame:
        """UHS for many places in one table: one row per place, one column per period."""
        rows = []
        for s in self.sites(places, **kw):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = s.uhs(tr, stat)
            rows.append({"place": s.label, "lat": s.lat, "lon": s.lon, "cell_id": s.cell_id,
                         **dict(zip(_data.PERIODS, u.Sa))})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ maps

    @staticmethod
    def hazard_map(tr: float = 475, period: float = 0.0, stat: Stat = "mean"
                   ) -> pd.DataFrame:
        """Sa [g] in every cell for one period and return period.

        TR other than 475 / 2475 uses the per-cell power-law fit (extrapolated outside).
        Returns cell_id, lat, lon, sa.
        """
        idx = np.where(np.isclose(_data.PERIODS, period))[0]
        if not idx.size:
            raise ValueError(f"period must be one of {_data.PERIODS}")
        col = _data.PERIOD_COLUMNS[int(idx[0])]
        g = _data.grid()[col].unstack(["tr", "stat"])
        if tr in _data.RETURN_PERIODS:
            sa = g[(int(tr), stat)]
        else:
            k0, k = powerlaw_fit(g[(475, stat)], 475, g[(2475, stat)], 2475)
            sa = pd.Series(powerlaw_sa(tr, k0, k), index=g.index)
        out = _data.cells()[["cell_id", "lat", "lon"]].copy()
        out["sa"] = out.cell_id.map(sa)
        return out

    def plot_map(self, tr: float = 475, period: float = 0.0, stat: Stat = "mean",
                 ax: "Axes | None" = None, *, provinces: bool = True, faults: bool = False,
                 sources: "bool | str | Sequence[str]" = False,
                 catalog: "str | Sequence[str] | pd.DataFrame | None" = None,
                 min_mw: float | None = None, capitals: bool = False, points=None,
                 annotate: bool = True, extent: "str | Sequence[float] | None" = None,
                 cmap=None, vmin: float | None = None, vmax: float | None = None,
                 legend: bool = True, theme: Literal["light", "dark"] = "light") -> "Axes":
        """Static map of the hazard grid with overlays.

        Parameters
        ----------
        tr, period, stat
            What to color the 0.08 deg cells by (see :meth:`hazard_map`).
        provinces, faults, capitals : bool
            Province outlines, the fault model, the provincial capitals.
        sources : bool, str or list of str
            Area sources of Beauval et al. (2018): ``True`` for all, or any of
            ``"crustal"``, ``"interface"``, ``"inslab"``, ``"background"`` (they overlap,
            so picking types keeps the map readable).
        catalog : str, list of str or DataFrame
            Epicenters sized by magnitude: ``"shallow"``, ``"deep"``, ``"historical"``,
            a list of them, or any DataFrame with ``lat``, ``lon`` and ``mw`` (or
            ``magnitude``, e.g. :func:`fetch_recent_events`).
        min_mw : float, optional
            Only plot events with Mw >= ``min_mw``.
        points
            Sites to mark: names, ``(lat, lon)`` or ``(lat, lon, label)`` tuples,
            HazardSite objects or a DataFrame with ``lat`` / ``lon`` [/ ``label``].
        annotate : bool
            Label the points with their value (from :meth:`HazardSite.uhs`).
        extent : None, "points" or (lon_min, lon_max, lat_min, lat_max)
            Map window; ``"points"`` zooms on the given points.
        cmap, vmin, vmax, legend
            Styling. The default colormap is a single-hue blue ramp (light = low hazard),
            so faults, epicenters and sites can be drawn in ink on top of it.
        theme : {"light", "dark"}
            Color tokens; in dark mode the ramp flips so low hazard recedes into the
            dark surface.

        Source-zone types are the only colored overlays (orange / aqua / violet,
        background zones neutral). Faults, epicenters and sites are ink with a surface
        halo; catalogs differ by fill: shallow hollow, deep tinted, historical dashed,
        live / custom solid.
        """
        from .maps import plot_map

        return plot_map(self, tr, period, stat, ax, provinces=provinces, faults=faults,
                        sources=sources, catalog=catalog, min_mw=min_mw, capitals=capitals,
                        points=points, annotate=annotate, extent=extent, cmap=cmap,
                        vmin=vmin, vmax=vmax, legend=legend, theme=theme)

    def explore(self, path: str = "igepn_hazard_map.html", tr: float = 475,
                period: float = 0.0, stat: Stat = "mean", points=None,
                catalogs: Sequence[str] = ("shallow", "deep", "historical"),
                recent: "bool | pd.DataFrame" = False,
                point_trs: Sequence[float] = (475, 975, 2475),
                open_browser: bool = False, inline_leaflet: bool = True):
        """Write an interactive hazard map (standalone HTML, Leaflet); return its path.

        The page has street / terrain base maps and layers that can be switched on and
        off: the hazard grid (re-colored in the browser for any return period, period and
        statistic), faults, source zones by type, the earthquake catalogs (sized by
        magnitude), the cantonal capitals and your points. Clicking a cell shows its
        values and UHS; a "go to lat, lon" box finds the cell containing any point.

        Parameters
        ----------
        path : str
            Output file.
        tr, period, stat
            Initial map; all can be changed in the page. TR other than 475 / 2475 uses
            the per-cell power-law fit, as in :meth:`hazard_map`.
        points
            Sites to mark (same forms as :meth:`plot_map`). Their popups show the mean
            UHS from :meth:`HazardSite.uhs` (digitized curves where available) at
            ``point_trs``.
        catalogs : sequence of str
            Bundled catalogs to include: ``"shallow"``, ``"deep"``, ``"historical"``.
        recent : bool or DataFrame
            ``True`` fetches the IG-EPN last-180-days events now (needs internet), or
            pass a DataFrame from :func:`fetch_recent_events`.
        open_browser : bool
            Open the file in the default browser.
        inline_leaflet : bool
            Embed the Leaflet map library in the file (default, about +150 kB), so the
            page also works where external scripts are blocked: sandboxed file previews,
            mail or chat attachments, offline. ``False`` loads it from the unpkg CDN
            with integrity checks, for a smaller file.

        Everything but the base-map tiles is embedded in the file: without internet
        the hazard and all overlays still draw, on a blank background.
        """
        from .maps import explore

        return explore(self, path, tr, period, stat, points, catalogs, recent, point_trs,
                       open_browser, inline_leaflet)

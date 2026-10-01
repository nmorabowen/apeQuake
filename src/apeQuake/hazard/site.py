"""Seismic hazard at one site: UHS at any return period, hazard curves, plots."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from . import _data
from .curves import loglog_interp, powerlaw_fit, powerlaw_rate, powerlaw_sa

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Stat = Literal["mean", "q16", "q50", "q84"]
Method = Literal["auto", "published", "digitized", "fit"]

_TR1, _TR2 = _data.RETURN_PERIODS


class HazardSite:
    """IG-EPN seismic hazard (rock, Vs30 = 760 m/s) at one location.

    Created by :meth:`EcuadorHazard.site`; not meant to be built directly.

    Three sources of numbers, always labelled in the ``source`` column of the outputs:

    * ``published``: the IG-EPN UHS ordinates at TR = 475 and 2475 yr (mean, q16, q50,
      q84), exactly as published.
    * ``digitized``: the mean hazard curves digitized from the IG-EPN images. Only for
      the 365 cells containing a cantonal capital, and only over the span that could be
      read from the image (see ``data/igepn/README.md``).
    * ``fit`` / ``fit-extrapolated``: lambda = k0 Sa^-k through the two published points
      of each period. Exact at 475 and 2475 yr, a log-log interpolation between them, an
      extrapolation outside them.

    Attributes
    ----------
    lat, lon : float
        Query coordinates.
    cell_id : str
        IG-EPN grid cell (0.08 deg) containing / nearest to the query point.
    cell_lat, cell_lon : float
        Centroid of that cell.
    distance_km : float
        Distance from the query point to the cell centroid.
    label : str
        Human-readable name (place name for name queries, else coordinates).
    interp : {"cell", "bilinear"}
        How the UHS ordinates were obtained from the grid.
    """

    def __init__(self, *, lat: float, lon: float, cell_id: str, cell_lat: float,
                 cell_lon: float, distance_km: float, values: pd.DataFrame,
                 interp: str = "cell", label: str | None = None) -> None:
        self.lat, self.lon = float(lat), float(lon)
        self.cell_id = cell_id
        self.cell_lat, self.cell_lon = float(cell_lat), float(cell_lon)
        self.distance_km = float(distance_km)
        self.interp = interp
        self.label = label or f"({self.lat:.4f}, {self.lon:.4f})"
        # values: index (tr, stat), columns T0.00 ... T2.00
        self._values = values

    # ------------------------------------------------------------------ basics

    def __repr__(self) -> str:
        pga = self._values.loc[(_TR1, "mean"), "T0.00"]
        return (f"HazardSite({self.label!r}, cell={self.cell_id}, "
                f"PGA475={pga:.3f} g, digitized_curves={self.has_digitized_curves})")

    @property
    def periods(self) -> np.ndarray:
        """Spectral periods [s] of the model (0.0 = PGA)."""
        return np.asarray(_data.PERIODS)

    @property
    def capital(self) -> pd.DataFrame:
        """Cantonal capitals located in this cell (empty if none)."""
        c = _data.capitals()
        return c[c.cell_id == self.cell_id].reset_index(drop=True)

    @property
    def has_digitized_curves(self) -> bool:
        """True if digitized hazard curves exist for this site's cell.

        They are used only with ``interp="cell"``: with bilinear interpolation the UHS
        no longer belongs to a single cell, so the cell's curve would not match it.
        """
        return self.interp == "cell" and self.cell_id in set(_data.curves_qc().cell_id)

    def published(self, tr: int = 475, stat: Stat = "mean") -> np.ndarray:
        """Published UHS ordinates [g] for ``tr`` in (475, 2475), ordered like ``periods``."""
        if tr not in _data.RETURN_PERIODS:
            raise ValueError(f"IG-EPN publishes TR = {_data.RETURN_PERIODS}; got {tr}. "
                             "Use uhs(tr) for other return periods.")
        if stat not in _data.STATS:
            raise ValueError(f"stat must be one of {_data.STATS}")
        return self._values.loc[(tr, stat), list(_data.PERIOD_COLUMNS)].to_numpy(float)

    def table(self) -> pd.DataFrame:
        """All published ordinates: index = period [s], columns = (tr, stat)."""
        t = self._values.T
        t.index = list(_data.PERIODS)
        t.index.name = "T"
        return t

    # ------------------------------------------------------------------ fits

    def _fit(self, stat: Stat) -> tuple[np.ndarray, np.ndarray]:
        return powerlaw_fit(self.published(_TR1, stat), _TR1, self.published(_TR2, stat), _TR2)

    def _digitized(self, period: float) -> pd.DataFrame | None:
        if not self.has_digitized_curves:
            return None
        c = _data.curves()
        c = c[(c.cell_id == self.cell_id) & np.isclose(c.period, period)]
        return c[["sa_g", "rate"]].reset_index(drop=True) if len(c) else None

    # ------------------------------------------------------------------ UHS

    def uhs(self, tr: float = 475, stat: Stat = "mean", method: Method = "auto"
            ) -> pd.DataFrame:
        """Uniform hazard spectrum for return period ``tr`` [yr].

        Parameters
        ----------
        tr : float
            Return period [yr]. Use :func:`apeQuake.hazard.tr_from_poe` to convert from a
            probability of exceedance (e.g. 10 % in 50 yr -> 475).
        stat : {"mean", "q16", "q50", "q84"}
            Logic-tree statistic. Digitized curves exist only for the mean.
        method : {"auto", "published", "digitized", "fit"}
            ``auto`` uses the published ordinates at 475 / 2475, else the digitized
            curve where it covers 1/tr, else the power-law fit.

        Returns
        -------
        DataFrame with columns ``T`` [s], ``Sa`` [g] and ``source``.
        """
        tr = float(tr)
        sa = np.full(len(_data.PERIODS), np.nan)
        src = np.full(len(_data.PERIODS), "", dtype=object)

        if method in ("auto", "published") and tr in _data.RETURN_PERIODS:
            sa[:] = self.published(int(tr), stat)
            src[:] = "published"
        elif method == "published":
            raise ValueError(f"published values exist only for TR = {_data.RETURN_PERIODS}")

        if method in ("auto", "digitized") and stat == "mean":
            for i, T in enumerate(_data.PERIODS):
                if src[i]:
                    continue
                d = self._digitized(T)
                if d is not None:
                    v = loglog_interp(1.0 / tr, d.rate.to_numpy(), d.sa_g.to_numpy())
                    if np.isfinite(v):
                        sa[i], src[i] = float(v), "digitized"

        if method in ("auto", "fit"):
            k0, k = self._fit(stat)
            fit = powerlaw_sa(tr, k0, k)
            inside = _TR1 <= tr <= _TR2
            for i in range(len(sa)):
                if not src[i]:
                    sa[i] = fit[i]
                    src[i] = "fit" if inside else "fit-extrapolated"
            if not inside and (src == "fit-extrapolated").any():
                warnings.warn(
                    f"TR = {tr:g} yr is outside the published {_TR1}-{_TR2} yr range; the "
                    "power-law extrapolation ignores the curvature of the hazard curve "
                    "(typically overestimates Sa at long TR, underestimates at short TR).",
                    stacklevel=2)
        src[src == ""] = "unavailable"
        return pd.DataFrame({"T": _data.PERIODS, "Sa": sa, "source": src})

    def uhs_table(self, trs: Iterable[float] = (72, 225, 475, 975, 2475),
                  stat: Stat = "mean", method: Method = "auto") -> pd.DataFrame:
        """UHS for several return periods: index = period [s], one column per TR."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cols = {tr: self.uhs(tr, stat, method).set_index("T")["Sa"] for tr in trs}
        out = pd.DataFrame(cols)
        out.columns.name = "TR [yr]"
        return out

    # ------------------------------------------------------------------ hazard curves

    def hazard_curve(self, period: float = 0.0, sa: Sequence[float] | None = None,
                     method: Method = "auto") -> pd.DataFrame:
        """Mean hazard curve (annual rate of exceedance vs. Sa) for one period.

        Parameters
        ----------
        period : float
            Spectral period [s], one of ``periods`` (0.0 = PGA).
        sa : sequence of float, optional
            Accelerations [g] at which to evaluate. Default: the digitized points when
            available, else 60 log-spaced values spanning the published range.
        method : {"auto", "digitized", "fit"}
            ``auto`` uses the digitized curve within its span and the fit elsewhere.

        Returns
        -------
        DataFrame with ``sa_g``, ``rate`` [1/yr], ``tr`` [yr] and ``source``.
        """
        i = self._period_index(period)
        k0, k = (v[i] for v in self._fit("mean"))
        sa1, sa2 = self.published(_TR1)[i], self.published(_TR2)[i]
        d = self._digitized(_data.PERIODS[i]) if method in ("auto", "digitized") else None
        if method == "digitized" and d is None:
            raise ValueError(f"no digitized curve for cell {self.cell_id}")

        if sa is None:
            if d is not None:
                sa = d.sa_g.to_numpy()
            else:
                sa = np.geomspace(sa1 / 3.0, sa2 * 2.0, 60)
        sa = np.asarray(sa, float)

        rate = np.full(sa.shape, np.nan)
        src = np.full(sa.shape, "unavailable", dtype=object)
        if d is not None:
            rate = loglog_interp(sa, d.sa_g.to_numpy(), d.rate.to_numpy())
            src[np.isfinite(rate)] = "digitized"
        if method in ("auto", "fit"):
            miss = ~np.isfinite(rate)
            rate[miss] = powerlaw_rate(sa[miss], k0, k)
            inside = (sa >= sa1) & (sa <= sa2)
            src[miss & inside] = "fit"
            src[miss & ~inside] = "fit-extrapolated"
        with np.errstate(divide="ignore"):
            tr = 1.0 / rate
        return pd.DataFrame({"sa_g": sa, "rate": rate, "tr": tr, "source": src})

    def hazard_curves(self, method: Method = "auto") -> pd.DataFrame:
        """Hazard curves of all periods, long format (adds a ``period`` column)."""
        frames = []
        for T in _data.PERIODS:
            c = self.hazard_curve(T, method=method)
            c.insert(0, "period", T)
            frames.append(c)
        return pd.concat(frames, ignore_index=True)

    def return_period(self, period: float, sa: float | Sequence[float]) -> np.ndarray:
        """Return period [yr] of exceeding ``sa`` [g] at ``period``."""
        return self.hazard_curve(period, sa=np.atleast_1d(sa))["tr"].to_numpy()

    def _period_index(self, period: float) -> int:
        idx = np.where(np.isclose(_data.PERIODS, period))[0]
        if not idx.size:
            raise ValueError(f"period must be one of {_data.PERIODS}; got {period}")
        return int(idx[0])

    # ------------------------------------------------------------------ plots

    def plot_uhs(self, trs: Iterable[float] = (475, 2475), stat: Stat = "mean",
                 band: bool = True, ax: "Axes | None" = None, logx: bool = False,
                 theme: Literal["light", "dark"] = "light") -> "Axes":
        """Plot UHS for the given return periods.

        Return periods are ordered, so they share one hue from light (short TR) to dark
        (long TR). ``band`` adds the q16-q84 wash for the published return periods.
        Extrapolated spectra (outside 475-2475 yr) are dashed. Up to four spectra are
        labelled at their right end as well as in the legend.
        """
        import matplotlib.pyplot as plt

        from . import _style

        t = _style.theme(theme)
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.6))
        _style.style_axes(ax, t)
        trs = sorted(float(v) for v in trs)
        colors, _ = _style.ordinal(len(trs), t.name)
        ends: list[tuple[float, float, str]] = []
        T = np.asarray(_data.PERIODS)
        x = np.where(T == 0, 0.01, T) if logx else T
        for tr, col in zip(trs, colors):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = self.uhs(tr, stat)
            extrap = (u.source == "fit-extrapolated").any()
            if band and tr in _data.RETURN_PERIODS:
                ax.fill_between(x, self.published(int(tr), "q16"),
                                self.published(int(tr), "q84"), color=col, alpha=0.12, lw=0)
            ax.plot(x, u.Sa, "--" if extrap else "-", color=col, lw=2,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=f"{tr:g} yr" + (" (extrapolated)" if extrap else ""))
            ax.plot(x, u.Sa, "o", ms=6, color=col, mec=t.surface, mew=1.6, zorder=3)
            ends.append((x[-1], u.Sa.iloc[-1], f"{tr:g} yr"))
        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, x[-1] * 1.16)
        ax.set_ylim(0, None)
        if len(trs) <= 4:                          # direct labels supplement the legend
            _style.end_labels(ax, t, ends)
        ax.set_xlabel("Period T [s]" + (" (PGA plotted at 0.01 s)" if logx else ""))
        ax.set_ylabel("Sa [g]")
        sub = f"{stat}, rock Vs30 = 760 m/s" + (", shaded q16-q84" if band else "")
        _style.title(ax, t, f"Uniform hazard spectra - {self.label}", sub)
        _style.legend(ax, t, loc="upper right", title="Return period",
                      title_fontsize=8).get_title().set_color(t.secondary)
        return ax

    def plot_hazard_curves(self, periods: Iterable[float] | None = None,
                           ax: "Axes | None" = None, anchors: bool = True,
                           theme: Literal["light", "dark"] = "light") -> "Axes":
        """Plot mean hazard curves.

        Digitized IG-EPN curves are solid; the power-law fit is dashed (it is a model
        between / beyond the two published points). Periods are ordered, so they share one
        hue; the default shows five (PGA, 0.2, 0.5, 1 and 2 s). With more than five the
        curves are also labelled at their ends, since one hue cannot separate them alone.
        ``anchors`` marks the published UHS points (1/475, 1/2475).
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from . import _style

        t = _style.theme(theme)
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.2))
        _style.style_axes(ax, t)
        periods = (0.0, 0.2, 0.5, 1.0, 2.0) if periods is None else tuple(periods)
        periods = tuple(sorted(periods))
        colors, label_ends = _style.ordinal(len(periods), t.name)
        ends: list[tuple[float, float, str]] = []
        for T, col in zip(periods, colors):
            i = self._period_index(T)
            lbl = "PGA" if T == 0 else f"{T:g} s"
            sa1, sa2 = self.published(_TR1)[i], self.published(_TR2)[i]
            fs = np.geomspace(sa1 / 3.0, sa2 * 2.0, 60)
            fc = self.hazard_curve(T, sa=fs, method="fit")
            dig = self.hazard_curve(T)
            dig = dig[dig.source == "digitized"]
            # the fit is context when a digitized curve exists: lighter, thinner
            ax.loglog(fc.sa_g, fc.rate, linestyle=(0, (4, 3)), color=col,
                      lw=1.0 if len(dig) else 1.6, alpha=0.55 if len(dig) else 1.0,
                      label=None if len(dig) else lbl)
            end = fc
            if len(dig):
                ax.loglog(dig.sa_g, dig.rate, "-", color=col, lw=2, label=lbl,
                          solid_capstyle="round")
                end = dig
            if anchors:
                ax.plot([sa1, sa2], [1 / _TR1, 1 / _TR2], "o", ms=6, color=col,
                        mec=t.surface, mew=1.6, zorder=3)
            ends.append((end.sa_g.iloc[-1], end.rate.iloc[-1], lbl))
        ax.grid(True, which="minor", color=t.grid, linewidth=0.4, alpha=0.6)
        for tr in _data.RETURN_PERIODS:
            ax.axhline(1 / tr, color=t.baseline, lw=0.8, zorder=1)
            ax.annotate(f"{tr} yr", (1.0, 1 / tr), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right",
                        fontsize=7.5, color=t.muted)
        ax.set_xlabel("Sa [g]")
        ax.set_ylabel("Annual rate of exceedance [1/yr]")
        src = ("solid: digitized IG-EPN curve, dashed: power-law fit"
               if self.has_digitized_curves else
               "power-law fit through the published 475 / 2475 yr values "
               "(no digitized curve for this cell)")
        _style.title(ax, t, f"Mean hazard curves - {self.label}", src)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], ls="", marker="o", ms=6, color=t.secondary,
                              mec=t.surface, mew=1.6))
        labels.append("published UHS point")
        _style.legend(ax, t, handles=handles, labels=labels, loc="lower left")
        if label_ends:                             # > 5 periods: one hue is not enough
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo, hi * 2.2)              # room for the labels
            _style.end_labels(ax, t, ends)
        return ax

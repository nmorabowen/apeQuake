"""Digitized NEC-SE-DS seismic-hazard curves, with interpolation and extrapolation.

The NEC-15 *Peligro sismico* chapter (section 10.3, Figures 10-32) plots, for each
provincial capital, the annual exceedance rate of PGA and of Sa at T = 0.1, 0.2,
0.5 and 1.0 s (rock, 5 % damping).  This module ships those curves as numbers
(see ``scripts/digitize_nec_hazard.py`` for how they were recovered) and adds the
operations a designer needs:

* ``rate_at(a)`` / ``a_at_rate(rate)`` / ``a_at_return_period(Tr)``;
* a log-log tail model so the curves can be pushed past the plotted range
  (long return periods, rates below 1e-5, accelerations beyond the x-axis);
* a uniform-hazard spectrum built from the five measures.

Accuracy: the plots are raster images, so values carry roughly +-2-4 % in
annual rate (1-2 px on a 5-decade axis).  Treat results as design-aid numbers,
not as a replacement for the IG-EPN hazard model.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Iterable, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = [
    "NEC_RETURN_PERIODS",
    "HazardCurve",
    "CityHazard",
    "HazardDatabase",
    "load_hazard_database",
]

#: The four NEC hazard levels (Table 9): name -> return period in years.
NEC_RETURN_PERIODS: dict[str, float] = {
    "frecuente": 72.0,
    "ocasional": 225.0,
    "raro": 475.0,
    "muy raro": 2500.0,
}

#: Structural periods (s) of the five measures; 0 is PGA.
_PERIOD_OF = {"PGA": 0.0, "0.1": 0.1, "0.2": 0.2, "0.5": 0.5, "1.0": 1.0}

TailModel = Literal["quadratic", "linear"]


@dataclass(frozen=True, eq=False)
class HazardCurve:
    """Annual exceedance rate versus acceleration for one measure.

    Parameters
    ----------
    key
        ``"PGA"``, ``"0.1"``, ``"0.2"``, ``"0.5"`` or ``"1.0"`` (period in s).
    a
        Acceleration in g, strictly increasing.
    rate
        Annual exceedance rate (1/yr), strictly decreasing.
    aliased_from
        Set when the source plot hides this curve under another one (Sa(0.2 s)
        under Sa(0.1 s) for a few cities); the values are then the other curve's.
    """

    key: str
    a: np.ndarray
    rate: np.ndarray
    aliased_from: str | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        a = np.asarray(self.a, float)
        r = np.asarray(self.rate, float)
        # The digitized path is monotone in both axes by construction; drop any
        # repeated sample so interpolation stays well defined.
        # Compare against the last *kept* point, so a stray out-of-order sample
        # (a slightly smaller a on a near-vertical start) cannot leave duplicates.
        keep = [0]
        for i in range(1, len(a)):
            if a[i] > a[keep[-1]] + 1e-9 and r[i] < r[keep[-1]] * (1 - 1e-9):
                keep.append(i)
        object.__setattr__(self, "a", a[keep])
        object.__setattr__(self, "rate", r[keep])

    @property
    def period(self) -> float:
        """Structural period in seconds (0 for PGA)."""
        return _PERIOD_OF[self.key]

    @property
    def a_range(self) -> tuple[float, float]:
        """Acceleration range (g) covered by the digitized data."""
        return float(self.a[0]), float(self.a[-1])

    # -- interpolation -----------------------------------------------------
    def _interp(self) -> PchipInterpolator:
        if "f" not in self._cache:
            self._cache["f"] = PchipInterpolator(np.log(self.a), np.log(self.rate))
        return self._cache["f"]

    def _tail(self, model: TailModel, frac: float) -> tuple[np.ndarray, float]:
        """Fit ln(rate) on the high-acceleration tail; returns (coeffs, ln a_end)."""
        ck = ("tail", model, frac)
        if ck not in self._cache:
            n = len(self.a)
            sel = slice(int(n * (1.0 - frac)), n)
            x, y = np.log(self.a[sel]), np.log(self.rate[sel])
            coef = np.polyfit(x, y, 2 if model == "quadratic" else 1)
            if model == "quadratic":
                # a rising / flattening tail is unphysical: fall back to the end slope
                slope_end = 2.0 * coef[0] * x[-1] + coef[1]
                if slope_end >= 0 or coef[0] > 0:
                    k = max(8, len(x) // 4)
                    coef = np.polyfit(x[-k:], y[-k:], 1)
            self._cache[ck] = (coef, float(np.log(self.a[-1])))
        return self._cache[ck]

    def rate_at(
        self,
        a: float | Iterable[float],
        *,
        extrapolate: bool = True,
        tail: TailModel = "quadratic",
        tail_fraction: float = 0.3,
    ) -> np.ndarray:
        """Annual exceedance rate at acceleration(s) *a* (g).

        Inside the digitized range the curve is interpolated (monotone PCHIP in
        log-log).  Above it, with ``extrapolate=True``, a log-log polynomial
        fitted to the last ``tail_fraction`` of the curve is used (see
        :meth:`extrapolate`); with ``False`` the result is NaN there.  Below the
        first digitized point the rate is held at its initial value.
        """
        a = np.atleast_1d(np.asarray(a, float))
        out = np.empty_like(a)
        lo, hi = self.a_range
        inside = (a >= lo) & (a <= hi)
        low = a < lo
        high = a > hi
        out[inside] = np.exp(self._interp()(np.log(a[inside])))
        out[low] = self.rate[0]
        if high.any():
            if not extrapolate:
                out[high] = np.nan
            else:
                coef, _ = self._tail(tail, tail_fraction)
                out[high] = np.exp(np.polyval(coef, np.log(a[high])))
        return out

    def a_at_rate(
        self,
        rate: float | Iterable[float],
        *,
        extrapolate: bool = True,
        tail: TailModel = "quadratic",
        tail_fraction: float = 0.3,
    ) -> np.ndarray:
        """Acceleration (g) whose annual exceedance rate equals *rate*."""
        rate = np.atleast_1d(np.asarray(rate, float))
        out = np.empty_like(rate)
        for i, r in enumerate(rate):
            if r >= self.rate[0]:
                out[i] = self.a[0]
            elif r >= self.rate[-1]:
                out[i] = float(
                    np.exp(np.interp(np.log(r), np.log(self.rate[::-1]), np.log(self.a[::-1])))
                )
            elif not extrapolate:
                out[i] = np.nan
            else:
                out[i] = self._invert_tail(r, tail, tail_fraction)
        return out

    def _invert_tail(self, rate: float, tail: TailModel, frac: float) -> float:
        coef, x_end = self._tail(tail, frac)
        y = np.log(rate)
        if len(coef) == 2:
            return float(np.exp((y - coef[1]) / coef[0]))
        c2, c1, c0 = coef
        roots = np.roots([c2, c1, c0 - y])  # c2 x^2 + c1 x + (c0 - y) = 0
        roots = roots[np.isreal(roots)].real
        roots = roots[roots >= x_end]  # only the branch beyond the data
        return float(np.exp(roots.min())) if roots.size else float("nan")

    def a_at_return_period(self, return_period: float | Iterable[float], **kw) -> np.ndarray:
        """Acceleration (g) with mean return period *Tr* years (rate = 1/Tr)."""
        return self.a_at_rate(1.0 / np.atleast_1d(np.asarray(return_period, float)), **kw)

    def probability_of_exceedance(
        self, a: float | Iterable[float], years: float = 50.0, **kw
    ) -> np.ndarray:
        """Poisson probability of exceeding *a* at least once in *years*."""
        return 1.0 - np.exp(-self.rate_at(a, **kw) * years)

    def extrapolate(
        self,
        a_max: float,
        *,
        n: int = 60,
        tail: TailModel = "quadratic",
        tail_fraction: float = 0.3,
    ) -> "HazardCurve":
        """Return a new curve extended to *a_max* g with the tail model."""
        hi = self.a_range[1]
        if a_max <= hi:
            return self
        ext = np.geomspace(hi, a_max, n + 1)[1:]
        r = self.rate_at(ext, tail=tail, tail_fraction=tail_fraction)
        return HazardCurve(
            self.key,
            np.concatenate([self.a, ext]),
            np.concatenate([self.rate, r]),
            self.aliased_from,
        )


@dataclass(frozen=True, eq=False)
class CityHazard:
    """The five hazard curves of one provincial capital."""

    name: str
    figure: int
    lat: float
    lon: float
    curves: dict[str, HazardCurve]

    def __getitem__(self, key: str | float) -> HazardCurve:
        """Curve by key (``"PGA"``, ``"0.2"``) or period in s (``0``, ``0.2``)."""
        if isinstance(key, (int, float)):
            key = "PGA" if key == 0 else f"{float(key):.1f}"
        return self.curves[key]

    @property
    def periods(self) -> np.ndarray:
        """Structural periods (s) of the available measures, ascending."""
        return np.array(sorted(c.period for c in self.curves.values()))

    def uhs(self, return_period: float = 475.0, **kw) -> tuple[np.ndarray, np.ndarray]:
        """Uniform-hazard spectrum: ``(periods, Sa in g)`` at one return period.

        Built from PGA (T = 0) and Sa at 0.1, 0.2, 0.5, 1.0 s only; the plots
        carry no other periods, so the shape between them is a straight line.
        """
        periods = self.periods
        sa = np.array([self[p].a_at_return_period(return_period, **kw)[0] for p in periods])
        return periods, sa

    def hazard_table(self, return_periods: dict[str, float] | None = None, **kw):
        """DataFrame of PGA / Sa (g) at the NEC hazard levels (or *return_periods*)."""
        import pandas as pd

        rows = {}
        for label, tr in (return_periods or NEC_RETURN_PERIODS).items():
            rows[f"{label} (Tr={tr:g})"] = {
                ("PGA" if p == 0 else f"Sa({p:g}s)"): self[p].a_at_return_period(tr, **kw)[0]
                for p in self.periods
            }
        return pd.DataFrame(rows).T

    def plot(
        self,
        *,
        extrapolate_to: float | None = None,
        return_periods: Iterable[float] | None = (72, 225, 475, 2500),
        ax=None,
        **kw,
    ):
        """NEC-style plot (log rate vs g) with optional dashed extrapolation."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5))
        colors = {"PGA": "k", "0.1": "#b34696", "0.2": "#96612b", "0.5": "#4fae3a", "1.0": "#2f46a0"}
        for key in ("1.0", "0.5", "0.2", "0.1", "PGA"):
            c = self.curves[key]
            label = "PGA" if key == "PGA" else f"Sa({key} s)"
            ax.semilogy(c.a, c.rate, color=colors[key], lw=1.8, label=label)
            if extrapolate_to and extrapolate_to > c.a_range[1]:
                e = c.extrapolate(extrapolate_to, **kw)
                m = e.a >= c.a_range[1]
                ax.semilogy(e.a[m], e.rate[m], color=colors[key], lw=1.8, ls="--")
        xmax = ax.get_xlim()[1]
        for tr in return_periods or ():
            ax.axhline(1.0 / tr, color="0.6", lw=0.8, ls=":")
            ax.text(xmax, 1.0 / tr, f" Tr={tr:g}", va="center", fontsize=8, color="0.4")
        ax.set_ylim(1e-6 if extrapolate_to else 1e-5, 1.0)
        ax.set_xlabel("Acceleration (g)")
        ax.set_ylabel("Annual exceedance rate (1/yr)")
        ax.set_title(f"{self.name} ({self.lat:g}, {self.lon:g}) - NEC-SE-DS Fig. {self.figure}")
        ax.grid(True, which="both", color="0.85", lw=0.5)
        ax.legend(loc="upper right", frameon=False)
        return ax


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


class HazardDatabase:
    """All 23 digitized cities; a read-only ``name -> CityHazard`` mapping."""

    def __init__(self, raw: dict) -> None:
        self.source: str = raw["source"]
        self._cities: dict[str, CityHazard] = {}
        for name, c in raw["cities"].items():
            curves = {
                k: HazardCurve(k, np.array(v["a"]), np.array(v["rate"]), v.get("aliased_from"))
                for k, v in c["curves"].items()
            }
            self._cities[name] = CityHazard(name, c["figure"], c["lat"], c["lon"], curves)
        self._lookup = {_norm(n): n for n in self._cities}

    def __len__(self) -> int:
        return len(self._cities)

    def __iter__(self):
        return iter(self._cities.values())

    @property
    def names(self) -> list[str]:
        """City names in NEC figure order."""
        return list(self._cities)

    def __getitem__(self, name: str) -> CityHazard:
        """Case- and accent-insensitive lookup (``"Tulcán"`` == ``"tulcan"``)."""
        try:
            return self._cities[self._lookup[_norm(name)]]
        except KeyError:
            raise KeyError(f"{name!r} not found; available: {', '.join(self._cities)}") from None

    def nearest(self, lat: float, lon: float) -> CityHazard:
        """City closest to (lat, lon) by great-circle distance."""
        la, lo = np.radians(lat), np.radians(lon)

        def dist(c: CityHazard) -> float:
            la2, lo2 = np.radians(c.lat), np.radians(c.lon)
            h = np.sin((la2 - la) / 2) ** 2 + np.cos(la) * np.cos(la2) * np.sin((lo2 - lo) / 2) ** 2
            return float(2 * np.arcsin(np.sqrt(h)))

        return min(self._cities.values(), key=dist)


@lru_cache(maxsize=1)
def load_hazard_database() -> HazardDatabase:
    """Load the packaged NEC hazard curves (cached)."""
    path = resources.files("apeQuake.nec").joinpath("data/nec_hazard_curves.json")
    with path.open("r", encoding="utf-8") as fh:
        return HazardDatabase(json.load(fh))

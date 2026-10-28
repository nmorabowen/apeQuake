from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import numpy as np
import pandas as pd

from apeQuake.core.integration import integrate
from apeQuake.core.types import ComponentName
from apeQuake.core.units import G, TO_MPS2, normalize_unit

if TYPE_CHECKING:
    from apeQuake.core.record import Record
    from apeQuake.filters.filters import Filter  # adjust to your package


class IntensityMeasures:
    """
    IM composite following Spectrogram/Spectrum stateful pattern:
    - df_im is the working dataframe (mutable by apply_* wrappers)
    - outputs are computed on current df_im unless a custom df is passed
    - applied_filters records what was applied to df_im
    """

    def __init__(self, record: "Record") -> None:
        self.record = record
        self.filter: "Filter" = record.filter

        self.df_im: pd.DataFrame = record.df.copy()

        # bookkeeping / logs
        self.applied_filters: list[str] = []

    # ------------------------- DF lifecycle ------------------------- #

    def set_df(self, df: pd.DataFrame) -> "IntensityMeasures":
        self.df_im = df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def reset_df(self) -> "IntensityMeasures":
        self.df_im = self.record.df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def _invalidate(self) -> None:
        # Currently no cached IM outputs, but kept for symmetry/future caching.
        pass

    def _log(self, s: str) -> None:
        self.applied_filters.append(s)

    # ------------------------- wrappers ---------------------------- #
    # (match your Filter API: each takes df= and returns a DataFrame)

    def apply_low_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "IntensityMeasures":
        self.df_im = self.filter.low_pass(df=self.df_im, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"low_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_high_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "IntensityMeasures":
        self.df_im = self.filter.high_pass(df=self.df_im, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"high_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_pass(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "IntensityMeasures":
        self.df_im = self.filter.band_pass(
            df=self.df_im, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_pass(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_stop(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "IntensityMeasures":
        self.df_im = self.filter.band_stop(
            df=self.df_im, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_stop(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_detrend(self, type: str = "demean") -> "IntensityMeasures":
        self.df_im = self.filter.detrend(df=self.df_im, type=type)
        self._invalidate()
        self._log(f"detrend(type='{type}')")
        return self

    def apply_taper(self, max_percentage: float = 0.05, *, type: str = "cosine") -> "IntensityMeasures":
        self.df_im = self.filter.taper(df=self.df_im, max_percentage=max_percentage, type=type)
        self._invalidate()
        self._log(f"taper(max_percentage={max_percentage}, type='{type}')")
        return self


    def apply_baseline_correction(self, order: int = 2) -> "IntensityMeasures":
        self.df_im = self.filter.baseline_correction(df=self.df_im, order=order)
        self._invalidate()
        self._log(f"baseline_correction(order={order})")
        return self

    def apply_base_filters(
        self,
        *,
        detrend: bool = True,
        demean: bool = True,
        taper: float | None = 0.05,
        band: tuple[float, float] | None = None,
    ) -> "IntensityMeasures":
        """
        Apply the 'standard' preprocessing in a single call.

        Mutates df_im (stateful).
        """
        if band is not None:
            Tc_low, Tc_high = band
            self.apply_band_pass(Tc_low=Tc_low, Tc_high=Tc_high)

        if detrend:
            self.apply_detrend(type="linear")

        if demean:
            self.apply_detrend(type="demean")

        if taper is not None and taper > 0.0:
            self.apply_taper(max_percentage=taper)

        return self

    # ------------------------- helpers ---------------------------- #

    def _get_work_df(self, df: pd.DataFrame | None) -> pd.DataFrame:
        """
        If df is None -> use current stateful df_im (no copy).
        If df is provided -> use df.copy() (no mutation of caller df).
        """
        if df is None:
            return self.df_im
        return df.copy()

    def _husid(self, acc: np.ndarray, dt: float) -> np.ndarray:
        """
        Normalized cumulative Arias-like Husid curve in [0,1].
        Constants cancel under normalization, so use cumsum(a^2*dt)/final.
        """
        ai = np.cumsum(acc * acc) * dt
        if ai.size == 0 or ai[-1] <= 0.0:
            return np.zeros_like(ai)
        return ai / ai[-1]

    # ------------------------- IMs ---------------------------- #

    def significant_duration(
        self,
        component: ComponentName,
        *,
        p1: float = 0.05,
        p2: float = 0.75,
        df: pd.DataFrame | None = None,
    ) -> float:
        """
        D(p1–p2) for a single component using current df_im, unless df is provided.
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute durations.")

        if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
            raise ValueError("p1 and p2 must be in [0,1].")
        if p2 <= p1:
            raise ValueError("p2 must be > p1.")

        df_work = self._get_work_df(df)

        if "time" not in df_work.columns:
            raise ValueError("DataFrame must contain a 'time' column.")
        if component not in df_work.columns:
            raise ValueError(f"Component '{component}' not found in DataFrame columns.")

        time = df_work["time"].to_numpy(dtype=float)
        acc = df_work[component].to_numpy(dtype=float)

        hus = self._husid(acc, dt=float(rec.dt))
        if hus.size == 0 or np.allclose(hus, 0.0):
            return 0.0

        # np.interp expects x increasing; husid is monotone nondecreasing
        t1 = float(np.interp(p1, hus, time))
        t2 = float(np.interp(p2, hus, time))
        return t2 - t1

    def significant_all(
        self,
        *,
        p1: float = 0.05,
        p2: float = 0.75,
        components: Sequence[ComponentName] | None = None,
        df: pd.DataFrame | None = None,
    ) -> dict[ComponentName, float]:
        """
        D(p1–p2) for multiple components using current df_im, unless df is provided.
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute durations.")

        if components is None:
            comps = list(rec.components)
        else:
            comps = list(components)

        df_work = self._get_work_df(df)
        if "time" not in df_work.columns:
            raise ValueError("DataFrame must contain a 'time' column.")

        time = df_work["time"].to_numpy(dtype=float)

        out: dict[ComponentName, float] = {}
        for comp in comps:
            if comp not in df_work.columns:
                continue
            acc = df_work[comp].to_numpy(dtype=float)
            hus = self._husid(acc, dt=float(rec.dt))
            if hus.size == 0 or np.allclose(hus, 0.0):
                out[comp] = 0.0
                continue
            t1 = float(np.interp(p1, hus, time))
            t2 = float(np.interp(p2, hus, time))
            out[comp] = t2 - t1

        return out

    def husid(
        self,
        component: ComponentName,
        *,
        df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (time, Husid) using current df_im, unless df is provided.
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute Husid.")

        df_work = self._get_work_df(df)

        if "time" not in df_work.columns:
            raise ValueError("DataFrame must contain a 'time' column.")
        if component not in df_work.columns:
            raise ValueError(f"Component '{component}' not found in DataFrame columns.")

        time = df_work["time"].to_numpy(dtype=float)
        acc = df_work[component].to_numpy(dtype=float)
        hus = self._husid(acc, dt=float(rec.dt))
        return time, hus

    # ------------------------- kinematics & peaks ---------------------------- #

    def _component(self, component: ComponentName, df: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot integrate.")
        df_work = self._get_work_df(df)
        if "time" not in df_work.columns:
            raise ValueError("DataFrame must contain a 'time' column.")
        if component not in df_work.columns:
            raise ValueError(f"Component '{component}' not found in DataFrame columns.")
        return df_work["time"].to_numpy(dtype=float), df_work[component].to_numpy(dtype=float)

    def velocity(
        self, component: ComponentName, *, df: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (time, velocity) by integrating the current df_im (or df).

        Exact for acceleration linear between samples, zero initial velocity.
        Units: record acceleration units x s (cm/s^2 -> cm/s).  Integrate a
        baseline-corrected or high-passed record (``apply_baseline_correction``),
        or drift dominates.
        """
        time, acc = self._component(component, df)
        vel, _ = integrate(acc, float(self.record.dt))  # type: ignore[arg-type]
        return time, vel

    def displacement(
        self, component: ComponentName, *, df: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (time, displacement); units: record acceleration units x s^2.
        Same integration and caveats as :meth:`velocity`.
        """
        time, acc = self._component(component, df)
        _, disp = integrate(acc, float(self.record.dt))  # type: ignore[arg-type]
        return time, disp

    def pga(self, component: ComponentName, *, df: pd.DataFrame | None = None) -> float:
        """Peak ground acceleration max|a|, in the record units."""
        _, acc = self._component(component, df)
        return float(np.max(np.abs(acc)))

    def pgv(self, component: ComponentName, *, df: pd.DataFrame | None = None) -> float:
        """Peak ground velocity max|v|, in record units x s."""
        return float(np.max(np.abs(self.velocity(component, df=df)[1])))

    def pgd(self, component: ComponentName, *, df: pd.DataFrame | None = None) -> float:
        """Peak ground displacement max|d|, in record units x s^2."""
        return float(np.max(np.abs(self.displacement(component, df=df)[1])))

    def arias_intensity(
        self,
        component: ComponentName,
        *,
        units: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> float:
        """
        Arias intensity Ia = pi / (2 g) * integral of a(t)^2 dt, in m/s.

        Parameters
        ----------
        units
            Acceleration unit of the data (``"g"``, ``"m/s2"``, ``"cm/s2"``).
            Defaults to ``Record.units``; one of the two is required because
            Ia is not unit-agnostic.
        """
        u = units if units is not None else self.record.units
        if u is None:
            raise ValueError(
                "Arias intensity needs physical units: pass units= or build the "
                "Record with units= (Record.from_files sets it from the file header)."
            )
        _, acc = self._component(component, df)
        a = acc * TO_MPS2[normalize_unit(u)]
        dt = float(self.record.dt)  # type: ignore[arg-type]
        integral = dt * (np.sum(a * a) - 0.5 * (a[0] ** 2 + a[-1] ** 2)) if a.size > 1 else 0.0
        return float(np.pi / (2.0 * G) * integral)

    def zero_crossing_rate(
        self, component: ComponentName, *, df: pd.DataFrame | None = None
    ) -> float:
        """Zero crossings of the acceleration per second, ``nu0``.

        Counts sign changes between consecutive non-zero samples (exact zeros,
        e.g. trailing padding, are skipped rather than counted) over the record
        duration.
        """
        time, acc = self._component(component, df)
        nz = acc[acc != 0.0]
        crossings = int(np.count_nonzero(np.signbit(nz[1:]) != np.signbit(nz[:-1])))
        duration = float(time[-1] - time[0])
        return crossings / duration if duration > 0.0 else 0.0

    def destructiveness_potential(
        self,
        component: ComponentName,
        *,
        units: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> float:
        """Araya & Saragoni (1984) destructiveness potential ``PD = Ia / nu0^2``.

        ``Ia`` in m/s (:meth:`arias_intensity`, so units are required) and
        ``nu0`` in crossings/s (:meth:`zero_crossing_rate`): ``PD`` in m s.
        Multiply by 100 for the cm s of the original paper.  NaN if the record
        never crosses zero.
        """
        ia = self.arias_intensity(component, units=units, df=df)
        nu0 = self.zero_crossing_rate(component, df=df)
        return ia / nu0**2 if nu0 > 0.0 else float("nan")

    def summary(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        units: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        One row per component: PGA, PGV, PGD, D5-75, D5-95 and, when units are
        known (``units=`` or ``Record.units``), Arias intensity ``Ia`` [m/s] and
        destructiveness potential ``PD`` [m s].

        PGA/PGV/PGD are in record units, x s, x s^2.
        """
        rec = self.record
        comps = list(rec.components) if components is None else list(components)
        has_units = units is not None or rec.units is not None
        rows = {}
        for c in comps:
            row = {
                "PGA": self.pga(c, df=df),
                "PGV": self.pgv(c, df=df),
                "PGD": self.pgd(c, df=df),
                "D5-75": self.significant_duration(c, p1=0.05, p2=0.75, df=df),
                "D5-95": self.significant_duration(c, p1=0.05, p2=0.95, df=df),
            }
            if has_units:
                row["Ia"] = self.arias_intensity(c, units=units, df=df)
                row["PD"] = self.destructiveness_potential(c, units=units, df=df)
            rows[c] = row
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "component"
        return out

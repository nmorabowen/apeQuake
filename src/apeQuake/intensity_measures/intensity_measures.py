from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import numpy as np
import pandas as pd

from apeQuake.core.types import ComponentName

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

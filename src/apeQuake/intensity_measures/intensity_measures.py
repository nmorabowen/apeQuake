from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence
import numpy as np
import pandas as pd

from apeQuake.core.types import ComponentName

if TYPE_CHECKING:
    from apeQuake.core.record import Record

FilterFunc = Callable[..., pd.DataFrame]


class IntensityMeasures:
    """
    Composite for ground–motion intensity measures (IMs).

    Attached to a Record as `record.IM`.

    Current features
    ----------------
    - Significant duration based on normalized Arias (Husid curve):
        D(p1–p2) = t(p2) – t(p1), with p1,p2 in [0,1].

    Design
    ------
    - Uses Record.df by default (`time` + component columns).
    - Optional filter pipeline:
        record.IM.add_filter(record.filter.detrend, type="demean")
        record.IM.add_filter(record.filter.taper, max_percentage=0.05)
    - You may also pass a custom df and skip/force pipeline application.
    """

    def __init__(self, record: "Record") -> None:
        self._record = record

        # Filter pipeline: list of (Filter method, kwargs)
        self._filter_steps: list[tuple[FilterFunc, dict]] = []

        # bookkeeping
        self._df_used: pd.DataFrame | None = None

    # ---------------------------------------------------------- #
    # Filter pipeline
    # ---------------------------------------------------------- #

    def add_filter(self, func: FilterFunc, **params) -> "IntensityMeasures":
        """
        Add a filter step to the IM computation pipeline.

        Example
        -------
        rec.IM.add_filter(rec.filter.detrend, type="demean")
        rec.IM.add_filter(rec.filter.taper, max_percentage=0.05)
        """
        self._filter_steps.append((func, params))
        return self

    def clear_filters(self) -> None:
        """Remove all registered filters."""
        self._filter_steps.clear()

    def describe_filters(self) -> str:
        """Human-readable description of current pipeline."""
        if not self._filter_steps:
            return "(no filters)"

        parts: list[str] = []
        for func, params in self._filter_steps:
            name = getattr(func, "__name__", "<fn>")
            args = ", ".join(f"{k}={v!r}" for k, v in params.items())
            parts.append(f"{name}({args})")
        return " → ".join(parts)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered filter steps to df.

        Each func must accept `df=` and return a DataFrame.
        """
        out = df
        for func, params in self._filter_steps:
            out = func(df=out, **params)
        return out

    def _prepare_df(
        self,
        df: pd.DataFrame | None,
        use_filters: bool | None,
    ) -> pd.DataFrame:
        """
        Common logic for selecting base df and deciding on pipeline.

        Convention
        ----------
        - If df is None:
            * base = record.df.copy()
            * if use_filters is None/True  → apply pipeline
            * if use_filters is False      → no pipeline
        - If df is not None:
            * base = df.copy()
            * if use_filters is True       → apply pipeline
            * if use_filters is None/False → no pipeline
        """
        rec = self._record

        if df is None:
            if rec.df is None:
                raise ValueError("Record.df is not set; cannot compute IMs.")
            df_work = rec.df.copy()
            effective_use_filters = True if use_filters is None else use_filters
        else:
            df_work = df.copy()
            effective_use_filters = True if use_filters is True else False

        if effective_use_filters and self._filter_steps:
            df_work = self._apply_filters(df_work)

        self._df_used = df_work
        return df_work

    # ---------------------------------------------------------- #
    # Husid and significant duration
    # ---------------------------------------------------------- #

    def _husid(
        self,
        acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute normalized Arias-intensity Husid curve.

        Shape is what matters for D(p1–p2). Constants (π/2g) cancel
        when normalizing, so we simply do cumsum(a^2 * dt) / final.
        """
        # cumulative Arias-like measure
        ai = np.cumsum(acc**2) * dt
        if ai[-1] <= 0.0:
            # Degenerate case: zero or near-zero motion
            return np.zeros_like(ai)
        return ai / ai[-1]

    def significant_duration(
        self,
        component: ComponentName,
        *,
        p1: float = 0.05,
        p2: float = 0.75,
        df: pd.DataFrame | None = None,
        use_filters: bool | None = None,
    ) -> float:
        """
        Compute significant duration D(p1–p2) for a single component.

        Parameters
        ----------
        component : {'X','Y','Z'}
            Component to use.

        p1, p2 : float
            Lower and upper Husid fractions in [0,1].
            Typical: (0.05, 0.75), (0.05, 0.95).

        df : DataFrame or None
            Time series with columns 'time' and the component.
            If None, uses Record.df.

        use_filters : {True, False, None}
            See _prepare_df() docstring for semantics.

        Returns
        -------
        float
            D(p1–p2) in seconds.
        """
        rec = self._record

        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute durations.")

        df_work = self._prepare_df(df, use_filters)

        if "time" not in df_work.columns:
            raise ValueError("DataFrame must contain a 'time' column.")

        if component not in df_work.columns:
            raise ValueError(f"Component '{component}' not found in DataFrame columns.")

        time = df_work["time"].to_numpy(dtype=float)
        acc = df_work[component].to_numpy(dtype=float)

        husid = self._husid(acc, dt=rec.dt)

        # handle edge case: flat Husid (zero motion)
        if np.allclose(husid, 0.0):
            return 0.0

        t1 = float(np.interp(p1, husid, time))
        t2 = float(np.interp(p2, husid, time))
        return t2 - t1

    def significant_all(
        self,
        *,
        p1: float = 0.05,
        p2: float = 0.75,
        components: Sequence[ComponentName] | None = None,
        df: pd.DataFrame | None = None,
        use_filters: bool | None = None,
    ) -> dict[ComponentName, float]:
        """
        Compute significant duration D(p1–p2) for multiple components.

        Parameters
        ----------
        p1, p2 : float
            Lower and upper Husid fractions.
        components : sequence or None
            Subset of components. If None, use all present in Record.components.
        df, use_filters : see significant_duration().

        Returns
        -------
        dict
            {component_name: D(p1–p2)}
        """
        rec = self._record

        if components is None:
            comps = list(rec.components)
        else:
            comps = list(components)

        # we want to prepare df only once, not per component
        df_work = self._prepare_df(df, use_filters)

        results: dict[ComponentName, float] = {}
        for comp in comps:
            if comp not in df_work.columns:
                continue

            time = df_work["time"].to_numpy(dtype=float)
            acc = df_work[comp].to_numpy(dtype=float)
            husid = self._husid(acc, dt=rec.dt)

            if np.allclose(husid, 0.0):
                results[comp] = 0.0
                continue

            t1 = float(np.interp(p1, husid, time))
            t2 = float(np.interp(p2, husid, time))
            results[comp] = t2 - t1

        return results

    def husid(
        self,
        component: ComponentName,
        *,
        df: pd.DataFrame | None = None,
        use_filters: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (time, Husid) for a given component.

        Husid is the normalized cumulative Arias-like curve in [0,1].
        """
        rec = self._record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute Husid.")

        df_work = self._prepare_df(df, use_filters)

        if "time" not in df_work.columns:
            raise ValueError("DataFrame must contain a 'time' column.")
        if component not in df_work.columns:
            raise ValueError(f"Component '{component}' not found in DataFrame columns.")

        time = df_work["time"].to_numpy(dtype=float)
        acc = df_work[component].to_numpy(dtype=float)
        husid = self._husid(acc, dt=rec.dt)
        return time, husid

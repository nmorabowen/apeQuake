from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..filters.filters import Filter as FilterType
    from ..spectrum.spectrum import Spectrum
    from ..spectrogram.spectrogram import Spectrogram
    from ..response_spectra import ResponseSpectra
    from ..intensity_measures import IntensityMeasures


class Record:
    """
    Multicomponent ground-motion record with 1–3 components,
    stored internally as a pandas DataFrame.

    DataFrame structure:
        time | X | Y | Z
        -----+---+---+---
        t0   | . | . | .
        t1   | . | . | .
        ...

    Only the present components are included (e.g., time|X|Z).
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        *,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        time_array: np.ndarray | None = None,
        dt: float | None = None,
    ) -> None:
        """
        Initialize a Record either from an existing DataFrame or from arrays.

        Parameters
        ----------
        df : DataFrame or None
            If provided, must contain at least a 'time' column and
            one or more of ['X', 'Y', 'Z'].
        x, y, z : np.ndarray or None
            Component arrays. At least one must be provided if df is None.
        time_array : np.ndarray or None
            Time array. If None, it will be generated from dt and length of component.
        dt : float or None
            Time step. Required if time_array is None.
        """
        self._interpolation_flag = False
        self.dt: float | None = dt

        if df is not None:
            self._init_from_df(df)
        else:
            self._init_from_arrays(x=x, y=y, z=z, time_array=time_array, dt=dt)

        # Attach filter toolbox (stateless, uses self.df)
        from ..filters import Filter  # local import to avoid circular
        from ..spectrum import Spectrum
        from ..spectrogram import Spectrogram
        from ..response_spectra import ResponseSpectra
        from ..intensity_measures import IntensityMeasures
        self.filter: FilterType = Filter(self)
        self.spectrum: Spectrum = Spectrum(self)
        self.spectrogram: Spectrogram = Spectrogram(self)
        self.response_spectra: ResponseSpectra = ResponseSpectra(self)
        self.intensity_measures: IntensityMeasures = IntensityMeasures(self)

    # -------------------------------------------------------------- #
    # Initialization helpers
    # -------------------------------------------------------------- #

    def _init_from_df(self, df: pd.DataFrame) -> None:
        # Ensure copy
        df = df.copy()

        if "time" not in df.columns:
            raise ValueError("DataFrame must contain a 'time' column.")

        # Detect which components exist
        comps = [c for c in ("X", "Y", "Z") if c in df.columns]
        if not comps:
            raise ValueError("DataFrame must contain at least one of 'X', 'Y', 'Z'.")

        self.components: Tuple[ComponentName, ...] = tuple(comps)  # type: ignore[assignment]

        time_original = df["time"].to_numpy(dtype=float)
        data_original = df[self.components].to_numpy(dtype=float)

        # Compute dt and check uniformity / interpolation
        self.dt = self._get_dt(time_original)

        if self._interpolation_flag:
            t0, t1 = time_original[0], time_original[-1]
            time_new = np.arange(t0, t1 + 0.5 * self.dt, self.dt)
            f = interp1d(
                time_original,
                data_original,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            data_new = f(time_new)
        else:
            time_new = time_original
            data_new = data_original

        # Build canonical df
        out = {"time": time_new}
        for j, comp in enumerate(self.components):
            out[comp] = data_new[:, j]

        self.df = pd.DataFrame(out)

    def _init_from_arrays(
        self,
        *,
        x: np.ndarray | None,
        y: np.ndarray | None,
        z: np.ndarray | None,
        time_array: np.ndarray | None,
        dt: float | None,
    ) -> None:
        # Collect components
        names: List[ComponentName] = []
        cols: List[np.ndarray] = []

        for name, arr in (("X", x), ("Y", y), ("Z", z)):
            if arr is not None:
                names.append(name)  # type: ignore[arg-type]
                cols.append(np.asarray(arr, dtype=float))

        if not names:
            raise ValueError(
                "At least one component (x, y, or z) must be provided "
                "when df is not given."
            )

        lengths = [len(c) for c in cols]
        if len(set(lengths)) != 1:
            raise ValueError(f"All components must have same length, got {lengths}.")

        npts = lengths[0]

        self.components = tuple(names)
        self.dt = dt
        self._interpolation_flag = False

        if time_array is None:
            if dt is None:
                raise ValueError("If time_array is None, dt must be provided.")
            dt_val = float(dt)
            time_new = np.arange(npts, dtype=float) * dt_val
            data_new = np.column_stack(cols)
            self.dt = dt_val
        else:
            time_original = np.asarray(time_array, dtype=float)
            if len(time_original) != npts:
                raise ValueError(
                    f"time_array length {len(time_original)} "
                    f"!= component length {npts}"
                )

            data_original = np.column_stack(cols)
            self.dt = self._get_dt(time_original)

            if self._interpolation_flag:
                t0, t1 = time_original[0], time_original[-1]
                time_new = np.arange(t0, t1 + 0.5 * self.dt, self.dt)
                f = interp1d(
                    time_original,
                    data_original,
                    axis=0,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                data_new = f(time_new)
            else:
                time_new = time_original
                data_new = data_original

        out = {"time": time_new}
        for j, comp in enumerate(self.components):
            out[comp] = data_new[:, j]
        self.df = pd.DataFrame(out)

    # -------------------------------------------------------------- #
    # Properties: time, data, components, x/y/z
    # -------------------------------------------------------------- #

    @property
    def time(self) -> np.ndarray:
        return self.df["time"].to_numpy(dtype=float)

    @property
    def data(self) -> np.ndarray:
        # (N, n_comp) matrix for components in canonical order
        return self.df[list(self.components)].to_numpy(dtype=float)

    @property
    def x(self) -> np.ndarray:
        if "X" not in self.components:
            raise AttributeError("Record has no X component.")
        return self.df["X"].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        if "Y" not in self.components:
            raise AttributeError("Record has no Y component.")
        return self.df["Y"].to_numpy(dtype=float)

    @property
    def z(self) -> np.ndarray:
        if "Z" not in self.components:
            raise AttributeError("Record has no Z component.")
        return self.df["Z"].to_numpy(dtype=float)

    # -------------------------------------------------------------- #
    # Time helpers
    # -------------------------------------------------------------- #

    def _get_dt(self, time: np.ndarray) -> float:
        dt_array = np.diff(time)
        dt_unique = np.unique(dt_array)

        if self.dt is not None:
            dt_user = float(self.dt)
            if (
                len(dt_unique) > 1
                or not np.isclose(dt_unique[0], dt_user, rtol=1e-6, atol=1e-9)
            ):
                self._interpolation_flag = True
                print(
                    f"dt provided by user: {dt_user}. "
                    "Components will be interpolated to this dt."
                )
                return dt_user
            self._interpolation_flag = False
            return dt_user

        if len(dt_unique) > 1:
            self._interpolation_flag = True
            dt = float(dt_array.min())
            print(f"Setting dt to the lowest value found in time array: {dt}")
            print("Components will be interpolated to this dt.")
        else:
            self._interpolation_flag = False
            dt = float(dt_unique[0])

        return dt

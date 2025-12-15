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
    from ..core.record import Record


class Record:
    """
    Multicomponent ground-motion record with 1–3 components,
    stored internally as a canonical pandas DataFrame.

    Canonical DataFrame:
        time | X | Y | Z
    (Only present components are included.)
    """

    def __init__(
        self,
        *,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        time_array: np.ndarray | None = None,
        dt: float | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize a Record from arrays (df input removed).

        Parameters
        ----------
        x, y, z
            Component arrays (at least one must be provided).
        time_array
            Optional time array (same length as components).
            If provided and not uniform, the record may be interpolated.
        dt
            Desired time step.
            - If time_array is None -> dt is required (time generated as arange * dt).
            - If time_array is provided:
                * if dt is provided -> enforce/validate it; interpolate if needed.
                * if dt is None -> infer dt; interpolate if time_array is not uniform.
        """
        self._interpolation_flag: bool = False
        self.dt: float | None = dt
        self.init_logs: list[str] = []
        self.name: str | None = name

        self._init_from_arrays(x=x, y=y, z=z, time_array=time_array, dt=dt)

        # Attach composites
        from ..filters import Filter
        from ..spectrum import Spectrum
        from ..spectrogram import Spectrogram
        from ..response_spectra import ResponseSpectra
        from ..intensity_measures import IntensityMeasures
        from ..plot_record import PlotRecord

        self.filter: FilterType = Filter(self)
        self.spectrum: Spectrum = Spectrum(self)
        self.spectrogram: Spectrogram = Spectrogram(self)
        self.response_spectra: ResponseSpectra = ResponseSpectra(self)
        self.intensity_measures: IntensityMeasures = IntensityMeasures(self)
        self.plot_record: PlotRecord = PlotRecord(self)

    # -------------------------------------------------------------- #
    # Initialization
    # -------------------------------------------------------------- #

    def _init_from_arrays(
        self,
        *,
        x: np.ndarray | None,
        y: np.ndarray | None,
        z: np.ndarray | None,
        time_array: np.ndarray | None,
        dt: float | None,
    ) -> None:
        # Collect present components
        names: List[ComponentName] = []
        cols: List[np.ndarray] = []

        for name, arr in (("X", x), ("Y", y), ("Z", z)):
            if arr is not None:
                names.append(name)  # type: ignore[arg-type]
                cols.append(np.asarray(arr, dtype=float))

        if not names:
            raise ValueError("At least one component (x, y, or z) must be provided.")

        lengths = [len(c) for c in cols]
        if len(set(lengths)) != 1:
            raise ValueError(f"All components must have same length, got {lengths}.")

        npts = lengths[0]
        self.components = tuple(names)

        # Case A: no time_array -> generate time from dt
        if time_array is None:
            if dt is None:
                raise ValueError("If time_array is None, dt must be provided.")
            dt_val = float(dt)
            if dt_val <= 0.0:
                raise ValueError(f"dt must be > 0, got {dt_val}.")

            time_new = np.arange(npts, dtype=float) * dt_val
            data_new = np.column_stack(cols)

            self.dt = dt_val
            self._interpolation_flag = False
            self.init_logs.append(f"Generated time array with dt={dt_val} (npts={npts}).")

        # Case B: time_array provided -> infer/enforce dt, maybe interpolate
        else:
            time_original = np.asarray(time_array, dtype=float)
            if time_original.ndim != 1:
                raise ValueError("time_array must be 1D.")
            if len(time_original) != npts:
                raise ValueError(
                    f"time_array length {len(time_original)} != component length {npts}."
                )

            # Basic sanity: strictly increasing time
            if not np.all(np.diff(time_original) > 0.0):
                raise ValueError("time_array must be strictly increasing.")

            data_original = np.column_stack(cols)

            # Compute dt and interpolation decision
            dt_eff = self._get_dt(time_original, dt_user=dt)
            self.dt = dt_eff

            if self._interpolation_flag:
                t0, t1 = float(time_original[0]), float(time_original[-1])
                time_new = np.arange(t0, t1 + 0.5 * dt_eff, dt_eff)

                f = interp1d(
                    time_original,
                    data_original,
                    axis=0,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                data_new = f(time_new)

                self.init_logs.append(
                    f"Interpolated components to uniform dt={dt_eff} "
                    f"(npts: {npts} -> {len(time_new)})."
                )
            else:
                time_new = time_original
                data_new = data_original
                self.init_logs.append(f"Using provided time_array with uniform dt={dt_eff}.")

        # Build canonical df
        out: dict[str, np.ndarray] = {"time": time_new}
        for j, comp in enumerate(self.components):
            out[comp] = data_new[:, j]

        self.df = pd.DataFrame(out)

    # -------------------------------------------------------------- #
    # Properties
    # -------------------------------------------------------------- #

    @property
    def time(self) -> np.ndarray:
        return self.df["time"].to_numpy(dtype=float)

    @property
    def data(self) -> np.ndarray:
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

    def _get_dt(self, time: np.ndarray, *, dt_user: float | None) -> float:
        """
        Decide effective dt and whether interpolation is needed.

        Rules (keeps your original behavior):
        - If dt_user provided:
            * If time is not uniform OR differs from dt_user -> interpolate to dt_user.
            * Else use dt_user as-is.
        - If dt_user not provided:
            * If time uniform -> use that dt.
            * If time not uniform -> use min(dt_array) and interpolate.
        """
        dt_array = np.diff(time)
        if dt_array.size == 0:
            raise ValueError("time_array must have at least 2 points to infer dt.")

        dt_unique = np.unique(dt_array)

        if dt_user is not None:
            dt_user = float(dt_user)
            if dt_user <= 0.0:
                raise ValueError(f"dt must be > 0, got {dt_user}.")

            if len(dt_unique) > 1 or not np.isclose(dt_unique[0], dt_user, rtol=1e-6, atol=1e-9):
                self._interpolation_flag = True
                self.init_logs.append(
                    f"User dt={dt_user} differs from/incompatible with time_array; will interpolate."
                )
                return dt_user

            self._interpolation_flag = False
            return dt_user

        # No dt_user: infer
        if len(dt_unique) > 1:
            self._interpolation_flag = True
            dt_eff = float(dt_array.min())
            self.init_logs.append(
                f"time_array is non-uniform; using dt=min(diff)={dt_eff} and interpolating."
            )
            return dt_eff

        self._interpolation_flag = False
        return float(dt_unique[0])

    def __repr__(self) -> str:
        comps = ",".join(self.record.components)
        npts = len(self.df_im)

        if self.applied_filters:
            filt = f"{len(self.applied_filters)} ({self.applied_filters[-1]})"
        else:
            filt = "0"

        return (
            f"<IntensityMeasures "
            f"components=[{comps}] "
            f"npts={npts} "
            f"filters={filt}>"
        )

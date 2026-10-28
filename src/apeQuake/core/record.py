from __future__ import annotations

import os
from pathlib import Path
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
    from ..code_spectrum import CodeSpectrum
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
        units: str | None = None,
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
        units
            Optional acceleration unit of the components: ``"g"``, ``"m/s2"`` or
            ``"cm/s2"`` (aliases ``"m/s^2"``, ``"cm/s^2"``, ``"gal"``).  Only a
            label: the data are never rescaled.  Measures that need physical
            units (Arias intensity) read it.
        """
        from .units import normalize_unit

        self._interpolation_flag: bool = False
        self.dt: float | None = dt
        self.init_logs: list[str] = []
        self.name: str | None = name
        self.units: str | None = None if units is None else normalize_unit(units)
        self.source_files: dict[str, str] = {}

        self._init_from_arrays(x=x, y=y, z=z, time_array=time_array, dt=dt)

        # Attach composites
        from ..filters import Filter
        from ..spectrum import Spectrum
        from ..spectrogram import Spectrogram
        from ..response_spectra import ResponseSpectra
        from ..intensity_measures import IntensityMeasures
        from ..code_spectrum import CodeSpectrum
        from ..plot_record import PlotRecord

        self.filter: FilterType = Filter(self)
        self.spectrum: Spectrum = Spectrum(self)
        self.spectrogram: Spectrogram = Spectrogram(self)
        self.response_spectra: ResponseSpectra = ResponseSpectra(self)
        self.intensity_measures: IntensityMeasures = IntensityMeasures(self)
        self.code_spectrum: CodeSpectrum = CodeSpectrum(self)
        self.plot_record: PlotRecord = PlotRecord(self)

    # -------------------------------------------------------------- #
    # Construction from files
    # -------------------------------------------------------------- #

    @classmethod
    def from_files(
        cls,
        x: str | os.PathLike[str] | None = None,
        y: str | os.PathLike[str] | None = None,
        z: str | os.PathLike[str] | None = None,
        *,
        format: str = "auto",
        dt: float | None = None,
        units: str | None = None,
        name: str | None = None,
    ) -> "Record":
        """
        Build a Record from one file per component (PEER ``.AT2``, RENAC
        ``.txt`` or generic text; see :mod:`apeQuake.readers`).

        Parameters
        ----------
        x, y, z
            Paths of the component files (at least one).  Map the file
            components yourself, e.g. ``x=..._E_100.txt, y=..._N_100.txt,
            z=..._Z_100.txt`` for RENAC.
        format
            ``"auto"`` (default: by extension and header), ``"at2"``,
            ``"renac"`` or ``"txt"``; applies to every file.
        dt
            Time step for ``"txt"`` files whose header gives none.
        units
            Target unit (``"g"``, ``"m/s2"``, ``"cm/s2"``).  Files declaring
            another unit are converted to it; files declaring none are taken
            to be in it already.  If omitted, all files that declare a unit
            must agree, and that unit becomes ``Record.units``.

        Notes
        -----
        All files must share the same time step.  Components of different
        length are padded with trailing zeros to the longest (logged in
        ``init_logs``).
        """
        from ..readers import read_record_file
        from .units import conversion_factor, normalize_unit

        paths = {c: p for c, p in (("X", x), ("Y", y), ("Z", z)) if p is not None}
        if not paths:
            raise ValueError("At least one component file (x, y, or z) must be given.")
        files = {c: read_record_file(p, format=format, dt=dt) for c, p in paths.items()}  # type: ignore[arg-type]

        dts = {c: f.dt for c, f in files.items()}
        dt0 = next(iter(dts.values()))
        if any(not np.isclose(d, dt0, rtol=1e-6) for d in dts.values()):
            raise ValueError(f"Component files have different time steps: {dts}.")

        logs: list[str] = []
        declared = {c: f.units for c, f in files.items()}
        if units is None:
            known = {u for u in declared.values() if u is not None}
            if len(known) > 1:
                raise ValueError(
                    f"Component files declare different units {declared}; pass units= "
                    "to convert them to one."
                )
            if known and None in declared.values():
                raise ValueError(
                    f"Some component files declare no unit {declared}; pass units= "
                    "to state it."
                )
            target = next(iter(known), None)
            arrays = {c: f.data for c, f in files.items()}
        else:
            target = normalize_unit(units)
            arrays = {}
            for c, f in files.items():
                if f.units is None or f.units == target:
                    arrays[c] = f.data
                else:
                    arrays[c] = f.data * conversion_factor(f.units, target)
                    logs.append(f"{c}: converted {f.units} -> {target}.")

        n = max(a.size for a in arrays.values())
        for c, a in arrays.items():
            if a.size < n:
                arrays[c] = np.concatenate([a, np.zeros(n - a.size)])
                logs.append(f"{c}: padded with {n - a.size} trailing zeros to {n} samples.")

        if name is None:
            first = files[next(iter(files))]
            meta = first.metadata
            if "station" in meta:
                name = " ".join(v for v in (meta.get("station"), meta.get("event")) if v)
            else:
                name = meta.get("title") or Path(first.path).stem

        rec = cls(
            x=arrays.get("X"), y=arrays.get("Y"), z=arrays.get("Z"),
            dt=dt0, name=name, units=target,
        )
        rec.source_files = {c: f.path for c, f in files.items()}
        rec.init_logs[:0] = [f"{c}: read {f.format} file {f.path}" for c, f in files.items()] + logs
        return rec

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

        # Representative step and a tolerance-based uniformity test.
        # Exact comparison (np.unique) would flag floating-point round-off in an
        # otherwise-uniform grid (e.g. np.arange) as "non-uniform" and trigger a
        # spurious interpolation, so compare the spread of spacings to the step.
        dt_repr = float(np.median(dt_array))
        spread = float(np.ptp(dt_array))
        is_uniform = spread <= 1e-6 * abs(dt_repr)

        if dt_user is not None:
            dt_user = float(dt_user)
            if dt_user <= 0.0:
                raise ValueError(f"dt must be > 0, got {dt_user}.")

            if (not is_uniform) or not np.isclose(dt_repr, dt_user, rtol=1e-6, atol=1e-9):
                self._interpolation_flag = True
                self.init_logs.append(
                    f"User dt={dt_user} differs from/incompatible with time_array; will interpolate."
                )
                return dt_user

            self._interpolation_flag = False
            return dt_user

        # No dt_user: infer
        if not is_uniform:
            self._interpolation_flag = True
            dt_eff = float(dt_array.min())
            self.init_logs.append(
                f"time_array is non-uniform; using dt=min(diff)={dt_eff} and interpolating."
            )
            return dt_eff

        self._interpolation_flag = False
        return dt_repr

    def __repr__(self) -> str:
        comps = ",".join(self.components)
        npts = len(self.df)
        name = f"'{self.name}'" if self.name is not None else "None"
        interp = "interpolated" if self._interpolation_flag else "as-provided"

        return (
            f"<Record name={name} "
            f"components=[{comps}] "
            f"npts={npts} "
            f"dt={self.dt} "
            f"({interp})>"
        )

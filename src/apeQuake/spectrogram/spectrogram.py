from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Callable
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab

from apeQuake.core.types import ComponentName

if TYPE_CHECKING:
    from apeQuake.core.record import Record

FilterFunc = Callable[..., pd.DataFrame]


class Spectrogram:
    """
    Spectrogram composite:

    - Attached to a Record.
    - Optionally applies a pipeline of Record.filter.* steps before
      computing time–frequency spectrograms.
    - Retains your original style:
        * contourf
        * cmap='seismic' by default
        * shared axes, fmin cutoff
        * right-hand secondary axis in period
        * colorbar on the right
    """

    def __init__(self, record: "Record") -> None:
        self._record = record

        # List of (filter_function, params_dict) – applied in order
        self._filter_steps: list[tuple[FilterFunc, dict]] = []

        # Stored spectrogram results
        self.freqs: np.ndarray | None = None     # (nfreq,)
        self.bins: np.ndarray | None = None      # (nbins,)
        # component -> Pxx_dB array of shape (nfreq, nbins)
        self.power: dict[ComponentName, np.ndarray] = {}
        self._df_used: pd.DataFrame | None = None

    # ---------------------------------------------------------- #
    # Filter pipeline management
    # ---------------------------------------------------------- #

    def add_filter(self, func: FilterFunc, **params) -> "Spectrogram":
        """
        Add a filter step to the spectrogram computation pipeline.

        Example
        -------
        rec.spectrogram.add_filter(rec.filter.detrend, type="demean")
        rec.spectrogram.add_filter(rec.filter.taper, max_percentage=0.05)
        """
        self._filter_steps.append((func, params))
        return self

    def clear_filters(self) -> None:
        """Remove all registered filters for the spectrogram pipeline."""
        self._filter_steps.clear()

    def describe_filters(self) -> str:
        """Human-readable description of the current pipeline."""
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

        Each func must be a Filter method with signature:
            func(..., df=<DataFrame>) -> DataFrame
        """
        out = df
        for func, params in self._filter_steps:
            out = func(df=out, **params)
        return out

    # ---------------------------------------------------------- #
    # Core spectrogram computation
    # ---------------------------------------------------------- #

    def compute(
        self,
        df: pd.DataFrame | None = None,
        *,
        use_filters: bool = True,
        components: Sequence[ComponentName] | None = None,
        NFFT: int = 256,
        noverlap: int = 128,
        detrend: bool = True,
        taper: float = 0.05,
        band: tuple[float, float] | None = None,   # (Tc_low, Tc_high)
    ) -> None:
        """
        Compute spectrograms (power in dB) for the selected components.

        Parameters
        ----------
        df : DataFrame or None
            Base time series with 'time' and component columns.
            If None, uses record.df.

        use_filters : bool
            If True, applies any filters added via add_filter() before
            computing the spectrogram.

        components : sequence or None
            Which components to include. If None, use all components
            present in both Record.components and df columns.

        NFFT : int
            FFT window size.

        noverlap : int
            Overlap between windows.

        detrend : bool
            If True, applies simple linear detrend + demean before mlab.specgram.

        taper : float
            Fraction of record (0–1) to taper at each end with a cosine taper.

        band : (Tc_low, Tc_high) or None
            Optional additional band-pass filter in period [s],
            applied via `record.filter.band_pass(...)` on top of
            any pipeline filters.

        Side effects
        ------------
        - Sets self.freqs (Hz), self.bins (time), and self.power[comp]
          as dB matrices of shape (nfreq, nbins).
        """
        from scipy.signal import detrend as sp_detrend

        rec = self._record

        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute spectrogram.")

        # Base df
        if df is None:
            df_work = rec.df.copy()
        else:
            df_work = df.copy()

        # Apply the generic filter pipeline
        if use_filters and self._filter_steps:
            df_work = self._apply_filters(df_work)

        # Optional (extra) band-pass here, in period space
        if band is not None:
            Tc_low, Tc_high = band
            if hasattr(rec, "filter"):
                df_work = rec.filter.band_pass(
                    Tc_low=Tc_low,
                    Tc_high=Tc_high,
                    df=df_work,
                )

        # Which components
        if components is None:
            comps: list[ComponentName] = [
                c for c in ("X", "Y", "Z")
                if c in rec.components and c in df_work.columns
            ]  # type: ignore[list-item]
        else:
            comps = [c for c in components if c in df_work.columns]

        if not comps:
            raise ValueError(
                "No components to compute spectrogram for. "
                "Check df and 'components' argument."
            )

        dt = rec.dt
        fs = 1.0 / dt

        freqs_global: np.ndarray | None = None
        bins_global: np.ndarray | None = None
        power: dict[ComponentName, np.ndarray] = {}

        for comp in comps:
            data = df_work[comp].to_numpy(dtype=float)

            # Detrend (demean + linear)
            if detrend:
                data = sp_detrend(data, type="linear")
                data = data - data.mean()

            # Taper at both ends with a cosine taper
            if taper and taper > 0.0:
                n = len(data)
                ntap = int(n * taper)
                if ntap > 0:
                    window = np.ones(n, dtype=float)
                    # cosine ramp
                    x = np.linspace(0.0, np.pi, ntap)
                    ramp = 0.5 * (1.0 - np.cos(x))
                    window[:ntap] = ramp
                    window[-ntap:] = ramp[::-1]
                    data = data * window

            # Spectrogram via mlab
            Pxx, freqs, bins = mlab.specgram(
                data,
                NFFT=NFFT,
                Fs=fs,
                noverlap=noverlap,
            )

            Pxx_dB = 10.0 * np.log10(Pxx + 1e-12)

            if freqs_global is None:
                freqs_global = freqs
                bins_global = bins
            else:
                # Sanity check: same freq/time axes
                if not np.allclose(freqs_global, freqs) or not np.allclose(
                    bins_global, bins
                ):
                    raise ValueError("Frequency or time axis mismatch between components.")

            power[comp] = Pxx_dB

        self.freqs = freqs_global
        self.bins = bins_global
        self.power = power
        self._df_used = df_work

    # ---------------------------------------------------------- #
    # Plotting
    # ---------------------------------------------------------- #

    def _ensure_computed(self) -> None:
        if self.freqs is None or self.bins is None or not self.power:
            self.compute()

    def plot(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        fmin: float = 0.1,
        cmap: str = "seismic",
        vmin: float | None = None,
        vmax: float | None = None,
        fig=None,
        axes=None,
    ):
        """
        Plot contour spectrograms for the selected components.

        Parameters
        ----------
        components : sequence or None
            Components to plot. If None, uses whatever is in self.power.
        fmin : float
            Minimum frequency in Hz (to avoid infinite period on secondary axis).
        cmap : str
            Matplotlib colormap (default 'seismic').
        vmin, vmax : float or None
            Optional limits for the dB colormap.
        fig, axes : matplotlib Figure/Axes or None
            Optionally reuse an existing figure/axes.

        Returns
        -------
        fig, axes
        """
        import matplotlib.pyplot as plt

        self._ensure_computed()

        assert self.freqs is not None
        assert self.bins is not None
        assert self.power

        freqs = self.freqs
        bins = self.bins

        # Components to plot
        if components is None:
            comps_plot: list[ComponentName] = list(self.power.keys())
        else:
            comps_plot = [c for c in components if c in self.power]

        if not comps_plot:
            raise ValueError("No components available to plot.")

        nrows = len(comps_plot)

        if fig is None or axes is None:
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=1,
                figsize=(12, 3.5 * nrows),
                sharex=True,
                sharey=True,
            )

        if nrows == 1:
            axes_list = [axes]
        else:
            axes_list = list(axes)

        # Helper transforms for secondary y-axis
        def freq2period(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / x

        def period2freq(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / x

        last_cs = None

        for comp, ax in zip(comps_plot, axes_list):
            Pxx_dB = self.power[comp]

            cs = ax.contourf(
                bins,
                freqs,
                Pxx_dB,
                levels=50,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend="both",
            )
            last_cs = cs

            ax.set_ylim(bottom=fmin)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"Component {comp}")
            ax.grid(True, alpha=0.3, ls="--")

            # Secondary y-axis in period
            secax = ax.secondary_yaxis("right", functions=(freq2period, period2freq))
            secax.set_ylabel("Period (s)")

        axes_list[-1].set_xlabel("Time (s)")

        # Colorbar on the right, as in your original version
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(last_cs, cax=cbar_ax, label="Power (dB)")

        plt.subplots_adjust(right=0.9)
        return fig, axes_list

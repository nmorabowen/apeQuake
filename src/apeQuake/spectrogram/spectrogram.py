from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record
    from ..filters.filters import Filter


class Spectrogram:
    def __init__(self, record: "Record") -> None:
        self.record = record
        self.filter: "Filter" = record.filter

        self.df_spectrogram: pd.DataFrame = record.df.copy()

        self.freqs: np.ndarray | None = None
        self.bins: np.ndarray | None = None
        self.power: dict[ComponentName, np.ndarray] = {}

        self.applied_filters: list[str] = []

    # ------------------------- DF lifecycle ------------------------- #

    def set_df(self, df: pd.DataFrame) -> "Spectrogram":
        self.df_spectrogram = df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def reset_df(self) -> "Spectrogram":
        self.df_spectrogram = self.record.df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def _invalidate(self) -> None:
        self.freqs = None
        self.bins = None
        self.power = {}

    def _log(self, s: str) -> None:
        self.applied_filters.append(s)

    # ------------------------- wrappers ---------------------------- #

    def apply_low_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrogram":
        self.df_spectrogram = self.filter.low_pass(df=self.df_spectrogram, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"low_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_high_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrogram":
        self.df_spectrogram = self.filter.high_pass(df=self.df_spectrogram, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"high_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_pass(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrogram":
        self.df_spectrogram = self.filter.band_pass(
            df=self.df_spectrogram, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_pass(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_stop(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrogram":
        self.df_spectrogram = self.filter.band_stop(
            df=self.df_spectrogram, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_stop(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_detrend(self, type: str = "demean") -> "Spectrogram":
        self.df_spectrogram = self.filter.detrend(df=self.df_spectrogram, type=type)
        self._invalidate()
        self._log(f"detrend(type='{type}')")
        return self

    def apply_taper(self, max_percentage: float = 0.05, *, type: str = "cosine") -> "Spectrogram":
        self.df_spectrogram = self.filter.taper(df=self.df_spectrogram, max_percentage=max_percentage, type=type)
        self._invalidate()
        self._log(f"taper(max_percentage={max_percentage}, type='{type}')")
        return self

    # --------------------- common/base preprocessing --------------------- #

    def apply_base_filters(
        self,
        *,
        detrend: bool = True,
        demean: bool = True,
        taper: float | None = 0.05,
        band: tuple[float, float] | None = None,
    ) -> "Spectrogram":
        """
        Apply the 'standard' preprocessing in a single call.

        This MUTATES df_spectrogram (stateful) by calling the wrapper methods.
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

    # ------------------------- computation ---------------------------- #

    def compute(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        NFFT: int = 256,
        noverlap: int = 128,
    ) -> None:
        """
        Compute spectrograms (power in dB) for the selected components.

        IMPORTANT: This function does NOT apply filters. It uses the current
        df_spectrogram as-is.
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute spectrogram.")

        df_work = self.df_spectrogram

        if components is None:
            comps: list[ComponentName] = [
                c for c in ("X", "Y", "Z")
                if c in rec.components and c in df_work.columns
            ]  # type: ignore[list-item]
        else:
            comps = [c for c in components if c in df_work.columns]

        if not comps:
            raise ValueError("No components to compute spectrogram for. Check df and 'components' argument.")

        fs = 1.0 / rec.dt

        freqs_global: np.ndarray | None = None
        bins_global: np.ndarray | None = None
        power: dict[ComponentName, np.ndarray] = {}

        for comp in comps:
            data = df_work[comp].to_numpy(dtype=float)

            Pxx, freqs, bins_ = mlab.specgram(
                data,
                NFFT=NFFT,
                Fs=fs,
                noverlap=noverlap,
            )
            Pxx_dB = 10.0 * np.log10(Pxx + 1e-12)

            if freqs_global is None:
                freqs_global = freqs
                bins_global = bins_
            else:
                if not np.allclose(freqs_global, freqs) or not np.allclose(bins_global, bins_):
                    raise ValueError("Frequency or time axis mismatch between components.")

            power[comp] = Pxx_dB

        self.freqs = freqs_global
        self.bins = bins_global
        self.power = power

    def plot_spectrogram(
        self,
        *,
        NFFT: int = 256,
        noverlap: int = 128,
        cmap: str = "seismic",
        fmin: float = 0.1,
        detrend: bool = True,
        taper: float = 0.05,
        band: tuple[float, float] | None = None,  # (Tc_low, Tc_high) in seconds
        corners: int = 4,
        zerophase: bool = True,
        show: bool = True,
    ):
        """
        Plot contour spectrograms for available components (X,Y,Z) with a colorbar
        and a secondary y-axis showing period.

        This method does NOT mutate df_spectrogram; it works on a copied ObsPy Stream.

        Parameters
        ----------
        NFFT, noverlap, cmap, fmin
            Same meaning as your draft.
        detrend
            If True: apply demean + linear detrend (plot-time only).
        taper
            Fractional taper (0..1) for ObsPy taper().
        band
            Optional band-pass in period (Tc_low, Tc_high) [s].
            Converted to freqmin=1/Tc_high and freqmax=1/Tc_low.
        corners, zerophase
            Passed to ObsPy filter() when band is not None.
        show
            If True calls plt.show().
        """
        try:
            import obspy
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError(
                "plot_spectrogram() requires obspy and matplotlib. Install with: pip install obspy matplotlib"
            ) from e

        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot plot spectrogram.")

        df = self.df_spectrogram
        comps = [c for c in ("X", "Y", "Z") if c in df.columns]
        if not comps:
            raise ValueError("No components (X,Y,Z) found in df_spectrogram.")

        # Step 1 — construct ObsPy stream from current working df (copy)
        st = obspy.Stream()
        for name in comps:
            data = df[name].to_numpy(dtype=float).copy()
            tr = obspy.Trace(data=data)
            tr.stats.delta = rec.dt
            tr.stats.channel = name
            st.append(tr)

        # Step 2 — preprocessing (plot-time only; does NOT affect df_spectrogram)
        st_clean = st.copy()

        if detrend:
            st_clean.detrend("demean")
            st_clean.detrend("linear")

        if taper and taper > 0.0:
            st_clean.taper(taper)

        # Optional band-pass (period -> frequency)
        if band is not None:
            Tc_low, Tc_high = band
            if Tc_low <= 0 or Tc_high <= 0:
                raise ValueError("band periods must be > 0.")
            if Tc_low >= Tc_high:
                raise ValueError("band must satisfy Tc_low < Tc_high.")

            f1 = 1.0 / Tc_high  # freqmin
            f2 = 1.0 / Tc_low   # freqmax
            st_clean.filter(
                "bandpass",
                freqmin=f1,
                freqmax=f2,
                corners=corners,
                zerophase=zerophase,
            )

        # Step 3 — setup figure
        fig, axes = plt.subplots(
            nrows=len(st_clean),
            figsize=(12, 3.5 * len(st_clean)),
            sharex=True,
            sharey=True,
        )
        if len(st_clean) == 1:
            axes = [axes]

        # Helper functions for secondary axis
        def freq2period(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / x

        def period2freq(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / x

        last_cs = None

        for ax, tr in zip(axes, st_clean):
            fs = tr.stats.sampling_rate

            # Compute spectrogram numerically
            Pxx, freqs, bins = mlab.specgram(
                tr.data,
                NFFT=NFFT,
                Fs=fs,
                noverlap=noverlap,
            )

            # Convert to dB
            Pxx_dB = 10.0 * np.log10(Pxx + 1e-12)

            # Plot contour spectrogram
            cs = ax.contourf(
                bins,
                freqs,
                Pxx_dB,
                levels=50,
                cmap=cmap,
                extend="both",
            )
            last_cs = cs

            ax.set_ylim(bottom=fmin)
            ax.set_title(f"Component {tr.stats.channel}")
            ax.set_ylabel("Frequency (Hz)")
            ax.grid(True, alpha=0.3, ls="--")

            secax = ax.secondary_yaxis("right", functions=(freq2period, period2freq))
            secax.set_ylabel("Period (s)")

        axes[-1].set_xlabel("Time (s)")

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(last_cs, cax=cbar_ax, label="Power (dB)")

        plt.subplots_adjust(right=0.9)

        if show:
            plt.show()

        return fig, axes

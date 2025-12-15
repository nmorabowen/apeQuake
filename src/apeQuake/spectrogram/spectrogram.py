from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record
    from ..filters.filters import Filter

@dataclass(slots=True)
class SpectrogramPlotLimits:
    # x-axis (time in seconds)
    tmin: float | None = None
    tmax: float | None = None

    # y-axis (frequency in Hz)
    fmin: float | None = None
    fmax: float | None = None

    # color axis (power in dB)
    pmin: float | None = None
    pmax: float | None = None
    

class Spectrogram:
    def __init__(self, record: "Record", name:str | None = None) -> None:
        self.record = record
        self.filter: "Filter" = record.filter
        self.name: str | None = name

        self.df_spectrogram: pd.DataFrame = record.df.copy()

        self.freqs: np.ndarray | None = None
        self.bins: np.ndarray | None = None
        self.power: dict[ComponentName, np.ndarray] = {}

        self.applied_filters: list[str] = []

        # --- NEW: plot limits as instance attribute
        self.plot_limits: SpectrogramPlotLimits = SpectrogramPlotLimits()

    # ------------------------- plot config ------------------------- #

    def set_plot_limits(
        self,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        pmin: float | None = None,
        pmax: float | None = None,
    ) -> "Spectrogram":
        if tmin is not None and tmax is not None and tmin >= tmax:
            raise ValueError("Require tmin < tmax.")
        if fmin is not None and fmax is not None and fmin >= fmax:
            raise ValueError("Require fmin < fmax.")
        if pmin is not None and pmax is not None and pmin >= pmax:
            raise ValueError("Require pmin < pmax.")

        if tmin is not None:
            self.plot_limits.tmin = float(tmin)
        if tmax is not None:
            self.plot_limits.tmax = float(tmax)
        if fmin is not None:
            self.plot_limits.fmin = float(fmin)
        if fmax is not None:
            self.plot_limits.fmax = float(fmax)
        if pmin is not None:
            self.plot_limits.pmin = float(pmin)
        if pmax is not None:
            self.plot_limits.pmax = float(pmax)

        return self

    def reset_plot_limits(self) -> "Spectrogram":
        self.plot_limits = SpectrogramPlotLimits()
        return self

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
        # fallback default if self.plot_limits.fmin is None
        fmin: float = 0.1,
        detrend: bool = True,
        taper: float = 0.05,
        band: tuple[float, float] | None = None,
        corners: int = 4,
        zerophase: bool = True,
        show: bool = True,
    ):
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

        # Step 2 — preprocessing (plot-time only)
        st_clean = st.copy()
        if detrend:
            st_clean.detrend("demean")
            st_clean.detrend("linear")
        if taper and taper > 0.0:
            st_clean.taper(taper)

        if band is not None:
            Tc_low, Tc_high = band
            if Tc_low <= 0 or Tc_high <= 0:
                raise ValueError("band periods must be > 0.")
            if Tc_low >= Tc_high:
                raise ValueError("band must satisfy Tc_low < Tc_high.")
            f1 = 1.0 / Tc_high
            f2 = 1.0 / Tc_low
            st_clean.filter("bandpass", freqmin=f1, freqmax=f2, corners=corners, zerophase=zerophase)

        # Limits (instance attributes take priority; fall back to fmin argument)
        lim = self.plot_limits
        tmin = lim.tmin
        tmax_eff = lim.tmax  # <-- do NOT mutate lim.tmax
        fmin_eff = lim.fmin if lim.fmin is not None else fmin
        fmax_eff = lim.fmax
        pmin = lim.pmin
        pmax = lim.pmax

        if fmin_eff is not None and fmin_eff <= 0.0:
            raise ValueError("fmin must be > 0 to use the period secondary axis.")

        # Step 3 — setup figure
        fig, axes = plt.subplots(
            nrows=len(st_clean),
            figsize=(12, 3.5 * len(st_clean)),
            sharex=True,
            sharey=True,
        )
        if len(st_clean) == 1:
            axes = [axes]

        # Figure-level title (metadata)
        if self.name:
            fig.suptitle(self.name, fontsize=14, y=0.98)

        def freq2period(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / x

        def period2freq(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 1.0 / x

        last_cs = None

        for ax, tr in zip(axes, st_clean):
            fs = tr.stats.sampling_rate

            Pxx, freqs, bins = mlab.specgram(tr.data, NFFT=NFFT, Fs=fs, noverlap=noverlap)
            Pxx_dB = 10.0 * np.log10(Pxx + 1e-12)

            # Clip requested tmax to available bins (do it on the local variable, not the attribute)
            if tmax_eff is not None:
                t_available_max = float(bins.max()) if len(bins) else 0.0
                if tmax_eff > t_available_max:
                    tmax_eff = t_available_max

            # Power levels (fixed scale if pmin/pmax provided)
            if (pmin is not None) and (pmax is not None):
                levels = np.linspace(pmin, pmax, 51)  # 50 bands
            else:
                levels = 50

            cs = ax.contourf(
                bins,
                freqs,
                Pxx_dB,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            last_cs = cs

            # Frequency limits
            if fmin_eff is not None:
                ax.set_ylim(bottom=fmin_eff)
            if fmax_eff is not None:
                ax.set_ylim(top=fmax_eff)

            # Time limits
            if tmin is not None or tmax_eff is not None:
                ax.set_xlim(left=tmin, right=tmax_eff)

            ax.set_title(f"Component {tr.stats.channel}")
            ax.set_ylabel("Frequency (Hz)")
            ax.grid(True, alpha=0.3, ls="--")

            secax = ax.secondary_yaxis("right", functions=(freq2period, period2freq))
            secax.set_ylabel("Period (s)")

        axes[-1].set_xlabel("Time (s)")

        # Add colorbar
        if last_cs is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(last_cs, cax=cbar_ax, label="Power (dB)")

        plt.subplots_adjust(right=0.9, top=0.92)

        if show:
            plt.show()

        return fig, axes
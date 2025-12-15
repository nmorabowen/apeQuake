from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Literal
import numpy as np
import pandas as pd

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record
    from ..filters.filters import Filter  # adjust import to your project layout


SpectrumRepresentation = Literal["linear", "semilogx", "semilogy", "loglog"]


class Spectrum:
    def __init__(self, record: "Record") -> None:
        self.record = record
        self.filter: "Filter" = record.filter

        # Working df (stateful like Spectrogram)
        self.df_spectrum: pd.DataFrame = record.df.copy()

        # Outputs (invalidated when df changes)
        self.freqs: np.ndarray | None = None
        self.amplitudes: dict[ComponentName, np.ndarray] = {}

        # Logging (like Spectrogram)
        self.applied_filters: list[str] = []

    # ------------------------- DF lifecycle ------------------------- #

    def set_df(self, df: pd.DataFrame) -> "Spectrum":
        self.df_spectrum = df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def reset_df(self) -> "Spectrum":
        self.df_spectrum = self.record.df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def _invalidate(self) -> None:
        self.freqs = None
        self.amplitudes = {}

    def _log(self, s: str) -> None:
        self.applied_filters.append(s)

    # ------------------------- wrappers ---------------------------- #

    def apply_low_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrum":
        self.df_spectrum = self.filter.low_pass(df=self.df_spectrum, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"low_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_high_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrum":
        self.df_spectrum = self.filter.high_pass(df=self.df_spectrum, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"high_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_pass(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrum":
        self.df_spectrum = self.filter.band_pass(
            df=self.df_spectrum, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_pass(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_stop(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "Spectrum":
        self.df_spectrum = self.filter.band_stop(
            df=self.df_spectrum, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_stop(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_detrend(self, type: str = "demean") -> "Spectrum":
        self.df_spectrum = self.filter.detrend(df=self.df_spectrum, type=type)
        self._invalidate()
        self._log(f"detrend(type='{type}')")
        return self

    def apply_taper(self, max_percentage: float = 0.05, *, type: str = "cosine") -> "Spectrum":
        self.df_spectrum = self.filter.taper(df=self.df_spectrum, max_percentage=max_percentage, type=type)
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
    ) -> "Spectrum":
        """
        Apply the 'standard' preprocessing in a single call.

        This MUTATES df_spectrum (stateful) by calling the wrapper methods.
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
        remove_mean: bool = True,
        scale: Literal["none", "amplitude"] = "amplitude",
    ) -> None:
        """
        Compute FFT amplitude spectra for the selected components.

        IMPORTANT: This function does NOT apply filters.
        It uses the current df_spectrum as-is.
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute spectrum.")

        df_work = self.df_spectrum

        if "time" not in df_work.columns:
            raise ValueError("df_spectrum must contain a 'time' column.")

        if components is None:
            comps: list[ComponentName] = [
                c for c in ("X", "Y", "Z")
                if c in rec.components and c in df_work.columns
            ]  # type: ignore[list-item]
        else:
            comps = [c for c in components if c in df_work.columns]

        if not comps:
            raise ValueError("No components to compute spectrum for. Check df and 'components' argument.")

        dt = float(rec.dt)
        n = len(df_work["time"])

        freqs = np.fft.rfftfreq(n, dt)
        amps: dict[ComponentName, np.ndarray] = {}

        for comp in comps:
            x = df_work[comp].to_numpy(dtype=float)
            if remove_mean:
                x = x - np.mean(x)

            X = np.fft.rfft(x)

            A = np.abs(X)
            if scale == "amplitude":
                # Common “single-sided amplitude spectrum” scaling convention
                # (keeps DC/Nyquist as-is; doubles interior bins)
                A = A / n
                if len(A) > 2:
                    A[1:-1] *= 2.0

            amps[comp] = A

        self.freqs = freqs
        self.amplitudes = amps

    # ------------------------- plotting ---------------------------- #

    def _ensure_computed(self) -> None:
        if self.freqs is None or not self.amplitudes:
            self.compute()

    def plot(
        self,
        *,
        representation: SpectrumRepresentation = "loglog",
        combined: bool = False,
        components: Sequence[ComponentName] | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        show: bool = True,
        fig=None,
        axes=None,
        **plot_kwargs,
    ):
        import matplotlib.pyplot as plt

        self._ensure_computed()

        freqs = self.freqs
        amps = self.amplitudes

        if freqs is None:
            raise RuntimeError("Spectrum.freqs is None after compute().")

        # components to plot
        if components is None:
            comps = list(amps.keys())
        else:
            comps = [c for c in components if c in amps]

        if not comps:
            raise ValueError("No available components to plot.")

        # frequency window mask
        mask = np.ones_like(freqs, dtype=bool)
        if fmin is not None:
            mask &= freqs >= fmin
        if fmax is not None:
            mask &= freqs <= fmax
        # avoid plotting DC by default in log plots
        if representation in ("semilogx", "loglog"):
            mask &= freqs > 0.0

        def plot_fn(ax):
            match representation:
                case "linear": return ax.plot
                case "semilogx": return ax.semilogx
                case "semilogy": return ax.semilogy
                case "loglog": return ax.loglog
                case _: raise ValueError("Bad representation")

        if combined:
            if fig is None or axes is None:
                fig, ax = plt.subplots(figsize=(7, 5))
            else:
                ax = axes

            P = plot_fn(ax)
            for comp in comps:
                P(freqs[mask], amps[comp][mask], label=comp, **plot_kwargs)

            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Spectrum" + ("" if not self.applied_filters else f" | {self.applied_filters[-1]}"))
            fig.tight_layout()
            if show:
                plt.show()
            return fig, ax

        # Separate subplots
        if fig is None or axes is None:
            fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 4), sharey=True)

        if len(comps) == 1:
            axes = [axes]

        for comp, ax in zip(comps, axes):
            P = plot_fn(ax)
            P(freqs[mask], amps[comp][mask], **plot_kwargs)
            ax.set_title(comp)
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(True, which="both", ls="-", alpha=0.5)

        axes[0].set_ylabel("Amplitude")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_combined(self, **kw):
        return self.plot(combined=True, **kw)

    def plot_separate(self, **kw):
        return self.plot(combined=False, **kw)

    def get_arrays(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        representation: SpectrumRepresentation = "loglog",
    ) -> tuple[np.ndarray, dict[ComponentName, np.ndarray], list[ComponentName]]:
        """Return (freqs, amps_dict, comps) after compute(), with masking applied."""
        self._ensure_computed()

        if self.freqs is None:
            raise RuntimeError("Spectrum.freqs is None after compute().")

        freqs = self.freqs
        amps = self.amplitudes

        if components is None:
            comps = list(amps.keys())
        else:
            comps = [c for c in components if c in amps]

        if not comps:
            raise ValueError("No available components to extract.")

        mask = np.ones_like(freqs, dtype=bool)
        if fmin is not None:
            mask &= freqs >= fmin
        if fmax is not None:
            mask &= freqs <= fmax
        if representation in ("semilogx", "loglog"):
            mask &= freqs > 0.0

        freqs_m = freqs[mask]
        amps_m: dict[ComponentName, np.ndarray] = {c: amps[c][mask] for c in comps}
        return freqs_m, amps_m, comps

    def compare(
        self,
        *,
        other: Sequence["Spectrum"] | None = None,
        other_records: Sequence["Record"] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        components: Sequence[ComponentName] | None = None,
        representation: SpectrumRepresentation = "loglog",
        combined: bool = True,              # True = one plot (all comps) or one plot per comp? (see below)
        per_component_axes: bool = False,    # if True -> one subplot per component, overlay records in each
        fmin: float | None = None,
        fmax: float | None = None,
        grid: bool = True,
        show: bool = True,
        figsize: tuple[float, float] | None = None,
        align: Literal["strict", "interp"] = "strict",
        **plot_kwargs,
    ):
        """
        Compare spectra by overlaying multiple Spectrum objects (or Records).

        align:
          - "strict": requires identical frequency grids (same dt & n after masking)
          - "interp": interpolates amplitudes onto the base (self) frequency grid
        """
        import matplotlib.pyplot as plt

        # --- build list of Spectrum objects ---
        specs: list[Spectrum] = [self]

        if other is not None:
            specs.extend(list(other))

        if other_records is not None:
            specs.extend([Spectrum(r) for r in other_records])

        # --- labels / colors ---
        if labels is None:
            labels = [getattr(s.record, "name", f"rec{i}") for i, s in enumerate(specs)]
        if len(labels) != len(specs):
            raise ValueError("labels must match number of spectra.")

        if colors is not None and len(colors) != len(specs):
            raise ValueError("colors must match number of spectra.")

        # --- base arrays (self) ---
        f_base, A_base, comps = self.get_arrays(
            components=components,
            fmin=fmin,
            fmax=fmax,
            representation=representation,
        )

        # --- gather & (optionally) align others ---
        all_freqs: list[np.ndarray] = [f_base]
        all_amps: list[dict[ComponentName, np.ndarray]] = [A_base]

        for s in specs[1:]:
            f_i, A_i, comps_i = s.get_arrays(
                components=comps,          # force same component set for plotting
                fmin=fmin,
                fmax=fmax,
                representation=representation,
            )

            if align == "strict":
                if f_i.shape != f_base.shape or not np.allclose(f_i, f_base, rtol=0, atol=0):
                    raise ValueError(
                        "Frequency grids differ. Use align='interp' or ensure same dt and n."
                    )
                all_freqs.append(f_i)
                all_amps.append(A_i)

            elif align == "interp":
                # interpolate each component amplitude onto base grid
                A_i_interp: dict[ComponentName, np.ndarray] = {}
                for c in comps:
                    # safe: f_i is increasing, use np.interp
                    A_i_interp[c] = np.interp(f_base, f_i, A_i[c], left=np.nan, right=np.nan)
                all_freqs.append(f_base)
                all_amps.append(A_i_interp)
            else:
                raise ValueError("align must be 'strict' or 'interp'.")

        # --- plotting function ---
        def plot_fn(ax):
            match representation:
                case "linear": return ax.plot
                case "semilogx": return ax.semilogx
                case "semilogy": return ax.semilogy
                case "loglog": return ax.loglog
                case _: raise ValueError("Bad representation")

        # =======================
        # layout choices
        # =======================

        if per_component_axes:
            # one subplot per component, overlay records in each
            n = len(comps)
            if figsize is None:
                figsize = (5.5 * n, 4.2)

            fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
            if n == 1:
                axes = [axes]

            for ax, c in zip(axes, comps):
                P = plot_fn(ax)
                for k, (lab, A_k) in enumerate(zip(labels, all_amps)):
                    kw = dict(plot_kwargs)
                    if colors is not None:
                        kw["color"] = colors[k]
                    P(f_base, A_k[c], label=lab, **kw)

                ax.set_title(c)
                ax.set_xlabel("Frequency (Hz)")
                if grid:
                    ax.grid(True, which="both", alpha=0.5)

            axes[0].set_ylabel("Amplitude")
            axes[-1].legend()
            fig.tight_layout()
            if show:
                plt.show()
            return fig, axes

        # else: combined axis
        if figsize is None:
            figsize = (7.2, 5.2)
        fig, ax = plt.subplots(figsize=figsize)

        P = plot_fn(ax)

        # combined=True here means "plot all components on same axes, but keep label readable"
        # label format: "<record>.<component>"
        for k, (lab, A_k) in enumerate(zip(labels, all_amps)):
            for c in comps:
                kw = dict(plot_kwargs)
                if colors is not None:
                    kw["color"] = colors[k]
                P(f_base, A_k[c], label=f"{lab}.{c}", **kw)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        if grid:
            ax.grid(True, which="both", alpha=0.5)
        ax.legend(ncol=2)
        ax.set_title("Spectrum comparison")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

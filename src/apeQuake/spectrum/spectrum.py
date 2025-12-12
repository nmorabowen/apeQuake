from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Callable, Literal
import numpy as np
import pandas as pd

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record
    from ..filters import Filter

SpectrumRepresentation = Literal["linear", "semilogx", "semilogy", "loglog"]
FilterFunc = Callable[..., pd.DataFrame]


class Spectrum:
    """
    Spectrum composite:
    - attaches to Record
    - optional filter pipeline using existing rec.filter.*
    """

    def __init__(self, record: "Record") -> None:
        self._record = record

        # Pipeline: list[(function, {kwargs})]
        self._filter_steps: list[tuple[FilterFunc, dict]] = []

        # Outputs
        self.freqs: np.ndarray | None = None
        self.amplitudes: dict[ComponentName, np.ndarray] = {}
        self._df_used: pd.DataFrame | None = None

    # ---------------------------------------------------------- #
    # Filter pipeline
    # ---------------------------------------------------------- #

    def add_filter(self, func: FilterFunc, **params) -> "Spectrum":
        """
        Add a filter step to the spectrum computation.

        Example:
            rec.spectrum.add_filter(rec.filter.high_pass, Tc=0.1)
        """
        self._filter_steps.append((func, params))
        return self

    def clear_filters(self) -> None:
        self._filter_steps.clear()

    def describe_filters(self) -> str:
        if not self._filter_steps:
            return "(no filters)"
        parts = []
        for func, params in self._filter_steps:
            name = getattr(func, "__name__", "<fn>")
            args = ", ".join(f"{k}={v!r}" for k, v in params.items())
            parts.append(f"{name}({args})")
        return " → ".join(parts)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        for func, params in self._filter_steps:
            out = func(df=out, **params)
        return out

    # ---------------------------------------------------------- #
    # Spectrum computation
    # ---------------------------------------------------------- #

    def compute(
        self,
        df: pd.DataFrame | None = None,
        *,
        use_filters: bool = True,
        components: Sequence[ComponentName] | None = None,
    ) -> None:

        rec = self._record

        if df is None:
            df_work = rec.df.copy()
        else:
            df_work = df.copy()

        # Apply Spectrum filters
        if use_filters and self._filter_steps:
            df_work = self._apply_filters(df_work)

        # Which components
        if components is None:
            comps = [
                c for c in ("X", "Y", "Z")
                if c in rec.components and c in df_work.columns
            ]
        else:
            comps = [c for c in components if c in df_work.columns]

        if not comps:
            raise ValueError("No components to compute spectrum for.")

        dt = rec.dt
        time_arr = df_work["time"].to_numpy(float)
        n = len(time_arr)

        freqs = np.fft.rfftfreq(n, dt)
        amplitudes = {}

        for comp in comps:
            data = df_work[comp].to_numpy(float)
            amplitudes[comp] = np.abs(np.fft.rfft(data))

        self.freqs = freqs
        self.amplitudes = amplitudes
        self._df_used = df_work

    # ---------------------------------------------------------- #
    # Plotting
    # ---------------------------------------------------------- #

    def _ensure_computed(self):
        if self.freqs is None or not self.amplitudes:
            self.compute()

    def plot(
        self,
        *,
        representation: SpectrumRepresentation = "loglog",
        combined: bool = False,
        components: Sequence[ComponentName] | None = None,
        fig=None,
        axes=None,
        **plot_kwargs,
    ):
        import matplotlib.pyplot as plt

        self._ensure_computed()

        freqs = self.freqs
        amps = self.amplitudes

        # components to plot
        if components is None:
            comps = list(amps.keys())
        else:
            comps = [c for c in components if c in amps]

        # choose plot method
        def P(ax):
            match representation:
                case "linear": return ax.plot
                case "semilogx": return ax.semilogx
                case "semilogy": return ax.semilogy
                case "loglog": return ax.loglog
                case _: raise ValueError("Bad representation")

        # Combined plot
        if combined:
            if fig is None or axes is None:
                fig, ax = plt.subplots(figsize=(7, 5))
            else:
                ax = axes

            plot_fn = P(ax)
            for comp in comps:
                plot_fn(freqs[1:], amps[comp][1:], label=comp, **plot_kwargs)

            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            fig.tight_layout()
            return fig, ax

        # Separate subplots
        if fig is None or axes is None:
            fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 4))

        if len(comps) == 1:
            axes = [axes]

        for comp, ax in zip(comps, axes):
            plot_fn = P(ax)
            plot_fn(freqs[1:], amps[comp][1:], label=comp, **plot_kwargs)
            ax.set_title(comp)
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(True, which="both", ls="-", alpha=0.5)

        axes[0].set_ylabel("Amplitude")
        fig.tight_layout()
        return fig, axes

    def plot_combined(self, **kw):
        return self.plot(combined=True, **kw)

    def plot_separate(self, **kw):
        return self.plot(combined=False, **kw)

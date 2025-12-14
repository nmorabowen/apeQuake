from __future__ import annotations

from typing import Sequence
import matplotlib.pyplot as plt
from ..core.types import ComponentName


class PlotRecord:
    def __init__(self, record: "Record") -> None:
        self.record = record

    def _select_components(self, components: Sequence[ComponentName] | None) -> list[ComponentName]:
        df = self.record.df
        if components is None:
            comps = [c for c in ("X", "Y", "Z") if c in df.columns]
        else:
            comps = [c for c in components if c in df.columns]
        if not comps:
            avail = [c for c in ("X", "Y", "Z") if c in df.columns]
            raise ValueError(f"No components to plot. Requested={components}, available={avail}.")
        return comps

    def plot(
        self,
        components: Sequence[ComponentName] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax=None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:

        df = self.record.df
        if "time" not in df.columns:
            raise ValueError("Record.df must contain a 'time' column.")

        comps = self._select_components(components)
        t = df["time"].to_numpy(float)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for c in comps:
            ax.plot(t, df[c].to_numpy(float), label=c, **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if title:
            ax.set_title(title)

        fig.tight_layout()
        if show:
            plt.show()

        return fig, ax

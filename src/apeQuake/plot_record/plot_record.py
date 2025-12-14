from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import matplotlib.pyplot as plt
import pandas as pd

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record


class PlotRecord:
    def __init__(self, record: "Record") -> None:
        self.record = record

    # ------------------------- helpers ------------------------- #

    def _select_components(
        self,
        components: Sequence[ComponentName] | None,
        *,
        df: pd.DataFrame | None = None,
    ) -> list[ComponentName]:
        df_work = self.record.df if df is None else df

        if components is None:
            comps = [c for c in ("X", "Y", "Z") if c in df_work.columns]
        else:
            comps = [c for c in components if c in df_work.columns]

        if not comps:
            avail = [c for c in ("X", "Y", "Z") if c in df_work.columns]
            raise ValueError(
                f"No components selected/found. Requested={list(components) if components is not None else None}, "
                f"available={avail}."
            )
        return comps

    def _as_list_axes(self, axes, n: int) -> list[plt.Axes]:
        if n == 1:
            return [axes]
        return list(axes)

    # ------------------------- primitives ------------------------- #

    def plot_component(
        self,
        component: ComponentName,
        *,
        ax: plt.Axes,
        linewidth: float = 1.0,
        linestyle: str = "-",
        grid: bool = True,
        ylabel: str | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Primitive: draw ONE component vs time into the provided ax.
        Does not create figures, does not call show(), does not set xlabel().
        """
        df = self.record.df
        if "time" not in df.columns:
            raise ValueError("Record.df must contain a 'time' column.")
        if component not in df.columns:
            raise ValueError(f"Component '{component}' not found in Record.df.")

        t = df["time"].to_numpy(float)
        y = df[component].to_numpy(float)

        ax.plot(t, y, linewidth=linewidth, linestyle=linestyle, **kwargs)
        if grid:
            ax.grid(True, alpha=0.3)

        ax.set_ylabel(ylabel if ylabel is not None else component)
        return ax

    # ------------------------- public API ------------------------- #

    def plot(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = False,
        linewidth: float = 1.0,
        linestyle: str = "-",
        grid: bool = True,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Stacked record plot: nrows=len(components), ncols=1.
        Returns (fig, axes) where axes is always a list.
        """
        df = self.record.df
        if "time" not in df.columns:
            raise ValueError("Record.df must contain a 'time' column.")

        comps = self._select_components(components, df=df)
        n = len(comps)

        if figsize is None:
            figsize = (10, max(2.6 * n, 3.2))

        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
        )
        axes_list = self._as_list_axes(axes, n)

        for ax, c in zip(axes_list, comps):
            self.plot_component(
                c,
                ax=ax,
                linewidth=linewidth,
                linestyle=linestyle,
                grid=grid,
                **kwargs,
            )

        axes_list[-1].set_xlabel("Time (s)")

        if title:
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()

        if show:
            plt.show()

        return fig, axes_list

    def plot_record(self, **kw) -> tuple[plt.Figure, list[plt.Axes]]:
        """Alias for plot(), keeps naming consistent with your other composites."""
        return self.plot(**kw)

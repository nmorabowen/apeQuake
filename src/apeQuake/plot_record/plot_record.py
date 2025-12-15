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

    def plot_band_pass(
        self,
        *,
        Tc_low_list: Sequence[float],
        Tc_high_list: Sequence[float],
        components: Sequence[ComponentName] | None = None,
        title: str | None = None,
        corners: int = 4,
        zerophase: bool = True,
        sharex: bool = True,
        sharey: bool = True,
        figsize: tuple[float, float] | None = None,
        linewidth: float = 1.0,
        linestyle: str = "-",
        grid: bool = True,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, list[list[plt.Axes]]]:
        """
        Grid plot: rows = components, columns = [Original] + each band-pass filter.

        xlim, ylim:
            If provided, applied identically to ALL subplots.
        """
        rec = self.record
        df0 = rec.df
        if "time" not in df0.columns:
            raise ValueError("Record.df must contain a 'time' column.")

        if len(Tc_low_list) != len(Tc_high_list):
            raise ValueError("Tc_low_list and Tc_high_list must have the same length.")

        comps = self._select_components(components, df=df0)
        n_rows = len(comps)
        n_cols = 1 + len(Tc_low_list)

        if figsize is None:
            figsize = (4.8 * n_cols, max(2.6 * n_rows, 3.2))

        fig, ax = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
        )

        # normalize ax to a 2D list
        if n_rows == 1 and n_cols == 1:
            axes = [[ax]]
        elif n_rows == 1:
            axes = [list(ax)]
        elif n_cols == 1:
            axes = [[a] for a in ax]
        else:
            axes = [list(row) for row in ax]

        # ---- column 0: original ----
        for i, c in enumerate(comps):
            self.plot_component(
                c,
                ax=axes[i][0],
                linewidth=linewidth,
                linestyle=linestyle,
                grid=grid,
                **kwargs,
            )
            if i == 0:
                axes[i][0].set_title("Original")

        # ---- filtered columns ----
        for j, (Tc_low, Tc_high) in enumerate(zip(Tc_low_list, Tc_high_list), start=1):
            Tc_low = float(Tc_low)
            Tc_high = float(Tc_high)
            if Tc_low <= 0 or Tc_high <= 0:
                raise ValueError("Tc_low and Tc_high must be > 0.")
            if Tc_low >= Tc_high:
                raise ValueError("Require Tc_low < Tc_high.")

            df_f = rec.filter.band_pass(
                df=df0,
                Tc_low=Tc_low,
                Tc_high=Tc_high,
                corners=corners,
                zerophase=zerophase,
            )

            for i, c in enumerate(comps):
                t = df_f["time"].to_numpy(float)
                y = df_f[c].to_numpy(float)

                axes[i][j].plot(t, y, linewidth=linewidth, linestyle=linestyle, **kwargs)
                if grid:
                    axes[i][j].grid(True, alpha=0.3)
                axes[i][j].set_ylabel(c)

                if i == 0:
                    axes[i][j].set_title(f"Band-pass\nTc={Tc_low:g}–{Tc_high:g} s")

        # ---- apply global axis limits ----
        if xlim is not None:
            for row in axes:
                for ax_ in row:
                    ax_.set_xlim(*xlim)

        if ylim is not None:
            for row in axes:
                for ax_ in row:
                    ax_.set_ylim(*ylim)

        # x-label only on bottom row
        for j in range(n_cols):
            axes[-1][j].set_xlabel("Time (s)")

        if title:
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()

        if show:
            plt.show()

        return fig, axes


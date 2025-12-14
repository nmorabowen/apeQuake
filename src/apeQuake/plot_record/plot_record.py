from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record
    from ..filters.filters import Filter as FilterType
    from ..spectrum.spectrum import Spectrum
    from ..spectrogram.spectrogram import Spectrogram
    from ..response_spectra import ResponseSpectra
    from ..intensity_measures import IntensityMeasures

class PlotRecord:
    def __init__(self, record: "Record") -> None:
        self.record = record

    def plot(
        self,
        components: list[ComponentName] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 6),
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the record components.

        Parameters
        ----------
        components
            List of components to plot. If None, plot all available components.
        title
            Title of the plot.
        figsize
            Size of the figure.
        **kwargs
            Additional keyword arguments passed to pandas DataFrame plot method.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.
        """
        df_to_plot = self.record.df[components] if components is not None else self.record.df

        fig, ax = plt.subplots(figsize=figsize)
        df_to_plot.plot(ax=ax, **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig
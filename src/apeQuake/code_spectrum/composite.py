"""``rec.code_spectrum``: building-code design spectra next to a record's spectrum."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from apeQuake.core.types import ComponentName

from .base import CodeSpectrumModel, available_codes, get_model_class

if TYPE_CHECKING:
    from apeQuake.core.record import Record


class CodeSpectrum:
    """Code-spectrum composite for a :class:`~apeQuake.core.record.Record`.

    Holds any number of named code models (NEC, ASCE 7-10 / 7-16 / 7-22 ...),
    evaluates them on a common period grid and, optionally, overlays the
    5 %-damped pseudo-acceleration spectrum of the record.  It never touches
    ``record.df``: the record spectrum is computed by a private
    :class:`~apeQuake.response_spectra.ResponseSpectra`.

    Example
    -------
    >>> rec.code_spectrum.add("ASCE7-16", ss=1.5, s1=0.6, site_class="D", tl=8.0)
    >>> rec.code_spectrum.compute()
    >>> rec.code_spectrum.compute_record(units_per_g=9.81)  # record in m/s^2
    >>> rec.code_spectrum.plot()
    """

    def __init__(self, record: "Record") -> None:
        self._record = record
        self.models: dict[str, CodeSpectrumModel] = {}

        # Results (cache). Sa in g.
        self.periods: np.ndarray | None = None
        self.Sa: dict[str, np.ndarray] = {}
        self.record_Sa: dict[ComponentName, np.ndarray] = {}
        self._record_periods: np.ndarray | None = None

    # ---------------------------------------------------------- #
    # Model management
    # ---------------------------------------------------------- #
    @staticmethod
    def available_codes() -> list[str]:
        """Canonical names of the registered codes."""
        return available_codes()

    def add(
        self,
        model: str | CodeSpectrumModel,
        label: str | None = None,
        **inputs: Any,
    ) -> CodeSpectrumModel:
        """Add a model, by code name (``**inputs`` go to its constructor) or instance.

        ``label`` names the model in results and plots (default: its ``code``).
        Returns the model.  Adding invalidates cached results.
        """
        if isinstance(model, str):
            model = get_model_class(model)(**inputs)
        elif inputs:
            raise TypeError("keyword inputs are only valid when adding by code name")
        label = label or model.code
        if label in self.models:
            raise ValueError(f"A model labelled {label!r} already exists; pass label=...")
        self.models[label] = model
        self._invalidate()
        return model

    def remove(self, label: str) -> None:
        """Remove the model *label* and invalidate cached results."""
        del self.models[label]
        self._invalidate()

    def clear(self) -> None:
        """Remove every model and every cached result."""
        self.models.clear()
        self._invalidate()

    def _invalidate(self) -> None:
        self.periods = None
        self.Sa = {}
        self.record_Sa = {}
        self._record_periods = None

    # ---------------------------------------------------------- #
    # Compute
    # ---------------------------------------------------------- #
    def compute(
        self,
        periods: Sequence[float] | None = None,
        *,
        t_max: float = 4.0,
        n: int = 400,
    ) -> dict[str, np.ndarray]:
        """Evaluate every model on *periods* (default: a shared grid to *t_max*).

        The default grid is the union of each model's own grid, so every knee
        period (``T0``, ``Ts``, ``TL`` ...) is sampled exactly.  Results are
        cached in ``self.periods`` / ``self.Sa`` (g).
        """
        if not self.models:
            raise RuntimeError("No models: call add() first.")
        if periods is None:
            T = np.unique(
                np.concatenate([m.default_periods(t_max, n) for m in self.models.values()])
            )
        else:
            T = np.asarray(periods, dtype=float)
        self.Sa = {label: m.sa(T) for label, m in self.models.items()}
        self.periods = T
        self.record_Sa = {}
        self._record_periods = None
        return self.Sa

    def compute_record(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        damping: float = 0.05,
        units_per_g: float = 1.0,
        df: pd.DataFrame | None = None,
        parallel: bool = True,
    ) -> dict[ComponentName, np.ndarray]:
        """5 %-damped pseudo-acceleration spectrum of the record, in g.

        Evaluated at the cached ``periods`` greater than zero (call
        :meth:`compute` first, or it is called with defaults).  ``units_per_g``
        is how many acceleration units of the record make one g (``1`` if the
        record is in g, ``9.80665`` if in m/s^2, ``981`` if in cm/s^2).
        ``df`` optionally replaces ``record.df`` (it is copied, never modified).
        """
        from apeQuake.response_spectra import ResponseSpectra

        if units_per_g <= 0.0:
            raise ValueError("units_per_g must be > 0")
        if self.periods is None:
            self.compute()
        assert self.periods is not None
        T = self.periods[self.periods > 0.0]
        rs = ResponseSpectra(self._record)
        rs.compute(
            T,
            df=None if df is None else df.copy(),
            use_filters=False,
            components=components,
            damping=damping,
            parallel=parallel,
        )
        self.record_Sa = {c: np.asarray(v) / units_per_g for c, v in rs.Sa.items()}
        self._record_periods = T
        return self.record_Sa

    def summary(self) -> pd.DataFrame:
        """One row per model: its code and defining parameters."""
        rows = {
            label: {"code": m.code, **m.parameters()} for label, m in self.models.items()
        }
        return pd.DataFrame(rows).T

    # ---------------------------------------------------------- #
    # Plot (consumes the cache; never recomputes)
    # ---------------------------------------------------------- #
    def plot(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        ax=None,
        figsize: tuple[float, float] = (7.5, 5.0),
        show: bool = True,
    ):
        """Plot cached code spectra (and the record spectrum if computed)."""
        import matplotlib.pyplot as plt

        if self.periods is None or not self.Sa:
            raise RuntimeError("Nothing to plot: call compute() first.")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for label, sa in self.Sa.items():
            ax.plot(self.periods, sa, lw=1.8, label=label)
        if self.record_Sa and self._record_periods is not None:
            for comp, sa in self.record_Sa.items():
                if components is not None and comp not in components:
                    continue
                ax.plot(self._record_periods, sa, lw=1.0, ls="--", label=f"Record {comp}")
        ax.set_xlabel("Period T [s]")
        ax.set_ylabel("Sa [g]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

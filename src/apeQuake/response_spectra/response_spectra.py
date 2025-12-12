from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Callable, Literal
import numpy as np
import pandas as pd

from apeQuake.core.types import ComponentName

if TYPE_CHECKING:
    from apeQuake.core.record import Record

FilterFunc = Callable[..., pd.DataFrame]
RSQuantity = Literal["Sd", "Sv", "Sa"]   # Sa = pseudo-acceleration ω² Sd
RSRepresentation = Literal["linear", "semilogx", "loglog"]


class ResponseSpectra:
    """
    Response spectra composite for a multicomponent Record.

    - Solves SDOF response with Newmark average acceleration (γ=0.5, β=0.25)
      for each oscillator period and component.
    - Input: ground acceleration time series (from Record or a user-supplied df).
    - Output: Sd, Sv, Sa (pseudo) vs period, for each component.
    - Optional filter pipeline via `add_filter(rec.filter.*)`.

    Equation of motion (m=1):
        u¨ + 2 ξ ω u˙ + ω² u = -a_g(t)
    """

    def __init__(self, record: "Record") -> None:
        self._record = record

        # Pipeline: list of (filter_function, params)
        self._filter_steps: list[tuple[FilterFunc, dict]] = []

        # Results
        self.periods: np.ndarray | None = None
        self.Sd: dict[ComponentName, np.ndarray] = {}
        self.Sv: dict[ComponentName, np.ndarray] = {}
        self.Sa: dict[ComponentName, np.ndarray] = {}  # pseudo-acceleration
        self._df_used: pd.DataFrame | None = None

    # ---------------------------------------------------------- #
    # Filter pipeline
    # ---------------------------------------------------------- #

    def add_filter(self, func: FilterFunc, **params) -> "ResponseSpectra":
        """
        Add a filter step to the RS computation pipeline.

        Example
        -------
        rec.response_spectra.add_filter(rec.filter.detrend, type="demean")
        rec.response_spectra.add_filter(rec.filter.taper, max_percentage=0.05)
        rec.response_spectra.add_filter(rec.filter.band_pass, Tc_low=0.1, Tc_high=5.0)
        """
        self._filter_steps.append((func, params))
        return self

    def clear_filters(self) -> None:
        """Remove all registered filters from the pipeline."""
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

    def _apply_filters(self, df: pd.DataFrame, use_filters: bool | None) -> pd.DataFrame:
        """
        Apply the pipeline depending on use_filters and whether df is user-supplied.

        Convention
        ----------
        - If df comes from record (df=None in compute) and use_filters is
          True or None → apply pipeline.
        - If df is user-supplied and use_filters is True → apply pipeline.
        - If use_filters is False → never apply pipeline.
        """
        if not self._filter_steps:
            return df

        if use_filters is False:
            return df

        out = df
        for func, params in self._filter_steps:
            out = func(df=out, **params)
        return out

    # ---------------------------------------------------------- #
    # Newmark solver (internal)
    # ---------------------------------------------------------- #

    @staticmethod
    def _newmark_sdof(
        ag: np.ndarray,
        dt: float,
        T: float,
        damping: float,
        gamma: float = 0.5,
        beta: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Newmark average acceleration SDOF integration for one oscillator.

        Parameters
        ----------
        ag : np.ndarray
            Ground acceleration time history (a_g(t)).
        dt : float
            Time step.
        T : float
            Natural period of the SDOF [s].
        damping : float
            Damping ratio (ξ).
        gamma, beta : float
            Newmark parameters (average acceleration → γ=0.5, β=0.25).

        Returns
        -------
        u, v, a : np.ndarray
            Relative displacement, velocity, and acceleration time histories.
        """
        # Basic properties (m=1)
        w = 2.0 * np.pi / T
        k = w**2        # stiffness
        c = 2.0 * damping * w  # viscous damping

        n = len(ag)
        u = np.zeros(n, dtype=float)
        v = np.zeros(n, dtype=float)
        a = np.zeros(n, dtype=float)

        # Initial acceleration from equilibrium: u0 = v0 = 0
        # u¨0 = -a_g(0) (since m=1)
        a[0] = -ag[0]

        # Newmark constants
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)

        k_hat = k + a0 + a1 * c  # m=1

        # External load: p = -a_g
        p = -ag

        for i in range(n - 1):
            # Effective load at step i+1
            p_eff = (
                p[i + 1]
                + a0 * u[i]
                + a2 * v[i]
                + a3 * a[i]
                + a1 * c * u[i]
                + a4 * c * v[i]
                + a5 * c * a[i]
            )

            u_new = p_eff / k_hat
            du = u_new - u[i]

            v_new = a1 * du - a4 * v[i] - a5 * a[i]
            a_new = a0 * du - a2 * v[i] - a3 * a[i]

            u[i + 1] = u_new
            v[i + 1] = v_new
            a[i + 1] = a_new

        return u, v, a

    # ---------------------------------------------------------- #
    # Core RS computation
    # ---------------------------------------------------------- #

    def compute(
        self,
        periods: Sequence[float],
        *,
        df: pd.DataFrame | None = None,
        use_filters: bool | None = None,
        components: Sequence[ComponentName] | None = None,
        damping: float = 0.05,
        gamma: float = 0.5,
        beta: float = 0.25,
    ) -> None:
        """
        Compute response spectra (Sd, Sv, Sa_pseudo) for given periods.

        Parameters
        ----------
        periods : sequence of float
            Oscillator periods [s] for the response spectra.

        df : DataFrame or None
            Input acceleration time histories. Must contain 'time' and
            component columns. If None, uses record.df.

        use_filters : {True, False, None}
            - If df is None:
                * None/True → apply internal filter pipeline on record.df.
                * False      → use record.df as-is.
            - If df is provided:
                * True  → apply pipeline on provided df.
                * None/False → DO NOT apply pipeline.

        components : sequence or None
            Components to consider. If None, use all that exist in both
            Record.components and df columns.

        damping : float
            Damping ratio (ξ).

        gamma, beta : float
            Newmark integration parameters. Default is average acceleration
            (γ=0.5, β=0.25).
        """
        rec = self._record

        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute response spectra.")

        # --- Base df selection and filter logic ---
        if df is None:
            df_work = rec.df.copy()
            # default: if user doesn't say anything, apply filters
            effective_use_filters = True if use_filters is None else use_filters
        else:
            df_work = df.copy()
            # default: if user passes df, do NOT apply filters unless explicit
            effective_use_filters = True if use_filters is True else False

        if effective_use_filters and self._filter_steps:
            df_work = self._apply_filters(df_work, use_filters=True)

        # --- Components selection ---
        if components is None:
            comps: list[ComponentName] = [
                c for c in ("X", "Y", "Z")
                if c in rec.components and c in df_work.columns
            ]  # type: ignore[list-item]
        else:
            comps = [c for c in components if c in df_work.columns]

        if not comps:
            raise ValueError("No components available for response spectra.")

        # --- Period array ---
        periods_arr = np.asarray(periods, dtype=float)
        if np.any(periods_arr <= 0.0):
            raise ValueError("All periods must be > 0.")

        dt = rec.dt
        nT = len(periods_arr)

        # Initialize result dicts
        Sd: dict[ComponentName, np.ndarray] = {}
        Sv: dict[ComponentName, np.ndarray] = {}
        Sa: dict[ComponentName, np.ndarray] = {}

        # --- Loop over components and periods ---
        for comp in comps:
            ag = df_work[comp].to_numpy(dtype=float)

            Sd_comp = np.zeros(nT, dtype=float)
            Sv_comp = np.zeros(nT, dtype=float)
            Sa_comp = np.zeros(nT, dtype=float)

            for j, T in enumerate(periods_arr):
                u, v, a_rel = self._newmark_sdof(
                    ag=ag,
                    dt=dt,
                    T=T,
                    damping=damping,
                    gamma=gamma,
                    beta=beta,
                )

                # Relative maxima
                Sd_comp[j] = np.max(np.abs(u))
                Sv_comp[j] = np.max(np.abs(v))

                # Pseudo-acceleration spectrum: Sa = ω² Sd
                w = 2.0 * np.pi / T
                Sa_comp[j] = (w**2) * Sd_comp[j]

            Sd[comp] = Sd_comp
            Sv[comp] = Sv_comp
            Sa[comp] = Sa_comp

        # Store
        self.periods = periods_arr
        self.Sd = Sd
        self.Sv = Sv
        self.Sa = Sa
        self._df_used = df_work

    # ---------------------------------------------------------- #
    # Plotting
    # ---------------------------------------------------------- #

    def _ensure_computed(self) -> None:
        if (
            self.periods is None
            or (not self.Sd and not self.Sv and not self.Sa)
        ):
            raise RuntimeError("ResponseSpectra.compute() has not been called yet.")

    def plot(
        self,
        quantity: RSQuantity = "Sa",
        *,
        components: Sequence[ComponentName] | None = None,
        representation: RSRepresentation = "loglog",
        combined: bool = False,
        fig=None,
        axes=None,
        **plot_kwargs,
    ):
        """
        Plot response spectra.

        Parameters
        ----------
        quantity : {'Sd','Sv','Sa'}
            Which spectrum to plot.
            Sa is pseudo-acceleration (ω² Sd).

        components : sequence or None
            Components to plot. If None, use all that are available.

        representation : {'linear','semilogx','loglog'}
            Plot style in (T, quantity) space.

        combined : bool
            If True, all components in one axes. If False, separate subplots.

        fig, axes : optional Matplotlib Figure/Axes to draw into.

        **plot_kwargs :
            Forwarded to the matplotlib plotting function
            (color, linewidth, alpha, etc.).

        Returns
        -------
        fig, axes
        """
        import matplotlib.pyplot as plt

        self._ensure_computed()
        assert self.periods is not None

        T = self.periods

        if quantity == "Sd":
            data_dict = self.Sd
            ylabel = "Spectral Displacement (Sd)"
        elif quantity == "Sv":
            data_dict = self.Sv
            ylabel = "Spectral Velocity (Sv)"
        elif quantity == "Sa":
            data_dict = self.Sa
            ylabel = "Pseudo-Acceleration (Sa)"
        else:
            raise ValueError("quantity must be one of 'Sd', 'Sv', 'Sa'.")

        # Components to plot
        if components is None:
            comps_plot: list[ComponentName] = list(data_dict.keys())
        else:
            comps_plot = [c for c in components if c in data_dict]

        if not comps_plot:
            raise ValueError("No components available to plot for the chosen quantity.")

        # Plot style helper
        def P(ax):
            if representation == "linear":
                return ax.plot
            elif representation == "semilogx":
                return ax.semilogx
            elif representation == "loglog":
                return ax.loglog
            else:
                raise ValueError(
                    "representation must be one of 'linear','semilogx','loglog'"
                )

        # Combined
        if combined:
            if fig is None or axes is None:
                fig, ax = plt.subplots(figsize=(7, 5))
            else:
                ax = axes

            plot_fn = P(ax)
            for comp in comps_plot:
                plot_fn(
                    T,
                    data_dict[comp],
                    label=f"Comp {comp}",
                    **plot_kwargs,
                )

            ax.set_xlabel("Period T [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            return fig, ax

        # Separate subplots
        n = len(comps_plot)
        if fig is None or axes is None:
            fig, axes = plt.subplots(
                nrows=1,
                ncols=n,
                figsize=(5 * n, 4),
                sharey=True,
            )

        if n == 1:
            axes_list = [axes]
        else:
            axes_list = list(axes)

        for comp, ax in zip(comps_plot, axes_list):
            plot_fn = P(ax)
            plot_fn(T, data_dict[comp], label=f"Comp {comp}", **plot_kwargs)
            ax.set_title(f"Component {comp}")
            ax.set_xlabel("Period T [s]")
            ax.grid(True, which="both", ls="-", alpha=0.5)

        axes_list[0].set_ylabel(ylabel)
        fig.tight_layout()
        return fig, axes_list

    def plot_combined(self, quantity: RSQuantity = "Sa", **kwargs):
        return self.plot(quantity=quantity, combined=True, **kwargs)

    def plot_separate(self, quantity: RSQuantity = "Sa", **kwargs):
        return self.plot(quantity=quantity, combined=False, **kwargs)

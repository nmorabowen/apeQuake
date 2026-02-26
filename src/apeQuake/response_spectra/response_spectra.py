from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Callable, Literal
import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration dependency
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap

    def prange(*args):
        return range(*args)

from apeQuake.core.types import ComponentName

if TYPE_CHECKING:
    from apeQuake.core.record import Record

FilterFunc = Callable[..., pd.DataFrame]
RSQuantity = Literal["Sd", "Sv", "Sa"]   # Sa = pseudo-acceleration ω² Sd
RSRepresentation = Literal["linear", "semilogx", "loglog"]
SaMode = Literal["pseudo", "absolute"]


@njit(cache=True, fastmath=True, parallel=True)
def _newmark_spectra_kernel(
    ag: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float,
    gamma: float,
    beta: float,
    sa_mode_abs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sd, Sv, Sa for one acceleration series (parallel over periods)."""
    nT = len(periods)
    n = len(ag)

    Sd = np.zeros(nT, dtype=np.float64)
    Sv = np.zeros(nT, dtype=np.float64)
    Sa = np.zeros(nT, dtype=np.float64)

    for j in prange(nT):
        T = periods[j]
        if T <= 0.0:
            Sd[j] = np.nan
            Sv[j] = np.nan
            Sa[j] = np.nan
            continue

        w = 2.0 * np.pi / T
        k = w * w
        c = 2.0 * damping * w

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)

        k_hat = k + a0 + a1 * c

        u = 0.0
        v = 0.0
        a = -ag[0]

        sd_max = 0.0
        sv_max = 0.0
        sa_abs_max = abs(a + ag[0])

        for i in range(n - 1):
            p_eff = (
                -ag[i + 1]
                + a0 * u
                + a2 * v
                + a3 * a
                + a1 * c * u
                + a4 * c * v
                + a5 * c * a
            )

            u_new = p_eff / k_hat
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)

            abs_u = abs(u_new)
            if abs_u > sd_max:
                sd_max = abs_u

            abs_v = abs(v_new)
            if abs_v > sv_max:
                sv_max = abs_v

            abs_a_abs = abs(a_new + ag[i + 1])
            if abs_a_abs > sa_abs_max:
                sa_abs_max = abs_a_abs

            u = u_new
            v = v_new
            a = a_new

        Sd[j] = sd_max
        Sv[j] = sv_max
        if sa_mode_abs == 1:
            Sa[j] = sa_abs_max
        else:
            Sa[j] = (w * w) * sd_max

    return Sd, Sv, Sa


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
        sa_mode: SaMode = "pseudo",
        parallel: bool = True,
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

        sa_mode : {'pseudo', 'absolute'}
            Definition used for spectral acceleration Sa:
            - 'pseudo'   -> Sa = ω² Sd
            - 'absolute' -> Sa = max(|u¨ + a_g|)

        parallel : bool
            If True, use a Numba-parallel kernel over periods when available.
            Falls back to Python loops if Numba is not installed.
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
        if sa_mode not in ("pseudo", "absolute"):
            raise ValueError("sa_mode must be one of {'pseudo', 'absolute'}.")

        dt = rec.dt
        nT = len(periods_arr)
        sa_mode_abs = 1 if sa_mode == "absolute" else 0

        # Initialize result dicts
        Sd: dict[ComponentName, np.ndarray] = {}
        Sv: dict[ComponentName, np.ndarray] = {}
        Sa: dict[ComponentName, np.ndarray] = {}

        # --- Loop over components and periods ---
        for comp in comps:
            ag = df_work[comp].to_numpy(dtype=float)

            if parallel and _NUMBA_AVAILABLE:
                Sd_comp, Sv_comp, Sa_comp = _newmark_spectra_kernel(
                    ag=np.ascontiguousarray(ag, dtype=np.float64),
                    dt=float(dt),
                    periods=np.ascontiguousarray(periods_arr, dtype=np.float64),
                    damping=float(damping),
                    gamma=float(gamma),
                    beta=float(beta),
                    sa_mode_abs=sa_mode_abs,
                )
            else:
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

                    Sd_comp[j] = np.max(np.abs(u))
                    Sv_comp[j] = np.max(np.abs(v))

                    if sa_mode == "pseudo":
                        w = 2.0 * np.pi / T
                        Sa_comp[j] = (w**2) * Sd_comp[j]
                    else:
                        a_abs = a_rel + ag
                        Sa_comp[j] = np.max(np.abs(a_abs))

            Sd[comp] = Sd_comp
            Sv[comp] = Sv_comp
            Sa[comp] = Sa_comp

        # Store
        self.periods = periods_arr
        self.Sd = Sd
        self.Sv = Sv
        self.Sa = Sa
        self._df_used = df_work

    def newmark_sdof(
        self,
        T: float,
        *,
        component: ComponentName | None = None,
        df: pd.DataFrame | None = None,
        use_filters: bool | None = None,
        damping: float = 0.05,
        gamma: float = 0.5,
        beta: float = 0.25,
        plot: bool = False,
        fig=None,
        axes=None,
        **plot_kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Compute one SDOF response history for a single component.

        Returns
        -------
        dict with keys:
            'time', 'ag', 'displacement', 'velocity',
            'acceleration' (relative), 'total_acceleration' (absolute),
            'component', 'period', 'damping'
        """
        rec = self._record

        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute SDOF response.")
        if T <= 0.0:
            raise ValueError("T must be > 0.")

        if df is None:
            df_work = rec.df.copy()
            effective_use_filters = True if use_filters is None else use_filters
        else:
            df_work = df.copy()
            effective_use_filters = True if use_filters is True else False

        if effective_use_filters and self._filter_steps:
            df_work = self._apply_filters(df_work, use_filters=True)

        available = [c for c in ("X", "Y", "Z") if c in df_work.columns]
        if not available:
            raise ValueError("No components available for SDOF response.")

        comp: ComponentName
        if component is None:
            comp = available[0]  # type: ignore[assignment]
        else:
            if component not in df_work.columns:
                raise ValueError(f"Component '{component}' not found in selected DataFrame.")
            comp = component

        ag = df_work[comp].to_numpy(dtype=float)
        time = df_work["time"].to_numpy(dtype=float)
        u, v, a_rel = self._newmark_sdof(
            ag=ag,
            dt=rec.dt,
            T=float(T),
            damping=damping,
            gamma=gamma,
            beta=beta,
        )
        a_abs = a_rel + ag

        out = {
            "time": time,
            "ag": ag,
            "displacement": u,
            "velocity": v,
            "acceleration": a_rel,
            "total_acceleration": a_abs,
            "component": np.array([comp]),
            "period": np.array([float(T)]),
            "damping": np.array([float(damping)]),
        }

        if plot:
            self.plot_sdof_response(
                out,
                fig=fig,
                axes=axes,
                **plot_kwargs,
            )

        return out

    def compute_response_spectrum(
        self,
        periods: Sequence[float],
        *,
        component: ComponentName | None = None,
        df: pd.DataFrame | None = None,
        use_filters: bool | None = None,
        damping: float = 0.05,
        gamma: float = 0.5,
        beta: float = 0.25,
        sa_mode: SaMode = "absolute",
        parallel: bool = True,
        plot: bool = False,
        fig=None,
        axes=None,
        **plot_kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Compute Sd, Sv, Sa for one component and return them as arrays.

        Notes
        -----
        This is a convenience API for single-component workflows.
        For multi-component workflows, use `compute(...)` and `plot(...)`.
        """
        rec = self._record
        if df is None:
            df_work = rec.df.copy()
        else:
            df_work = df.copy()

        available = [c for c in ("X", "Y", "Z") if c in df_work.columns]
        if not available:
            raise ValueError("No components available for response spectrum.")

        comp: ComponentName
        if component is None:
            comp = available[0]  # type: ignore[assignment]
        else:
            if component not in df_work.columns:
                raise ValueError(f"Component '{component}' not found in selected DataFrame.")
            comp = component

        self.compute(
            periods=periods,
            df=df_work,
            use_filters=use_filters,
            components=[comp],
            damping=damping,
            gamma=gamma,
            beta=beta,
            sa_mode=sa_mode,
            parallel=parallel,
        )

        assert self.periods is not None
        result = {
            "T": self.periods.copy(),
            "Sa": self.Sa[comp].copy(),
            "Sv": self.Sv[comp].copy(),
            "Sd": self.Sd[comp].copy(),
            "component": np.array([comp]),
            "units": {
                "T": "s",
                "Sa": "accel-units",
                "Sv": "velocity-units",
                "Sd": "length-units",
            },
        }

        if plot:
            self.plot_response_spectrum(result, fig=fig, axes=axes, **plot_kwargs)

        return result

    def plot_response_spectrum(
        self,
        spectrum_results: dict[str, np.ndarray],
        *,
        fig=None,
        axes=None,
        figsize: tuple[float, float] = (6, 8),
        linewidth: float = 1.0,
        linestyle: str = "-",
        show: bool = True,
    ):
        """Plot Sa, Sv, Sd in 3 stacked subplots from a spectrum-results dictionary."""
        import matplotlib.pyplot as plt

        T = np.asarray(spectrum_results["T"], dtype=float)
        Sa = np.asarray(spectrum_results["Sa"], dtype=float)
        Sv = np.asarray(spectrum_results["Sv"], dtype=float)
        Sd = np.asarray(spectrum_results["Sd"], dtype=float)

        if fig is None or axes is None:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

        axes_list = [axes] if not isinstance(axes, (list, tuple, np.ndarray)) else list(axes)
        if len(axes_list) != 3:
            raise ValueError("axes must contain exactly 3 matplotlib axes.")

        axes_list[0].plot(T, Sa, linewidth=linewidth, linestyle=linestyle, label="Sa")
        axes_list[1].plot(T, Sv, linewidth=linewidth, linestyle=linestyle, label="Sv")
        axes_list[2].plot(T, Sd, linewidth=linewidth, linestyle=linestyle, label="Sd")

        axes_list[0].set_ylabel("Sa")
        axes_list[1].set_ylabel("Sv")
        axes_list[2].set_ylabel("Sd")
        axes_list[2].set_xlabel("Period T [s]")

        for ax in axes_list:
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes_list

    def plot_sdof_response(
        self,
        response: dict[str, np.ndarray],
        *,
        fig=None,
        axes=None,
        figsize: tuple[float, float] = (6, 8),
        linewidth: float = 1.0,
        linestyle: str = "-",
        show: bool = True,
    ):
        """
        Plot SDOF response arrays returned by `newmark_sdof(...)`.

        Subplots (top to bottom): absolute acceleration, relative velocity,
        relative displacement.
        """
        import matplotlib.pyplot as plt

        t = np.asarray(response["time"], dtype=float)
        a_abs = np.asarray(response["total_acceleration"], dtype=float)
        v = np.asarray(response["velocity"], dtype=float)
        u = np.asarray(response["displacement"], dtype=float)

        if fig is None or axes is None:
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

        axes_list = [axes] if not isinstance(axes, (list, tuple, np.ndarray)) else list(axes)
        if len(axes_list) != 3:
            raise ValueError("axes must contain exactly 3 matplotlib axes.")

        axes_list[0].plot(t, a_abs, linewidth=linewidth, linestyle=linestyle, label="a_abs")
        axes_list[1].plot(t, v, linewidth=linewidth, linestyle=linestyle, label="v")
        axes_list[2].plot(t, u, linewidth=linewidth, linestyle=linestyle, label="u")

        axes_list[0].set_ylabel("Abs. Accel")
        axes_list[1].set_ylabel("Rel. Vel")
        axes_list[2].set_ylabel("Rel. Disp")
        axes_list[2].set_xlabel("Time (s)")

        for ax in axes_list:
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes_list

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

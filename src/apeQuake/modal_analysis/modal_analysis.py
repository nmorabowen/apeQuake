from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import svd, eig

from ..core.types import ComponentName

if TYPE_CHECKING:
    from ..core.record import Record
    from ..filters.filters import Filter


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

@dataclass(slots=True)
class SSIConfig:
    window_length: float = 10.0
    overlap: float = 0.5
    max_order: int = 50
    min_order: int = 2
    order_step: int = 2
    n_block_rows: int = 15
    freq_tol: float = 0.05
    damp_tol: float = 0.02
    max_damping: float = 0.20
    min_frequency: float = 0.0


# ------------------------------------------------------------------ #
# ModalAnalysis
# ------------------------------------------------------------------ #

class ModalAnalysis:

    def __init__(self, record: "Record") -> None:
        self.record = record
        self.filter: "Filter" = record.filter

        self.df_modal: pd.DataFrame = record.df.copy()
        self.applied_filters: list[str] = []
        self.config: SSIConfig = SSIConfig()

        # windowed SSI results
        self.window_times: np.ndarray | None = None
        self.frequencies: dict[ComponentName, np.ndarray] = {}
        self.damping_ratios: dict[ComponentName, np.ndarray] = {}

        # stabilization diagram results
        self.stab_frequencies: dict[ComponentName, list[np.ndarray]] = {}
        self.stab_damping: dict[ComponentName, list[np.ndarray]] = {}
        self.stab_orders: np.ndarray | None = None
        self.stab_stable_mask: dict[ComponentName, list[np.ndarray]] = {}

    # ----------------------- DF lifecycle -------------------------- #

    def set_df(self, df: pd.DataFrame) -> "ModalAnalysis":
        self.df_modal = df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def reset_df(self) -> "ModalAnalysis":
        self.df_modal = self.record.df.copy()
        self._invalidate()
        self.applied_filters = []
        return self

    def _invalidate(self) -> None:
        self.window_times = None
        self.frequencies = {}
        self.damping_ratios = {}
        self.stab_frequencies = {}
        self.stab_damping = {}
        self.stab_orders = None
        self.stab_stable_mask = {}

    def _log(self, s: str) -> None:
        self.applied_filters.append(s)

    # ----------------------- filter wrappers ----------------------- #

    def apply_low_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "ModalAnalysis":
        self.df_modal = self.filter.low_pass(df=self.df_modal, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"low_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_high_pass(self, Tc: float, *, corners: int = 4, zerophase: bool = True) -> "ModalAnalysis":
        self.df_modal = self.filter.high_pass(df=self.df_modal, Tc=Tc, corners=corners, zerophase=zerophase)
        self._invalidate()
        self._log(f"high_pass(Tc={Tc}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_pass(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "ModalAnalysis":
        self.df_modal = self.filter.band_pass(
            df=self.df_modal, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_pass(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_band_stop(self, Tc_low: float, Tc_high: float, *, corners: int = 4, zerophase: bool = True) -> "ModalAnalysis":
        self.df_modal = self.filter.band_stop(
            df=self.df_modal, Tc_low=Tc_low, Tc_high=Tc_high, corners=corners, zerophase=zerophase
        )
        self._invalidate()
        self._log(f"band_stop(Tc_low={Tc_low}, Tc_high={Tc_high}, corners={corners}, zerophase={zerophase})")
        return self

    def apply_detrend(self, type: str = "demean") -> "ModalAnalysis":
        self.df_modal = self.filter.detrend(df=self.df_modal, type=type)
        self._invalidate()
        self._log(f"detrend(type='{type}')")
        return self

    def apply_taper(self, max_percentage: float = 0.05, *, type: str = "cosine") -> "ModalAnalysis":
        self.df_modal = self.filter.taper(df=self.df_modal, max_percentage=max_percentage, type=type)
        self._invalidate()
        self._log(f"taper(max_percentage={max_percentage}, type='{type}')")
        return self

    def apply_base_filters(
        self,
        *,
        detrend: bool = True,
        demean: bool = True,
        taper: float | None = 0.05,
        band: tuple[float, float] | None = None,
    ) -> "ModalAnalysis":
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

    # ----------------------- configuration ------------------------- #

    def set_config(self, **kwargs) -> "ModalAnalysis":
        for key, val in kwargs.items():
            if not hasattr(self.config, key):
                raise ValueError(f"Unknown SSIConfig parameter: {key}")
            setattr(self.config, key, val)
        self._invalidate()
        return self

    # ================================================================ #
    #                      SSI-COV ALGORITHM                           #
    # ================================================================ #

    @staticmethod
    def _compute_covariances_single(y: np.ndarray, n_lags: int) -> np.ndarray:
        """Output covariance R(k) for scalar channel, k = 0 .. n_lags-1."""
        N = len(y)
        R = np.empty(n_lags, dtype=float)
        for k in range(n_lags):
            R[k] = np.dot(y[: N - k], y[k:]) / N
        return R

    @staticmethod
    def _compute_covariances_multi(Y: np.ndarray, n_lags: int) -> np.ndarray:
        """
        Block covariance matrices for multi-channel data.

        Parameters
        ----------
        Y : (N, n_ch) array
        n_lags : number of lags

        Returns
        -------
        R : (n_lags, n_ch, n_ch) array
        """
        N, n_ch = Y.shape
        R = np.zeros((n_lags, n_ch, n_ch), dtype=float)
        for k in range(n_lags):
            R[k] = (Y[: N - k].T @ Y[k:]) / N
        return R

    @staticmethod
    def _build_hankel_single(R: np.ndarray, i: int) -> np.ndarray:
        """
        Block Hankel matrix for scalar channel.

        H[r, c] = R[r + c + 1] for r, c = 0..i-1
        Requires R with indices 1..2i-1 (i.e. len(R) >= 2*i).
        """
        H = np.empty((i, i), dtype=float)
        for r in range(i):
            for c in range(i):
                H[r, c] = R[r + c + 1]
        return H

    @staticmethod
    def _build_hankel_multi(R: np.ndarray, i: int) -> np.ndarray:
        """
        Block Hankel matrix from matrix covariances.

        R : (n_lags, n_ch, n_ch), indices 0..2i-1
        H[r,c] block = R[r + c + 1]
        Returns H of shape (i*n_ch, i*n_ch).
        """
        n_ch = R.shape[1]
        H = np.zeros((i * n_ch, i * n_ch), dtype=float)
        for r in range(i):
            for c in range(i):
                lag = r + c + 1
                H[r * n_ch: (r + 1) * n_ch, c * n_ch: (c + 1) * n_ch] = R[lag]
        return H

    @staticmethod
    def _extract_modal_params(
        T: np.ndarray,
        model_order: int,
        dt: float,
        n_ch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        From the block Toeplitz matrix, extract frequencies and damping.

        Returns (frequencies_hz, damping_ratios) arrays for physical modes.
        """
        U, S, _ = svd(T, full_matrices=False)

        if model_order > len(S):
            model_order = len(S)

        U_n = U[:, :model_order]
        S_n = S[:model_order]

        # Observability matrix
        O = U_n * np.sqrt(S_n)[np.newaxis, :]

        # Extract system matrix A from shift structure
        O_up = O[n_ch:, :]
        O_down = O[:-n_ch, :]
        A, _, _, _ = np.linalg.lstsq(O_down, O_up, rcond=None)

        eigenvalues, _ = eig(A)

        # Convert discrete -> continuous eigenvalues
        # Filter out zero or negative-magnitude eigenvalues
        mag = np.abs(eigenvalues)
        valid = mag > 1e-12
        eigenvalues = eigenvalues[valid]

        lam_c = np.log(eigenvalues.astype(complex)) / dt
        omega = np.abs(lam_c)

        # Avoid division by zero
        nonzero = omega > 1e-12
        lam_c = lam_c[nonzero]
        omega = omega[nonzero]

        freq_hz = omega / (2.0 * np.pi)
        damping = -np.real(lam_c) / omega

        return freq_hz, damping

    @staticmethod
    def _filter_physical_modes(
        freq_hz: np.ndarray,
        damping: np.ndarray,
        max_damping: float,
        min_frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only physical, positive-frequency, conjugate-unique modes."""
        # Keep one from each conjugate pair (positive imaginary part in continuous domain)
        # Since freq is always positive, filter by positive damping
        mask = (
            (damping > 0.0)
            & (damping < max_damping)
            & (freq_hz > min_frequency)
        )

        # Remove near-duplicate frequencies (conjugate pairs produce identical freq)
        f_out = freq_hz[mask]
        d_out = damping[mask]

        if len(f_out) == 0:
            return f_out, d_out

        # Sort by frequency and remove duplicates within 0.1% tolerance
        order = np.argsort(f_out)
        f_out = f_out[order]
        d_out = d_out[order]

        keep = [True]
        for j in range(1, len(f_out)):
            if abs(f_out[j] - f_out[j - 1]) / max(f_out[j], 1e-12) < 0.001:
                keep.append(False)
            else:
                keep.append(True)
        keep_arr = np.array(keep)
        return f_out[keep_arr], d_out[keep_arr]

    @classmethod
    def _ssi_cov_single(
        cls,
        y_window: np.ndarray,
        dt: float,
        i: int,
        model_order: int,
        max_damping: float,
        min_frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SSI-COV for a single window, single channel."""
        n_lags = 2 * i
        if len(y_window) < n_lags + 1:
            return np.array([]), np.array([])

        R = cls._compute_covariances_single(y_window, n_lags)
        H = cls._build_hankel_single(R, i)
        freq_hz, damping = cls._extract_modal_params(H, model_order, dt, n_ch=1)
        return cls._filter_physical_modes(freq_hz, damping, max_damping, min_frequency)

    @classmethod
    def _ssi_cov_multi(
        cls,
        Y_window: np.ndarray,
        dt: float,
        i: int,
        model_order: int,
        max_damping: float,
        min_frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SSI-COV for a single window, multi-channel joint identification."""
        n_ch = Y_window.shape[1]
        n_lags = 2 * i
        if Y_window.shape[0] < n_lags + 1:
            return np.array([]), np.array([])

        R = cls._compute_covariances_multi(Y_window, n_lags)
        H = cls._build_hankel_multi(R, i)
        freq_hz, damping = cls._extract_modal_params(H, model_order, dt, n_ch=n_ch)
        return cls._filter_physical_modes(freq_hz, damping, max_damping, min_frequency)

    # ================================================================ #
    #                       MODE TRACKING                              #
    # ================================================================ #

    @staticmethod
    def _track_modes(
        all_freqs: list[np.ndarray],
        all_damps: list[np.ndarray],
        freq_tol: float,
        damp_tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Track modes across windows by frequency proximity.

        Returns NaN-padded arrays of shape (n_windows, n_tracked_modes).
        """
        n_windows = len(all_freqs)
        if n_windows == 0:
            return np.array([]).reshape(0, 0), np.array([]).reshape(0, 0)

        # Each track: list of (freq, damp) per window, NaN if unmatched
        tracks_f: list[list[float]] = []
        tracks_d: list[list[float]] = []

        for w in range(n_windows):
            f_w = all_freqs[w]
            d_w = all_damps[w]
            n_modes = len(f_w)
            matched = np.zeros(n_modes, dtype=bool)

            # Try to extend existing tracks
            for t in range(len(tracks_f)):
                prev_f = tracks_f[t][-1]
                if np.isnan(prev_f):
                    # Look back for last valid frequency in this track
                    for back in range(len(tracks_f[t]) - 1, -1, -1):
                        if not np.isnan(tracks_f[t][back]):
                            prev_f = tracks_f[t][back]
                            break
                    else:
                        # Entire track is NaN so far
                        tracks_f[t].append(np.nan)
                        tracks_d[t].append(np.nan)
                        continue

                if n_modes == 0:
                    tracks_f[t].append(np.nan)
                    tracks_d[t].append(np.nan)
                    continue

                # Find closest unmatched mode
                rel_diff = np.abs(f_w - prev_f) / max(prev_f, 1e-12)
                rel_diff[matched] = np.inf
                best = int(np.argmin(rel_diff))

                if rel_diff[best] < freq_tol:
                    tracks_f[t].append(float(f_w[best]))
                    tracks_d[t].append(float(d_w[best]))
                    matched[best] = True
                else:
                    tracks_f[t].append(np.nan)
                    tracks_d[t].append(np.nan)

            # Start new tracks for unmatched modes
            for j in range(n_modes):
                if not matched[j]:
                    pad = [np.nan] * w
                    tracks_f.append(pad + [float(f_w[j])])
                    tracks_d.append(pad + [float(d_w[j])])

        if not tracks_f:
            return np.full((n_windows, 0), np.nan), np.full((n_windows, 0), np.nan)

        # Pad all tracks to full length
        n_tracks = len(tracks_f)
        freq_arr = np.full((n_windows, n_tracks), np.nan)
        damp_arr = np.full((n_windows, n_tracks), np.nan)
        for t in range(n_tracks):
            L = len(tracks_f[t])
            freq_arr[:L, t] = tracks_f[t]
            damp_arr[:L, t] = tracks_d[t]

        return freq_arr, damp_arr

    # ================================================================ #
    #                        COMPUTE                                   #
    # ================================================================ #

    def _resolve_components(
        self, components: Sequence[ComponentName] | None
    ) -> list[ComponentName]:
        df_work = self.df_modal
        if components is None:
            return [
                c for c in ("X", "Y", "Z")
                if c in self.record.components and c in df_work.columns
            ]  # type: ignore[list-item]
        return [c for c in components if c in df_work.columns]

    def compute(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        mode: Literal["independent", "joint"] = "independent",
        window_length: float | None = None,
        overlap: float | None = None,
        model_order: int | None = None,
        n_block_rows: int | None = None,
        max_damping: float | None = None,
        min_frequency: float | None = None,
    ) -> None:
        """
        Run windowed SSI-COV analysis.

        Parameters
        ----------
        components : which components to analyse (default: all present)
        mode : "independent" processes each component separately;
               "joint" stacks components as multi-output channels
        window_length : override config window length (seconds)
        overlap : override config overlap fraction
        model_order : if given, use this order; otherwise use config.max_order
        n_block_rows : override config block rows
        max_damping : reject modes above this damping ratio
        min_frequency : reject modes below this frequency (Hz)
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot compute SSI.")

        dt = float(rec.dt)
        cfg = self.config

        wl = window_length if window_length is not None else cfg.window_length
        ovlp = overlap if overlap is not None else cfg.overlap
        order = model_order if model_order is not None else cfg.max_order
        i = n_block_rows if n_block_rows is not None else cfg.n_block_rows
        md = max_damping if max_damping is not None else cfg.max_damping
        mf = min_frequency if min_frequency is not None else cfg.min_frequency

        comps = self._resolve_components(components)
        if not comps:
            raise ValueError("No components to compute. Check df and 'components' argument.")

        df_work = self.df_modal
        time_arr = df_work["time"].to_numpy(dtype=float)

        n_samples = len(time_arr)
        n_window = int(wl / dt)
        n_step = max(1, int(n_window * (1.0 - ovlp)))

        # Window start indices
        starts = list(range(0, n_samples - n_window + 1, n_step))
        if not starts:
            raise ValueError(
                f"Window length ({wl}s = {n_window} samples) exceeds record length ({n_samples} samples)."
            )

        self.window_times = np.array([
            time_arr[s] + wl / 2.0 for s in starts
        ])

        if mode == "independent":
            self._compute_independent(comps, df_work, starts, n_window, dt, i, order, md, mf)
        elif mode == "joint":
            self._compute_joint(comps, df_work, starts, n_window, dt, i, order, md, mf)
        else:
            raise ValueError(f"mode must be 'independent' or 'joint', got '{mode}'")

    def _compute_independent(
        self,
        comps: list[ComponentName],
        df_work: pd.DataFrame,
        starts: list[int],
        n_window: int,
        dt: float,
        i: int,
        order: int,
        max_damping: float,
        min_frequency: float,
    ) -> None:
        for comp in comps:
            data = df_work[comp].to_numpy(dtype=float)
            all_f: list[np.ndarray] = []
            all_d: list[np.ndarray] = []

            for s in starts:
                y_win = data[s: s + n_window]
                try:
                    f, d = self._ssi_cov_single(y_win, dt, i, order, max_damping, min_frequency)
                except Exception:
                    f, d = np.array([]), np.array([])
                all_f.append(f)
                all_d.append(d)

            freq_arr, damp_arr = self._track_modes(
                all_f, all_d, self.config.freq_tol, self.config.damp_tol
            )
            self.frequencies[comp] = freq_arr
            self.damping_ratios[comp] = damp_arr

    def _compute_joint(
        self,
        comps: list[ComponentName],
        df_work: pd.DataFrame,
        starts: list[int],
        n_window: int,
        dt: float,
        i: int,
        order: int,
        max_damping: float,
        min_frequency: float,
    ) -> None:
        data_cols = np.column_stack([
            df_work[c].to_numpy(dtype=float) for c in comps
        ])

        all_f: list[np.ndarray] = []
        all_d: list[np.ndarray] = []

        for s in starts:
            Y_win = data_cols[s: s + n_window, :]
            try:
                f, d = self._ssi_cov_multi(Y_win, dt, i, order, max_damping, min_frequency)
            except Exception:
                f, d = np.array([]), np.array([])
            all_f.append(f)
            all_d.append(d)

        freq_arr, damp_arr = self._track_modes(
            all_f, all_d, self.config.freq_tol, self.config.damp_tol
        )
        # Store under a synthetic key for joint results
        joint_key: ComponentName = comps[0]
        for c in comps:
            self.frequencies[c] = freq_arr
            self.damping_ratios[c] = damp_arr

    # ================================================================ #
    #                   STABILIZATION DIAGRAM                          #
    # ================================================================ #

    def compute_stabilization(
        self,
        *,
        component: ComponentName | None = None,
        window_index: int = 0,
        min_order: int | None = None,
        max_order: int | None = None,
        order_step: int | None = None,
    ) -> None:
        """
        Compute stabilization diagram data for a specific window.

        Sweeps model orders and classifies poles as stable/unstable.
        """
        rec = self.record
        if rec.dt is None:
            raise ValueError("Record.dt is None.")

        dt = float(rec.dt)
        cfg = self.config

        mn = min_order if min_order is not None else cfg.min_order
        mx = max_order if max_order is not None else cfg.max_order
        step = order_step if order_step is not None else cfg.order_step
        i = cfg.n_block_rows

        df_work = self.df_modal
        comps = self._resolve_components([component] if component else None)
        if not comps:
            raise ValueError("No valid component for stabilization.")

        time_arr = df_work["time"].to_numpy(dtype=float)
        n_samples = len(time_arr)
        wl = cfg.window_length
        n_window = int(wl / dt)
        ovlp = cfg.overlap
        n_step = max(1, int(n_window * (1.0 - ovlp)))

        starts = list(range(0, n_samples - n_window + 1, n_step))
        if window_index >= len(starts):
            raise IndexError(
                f"window_index {window_index} out of range (have {len(starts)} windows)."
            )

        s = starts[window_index]
        orders = np.arange(mn, mx + 1, step)
        self.stab_orders = orders

        for comp in comps:
            data = df_work[comp].to_numpy(dtype=float)
            y_win = data[s: s + n_window]

            freq_list: list[np.ndarray] = []
            damp_list: list[np.ndarray] = []
            stable_list: list[np.ndarray] = []

            prev_f: np.ndarray | None = None
            prev_d: np.ndarray | None = None

            for n in orders:
                try:
                    f, d = self._ssi_cov_single(
                        y_win, dt, i, int(n),
                        cfg.max_damping, cfg.min_frequency,
                    )
                except Exception:
                    f, d = np.array([]), np.array([])

                freq_list.append(f)
                damp_list.append(d)

                # Classify stability
                stable = np.zeros(len(f), dtype=bool)
                if prev_f is not None and len(prev_f) > 0 and len(f) > 0:
                    for j, fj in enumerate(f):
                        rel = np.abs(prev_f - fj) / max(fj, 1e-12)
                        best = int(np.argmin(rel))
                        if rel[best] < cfg.freq_tol and abs(d[j] - prev_d[best]) < cfg.damp_tol:
                            stable[j] = True

                stable_list.append(stable)
                prev_f = f
                prev_d = d

            self.stab_frequencies[comp] = freq_list
            self.stab_damping[comp] = damp_list
            self.stab_stable_mask[comp] = stable_list

    # ================================================================ #
    #                         PLOTTING                                 #
    # ================================================================ #

    def _ensure_computed(self) -> None:
        if self.window_times is None or not self.frequencies:
            raise RuntimeError("Call compute() before plotting windowed results.")

    def _ensure_stab_computed(self) -> None:
        if self.stab_orders is None or not self.stab_frequencies:
            raise RuntimeError("Call compute_stabilization() before plotting the stabilization diagram.")

    def plot_stabilization(
        self,
        *,
        component: ComponentName | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        show: bool = True,
        fig=None,
        ax=None,
    ):
        """
        Stabilization diagram: frequency (x) vs model order (y).

        Stable poles are shown as filled circles, unstable as crosses.
        An FFT amplitude spectrum of the window is overlaid for reference.
        """
        import matplotlib.pyplot as plt

        self._ensure_stab_computed()

        comp = component or next(iter(self.stab_frequencies))
        if comp not in self.stab_frequencies:
            raise ValueError(f"No stabilization data for component '{comp}'.")

        freq_list = self.stab_frequencies[comp]
        stable_list = self.stab_stable_mask[comp]
        orders = self.stab_orders

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for idx, (n, f, s_mask) in enumerate(zip(orders, freq_list, stable_list)):
            if len(f) == 0:
                continue
            stable_f = f[s_mask]
            unstable_f = f[~s_mask]

            if len(stable_f) > 0:
                ax.plot(
                    stable_f,
                    np.full_like(stable_f, n),
                    "o",
                    color="green",
                    markersize=4,
                    label="Stable" if idx == 0 else None,
                )
            if len(unstable_f) > 0:
                ax.plot(
                    unstable_f,
                    np.full_like(unstable_f, n),
                    "x",
                    color="red",
                    markersize=3,
                    alpha=0.4,
                    label="Unstable" if idx == 0 else None,
                )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Model Order")
        ax.set_title(f"Stabilization Diagram — Component {comp}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, ls="--")

        if fmin is not None:
            ax.set_xlim(left=fmin)
        if fmax is not None:
            ax.set_xlim(right=fmax)

        # Overlay FFT spectrum on secondary y-axis for reference
        ax2 = ax.twinx()
        df_work = self.df_modal
        if comp in df_work.columns:
            data = df_work[comp].to_numpy(dtype=float)
            dt = float(self.record.dt)
            wl = self.config.window_length
            n_win = int(wl / dt)
            n_step = max(1, int(n_win * (1.0 - self.config.overlap)))
            # Use the same window as compute_stabilization (first window by default)
            seg = data[:n_win]
            N = len(seg)
            freqs_fft = np.fft.rfftfreq(N, d=dt)
            amp = np.abs(np.fft.rfft(seg)) * 2.0 / N
            ax2.plot(freqs_fft, amp, color="steelblue", alpha=0.35, lw=0.8)
            ax2.set_ylabel("FFT Amplitude", color="steelblue", alpha=0.5)
            ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=8)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_frequency_tracking(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        combined: bool = False,
        fmin: float | None = None,
        fmax: float | None = None,
        show: bool = True,
        fig=None,
        axes=None,
    ):
        """Plot identified natural frequencies vs time."""
        import matplotlib.pyplot as plt

        self._ensure_computed()

        comps = self._resolve_components(components)
        comps = [c for c in comps if c in self.frequencies]
        if not comps:
            raise ValueError("No frequency data to plot.")

        t = self.window_times

        if combined:
            if fig is None or axes is None:
                fig, ax = plt.subplots(figsize=(10, 5))
            else:
                ax = axes
            for comp in comps:
                f_arr = self.frequencies[comp]
                n_modes = f_arr.shape[1] if f_arr.ndim == 2 else 0
                for m in range(n_modes):
                    ax.scatter(t, f_arr[:, m], s=8, label=f"{comp} mode {m + 1}", alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("Identified Natural Frequencies")
            if fmin is not None:
                ax.set_ylim(bottom=fmin)
            if fmax is not None:
                ax.set_ylim(top=fmax)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, ncol=2)
            if self.record.name:
                fig.suptitle(self.record.name, fontsize=14, y=0.98)
            fig.tight_layout()
            if show:
                plt.show()
            return fig, ax

        # Separate subplots
        n = len(comps)
        if fig is None or axes is None:
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, comp in zip(axes, comps):
            f_arr = self.frequencies[comp]
            n_modes = f_arr.shape[1] if f_arr.ndim == 2 else 0
            for m in range(n_modes):
                ax.scatter(t, f_arr[:, m], s=8, label=f"Mode {m + 1}", alpha=0.7)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"Component {comp}")
            if fmin is not None:
                ax.set_ylim(bottom=fmin)
            if fmax is not None:
                ax.set_ylim(top=fmax)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, ncol=2)

        axes[-1].set_xlabel("Time (s)")

        if self.record.name:
            fig.suptitle(self.record.name, fontsize=14, y=0.98)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_damping_tracking(
        self,
        *,
        components: Sequence[ComponentName] | None = None,
        combined: bool = False,
        show: bool = True,
        fig=None,
        axes=None,
    ):
        """Plot identified damping ratios vs time."""
        import matplotlib.pyplot as plt

        self._ensure_computed()

        comps = self._resolve_components(components)
        comps = [c for c in comps if c in self.damping_ratios]
        if not comps:
            raise ValueError("No damping data to plot.")

        t = self.window_times

        if combined:
            if fig is None or axes is None:
                fig, ax = plt.subplots(figsize=(10, 5))
            else:
                ax = axes
            for comp in comps:
                d_arr = self.damping_ratios[comp]
                n_modes = d_arr.shape[1] if d_arr.ndim == 2 else 0
                for m in range(n_modes):
                    ax.scatter(t, d_arr[:, m] * 100, s=8, label=f"{comp} mode {m + 1}", alpha=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Damping Ratio (%)")
            ax.set_title("Identified Damping Ratios")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, ncol=2)
            if self.record.name:
                fig.suptitle(self.record.name, fontsize=14, y=0.98)
            fig.tight_layout()
            if show:
                plt.show()
            return fig, ax

        n = len(comps)
        if fig is None or axes is None:
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, comp in zip(axes, comps):
            d_arr = self.damping_ratios[comp]
            n_modes = d_arr.shape[1] if d_arr.ndim == 2 else 0
            for m in range(n_modes):
                ax.scatter(t, d_arr[:, m] * 100, s=8, label=f"Mode {m + 1}", alpha=0.7)
            ax.set_ylabel("Damping Ratio (%)")
            ax.set_title(f"Component {comp}")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, ncol=2)

        axes[-1].set_xlabel("Time (s)")

        if self.record.name:
            fig.suptitle(self.record.name, fontsize=14, y=0.98)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    # ================================================================ #
    #                       CONVENIENCE                                #
    # ================================================================ #

    def get_results(
        self,
        component: ComponentName | None = None,
    ) -> dict[str, np.ndarray]:
        """Return results as a dict for programmatic access."""
        self._ensure_computed()
        if component is None:
            component = next(iter(self.frequencies))
        return {
            "window_times": self.window_times,
            "frequencies": self.frequencies.get(component, np.array([])),
            "damping_ratios": self.damping_ratios.get(component, np.array([])),
        }

    def __repr__(self) -> str:
        comps = ",".join(self.record.components)
        npts = len(self.df_modal)
        n_win = len(self.window_times) if self.window_times is not None else 0

        if self.applied_filters:
            filt = f"{len(self.applied_filters)} ({self.applied_filters[-1]})"
        else:
            filt = "0"

        return (
            f"<ModalAnalysis "
            f"components=[{comps}] "
            f"npts={npts} "
            f"windows={n_win} "
            f"filters={filt}>"
        )

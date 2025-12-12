from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from obspy import Stream, Trace

if TYPE_CHECKING:
    from ..core.record import Record


class Filter:
    """
    Filtering toolbox attached to a Record.

    - Never modifies the original Record (baseline stored in Record.df).
    - All methods return a pandas.DataFrame with columns:
          'time', plus component columns ('X','Y','Z' subset).
    - To chain filters, pass the previous DataFrame via `df=...`.
    """

    def __init__(self, record: "Record") -> None:
        self._record = record

    # ------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------- #

    def _df_to_stream(self, df: pd.DataFrame) -> Stream:
        """
        Convert DataFrame to ObsPy Stream for filtering.

        Expects a 'time' column and the component columns present in
        the underlying Record (record.components).
        """
        rec = self._record

        if "time" not in df.columns:
            raise ValueError("DataFrame must contain 'time' column.")

        if rec.dt is None:
            raise ValueError("Record.dt is None; cannot build Stream.")

        traces = []
        for comp in rec.components:
            if comp not in df.columns:
                raise ValueError(
                    f"DataFrame missing component '{comp}' required by Record."
                )
            tr = Trace(df[comp].to_numpy().copy())
            tr.stats.delta = rec.dt
            tr.stats.channel = comp
            traces.append(tr)

        return Stream(traces)

    def _stream_to_df(self, st: Stream) -> pd.DataFrame:
        """
        Convert Stream → DataFrame with structure:
            time | X | Y | Z
        for the components present in the underlying Record.
        """
        rec = self._record

        if not st:
            raise ValueError("Empty Stream provided to _stream_to_df().")

        n = st[0].stats.npts
        dt = st[0].stats.delta

        data: dict[str, np.ndarray] = {}
        data["time"] = np.arange(n, dtype=float) * dt

        for comp in rec.components:
            for tr in st:
                if tr.stats.channel == comp:
                    data[comp] = tr.data.copy()
                    break
            else:
                raise ValueError(f"Stream missing component '{comp}'")

        return pd.DataFrame(data)

    def _input_df(self, df: pd.DataFrame | None) -> pd.DataFrame:
        """
        If df is None, use the baseline Record.df.
        Otherwise, return df unchanged.
        """
        if df is None:
            # Use a copy to avoid accidental mutation outside
            return self._record.df.copy()
        return df

    # ------------------------------------------------------- #
    # Filters
    # ------------------------------------------------------- #

    def low_pass(
        self,
        Tc: float,
        *,
        corners: int = 4,
        zerophase: bool = True,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Low-pass Butterworth filter with corner period Tc [s].
        """
        in_df = self._input_df(df)

        if self._record.dt is None:
            raise ValueError("Record.dt is None; cannot apply low-pass.")

        fc = 1.0 / Tc

        st = self._df_to_stream(in_df)
        st.filter("lowpass", freq=fc, corners=corners, zerophase=zerophase)

        return self._stream_to_df(st)

    def high_pass(
        self,
        Tc: float,
        *,
        corners: int = 4,
        zerophase: bool = True,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        High-pass Butterworth filter with corner period Tc [s].
        """
        in_df = self._input_df(df)

        if self._record.dt is None:
            raise ValueError("Record.dt is None; cannot apply high-pass.")

        fc = 1.0 / Tc

        st = self._df_to_stream(in_df)
        st.filter("highpass", freq=fc, corners=corners, zerophase=zerophase)

        return self._stream_to_df(st)

    def band_pass(
        self,
        Tc_low: float,
        Tc_high: float,
        *,
        corners: int = 4,
        zerophase: bool = True,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Band-pass between periods Tc_low and Tc_high [s].

        freqmin = 1 / Tc_high
        freqmax = 1 / Tc_low
        """
        in_df = self._input_df(df)

        if self._record.dt is None:
            raise ValueError("Record.dt is None; cannot apply band-pass.")

        fc_low = 1.0 / Tc_high
        fc_high = 1.0 / Tc_low

        st = self._df_to_stream(in_df)
        st.filter(
            "bandpass",
            freqmin=fc_low,
            freqmax=fc_high,
            corners=corners,
            zerophase=zerophase,
        )

        return self._stream_to_df(st)

    def band_stop(
        self,
        Tc_low: float,
        Tc_high: float,
        *,
        corners: int = 4,
        zerophase: bool = True,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Band-stop (notch) between periods Tc_low and Tc_high [s].
        """
        in_df = self._input_df(df)

        if self._record.dt is None:
            raise ValueError("Record.dt is None; cannot apply band-stop.")

        fc_low = 1.0 / Tc_high
        fc_high = 1.0 / Tc_low

        st = self._df_to_stream(in_df)
        st.filter(
            "bandstop",
            freqmin=fc_low,
            freqmax=fc_high,
            corners=corners,
            zerophase=zerophase,
        )

        return self._stream_to_df(st)

    def detrend(
        self,
        type: str = "linear",
        *,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Detrend the data ('demean' or 'linear').
        """
        if type not in {"demean", "linear"}:
            raise ValueError("type must be 'demean' or 'linear'")

        in_df = self._input_df(df)

        st = self._df_to_stream(in_df)
        st.detrend(type)

        return self._stream_to_df(st)

    def taper(
        self,
        max_percentage: float = 0.05,
        type: str = "cosine",
        *,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Apply a taper to all components.
        """
        in_df = self._input_df(df)

        st = self._df_to_stream(in_df)
        st.taper(max_percentage=max_percentage, type=type)

        return self._stream_to_df(st)

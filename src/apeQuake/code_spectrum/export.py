"""Write a code spectrum in the formats analysis programs read.

Every writer samples the model on a grid that contains its knee periods
(``T0``, ``Ts``, ``TL``, the multi-period nodes ...), so the exported curve keeps
the corners of the code spectrum instead of chamfering them.  Periods equal at
the output precision are written once: a duplicated abscissa is invalid for an
OpenSees ``Path`` series and meaningless for ETABS / SAP2000.

* :func:`to_csv` — ``T,Sa`` with a header row.
* :func:`to_etabs` / :func:`to_sap2000` — a two-column ``period value`` text
  file for *Define > Functions > Response Spectrum > From File*.  The first
  line states how many header lines to skip; choose *Period vs Value* and set
  the function's acceleration units to ``units``.
* :func:`to_opensees` — a ``timeSeries Path`` (Tcl) or ``ops.timeSeries('Path',
  ...)`` (OpenSeesPy) whose "time" is the period, as used by OpenSees
  response-spectrum analysis.  It is not a ground motion: never feed it to a
  ``UniformExcitation`` pattern.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import G

if TYPE_CHECKING:
    import os

    from .base import CodeSpectrumModel

__all__ = ["sampled", "to_csv", "to_etabs", "to_opensees", "to_sap2000"]

SpectrumUnit = Literal["g", "m/s2", "cm/s2"]
_PER_G: dict[str, float] = {"g": 1.0, "m/s2": G, "cm/s2": 100.0 * G}

#: Decimal places written for periods and ordinates.
PRECISION = 6


def sampled(
    model: CodeSpectrumModel,
    periods: Iterable[float] | None = None,
    *,
    t_max: float = 4.0,
    n: int = 200,
    units: SpectrumUnit = "g",
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, Sa)`` exactly as the writers emit them.

    ``periods=None`` uses ``model.default_periods(t_max, n)`` (knees included).
    ``Sa`` is converted from g to ``units``.
    """
    if units not in _PER_G:
        raise ValueError(f"units must be one of {sorted(_PER_G)}; got {units!r}")
    T = model.default_periods(t_max, n) if periods is None else model._periods(periods)
    T = np.round(np.sort(T), PRECISION)
    T = T[np.concatenate(([True], np.diff(T) > 0.0))]
    return T, model.sa(T) * _PER_G[units]


def _title(model: CodeSpectrumModel) -> str:
    p = model.parameters()
    extras = ", ".join(f"{k}={p[k]}" for k in ("site_class", "level") if k in p)
    return f"{model.code} elastic spectrum, 5% damping" + (f" ({extras})" if extras else "")


def _write(path: str | os.PathLike[str], lines: list[str]) -> Path:
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def to_csv(
    model: CodeSpectrumModel,
    path: str | os.PathLike[str],
    periods: Iterable[float] | None = None,
    *,
    t_max: float = 4.0,
    n: int = 200,
    units: SpectrumUnit = "g",
) -> Path:
    """Write ``T [s],Sa [units]`` rows with one header row."""
    T, sa = sampled(model, periods, t_max=t_max, n=n, units=units)
    return _write(path, [f"T [s],Sa [{units}]"] + [f"{t:.6f},{s:.6f}" for t, s in zip(T, sa)])


def to_etabs(
    model: CodeSpectrumModel,
    path: str | os.PathLike[str],
    periods: Iterable[float] | None = None,
    *,
    t_max: float = 4.0,
    n: int = 200,
    units: SpectrumUnit = "g",
) -> Path:
    """Write an ETABS response-spectrum function file (``period<TAB>value``).

    Import with *Define > Functions > Response Spectrum > From File*, *Period
    vs Value*, *Header Lines to Skip* as stated on the first line, and the
    function's units set to ``units``.
    """
    T, sa = sampled(model, periods, t_max=t_max, n=n, units=units)
    params = model.parameters()
    info = [
        _title(model),
        f"Units: period [s], acceleration [{units}]",
        ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in params.items()),
        "Written by apeQuake",
    ]
    head = [f"Header lines to skip: {len(info) + 1}"] + info
    return _write(path, head + [f"{t:.6f}\t{s:.6f}" for t, s in zip(T, sa)])


def to_sap2000(
    model: CodeSpectrumModel,
    path: str | os.PathLike[str],
    periods: Iterable[float] | None = None,
    *,
    t_max: float = 4.0,
    n: int = 200,
    units: SpectrumUnit = "g",
) -> Path:
    """Write a SAP2000 response-spectrum function file.

    SAP2000 and ETABS share the *From File* import (same dialog fields), so
    this is :func:`to_etabs`.
    """
    return to_etabs(model, path, periods, t_max=t_max, n=n, units=units)


def to_opensees(
    model: CodeSpectrumModel,
    path: str | os.PathLike[str],
    periods: Iterable[float] | None = None,
    *,
    t_max: float = 4.0,
    n: int = 200,
    units: SpectrumUnit = "g",
    tag: int = 1,
    style: Literal["tcl", "python"] = "tcl",
) -> Path:
    """Write the spectrum as an OpenSees ``Path`` time series indexed by period.

    ``style="tcl"`` writes ``timeSeries Path <tag> -time {...} -values {...}``;
    ``style="python"`` writes the OpenSeesPy call, ready to ``exec`` or import.
    Most models are in length units of m or mm: pick ``units`` to match the
    model (``"m/s2"``), or keep ``"g"`` and scale in the analysis.
    """
    if int(tag) != tag or tag < 1:
        raise ValueError(f"tag must be a positive integer, got {tag!r}")
    T, sa = sampled(model, periods, t_max=t_max, n=n, units=units)
    head = [f"# {_title(model)}", f"# period [s] vs Sa [{units}]; written by apeQuake"]
    if style == "tcl":
        t_txt = " ".join(f"{t:.6f}" for t in T)
        s_txt = " ".join(f"{s:.6f}" for s in sa)
        body = [f"timeSeries Path {int(tag)} -time {{{t_txt}}} -values {{{s_txt}}}"]
    elif style == "python":
        # Plain floats: repr of a numpy scalar ("np.float64(...)") is not valid
        # in a script that only imports openseespy.
        t_txt = ", ".join(f"{t:.6f}" for t in T)
        s_txt = ", ".join(f"{s:.6f}" for s in sa)
        body = [
            "import openseespy.opensees as ops",
            "",
            f"ops.timeSeries('Path', {int(tag)},",
            f"               '-time', [{t_txt}],",
            f"               '-values', [{s_txt}])",
        ]
    else:
        raise ValueError(f"style must be 'tcl' or 'python'; got {style!r}")
    return _write(path, head + body)

"""Readers for ground-motion record files.

Three formats, one result type (:class:`RecordFile`, one component per file):

* ``"at2"`` — PEER NGA ``.AT2``: four header lines, the fourth giving ``NPTS``
  and ``DT`` (``NPTS=  7278, DT=  .0050 SEC`` or the older
  ``7278  0.0050  NPTS, DT``), then values row by row.  Units from the third
  line (``... IN UNITS OF G``).
* ``"renac"`` — Red Nacional de Acelerografos (IG-EPN, Ecuador) ``.txt``:
  ``key: value`` header lines (``Estacion``, ``Componente``,
  ``Frecuencia de muestreo (Hz)``, ``Unidades``...), a line of underscores,
  then values row by row.  ``dt = 1 / frecuencia``.
* ``"txt"`` — generic text: non-numeric header lines, then values row by row
  (one or several per line).  ``dt`` from a header line containing ``DT``
  (``DT = 0.005``) or from the ``dt=`` argument.

A non-numeric line inside the data block raises ``ValueError`` naming the line:
the readers never fill gaps with zeros, which would silently change the motion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.units import AccelUnit

FileFormat = Literal["auto", "at2", "renac", "txt"]

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class RecordFile:
    """One component read from a file.

    Attributes
    ----------
    data : ndarray
        Acceleration samples, in ``units`` when known.
    dt : float
        Time step [s].
    units : {"g", "m/s2", "cm/s2"} or None
        Units declared by the file; ``None`` if it declares none.
    format : str
        ``"at2"``, ``"renac"`` or ``"txt"``.
    path : str
        File read.
    metadata : dict
        Header fields (RENAC: station, component, event...; AT2: title).
    """

    data: np.ndarray
    dt: float
    units: AccelUnit | None
    format: str
    path: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def npts(self) -> int:
        return int(self.data.size)


def _lines(path: str | Path) -> list[str]:
    # RENAC headers are Latin-1 (accented Spanish); AT2 and plain text are ASCII.
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return text.splitlines()


def _values(lines: list[str], first_line: int, path: str | Path) -> np.ndarray:
    """Parse ``lines`` (data block starting at file line ``first_line``) row by row."""
    out: list[float] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        try:
            out.extend(float(v) for v in s.replace(",", " ").split())
        except ValueError:
            raise ValueError(
                f"{path}: non-numeric data on line {first_line + i + 1}: {s[:60]!r}"
            ) from None
    if not out:
        raise ValueError(f"{path}: no data values found.")
    return np.asarray(out, dtype=float)


def _is_numeric(line: str) -> bool:
    parts = line.replace(",", " ").split()
    return bool(parts) and all(re.fullmatch(_NUM, p) for p in parts)


def _units_from_text(text: str) -> AccelUnit | None:
    t = text.upper()
    if re.search(r"CM\s*/\s*S", t) or "GAL" in t:
        return "cm/s2"
    if re.search(r"\bM\s*/\s*S", t):
        return "m/s2"
    if re.search(r"\bG\b", t):
        return "g"
    return None


def read_at2(path: str | Path) -> RecordFile:
    """Read a PEER NGA ``.AT2`` acceleration file."""
    lines = _lines(path)
    if len(lines) < 5:
        raise ValueError(f"{path}: too short for an AT2 file (4 header lines expected).")
    head = lines[3]
    m_npts = re.search(r"NPTS\s*=\s*(\d+)", head, re.IGNORECASE)
    m_dt = re.search(rf"DT\s*=\s*({_NUM})", head, re.IGNORECASE)
    if m_npts and m_dt:
        npts, dt = int(m_npts.group(1)), float(m_dt.group(1))
    else:  # older layout: "7278    0.0050    NPTS, DT"
        nums = re.findall(_NUM, head)
        if len(nums) < 2 or "NPTS" not in head.upper():
            raise ValueError(f"{path}: line 4 does not give NPTS and DT: {head.strip()!r}")
        npts, dt = int(float(nums[0])), float(nums[1])
    data = _values(lines[4:], 4, path)
    if data.size < npts:
        raise ValueError(f"{path}: header says NPTS={npts} but only {data.size} values found.")
    return RecordFile(
        data=data[:npts],
        dt=dt,
        units=_units_from_text(lines[2]),
        format="at2",
        path=str(path),
        metadata={"title": lines[1].strip(), "units_line": lines[2].strip()},
    )


def read_renac(path: str | Path) -> RecordFile:
    """Read a RENAC (IG-EPN, Ecuador) acceleration ``.txt`` file."""
    lines = _lines(path)
    sep = next((i for i, s in enumerate(lines) if s.strip() and set(s.strip()) == {"_"}), None)
    if sep is None:
        raise ValueError(f"{path}: no '____' separator line ending the RENAC header.")
    meta: dict[str, str] = {}
    freq = None
    units = None
    for line in lines[:sep]:
        if ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        k = key.lower()
        if "frecuencia" in k:
            m = re.search(_NUM, val)
            if m:
                freq = float(m.group(0))
        elif "unidad" in k:
            units = _units_from_text(val)
        elif "estaci" in k:
            meta["station"] = val
        elif "componente" in k:
            meta["component"] = val
        elif k == "evento":  # not "Fecha del evento ..."
            meta["event"] = val
        else:
            meta[key] = val
    if not freq or freq <= 0.0:
        raise ValueError(f"{path}: no 'Frecuencia de muestreo (Hz)' in the RENAC header.")
    return RecordFile(
        data=_values(lines[sep + 1:], sep + 1, path),
        dt=1.0 / freq,
        units=units,
        format="renac",
        path=str(path),
        metadata=meta,
    )


def read_txt(path: str | Path, *, dt: float | None = None) -> RecordFile:
    """Read a generic text file: header lines, then values row by row.

    ``dt`` comes from a header line containing ``DT`` (``DT = 0.005``) unless
    given; a ``dt`` argument that disagrees with the header raises.
    """
    lines = _lines(path)
    start = next((i for i, s in enumerate(lines) if _is_numeric(s)), None)
    if start is None:
        raise ValueError(f"{path}: no numeric data lines found.")
    header = lines[:start]
    dt_head = None
    for line in header:
        m = re.search(rf"\bDT\b\s*[=:]?\s*({_NUM})", line, re.IGNORECASE)
        if m:
            dt_head = float(m.group(1))
            break
    if dt is None and dt_head is None:
        raise ValueError(f"{path}: the header gives no DT; pass dt=.")
    if dt is not None and dt_head is not None and not np.isclose(dt, dt_head, rtol=1e-6):
        raise ValueError(f"{path}: dt={dt} disagrees with the header DT={dt_head}.")
    unit_lines = [s for s in header if re.search(r"UNIT|UNIDAD", s, re.IGNORECASE)]
    units: AccelUnit | None = next(
        (u for u in map(_units_from_text, unit_lines) if u is not None), None
    )
    return RecordFile(
        data=_values(lines[start:], start, path),
        dt=float(dt if dt is not None else dt_head),  # type: ignore[arg-type]
        units=units,
        format="txt",
        path=str(path),
        metadata={},
    )


def detect_format(path: str | Path) -> Literal["at2", "renac", "txt"]:
    """Guess the format from the extension and the first header lines."""
    if Path(path).suffix.upper() == ".AT2":
        return "at2"
    head = "\n".join(_lines(path)[:15]).upper()
    if "PEER" in head or re.search(r"NPTS\s*=", head):
        return "at2"
    if "RENAC" in head or "FRECUENCIA DE MUESTREO" in head:
        return "renac"
    return "txt"


def read_record_file(
    path: str | Path, *, format: FileFormat = "auto", dt: float | None = None
) -> RecordFile:
    """Read one component file in any supported format.

    ``format="auto"`` uses :func:`detect_format`.  ``dt`` is only used by the
    ``"txt"`` reader (the other formats carry their own time step).
    """
    fmt = detect_format(path) if format == "auto" else format
    if fmt == "at2":
        return read_at2(path)
    if fmt == "renac":
        return read_renac(path)
    if fmt == "txt":
        return read_txt(path, dt=dt)
    raise ValueError(f"format must be 'auto', 'at2', 'renac' or 'txt'; got {format!r}")


__all__ = [
    "RecordFile",
    "detect_format",
    "read_at2",
    "read_record_file",
    "read_renac",
    "read_txt",
]

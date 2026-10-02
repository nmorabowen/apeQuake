"""Typst report of a site seismic assessment (NEC-SE-DS vs ASCE 7 vs IG-EPN).

``build_report(assessment, meta)`` returns the Typst source, the SVG figures and,
when the ``typst`` CLI and the local ``ape-informes`` package are available, the
compiled PDF::

    from apeQuake.site_assessment import assess_site
    from apeQuake.report import ReportMeta, build_report

    rep = build_report(assess_site(-0.22, -78.51, site_class="D"),
                       ReportMeta(project="Ejemplo", site_name="Quito"))
    open("informe.pdf", "wb").write(rep.pdf)

The report uses the ``@local/ape-informes:0.1.0`` Typst package (``ape-report``
template).  Compilation runs in a temporary folder; nothing is written elsewhere.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import figures
from .document import ReportMeta, render

__all__ = ["ReportMeta", "Report", "build_report", "compile_typst", "TypstError"]


class TypstError(RuntimeError):
    """``typst compile`` is missing or failed."""


@dataclass
class Report:
    typ: str
    figures: dict[str, str] = field(default_factory=dict)   # file name -> SVG text
    pdf: bytes | None = None


def _font_path() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    p = Path(appdata) / "typst" / "packages" / "local" / "ape-informes" / "0.1.0" / "fonts"
    return p if p.is_dir() else None


def compile_typst(typ: str, files: dict[str, str], *, timeout: float = 120.0) -> bytes:
    """Compile Typst source (plus side files) to PDF bytes in a temporary folder."""
    for name in files:
        if Path(name).name != name or name in ("main.typ", "main.pdf"):
            raise ValueError(f"invalid file name {name!r}")   # flat names, never paths
    exe = shutil.which("typst")
    if exe is None:
        raise TypstError("the typst CLI is not on PATH")
    with tempfile.TemporaryDirectory(prefix="apequake-report-") as tmp:
        root = Path(tmp)
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")
        (root / "main.typ").write_text(typ, encoding="utf-8")
        cmd = [exe, "compile", "--root", str(root)]
        fonts = _font_path()
        if fonts is not None:
            cmd += ["--font-path", str(fonts)]
        cmd += [str(root / "main.typ"), str(root / "main.pdf")]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise TypstError(p.stderr.strip()[-2000:] or "typst compile failed")
        return (root / "main.pdf").read_bytes()


def build_report(assessment: Any, meta: ReportMeta | None = None, *,
                 compile: bool = True) -> Report:
    """Build the report for a :class:`~apeQuake.site_assessment.SiteAssessment`
    (or its ``to_dict()``)."""
    d = assessment.to_dict() if hasattr(assessment, "to_dict") else assessment
    meta = meta or ReportMeta()
    figs = {"fig-roca.svg": figures.rock_spectra(d)}
    if d["site"] is not None:
        figs["fig-sitio.svg"] = figures.site_spectra(d)
    figs["fig-ubicacion.svg"] = figures.location_map(d)
    rep = Report(typ=render(d, meta), figures=figs)
    if compile:
        rep.pdf = compile_typst(rep.typ, figs)
    return rep

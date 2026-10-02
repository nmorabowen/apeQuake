"""Typst site report: source, figures and (when typst + ape-informes exist) the PDF."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from apeQuake.report import ReportMeta, build_report, compile_typst
from apeQuake.report.document import esc
from apeQuake.site_assessment import assess_site

QUITO = (-0.2200, -78.5125)
GYE = (-2.1894, -79.8891)


def _has_typst_package() -> bool:
    appdata = os.environ.get("APPDATA", "")
    pkg = Path(appdata) / "typst" / "packages" / "local" / "ape-informes" / "0.1.0"
    return shutil.which("typst") is not None and (pkg / "typst.toml").is_file() \
        and (pkg / "images" / "APE_LOGO.png").is_file()


def test_escape():
    assert esc("A#B$C_D") == "A\\#B\\$C\\_D"
    assert esc("[x] <y> @z") == "\\[x\\] \\<y\\> \\@z"


def test_source_and_figures_class_d():
    rep = build_report(assess_site(*QUITO, site_class="D"),
                       ReportMeta(site_name="Quito"), compile=False)
    assert rep.pdf is None
    assert '#import "@local/ape-informes:0.1.0"' in rep.typ
    assert "= Introducción" in rep.typ and "= Comparación de espectros" in rep.typ
    assert "zona sísmica V" in rep.typ and "$Z = 0.40$" in rep.typ
    assert set(rep.figures) == {"fig-roca.svg", "fig-sitio.svg", "fig-ubicacion.svg"}
    assert all(s.lstrip().startswith("<?xml") or "<svg" in s[:500] for s in rep.figures.values())


def test_class_f_has_no_site_section():
    rep = build_report(assess_site(*QUITO, site_class="F"), compile=False)
    assert "== Espectros para el perfil de suelo" not in rep.typ
    assert "fig-sitio.svg" not in rep.figures


def test_class_e_without_asce_site_spectra_renders():
    rep = build_report(assess_site(*GYE, site_class="E"), compile=False)
    assert "== Espectros para el perfil de suelo" in rep.typ


def test_names_from_data_are_escaped():
    rep = build_report(assess_site(*QUITO, site_class="D"),
                       ReportMeta(site_name="Obra #1 $x$"), compile=False)
    assert "Obra \\#1 \\$x\\$" in rep.typ


def test_side_file_names_are_flat():
    with pytest.raises(ValueError):
        compile_typst("x", {"../evil.svg": "<svg/>"})


@pytest.mark.skipif(not _has_typst_package(), reason="typst CLI or @local/ape-informes not installed")
def test_compiles_to_pdf():
    rep = build_report(assess_site(*QUITO, vs30=250),
                       ReportMeta(site_name="Quito", project="Prueba", date="Octubre 2026"))
    assert rep.pdf is not None and rep.pdf[:5] == b"%PDF-" and len(rep.pdf) > 50_000

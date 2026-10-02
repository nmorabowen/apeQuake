# Typst site report: `apeQuake.report`

apeQuake#13 (sample sections) and the full chapter set (2026-10-02): Resumen ejecutivo,
Introducción, Ubicación y zonificación sísmica (map), Parámetros en roca (Ss/S1 derivation),
Espectros NEC (uses `txt-espectro-intro`), Espectros ASCE/SEI 7 (7-16 with substitutions and the
11.4.8 exception; 7-22 approximate), Peligro IG-EPN (UHS table with fractiles), Comparación,
Conclusiones, Supuestos y limitaciones (fixed limitations + every notice in Spanish via
`report.chapters.notice_es`). Chapters live in `report/chapters.py`.

```python
from apeQuake.report import ReportMeta, build_report, compile_typst, TypstError

meta = ReportMeta(title="Parámetros Sísmicos del Sitio", project="...", client="...",
                  document_code="APE-....", revision="0", date="Octubre 2026",
                  site_name="Quito", authors=[{"name": ..., "email": ..., "affiliation": ...}],
                  revisions=None)          # default: one "Emisión inicial" row
rep = build_report(assessment_or_dict, meta, compile=True)
rep.typ        # Typst source (main.typ), imports @local/ape-informes:0.1.0, ape-report template
rep.figures    # {"fig-roca.svg", "fig-sitio.svg" (not for F), "fig-ubicacion.svg"}
rep.pdf        # bytes, or None with compile=False
```

- `compile_typst(typ, files)` runs `typst compile` in a temp folder (`--root` = that folder,
  `--font-path` = the package's bundled fonts if installed). Side files must be flat names.
- Figures: matplotlib SVG with `svg.fonttype = "path"` (no font dependency in Typst).
- Prose (`report/document.py`): informe register of the APE tone standard (load `ape-tono`
  before editing any Spanish text; run its `tone-lint.py` on the generated `.typ`). Every number
  comes from the assessment dict; comparison phrases from `_cmp(value, ref)`; data strings go
  through `esc()` before entering markup. Use "período", accents everywhere.
- Known lint error: edition drift ASCE 7-16 + 7-22 in one document (rule 11). Inherent to the
  comparison; pending the owner's decision (exception in the standard, or one edition).

## Prerequisite: `@local/ape-informes:0.1.0`

Install by copying `ape-informes/package/*` to
`%APPDATA%\typst\packages\local\ape-informes\0.1.0\` (what `install-package.ps1` does; its
per-user font registration is not needed because the report passes `--font-path`).
The logo is tracked since ape-informes#2 and the NEC spectrum text (`txt-espectro-intro`,
plateau to T_C) was fixed in ape-informes#1; reinstall the package after pulling.

## API: `report.build` (apeQuake/1 1.3.0)

Args = `site.assess` args + optional `report` {title, project, client, documentCode, revision,
date, siteName, authors[{name, email?, affiliation?}]}. Result {pdfBase64, typ, figures
{name: svg}, fileName}. `report_unavailable` when the typst CLI or the local package is
missing (`TypstError`); a failing compile of the document is `internal_error`
(`TypstCompileError`). Author dicts are padded with empty email/affiliation (the cover needs them).

## Lint status

Two findings remain and are not report defects: "Conclusiones" flagged as unaccented (linter
false positive, the plural has no accent; fix queued for ape-workflow) and ASCE 7-16 + 7-22
edition drift (pending owner decision). Placeholders for missing values are "no aplica", never "—".

# Typst site report: `apeQuake.report`

Merged in apeQuake#13 (2026-10-01) with the **sample sections only** (Resumen ejecutivo,
Introducción, Comparación de espectros). The rest waits for the owner's tone approval.

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
**Gotcha:** the ape-informes repo `.gitignore` excludes `*.png`, so `images/APE_LOGO.png` is not
in git; copy it from `ape-ofertas/package/images/APE_LOGO.png` or the default logo fails.
Also: `txt-espectro-intro(fuente: "nec")` in ape-informes states the NEC spectrum with T0
where Tc belongs (fix in progress, separate session) — the site report does not use it.

## Remaining chapters (planned)

Ubicación y zonificación (with `fig-ubicacion.svg`), Definición de S_S y S_1, Parámetros NEC y
ASCE, Peligro IG-EPN (UHS with fractiles), Conclusiones, Supuestos y limitaciones (all
`warnings`), then `report.build` in the API (PDF base64 + `.typ` + figures; never a path).

---
name: apequake
description: >
  apeQuake is APE's Python library for ground motions and seismic hazard of Ecuador.
  Use this skill whenever code imports apeQuake (``from apeQuake import Record``,
  ``apeQuake.hazard``, ``apeQuake.nec``, ``apeQuake.site_assessment``, ``apeQuake.api``,
  ``apeQuake.report``, ``apeQuake.code_spectrum``) or touches the repo at
  ``C:\Users\nmora\Documents\Github\apeQuake``, and for the ape-tools "sismo" module that
  calls it. Trigger for: a point in Ecuador -> NEC-SE-DS zone factor Z / zone / region /
  eta (``zone_at``, digitized Figura 1, Tabla 19 cross-check); comparing NEC-SE-DS vs
  ASCE 7-16 vs ASCE 7-22 vs IG-EPN at a site (``assess_site``: Ss/S1 on rock by method
  nec | igepn | manual, per-code site factors, Vs30 or site class, rock and site views);
  the ``apeQuake/1`` JSON API (``dispatch``, ``zoning.at``, ``site.assess``, schemas,
  error codes); the Typst site report on ape-informes (``build_report``); IG-EPN
  probabilistic hazard (``EcuadorHazard``, ``HazardSite``, UHS 475/2475, hazard curves,
  maps, ``explore``); digitized NEC hazard curves (``load_hazard_database``); code spectra
  (``NECSpectrum``, ``ASCE7_16Spectrum``, ``ASCE7_22Spectrum``); and record processing
  (``Record``: filters, Fourier spectrum, response spectra, spectrogram, intensity
  measures). For NEC theory use ``nec-ds-sismico``; for ASCE 7 theory ``asce7-ground-motion``;
  for report prose ``ape-tono``.
---

# apeQuake

Python library (numpy / pandas / scipy / matplotlib / numba / obspy) for ground-motion
processing and the seismic hazard of Ecuador. Repo
`C:\Users\nmora\Documents\Github\apeQuake`, GitHub `nmorabowen/apeQuake`, branch `main`.
Tests: `python -m pytest -q` (≈455 tests). Docs: mkdocs under `docs/`.

## Package map

| Module | What it owns | Reference |
|---|---|---|
| `apeQuake.Record` (`core/`) | Multi-component records and their composites: `filter`, `spectrum`, `response_spectra`, `spectrogram`, `intensity_measures`, `plot_record`, `code_spectrum` | [records.md](references/records.md) |
| `apeQuake.code_spectrum` | `NECSpectrum`, `ASCE7_10Spectrum`, `ASCE7_16Spectrum`, `ASCE7_22Spectrum`; registry `get_model_class` | [site-assessment.md](references/site-assessment.md) |
| `apeQuake.hazard` | IG-EPN probabilistic hazard snapshot: `EcuadorHazard`, `HazardSite`, maps, `explore()` | [hazard-igepn.md](references/hazard-igepn.md) |
| `apeQuake.nec` | Digitized NEC hazard curves (23 capitals) and **zoning**: `zone_at`, `region_at`, `table19` | [nec-zoning.md](references/nec-zoning.md) |
| `apeQuake.site_assessment` | `assess_site`: NEC vs ASCE 7-16 vs 7-22 (approx.) vs IG-EPN at a point | [site-assessment.md](references/site-assessment.md) |
| `apeQuake.api` | `apeQuake/1` JSON API for ape-tools: `dispatch`, schemas | [json-api.md](references/json-api.md) |
| `apeQuake.report` | Typst site report on `@local/ape-informes` | [report.md](references/report.md) |

## The site workflow in four lines

```python
from apeQuake.nec import zone_at
from apeQuake.site_assessment import assess_site
from apeQuake.report import ReportMeta, build_report

z = zone_at(-0.22, -78.51)                         # Z 0.40, zone V, sierra, eta 2.48 (+ warnings)
a = assess_site(-0.22, -78.51, vs30=250)           # or site_class="D"; method="nec"|"igepn"|"manual"
rep = build_report(a, ReportMeta(site_name="Quito"))   # rep.typ, rep.figures (SVG), rep.pdf
```

Same through the API: `dispatch({"api": "apeQuake/1", "command": "site.assess", "args": {"lat": -0.22, "lon": -78.51, "vs30": 250}})`.

## Decisions that are not obvious from the code (owner, 2026-10-01)

- **Ss and S1 are defined once, on rock** (Vs30 = 760 m/s); each code then applies its own
  site factors. Method `nec`: `Ss = 1.5·Sa_B(0.2 s)`, `S1 = 1.5·Sa_B(1.0 s)` from the NEC class-B
  spectrum (1.5 = 475-yr design level -> MCE-type, mirroring SD = 2/3 SM). Not risk-targeted.
- **ASCE 7-22 is approximate in Ecuador.** 7-22 has no Fa/Fv tables (SMS/SM1 come from the USGS
  geodatabase, which does not cover Ecuador): SMS/SM1 from 7-16 tables + 7-22 two-period shape.
- **Z comes from the digitized map only** (user choice); Tabla 19 is a cross-check reported in
  `nearest_listed` and warnings, not the lookup. NEC's own rule is Tabla 19 first.
- **eta by province**, exactly as NEC words it ("Provincias de la Costa / Sierra / Oriente").
- **Site class input:** Vs30 (each code classifies with its own table; 7-22 has BC/CD/DE) or one
  A-F class for all codes. F -> rock view only.
- **TL for ASCE** defaults to the NEC TL of the site class (no ASCE TL map for Ecuador).

## Gotchas

- ASCE 7-16 §11.4.8 fires for most of Ecuador (class D with S1 ≥ 0.2): `assess_site` applies the
  exception and reports it. Class E with S1 > 0.1 has **no Fv** (Table 11.4-2): no ASCE site
  spectra, NEC and IG-EPN still returned.
- The NEC figure and Tabla 19 disagree in places (Quevedo, Vinces, Milagro listed 0.35, inside the
  figure's 0.30 ellipse; 10 parishes listed twice with different Z). Map vs table: 91 % at the
  parish point. Do not "fix" the map to match the table.
- Use `EcuadorHazard().site(lat, lon, interp="bilinear")` for IG-EPN at a point; do not write a
  lattice lookup: IG-EPN cells are merged in places (Guayas river), point-in-polygon is correct.
- `apeQuake.api` never imports `jsonschema` at runtime (dev dependency only); schemas in
  `api/schemas/*.json` are the contract and the tests prove the hand checks agree.
- The report needs the `typst` CLI and `@local/ape-informes:0.1.0` installed (see report.md).

## Status of PLAN-002 (ape-tools "sismo" module), 2026-10-01

Merged in apeQuake: #10 zoning, #11 `assess_site`, #12 JSON API, #13 report (sample sections:
Resumen ejecutivo, Introducción, Comparación de espectros). Pending: the remaining report chapters
and `report.build` (API 1.3.0) after the owner approves the sample tone; the ASCE 7-16/7-22
edition-drift rule (tone standard rule 11); ape-tools T1 (pin, type generation, registry) and T2
(screens). Plan: `ape-tools/docs/plans/PLAN-002-sismo.md`.

# JSON API (`apeQuake/1`)

`apeQuake.api.dispatch(request) -> response` is the entry point ape-tools uses. It
is one pure function that never raises. Every command's arguments and result
are described by a JSON Schema (draft 2020-12) shipped in `apeQuake/api/schemas/`,
and those files are the contract. The design follows apeLoads ADR-0006.

```python
from apeQuake.api import dispatch

dispatch({"api": "apeQuake/1", "command": "site.assess",
          "args": {"lat": -0.22, "lon": -78.51, "vs30": 250, "method": "nec"}})
```

| Command | What it returns |
|---|---|
| `api.describe` | The version, the commands, and their schemas. |
| `zoning.at` | NEC Z, zone, province, region, η, distance to the nearest zone boundary, and the nearest Tabla 19 town. |
| `map.layer` | One map layer: `necZones` (run-length encoded grid of Figura 1), `igepnHazard` (cell polygons with Sa for `tr` 475/2475, `period`, `stat`), `faults`, `sourceZones`, `capitals`, `provinces`. |
| `report.build` | The Typst site report: PDF (base64), `.typ` source and SVG figures (needs the typst CLI and `@local/ape-informes`). |
| `site.assess` | The NEC-SE-DS, ASCE 7-16, ASCE 7-22 (approximate) and IG-EPN comparison: parameters, spectra, the rock and site views, and warnings. |
| `spectrum.code` | One code spectrum by `code`: `ASCE7-10` / `ASCE7-16` (ss, s1, siteClass, tl), `ASCE7-22` (sms, sm1, siteClass, tl), `ASCE7-22-MPRS` (the 22 MCE_R ordinates `saMcer`, siteClass, tl, optional sms/sm1), `NEC-15` (z, siteClass, region). Returns the parameters, knee periods and Sa(T); with `export` {format: `csv`, `etabs`, `sap2000`, `opensees-tcl`, `opensees-python`; units} also the file text. |
| `design.sdc` | ASCE 7 §11.6 seismic design category from sds, sd1, s1 and riskCategory, with Ie (Table 1.5-2). |
| `design.elf` | ELF base shear. `code: "asce7"` (edition 7-10 / 7-16 / 7-22, §12.8 Method 2) or `"nec"` (NEC-SE-DS 6.3); `t` and/or `hn`; optional `stories` [{weight, height}] for Fx and Vx. ASCE results list every Cs limit with this edition's equation number. |

**Error codes:**

- `bad_request`: the request doesn't match the schema.
- `unknown_command`: no command by that name.
- `out_of_area`: the point is outside Ecuador or the NEC zone map.
- `value_error`: the domain refused a value.
- `report_unavailable`: the typst CLI or the ape-informes package is not installed.
- `internal_error`: a bug. The message carries only the exception type.

**Units:** degrees (WGS-84), g, s, m/s for Vs30, km, and m for building heights. Forces in `design.elf` come back in the unit of `w`.

`design.elf` does not take a spectrum, so ASCE 7-22 Method 1 (Cs from the multi-period spectrum) is library-only for now: `apeQuake.design.asce7.base_shear(..., spectrum=...)`.

**Command line:** `python -m apeQuake.api` reads one request on stdin and writes
the response to stdout.

::: apeQuake.api.dispatch

# JSON API `apeQuake/1`: `apeQuake.api`

Merged in apeQuake#12 (2026-10-01). Modelled on apeLoads ADR-0006 so ape-tools registers it
like Loads: `Module("sismo", "Sismo", api=apeQuake.api.API, dispatch=apeQuake.api.dispatch)`.

```python
from apeQuake.api import API, API_VERSION, ERROR_CODES, commands, dispatch, schema, schema_names

dispatch({"api": "apeQuake/1", "command": "zoning.at", "args": {"lat": -0.22, "lon": -78.51}})
# {"api": "apeQuake/1", "ok": true, "result": {...}}  |  {"ok": false, "error": {"code", "message"}}
```

- `API_VERSION = "1.1.0"`. History: 1.0.1 envelope branches titled Ok / Error and `py.typed`; 1.0.2 every `$ref` names its file (`common.json#/...`, even inside common.json, for json-schema-ref-parser); 1.1.0 `map.layer`. Additive changes bump the minor (`report.build` → 1.2.0).
- Commands: `api.describe` (version, commands, schemas, `common`), `zoning.at` {lat, lon},
  `site.assess` {lat, lon, vs30 | siteClass, method?, ss?, s1?, z?, region?, tlAsce?},
  `map.layer` {layer: necZones | igepnHazard | faults | sourceZones | capitals | provinces;
  tr? 475|2475, period?, stat? only with igepnHazard}. Results by `kind`: `grid` (necZones:
  lon0/dlon/lat0/dlat, nx/ny, `rows` of [code, count] runs north→south, `classes`), `geojson`
  (FeatureCollection; hazard properties {cellId, sa}, plus min/max), `points` (capitals).
  Layer builders live in `api/layers.py`; `nec.zone_grid()` is the public raster accessor.
- Error codes: `bad_request`, `unknown_command`, `out_of_area` (`OutsideEcuadorError`),
  `value_error`, `internal_error` (message = exception type only).
- `dispatch` never raises; results must pass `json.dumps(allow_nan=False)`.
- Payloads are **camelCase**, built explicitly in `api/handlers.py` (`assessment_payload`,
  `_zone_payload`), not dumped from `to_dict()`. Units: degrees, g, s, m/s, km.
- Schemas: `api/schemas/*.json`, draft 2020-12, `$id` =
  `https://nmorabowen.github.io/apeQuake/schemas/apeQuake-1/<name>.json`, cross-refs
  `common.json#/$defs/...`. They are the contract (ape-tools generates TS types from them).
- Runtime checks (`api/checks.py`) mirror the schemas without jsonschema: closed objects,
  exactly one of vs30/siteClass, ss/s1 only with method manual, finite numbers, bool ≠ number.
- CLI: `python -m apeQuake.api` (stdin → stdout, exit 0 ok / 1 not ok; NaN rejected).

## Adding a command

Handler in `handlers.py` + registry entry in `dispatcher.py` + `<cmd>.args.json` /
`<cmd>.result.json` + valid fixtures, shape-error fixtures and domain-error fixtures in
`tests/test_api.py` (schema and dispatch must agree on each) + bump `API_VERSION`.
